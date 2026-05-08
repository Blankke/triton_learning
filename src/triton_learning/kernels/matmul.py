"""
Triton 主干矩阵乘 kernel。

使用示例：
    from triton_learning.kernels.matmul import triton_matmul
    out = triton_matmul(x, c)

说明：
    本文件只实现 Step 2 需要的主干矩阵乘 W = X @ C。
    输入矩阵形状为：
        X: [M, K]
        C: [K, N]
        W: [M, N]
    后续 Step 3 会在这个 kernel 的基础上融合 Z = Y @ B。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def matmul_autotune_configs() -> list[triton.Config]:
    """
    基于 Triton 官方 matmul 教程整理 autotune 配置。

    `docs/Matmul.md` 已经记录了 grouped ordering、tile 和 cache 复用思路，
    这里直接把这些配置封装出来，供 Step 2 和 Step 3 复用。
    """
    configs: list[triton.Config] = []
    for block_m in (16, 32, 64):
        for block_n in (64, 128):
            for block_k in (32, 64):
                for group_m in (4, 8):
                    for num_warps in (4, 8):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_m,
                                },
                                num_warps=num_warps,
                                num_stages=3,
                            )
                        )
    return configs


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """计算 C = A @ B 的一个输出 tile。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # grouped ordering：让相邻 program 尽量复用 B 的 tile，提高 L2 cache 命中率。
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # fp32 累加能降低 fp16 输入下的数值误差。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper：检查输入、分配输出、启动 Triton kernel。

    参数：
        a: [M, K] CUDA tensor，建议 fp16
        b: [K, N] CUDA tensor，建议 fp16

    返回：
        c: [M, N]，即 a @ b
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_matmul 只支持二维矩阵。")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"矩阵形状不匹配：a={tuple(a.shape)}, b={tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_matmul 需要 CUDA tensor。")
    if a.dtype != b.dtype:
        raise ValueError(f"a 和 b 的 dtype 必须一致：a={a.dtype}, b={b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"暂不支持 dtype={a.dtype}。")

    # contiguous 能让 stride 规则更简单，便于学习和复现实验。
    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
