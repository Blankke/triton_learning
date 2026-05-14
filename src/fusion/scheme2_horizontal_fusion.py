"""
方案2：single-kernel horizontal fusion via pid partition。

使用示例：
    from fusion.scheme2_horizontal_fusion import triton_horizontal_fused_down_main
    y, w = triton_horizontal_fused_down_main(x, a, c)

说明：
    一个 Triton kernel 内部把 program id 分成两段：
        pid < num_tiles_1      -> 计算 Y = X @ A 的一个 tile
        pid >= num_tiles_1     -> 计算 W = X @ C 的一个 tile
    这里直接复用 Step 2 GEMM 的调度骨架：
        autotune + grouped ordering + fp32 accumulator
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_learning.kernels.matmul import matmul_autotune_configs


REFERENCE_BLOCK_M_DOWN = 16
REFERENCE_BLOCK_N_DOWN = 16
REFERENCE_BLOCK_M_MAIN = 16
REFERENCE_BLOCK_N_MAIN = 128


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "H", "R", "N"],
)
@triton.jit
def _horizontal_fused_kernel(
    x_ptr,
    a_ptr,
    c_ptr,
    y_ptr,
    w_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ar: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yr: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """基于 Step 2 GEMM 骨架，用 pid 区间分别计算 Y tile 或 W tile。"""
    pid = tl.program_id(axis=0)

    num_pid_m_down = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_down = tl.cdiv(R, BLOCK_SIZE_N)
    num_tiles_down = num_pid_m_down * num_pid_n_down

    if pid < num_tiles_down:
        num_pid_in_group_down = GROUP_SIZE_M * num_pid_n_down
        group_id_down = pid // num_pid_in_group_down
        first_pid_m_down = group_id_down * GROUP_SIZE_M
        group_size_m_down = tl.minimum(num_pid_m_down - first_pid_m_down, GROUP_SIZE_M)
        pid_m_down = first_pid_m_down + ((pid % num_pid_in_group_down) % group_size_m_down)
        pid_n_down = (pid % num_pid_in_group_down) // group_size_m_down

        offs_m = pid_m_down * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n_down * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
        a_ptrs = a_ptr + offs_k[:, None] * stride_ah + offs_n[None, :] * stride_ar
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_start in range(0, H, BLOCK_SIZE_K):
            x_tile = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < H),
                other=0.0,
            )
            a_tile = tl.load(
                a_ptrs,
                mask=((k_start + offs_k[:, None]) < H) & (offs_n[None, :] < R),
                other=0.0,
            )
            acc += tl.dot(x_tile, a_tile)
            x_ptrs += BLOCK_SIZE_K * stride_xh
            a_ptrs += BLOCK_SIZE_K * stride_ah

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yr
        tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < R))
        return

    pid_main = pid - num_tiles_down
    num_pid_m_main = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_main = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group_main = GROUP_SIZE_M * num_pid_n_main
    group_id_main = pid_main // num_pid_in_group_main
    first_pid_m_main = group_id_main * GROUP_SIZE_M
    group_size_m_main = tl.minimum(num_pid_m_main - first_pid_m_main, GROUP_SIZE_M)
    pid_m_main = first_pid_m_main + ((pid_main % num_pid_in_group_main) % group_size_m_main)
    pid_n_main = (pid_main % num_pid_in_group_main) // group_size_m_main

    offs_m = pid_m_main * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n_main * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + offs_n[None, :] * stride_cn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_SIZE_K):
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < H),
            other=0.0,
        )
        c_tile = tl.load(
            c_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x_tile, c_tile)
        x_ptrs += BLOCK_SIZE_K * stride_xh
        c_ptrs += BLOCK_SIZE_K * stride_ch

    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
    tl.store(w_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def down_tile_count(m: int, r: int) -> int:
    """按题目分析时采用的参考块大小估算算子1 program 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_DOWN) * triton.cdiv(r, REFERENCE_BLOCK_N_DOWN)


def main_tile_count(m: int, n: int) -> int:
    """按题目分析时采用的参考块大小估算算子3 program 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_MAIN) * triton.cdiv(n, REFERENCE_BLOCK_N_MAIN)


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("方案2只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("方案2需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


def triton_horizontal_fused_down_main(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    方案2 wrapper：一个 Triton kernel 通过 pid 分区同时产生 Y 和 W。

    返回：
        y: [M, r]
        w: [M, H']
    """
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(r, meta["BLOCK_SIZE_N"])
        + triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )

    _horizontal_fused_kernel[grid](
        x,
        a,
        c,
        y,
        w,
        m,
        h,
        r,
        n,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        c.stride(0),
        c.stride(1),
        y.stride(0),
        y.stride(1),
        w.stride(0),
        w.stride(1),
    )
    return y, w
