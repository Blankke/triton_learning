"""
Step 3 的单 LoRA 融合 kernel。

使用示例：
    from triton_learning.kernels.fused_matmul_expand import triton_fused_matmul_expand
    out = triton_fused_matmul_expand(x, c, y, b)

说明：
    这里只实现问题1要求的单 LoRA 版本：
        O = X @ C + Y @ B
    其中 Y = X @ A 已经在 kernel 外部先算好。
    这一步只参考 Punica expand 的“对同一个输出 tile 做额外累加”的思想，
    不引入 SGMV、多 adapter、segment 或 token 重排逻辑。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_learning.kernels.matmul import matmul_autotune_configs


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "N", "K", "R"],
)
@triton.jit
def _fused_matmul_expand_kernel(
    x_ptr,  # 输入矩阵 X 的首地址，形状为 [M, K]
    c_ptr,  # 主干权重矩阵 C 的首地址，形状为 [K, N]
    y_ptr,  # LoRA 中间结果矩阵 Y 的首地址，形状为 [M, R]
    b_ptr,  # LoRA expand 权重矩阵 B 的首地址，形状为 [R, N]
    o_ptr,  # 输出矩阵 O 的首地址，形状为 [M, N]
    M: tl.constexpr,  # 输出行数，也是 X / Y / O 的第 0 维
    N: tl.constexpr,  # 输出列数，也是 C / B / O 的第 1 维
    K: tl.constexpr,  # 主干 GEMM 的 reduction 维，对应 X 的列数与 C 的行数
    R: tl.constexpr,  # LoRA expand 的 reduction 维，对应 Y 的列数与 B 的行数
    stride_xm: tl.constexpr,  # X 沿第 0 维（行方向）的 stride
    stride_xk: tl.constexpr,  # X 沿第 1 维（K 方向）的 stride
    stride_ck: tl.constexpr,  # C 沿第 0 维（K 方向）的 stride
    stride_cn: tl.constexpr,  # C 沿第 1 维（列方向）的 stride
    stride_ym: tl.constexpr,  # Y 沿第 0 维（行方向）的 stride
    stride_yr: tl.constexpr,  # Y 沿第 1 维（R 方向）的 stride
    stride_br: tl.constexpr,  # B 沿第 0 维（R 方向）的 stride
    stride_bn: tl.constexpr,  # B 沿第 1 维（列方向）的 stride
    stride_om: tl.constexpr,  # O 沿第 0 维（行方向）的 stride
    stride_on: tl.constexpr,  # O 沿第 1 维（列方向）的 stride
    BLOCK_SIZE_M: tl.constexpr,  # 单个 program 在 M 方向一次处理多少行
    BLOCK_SIZE_N: tl.constexpr,  # 单个 program 在 N 方向一次处理多少列
    BLOCK_SIZE_K: tl.constexpr,  # 两个 reduction 循环共用的分块深度
    GROUP_SIZE_M: tl.constexpr,  # program id 分组参数，用于提升 L2 cache 命中
):
    """
    先算主干 GEMM，再把 expand 分支累加到同一个输出 tile。

    对齐逻辑来自 `docs/fusion.md`：
    两个分支虽然 reduction 维度不同，但都写到同一个 `[BLOCK_M, BLOCK_N]` 输出 tile。
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    c_ptrs = c_ptr + offs_k[:, None] * stride_ck + offs_n[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        c_tile = tl.load(
            c_ptrs,
            mask=((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(x_tile, c_tile)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        c_ptrs += BLOCK_SIZE_K * stride_ck

    # Punica expand 的单 adapter 简化版：直接对同一输出 tile 做第二次累加。
    offs_r = tl.arange(0, BLOCK_SIZE_K)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_r[None, :] * stride_yr
    b_ptrs = b_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn
    for r_start in range(0, R, BLOCK_SIZE_K):
        y_tile = tl.load(
            y_ptrs,
            mask=(offs_m[:, None] < M) & ((r_start + offs_r[None, :]) < R),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=((r_start + offs_r[:, None]) < R) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(y_tile, b_tile)
        y_ptrs += BLOCK_SIZE_K * stride_yr
        b_ptrs += BLOCK_SIZE_K * stride_br

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_fused_matmul_expand(
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper：执行单 LoRA 融合版本 `X@C + Y@B`。"""
    if x.ndim != 2 or c.ndim != 2 or y.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_fused_matmul_expand 只支持二维矩阵。")
    if x.shape[1] != c.shape[0]:
        raise ValueError(f"主干矩阵形状不匹配：x={tuple(x.shape)}, c={tuple(c.shape)}")
    if y.shape[1] != b.shape[0]:
        raise ValueError(f"expand 矩阵形状不匹配：y={tuple(y.shape)}, b={tuple(b.shape)}")
    if x.shape[0] != y.shape[0] or c.shape[1] != b.shape[1]:
        raise ValueError("主干分支与 expand 分支的输出 tile 无法对齐。")
    if not all(tensor.is_cuda for tensor in (x, c, y, b)):
        raise ValueError("triton_fused_matmul_expand 需要 CUDA tensor。")
    if len({x.dtype, c.dtype, y.dtype, b.dtype}) != 1:
        raise ValueError("x、c、y、b 的 dtype 必须一致。")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"暂不支持 dtype={x.dtype}。")

    x = x.contiguous()
    c = c.contiguous()
    y = y.contiguous()
    b = b.contiguous()

    m, k = x.shape
    _, n = c.shape
    _, r = y.shape
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    _fused_matmul_expand_kernel[grid](
        x,
        c,
        y,
        b,
        out,
        m,
        n,
        k,
        r,
        x.stride(0),
        x.stride(1),
        c.stride(0),
        c.stride(1),
        y.stride(0),
        y.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
