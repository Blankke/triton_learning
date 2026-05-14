"""
方案3：column-concatenated GEMM / 列拼接融合。

使用示例：
    from fusion.scheme3_column_concat import triton_logical_concat_down_main
    y, w = triton_logical_concat_down_main(x, a, c, r_pad=0)

说明：
    利用恒等式：
        [X @ A, X @ C] = X @ [A, C]
    本文件保留三类实现口径：
        1. physical_precat：调用方提前构造好 AC，再直接复用 Step 2 的高性能 GEMM
        2. logical [A, C]：不生成 AC，在 Triton kernel 内按列号从 A/C 逻辑加载
        3. logical [C, A]：不生成 AC，在 Triton kernel 内按列号从 C/A 逻辑加载
    三类实现都以 Step 2 的 matmul 为骨架，避免再维护一套低性能简化 kernel。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_learning.kernels.matmul import launch_triton_matmul, matmul_autotune_configs


REFERENCE_BLOCK_M = 16
REFERENCE_BLOCK_N = 128


def concat_tile_count(m: int, total_n: int) -> int:
    """按题目分析时采用的参考块大小估算列拼接 GEMM 的 program 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M) * triton.cdiv(total_n, REFERENCE_BLOCK_N)


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("方案3只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("方案3需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "H", "R", "N", "TOTAL_N", "C_START"],
)
@triton.jit
def _logical_concat_a_first_kernel(
    x_ptr,
    a_ptr,
    c_ptr,
    y_ptr,
    w_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    TOTAL_N: tl.constexpr,
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
    C_START: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """逻辑拼接 [A, C]：使用 Step 2 的 grouped ordering 和 autotune 骨架。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_logic_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    c_cols = offs_logic_n - C_START
    safe_a_cols = tl.minimum(offs_logic_n, R - 1)
    safe_c_cols = tl.minimum(tl.maximum(c_cols, 0), N - 1)

    is_a_col = offs_logic_n < R
    is_c_col = (offs_logic_n >= C_START) & (c_cols < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    a_ptrs = a_ptr + offs_k[:, None] * stride_ah + safe_a_cols[None, :] * stride_ar
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + safe_c_cols[None, :] * stride_cn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_SIZE_K):
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < H),
            other=0.0,
        )
        a_tile = tl.load(
            a_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & is_a_col[None, :],
            other=0.0,
        )
        c_tile = tl.load(
            c_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & is_c_col[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x_tile, a_tile + c_tile)
        x_ptrs += BLOCK_SIZE_K * stride_xh
        a_ptrs += BLOCK_SIZE_K * stride_ah
        c_ptrs += BLOCK_SIZE_K * stride_ch

    safe_y_cols = tl.minimum(offs_logic_n, R - 1)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + safe_y_cols[None, :] * stride_yr
    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + safe_c_cols[None, :] * stride_wn
    tl.store(y_ptrs, accumulator, mask=(offs_m[:, None] < M) & is_a_col[None, :])
    tl.store(w_ptrs, accumulator, mask=(offs_m[:, None] < M) & is_c_col[None, :])


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "H", "R", "N", "TOTAL_N"],
)
@triton.jit
def _logical_concat_c_first_kernel(
    x_ptr,
    a_ptr,
    c_ptr,
    y_ptr,
    w_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    TOTAL_N: tl.constexpr,
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
    """逻辑拼接 [C, A]：让大矩阵 C 从 0 列开始，尾部只保留 A 的剩余列。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_logic_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_cols = offs_logic_n - N
    safe_a_cols = tl.minimum(tl.maximum(a_cols, 0), R - 1)
    safe_c_cols = tl.minimum(offs_logic_n, N - 1)

    is_c_col = offs_logic_n < N
    is_a_col = (offs_logic_n >= N) & (a_cols < R)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + safe_c_cols[None, :] * stride_cn
    a_ptrs = a_ptr + offs_k[:, None] * stride_ah + safe_a_cols[None, :] * stride_ar
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_SIZE_K):
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < H),
            other=0.0,
        )
        c_tile = tl.load(
            c_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & is_c_col[None, :],
            other=0.0,
        )
        a_tile = tl.load(
            a_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & is_a_col[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x_tile, c_tile + a_tile)
        x_ptrs += BLOCK_SIZE_K * stride_xh
        c_ptrs += BLOCK_SIZE_K * stride_ch
        a_ptrs += BLOCK_SIZE_K * stride_ah

    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + safe_c_cols[None, :] * stride_wn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + safe_a_cols[None, :] * stride_yr
    tl.store(w_ptrs, accumulator, mask=(offs_m[:, None] < M) & is_c_col[None, :])
    tl.store(y_ptrs, accumulator, mask=(offs_m[:, None] < M) & is_a_col[None, :])


def triton_physical_concat_precat(x: torch.Tensor, ac: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    物理拼接版本：调用方提前准备好 AC，然后直接复用 Step 2 的高性能 GEMM。

    这个版本不再维护额外 kernel；其本质就是把方案3退化成一次标准 matmul。
    """
    if x.ndim != 2 or ac.ndim != 2:
        raise ValueError("precat 版本只支持二维矩阵。")
    if x.shape[1] != ac.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, ac={tuple(ac.shape)}")
    if not (x.is_cuda and ac.is_cuda):
        raise ValueError("precat 版本需要 CUDA tensor。")
    if x.dtype != ac.dtype:
        raise ValueError("x 和 ac 的 dtype 必须一致。")
    if r <= 0 or r >= ac.shape[1]:
        raise ValueError(f"r 必须落在 (0, {ac.shape[1]}) 内，当前 r={r}")

    x = x.contiguous()
    ac = ac.contiguous()
    t = torch.empty((x.shape[0], ac.shape[1]), device=x.device, dtype=x.dtype)
    launch_triton_matmul(x, ac, t)
    return t[:, :r], t[:, r:]


def triton_logical_concat_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    r_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    逻辑拼接 [A, C]：不生成 AC，在 Triton kernel 内按列号从 A 或 C 加载。

    参数：
        r_pad=0   -> C 紧跟在 r 后面
        r_pad=128 -> C 从 128 对齐边界开始
    """
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    if r_pad != 0 and r_pad < r:
        raise ValueError(f"r_pad 必须为 0 或者不小于 r，当前 r={r}, r_pad={r_pad}")

    c_start = r_pad if r_pad > 0 else r
    total_n = c_start + n
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(total_n, meta["BLOCK_SIZE_N"]),
    )

    _logical_concat_a_first_kernel[grid](
        x,
        a,
        c,
        y,
        w,
        m,
        h,
        r,
        n,
        total_n,
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
        C_START=c_start,
    )
    return y, w


def triton_logical_concat_c_first_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    逻辑拼接 [C, A]：让主干矩阵 C 保持完整对齐，最后一个尾 tile 再处理 A。
    """
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    total_n = n + r
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(total_n, meta["BLOCK_SIZE_N"]),
    )

    _logical_concat_c_first_kernel[grid](
        x,
        a,
        c,
        y,
        w,
        m,
        h,
        r,
        n,
        total_n,
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
