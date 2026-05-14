"""
方案3：column-concatenated GEMM / 列拼接融合。

使用示例：
    from fusion.scheme3_column_concat import triton_logical_concat_c_first_down_main
    y, w = triton_logical_concat_c_first_down_main(x, a, c)

说明：
    利用恒等式：
        [X @ A, X @ C] = X @ [A, C]
    本文件提供三类执行口径：
        1. physical_precat: 预先传入已经拼好的 AC，再用 Triton GEMM
        2. logical [A, C]: 不生成 AC，在 Triton kernel 内按列号从 A/C 加载
        3. logical [C, A]: 不生成 AC，在 Triton kernel 内按列号从 C/A 加载
    其中 `[C, A]` 版本会让大矩阵 C 从 0 列开始，最后一个尾 tile 只保留 A 的 8 列。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 128
BLOCK_K = 64


@triton.jit
def _physical_concat_kernel(
    x_ptr,
    ac_ptr,
    t_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    TOTAL_N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_ach: tl.constexpr,
    stride_acn: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_tn: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """普通 GEMM：T = X @ AC。"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_N_SIZE)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    offs_n = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    offs_k = tl.arange(0, BLOCK_K_SIZE)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    ac_ptrs = ac_ptr + offs_k[:, None] * stride_ach + offs_n[None, :] * stride_acn
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < H),
            other=0.0,
        )
        ac_tile = tl.load(
            ac_ptrs,
            mask=((k_start + offs_k[:, None]) < H) & (offs_n[None, :] < TOTAL_N),
            other=0.0,
        )
        acc += tl.dot(x_tile, ac_tile)
        x_ptrs += BLOCK_K_SIZE * stride_xh
        ac_ptrs += BLOCK_K_SIZE * stride_ach

    t_ptrs = t_ptr + offs_m[:, None] * stride_tm + offs_n[None, :] * stride_tn
    tl.store(t_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < TOTAL_N))


@triton.jit
def _logical_concat_kernel(
    x_ptr,
    a_ptr,
    c_ptr,
    y_ptr,
    w_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    R_PAD: tl.constexpr,
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
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """逻辑拼接 GEMM：不生成 AC，按输出列号选择 A 或 C。"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_N_SIZE)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    offs_tn = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    offs_k = tl.arange(0, BLOCK_K_SIZE)

    # r_pad=0 时 C 从 R 后面开始；r_pad>0 时 C 从 R_PAD 对齐边界开始。
    c_start = tl.where(R_PAD > 0, R_PAD, R)
    c_cols = offs_tn - c_start
    is_a_col = offs_tn < R
    is_c_col = (offs_tn >= c_start) & (c_cols < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    a_ptrs = a_ptr + offs_k[:, None] * stride_ah + offs_tn[None, :] * stride_ar
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + c_cols[None, :] * stride_cn
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
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
        acc += tl.dot(x_tile, a_tile + c_tile)
        x_ptrs += BLOCK_K_SIZE * stride_xh
        a_ptrs += BLOCK_K_SIZE * stride_ah
        c_ptrs += BLOCK_K_SIZE * stride_ch

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_tn[None, :] * stride_yr
    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + c_cols[None, :] * stride_wn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & is_a_col[None, :])
    tl.store(w_ptrs, acc, mask=(offs_m[:, None] < M) & is_c_col[None, :])


def concat_tile_count(m: int, total_n: int) -> int:
    """列拼接 GEMM 的 program 数。"""
    return triton.cdiv(m, BLOCK_M) * triton.cdiv(total_n, BLOCK_N)


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


def _run_physical_concat_gemm(x: torch.Tensor, ac: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """给定已经拼好的 AC，执行 T = X @ AC，并返回 Y/W 视图。"""
    m, h = x.shape
    total_n = ac.shape[1]
    t = torch.empty((m, total_n), device=x.device, dtype=x.dtype)
    grid = (concat_tile_count(m, total_n),)
    _physical_concat_kernel[grid](
        x,
        ac,
        t,
        m,
        h,
        total_n,
        x.stride(0),
        x.stride(1),
        ac.stride(0),
        ac.stride(1),
        t.stride(0),
        t.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return t[:, :r], t[:, r:]


def triton_physical_concat_precat(x: torch.Tensor, ac: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """物理拼接版本：调用方提前准备好 AC，用于模拟权重预处理。"""
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
    return _run_physical_concat_gemm(x.contiguous(), ac.contiguous(), r)


def triton_logical_concat_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    r_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    逻辑拼接版本：不生成 AC，在 Triton kernel 内按列号从 A 或 C 加载。

    r_pad=0 表示 C 紧跟在 r 后面；r_pad=128 表示 C 从 BLOCK_N 对齐边界开始。
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
    grid = (concat_tile_count(m, total_n),)

    _logical_concat_kernel[grid](
        x,
        a,
        c,
        y,
        w,
        m,
        h,
        r,
        n,
        r_pad,
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
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y, w


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
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """逻辑拼接 GEMM：按 [C, A] 的列顺序组织输出。"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_N_SIZE)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    offs_tn = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    offs_k = tl.arange(0, BLOCK_K_SIZE)

    a_cols = offs_tn - N
    is_c_col = offs_tn < N
    is_a_col = (offs_tn >= N) & (a_cols < R)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + offs_tn[None, :] * stride_cn
    a_ptrs = a_ptr + offs_k[:, None] * stride_ah + a_cols[None, :] * stride_ar
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
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
        acc += tl.dot(x_tile, c_tile + a_tile)
        x_ptrs += BLOCK_K_SIZE * stride_xh
        c_ptrs += BLOCK_K_SIZE * stride_ch
        a_ptrs += BLOCK_K_SIZE * stride_ah

    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_tn[None, :] * stride_wn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + a_cols[None, :] * stride_yr
    tl.store(w_ptrs, acc, mask=(offs_m[:, None] < M) & is_c_col[None, :])
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & is_a_col[None, :])


def triton_logical_concat_c_first_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    逻辑拼接版本：按 [C, A] 顺序组织列，不需要额外 padding。

    返回：
        y: [M, r]
        w: [M, N]
    """
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    total_n = n + r
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (concat_tile_count(m, total_n),)

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
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y, w
