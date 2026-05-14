"""
方案1：two-stream concurrent Triton kernels / 空分复用近似实验。

使用示例：
    from fusion.scheme1_spatial_sharing import triton_spatial_sharing_down_main
    y, w = triton_spatial_sharing_down_main(x, a, c, concurrent=True)

说明：
    这不是 single-kernel fusion，而是两个 Triton kernel 分别 launch：
        kernel1: Y = X @ A
        kernel3: W = X @ C
    Python 侧用 torch.cuda.Stream 让两个 Triton kernel 尽量并发执行。
    这只能近似观察空闲 SM 是否被利用，不能严格指定每个 SM 归属。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_M_DOWN = 16
BLOCK_N_DOWN = 16
BLOCK_M_MAIN = 16
BLOCK_N_MAIN = 128
BLOCK_K = 64


@triton.jit
def _down_kernel(
    x_ptr,
    a_ptr,
    y_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ar: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yr: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """计算 Y = X @ A 的一个窄输出 tile。"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(R, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K_SIZE)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    a_ptrs = a_ptr + offs_k[:, None] * stride_ah + offs_n[None, :] * stride_ar
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
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
        x_ptrs += BLOCK_K_SIZE * stride_xh
        a_ptrs += BLOCK_K_SIZE * stride_ah

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yr
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < R))


@triton.jit
def _main_kernel(
    x_ptr,
    c_ptr,
    w_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """计算 W = X @ C 的一个主干输出 tile。"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K_SIZE)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    c_ptrs = c_ptr + offs_k[:, None] * stride_ch + offs_n[None, :] * stride_cn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
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
        x_ptrs += BLOCK_K_SIZE * stride_xh
        c_ptrs += BLOCK_K_SIZE * stride_ch

    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
    tl.store(w_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def down_tile_count(m: int, r: int) -> int:
    """估算算子1的 tile 数。"""
    return triton.cdiv(m, BLOCK_M_DOWN) * triton.cdiv(r, BLOCK_N_DOWN)


def main_tile_count(m: int, n: int) -> int:
    """估算算子3的 tile 数。"""
    return triton.cdiv(m, BLOCK_M_MAIN) * triton.cdiv(n, BLOCK_N_MAIN)


def _launch_down(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> None:
    """发射 Y = X @ A 的 Triton kernel。"""
    m, h = x.shape
    _, r = a.shape
    grid = (down_tile_count(m, r),)
    _down_kernel[grid](
        x,
        a,
        y,
        m,
        h,
        r,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M_DOWN,
        BLOCK_N_DOWN,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )


def _launch_main(x: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> None:
    """发射 W = X @ C 的 Triton kernel。"""
    m, h = x.shape
    _, n = c.shape
    grid = (main_tile_count(m, n),)
    _main_kernel[grid](
        x,
        c,
        w,
        m,
        h,
        n,
        x.stride(0),
        x.stride(1),
        c.stride(0),
        c.stride(1),
        w.stride(0),
        w.stride(1),
        BLOCK_M_MAIN,
        BLOCK_N_MAIN,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous，降低 stride 理解成本。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("方案1只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("方案1需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


def triton_spatial_sharing_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    concurrent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    方案1 wrapper：分别 launch 两个 Triton kernel，可选择双 stream 并发。

    返回：
        y: [M, r]
        w: [M, H']
    """
    x, a, c = _check_inputs(x, a, c)
    m = x.shape[0]
    r = a.shape[1]
    n = c.shape[1]
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    if not concurrent:
        _launch_down(x, a, y)
        _launch_main(x, c, w)
        return y, w

    current = torch.cuda.current_stream()
    down_stream = torch.cuda.Stream(device=x.device)
    main_stream = torch.cuda.Stream(device=x.device)
    down_stream.wait_stream(current)
    main_stream.wait_stream(current)

    with torch.cuda.stream(down_stream):
        _launch_down(x, a, y)
    with torch.cuda.stream(main_stream):
        _launch_main(x, c, w)

    current.wait_stream(down_stream)
    current.wait_stream(main_stream)
    return y, w

