"""
方案2：single-kernel horizontal fusion via pid partition。

使用示例：
    from fusion.scheme2_horizontal_fusion import triton_horizontal_fused_down_main
    y, w = triton_horizontal_fused_down_main(x, a, c)

说明：
    一个 Triton kernel 内部把 program id 分成两段：
        pid < num_tiles_1      -> 计算 Y = X @ A 的一个 tile
        pid >= num_tiles_1     -> 计算 W = X @ C 的一个 tile
    该方案不做物理拼接，也不严格指定 SM，只减少一次单独 kernel launch。
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
    NUM_TILES_DOWN: tl.constexpr,
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
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M3: tl.constexpr,
    BLOCK_N3: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
):
    """按 pid 区间分别计算 Y tile 或 W tile。"""
    pid = tl.program_id(axis=0)

    if pid < NUM_TILES_DOWN:
        num_pid_n1 = tl.cdiv(R, BLOCK_N1)
        pid_m1 = pid // num_pid_n1
        pid_n1 = pid % num_pid_n1

        offs_m1 = pid_m1 * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        offs_n1 = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
        offs_k1 = tl.arange(0, BLOCK_K_SIZE)

        x_ptrs1 = x_ptr + offs_m1[:, None] * stride_xm + offs_k1[None, :] * stride_xh
        a_ptrs = a_ptr + offs_k1[:, None] * stride_ah + offs_n1[None, :] * stride_ar
        acc1 = tl.zeros((BLOCK_M1, BLOCK_N1), dtype=tl.float32)

        for k_start in range(0, H, BLOCK_K_SIZE):
            x_tile = tl.load(
                x_ptrs1,
                mask=(offs_m1[:, None] < M) & ((k_start + offs_k1[None, :]) < H),
                other=0.0,
            )
            a_tile = tl.load(
                a_ptrs,
                mask=((k_start + offs_k1[:, None]) < H) & (offs_n1[None, :] < R),
                other=0.0,
            )
            acc1 += tl.dot(x_tile, a_tile)
            x_ptrs1 += BLOCK_K_SIZE * stride_xh
            a_ptrs += BLOCK_K_SIZE * stride_ah

        y_ptrs = y_ptr + offs_m1[:, None] * stride_ym + offs_n1[None, :] * stride_yr
        tl.store(y_ptrs, acc1, mask=(offs_m1[:, None] < M) & (offs_n1[None, :] < R))
        return

    pid3 = pid - NUM_TILES_DOWN
    num_pid_n3 = tl.cdiv(N, BLOCK_N3)
    pid_m3 = pid3 // num_pid_n3
    pid_n3 = pid3 % num_pid_n3

    offs_m3 = pid_m3 * BLOCK_M3 + tl.arange(0, BLOCK_M3)
    offs_n3 = pid_n3 * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    offs_k3 = tl.arange(0, BLOCK_K_SIZE)

    x_ptrs3 = x_ptr + offs_m3[:, None] * stride_xm + offs_k3[None, :] * stride_xh
    c_ptrs = c_ptr + offs_k3[:, None] * stride_ch + offs_n3[None, :] * stride_cn
    acc3 = tl.zeros((BLOCK_M3, BLOCK_N3), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_SIZE):
        x_tile = tl.load(
            x_ptrs3,
            mask=(offs_m3[:, None] < M) & ((k_start + offs_k3[None, :]) < H),
            other=0.0,
        )
        c_tile = tl.load(
            c_ptrs,
            mask=((k_start + offs_k3[:, None]) < H) & (offs_n3[None, :] < N),
            other=0.0,
        )
        acc3 += tl.dot(x_tile, c_tile)
        x_ptrs3 += BLOCK_K_SIZE * stride_xh
        c_ptrs += BLOCK_K_SIZE * stride_ch

    w_ptrs = w_ptr + offs_m3[:, None] * stride_wm + offs_n3[None, :] * stride_wn
    tl.store(w_ptrs, acc3, mask=(offs_m3[:, None] < M) & (offs_n3[None, :] < N))


def down_tile_count(m: int, r: int) -> int:
    """方案2中算子1的 program 数。"""
    return triton.cdiv(m, BLOCK_M_DOWN) * triton.cdiv(r, BLOCK_N_DOWN)


def main_tile_count(m: int, n: int) -> int:
    """方案2中算子3的 program 数。"""
    return triton.cdiv(m, BLOCK_M_MAIN) * triton.cdiv(n, BLOCK_N_MAIN)


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

    num_tiles_down = down_tile_count(m, r)
    num_tiles_main = main_tile_count(m, n)
    grid = (num_tiles_down + num_tiles_main,)

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
        num_tiles_down,
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
        BLOCK_M_DOWN,
        BLOCK_N_DOWN,
        BLOCK_M_MAIN,
        BLOCK_N_MAIN,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y, w

