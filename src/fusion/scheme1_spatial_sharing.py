"""
方案1：two-stream concurrent Triton kernels / 空分复用近似实验。

使用示例：
    from fusion.scheme1_spatial_sharing import triton_spatial_sharing_down_main
    y, w = triton_spatial_sharing_down_main(x, a, c, concurrent=True)

说明：
    这不是 single-kernel fusion，而是两个 Triton kernel 分别 launch：
        kernel1: Y = X @ A
        kernel3: W = X @ C
    这里不再维护一份“简化版 matmul”，而是直接复用 Step 2 的高性能 GEMM。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from triton_learning.kernels.matmul import launch_triton_matmul


REFERENCE_BLOCK_M_DOWN = 16
REFERENCE_BLOCK_N_DOWN = 16
REFERENCE_BLOCK_M_MAIN = 16
REFERENCE_BLOCK_N_MAIN = 128


@dataclass(frozen=True)
class _StreamPair:
    """按 device 复用的双 stream 句柄。"""

    down_stream: torch.cuda.Stream
    main_stream: torch.cuda.Stream


# 方案1的重点是比较“两个 Triton GEMM 并发 launch”本身是否有收益。
# 如果每次调用都新建 stream，会把 stream 创建成本混进 benchmark 热路径，
# 让测量结果偏向悲观。因此这里按 device 缓存并复用 stream。
_STREAM_CACHE: dict[int, _StreamPair] = {}


def down_tile_count(m: int, r: int) -> int:
    """按题目分析时采用的参考块大小估算算子1 tile 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_DOWN) * triton.cdiv(r, REFERENCE_BLOCK_N_DOWN)


def main_tile_count(m: int, n: int) -> int:
    """按题目分析时采用的参考块大小估算算子3 tile 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_MAIN) * triton.cdiv(n, REFERENCE_BLOCK_N_MAIN)


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous，便于后续复用 Step 2 GEMM。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("方案1只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("方案1需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


def _device_index(device: torch.device) -> int:
    """把 torch.device 归一化为可缓存的 CUDA device index。"""
    if device.type != "cuda":
        raise ValueError("方案1只支持 CUDA device。")
    return torch.cuda.current_device() if device.index is None else device.index


def _get_stream_pair(device: torch.device) -> _StreamPair:
    """按 device 复用双 stream，避免每次调用都重复创建。"""
    device_index = _device_index(device)
    cached = _STREAM_CACHE.get(device_index)
    if cached is not None:
        return cached
    stream_pair = _StreamPair(
        down_stream=torch.cuda.Stream(device=device_index),
        main_stream=torch.cuda.Stream(device=device_index),
    )
    _STREAM_CACHE[device_index] = stream_pair
    return stream_pair


def triton_spatial_sharing_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    concurrent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    方案1 wrapper：分别 launch 两个 Triton kernel，可选择双 stream 并发。

    两个 matmul 都直接复用 Step 2 的高性能 Triton GEMM。
    """
    x, a, c = _check_inputs(x, a, c)
    m = x.shape[0]
    r = a.shape[1]
    n = c.shape[1]
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    if not concurrent:
        launch_triton_matmul(x, a, y)
        launch_triton_matmul(x, c, w)
        return y, w

    current = torch.cuda.current_stream()
    stream_pair = _get_stream_pair(x.device)
    down_stream = stream_pair.down_stream
    main_stream = stream_pair.main_stream
    down_stream.wait_stream(current)
    main_stream.wait_stream(current)

    with torch.cuda.stream(down_stream):
        launch_triton_matmul(x, a, y)
    with torch.cuda.stream(main_stream):
        launch_triton_matmul(x, c, w)

    current.wait_stream(down_stream)
    current.wait_stream(main_stream)
    return y, w
