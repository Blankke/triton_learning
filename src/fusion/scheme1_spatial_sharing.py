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

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import triton

from triton_learning.benchmark_utils import cuda_nvtx_range
from triton_learning.kernels.matmul import launch_triton_matmul


REFERENCE_BLOCK_M_DOWN = 16
REFERENCE_BLOCK_N_DOWN = 16
REFERENCE_BLOCK_M_MAIN = 16
REFERENCE_BLOCK_N_MAIN = 128


@dataclass(frozen=True)
class _ConcurrentLaunchResources:
    """按 device 复用的双 stream 与同步 event。"""

    down_stream: torch.cuda.Stream
    main_stream: torch.cuda.Stream
    ready_event: torch.cuda.Event
    down_done_event: torch.cuda.Event
    main_done_event: torch.cuda.Event


# 方案1的重点是比较“两个 Triton GEMM 并发 launch”本身是否有收益。
# 如果每次调用都新建 stream，会把 stream 创建成本混进 benchmark 热路径，
# 让测量结果偏向悲观。因此这里按 device 缓存并复用 stream / event。
_RESOURCE_CACHE: dict[int, _ConcurrentLaunchResources] = {}


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


def _get_concurrent_launch_resources(device: torch.device) -> _ConcurrentLaunchResources:
    """按 device 复用双 stream 和 event，避免热路径里反复创建对象。"""
    device_index = _device_index(device)
    cached = _RESOURCE_CACHE.get(device_index)
    if cached is not None:
        return cached
    resources = _ConcurrentLaunchResources(
        down_stream=torch.cuda.Stream(device=device_index),
        main_stream=torch.cuda.Stream(device=device_index),
        ready_event=torch.cuda.Event(enable_timing=False),
        down_done_event=torch.cuda.Event(enable_timing=False),
        main_done_event=torch.cuda.Event(enable_timing=False),
    )
    _RESOURCE_CACHE[device_index] = resources
    return resources


def launch_triton_matmul_pair(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    concurrent: bool,
    range_titles: tuple[str, str] | None = None,
) -> None:
    """
    按顺序或双 stream 并发 launch 两个 Triton GEMM。

    这里把并发路径里的同步方式统一成“显式 event 边界”：
        1. 先记录当前 stream 已经完成到哪里；
        2. 两个工作 stream 只等待这一个 ready event；
        3. 两个 kernel launch 完成后，各自记录 done event；
        4. 当前 stream 只在尾部等待这两个 done event。

    这样可以避免把默认流上的宽泛依赖传播进两个工作 stream，
    更容易在 Nsight Systems 里看到真实的双 stream launch 关系。
    """
    x, a, c = _check_inputs(x, a, c)
    if y.shape != (x.shape[0], a.shape[1]) or w.shape != (x.shape[0], c.shape[1]):
        raise ValueError(
            "输出 tensor 形状不匹配："
            f"y={tuple(y.shape)}, 期望={(x.shape[0], a.shape[1])}; "
            f"w={tuple(w.shape)}, 期望={(x.shape[0], c.shape[1])}"
        )
    if y.dtype != x.dtype or w.dtype != x.dtype:
        raise ValueError("输出 tensor 的 dtype 必须与输入一致。")
    if not y.is_cuda or not w.is_cuda:
        raise ValueError("输出 tensor 必须位于 CUDA 上。")

    if range_titles is not None and len(range_titles) != 2:
        raise ValueError("range_titles 必须恰好包含两个标题。")

    down_ctx = cuda_nvtx_range(range_titles[0]) if range_titles is not None else nullcontext()
    main_ctx = cuda_nvtx_range(range_titles[1]) if range_titles is not None else nullcontext()

    if not concurrent:
        with down_ctx:
            launch_triton_matmul(x, a, y)
        with main_ctx:
            launch_triton_matmul(x, c, w)
        return

    current = torch.cuda.current_stream(device=x.device)
    resources = _get_concurrent_launch_resources(x.device)
    resources.ready_event.record(current)
    resources.down_stream.wait_event(resources.ready_event)
    resources.main_stream.wait_event(resources.ready_event)

    with torch.cuda.stream(resources.down_stream):
        with down_ctx:
            launch_triton_matmul(x, a, y)
        resources.down_done_event.record(resources.down_stream)

    with torch.cuda.stream(resources.main_stream):
        with main_ctx:
            launch_triton_matmul(x, c, w)
        resources.main_done_event.record(resources.main_stream)

    current.wait_event(resources.down_done_event)
    current.wait_event(resources.main_done_event)


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
    launch_triton_matmul_pair(x, a, c, y, w, concurrent=concurrent)
    return y, w
