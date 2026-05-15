"""
GPU benchmark 复用工具。

使用示例：
    from triton_learning.benchmark_utils import measure_cuda_time, require_cuda
    require_cuda()

说明：
    本模块统一处理 CUDA 环境检查、计时统计与 CSV 结果写出，
    让 Step 1/2/3 只保留实验逻辑本身。
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

STEADY_STATE_CAPTURE_RANGE = "steady_state_capture"


@dataclass(frozen=True)
class TimingResult:
    """保存一次 benchmark 的中位数与波动区间。"""

    median_ms: float
    p20_ms: float
    p80_ms: float


def require_cuda() -> None:
    """实验必须在能看到 CUDA 的 PyTorch 环境中运行。"""
    import sys

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 当前看不到 CUDA GPU，请先修复环境。")


@contextmanager
def cuda_nvtx_range(title: str):
    """
    为 Nsight Systems / Nsight Compute 加一层可读的 NVTX 标记。

    这样在远端 A100 上做 profiling 时，可以直接按 range 名字定位：
        op1_XA / scheme2_horizontal_pair / scheme3_physical_precat_pair
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(title)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def run_profiled_callable(
    title: str,
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    repeat: int,
) -> None:
    """
    只执行少量带 NVTX 的 workload，不做 benchmark 统计。

    这个入口专门给 nsys / ncu 使用：
        1. warmup 阶段先把 Triton kernel 编译好；
        2. repeat 阶段再用 NVTX 包住真正想观察的 kernel。
    """
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc=f"{title} profile warmup", leave=False):
            fn()
        torch.cuda.synchronize()

        # 外层 steady_state range 专门给 Nsight Systems 做 capture-range 过滤，
        # 这样最终报告里不会再混入 warmup / autotune 阶段的时间线。
        with cuda_nvtx_range(STEADY_STATE_CAPTURE_RANGE):
            for _ in tqdm(range(repeat), desc=f"{title} profile", leave=False):
                with cuda_nvtx_range(title):
                    fn()
                torch.cuda.synchronize()


def measure_cuda_time(
    title: str,
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    repeat: int,
) -> TimingResult:
    """
    使用 CUDA Event 统计一段 GPU 计算的耗时。

    warmup 只负责把 kernel 与 allocator 预热到稳定状态，
    repeat 才是最终记入统计的正式测量。
    """
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc=f"{title} warmup", leave=False):
            fn()
        torch.cuda.synchronize()

        elapsed_ms: list[float] = []
        for _ in tqdm(range(repeat), desc=f"{title} bench", leave=False):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms.append(start.elapsed_time(end))

    sorted_ms = sorted(elapsed_ms)
    p20_idx = int(0.2 * (len(sorted_ms) - 1))
    p80_idx = int(0.8 * (len(sorted_ms) - 1))
    return TimingResult(
        median_ms=statistics.median(sorted_ms),
        p20_ms=sorted_ms[p20_idx],
        p80_ms=sorted_ms[p80_idx],
    )


def append_csv(path: Path, row: dict[str, object]) -> None:
    """向 benchmark CSV 追加一行结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = True
    file_mode = "a"
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
        existing_header = first_line.split(",") if first_line else []
        if existing_header == fieldnames:
            write_header = False
        else:
            # 当 benchmark schema 变化时，直接重写文件，避免新旧列错位。
            file_mode = "w"

    with path.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tflops(m: int, n: int, k: int, ms: float) -> float:
    """根据 GEMM 公式估算 TFLOPS。"""
    seconds = ms * 1e-3
    return 2.0 * m * n * k / seconds / 1e12
