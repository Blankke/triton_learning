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
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


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


def measure_cuda_time(
    title: str,
    fn: Callable[[], torch.Tensor],
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
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tflops(m: int, n: int, k: int, ms: float) -> float:
    """根据 GEMM 公式估算 TFLOPS。"""
    seconds = ms * 1e-3
    return 2.0 * m * n * k / seconds / 1e12
