"""
三个融合方案共用的 benchmark 工具。

使用示例：
    from fusion.common import parse_common_args, create_inputs, check_outputs

说明：
    本文件只放参数解析、参考结果、误差检查和结果打印。
    三个方案的 Triton kernel 分别保存在各自独立文件中。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from triton_learning.problem_spec import DEFAULT_GATEUP_PROBLEM, ProblemShape, resolve_dtype
from triton_learning.reference_ops import run_reference_pipeline


class Inputs(NamedTuple):
    """保存本实验需要的四个输入矩阵。"""

    x: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


class References(NamedTuple):
    """保存 PyTorch 参考结果。"""

    y: torch.Tensor
    w: torch.Tensor
    o: torch.Tensor


@dataclass(frozen=True)
class ErrorReport:
    """保存一次正确性检查的主要误差。"""

    y_correct: bool
    w_correct: bool
    o_correct: bool
    y_max_abs: float
    w_max_abs: float
    o_max_abs: float
    y_max_rel: float
    w_max_rel: float
    o_max_rel: float

    @property
    def all_correct(self) -> bool:
        """三个输出都正确时，整体才算正确。"""
        return self.y_correct and self.w_correct and self.o_correct


def parse_common_args(
    description: str,
    default_output: str,
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """解析三个 benchmark 共用的命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--m", type=int, default=DEFAULT_GATEUP_PROBLEM.m, help="输入 token/batch 维度 M。")
    parser.add_argument("--h", type=int, default=DEFAULT_GATEUP_PROBLEM.h, help="输入隐藏维度 H。")
    parser.add_argument("--n", type=int, default=DEFAULT_GATEUP_PROBLEM.n, help="主干输出维度 H'。")
    parser.add_argument("--r", type=int, default=DEFAULT_GATEUP_PROBLEM.r, help="低秩维度 r。")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default=DEFAULT_GATEUP_PROBLEM.dtype_name,
        help="矩阵元素格式。",
    )
    parser.add_argument("--warmup", type=int, default=30, help="正式计时前的预热次数。")
    parser.add_argument("--repeat", type=int, default=100, help="正式计时重复次数。")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="进入 profiling 模式：不写 CSV，不做统计，只执行少量带 NVTX 标记的 workload。",
    )
    parser.add_argument("--profile-warmup", type=int, default=1, help="profiling 模式下的预热次数。")
    parser.add_argument("--profile-repeat", type=int, default=1, help="profiling 模式下实际执行次数。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--output", type=Path, default=Path(default_output), help="CSV 结果输出路径。")
    if configure_parser is not None:
        configure_parser(parser)
    return parser.parse_args()


def shape_from_args(args: argparse.Namespace) -> ProblemShape:
    """从命令行参数构造统一形状对象。"""
    return ProblemShape(m=args.m, h=args.h, n=args.n, r=args.r, dtype_name=args.dtype)


def create_inputs(shape: ProblemShape, dtype: torch.dtype, device: torch.device, seed: int) -> Inputs:
    """按固定随机种子创建输入矩阵，保证三个方案可横向比较。"""
    torch.manual_seed(seed)
    return Inputs(
        x=torch.randn((shape.m, shape.h), device=device, dtype=dtype),
        a=torch.randn((shape.h, shape.r), device=device, dtype=dtype),
        b=torch.randn((shape.r, shape.n), device=device, dtype=dtype),
        c=torch.randn((shape.h, shape.n), device=device, dtype=dtype),
    )


def compute_references(inputs: Inputs) -> References:
    """使用 PyTorch 计算权威参考结果。"""
    y = inputs.x @ inputs.a
    w = inputs.x @ inputs.c
    o = w + y @ inputs.b
    return References(y=y, w=w, o=o)


def run_pytorch_pair(inputs: Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 PyTorch/cuBLAS 的两步输出：Y = X@A, W = X@C。"""
    y = inputs.x @ inputs.a
    w = inputs.x @ inputs.c
    return y, w


def compute_full_output(y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """融合 kernel 产出 Y/W 后，继续执行后续算子2和加法。"""
    return w + y @ b


def run_pytorch_full_pipeline(inputs: Inputs) -> torch.Tensor:
    """执行原版 PyTorch/cuBLAS 串行 full pipeline。"""
    return run_reference_pipeline(inputs.x, inputs.a, inputs.b, inputs.c)


def tolerance_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """给不同 dtype 设置用于实验的正确性容忍度。"""
    if dtype == torch.float16:
        return 1.0, 5e-2
    if dtype == torch.bfloat16:
        return 2.0, 8e-2
    return 1e-4, 1e-4


def _max_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """计算最大相对误差，避免 expected 接近 0 时除零。"""
    return ((actual - expected).abs() / expected.abs().clamp_min(1e-5)).max().item()


def check_outputs(
    y: torch.Tensor,
    w: torch.Tensor,
    o: torch.Tensor,
    refs: References,
    dtype: torch.dtype,
) -> ErrorReport:
    """检查 Y、W 和最终 O 三个结果。"""
    atol, rtol = tolerance_for_dtype(dtype)
    return ErrorReport(
        y_correct=torch.allclose(y, refs.y, atol=atol, rtol=rtol),
        w_correct=torch.allclose(w, refs.w, atol=atol, rtol=rtol),
        o_correct=torch.allclose(o, refs.o, atol=atol, rtol=rtol),
        y_max_abs=(y - refs.y).abs().max().item(),
        w_max_abs=(w - refs.w).abs().max().item(),
        o_max_abs=(o - refs.o).abs().max().item(),
        y_max_rel=_max_rel_error(y, refs.y),
        w_max_rel=_max_rel_error(w, refs.w),
        o_max_rel=_max_rel_error(o, refs.o),
    )


def print_shape(title: str, shape: ProblemShape) -> None:
    """打印当前 benchmark 使用的矩阵形状。"""
    print(f"\n{title}")
    print(f"  X: [{shape.m}, {shape.h}]")
    print(f"  A: [{shape.h}, {shape.r}]")
    print(f"  C: [{shape.h}, {shape.n}]")
    print(f"  B: [{shape.r}, {shape.n}]")
    print(f"  dtype: {shape.dtype_name}")


def add_error_fields(row: dict[str, object], errors: ErrorReport) -> dict[str, object]:
    """把正确性字段追加到 CSV 行。"""
    row.update(
        {
            "correct": errors.all_correct,
            "y_correct": errors.y_correct,
            "w_correct": errors.w_correct,
            "o_correct": errors.o_correct,
            "y_max_abs": errors.y_max_abs,
            "w_max_abs": errors.w_max_abs,
            "o_max_abs": errors.o_max_abs,
            "y_max_rel": errors.y_max_rel,
            "w_max_rel": errors.w_max_rel,
            "o_max_rel": errors.o_max_rel,
        }
    )
    return row


def add_shape_fields(row: dict[str, object], shape: ProblemShape, warmup: int, repeat: int) -> dict[str, object]:
    """把形状与计时参数追加到 CSV 行。"""
    row.update(
        {
            "m": shape.m,
            "h": shape.h,
            "n": shape.n,
            "r": shape.r,
            "dtype": shape.dtype_name,
            "warmup": warmup,
            "repeat": repeat,
        }
    )
    return row


def print_error_report(errors: ErrorReport) -> None:
    """打印正确性检查摘要。"""
    print(f"  correct: {errors.all_correct}")
    print(f"  Y max_abs={errors.y_max_abs:.6f}, max_rel={errors.y_max_rel:.6f}")
    print(f"  W max_abs={errors.w_max_abs:.6f}, max_rel={errors.w_max_rel:.6f}")
    print(f"  O max_abs={errors.o_max_abs:.6f}, max_rel={errors.o_max_rel:.6f}")


def dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    """解析 dtype，并保持入口文件更清爽。"""
    return resolve_dtype(args.dtype)
