"""
Step 1：用 PyTorch 完成 baseline 验证。

使用示例：
    python -m triton_learning.bench_step1_baseline --m 64 --h 4096 --n 28672 --r 8 --dtype fp16

说明：
    本脚本只做原始串行流程：
        Y = X @ A
        Z = Y @ B
        W = X @ C
        O = W + Z
    重点输出完整串行流程的总时间，同时保留每个子算子的耗时，方便后续对比 Step 3。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda, run_profiled_callable
from triton_learning.problem_spec import DEFAULT_GATEUP_PROBLEM, ProblemShape, resolve_dtype
from triton_learning.reference_ops import (
    compute_lora_down,
    compute_lora_expand,
    compute_main_matmul,
    compute_output_add,
    create_problem_tensors,
    run_reference_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1：测量三个独立矩阵乘的 baseline。")
    parser.add_argument("--m", type=int, default=DEFAULT_GATEUP_PROBLEM.m, help="输入 token/batch 维度 M。")
    parser.add_argument("--h", type=int, default=DEFAULT_GATEUP_PROBLEM.h, help="隐藏层维度 H。")
    parser.add_argument("--n", type=int, default=DEFAULT_GATEUP_PROBLEM.n, help="输出维度 N。")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/step1_baseline.csv"),
        help="CSV 结果输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()

    shape = ProblemShape(m=args.m, h=args.h, n=args.n, r=args.r, dtype_name=args.dtype)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")
    tensors = create_problem_tensors(shape=shape, dtype=dtype, device=device)

    print("\nStep 1 baseline 矩阵形状：")
    print(f"  X: [{shape.m}, {shape.h}]")
    print(f"  A: [{shape.h}, {shape.r}]")
    print(f"  B: [{shape.r}, {shape.n}]")
    print(f"  C: [{shape.h}, {shape.n}]")
    print(f"  dtype: {shape.dtype_name}")

    def run_op1_once() -> torch.Tensor:
        return compute_lora_down(tensors.x, tensors.a)

    y_for_op2 = compute_lora_down(tensors.x, tensors.a)

    def run_op2_once() -> torch.Tensor:
        return compute_lora_expand(y_for_op2, tensors.b)

    def run_op3_once() -> torch.Tensor:
        return compute_main_matmul(tensors.x, tensors.c)

    z_for_add = compute_lora_expand(y_for_op2, tensors.b)
    w_for_add = compute_main_matmul(tensors.x, tensors.c)

    def run_add_once() -> torch.Tensor:
        return compute_output_add(w_for_add, z_for_add)

    def run_full_once() -> torch.Tensor:
        return run_reference_pipeline(tensors.x, tensors.a, tensors.b, tensors.c)

    if args.profile_only:
        print("\nStep 1 profiling 模式：不写 CSV，只执行 NVTX 标记的 workload。")
        run_profiled_callable(
            "step1/op1_xa",
            run_op1_once,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        run_profiled_callable(
            "step1/op2_yb",
            run_op2_once,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        run_profiled_callable(
            "step1/op3_xc",
            run_op3_once,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        run_profiled_callable(
            "step1/add",
            run_add_once,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        run_profiled_callable(
            "step1/full_pipeline",
            run_full_once,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        print("Step 1 profiling workload 执行完成。")
        return

    op1 = measure_cuda_time("op1: Y = X @ A", run_op1_once, args.warmup, args.repeat)
    op2 = measure_cuda_time("op2: Z = Y @ B", run_op2_once, args.warmup, args.repeat)
    op3 = measure_cuda_time("op3: W = X @ C", run_op3_once, args.warmup, args.repeat)
    add = measure_cuda_time("add: O = W + Z", run_add_once, args.warmup, args.repeat)
    full = measure_cuda_time("full: O = X@C + (X@A)@B", run_full_once, args.warmup, args.repeat)

    total_split_ms = op1.median_ms + op2.median_ms + op3.median_ms + add.median_ms
    row = {
        "m": shape.m,
        "h": shape.h,
        "n": shape.n,
        "r": shape.r,
        "dtype": shape.dtype_name,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "op1_xa_ms": op1.median_ms,
        "op2_yb_ms": op2.median_ms,
        "op3_xc_ms": op3.median_ms,
        "add_ms": add.median_ms,
        "sum_individual_ms": total_split_ms,
        "full_pipeline_ms": full.median_ms,
        "op1_xa_p20_ms": op1.p20_ms,
        "op1_xa_p80_ms": op1.p80_ms,
        "op2_yb_p20_ms": op2.p20_ms,
        "op2_yb_p80_ms": op2.p80_ms,
        "op3_xc_p20_ms": op3.p20_ms,
        "op3_xc_p80_ms": op3.p80_ms,
        "add_p20_ms": add.p20_ms,
        "add_p80_ms": add.p80_ms,
        "full_p20_ms": full.p20_ms,
        "full_p80_ms": full.p80_ms,
    }
    append_csv(args.output, row)

    print("\nStep 1 baseline 结果：")
    print(f"  完整串行总时间 full_pipeline_ms: {full.median_ms:.6f} ms")
    print(f"  分项相加 sum_individual_ms: {total_split_ms:.6f} ms")
    print(f"  op1 X@A: {op1.median_ms:.6f} ms")
    print(f"  op2 Y@B: {op2.median_ms:.6f} ms")
    print(f"  op3 X@C: {op3.median_ms:.6f} ms")
    print(f"  add W+Z: {add.median_ms:.6f} ms")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
