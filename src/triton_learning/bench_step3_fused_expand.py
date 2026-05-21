"""
Step 3：开始实现单 LoRA 场景的 fused expand + main GEMM。

使用示例：
    python -m triton_learning.bench_step3_fused_expand --m 64 --h 4096 --n 28672 --r 8 --dtype fp16

说明：
    本脚本做问题1要求的第一版全流程融合验证：
        1. baseline 全流程：O = X @ C + (X @ A) @ B
        2. fused 全流程：先算 Y = X @ A，再调用一个 Triton kernel 计算 O = X @ C + Y @ B
    这里刻意不引入 SGMV / 多 LoRA / segment 逻辑，只验证 Punica expand 思想在单 adapter 场景下的可行性。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda
from triton_learning.kernels.fused_matmul_expand import triton_fused_matmul_expand
from triton_learning.problem_spec import DEFAULT_GATEUP_PROBLEM, ProblemShape, resolve_dtype
from triton_learning.reference_ops import (
    compute_lora_down,
    create_problem_tensors,
    run_reference_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3：单 LoRA fused expand + GEMM 验证。")
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
        "--output",
        type=Path,
        default=Path("output/triton_learning/benchmarks/step3_fused_expand.csv"),
        help="CSV 结果输出路径。",
    )
    return parser.parse_args()


def resolve_close_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """
    为 Step 3 的数值比较设置容忍范围。

    fused 路径会把 `X@C` 与 `Y@B` 都累加在同一个 fp32 accumulator 里，
    而 Step 1 baseline 是两个独立 matmul 各自回写后再做加法。
    两者的舍入路径不同，所以这里采用更符合 fp16/bf16 实验场景的容忍度。
    """
    if dtype == torch.float16:
        return 1.0, 5e-2
    if dtype == torch.bfloat16:
        return 2.0, 8e-2
    return 1e-4, 1e-4


def run_fused_pipeline(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Step 3 全流程：先算 Y = X @ A，再执行 fused kernel。"""
    y = compute_lora_down(x, a)
    return triton_fused_matmul_expand(x, c, y, b)


def main() -> None:
    args = parse_args()
    require_cuda()

    shape = ProblemShape(m=args.m, h=args.h, n=args.n, r=args.r, dtype_name=args.dtype)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")
    tensors = create_problem_tensors(shape=shape, dtype=dtype, device=device)
    atol, rtol = resolve_close_tolerance(dtype)

    print("\nStep 3 全流程矩阵形状：")
    print(f"  X: [{shape.m}, {shape.h}]")
    print(f"  A: [{shape.h}, {shape.r}]")
    print(f"  B: [{shape.r}, {shape.n}]")
    print(f"  C: [{shape.h}, {shape.n}]")
    print(f"  dtype: {shape.dtype_name}")

    print("\n先检查 fused 全流程输出是否接近 Step 1 串行结果。")
    with torch.no_grad():
        baseline_out = run_reference_pipeline(tensors.x, tensors.a, tensors.b, tensors.c)
        fused_out = run_fused_pipeline(tensors.x, tensors.a, tensors.b, tensors.c)
        torch.cuda.synchronize()
        diff = (fused_out - baseline_out).abs()
        max_abs_error = diff.max().item()
        max_rel_error = (diff / baseline_out.abs().clamp_min(1e-5)).max().item()
        correct = torch.allclose(fused_out, baseline_out, atol=atol, rtol=rtol)
        max_flat_index = diff.reshape(-1).argmax().item()
        max_row = max_flat_index // baseline_out.shape[1]
        max_col = max_flat_index % baseline_out.shape[1]
        baseline_at_max = baseline_out[max_row, max_col].item()
        fused_at_max = fused_out[max_row, max_col].item()

    def run_baseline_once() -> torch.Tensor:
        return run_reference_pipeline(tensors.x, tensors.a, tensors.b, tensors.c)

    def run_fused_once() -> torch.Tensor:
        return run_fused_pipeline(tensors.x, tensors.a, tensors.b, tensors.c)

    baseline = measure_cuda_time(
        "baseline full: X@C + (X@A)@B",
        run_baseline_once,
        args.warmup,
        args.repeat,
    )
    fused = measure_cuda_time(
        "fused full: X@A + fused(X@C, Y@B)",
        run_fused_once,
        args.warmup,
        args.repeat,
    )

    # Step 3 的 FLOPS 由两部分 GEMM 相加组成。
    down_flops = 2.0 * shape.m * shape.h * shape.r
    base_flops = 2.0 * shape.m * shape.n * shape.h
    expand_flops = 2.0 * shape.m * shape.n * shape.r
    total_flops = down_flops + base_flops + expand_flops
    fused_tflops = total_flops / (fused.median_ms * 1e-3) / 1e12
    baseline_tflops = total_flops / (baseline.median_ms * 1e-3) / 1e12
    time_ratio = fused.median_ms / baseline.median_ms
    speedup = baseline.median_ms / fused.median_ms

    row = {
        "m": shape.m,
        "h": shape.h,
        "n": shape.n,
        "r": shape.r,
        "dtype": shape.dtype_name,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "correct": correct,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "baseline_full_ms": baseline.median_ms,
        "fused_ms": fused.median_ms,
        "baseline_tflops": baseline_tflops,
        "fused_tflops": fused_tflops,
        "fused_time_vs_baseline": time_ratio,
        "speedup_vs_baseline": speedup,
        "baseline_p20_ms": baseline.p20_ms,
        "baseline_p80_ms": baseline.p80_ms,
        "fused_p20_ms": fused.p20_ms,
        "fused_p80_ms": fused.p80_ms,
        "atol": atol,
        "rtol": rtol,
    }
    append_csv(args.output, row)

    print("\nStep 3 单 LoRA 全流程融合结果：")
    print(f"  correct: {correct}")
    print(f"  max_abs_error: {max_abs_error:.6f}")
    print(f"  max_rel_error: {max_rel_error:.6f}")
    print(
        f"  max diff position: ({max_row}, {max_col}), "
        f"baseline={baseline_at_max:.6f}, fused={fused_at_max:.6f}"
    )
    print(f"  baseline full: {baseline.median_ms:.6f} ms, {baseline_tflops:.6f} TFLOPS")
    print(f"  fused: {fused.median_ms:.6f} ms, {fused_tflops:.6f} TFLOPS")
    print(f"  fused 耗时 / baseline 耗时: {time_ratio:.4f}")
    print(f"  baseline / fused 加速比: {speedup:.4f}")
    print(f"  compare tolerance: atol={atol}, rtol={rtol}")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
