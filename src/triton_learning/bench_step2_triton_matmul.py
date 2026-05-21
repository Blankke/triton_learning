"""
Step 2：使用 Triton 教程中的 GEMM 思路实现主干矩阵乘，并与 cuBLAS 对比。

使用示例：
    python -m triton_learning.bench_step2_triton_matmul --m 64 --h 4096 --n 28672 --dtype fp16

说明：
    本脚本只关注主干矩阵乘：
        W = X @ C
    这里不实现低秩分支，也不做融合；融合逻辑放在 Step 3。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda, tflops
from triton_learning.kernels.matmul import triton_matmul
from triton_learning.problem_spec import DEFAULT_GATEUP_PROBLEM, ProblemShape, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2：Triton GEMM 与 cuBLAS 对比。")
    parser.add_argument("--m", type=int, default=DEFAULT_GATEUP_PROBLEM.m, help="输入 token/batch 维度 M。")
    parser.add_argument("--h", type=int, default=DEFAULT_GATEUP_PROBLEM.h, help="隐藏层维度 H，也就是 GEMM 的 K。")
    parser.add_argument("--n", type=int, default=DEFAULT_GATEUP_PROBLEM.n, help="输出维度 N。")
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
        default=Path("output/triton_learning/benchmarks/step2_triton_vs_cublas.csv"),
        help="CSV 结果输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()

    shape = ProblemShape(m=args.m, h=args.h, n=args.n, r=DEFAULT_GATEUP_PROBLEM.r, dtype_name=args.dtype)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")

    torch.manual_seed(0)
    x = torch.randn((shape.m, shape.h), device=device, dtype=dtype)
    c = torch.randn((shape.h, shape.n), device=device, dtype=dtype)

    print("\nStep 2 主干 GEMM 矩阵形状：")
    print(f"  X: [{shape.m}, {shape.h}]")
    print(f"  C: [{shape.h}, {shape.n}]")
    print(f"  W: [{shape.m}, {shape.n}]")
    print(f"  dtype: {shape.dtype_name}")

    print("\n先检查 Triton 输出是否接近 PyTorch/cuBLAS 输出。首次运行会触发 Triton 编译和 autotune。")
    with torch.no_grad():
        torch_out = x @ c
        triton_out = triton_matmul(x, c)
        torch.cuda.synchronize()
        max_abs_error = (triton_out - torch_out).abs().max().item()
        correct = torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    def run_cublas_once() -> torch.Tensor:
        return x @ c

    def run_triton_once() -> torch.Tensor:
        return triton_matmul(x, c)

    cublas = measure_cuda_time("cuBLAS: X @ C", run_cublas_once, args.warmup, args.repeat)
    triton = measure_cuda_time("Triton: X @ C", run_triton_once, args.warmup, args.repeat)

    cublas_tflops = tflops(shape.m, shape.n, shape.h, cublas.median_ms)
    triton_tflops = tflops(shape.m, shape.n, shape.h, triton.median_ms)
    time_ratio = triton.median_ms / cublas.median_ms
    perf_ratio = triton_tflops / cublas_tflops

    row = {
        "m": shape.m,
        "h": shape.h,
        "n": shape.n,
        "dtype": shape.dtype_name,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "correct": correct,
        "max_abs_error": max_abs_error,
        "cublas_ms": cublas.median_ms,
        "triton_ms": triton.median_ms,
        "cublas_tflops": cublas_tflops,
        "triton_tflops": triton_tflops,
        "triton_time_vs_cublas": time_ratio,
        "triton_perf_vs_cublas": perf_ratio,
        "cublas_p20_ms": cublas.p20_ms,
        "cublas_p80_ms": cublas.p80_ms,
        "triton_p20_ms": triton.p20_ms,
        "triton_p80_ms": triton.p80_ms,
    }
    append_csv(args.output, row)

    print("\nStep 2 Triton GEMM 结果：")
    print(f"  correct: {correct}")
    print(f"  max_abs_error: {max_abs_error:.6f}")
    print(f"  cuBLAS: {cublas.median_ms:.6f} ms, {cublas_tflops:.6f} TFLOPS")
    print(f"  Triton: {triton.median_ms:.6f} ms, {triton_tflops:.6f} TFLOPS")
    print(f"  Triton 耗时 / cuBLAS 耗时: {time_ratio:.4f}")
    print(f"  Triton TFLOPS / cuBLAS TFLOPS: {perf_ratio:.4f}")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
