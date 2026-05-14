"""
Benchmark 方案2：single-kernel horizontal fusion via pid partition。

使用示例：
    python -m fusion.bench_scheme2_horizontal_fusion --m 64 --h 4096 --n 28672 --r 8

说明：
    baseline 使用方案1的 sequential two Triton kernels。
    方案2使用一个 Triton kernel，通过 pid 区间把 4 个 Y tile 和 896 个 W tile 放在同一个 launch 中。
"""

from __future__ import annotations

import torch

from fusion.common import (
    add_error_fields,
    add_shape_fields,
    check_outputs,
    compute_full_output,
    compute_references,
    create_inputs,
    dtype_from_args,
    parse_common_args,
    print_error_report,
    print_shape,
    run_pytorch_pair,
    run_pytorch_full_pipeline,
    shape_from_args,
)
from fusion.scheme1_spatial_sharing import triton_spatial_sharing_down_main
from fusion.scheme2_horizontal_fusion import (
    down_tile_count,
    main_tile_count,
    triton_horizontal_fused_down_main,
)
from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda


def main() -> None:
    """执行方案2 benchmark。"""
    args = parse_common_args(
        "方案2：single-kernel horizontal fusion via pid partition。",
        "outputs/benchmarks/fusion_scheme2_horizontal_fusion.csv",
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)
    refs = compute_references(inputs)

    print_shape("方案2矩阵形状：", shape)
    tiles_down = down_tile_count(shape.m, shape.r)
    tiles_main = main_tile_count(shape.m, shape.n)
    total_tiles = tiles_down + tiles_main
    print(f"  算子1 tile 数: {tiles_down}")
    print(f"  算子3 tile 数: {tiles_main}")
    print(f"  total grid: {total_tiles}")

    with torch.no_grad():
        y, w = triton_horizontal_fused_down_main(inputs.x, inputs.a, inputs.c)
        o = compute_full_output(y, w, inputs.b)
        torch.cuda.synchronize()
        errors = check_outputs(y, w, o, refs, dtype)

    def run_pytorch_full() -> torch.Tensor:
        return run_pytorch_full_pipeline(inputs)

    def run_pytorch_pair_once() -> tuple[torch.Tensor, torch.Tensor]:
        return run_pytorch_pair(inputs)

    def run_baseline_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_fused_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_horizontal_fused_down_main(inputs.x, inputs.a, inputs.c)

    def run_baseline_full() -> torch.Tensor:
        y_once, w_once = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)
        return compute_full_output(y_once, w_once, inputs.b)

    def run_fused_full() -> torch.Tensor:
        y_once, w_once = triton_horizontal_fused_down_main(inputs.x, inputs.a, inputs.c)
        return compute_full_output(y_once, w_once, inputs.b)

    pytorch_full = measure_cuda_time("scheme2 pytorch full", run_pytorch_full, args.warmup, args.repeat)
    pytorch_pair = measure_cuda_time("scheme2 pytorch Y/W", run_pytorch_pair_once, args.warmup, args.repeat)
    baseline_pair = measure_cuda_time("scheme2 baseline Y/W", run_baseline_pair, args.warmup, args.repeat)
    fused_pair = measure_cuda_time("scheme2 horizontal Y/W", run_fused_pair, args.warmup, args.repeat)
    baseline_full = measure_cuda_time("scheme2 baseline full", run_baseline_full, args.warmup, args.repeat)
    fused_full = measure_cuda_time("scheme2 horizontal full", run_fused_full, args.warmup, args.repeat)

    row: dict[str, object] = {
        "scheme": "scheme2_horizontal_fusion",
        "down_tiles": tiles_down,
        "main_tiles": tiles_main,
        "total_tiles": total_tiles,
        "pytorch_pair_ms": pytorch_pair.median_ms,
        "scheme_pair_ms": fused_pair.median_ms,
        "scheme_vs_pytorch_pair_speedup": pytorch_pair.median_ms / fused_pair.median_ms,
        "pytorch_full_ms": pytorch_full.median_ms,
        "triton_serial_full_ms": baseline_full.median_ms,
        "scheme_full_ms": fused_full.median_ms,
        "scheme_vs_pytorch_speedup": pytorch_full.median_ms / fused_full.median_ms,
        "scheme_vs_triton_serial_speedup": baseline_full.median_ms / fused_full.median_ms,
        "baseline_pair_ms": baseline_pair.median_ms,
        "fused_pair_ms": fused_pair.median_ms,
        "baseline_full_ms": baseline_full.median_ms,
        "fused_full_ms": fused_full.median_ms,
        "pair_speedup": baseline_pair.median_ms / fused_pair.median_ms,
        "full_speedup": baseline_full.median_ms / fused_full.median_ms,
        "pytorch_pair_p20_ms": pytorch_pair.p20_ms,
        "pytorch_pair_p80_ms": pytorch_pair.p80_ms,
        "pytorch_full_p20_ms": pytorch_full.p20_ms,
        "pytorch_full_p80_ms": pytorch_full.p80_ms,
        "baseline_pair_p20_ms": baseline_pair.p20_ms,
        "baseline_pair_p80_ms": baseline_pair.p80_ms,
        "fused_pair_p20_ms": fused_pair.p20_ms,
        "fused_pair_p80_ms": fused_pair.p80_ms,
        "baseline_full_p20_ms": baseline_full.p20_ms,
        "baseline_full_p80_ms": baseline_full.p80_ms,
        "fused_full_p20_ms": fused_full.p20_ms,
        "fused_full_p80_ms": fused_full.p80_ms,
    }
    add_shape_fields(row, shape, args.warmup, args.repeat)
    add_error_fields(row, errors)
    append_csv(args.output, row)

    print("\n方案2 benchmark 结果：")
    print_error_report(errors)
    print(f"  PyTorch Y/W: {pytorch_pair.median_ms:.6f} ms")
    print(f"  baseline Y/W: {baseline_pair.median_ms:.6f} ms")
    print(f"  horizontal fused Y/W: {fused_pair.median_ms:.6f} ms")
    print(f"  horizontal Y/W vs PyTorch speedup: {row['scheme_vs_pytorch_pair_speedup']:.4f}")
    print(f"  horizontal Y/W vs Triton serial speedup: {row['pair_speedup']:.4f}")
    print(f"  PyTorch full: {pytorch_full.median_ms:.6f} ms")
    print(f"  baseline full: {baseline_full.median_ms:.6f} ms")
    print(f"  horizontal fused full: {fused_full.median_ms:.6f} ms")
    print(f"  horizontal full vs PyTorch speedup: {row['scheme_vs_pytorch_speedup']:.4f}")
    print(f"  horizontal full vs Triton serial speedup: {row['scheme_vs_triton_serial_speedup']:.4f}")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
