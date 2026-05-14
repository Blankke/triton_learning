"""
Benchmark 方案3：column-concatenated GEMM / 列拼接融合。

使用示例：
    python -m fusion.bench_scheme3_column_concat --m 64 --h 4096 --n 28672 --r 8

说明：
    本脚本分别测量：
        1. physical_precat：AC 已经预处理好，只测 Triton GEMM
        2. logical_no_pad：Triton 内逻辑拼接，C 紧跟 r 后面
        3. logical_rpad_128：Triton 内逻辑拼接，C 从 128 列对齐边界开始
        4. logical_c_first_no_pad：Triton 内逻辑拼接，按 [C, A] 排列，尾 tile mask A 的剩余列
"""

from __future__ import annotations

from collections.abc import Callable

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
from fusion.scheme3_column_concat import (
    concat_tile_count,
    triton_logical_concat_c_first_down_main,
    triton_logical_concat_down_main,
    triton_physical_concat_precat,
)
from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda


def main() -> None:
    """执行方案3 benchmark。"""
    args = parse_common_args(
        "方案3：column-concatenated GEMM / 列拼接融合。",
        "outputs/benchmarks/fusion_scheme3_column_concat.csv",
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)
    refs = compute_references(inputs)
    ac = torch.cat([inputs.a, inputs.c], dim=1).contiguous()

    print_shape("方案3矩阵形状：", shape)
    total_no_pad = shape.r + shape.n
    total_rpad = 128 + shape.n if shape.r <= 128 else shape.r + shape.n
    total_c_first = shape.n + shape.r
    print(f"  no_pad total columns: {total_no_pad}, tile 数: {concat_tile_count(shape.m, total_no_pad)}")
    print(f"  r_pad=128 total columns: {total_rpad}, tile 数: {concat_tile_count(shape.m, total_rpad)}")
    print(f"  c_first total columns: {total_c_first}, tile 数: {concat_tile_count(shape.m, total_c_first)}")

    def run_pytorch_full() -> torch.Tensor:
        return run_pytorch_full_pipeline(inputs)

    def run_pytorch_pair_once() -> tuple[torch.Tensor, torch.Tensor]:
        return run_pytorch_pair(inputs)

    def run_baseline_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_baseline_full() -> torch.Tensor:
        y_once, w_once = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)
        return compute_full_output(y_once, w_once, inputs.b)

    pytorch_full = measure_cuda_time("scheme3 pytorch full", run_pytorch_full, args.warmup, args.repeat)
    pytorch_pair = measure_cuda_time("scheme3 pytorch Y/W", run_pytorch_pair_once, args.warmup, args.repeat)
    baseline_pair = measure_cuda_time("scheme3 baseline Y/W", run_baseline_pair, args.warmup, args.repeat)
    baseline_full = measure_cuda_time("scheme3 baseline full", run_baseline_full, args.warmup, args.repeat)

    variants: list[tuple[str, int, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = [
        (
            "physical_precat",
            concat_tile_count(shape.m, total_no_pad),
            lambda: triton_physical_concat_precat(inputs.x, ac, shape.r),
        ),
        (
            "logical_no_pad",
            concat_tile_count(shape.m, total_no_pad),
            lambda: triton_logical_concat_down_main(inputs.x, inputs.a, inputs.c, r_pad=0),
        ),
        (
            "logical_rpad_128",
            concat_tile_count(shape.m, total_rpad),
            lambda: triton_logical_concat_down_main(inputs.x, inputs.a, inputs.c, r_pad=128),
        ),
        (
            "logical_c_first_no_pad",
            concat_tile_count(shape.m, total_c_first),
            lambda: triton_logical_concat_c_first_down_main(inputs.x, inputs.a, inputs.c),
        ),
    ]

    print("\n方案3 baseline：")
    print(f"  PyTorch full: {pytorch_full.median_ms:.6f} ms")
    print(f"  PyTorch Y/W: {pytorch_pair.median_ms:.6f} ms")
    print(f"  baseline Y/W: {baseline_pair.median_ms:.6f} ms")
    print(f"  baseline full: {baseline_full.median_ms:.6f} ms")

    for variant_name, variant_tiles, run_variant_pair in variants:
        with torch.no_grad():
            y, w = run_variant_pair()
            o = compute_full_output(y, w, inputs.b)
            torch.cuda.synchronize()
            errors = check_outputs(y, w, o, refs, dtype)

        def run_variant_full() -> torch.Tensor:
            y_once, w_once = run_variant_pair()
            return compute_full_output(y_once, w_once, inputs.b)

        variant_pair = measure_cuda_time(
            f"scheme3 {variant_name} Y/W",
            run_variant_pair,
            args.warmup,
            args.repeat,
        )
        variant_full = measure_cuda_time(
            f"scheme3 {variant_name} full",
            run_variant_full,
            args.warmup,
            args.repeat,
        )

        row: dict[str, object] = {
            "scheme": "scheme3_column_concat",
            "variant": variant_name,
            "variant_tiles": variant_tiles,
            "pytorch_pair_ms": pytorch_pair.median_ms,
            "scheme_pair_ms": variant_pair.median_ms,
            "scheme_vs_pytorch_pair_speedup": pytorch_pair.median_ms / variant_pair.median_ms,
            "pytorch_full_ms": pytorch_full.median_ms,
            "triton_serial_full_ms": baseline_full.median_ms,
            "scheme_full_ms": variant_full.median_ms,
            "scheme_vs_pytorch_speedup": pytorch_full.median_ms / variant_full.median_ms,
            "scheme_vs_triton_serial_speedup": baseline_full.median_ms / variant_full.median_ms,
            "baseline_pair_ms": baseline_pair.median_ms,
            "variant_pair_ms": variant_pair.median_ms,
            "baseline_full_ms": baseline_full.median_ms,
            "variant_full_ms": variant_full.median_ms,
            "pair_speedup": baseline_pair.median_ms / variant_pair.median_ms,
            "full_speedup": baseline_full.median_ms / variant_full.median_ms,
            "pytorch_pair_p20_ms": pytorch_pair.p20_ms,
            "pytorch_pair_p80_ms": pytorch_pair.p80_ms,
            "pytorch_full_p20_ms": pytorch_full.p20_ms,
            "pytorch_full_p80_ms": pytorch_full.p80_ms,
            "baseline_pair_p20_ms": baseline_pair.p20_ms,
            "baseline_pair_p80_ms": baseline_pair.p80_ms,
            "variant_pair_p20_ms": variant_pair.p20_ms,
            "variant_pair_p80_ms": variant_pair.p80_ms,
            "baseline_full_p20_ms": baseline_full.p20_ms,
            "baseline_full_p80_ms": baseline_full.p80_ms,
            "variant_full_p20_ms": variant_full.p20_ms,
            "variant_full_p80_ms": variant_full.p80_ms,
        }
        add_shape_fields(row, shape, args.warmup, args.repeat)
        add_error_fields(row, errors)
        append_csv(args.output, row)

        print(f"\n方案3 {variant_name} 结果：")
        print_error_report(errors)
        print(f"  PyTorch Y/W: {pytorch_pair.median_ms:.6f} ms")
        print(f"  variant Y/W: {variant_pair.median_ms:.6f} ms")
        print(f"  variant Y/W vs PyTorch speedup: {row['scheme_vs_pytorch_pair_speedup']:.4f}")
        print(f"  variant Y/W vs Triton serial speedup: {row['pair_speedup']:.4f}")
        print(f"  variant full: {variant_full.median_ms:.6f} ms")
        print(f"  variant full vs PyTorch speedup: {row['scheme_vs_pytorch_speedup']:.4f}")
        print(f"  variant full vs Triton serial speedup: {row['scheme_vs_triton_serial_speedup']:.4f}")

    print(f"\nCSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
