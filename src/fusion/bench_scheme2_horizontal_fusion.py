"""
Benchmark 方案2：single-kernel horizontal fusion。

使用示例：
    python -m fusion.bench_scheme2_horizontal_fusion --m 64 --h 4096 --n 28672 --r 8
    python -m fusion.bench_scheme2_horizontal_fusion --variant grouped_persistent --num-down-workers 4 --chunk-size 4

说明：
    baseline 使用方案1的 sequential two Triton kernels。
    方案2当前保留两个单 kernel 口径：
        1. static_pid：按 pid 区间把 down/main tiles 放进同一个 launch
        2. grouped_persistent：按 grouped ordering 做 persistent worker 调度

Nsight Systems 对照关系：
    - benchmark 正常计时时，baseline 路径仍然会看到 `_matmul_kernel`，
      因为它本质上还是 Step 2 的两个独立 Triton GEMM。
    - profiling 模式下：
      - `scheme2/horizontal_pair` 对应静态 pid 版
      - `scheme2/grouped_persistent_pair` 对应 grouped persistent 版
    - `vectorized_elementwise_kernel<...FillFunctor<int>...>`
      多数是 Triton autotune / 运行时辅助产生的 fill/reset 内核，不是题目里的核心 GEMM 本体。
"""

from __future__ import annotations

import argparse
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
from fusion.scheme2_horizontal_fusion import (
    DEFAULT_PERSISTENT_CHUNK_SIZE,
    GROUPED_PERSISTENT_VARIANT,
    STATIC_PID_VARIANT,
    device_num_sms,
    down_tile_count,
    main_tile_count,
    resolve_grouped_persistent_scheduler,
    triton_horizontal_fused_down_main,
)
from triton_learning.benchmark_utils import append_csv, measure_cuda_time, require_cuda, run_profiled_callable


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    """补充方案2变体与 persistent 调度参数。"""
    parser.add_argument(
        "--variant",
        choices=("all", STATIC_PID_VARIANT, GROUPED_PERSISTENT_VARIANT),
        default="all",
        help="选择方案2中要执行的变体；profiling 时建议一次只抓一个变体。",
    )
    parser.add_argument(
        "--num-down-workers",
        type=int,
        default=0,
        help="grouped_persistent 版的 down worker 数；0 表示按当前形状自动推导。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_PERSISTENT_CHUNK_SIZE,
        help="grouped_persistent 版每个 worker 每轮处理的连续 main grouped tiles 数。",
    )


def main() -> None:
    """执行方案2 benchmark。"""
    args = parse_common_args(
        "方案2：single-kernel horizontal fusion。",
        "output/fusion/benchmarks/fusion_scheme2_horizontal_fusion.csv",
        configure_parser=_configure_parser,
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)
    num_sms = device_num_sms(inputs.x.device)

    print_shape("方案2矩阵形状：", shape)
    tiles_down = down_tile_count(shape.m, shape.r)
    tiles_main = main_tile_count(shape.m, shape.n)
    total_tiles = tiles_down + tiles_main
    print(f"  算子1 tile 数: {tiles_down}")
    print(f"  算子3 tile 数: {tiles_main}")
    print(f"  total grid（参考口径）: {total_tiles}")
    print(f"  device SM 数: {num_sms}")

    persistent_schedule = resolve_grouped_persistent_scheduler(
        shape.m,
        shape.r,
        shape.n,
        inputs.x.device,
        num_down_workers=args.num_down_workers,
        chunk_size=args.chunk_size,
    )

    def run_pytorch_full() -> torch.Tensor:
        return run_pytorch_full_pipeline(inputs)

    def run_pytorch_pair_once() -> tuple[torch.Tensor, torch.Tensor]:
        return run_pytorch_pair(inputs)

    def run_baseline_pair() -> tuple[torch.Tensor, torch.Tensor]:
        # Nsight Systems 中这里对应的主 kernel 名字是 `_matmul_kernel`。
        # 因为它本质上还是 Step 2 的两个独立 Triton GEMM：先算 Y=X@A，再算 W=X@C。
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_baseline_full() -> torch.Tensor:
        y_once, w_once = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)
        return compute_full_output(y_once, w_once, inputs.b)

    variants: list[tuple[str, str, Callable[[], tuple[torch.Tensor, torch.Tensor]], dict[str, int]]] = [
        (
            STATIC_PID_VARIANT,
            "scheme2/horizontal_pair",
            lambda: triton_horizontal_fused_down_main(
                inputs.x,
                inputs.a,
                inputs.c,
                variant=STATIC_PID_VARIANT,
            ),
            {
                "num_sms": num_sms,
                "launched_workers": total_tiles,
                "requested_num_down_workers": 0,
                "effective_num_down_workers": 0,
                "chunk_size": 0,
                "num_main_chunks": 0,
            },
        ),
        (
            GROUPED_PERSISTENT_VARIANT,
            "scheme2/grouped_persistent_pair",
            lambda: triton_horizontal_fused_down_main(
                inputs.x,
                inputs.a,
                inputs.c,
                variant=GROUPED_PERSISTENT_VARIANT,
                num_down_workers=args.num_down_workers,
                chunk_size=args.chunk_size,
            ),
            {
                "num_sms": persistent_schedule.num_sms,
                "launched_workers": persistent_schedule.launched_workers,
                "requested_num_down_workers": persistent_schedule.requested_num_down_workers,
                "effective_num_down_workers": persistent_schedule.effective_num_down_workers,
                "chunk_size": persistent_schedule.chunk_size,
                "num_main_chunks": persistent_schedule.num_main_chunks,
            },
        ),
    ]

    if args.variant != "all":
        variants = [variant for variant in variants if variant[0] == args.variant]
        if not variants:
            raise ValueError(f"未找到方案2变体：{args.variant}")

    if args.profile_only:
        print("\n方案2 profiling 模式：只执行方案2本体，不再混入 baseline 或 PyTorch workload。")
        if len(variants) != 1:
            raise ValueError("方案2 profiling 模式要求只选择一个变体，请通过 --variant 指定。")
        _, profile_title, run_variant_pair, _ = variants[0]
        run_profiled_callable(
            profile_title,
            run_variant_pair,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        print("方案2 profiling workload 执行完成。")
        return

    refs = compute_references(inputs)
    pytorch_full = measure_cuda_time("scheme2 pytorch full", run_pytorch_full, args.warmup, args.repeat)
    pytorch_pair = measure_cuda_time("scheme2 pytorch Y/W", run_pytorch_pair_once, args.warmup, args.repeat)
    baseline_pair = measure_cuda_time("scheme2 baseline Y/W", run_baseline_pair, args.warmup, args.repeat)
    baseline_full = measure_cuda_time("scheme2 baseline full", run_baseline_full, args.warmup, args.repeat)

    print("\n方案2 baseline：")
    print(f"  PyTorch full: {pytorch_full.median_ms:.6f} ms")
    print(f"  PyTorch Y/W: {pytorch_pair.median_ms:.6f} ms")
    print(f"  baseline Y/W: {baseline_pair.median_ms:.6f} ms")
    print(f"  baseline full: {baseline_full.median_ms:.6f} ms")
    print(
        "  grouped persistent 默认调度："
        f" workers≈{persistent_schedule.launched_workers},"
        f" down_workers={persistent_schedule.effective_num_down_workers},"
        f" chunk_size={persistent_schedule.chunk_size},"
        f" main_chunks≈{persistent_schedule.num_main_chunks}"
    )

    for variant_name, _, run_variant_pair, schedule_info in variants:
        with torch.no_grad():
            y, w = run_variant_pair()
            o = compute_full_output(y, w, inputs.b)
            torch.cuda.synchronize()
            errors = check_outputs(y, w, o, refs, dtype)

        def run_variant_full() -> torch.Tensor:
            y_once, w_once = run_variant_pair()
            return compute_full_output(y_once, w_once, inputs.b)

        variant_pair = measure_cuda_time(
            f"scheme2 {variant_name} Y/W",
            run_variant_pair,
            args.warmup,
            args.repeat,
        )
        variant_full = measure_cuda_time(
            f"scheme2 {variant_name} full",
            run_variant_full,
            args.warmup,
            args.repeat,
        )

        row: dict[str, object] = {
            "scheme": "scheme2_horizontal_fusion",
            "variant": variant_name,
            "down_tiles": tiles_down,
            "main_tiles": tiles_main,
            "total_tiles": total_tiles,
            "num_sms": schedule_info["num_sms"],
            "launched_workers": schedule_info["launched_workers"],
            "requested_num_down_workers": schedule_info["requested_num_down_workers"],
            "effective_num_down_workers": schedule_info["effective_num_down_workers"],
            "chunk_size": schedule_info["chunk_size"],
            "num_main_chunks": schedule_info["num_main_chunks"],
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

        print(f"\n方案2 {variant_name} 结果：")
        print_error_report(errors)
        print(f"  variant Y/W: {variant_pair.median_ms:.6f} ms")
        print(f"  variant Y/W vs PyTorch speedup: {row['scheme_vs_pytorch_pair_speedup']:.4f}")
        print(f"  variant Y/W vs Triton serial speedup: {row['pair_speedup']:.4f}")
        print(f"  variant full: {variant_full.median_ms:.6f} ms")
        print(f"  variant full vs PyTorch speedup: {row['scheme_vs_pytorch_speedup']:.4f}")
        print(f"  variant full vs Triton serial speedup: {row['scheme_vs_triton_serial_speedup']:.4f}")
        if variant_name == GROUPED_PERSISTENT_VARIANT:
            print(
                "  grouped persistent 调度："
                f" workers≈{schedule_info['launched_workers']},"
                f" down_workers={schedule_info['effective_num_down_workers']},"
                f" chunk_size={schedule_info['chunk_size']},"
                f" main_chunks≈{schedule_info['num_main_chunks']}"
            )

    print(f"\nCSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
