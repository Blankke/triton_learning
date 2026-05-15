"""
Benchmark 方案1：two-stream concurrent Triton kernels / 空分复用近似。

使用示例：
    python -m fusion.bench_scheme1_spatial_sharing --m 64 --h 4096 --n 28672 --r 8

说明：
    本脚本分别测量：
        1. sequential two Triton kernels: 先 Y=X@A，再 W=X@C
        2. concurrent two Triton kernels: 两个 PyTorch CUDA stream 并发 launch Triton kernel
        3. 两种方式接上后续 Z=Y@B 和 O=W+Z 后的 full pipeline 时间
"""

from __future__ import annotations

import argparse

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
from fusion.scheme1_spatial_sharing import (
    down_tile_count,
    main_tile_count,
    triton_spatial_sharing_down_main,
)
from triton_learning.benchmark_utils import (
    append_csv,
    cuda_nvtx_range,
    measure_cuda_time,
    require_cuda,
    run_profiled_callable,
)
from triton_learning.kernels.matmul import launch_triton_matmul


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    """补充 profiling 模式选择，让 baseline 与方案1本体复用同一个入口。"""
    parser.add_argument(
        "--profile-mode",
        choices=("sequential", "concurrent"),
        default="concurrent",
        help="profiling 模式下选择 Triton 串行 baseline，或方案1的双 stream 并发版本。",
    )


def main() -> None:
    """执行方案1 benchmark。"""
    args = parse_common_args(
        "方案1：two-stream concurrent Triton kernels / 空分复用近似。",
        "outputs/benchmarks/fusion_scheme1_spatial_sharing.csv",
        configure_parser=_configure_parser,
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)

    print_shape("方案1矩阵形状：", shape)
    tiles_down = down_tile_count(shape.m, shape.r)
    tiles_main = main_tile_count(shape.m, shape.n)
    print(f"  算子1 tile 数: {tiles_down}")
    print(f"  算子3 tile 数: {tiles_main}")
    y_profile = torch.empty((shape.m, shape.r), device=inputs.x.device, dtype=inputs.x.dtype)
    w_profile = torch.empty((shape.m, shape.n), device=inputs.x.device, dtype=inputs.x.dtype)

    def run_pytorch_pair_once() -> tuple[torch.Tensor, torch.Tensor]:
        return run_pytorch_pair(inputs)

    def run_sequential_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_sequential_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        # baseline 现在也统一切到 Triton 串行 GEMM。
        # 为了让 GUI 里同时看见“整体串行 pair”和里面的两个子算子，
        # 这里手动拆成两个 NVTX 子区间。
        with cuda_nvtx_range("baseline/op1_xa_triton"):
            launch_triton_matmul(inputs.x, inputs.a, y_profile)
        with cuda_nvtx_range("baseline/op3_xc_triton"):
            launch_triton_matmul(inputs.x, inputs.c, w_profile)
        return y_profile, w_profile

    def run_concurrent_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=True)

    def run_pytorch_full() -> torch.Tensor:
        return run_pytorch_full_pipeline(inputs)

    def run_sequential_full() -> torch.Tensor:
        y_once, w_once = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)
        return compute_full_output(y_once, w_once, inputs.b)

    def run_concurrent_full() -> torch.Tensor:
        y_once, w_once = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=True)
        return compute_full_output(y_once, w_once, inputs.b)

    if args.profile_only:
        if args.profile_mode == "sequential":
            print("\n方案1 profiling 模式：执行 Triton 串行 baseline，只看 Y=X@A 与 W=X@C。")
            run_profiled_callable(
                "baseline/triton_serial_pair",
                run_sequential_pair_profiled,
                warmup=args.profile_warmup,
                repeat=args.profile_repeat,
            )
        else:
            print("\n方案1 profiling 模式：只执行方案1本体，不再混入 baseline 或 PyTorch workload。")
            run_profiled_callable(
                "scheme1/concurrent_pair",
                run_concurrent_pair,
                warmup=args.profile_warmup,
                repeat=args.profile_repeat,
            )
        print("方案1 profiling workload 执行完成。")
        return

    refs = compute_references(inputs)
    with torch.no_grad():
        y, w = triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=True)
        o = compute_full_output(y, w, inputs.b)
        torch.cuda.synchronize()
        errors = check_outputs(y, w, o, refs, dtype)

    pytorch_full = measure_cuda_time("scheme1 pytorch full", run_pytorch_full, args.warmup, args.repeat)
    pytorch_pair = measure_cuda_time("scheme1 pytorch Y/W", run_pytorch_pair_once, args.warmup, args.repeat)
    sequential_pair = measure_cuda_time("scheme1 sequential Y/W", run_sequential_pair, args.warmup, args.repeat)
    concurrent_pair = measure_cuda_time("scheme1 concurrent Y/W", run_concurrent_pair, args.warmup, args.repeat)
    sequential_full = measure_cuda_time("scheme1 sequential full", run_sequential_full, args.warmup, args.repeat)
    concurrent_full = measure_cuda_time("scheme1 concurrent full", run_concurrent_full, args.warmup, args.repeat)

    row: dict[str, object] = {
        "scheme": "scheme1_spatial_sharing",
        "down_tiles": tiles_down,
        "main_tiles": tiles_main,
        "pytorch_pair_ms": pytorch_pair.median_ms,
        "scheme_pair_ms": concurrent_pair.median_ms,
        "scheme_vs_pytorch_pair_speedup": pytorch_pair.median_ms / concurrent_pair.median_ms,
        "pytorch_full_ms": pytorch_full.median_ms,
        "triton_serial_full_ms": sequential_full.median_ms,
        "scheme_full_ms": concurrent_full.median_ms,
        "scheme_vs_pytorch_speedup": pytorch_full.median_ms / concurrent_full.median_ms,
        "scheme_vs_triton_serial_speedup": sequential_full.median_ms / concurrent_full.median_ms,
        "sequential_pair_ms": sequential_pair.median_ms,
        "concurrent_pair_ms": concurrent_pair.median_ms,
        "sequential_full_ms": sequential_full.median_ms,
        "concurrent_full_ms": concurrent_full.median_ms,
        "pair_speedup": sequential_pair.median_ms / concurrent_pair.median_ms,
        "full_speedup": sequential_full.median_ms / concurrent_full.median_ms,
        "pytorch_pair_p20_ms": pytorch_pair.p20_ms,
        "pytorch_pair_p80_ms": pytorch_pair.p80_ms,
        "pytorch_full_p20_ms": pytorch_full.p20_ms,
        "pytorch_full_p80_ms": pytorch_full.p80_ms,
        "sequential_pair_p20_ms": sequential_pair.p20_ms,
        "sequential_pair_p80_ms": sequential_pair.p80_ms,
        "concurrent_pair_p20_ms": concurrent_pair.p20_ms,
        "concurrent_pair_p80_ms": concurrent_pair.p80_ms,
        "sequential_full_p20_ms": sequential_full.p20_ms,
        "sequential_full_p80_ms": sequential_full.p80_ms,
        "concurrent_full_p20_ms": concurrent_full.p20_ms,
        "concurrent_full_p80_ms": concurrent_full.p80_ms,
    }
    add_shape_fields(row, shape, args.warmup, args.repeat)
    add_error_fields(row, errors)
    append_csv(args.output, row)

    print("\n方案1 benchmark 结果：")
    print_error_report(errors)
    print(f"  PyTorch Y/W: {pytorch_pair.median_ms:.6f} ms")
    print(f"  sequential Y/W: {sequential_pair.median_ms:.6f} ms")
    print(f"  concurrent Y/W: {concurrent_pair.median_ms:.6f} ms")
    print(f"  concurrent Y/W vs PyTorch speedup: {row['scheme_vs_pytorch_pair_speedup']:.4f}")
    print(f"  concurrent Y/W vs Triton serial speedup: {row['pair_speedup']:.4f}")
    print(f"  PyTorch full: {pytorch_full.median_ms:.6f} ms")
    print(f"  sequential full: {sequential_full.median_ms:.6f} ms")
    print(f"  concurrent full: {concurrent_full.median_ms:.6f} ms")
    print(f"  concurrent full vs PyTorch speedup: {row['scheme_vs_pytorch_speedup']:.4f}")
    print(f"  concurrent full vs Triton serial speedup: {row['scheme_vs_triton_serial_speedup']:.4f}")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
