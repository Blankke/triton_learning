"""
sharing 五路对比 benchmark。

使用示例：
    python -m sharing.bench_five_way_comparison
    python -m sharing.bench_five_way_comparison --num-workers 108
    python -m sharing.bench_five_way_comparison --profile-only --profile-target single_fused_interleaved

说明：
    本入口把 sharing 当前真正不同的五种实现放到同一张表里：
        1. baseline：两个独立 Triton GEMM 串行执行
        2. stream_overlap：两个独立 Triton GEMM 在双 stream 中并发 launch
        3. single_fused_half_split：前半 pid range 做 op1，后半 pid range 做 op3
        4. single_fused_interleaved：四段 pid range 交错做 op1 / op3 / op1 / op3
        5. physical_concat：先物理拼接 [A, C]，再做一次大 GEMM
    这样 baseline / stream_overlap / physical_concat 只测一次，
    两种 pid range 单 kernel 方案直接并排对比，不再重复跑两份实验。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from fusion.common import (
    ErrorReport,
    add_shape_fields,
    check_outputs,
    compute_full_output,
    compute_references,
    create_inputs,
    dtype_from_args,
    parse_common_args,
    print_shape,
    shape_from_args,
)
from fusion.scheme1_spatial_sharing import triton_spatial_sharing_down_main
from fusion.scheme3_column_concat import triton_physical_concat_precat
from sharing.range_fusion import (
    DEFAULT_CONSTRUCTED_R,
    RangeSchedule,
    build_half_split_schedule,
    build_interleaved_schedule,
    format_schedule,
    schedule_logic_lines,
    schedule_segments,
    triton_range_fused_down_main,
)
from triton_learning.benchmark_utils import (
    append_csv,
    cuda_nvtx_range,
    measure_cuda_time,
    require_cuda,
    run_profiled_callable,
)
from triton_learning.kernels.matmul import launch_triton_matmul


PROFILE_BASELINE = "baseline"
PROFILE_STREAM_OVERLAP = "stream_overlap"
PROFILE_SINGLE_FUSED_HALF_SPLIT = "single_fused_half_split"
PROFILE_SINGLE_FUSED_INTERLEAVED = "single_fused_interleaved"
PROFILE_PHYSICAL_CONCAT = "physical_concat"
PROFILE_TARGETS = (
    PROFILE_BASELINE,
    PROFILE_STREAM_OVERLAP,
    PROFILE_SINGLE_FUSED_HALF_SPLIT,
    PROFILE_SINGLE_FUSED_INTERLEAVED,
    PROFILE_PHYSICAL_CONCAT,
)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    """补充 sharing 五路对比需要的参数。"""
    parser.set_defaults(r=DEFAULT_CONSTRUCTED_R)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="sharing 单 kernel 的 launch program 数；0 表示自动取 total_tiles，若显式传入也必须等于 total_tiles。",
    )
    parser.add_argument(
        "--profile-target",
        choices=PROFILE_TARGETS,
        default=PROFILE_SINGLE_FUSED_HALF_SPLIT,
        help="profiling 模式下要抓取的目标实现。",
    )


def _add_prefixed_error_fields(row: dict[str, object], prefix: str, errors: ErrorReport) -> None:
    """把某个方法的正确性字段追加到 CSV 行。"""
    row[f"{prefix}_correct"] = errors.all_correct
    row[f"{prefix}_y_correct"] = errors.y_correct
    row[f"{prefix}_w_correct"] = errors.w_correct
    row[f"{prefix}_o_correct"] = errors.o_correct
    row[f"{prefix}_y_max_abs"] = errors.y_max_abs
    row[f"{prefix}_w_max_abs"] = errors.w_max_abs
    row[f"{prefix}_o_max_abs"] = errors.o_max_abs
    row[f"{prefix}_y_max_rel"] = errors.y_max_rel
    row[f"{prefix}_w_max_rel"] = errors.w_max_rel
    row[f"{prefix}_o_max_rel"] = errors.o_max_rel


def _add_schedule_fields(row: dict[str, object], prefix: str, schedule: RangeSchedule) -> None:
    """把某一种 fused 调度的关键信息展开进 CSV 行。"""
    row[f"{prefix}_schedule_name"] = schedule.name
    row[f"{prefix}_num_sms"] = schedule.num_sms
    row[f"{prefix}_launched_workers"] = schedule.launched_workers
    row[f"{prefix}_op1_tiles"] = schedule.op1_tiles
    row[f"{prefix}_op3_tiles"] = schedule.op3_tiles
    row[f"{prefix}_op1_workers"] = schedule.op1_workers
    row[f"{prefix}_op3_workers"] = schedule.op3_workers
    row[f"{prefix}_ranges"] = format_schedule(schedule)
    row[f"{prefix}_range_ops"] = "|".join(schedule.range_ops)
    row[f"{prefix}_range_sizes"] = "|".join(str(size) for size in schedule.range_sizes)
    row[f"{prefix}_range_bases"] = "|".join(str(base) for base in schedule.op_worker_bases)
    row[f"{prefix}_range_pid_spans"] = "|".join(
        f"[{segment.pid_start},{segment.pid_end})->{segment.op_name}"
        for segment in schedule_segments(schedule)
    )
    row[f"{prefix}_range_tile_spans"] = "|".join(
        f"{segment.op_name}[{segment.op_tile_start},{segment.op_tile_end})"
        for segment in schedule_segments(schedule)
    )


def _print_schedule(title: str, schedule: RangeSchedule) -> None:
    """打印某一种 fused 调度的 pid range 详情。"""
    print(f"\n{title}：")
    print(f"  schedule: {schedule.name}")
    print(f"  device SM 数: {schedule.num_sms}")
    print(f"  op1 tile 数（16x128 参考口径）: {schedule.op1_tiles}")
    print(f"  op3 tile 数（16x128 参考口径）: {schedule.op3_tiles}")
    print(f"  launched workers（= total_tiles）: {schedule.launched_workers}")
    print(f"  ranges: {format_schedule(schedule)}")
    print(f"  op1 workers: {schedule.op1_workers}, op3 workers: {schedule.op3_workers}")
    print("  pid range 明细：")
    for segment in schedule_segments(schedule):
        print(
            "   "
            f"range{segment.range_index}: pid[{segment.pid_start}, {segment.pid_end}) -> {segment.op_name}, "
            f"{segment.op_name} tile[{segment.op_tile_start}, {segment.op_tile_end})"
        )
    print("  伪代码语义：")
    for line in schedule_logic_lines(schedule):
        print(f"    {line}")


def main() -> None:
    """执行 sharing 五路对比 benchmark。"""
    args = parse_common_args(
        "sharing 五路对比：baseline / stream overlap / 两种 pid-range fused / physical concat。",
        "output/sharing/benchmarks/sharing_five_way_comparison.csv",
        configure_parser=_configure_parser,
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)
    refs = compute_references(inputs)
    ac = torch.cat([inputs.a, inputs.c], dim=1).contiguous()
    stream_device_index = torch.cuda.current_device() if inputs.x.device.index is None else inputs.x.device.index
    half_split_schedule = build_half_split_schedule(shape.m, shape.r, shape.n, inputs.x.device, num_workers=args.num_workers)
    interleaved_schedule = build_interleaved_schedule(
        shape.m,
        shape.r,
        shape.n,
        inputs.x.device,
        num_workers=args.num_workers,
    )
    y_profile = torch.empty((shape.m, shape.r), device=inputs.x.device, dtype=inputs.x.dtype)
    w_profile = torch.empty((shape.m, shape.n), device=inputs.x.device, dtype=inputs.x.dtype)
    stream_overlap_down_profile_stream = torch.cuda.Stream(device=stream_device_index)
    stream_overlap_main_profile_stream = torch.cuda.Stream(device=stream_device_index)

    print_shape("sharing 五路对比矩阵形状：", shape)
    print(f"  sharing 默认构造 workload：r 建议使用 {DEFAULT_CONSTRUCTED_R}")
    print("  说明：baseline / stream_overlap / physical_concat 只测一次，")
    print("        两种 single_fused 调度直接并排对比，不再拆成两份重复实验。")
    _print_schedule("single_fused_half_split", half_split_schedule)
    _print_schedule("single_fused_interleaved", interleaved_schedule)

    def run_baseline_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_baseline_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        # baseline 仍然是两个独立 Triton GEMM；这里额外打子区间，便于在 nsys 里看清 op1/op3。
        with cuda_nvtx_range("sharing/five_way/baseline/op1_xa_triton"):
            launch_triton_matmul(inputs.x, inputs.a, y_profile)
        with cuda_nvtx_range("sharing/five_way/baseline/op3_xc_triton"):
            launch_triton_matmul(inputs.x, inputs.c, w_profile)
        return y_profile, w_profile

    def run_stream_overlap_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=True)

    def run_stream_overlap_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        # stream_overlap 本质上还是两个独立 Triton GEMM，只是放进两个 stream 并发 launch。
        current = torch.cuda.current_stream()
        stream_overlap_down_profile_stream.wait_stream(current)
        stream_overlap_main_profile_stream.wait_stream(current)
        with torch.cuda.stream(stream_overlap_down_profile_stream):
            with cuda_nvtx_range("sharing/five_way/stream_overlap/op1_xa_triton"):
                launch_triton_matmul(inputs.x, inputs.a, y_profile)
        with torch.cuda.stream(stream_overlap_main_profile_stream):
            with cuda_nvtx_range("sharing/five_way/stream_overlap/op3_xc_triton"):
                launch_triton_matmul(inputs.x, inputs.c, w_profile)
        current.wait_stream(stream_overlap_down_profile_stream)
        current.wait_stream(stream_overlap_main_profile_stream)
        return y_profile, w_profile

    def run_single_fused_half_split_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_range_fused_down_main(inputs.x, inputs.a, inputs.c, half_split_schedule)

    def run_single_fused_half_split_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        with cuda_nvtx_range("sharing/five_way/single_fused_half_split/range_fused_kernel"):
            return triton_range_fused_down_main(inputs.x, inputs.a, inputs.c, half_split_schedule)

    def run_single_fused_interleaved_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_range_fused_down_main(inputs.x, inputs.a, inputs.c, interleaved_schedule)

    def run_single_fused_interleaved_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        with cuda_nvtx_range("sharing/five_way/single_fused_interleaved/range_fused_kernel"):
            return triton_range_fused_down_main(inputs.x, inputs.a, inputs.c, interleaved_schedule)

    def run_physical_concat_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_physical_concat_precat(inputs.x, ac, shape.r)

    def run_physical_concat_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        with cuda_nvtx_range("sharing/five_way/physical_concat/matmul_kernel"):
            return triton_physical_concat_precat(inputs.x, ac, shape.r)

    profile_map: dict[str, tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = {
        PROFILE_BASELINE: ("sharing/five_way/baseline_pair", run_baseline_pair_profiled),
        PROFILE_STREAM_OVERLAP: ("sharing/five_way/stream_overlap_pair", run_stream_overlap_pair_profiled),
        PROFILE_SINGLE_FUSED_HALF_SPLIT: (
            "sharing/five_way/single_fused_half_split_pair",
            run_single_fused_half_split_pair_profiled,
        ),
        PROFILE_SINGLE_FUSED_INTERLEAVED: (
            "sharing/five_way/single_fused_interleaved_pair",
            run_single_fused_interleaved_pair_profiled,
        ),
        PROFILE_PHYSICAL_CONCAT: ("sharing/five_way/physical_concat_pair", run_physical_concat_pair_profiled),
    }

    if args.profile_only:
        title, profiled_fn = profile_map[args.profile_target]
        print(f"\nsharing profiling 模式：只执行 {args.profile_target}。")
        run_profiled_callable(
            title,
            profiled_fn,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        print("sharing profiling workload 执行完成。")
        return

    methods: list[tuple[str, str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = [
        ("baseline", "baseline serial", run_baseline_pair),
        ("stream_overlap", "stream overlap", run_stream_overlap_pair),
        ("single_fused_half_split", "single fused half split", run_single_fused_half_split_pair),
        ("single_fused_interleaved", "single fused interleaved", run_single_fused_interleaved_pair),
        ("physical_concat", "physical concat", run_physical_concat_pair),
    ]

    row: dict[str, object] = {
        "benchmark": "sharing_five_way_comparison",
    }
    add_shape_fields(row, shape, args.warmup, args.repeat)
    _add_schedule_fields(row, "half_split", half_split_schedule)
    _add_schedule_fields(row, "interleaved", interleaved_schedule)

    timing_by_name: dict[str, tuple[object, object]] = {}
    errors_by_name: dict[str, ErrorReport] = {}

    for method_name, timing_title, run_pair in methods:
        with torch.no_grad():
            y, w = run_pair()
            o = compute_full_output(y, w, inputs.b)
            torch.cuda.synchronize()
            errors = check_outputs(y, w, o, refs, dtype)
        errors_by_name[method_name] = errors

        def run_full() -> torch.Tensor:
            y_once, w_once = run_pair()
            return compute_full_output(y_once, w_once, inputs.b)

        pair_timing = measure_cuda_time(
            f"sharing {timing_title} pair",
            run_pair,
            args.warmup,
            args.repeat,
        )
        full_timing = measure_cuda_time(
            f"sharing {timing_title} full",
            run_full,
            args.warmup,
            args.repeat,
        )
        timing_by_name[method_name] = (pair_timing, full_timing)

        row[f"{method_name}_pair_ms"] = pair_timing.median_ms
        row[f"{method_name}_pair_p20_ms"] = pair_timing.p20_ms
        row[f"{method_name}_pair_p80_ms"] = pair_timing.p80_ms
        row[f"{method_name}_full_ms"] = full_timing.median_ms
        row[f"{method_name}_full_p20_ms"] = full_timing.p20_ms
        row[f"{method_name}_full_p80_ms"] = full_timing.p80_ms
        _add_prefixed_error_fields(row, method_name, errors)

    baseline_pair = timing_by_name["baseline"][0].median_ms
    baseline_full = timing_by_name["baseline"][1].median_ms
    for method_name in (
        "stream_overlap",
        "single_fused_half_split",
        "single_fused_interleaved",
        "physical_concat",
    ):
        row[f"{method_name}_pair_speedup_vs_baseline"] = baseline_pair / timing_by_name[method_name][0].median_ms
        row[f"{method_name}_full_speedup_vs_baseline"] = baseline_full / timing_by_name[method_name][1].median_ms

    append_csv(args.output, row)

    print("\nsharing 五路对比结果：")
    for method_name, timing_title, _ in methods:
        pair_timing, full_timing = timing_by_name[method_name]
        errors = errors_by_name[method_name]
        print(
            f"  {timing_title}: pair={pair_timing.median_ms:.6f} ms, "
            f"full={full_timing.median_ms:.6f} ms, correct={errors.all_correct}"
        )
    print(f"  stream overlap pair vs baseline: {row['stream_overlap_pair_speedup_vs_baseline']:.4f}")
    print(f"  single fused half split pair vs baseline: {row['single_fused_half_split_pair_speedup_vs_baseline']:.4f}")
    print(
        "  single fused interleaved pair vs baseline: "
        f"{row['single_fused_interleaved_pair_speedup_vs_baseline']:.4f}"
    )
    print(f"  physical concat pair vs baseline: {row['physical_concat_pair_speedup_vs_baseline']:.4f}")
    print(f"  CSV 已写入: {args.output}")


if __name__ == "__main__":
    main()
