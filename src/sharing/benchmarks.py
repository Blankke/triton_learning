"""
sharing 实验共用 benchmark 驱动。

使用示例：
    from sharing.benchmarks import SharingExperimentSpec, run_sharing_benchmark

说明：
    本模块把 sharing 目录下两个实验的公共逻辑集中到一起：
        - 构造 workload（默认 r=2048）
        - baseline / stream overlap / single fused / physical concat 统一计时
        - profiling 模式下按指定 target 输出 nsys/ncu 友好的 NVTX range
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

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
PROFILE_SINGLE_FUSED = "single_fused"
PROFILE_PHYSICAL_CONCAT = "physical_concat"
PROFILE_TARGETS = (
    PROFILE_BASELINE,
    PROFILE_STREAM_OVERLAP,
    PROFILE_SINGLE_FUSED,
    PROFILE_PHYSICAL_CONCAT,
)


@dataclass(frozen=True)
class SharingExperimentSpec:
    """描述一个 sharing benchmark 入口。"""

    experiment_name: str
    description: str
    default_output: str
    schedule_builder: Callable[[int, int, int, torch.device, int], RangeSchedule]
    profile_title_prefix: str


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    """给 sharing benchmark 补充默认构造 workload 与 profiling 选择项。"""
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
        default=PROFILE_SINGLE_FUSED,
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


def run_sharing_benchmark(spec: SharingExperimentSpec) -> None:
    """执行某一个 sharing 实验。"""
    args = parse_common_args(
        spec.description,
        spec.default_output,
        configure_parser=_configure_parser,
    )
    require_cuda()

    shape = shape_from_args(args)
    dtype = dtype_from_args(args)
    inputs = create_inputs(shape, dtype, torch.device("cuda"), args.seed)
    refs = compute_references(inputs)
    ac = torch.cat([inputs.a, inputs.c], dim=1).contiguous()
    schedule = spec.schedule_builder(shape.m, shape.r, shape.n, inputs.x.device, args.num_workers)
    y_profile = torch.empty((shape.m, shape.r), device=inputs.x.device, dtype=inputs.x.dtype)
    w_profile = torch.empty((shape.m, shape.n), device=inputs.x.device, dtype=inputs.x.dtype)

    print_shape(f"{spec.experiment_name} 矩阵形状：", shape)
    print(f"  sharing 默认构造 workload：r 建议使用 {DEFAULT_CONSTRUCTED_R}")
    print(f"  op1 tile 数（16x128 参考口径）: {schedule.op1_tiles}")
    print(f"  op3 tile 数（16x128 参考口径）: {schedule.op3_tiles}")
    print(f"  device SM 数: {schedule.num_sms}")
    print(f"  launched workers（= total_tiles）: {schedule.launched_workers}")
    print(f"  total_tiles = op1_tiles + op3_tiles = {schedule.op1_tiles} + {schedule.op3_tiles} = {schedule.launched_workers}")
    print(f"  schedule: {schedule.name}")
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
    print("  说明：当前 sharing 配置下 launched_workers == total_tiles，因此每个 pid 恰好对应 1 个 tile。")

    def run_baseline_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=False)

    def run_baseline_pair_profiled() -> tuple[torch.Tensor, torch.Tensor]:
        # baseline 仍然是两个独立 Triton GEMM；这里额外打子区间，便于在 nsys 里看清 op1/op3。
        with cuda_nvtx_range(f"{spec.profile_title_prefix}/baseline/op1_xa_triton"):
            launch_triton_matmul(inputs.x, inputs.a, y_profile)
        with cuda_nvtx_range(f"{spec.profile_title_prefix}/baseline/op3_xc_triton"):
            launch_triton_matmul(inputs.x, inputs.c, w_profile)
        return y_profile, w_profile

    def run_stream_overlap_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_spatial_sharing_down_main(inputs.x, inputs.a, inputs.c, concurrent=True)

    def run_single_fused_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_range_fused_down_main(inputs.x, inputs.a, inputs.c, schedule)

    def run_physical_concat_pair() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_physical_concat_precat(inputs.x, ac, shape.r)

    profile_map: dict[str, tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = {
        PROFILE_BASELINE: (f"{spec.profile_title_prefix}/baseline_pair", run_baseline_pair_profiled),
        PROFILE_STREAM_OVERLAP: (f"{spec.profile_title_prefix}/stream_overlap_pair", run_stream_overlap_pair),
        PROFILE_SINGLE_FUSED: (f"{spec.profile_title_prefix}/single_fused_pair", run_single_fused_pair),
        PROFILE_PHYSICAL_CONCAT: (f"{spec.profile_title_prefix}/physical_concat_pair", run_physical_concat_pair),
    }

    if args.profile_only:
        title, profiled_fn = profile_map[args.profile_target]
        print(f"\n{spec.experiment_name} profiling 模式：只执行 {args.profile_target}。")
        run_profiled_callable(
            title,
            profiled_fn,
            warmup=args.profile_warmup,
            repeat=args.profile_repeat,
        )
        print(f"{spec.experiment_name} profiling workload 执行完成。")
        return

    methods: list[tuple[str, str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = [
        ("baseline", "baseline serial", run_baseline_pair),
        ("stream_overlap", "stream overlap", run_stream_overlap_pair),
        ("single_fused", "single fused", run_single_fused_pair),
        ("physical_concat", "physical concat", run_physical_concat_pair),
    ]

    row: dict[str, object] = {
        "experiment": spec.experiment_name,
        "schedule_name": schedule.name,
        "num_sms": schedule.num_sms,
        "launched_workers": schedule.launched_workers,
        "op1_tiles": schedule.op1_tiles,
        "op3_tiles": schedule.op3_tiles,
        "op1_workers": schedule.op1_workers,
        "op3_workers": schedule.op3_workers,
        "range_ops": "|".join(schedule.range_ops),
        "range_sizes": "|".join(str(size) for size in schedule.range_sizes),
        "range_bases": "|".join(str(base) for base in schedule.op_worker_bases),
        "range_pid_spans": "|".join(
            f"[{segment.pid_start},{segment.pid_end})->{segment.op_name}"
            for segment in schedule_segments(schedule)
        ),
        "range_tile_spans": "|".join(
            f"{segment.op_name}[{segment.op_tile_start},{segment.op_tile_end})"
            for segment in schedule_segments(schedule)
        ),
    }
    add_shape_fields(row, shape, args.warmup, args.repeat)

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
            f"{spec.experiment_name} {timing_title} pair",
            run_pair,
            args.warmup,
            args.repeat,
        )
        full_timing = measure_cuda_time(
            f"{spec.experiment_name} {timing_title} full",
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
    for method_name in ("stream_overlap", "single_fused", "physical_concat"):
        row[f"{method_name}_pair_speedup_vs_baseline"] = baseline_pair / timing_by_name[method_name][0].median_ms
        row[f"{method_name}_full_speedup_vs_baseline"] = baseline_full / timing_by_name[method_name][1].median_ms

    append_csv(args.output, row)

    print(f"\n{spec.experiment_name} benchmark 结果：")
    for method_name, timing_title, _ in methods:
        pair_timing, full_timing = timing_by_name[method_name]
        errors = errors_by_name[method_name]
        print(
            f"  {timing_title}: pair={pair_timing.median_ms:.6f} ms, "
            f"full={full_timing.median_ms:.6f} ms, correct={errors.all_correct}"
        )
    print(f"  stream overlap pair vs baseline: {row['stream_overlap_pair_speedup_vs_baseline']:.4f}")
    print(f"  single fused pair vs baseline: {row['single_fused_pair_speedup_vs_baseline']:.4f}")
    print(f"  physical concat pair vs baseline: {row['physical_concat_pair_speedup_vs_baseline']:.4f}")
    print(f"  single fused full vs baseline: {row['single_fused_full_speedup_vs_baseline']:.4f}")
    print(f"  CSV 已写入: {args.output}")
