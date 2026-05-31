#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_sharing_profile_cases.sh
#   SCENARIO=real_r8 bash scripts/run_sharing_profile_cases.sh
#   SCENARIO=constructed_50_blocks PROFILE_METHOD=all bash scripts/run_sharing_profile_cases.sh
#   RUN_BENCH=0 TOOL=ncu NCU_SET=full bash scripts/run_sharing_profile_cases.sh
#
# 说明：
#   统一执行 sharing 的两组关键场景，并把结果拆分写入不同文件夹：
#     1. real_r8：真实 gateup 场景，M=64 H=4096 N=28672 R=8
#     2. constructed_50_blocks：构造场景，M=64 H=4096 N=1664 R=1664
#        按当前 sharing 代码采用的 16x128 参考口径，op1/op3 都约 52 blocks，
#        接近“每个 op 约 50 block”的目标。
#   默认行为：
#     - `SCENARIO=both`：依次执行上述两组场景
#     - 每组场景先跑一次 sharing 五路 benchmark
#     - 然后对 `PROFILE_METHOD=stream_overlap` 生成 Nsight Systems + Nsight Compute 报告
#   输出目录：
#     - `output/sharing/scenarios/<scenario>/benchmarks/`
#     - `output/sharing/scenarios/<scenario>/nsys/`
#     - `output/sharing/scenarios/<scenario>/ncu/`
#   常用环境变量：
#     - `SCENARIO=both|real_r8|constructed_50_blocks`
#     - `PROFILE_METHOD=stream_overlap|baseline|single_fused_half_split|single_fused_interleaved|physical_concat|all`
#     - `TOOL=all|nsys|ncu`
#     - `RUN_BENCH=0|1`
#     - `DTYPE=fp16|bf16|fp32`
#     - `WARMUP` / `REPEAT`
#     - `NSYS_PROFILE_WARMUP` / `NSYS_PROFILE_REPEAT` / `NSYS_USE_CAPTURE_RANGE`
#     - `NCU_PROFILE_WARMUP` / `NCU_PROFILE_REPEAT` / `NCU_SET`（默认 full）
#     - `SHARING_NUM_WORKERS`（若显式传入，必须等于 total_tiles）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO="${SCENARIO:-both}"
PROFILE_METHOD="${PROFILE_METHOD:-stream_overlap}"
TOOL="${TOOL:-all}"
RUN_BENCH="${RUN_BENCH:-1}"
NSYS_CAPTURE_RANGE_NAME="${NSYS_CAPTURE_RANGE_NAME:-steady_state_capture}"
declare -a FAILED_TARGETS=()

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

warn() {
  echo "警告: $*" >&2
}

record_failure() {
  local item="$1"
  FAILED_TARGETS+=("$item")
}

print_python_env() {
  echo "当前 Python："
  which python
  python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"
}

resolve_module() {
  printf '%s\n' "sharing.bench_five_way_comparison"
}

resolve_scenario() {
  local scenario="$1"
  case "$scenario" in
    real_r8)
      printf '%s\n' \
        "64" \
        "4096" \
        "28672" \
        "8" \
        "真实 gateup 场景（r=8）" \
        "4" \
        "896"
      ;;
    constructed_50_blocks)
      printf '%s\n' \
        "64" \
        "4096" \
        "1664" \
        "1664" \
        "构造场景（按 16x128 参考口径，op1/op3 各约 52 blocks）" \
        "52" \
        "52"
      ;;
    *)
      echo "不支持的 SCENARIO=$scenario，请使用 both / real_r8 / constructed_50_blocks" >&2
      return 1
      ;;
  esac
}

resolve_output_root() {
  local scenario="$1"
  printf '%s\n' "$ROOT_DIR/output/sharing/scenarios/$scenario"
}

write_scenario_config() {
  local scenario="$1"
  local output_root="$2"
  local m="$3"
  local h="$4"
  local n="$5"
  local r="$6"
  local description="$7"
  local op1_blocks="$8"
  local op3_blocks="$9"

  cat > "$output_root/run_config.txt" <<EOF
scenario=$scenario
description=$description
m=$m
h=$h
n=$n
r=$r
dtype=${DTYPE:-fp16}
profile_method=$PROFILE_METHOD
tool=$TOOL
run_bench=$RUN_BENCH
reference_block_estimate_op1=$op1_blocks
reference_block_estimate_op3=$op3_blocks
nsys_profile_warmup=${NSYS_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}
nsys_profile_repeat=${NSYS_PROFILE_REPEAT:-${PROFILE_REPEAT:-20}}
ncu_profile_warmup=${NCU_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}
ncu_profile_repeat=${NCU_PROFILE_REPEAT:-${PROFILE_REPEAT:-1}}
ncu_set=${NCU_SET:-full}
EOF
}

resolve_ncu_kernel_name() {
  local method="$1"
  case "$method" in
    baseline|stream_overlap|physical_concat)
      printf '%s\n' "regex:.*_matmul_kernel.*"
      ;;
    single_fused_half_split|single_fused_interleaved)
      printf '%s\n' "regex:.*_range_fused_kernel.*"
      ;;
    *)
      echo "不支持的 method: $method" >&2
      return 1
      ;;
  esac
}

resolve_ncu_launch_count() {
  local method="$1"
  case "$method" in
    baseline|stream_overlap)
      printf '%s\n' "2"
      ;;
    physical_concat|single_fused_half_split|single_fused_interleaved)
      printf '%s\n' "1"
      ;;
    *)
      echo "不支持的 method: $method" >&2
      return 1
      ;;
  esac
}

resolve_ncu_nvtx_include() {
  local method="$1"
  case "$method" in
    baseline)
      printf '%s\n' "sharing/five_way/baseline_pair/"
      ;;
    stream_overlap)
      printf '%s\n' "sharing/five_way/stream_overlap_pair/"
      ;;
    single_fused_half_split)
      printf '%s\n' "sharing/five_way/single_fused_half_split_pair/"
      ;;
    single_fused_interleaved)
      printf '%s\n' "sharing/five_way/single_fused_interleaved_pair/"
      ;;
    physical_concat)
      printf '%s\n' "sharing/five_way/physical_concat_pair/"
      ;;
    *)
      echo "不支持的 method: $method" >&2
      return 1
      ;;
  esac
}

run_benchmark_for_scenario() {
  local scenario="$1"
  local output_root="$2"
  local m="$3"
  local h="$4"
  local n="$5"
  local r="$6"
  local module
  module="$(resolve_module)"
  local benchmark_output="$output_root/benchmarks/sharing_five_way_comparison.csv"

  echo
  echo "===== sharing benchmark: $scenario ====="
  print_python_env
  python -m "$module" \
    --m "$m" \
    --h "$h" \
    --n "$n" \
    --r "$r" \
    --dtype "${DTYPE:-fp16}" \
    --num-workers "${SHARING_NUM_WORKERS:-0}" \
    --warmup "${WARMUP:-30}" \
    --repeat "${REPEAT:-100}" \
    --output "$benchmark_output"
  echo "benchmark 结果已写入：$benchmark_output"
}

run_nsys_target() {
  local scenario="$1"
  local method="$2"
  local output_root="$3"
  local m="$4"
  local h="$5"
  local n="$6"
  local r="$7"
  if ! command -v nsys >/dev/null 2>&1; then
    warn "未找到 nsys，跳过 $scenario/$method。"
    record_failure "nsys/$scenario/$method：本机未安装 nsys"
    return 0
  fi

  local module
  module="$(resolve_module)"
  local report_base="$output_root/nsys/sharing_${method}"
  local report_path="${report_base}.nsys-rep"
  local sqlite_path="${report_base}.sqlite"
  local profile_log="${report_base}_profile.log"
  local warmup="${NSYS_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NSYS_PROFILE_REPEAT:-${PROFILE_REPEAT:-20}}"
  local use_capture_range="${NSYS_USE_CAPTURE_RANGE:-1}"
  local profile_status=0
  local stats_status=0
  local stats_report=""
  local gpu_metrics_args=()

  if [[ -n "${GPU_METRICS_DEVICE:-}" ]]; then
    gpu_metrics_args+=(--gpu-metrics-device="${GPU_METRICS_DEVICE}")
  fi

  rm -f \
    "$report_path" \
    "$sqlite_path" \
    "$profile_log" \
    "${report_base}_cuda_gpu_kern_sum.txt" \
    "${report_base}_cuda_gpu_kern_sum.log" \
    "${report_base}_cuda_gpu_trace.txt" \
    "${report_base}_cuda_gpu_trace.log" \
    "${report_base}_cuda_api_sum.txt" \
    "${report_base}_cuda_api_sum.log"

  echo
  echo "===== Nsight Systems: $scenario / $method ====="
  print_python_env
  echo "开始采集 Nsight Systems: $report_path"
  if [[ "$use_capture_range" != "0" ]]; then
    echo "优先只抓 NVTX steady-state 区间: $NSYS_CAPTURE_RANGE_NAME"
  else
    echo "已关闭 NVTX capture-range，将直接抓整个 profiling workload"
  fi

  set +e
  if [[ "$use_capture_range" != "0" ]]; then
    nsys profile \
      -t cuda,nvtx,osrt \
      -s none \
      --cpuctxsw=none \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --capture-range=nvtx \
      --nvtx-capture="$NSYS_CAPTURE_RANGE_NAME" \
      --capture-range-end=stop-shutdown \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "$m" \
        --h "$h" \
        --n "$n" \
        --r "$r" \
        --dtype "${DTYPE:-fp16}" \
        --num-workers "${SHARING_NUM_WORKERS:-0}" \
        --profile-only \
        --profile-target "$method" \
        --profile-warmup "$warmup" \
        --profile-repeat "$repeat" 2>&1 | tee "$profile_log"
  else
    nsys profile \
      -t cuda,nvtx,osrt \
      -s none \
      --cpuctxsw=none \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "$m" \
        --h "$h" \
        --n "$n" \
        --r "$r" \
        --dtype "${DTYPE:-fp16}" \
        --num-workers "${SHARING_NUM_WORKERS:-0}" \
        --profile-only \
        --profile-target "$method" \
        --profile-warmup "$warmup" \
        --profile-repeat "$repeat" 2>&1 | tee "$profile_log"
  fi
  profile_status=${PIPESTATUS[0]}
  set -e

  if [[ ! -s "$report_path" && "$use_capture_range" != "0" ]]; then
    warn "NVTX capture-range 未生成报告，改为抓整个 profiling workload 重试一次。"
    rm -f "$report_path" "$sqlite_path"
    {
      echo
      echo "[nsys fallback] 不使用 capture-range 重试"
    } | tee -a "$profile_log"
    set +e
    nsys profile \
      -t cuda,nvtx,osrt \
      -s none \
      --cpuctxsw=none \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "$m" \
        --h "$h" \
        --n "$n" \
        --r "$r" \
        --dtype "${DTYPE:-fp16}" \
        --num-workers "${SHARING_NUM_WORKERS:-0}" \
        --profile-only \
        --profile-target "$method" \
        --profile-warmup "$warmup" \
        --profile-repeat "$repeat" 2>&1 | tee -a "$profile_log"
    profile_status=${PIPESTATUS[0]}
    set -e
  fi

  if [[ ! -s "$report_path" ]]; then
    warn "Nsight Systems 采集失败，未生成报告。日志：$profile_log"
    record_failure "nsys/$scenario/$method：采集失败（退出码 $profile_status）"
    return 0
  fi

  if [[ $profile_status -ne 0 ]]; then
    warn "Nsight Systems 返回码为 $profile_status，但报告已生成，继续导出 stats。日志：$profile_log"
  fi

  for stats_report in cuda_gpu_kern_sum cuda_gpu_trace cuda_api_sum; do
    set +e
    nsys stats --report "$stats_report" "$report_path" \
      > "${report_base}_${stats_report}.txt" \
      2> "${report_base}_${stats_report}.log"
    stats_status=$?
    set -e
    if [[ $stats_status -ne 0 ]]; then
      warn "nsys stats 导出失败：$scenario/$method/$stats_report"
    fi
  done

  echo "Nsight Systems 结果已写入："
  echo "  $report_path"
  echo "  $profile_log"
  echo "  ${report_base}_cuda_gpu_kern_sum.txt"
  echo "  ${report_base}_cuda_gpu_trace.txt"
  echo "  ${report_base}_cuda_api_sum.txt"
}

run_ncu_target() {
  local scenario="$1"
  local method="$2"
  local output_root="$3"
  local m="$4"
  local h="$5"
  local n="$6"
  local r="$7"
  if ! command -v ncu >/dev/null 2>&1; then
    warn "未找到 ncu，跳过 $scenario/$method。"
    record_failure "ncu/$scenario/$method：本机未安装 ncu"
    return 0
  fi

  local module
  module="$(resolve_module)"
  local report_base="$output_root/ncu/sharing_${method}"
  local report_path="${report_base}.ncu-rep"
  local collect_log="${report_base}_collect.log"
  local raw_file="${report_base}_raw.csv"
  local raw_log="${report_base}_raw.log"
  local details_file="${report_base}_details.csv"
  local details_log="${report_base}_details.log"
  local warmup="${NCU_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NCU_PROFILE_REPEAT:-${PROFILE_REPEAT:-1}}"
  local section_set="${NCU_SET:-full}"
  local kernel_name_filter="${NCU_KERNEL_NAME:-}"
  local nvtx_include_filter="${NCU_NVTX_INCLUDE:-}"
  local launch_count_override="${NCU_LAUNCH_COUNT:-}"
  local launch_skip="${NCU_LAUNCH_SKIP:-0}"
  local nvtx_rename_mode="${NCU_PRINT_NVTX_RENAME:-kernel}"
  local collect_status=0
  local raw_status=0
  local details_status=0
  local ncu_args=(
    --target-processes all
    --force-overwrite
    --export "$report_base"
    --set "$section_set"
    --replay-mode "${NCU_REPLAY_MODE:-application}"
    --app-replay-match "${NCU_APP_REPLAY_MATCH:-all}"
  )

  if [[ -z "$kernel_name_filter" ]]; then
    kernel_name_filter="$(resolve_ncu_kernel_name "$method")"
  fi
  if [[ -z "$nvtx_include_filter" ]]; then
    nvtx_include_filter="$(resolve_ncu_nvtx_include "$method")"
  fi
  if [[ -z "$launch_count_override" ]]; then
    launch_count_override="$(resolve_ncu_launch_count "$method")"
  fi
  if [[ -n "${NCU_IMPORT_SOURCE:-}" ]]; then
    ncu_args+=(--import-source "${NCU_IMPORT_SOURCE}")
  fi
  if [[ -n "$kernel_name_filter" ]]; then
    ncu_args+=(--kernel-name "$kernel_name_filter")
  fi
  if [[ -n "$nvtx_include_filter" ]]; then
    ncu_args+=(--nvtx --nvtx-include "$nvtx_include_filter")
  fi
  if [[ -n "$nvtx_rename_mode" ]]; then
    ncu_args+=(--print-nvtx-rename "$nvtx_rename_mode")
  fi
  if [[ -n "$launch_count_override" ]]; then
    ncu_args+=(--launch-count "$launch_count_override")
  fi
  ncu_args+=(--launch-skip "$launch_skip")

  rm -f \
    "$report_path" \
    "$collect_log" \
    "$raw_file" \
    "$raw_log" \
    "$details_file" \
    "$details_log"

  echo
  echo "===== Nsight Compute: $scenario / $method ====="
  print_python_env
  echo "开始采集 Nsight Compute: $report_path"
  echo "NCU section set: $section_set"
  echo "NCU replay mode: ${NCU_REPLAY_MODE:-application}"
  echo "NCU app replay match: ${NCU_APP_REPLAY_MATCH:-all}"
  echo "NCU 过滤条件："
  echo "  kernel-name: $kernel_name_filter"
  echo "  nvtx-include: $nvtx_include_filter"
  echo "  launch-skip: $launch_skip"
  echo "  launch-count: $launch_count_override"

  set +e
  ncu \
    "${ncu_args[@]}" \
    --log-file "$collect_log" \
    python -m "$module" \
      --m "$m" \
      --h "$h" \
      --n "$n" \
      --r "$r" \
      --dtype "${DTYPE:-fp16}" \
      --num-workers "${SHARING_NUM_WORKERS:-0}" \
      --profile-only \
      --profile-target "$method" \
      --profile-warmup "$warmup" \
      --profile-repeat "$repeat"
  collect_status=$?
  set -e

  if [[ $collect_status -ne 0 && ! -s "$report_path" ]]; then
    warn "Nsight Compute 采集失败，未生成报告。日志：$collect_log"
    record_failure "ncu/$scenario/$method：采集失败（退出码 $collect_status）"
    return 0
  fi

  if [[ $collect_status -ne 0 ]]; then
    warn "Nsight Compute 返回码为 $collect_status，但报告已生成，继续导出页面。日志：$collect_log"
  fi

  set +e
  ncu --import "$report_path" --page raw --csv > "$raw_file" 2> "$raw_log"
  raw_status=$?
  ncu --import "$report_path" --page details --print-details all --csv > "$details_file" 2> "$details_log"
  details_status=$?
  set -e

  if [[ $raw_status -ne 0 || ! -s "$raw_file" ]]; then
    warn "Nsight Compute raw 页面导出失败，但 .ncu-rep 已保留。详见 $raw_log"
  fi
  if [[ $details_status -ne 0 || ! -s "$details_file" ]]; then
    warn "Nsight Compute details 页面导出失败，但 .ncu-rep 已保留。详见 $details_log"
  fi

  echo "Nsight Compute 结果已写入："
  echo "  $report_path"
  echo "  $collect_log"
  echo "  $raw_file"
  echo "  $details_file"
}

run_profile_target() {
  local scenario="$1"
  local method="$2"
  local output_root="$3"
  local m="$4"
  local h="$5"
  local n="$6"
  local r="$7"
  case "$TOOL" in
    all)
      run_nsys_target "$scenario" "$method" "$output_root" "$m" "$h" "$n" "$r"
      run_ncu_target "$scenario" "$method" "$output_root" "$m" "$h" "$n" "$r"
      ;;
    nsys)
      run_nsys_target "$scenario" "$method" "$output_root" "$m" "$h" "$n" "$r"
      ;;
    ncu)
      run_ncu_target "$scenario" "$method" "$output_root" "$m" "$h" "$n" "$r"
      ;;
    *)
      echo "不支持的 TOOL=$TOOL，请使用 all / nsys / ncu" >&2
      exit 1
      ;;
  esac
}

run_profile_methods_for_scenario() {
  local scenario="$1"
  local output_root="$2"
  local m="$3"
  local h="$4"
  local n="$5"
  local r="$6"
  case "$PROFILE_METHOD" in
    all)
      run_profile_target "$scenario" baseline "$output_root" "$m" "$h" "$n" "$r"
      run_profile_target "$scenario" stream_overlap "$output_root" "$m" "$h" "$n" "$r"
      run_profile_target "$scenario" single_fused_half_split "$output_root" "$m" "$h" "$n" "$r"
      run_profile_target "$scenario" single_fused_interleaved "$output_root" "$m" "$h" "$n" "$r"
      run_profile_target "$scenario" physical_concat "$output_root" "$m" "$h" "$n" "$r"
      ;;
    baseline|stream_overlap|single_fused_half_split|single_fused_interleaved|physical_concat)
      run_profile_target "$scenario" "$PROFILE_METHOD" "$output_root" "$m" "$h" "$n" "$r"
      ;;
    *)
      echo "不支持的 PROFILE_METHOD=$PROFILE_METHOD" >&2
      exit 1
      ;;
  esac
}

run_scenario() {
  local scenario="$1"
  mapfile -t scenario_info < <(resolve_scenario "$scenario")
  local m="${scenario_info[0]}"
  local h="${scenario_info[1]}"
  local n="${scenario_info[2]}"
  local r="${scenario_info[3]}"
  local description="${scenario_info[4]}"
  local op1_blocks="${scenario_info[5]}"
  local op3_blocks="${scenario_info[6]}"
  local output_root
  output_root="$(resolve_output_root "$scenario")"

  mkdir -p "$output_root/benchmarks" "$output_root/nsys" "$output_root/ncu"
  write_scenario_config "$scenario" "$output_root" "$m" "$h" "$n" "$r" "$description" "$op1_blocks" "$op3_blocks"

  echo
  echo "===== sharing 场景: $scenario ====="
  echo "说明: $description"
  echo "形状: M=$m H=$h N=$n R=$r"
  echo "参考 block 估计: op1≈$op1_blocks, op3≈$op3_blocks"
  echo "输出目录: $output_root"

  if [[ "$RUN_BENCH" != "0" ]]; then
    run_benchmark_for_scenario "$scenario" "$output_root" "$m" "$h" "$n" "$r"
  fi
  run_profile_methods_for_scenario "$scenario" "$output_root" "$m" "$h" "$n" "$r"
}

case "$SCENARIO" in
  both)
    run_scenario real_r8
    run_scenario constructed_50_blocks
    ;;
  real_r8|constructed_50_blocks)
    run_scenario "$SCENARIO"
    ;;
  *)
    echo "不支持的 SCENARIO=$SCENARIO，请使用 both / real_r8 / constructed_50_blocks" >&2
    exit 1
    ;;
esac

if ((${#FAILED_TARGETS[@]} > 0)); then
  echo
  echo "以下 sharing 子任务失败："
  for failure_item in "${FAILED_TARGETS[@]}"; do
    echo "  - $failure_item"
  done
  exit 1
fi

echo
echo "全部 sharing 场景执行完成。"
