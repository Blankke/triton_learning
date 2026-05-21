#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_sharing.sh
#   bash scripts/profile_sharing.sh experiment1
#   bash scripts/profile_sharing.sh experiment2 single_fused
#   TOOL=nsys bash scripts/profile_sharing.sh experiment1 baseline
#   TOOL=ncu bash scripts/profile_sharing.sh experiment2 physical_concat
#
# 说明：
#   这是 sharing 实验的 profiling 总入口，会统一生成：
#     - `output/sharing/nsys/*.nsys-rep`
#     - `output/sharing/ncu/*.ncu-rep`
#     - 以及对应导出的文本 / CSV / log
#   目标定义如下：
#     - experiment1：前半 worker 做 op1，后半 worker 做 op3
#     - experiment2：四段 pid range 交错做 op1 / op3 / op1 / op3
#   方法定义如下：
#     - baseline
#     - stream_overlap
#     - single_fused
#     - physical_concat
#   默认行为：
#     - 不传参数时，会按 experiment1/2 的四种方法全部执行
#     - `TOOL=all` 时同时跑 nsys 与 ncu
#     - `TOOL=nsys` 或 `TOOL=ncu` 时只跑其中一种
#   常用环境变量：
#     - M/H/N/R/DTYPE（默认 R=2048）
#     - SHARING_NUM_WORKERS（若显式传入，必须等于 `num_tiles_1 + num_tiles_3`）
#     - NSYS_PROFILE_WARMUP / NSYS_PROFILE_REPEAT
#     - NCU_PROFILE_WARMUP / NCU_PROFILE_REPEAT
#     - NCU_SET=full
#     - GPU_METRICS_DEVICE=all
#     - NSYS_USE_CAPTURE_RANGE=0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-all}"
METHOD="${2:-all}"
TOOL="${TOOL:-all}"
NSYS_CAPTURE_RANGE_NAME="${NSYS_CAPTURE_RANGE_NAME:-steady_state_capture}"
declare -a FAILED_TARGETS=()

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

mkdir -p "$ROOT_DIR/output/sharing/nsys" "$ROOT_DIR/output/sharing/ncu"

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
  local target="$1"
  case "$target" in
    experiment1)
      printf '%s\n' "sharing.bench_experiment1_half_split"
      ;;
    experiment2)
      printf '%s\n' "sharing.bench_experiment2_interleaved"
      ;;
    *)
      echo "不支持的 experiment: $target" >&2
      return 1
      ;;
  esac
}

run_nsys_target() {
  local target="$1"
  local method="$2"
  if ! command -v nsys >/dev/null 2>&1; then
    warn "未找到 nsys，跳过 $target/$method。"
    record_failure "nsys/$target/$method：本机未安装 nsys"
    return 0
  fi

  local module
  module="$(resolve_module "$target")"
  local report_tag="${target}_${method}"
  local report_base="$ROOT_DIR/output/sharing/nsys/${report_tag}"
  local report_path="${report_base}.nsys-rep"
  local sqlite_path="${report_base}.sqlite"
  local profile_log="${report_base}_profile.log"
  local gpu_metrics_args=()
  local gpu_metrics_device_value="${GPU_METRICS_DEVICE:-}"
  local warmup="${NSYS_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NSYS_PROFILE_REPEAT:-${PROFILE_REPEAT:-20}}"
  local profile_status=0
  local stats_report=""
  local stats_status=0
  local use_capture_range="${NSYS_USE_CAPTURE_RANGE:-1}"

  if [[ -n "$gpu_metrics_device_value" ]]; then
    gpu_metrics_args+=(--gpu-metrics-devices="$gpu_metrics_device_value")
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
  echo "===== Nsight Systems: $target / $method ====="
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
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --capture-range=nvtx \
      --nvtx-capture="$NSYS_CAPTURE_RANGE_NAME" \
      --capture-range-end=stop-shutdown \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "${M:-64}" \
        --h "${H:-4096}" \
        --n "${N:-28672}" \
        --r "${R:-2048}" \
        --dtype "${DTYPE:-fp16}" \
        --num-workers "${SHARING_NUM_WORKERS:-0}" \
        --profile-only \
        --profile-target "$method" \
        --profile-warmup "$warmup" \
        --profile-repeat "$repeat" 2>&1 | tee "$profile_log"
  else
    nsys profile \
      -t cuda,nvtx,osrt \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "${M:-64}" \
        --h "${H:-4096}" \
        --n "${N:-28672}" \
        --r "${R:-2048}" \
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
    warn "使用 NVTX capture-range 未生成报告，改为抓整个 profiling workload 重试一次。"
    rm -f "$report_path" "$sqlite_path"
    {
      echo
      echo "[nsys fallback] 不使用 capture-range 重试"
    } | tee -a "$profile_log"
    set +e
    nsys profile \
      -t cuda,nvtx,osrt \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      "${gpu_metrics_args[@]}" \
      -o "$report_base" \
      python -m "$module" \
        --m "${M:-64}" \
        --h "${H:-4096}" \
        --n "${N:-28672}" \
        --r "${R:-2048}" \
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
    record_failure "nsys/$target/$method：采集失败（退出码 $profile_status）"
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
      warn "nsys stats 导出失败：$stats_report（详见 ${report_base}_${stats_report}.log）"
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
  local target="$1"
  local method="$2"
  if ! command -v ncu >/dev/null 2>&1; then
    warn "未找到 ncu，跳过 $target/$method。"
    record_failure "ncu/$target/$method：本机未安装 ncu"
    return 0
  fi

  local module
  module="$(resolve_module "$target")"
  local report_tag="${target}_${method}"
  local report_file="$ROOT_DIR/output/sharing/ncu/${report_tag}.csv"
  local report_base="$ROOT_DIR/output/sharing/ncu/${report_tag}"
  local report_path="${report_base}.ncu-rep"
  local collect_log="${report_base}_collect.log"
  local export_log="${report_base}_export.log"
  local warmup="${NCU_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NCU_PROFILE_REPEAT:-${PROFILE_REPEAT:-1}}"
  local default_metrics="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
  local ncu_args=(--target-processes all --force-overwrite --export "$report_base")
  local collect_status=0
  local export_status=0

  if [[ -n "${NCU_SET:-}" ]]; then
    ncu_args+=(--set "$NCU_SET")
  else
    ncu_args+=(--metrics "${NCU_METRICS:-$default_metrics}")
  fi

  rm -f "$report_file" "$report_path" "$collect_log" "$export_log"

  echo
  echo "===== Nsight Compute: $target / $method ====="
  print_python_env
  echo "开始采集 Nsight Compute: $report_file"
  echo "同时导出 Nsight Compute report: $report_path"

  set +e
  ncu \
    "${ncu_args[@]}" \
    --log-file "$collect_log" \
    python -m "$module" \
      --m "${M:-64}" \
      --h "${H:-4096}" \
      --n "${N:-28672}" \
      --r "${R:-2048}" \
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
    record_failure "ncu/$target/$method：采集失败（退出码 $collect_status）"
    return 0
  fi

  if [[ $collect_status -ne 0 ]]; then
    warn "Nsight Compute 返回码为 $collect_status，但报告已生成，继续导出 CSV。日志：$collect_log"
  fi

  set +e
  ncu --import "$report_path" --page raw --csv > "$report_file" 2> "$export_log"
  export_status=$?
  set -e

  if [[ $export_status -ne 0 || ! -s "$report_file" ]]; then
    warn "Nsight Compute CSV 导出失败，但 .ncu-rep 已保留。详见 $export_log"
  fi

  echo "Nsight Compute 结果已写入："
  echo "  $report_file"
  echo "  $report_path"
  echo "  $collect_log"
  echo "  $export_log"
}

run_target() {
  local target="$1"
  local method="$2"
  case "$TOOL" in
    all)
      run_nsys_target "$target" "$method"
      run_ncu_target "$target" "$method"
      ;;
    nsys)
      run_nsys_target "$target" "$method"
      ;;
    ncu)
      run_ncu_target "$target" "$method"
      ;;
    *)
      echo "不支持的 TOOL=$TOOL，请使用 all / nsys / ncu" >&2
      exit 1
      ;;
  esac
}

run_methods_for_target() {
  local target="$1"
  case "$METHOD" in
    all)
      run_target "$target" baseline
      run_target "$target" stream_overlap
      run_target "$target" single_fused
      run_target "$target" physical_concat
      ;;
    baseline|stream_overlap|single_fused|physical_concat)
      run_target "$target" "$METHOD"
      ;;
    *)
      echo "不支持的 method: $METHOD" >&2
      exit 1
      ;;
  esac
}

case "$TARGET" in
  all)
    run_methods_for_target experiment1
    run_methods_for_target experiment2
    ;;
  experiment1|experiment2)
    run_methods_for_target "$TARGET"
    ;;
  *)
    echo "用法：bash scripts/profile_sharing.sh [all|experiment1|experiment2] [all|baseline|stream_overlap|single_fused|physical_concat]" >&2
    exit 1
    ;;
esac

if ((${#FAILED_TARGETS[@]} > 0)); then
  echo
  echo "以下 profiling 子任务失败："
  for failure_item in "${FAILED_TARGETS[@]}"; do
    echo "  - $failure_item"
  done
  exit 1
fi

echo
echo "全部 sharing profiling 任务执行完成。"
