#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_suite.sh
#   bash scripts/profile_suite.sh baseline
#   bash scripts/profile_suite.sh scheme2
#   TOOL=nsys bash scripts/profile_suite.sh
#   TOOL=ncu VARIANT=physical_precat bash scripts/profile_suite.sh scheme3
#
# 说明：
#   这是仓库唯一的 profiling 总入口，会统一执行 Triton baseline / 方案1 / 方案2 / 方案3 的
#   Nsight Systems 与 Nsight Compute 采集。
#   目标定义如下：
#     - baseline：Triton 串行 baseline，只看 `Y=X@A` 与 `W=X@C`
#     - scheme1：方案1 的 two-stream concurrent pair
#     - scheme2：方案2 的 horizontal fused pair
#     - scheme3：方案3指定变体的 pair，默认 `physical_precat`
#   默认行为：
#     - 不传位置参数时，按 baseline -> scheme1 -> scheme2 -> scheme3 全部执行
#     - `TOOL=all` 时同时跑 nsys 与 ncu
#     - `TOOL=nsys` 或 `TOOL=ncu` 时只跑其中一种
#   常用环境变量：
#     - VARIANT=physical_precat|logical_no_pad|logical_rpad_128|logical_c_first_no_pad
#     - M/H/N/R/DTYPE
#     - GPU_METRICS_DEVICE=all
#     - NSYS_PROFILE_WARMUP / NSYS_PROFILE_REPEAT（默认 1 / 20）
#     - NCU_PROFILE_WARMUP / NCU_PROFILE_REPEAT（默认 1 / 1）
#     - NCU_SET=full

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-all}"
TOOL="${TOOL:-all}"
VARIANT_NAME="${VARIANT:-physical_precat}"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

mkdir -p "$ROOT_DIR/outputs/nsys" "$ROOT_DIR/outputs/ncu"

resolve_target() {
  local target="$1"
  local module=""
  local report_tag=""
  local extra_args=()

  case "$target" in
    baseline)
      module="fusion.bench_scheme1_spatial_sharing"
      report_tag="baseline_triton_serial_pair"
      extra_args+=(--profile-mode sequential)
      ;;
    scheme1)
      module="fusion.bench_scheme1_spatial_sharing"
      report_tag="scheme1_concurrent_pair"
      extra_args+=(--profile-mode concurrent)
      ;;
    scheme2)
      module="fusion.bench_scheme2_horizontal_fusion"
      report_tag="scheme2_horizontal_pair"
      ;;
    scheme3)
      module="fusion.bench_scheme3_column_concat"
      report_tag="scheme3_${VARIANT_NAME}_pair"
      extra_args+=(--variant "$VARIANT_NAME")
      ;;
    *)
      echo "不支持的 target: $target" >&2
      return 1
      ;;
  esac

  printf '%s\n' "$module"
  printf '%s\n' "$report_tag"
  printf '%s\n' "${extra_args[@]}"
}

run_nsys_target() {
  local target="$1"
  mapfile -t resolved < <(resolve_target "$target")
  local module="${resolved[0]}"
  local report_tag="${resolved[1]}"
  local extra_args=("${resolved[@]:2}")
  local report_base="$ROOT_DIR/outputs/nsys/${report_tag}"
  local gpu_metrics_args=()
  local gpu_metrics_device_value="${GPU_METRICS_DEVICE-all}"
  local warmup="${NSYS_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NSYS_PROFILE_REPEAT:-${PROFILE_REPEAT:-20}}"

  if [[ -n "$gpu_metrics_device_value" ]]; then
    gpu_metrics_args+=(--gpu-metrics-devices="$gpu_metrics_device_value")
  fi

  echo
  echo "===== Nsight Systems: $target ====="
  echo "当前 Python："
  which python
  python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"
  echo "开始采集 Nsight Systems: $report_base.nsys-rep"

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
      --r "${R:-8}" \
      --dtype "${DTYPE:-fp16}" \
      --profile-only \
      --profile-warmup "$warmup" \
      --profile-repeat "$repeat" \
      "${extra_args[@]}"

  nsys stats --report cuda_gpu_kern_sum "${report_base}.nsys-rep" > "${report_base}_cuda_gpu_kern_sum.txt"
  nsys stats --report cuda_gpu_trace "${report_base}.nsys-rep" > "${report_base}_cuda_gpu_trace.txt"
  nsys stats --report cuda_api_sum "${report_base}.nsys-rep" > "${report_base}_cuda_api_sum.txt"

  echo "Nsight Systems 结果已写入："
  echo "  ${report_base}.nsys-rep"
  echo "  ${report_base}_cuda_gpu_kern_sum.txt"
  echo "  ${report_base}_cuda_gpu_trace.txt"
  echo "  ${report_base}_cuda_api_sum.txt"
}

run_ncu_target() {
  local target="$1"
  mapfile -t resolved < <(resolve_target "$target")
  local module="${resolved[0]}"
  local report_tag="${resolved[1]}"
  local extra_args=("${resolved[@]:2}")
  local report_file="$ROOT_DIR/outputs/ncu/${report_tag}.csv"
  local report_base="$ROOT_DIR/outputs/ncu/${report_tag}"
  local warmup="${NCU_PROFILE_WARMUP:-${PROFILE_WARMUP:-1}}"
  local repeat="${NCU_PROFILE_REPEAT:-${PROFILE_REPEAT:-1}}"
  local default_metrics="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
  local ncu_args=(--target-processes all --force-overwrite --export "$report_base" --csv --log-file "$report_file")

  if [[ -n "${NCU_SET:-}" ]]; then
    ncu_args+=(--set "$NCU_SET")
  else
    ncu_args+=(--metrics "${NCU_METRICS:-$default_metrics}")
  fi

  echo
  echo "===== Nsight Compute: $target ====="
  echo "当前 Python："
  which python
  python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"
  echo "开始采集 Nsight Compute: $report_file"
  echo "同时导出 Nsight Compute report: ${report_base}.ncu-rep"

  ncu \
    "${ncu_args[@]}" \
    python -m "$module" \
      --m "${M:-64}" \
      --h "${H:-4096}" \
      --n "${N:-28672}" \
      --r "${R:-8}" \
      --dtype "${DTYPE:-fp16}" \
      --profile-only \
      --profile-warmup "$warmup" \
      --profile-repeat "$repeat" \
      "${extra_args[@]}"

  echo "Nsight Compute 结果已写入："
  echo "  $report_file"
  echo "  ${report_base}.ncu-rep"
}

run_target() {
  local target="$1"
  case "$TOOL" in
    all)
      run_nsys_target "$target"
      run_ncu_target "$target"
      ;;
    nsys)
      run_nsys_target "$target"
      ;;
    ncu)
      run_ncu_target "$target"
      ;;
    *)
      echo "不支持的 TOOL=$TOOL，请使用 all / nsys / ncu" >&2
      exit 1
      ;;
  esac
}

case "$TARGET" in
  all)
    run_target baseline
    run_target scheme1
    run_target scheme2
    run_target scheme3
    ;;
  baseline|scheme1|scheme2|scheme3)
    run_target "$TARGET"
    ;;
  *)
    echo "用法：bash scripts/profile_suite.sh [all|baseline|scheme1|scheme2|scheme3]" >&2
    exit 1
    ;;
esac

echo
echo "全部 profiling 任务执行完成。"
