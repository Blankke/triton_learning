#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_ncu.sh baseline
#   bash scripts/profile_ncu.sh scheme1
#   bash scripts/profile_ncu.sh scheme2
#   VARIANT=physical_precat bash scripts/profile_ncu.sh scheme3
#   NCU_SET=full bash scripts/profile_ncu.sh scheme2
#
# 说明：
#   每次只采一份“单一口径”的 Nsight Compute 报告，避免把 baseline 和方案本体混在同一份 CSV 里。
#   四类报告分别是：
#     - baseline：只看原始串行里的 `X@A` 与 `X@C`
#     - scheme1：只看方案1的 two-stream concurrent pair
#     - scheme2：只看方案2的 horizontal fused pair
#     - scheme3：只看方案3某一个变体的 pair（默认 physical_precat）
#   默认只抓老师当前关心的三层分析相关指标；若想切到完整采集，可设置环境变量 NCU_SET=full。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "用法：bash scripts/profile_ncu.sh <baseline|scheme1|scheme2|scheme3>"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

mkdir -p "$ROOT_DIR/outputs/ncu"

MODULE=""
REPORT_TAG=""
EXTRA_ARGS=()

case "$TARGET" in
  baseline)
    MODULE="triton_learning.bench_step1_baseline"
    REPORT_TAG="baseline_serial_yw"
    ;;
  scheme1)
    MODULE="fusion.bench_scheme1_spatial_sharing"
    REPORT_TAG="scheme1_concurrent_pair"
    ;;
  scheme2)
    MODULE="fusion.bench_scheme2_horizontal_fusion"
    REPORT_TAG="scheme2_horizontal_pair"
    ;;
  scheme3)
    MODULE="fusion.bench_scheme3_column_concat"
    VARIANT_NAME="${VARIANT:-physical_precat}"
    EXTRA_ARGS+=(--variant "$VARIANT_NAME")
    REPORT_TAG="scheme3_${VARIANT_NAME}_pair"
    ;;
  *)
    echo "不支持的 target: $TARGET"
    exit 1
    ;;
esac

REPORT_FILE="$ROOT_DIR/outputs/ncu/${REPORT_TAG}.csv"
REPORT_BASE="$ROOT_DIR/outputs/ncu/${REPORT_TAG}"
DEFAULT_METRICS="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

NCU_ARGS=(--target-processes all --force-overwrite --export "$REPORT_BASE" --csv --log-file "$REPORT_FILE")
if [[ -n "${NCU_SET:-}" ]]; then
  NCU_ARGS+=(--set "$NCU_SET")
else
  NCU_ARGS+=(--metrics "${NCU_METRICS:-$DEFAULT_METRICS}")
fi

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"
echo "开始采集 Nsight Compute: $REPORT_FILE"
echo "同时导出 Nsight Compute report: ${REPORT_BASE}.ncu-rep"

ncu \
  "${NCU_ARGS[@]}" \
  python -m "$MODULE" \
    --m "${M:-64}" \
    --h "${H:-4096}" \
    --n "${N:-28672}" \
    --r "${R:-8}" \
    --dtype "${DTYPE:-fp16}" \
    --profile-only \
    --profile-warmup "${PROFILE_WARMUP:-1}" \
    --profile-repeat "${PROFILE_REPEAT:-1}" \
    "${EXTRA_ARGS[@]}"

echo "Nsight Compute 结果已写入："
echo "  $REPORT_FILE"
echo "  ${REPORT_BASE}.ncu-rep"
