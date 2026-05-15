#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_nsys.sh baseline
#   bash scripts/profile_nsys.sh scheme1
#   bash scripts/profile_nsys.sh scheme2
#   VARIANT=physical_precat bash scripts/profile_nsys.sh scheme3
#
# 说明：
#   每次只采一份“单一口径”的 Nsight Systems 报告，避免把 baseline 和方案本体混在同一份时间线里。
#   四类报告分别是：
#     - baseline：只看原始串行里的 `X@A` 与 `X@C`
#     - scheme1：只看方案1的 two-stream concurrent pair
#     - scheme2：只看方案2的 horizontal fused pair
#     - scheme3：只看方案3某一个变体的 pair（默认 physical_precat）
#   结果会写入 outputs/nsys/<report_tag>.nsys-rep，并额外导出常用 stats 文本。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "用法：bash scripts/profile_nsys.sh <baseline|scheme1|scheme2|scheme3>"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

mkdir -p "$ROOT_DIR/outputs/nsys"

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

GPU_METRICS_ARGS=()
if [[ -n "${GPU_METRICS_DEVICE:-all}" ]]; then
  GPU_METRICS_ARGS+=(--gpu-metrics-device="${GPU_METRICS_DEVICE:-all}")
fi

REPORT_BASE="$ROOT_DIR/outputs/nsys/${REPORT_TAG}"

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"
echo "开始采集 Nsight Systems: $REPORT_BASE.nsys-rep"

nsys profile \
  -t cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  "${GPU_METRICS_ARGS[@]}" \
  -o "$REPORT_BASE" \
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

nsys stats --report cuda_gpu_kern_sum "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_cuda_gpu_kern_sum.txt"
nsys stats --report cuda_gpu_trace "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_cuda_gpu_trace.txt"
nsys stats --report cuda_api_sum "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_cuda_api_sum.txt"

echo "Nsight Systems 结果已写入："
echo "  ${REPORT_BASE}.nsys-rep"
echo "  ${REPORT_BASE}_cuda_gpu_kern_sum.txt"
echo "  ${REPORT_BASE}_cuda_gpu_trace.txt"
echo "  ${REPORT_BASE}_cuda_api_sum.txt"
