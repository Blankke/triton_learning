#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_nsys.sh step1
#   bash scripts/profile_nsys.sh scheme1
#   bash scripts/profile_nsys.sh scheme2
#   VARIANT=physical_precat bash scripts/profile_nsys.sh scheme3
#
# 说明：
#   用 benchmark 自带的 --profile-only 模式采集 Nsight Systems 时间线。
#   结果会写入 outputs/nsys/<target>.nsys-rep，并额外导出常用 stats 文本：
#     - cuda_gpu_kern_sum
#     - cuda_gpu_trace
#     - cuda_api_sum
#   可通过环境变量覆盖形状与 profiling 轮次：
#     M/H/N/R/DTYPE/PROFILE_WARMUP/PROFILE_REPEAT/GPU_METRICS_DEVICE

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "用法：bash scripts/profile_nsys.sh <step1|scheme1|scheme2|scheme3>"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

mkdir -p "$ROOT_DIR/outputs/nsys"

MODULE=""
REPORT_TAG="$TARGET"
EXTRA_ARGS=()

case "$TARGET" in
  step1)
    MODULE="triton_learning.bench_step1_baseline"
    ;;
  scheme1)
    MODULE="fusion.bench_scheme1_spatial_sharing"
    ;;
  scheme2)
    MODULE="fusion.bench_scheme2_horizontal_fusion"
    ;;
  scheme3)
    MODULE="fusion.bench_scheme3_column_concat"
    if [[ -n "${VARIANT:-}" ]]; then
      EXTRA_ARGS+=(--variant "$VARIANT")
      REPORT_TAG="${REPORT_TAG}_${VARIANT}"
    fi
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
