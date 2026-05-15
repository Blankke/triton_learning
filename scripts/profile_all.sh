#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/profile_all.sh
#   TOOL=nsys bash scripts/profile_all.sh
#   TOOL=ncu VARIANT=physical_precat bash scripts/profile_all.sh
#
# 说明：
#   一次性跑完四类独立 profiling 报告：
#     1. baseline：原始串行中的 X@A / X@C
#     2. scheme1：方案1 的 concurrent pair
#     3. scheme2：方案2 的 horizontal fused pair
#     4. scheme3：方案3 指定变体的 pair，默认 physical_precat
#   可通过环境变量控制：
#     - TOOL=all|nsys|ncu
#     - VARIANT=physical_precat|logical_no_pad|logical_rpad_128|logical_c_first_no_pad
#     - M/H/N/R/DTYPE/PROFILE_WARMUP/PROFILE_REPEAT/NCU_SET/GPU_METRICS_DEVICE

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOOL="${TOOL:-all}"
VARIANT_NAME="${VARIANT:-physical_precat}"

run_nsys() {
  echo
  echo "===== Nsight Systems: baseline ====="
  bash "$ROOT_DIR/scripts/profile_nsys.sh" baseline

  echo
  echo "===== Nsight Systems: scheme1 ====="
  bash "$ROOT_DIR/scripts/profile_nsys.sh" scheme1

  echo
  echo "===== Nsight Systems: scheme2 ====="
  bash "$ROOT_DIR/scripts/profile_nsys.sh" scheme2

  echo
  echo "===== Nsight Systems: scheme3 ($VARIANT_NAME) ====="
  VARIANT="$VARIANT_NAME" bash "$ROOT_DIR/scripts/profile_nsys.sh" scheme3
}

run_ncu() {
  echo
  echo "===== Nsight Compute: baseline ====="
  bash "$ROOT_DIR/scripts/profile_ncu.sh" baseline

  echo
  echo "===== Nsight Compute: scheme1 ====="
  bash "$ROOT_DIR/scripts/profile_ncu.sh" scheme1

  echo
  echo "===== Nsight Compute: scheme2 ====="
  bash "$ROOT_DIR/scripts/profile_ncu.sh" scheme2

  echo
  echo "===== Nsight Compute: scheme3 ($VARIANT_NAME) ====="
  VARIANT="$VARIANT_NAME" bash "$ROOT_DIR/scripts/profile_ncu.sh" scheme3
}

case "$TOOL" in
  all)
    run_nsys
    run_ncu
    ;;
  nsys)
    run_nsys
    ;;
  ncu)
    run_ncu
    ;;
  *)
    echo "不支持的 TOOL=$TOOL，请使用 all / nsys / ncu"
    exit 1
    ;;
esac

echo
echo "全部 profiling 任务执行完成。"
