#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_sharing_all.sh
#   RUN_PROFILE=0 bash scripts/run_sharing_all.sh
#   TOOL=nsys bash scripts/run_sharing_all.sh single_fused_half_split
#
# 说明：
#   先执行 sharing 五路对比 benchmark，并默认继续执行 sharing profiling。
#   默认使用构造 workload `r=2048`，输出会落到：
#     - output/sharing/benchmarks/sharing_five_way_comparison.csv
#     - output/sharing/nsys/*.nsys-rep
#     - output/sharing/ncu/*.ncu-rep
#   如只想跑 benchmark，不想顺带生成 profiling report，可设置 `RUN_PROFILE=0`。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE_METHOD="${1:-all}"
RUN_PROFILE="${RUN_PROFILE:-1}"

bash "$ROOT_DIR/scripts/run_sharing_benchmark.sh"

if [[ "$RUN_PROFILE" != "0" ]]; then
  TOOL="${TOOL:-all}" bash "$ROOT_DIR/scripts/profile_sharing.sh" "$PROFILE_METHOD"
fi
