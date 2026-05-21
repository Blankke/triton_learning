#!/usr/bin/env bash
# 用法：
#   cd /home/czc/triton_learning
#   bash scripts/run_sharing_all.sh
#
# 说明：
#   顺序执行 sharing 实验1 与 实验2。
#   默认使用构造 workload `r=2048`，输出会分别落到：
#     - output/sharing/benchmarks/experiment1_half_split.csv
#     - output/sharing/benchmarks/experiment2_interleaved.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash "$ROOT_DIR/scripts/run_sharing_experiment1.sh"
bash "$ROOT_DIR/scripts/run_sharing_experiment2.sh"

