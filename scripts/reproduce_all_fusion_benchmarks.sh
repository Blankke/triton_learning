#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/reproduce_all_fusion_benchmarks.sh
#
# 说明：
#   一条命令复现原版 baseline 与三个“算子1 + 算子3”融合 benchmark。
#   脚本会自动创建/复用仓库内 `.venv`，并安装 `requirements-cu128.txt` 中的依赖。
#   如需缩短验证时间，可以在命令前设置：
#   WARMUP=5 REPEAT=20 bash scripts/reproduce_all_fusion_benchmarks.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"

bash "$ROOT_DIR/scripts/run_step1_baseline.sh"
bash "$ROOT_DIR/scripts/run_scheme1_spatial_sharing.sh"
bash "$ROOT_DIR/scripts/run_scheme2_horizontal_fusion.sh"
bash "$ROOT_DIR/scripts/run_scheme3_column_concat.sh"
