#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_scheme2_horizontal_fusion.sh
#
# 说明：
#   运行方案2 benchmark。
#   默认会同时测量 `static_pid` 与 `grouped_persistent` 两个单 kernel 口径。
#   如需修改形状，可以在命令前设置环境变量：
#   M=64 H=4096 N=28672 R=8 DTYPE=fp16 bash scripts/run_scheme2_horizontal_fusion.sh
#   如需只跑 grouped_persistent，并手动指定调度超参：
#   SCHEME2_VARIANT=grouped_persistent SCHEME2_NUM_DOWN_WORKERS=4 SCHEME2_CHUNK_SIZE=4 bash scripts/run_scheme2_horizontal_fusion.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"

python -m fusion.bench_scheme2_horizontal_fusion \
  --m "${M:-64}" \
  --h "${H:-4096}" \
  --n "${N:-28672}" \
  --r "${R:-8}" \
  --dtype "${DTYPE:-fp16}" \
  --variant "${SCHEME2_VARIANT:-all}" \
  --num-down-workers "${SCHEME2_NUM_DOWN_WORKERS:-0}" \
  --chunk-size "${SCHEME2_CHUNK_SIZE:-4}" \
  --warmup "${WARMUP:-30}" \
  --repeat "${REPEAT:-100}"
