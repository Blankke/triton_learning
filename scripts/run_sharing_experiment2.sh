#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_sharing_experiment2.sh
#
# 说明：
#   运行 sharing 实验2：四段 pid range 交错做 op1 / op3 / op1 / op3。
#   默认使用构造 workload `r=2048`。
#   结果默认写入 `output/sharing/benchmarks/experiment2_interleaved.csv`。
#   如需覆盖默认参数，可在命令前设置环境变量：
#   M=64 H=4096 N=28672 R=2048 DTYPE=fp16 bash scripts/run_sharing_experiment2.sh
#   如需显式传 `SHARING_NUM_WORKERS`，它必须等于 `num_tiles_1 + num_tiles_3`。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"

python -m sharing.bench_experiment2_interleaved \
  --m "${M:-64}" \
  --h "${H:-4096}" \
  --n "${N:-28672}" \
  --r "${R:-2048}" \
  --dtype "${DTYPE:-fp16}" \
  --num-workers "${SHARING_NUM_WORKERS:-0}" \
  --warmup "${WARMUP:-30}" \
  --repeat "${REPEAT:-100}"
