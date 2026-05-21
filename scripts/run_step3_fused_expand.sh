#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_step3_fused_expand.sh
#
# 说明：
#   运行 Step 3 的单 LoRA fused expand + main GEMM 验证。
#   结果默认写入 `output/triton_learning/benchmarks/step3_fused_expand.csv`。
#   如需修改形状，可以在命令前设置环境变量：
#   M=64 H=4096 N=28672 R=8 DTYPE=fp16 bash scripts/run_step3_fused_expand.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_local_venv.sh"

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"

python -m triton_learning.bench_step3_fused_expand \
  --m "${M:-64}" \
  --h "${H:-4096}" \
  --n "${N:-28672}" \
  --r "${R:-8}" \
  --dtype "${DTYPE:-fp16}" \
  --warmup "${WARMUP:-30}" \
  --repeat "${REPEAT:-100}"
