#!/usr/bin/env bash
# 用法：
#   cd /home/starrys/triton_learning
#   bash scripts/run_step2_triton_matmul.sh
#
# 说明：
#   运行 Step 2 的 Triton 主干矩阵乘，并与 PyTorch/cuBLAS 对比。
#   如需修改形状，可以在命令前设置环境变量：
#   M=64 H=4096 N=28672 DTYPE=fp16 bash scripts/run_step2_triton_matmul.sh

set -euo pipefail

cd /home/starrys/triton_learning
source /home/starrys/venv/3DRF/bin/activate

echo "当前 Python："
which python
python -c "import sys, torch; print(sys.executable); print('torch.cuda.is_available() =', torch.cuda.is_available())"

export PYTHONPATH="/home/starrys/triton_learning/src:${PYTHONPATH:-}"

python -m triton_learning.bench_step2_triton_matmul \
  --m "${M:-64}" \
  --h "${H:-4096}" \
  --n "${N:-28672}" \
  --dtype "${DTYPE:-fp16}" \
  --warmup "${WARMUP:-30}" \
  --repeat "${REPEAT:-100}"

