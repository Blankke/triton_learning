#!/usr/bin/env bash
# 用法：
#   source scripts/activate_local_venv.sh
#
# 说明：
#   确保仓库根目录下的 `.venv` 存在，并安装运行 benchmark 所需依赖。
#   默认使用 `python3` 创建虚拟环境；如需覆盖，可在命令前设置 `PYTHON_BIN=/path/to/python`。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

_activate_local_venv_fail() {
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 1
  fi
  exit 1
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 Python 解释器：$PYTHON_BIN"
  _activate_local_venv_fail
fi

if [ ! -x "$ROOT_DIR/.venv/bin/python" ]; then
  echo "创建仓库本地虚拟环境：$ROOT_DIR/.venv"
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

if ! python -c "import numpy, torch, triton, tqdm" >/dev/null 2>&1; then
  echo "安装 benchmark 依赖到本地虚拟环境"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$ROOT_DIR/requirements-cu128.txt"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
