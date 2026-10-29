#!/usr/bin/env bash
# 在项目根目录启动网页推理，优先复用已有模型环境。
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"

if [[ -n "${GLDAS_PYTHON:-}" ]]; then
  python_bin="$GLDAS_PYTHON"
  if ! command -v -- "$python_bin" >/dev/null 2>&1; then
    echo "GLDAS_PYTHON 指向的 Python 不可执行：$python_bin" >&2
    exit 1
  fi
else
  python_bin=""
  for candidate in \
    "$project_dir/.web-runtime/python" \
    python python3; do
    if command -v -- "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import torch; from PIL import Image' \
       >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
  if [[ -z "$python_bin" ]]; then
    echo "未找到含 PyTorch 和 Pillow 的 Python，请设置 GLDAS_PYTHON。" >&2
    exit 1
  fi
fi

# 本机缓存的 NumPy 扩展为 CPython 3.10；其他版本使用自身环境依赖。
python_version="$("$python_bin" -c \
  'import sys; print("%d.%d" % sys.version_info[:2])')"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$python_version" == "3.10" ]] && \
   [[ -d "$project_dir/.web-runtime/deps/site" ]]; then
  export PYTHONPATH="$project_dir/.web-runtime/deps/site:\
$project_dir/.web-runtime/deps/detectron2:$PYTHONPATH"
fi

# 单图 CPU 推理使用适量线程，避免线程争用拖慢页面响应。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
printf '使用 Python：%s\n' "$python_bin"
exec "$python_bin" -m web_demo.server "$@"
