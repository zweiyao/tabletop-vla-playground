#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${TABLETOP_GPU:-0}"
export PYTHONPATH="$PWD/src:$PWD/../simulation/repos/RLinf"
export OPENPI_DATA_HOME="$PWD/../simulation/.cache/openpi"
export TORCH_COMPILE_DISABLE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
ready=0
for attempt in {1..180}; do
    if .venv/bin/python -c 'import json; d=json.load(open("data/tabletop-four-v1/manifest.json")); assert sum(e["split"]=="train" for e in d["episodes"])==320; assert sum(e["split"]=="val" for e in d["episodes"])==48' 2>/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
test "$ready" = 1
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,memory.used,memory.free --format=csv
.venv/bin/python -c 'import shutil; assert shutil.disk_usage(".").free > 8.5*1024**3, "Insufficient disk reserve"'
exec ../simulation/.venv/bin/python scripts/train_lora.py "$@"
