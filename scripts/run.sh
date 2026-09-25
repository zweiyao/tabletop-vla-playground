#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TABLETOP_GPU="${TABLETOP_GPU:-0}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m tabletop.app "$@"
