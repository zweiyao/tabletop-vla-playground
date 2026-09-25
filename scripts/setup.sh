#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then echo 'Run installation on the remote Linux GPU host.'; exit 1; fi
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
