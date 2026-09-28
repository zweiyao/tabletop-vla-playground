#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ready=0
for attempt in {1..180}; do
    if .venv/bin/python -c 'import json; d=json.load(open("data/tabletop-four-v1/manifest.json")); assert len(d["episodes"])==448' 2>/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
test "$ready" = 1
.venv/bin/python scripts/audit_demonstrations.py
exec .venv/bin/python scripts/evaluate_lora.py "$@"
