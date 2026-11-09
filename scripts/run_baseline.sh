#!/usr/bin/env bash
# Reproducible pipeline. Existing experiments are never overwritten.
# SMOKE=1 bash scripts/run_baseline.sh
# bash scripts/run_baseline.sh --experiment baseline_reproduction
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-.venv/bin/python}
exec "$PY" -m connectx_rl.pipeline "$@"
