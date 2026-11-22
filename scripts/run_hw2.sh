#!/usr/bin/env bash
# ДЗ2: configs -> train -> evaluate -> report. Готовые артефакты не пересчитываются.
#   SMOKE=1 bash scripts/run_hw2.sh                 # уменьшенный прогон, ~3 минуты
#   bash scripts/run_hw2.sh                         # полный прогон (15 обучений, ~1.5 ч на 15 ядрах)
#   bash scripts/run_hw2.sh --stage report          # только пересобрать отчёт
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-.venv/bin/python}
exec "$PY" -m connectx_rl.hw2 "$@"
