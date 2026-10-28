#!/bin/bash
# Nested XGBoost on the whole table (patients_full.csv) for one dataset x outcome.
# Usage: scripts/run_full_xgboost.sh DATASET OUTCOME
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
mkdir -p outputs/logs
LOG=outputs/logs/full_xgboost_$1_$2.log
echo "$(date -Is) START $1/$2" >> outputs/logs/full_xgboost_status.txt
if "${VENV:-.venv}/bin/python" -m jevtest.cli classical --full --dataset "$1" --outcome "$2" >"$LOG" 2>&1; then
  echo "$(date -Is) OK $1/$2" >> outputs/logs/full_xgboost_status.txt
else
  echo "$(date -Is) FAIL $1/$2" >> outputs/logs/full_xgboost_status.txt; tail -20 "$LOG" >> outputs/logs/full_xgboost_status.txt; exit 1
fi
