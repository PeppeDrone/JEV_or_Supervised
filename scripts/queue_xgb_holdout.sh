#!/bin/bash
# XGB_optuna hold-out arm on every task in jevtest.benchmark.ALL_PAIRS.
# Usage: scripts/queue_xgb_holdout.sh [primary|sensitivity|full] [WORKERS]
#   primary      one stratified 20% hold-out per task with the study seed (default)
#   sensitivity  five extra split seeds with the reduced budget (holdout.sensitivity_* in study.yaml)
#   full         XGB_optuna_full on the ten tables with patients_full.csv (needs primary + readout_to_full.json)
# WORKERS tasks run in parallel; each uses holdout.threads XGBoost threads (4 x 5 = 20 cores).
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
MODE="${1:-primary}"
WORKERS="${2:-4}"
case "$MODE" in
  primary) STAGE=xgb_holdout; DONE=holdout/metrics.json ;;
  sensitivity) STAGE=xgb_holdout_sensitivity; DONE= ;;
  full) STAGE=xgb_holdout_full; DONE=holdout/full/metrics.json ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
PY="${VENV:-.venv}/bin/python"
mkdir -p outputs/logs
STATUS="outputs/logs/xgb_holdout_${MODE}_status.txt"
echo "$(date -Is) QUEUE START mode=$MODE workers=$WORKERS" >> "$STATUS"

run_one() {
  local dataset="$1" outcome="$2"
  local log="outputs/logs/xgb_holdout_${MODE}_${dataset}_${outcome}.log"
  if [ -n "$DONE" ] && [ -f "outputs/$dataset/$outcome/$DONE" ]; then
    echo "$(date -Is) SKIP $dataset/$outcome (done)" >> "$STATUS"; return 0
  fi
  echo "$(date -Is) START $dataset/$outcome" >> "$STATUS"
  if "$PY" -m jevtest.cli "$STAGE" --dataset "$dataset" --outcome "$outcome" >"$log" 2>&1; then
    echo "$(date -Is) OK $dataset/$outcome" >> "$STATUS"
  else
    echo "$(date -Is) FAIL $dataset/$outcome" >> "$STATUS"; tail -5 "$log" >> "$STATUS"
  fi
}
export -f run_one
export PY STAGE DONE MODE STATUS

MODE="$MODE" "$PY" - <<'EOF' | xargs -P "$WORKERS" -n 2 bash -c 'run_one "$0" "$1"'
import os, sys
from pathlib import Path
sys.path.insert(0, "src")
from jevtest.benchmark import ALL_PAIRS
for dataset, outcome in ALL_PAIRS:
    if os.environ["MODE"] == "full" and not (Path("datasets") / dataset / "patients_full.csv").exists():
        continue
    print(dataset, outcome)
EOF
echo "$(date -Is) QUEUE END mode=$MODE" >> "$STATUS"
