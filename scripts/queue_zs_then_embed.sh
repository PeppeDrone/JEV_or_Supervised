#!/bin/bash
# Run remaining M2 / System 1 jobs for every prepared dataset, then rebuild plots.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/zs_then_embed.log
STATUS=outputs/logs/zs_then_embed_status.txt
: > "$STATUS"
log() { echo "$(date -Is) $*" | tee -a "$LOG" "$STATUS"; }

have() { [[ -f "outputs/$1/$2/predictions/$3.csv" ]]; }

run_stage() {
  local ds=$1 oc=$2 stage=$3
  log "START $ds/$oc $stage"
  if "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$ds" --outcome "$oc" \
      >"outputs/logs/${ds}_${oc}_${stage}.log" 2>&1; then
    log "OK $ds/$oc $stage"
    return 0
  fi
  log "FAIL $ds/$oc $stage exit=$?"
  return 1
}

mapfile -t PAIRS < <("$VENV/bin/python" - <<'PY'
from jevtest.benchmark import ALL_PAIRS
for ds, oc in ALL_PAIRS:
    print(ds, oc)
PY
)

log "=== ${#PAIRS[@]} dataset/outcome pairs from jevtest.benchmark.ALL_PAIRS ==="

log "=== phase 0: classical (M2) ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then
    log "SKIP $1/$2 classical"
  else
    run_stage "$1" "$2" classical || true
  fi
done
log "=== phase 0 done ==="
"$VENV/bin/python" scripts/plot_predictions.py --combined-only >>outputs/logs/zs_then_embed_plots.log 2>&1 || true

log "=== phase 1: zeroshot (ZS_raw / System 1) ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw; then
    log "SKIP $1/$2 zeroshot"
  else
    run_stage "$1" "$2" zeroshot || true
  fi
done
log "=== phase 1 done ==="

log "=== figures / complexity / summary ==="
"$VENV/bin/python" scripts/plot_predictions.py --combined-only >>outputs/logs/zs_then_embed_plots.log 2>&1 || true
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/zs_then_embed_plots.log 2>&1 || true
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/zs_then_embed_plots.log 2>&1 || true
echo "ZS_THEN_EMBED_DONE $(date -Is)" | tee -a "$STATUS"
log "=== finished ==="
