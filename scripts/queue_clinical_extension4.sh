#!/bin/bash
# Classical now; GPU zeroshot+embed+figures after arrhythmia ZS job finishes (or idle).
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/clinical_extension4.log
STATUS=outputs/logs/clinical_extension4_status.txt
: > "$STATUS"
log() { echo "$(date -Is) $*" | tee -a "$LOG" "$STATUS"; }

have() { [[ -f "outputs/$1/$2/predictions/$3.csv" ]]; }

run_stage() {
  local ds=$1 oc=$2 stage=$3
  log "START $ds/$oc $stage"
  if "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$ds" --outcome "$oc" \
      >"outputs/logs/${ds}_${oc}_${stage}.log" 2>&1; then
    log "OK $ds/$oc $stage"; return 0
  fi
  log "FAIL $ds/$oc $stage exit=$?"
  return 1
}

PAIRS=(
  "uci_heart_failure mortality"
  "uci_ilpd liver_disease"
  "uci_diabetes130 readmit_30d"
  "uci_cervical biopsy"
  "wids2021 diabetes_mellitus"
)

log "=== classical (CPU) ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then log "SKIP $1/$2 classical"
  else run_stage "$1" "$2" classical || true
  fi
done

log "=== wait for arrhythmia ZS full job / GPU idle ==="
while pgrep -f 'uci_arrhythmia_zeroshot_full|arrhythmia_class.*zeroshot|jevtest.cli zeroshot --dataset uci_arrhythmia' >/dev/null; do
  sleep 60
done
# also wait if some other jevtest GPU stage is mid-flight
while pgrep -f 'jevtest.cli (zeroshot|embed)' >/dev/null; do
  sleep 60
done
# Manual hold: figures / review before extension4 GPU (remove HOLD_EXTENSION4_GPU to release)
HOLD=outputs/logs/HOLD_EXTENSION4_GPU
if [[ -f "$HOLD" ]]; then
  log "=== held by $HOLD — waiting for release ==="
  while [[ -f "$HOLD" ]]; do sleep 30; done
  log "=== hold released ==="
fi

log "=== GPU zeroshot + embed for extension4 ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw; then log "SKIP $1/$2 zeroshot"
  else
    # diabetes130/cervical/hf/ilpd: default anyjev; arrhythmia-style OOM unlikely at these sizes
    run_stage "$1" "$2" zeroshot || true
  fi
  "$VENV/bin/python" scripts/refresh_outcome_figures.py --dataset "$1" --outcome "$2" \
    >>outputs/logs/clinical_extension4_plots.log 2>&1 || true
done

log "=== final summary / complexity ==="
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/clinical_extension4_plots.log 2>&1 || true
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/clinical_extension4_plots.log 2>&1 || true
echo "CLINICAL_EXT4_DONE $(date -Is)" | tee -a "$STATUS"
log "=== finished ==="
