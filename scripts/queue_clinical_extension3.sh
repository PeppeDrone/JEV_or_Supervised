#!/bin/bash
# Classical now; GPU after extension2 / extension4 / in-flight zeroshot|embed finish.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/clinical_extension3.log
STATUS=outputs/logs/clinical_extension3_status.txt
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
  "uci_maternal_risk risk_level"
  "uci_thyroid_recur recurred"
  "uci_cirrhosis cirrhosis_died"
  "uci_support2 hosp_death"
  "uci_aids_cid aids_event"
  "uci_glioma gbm"
  "uci_retinopathy retinopathy"
  "uci_cardiotocography nsp"
  "uci_thoracic died_1y"
  "uci_wbc_original malignant"
  "uci_vertebral pathology"
  "uci_autism asd"
  "uci_breast_recurrence recurrence"
  "uci_cdc_diabetes cdc_diabetes"
)

log "=== classical (CPU) ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then log "SKIP $1/$2 classical"
  else run_stage "$1" "$2" classical || true
  fi
done

log "=== wait for prior GPU queues / in-flight zeroshot|embed ==="
while pgrep -f 'queue_clinical_extension[24]|jevtest.cli (zeroshot|embed)' >/dev/null; do
  sleep 90
done

log "=== GPU zeroshot + embed for extension3 ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw; then log "SKIP $1/$2 zeroshot"
  else run_stage "$1" "$2" zeroshot || true
  fi
  "$VENV/bin/python" scripts/refresh_outcome_figures.py --dataset "$1" --outcome "$2" \
    >>outputs/logs/clinical_extension3_plots.log 2>&1 || true
done

log "=== final summary / complexity ==="
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/clinical_extension3_plots.log 2>&1 || true
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/clinical_extension3_plots.log 2>&1 || true
echo "CLINICAL_EXT3_DONE $(date -Is)" | tee -a "$STATUS"
log "=== finished ==="
