#!/bin/bash
# Overnight queue: finish original UCI GPU job, then run WDBC / Thyroid / Arrhythmia
# (classical → zeroshot), rebuild figures.
# Safe to re-run: skips stages that already have prediction artefacts.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/uci_extension_overnight.log
STATUS=outputs/logs/gpu_stages_status.txt
NEW_STATUS=outputs/logs/uci_extension_status.txt
: > "$NEW_STATUS"

log() { echo "$(date -Is) $*" | tee -a "$LOG" "$NEW_STATUS"; }

have() {
  local ds=$1 oc=$2 m=$3
  [[ -f "outputs/$ds/$oc/predictions/$m.csv" && -f "outputs/$ds/$oc/predictions/$m.meta.json" ]]
}

run_stage() {
  local ds=$1 oc=$2 stage=$3
  local outlog="outputs/logs/${ds}_${oc}_${stage}.log"
  log "START $ds/$oc $stage"
  if "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$ds" --outcome "$oc" >"$outlog" 2>&1; then
    log "OK $ds/$oc $stage"
    return 0
  fi
  local ec=$?
  log "FAIL $ds/$oc $stage exit=$ec"
  tail -40 "$outlog" | tee -a "$LOG" || true
  return "$ec"
}

ORIGINAL_GPU=(
  "cleveland_heart heart_disease"
  "uci_ckd ckd_status"
  "uci_dermatology dermatological_diagnosis"
  "uci_hcv hcv_category"
  "uci_mammographic_mass malignancy"
)
NEW_PAIRS=(
  "uci_wdbc diagnosis"
  "uci_thyroid hypothyroid_status"
  "uci_arrhythmia arrhythmia_class"
)
ALL_PAIRS=("${ORIGINAL_GPU[@]}" "${NEW_PAIRS[@]}")

log "=== classical for new datasets (CPU, immediate) ==="
for pair in "${NEW_PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then
    log "SKIP $1/$2 classical"
  else
    run_stage "$1" "$2" classical || log "classical failed for $1/$2 (continuing)"
  fi
done

log "=== wait for original GPU pipeline (ALL_DONE or mammo Probe) ==="
while true; do
  if grep -q 'ALL_DONE' "$STATUS" 2>/dev/null; then
    log "original GPU ALL_DONE"
    break
  fi
  if have uci_mammographic_mass malignancy Probe; then
    log "mammo Probe present; proceeding"
    break
  fi
  # still running?
  if ! pgrep -f 'jevtest.cli (zeroshot|embed) --dataset' >/dev/null; then
    # idle: if mammo ZS missing, original may have died early
    if have uci_hcv hcv_category Probe && ! have uci_mammographic_mass malignancy ZS_raw; then
      log "HCV Probe done but mammo not started; launching mammo zeroshot+embed"
      run_stage uci_mammographic_mass malignancy zeroshot || true
      run_stage uci_mammographic_mass malignancy embed || true
      echo "ALL_DONE $(date -Is)" >> "$STATUS"
      break
    fi
    if have uci_mammographic_mass malignancy Probe; then
      echo "ALL_DONE $(date -Is)" >> "$STATUS"
      break
    fi
  fi
  sleep 120
done

log "=== GPU zeroshot for extension datasets ==="
for pair in "${NEW_PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw && have "$1" "$2" ZS_cal; then
    log "SKIP $1/$2 zeroshot"
  else
    run_stage "$1" "$2" zeroshot || true
  fi
done

log "=== figures + summary + complexity ==="
for pair in "${ALL_PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then
    "$VENV/bin/python" scripts/plot_predictions.py --dataset "$1" --outcome "$2" --no-combined \
      >>outputs/logs/uci_extension_plots.log 2>&1 || true
  fi
done
"$VENV/bin/python" scripts/plot_predictions.py --combined-only >>outputs/logs/uci_extension_plots.log 2>&1 || true
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/uci_extension_plots.log 2>&1 || true
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/uci_extension_plots.log 2>&1 || true

echo "EXTENSION_ALL_DONE $(date -Is)" | tee -a "$NEW_STATUS"
log "=== finished ==="
