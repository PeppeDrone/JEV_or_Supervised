#!/bin/bash
# Classical then System 1 for the six public-extension tables (n=1000 each).
# Does not steal a live zeroshot GPU queue.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/public_six.log
STATUS=outputs/logs/public_six_status.txt
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
  tail -40 "outputs/logs/${ds}_${oc}_${stage}.log" | tee -a "$STATUS"
  return 1
}

PAIRS=(
  "physionet_2012 icu_death"
  "physionet_2019 sepsis"
  "nhanes_htn hypertension"
  "metabric pam50_subtype"
  "mexico_covid covid_death"
  "dryad_er_cbc er_death"
)

log "waiting for GPU zeroshot to finish (do not steal the running queue)"
while pgrep -f 'jevtest.cli zeroshot' >/dev/null; do
  sleep 20
done
log "GPU free"

OVERALL=0
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" M2; then
    log "SKIP $1/$2 classical"
  else
    run_stage "$1" "$2" check || OVERALL=1
    run_stage "$1" "$2" classical || OVERALL=1
  fi
done

for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw; then
    log "SKIP $1/$2 zeroshot"
  else
    run_stage "$1" "$2" zeroshot || OVERALL=1
  fi
done

log "rebuilding combined figures, complexity, summary"
"$VENV/bin/python" scripts/plot_predictions.py --combined-only >>outputs/logs/public_six_plots.log 2>&1 || OVERALL=1
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/public_six_plots.log 2>&1 || OVERALL=1
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/public_six_plots.log 2>&1 || OVERALL=1

if [ "$OVERALL" -eq 0 ]; then
  log "PUBLIC_SIX_DONE"
else
  log "PUBLIC_SIX_FAILED"
  exit 1
fi
