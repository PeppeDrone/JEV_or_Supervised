#!/bin/bash
# After the main zeroshot queue frees the GPU, run System 1 on the three
# non-UCI datasets and rebuild combined figures / complexity.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
LOG=outputs/logs/nonuci_three.log
STATUS=outputs/logs/nonuci_three_status.txt
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
  tail -30 "outputs/logs/${ds}_${oc}_${stage}.log" | tee -a "$STATUS"
  return 1
}

PAIRS=(
  "ppd_bangladesh ppd_high"
  "hip_fracture_elos extended_los"
  "berlin_appendicitis appendicitis"
)

log "waiting for GPU zeroshot to finish (do not steal the running queue)"
while true; do
  if pgrep -f 'jevtest.cli zeroshot' >/dev/null; then
    sleep 20
    continue
  fi
  if pgrep -f 'scripts/queue_zs_then_embed.sh' >/dev/null; then
    if grep -q 'phase 1 done' outputs/logs/zs_then_embed_status.txt 2>/dev/null; then
      break
    fi
    sleep 10
    continue
  fi
  break
done
log "GPU free"

OVERALL=0
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if have "$1" "$2" ZS_raw; then
    log "SKIP $1/$2 zeroshot"
  else
    run_stage "$1" "$2" zeroshot || OVERALL=1
  fi
done

log "rebuilding combined figures, complexity, summary"
"$VENV/bin/python" scripts/plot_predictions.py --combined-only >>outputs/logs/nonuci_three_plots.log 2>&1 || OVERALL=1
"$VENV/bin/python" scripts/analyse_uci_complexity_gaps.py >>outputs/logs/nonuci_three_plots.log 2>&1 || OVERALL=1
"$VENV/bin/python" scripts/summarise_uci_core.py >>outputs/logs/nonuci_three_plots.log 2>&1 || OVERALL=1

if [ "$OVERALL" -eq 0 ]; then
  log "NONUCI_THREE_DONE"
else
  log "NONUCI_THREE_FAILED"
  exit 1
fi
