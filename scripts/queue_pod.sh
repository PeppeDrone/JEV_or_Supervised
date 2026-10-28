#!/bin/bash
# Full rerun of the 45-task benchmark on a GPU pod with three backbones.
#
#   scripts/queue_pod.sh all                 everything below, in order
#   scripts/queue_pod.sh pin                 pin `revision: main` in the 32B/235B study files to the current HF commit
#   scripts/queue_pod.sh smoke               zeroshot + context + icl on uci_haberman --limit 40 for every backbone
#   scripts/queue_pod.sh classical           M2 (and M2 --full) on every task, once, into outputs/ (CPU)
#   scripts/queue_pod.sh zeroshot STUDY [holdout]   zeroshot only, on every task; 'holdout' scores just the
#                                            XGB_optuna hold-out patients (the comparable 20%, for large backbones)
#   scripts/queue_pod.sh gpu STUDY           zeroshot -> context -> icl on every task for one study file
#   scripts/queue_pod.sh holdout STUDY       XGB_optuna hold-out arm: tuned once (base study), re-scored for others
#   scripts/queue_pod.sh finish STUDY        share M2 into the study's outputs dir, ensemble, report
#   scripts/queue_pod.sh summarise STUDY     status, hold-out summary, complexity and codebook analyses, figures
#
# Studies: configs/study.yaml (Qwen3-4B, outputs/) is the base study whose XGB_optuna hold-out arm every other backbone
# reuses; default pod studies are the large Qwen3.5 sizes, configs/study_qwen3.5_27b.yaml (outputs_qwen3.5_27b/) and
# configs/study_qwen3.5_397b_a17b.yaml (outputs_qwen3.5_397b_a17b/). Override with STUDIES="a.yaml b.yaml".
# The small Qwen3.5 sizes run on the local DGX through scripts/queue_zeroshot_ladder.sh.
# Every stage is resumable: cached GPU units and finished predictions are skipped by jevtest itself.
# A quality-gate exit (code 2, e.g. the arrhythmia ICL prefix that cannot fit 8 x 10 rows) is logged as GATE
# and the queue continues; any other non-zero exit is logged as FAIL.
set -eu
cd "$(dirname "$0")/.."
unset PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
PY="${VENV:-.venv}/bin/python"
BASE_STUDY="configs/study.yaml"
STUDIES="${STUDIES:-configs/study.yaml configs/study_qwen3.5_27b.yaml configs/study_qwen3.5_397b_a17b.yaml}"
CPU_WORKERS="${CPU_WORKERS:-4}"
SMOKE_DATASET="${SMOKE_DATASET:-uci_haberman}"
SMOKE_OUTCOME="${SMOKE_OUTCOME:-died_5y}"
mkdir -p outputs/logs
STATUS="outputs/logs/pod_queue_status.txt"
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

outputs_of() {  # outputs dir declared by a study file
  JEVTEST_STUDY="$1" "$PY" -c 'import sys; sys.path.insert(0, "src"); from jevtest.config import outputs_root; print(outputs_root("."))'
}

pairs() {  # "dataset outcome" per line; with "full" only tasks that have patients_full.csv
  "$PY" - "$1" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from jevtest.benchmark import ALL_PAIRS
for dataset, outcome in ALL_PAIRS:
    if sys.argv[1] == "full" and not (Path("datasets") / dataset / "patients_full.csv").exists():
        continue
    print(dataset, outcome)
EOF
}

# run_stage STUDY STAGE DATASET OUTCOME [extra jevtest args]
run_stage() {
  local study="$1" stage="$2" dataset="$3" outcome="$4"; shift 4
  local tag; tag="$(basename "${study%.yaml}")"
  local logf="outputs/logs/pod_${tag}_${stage}_${dataset}_${outcome}.log"
  log "START $tag $stage $dataset/$outcome"
  set +e
  JEVTEST_STUDY="$study" "$PY" -m jevtest.cli "$stage" --dataset "$dataset" --outcome "$outcome" "$@" >"$logf" 2>&1
  local code=$?
  set -e
  case $code in
    0) log "OK    $tag $stage $dataset/$outcome" ;;
    2) log "GATE  $tag $stage $dataset/$outcome: $(grep -m1 'QUALITY GATE' "$logf" || tail -1 "$logf")" ;;
    *) log "FAIL  $tag $stage $dataset/$outcome (exit $code)"; tail -5 "$logf" | tee -a "$STATUS" ;;
  esac
  return 0
}
export -f run_stage log
export PY STATUS

phase_pin() {
  for study in $STUDIES; do
    [ "$study" = "$BASE_STUDY" ] && continue
    "$PY" - "$study" <<'EOF'
import re, sys
from pathlib import Path
from huggingface_hub import HfApi
path = Path(sys.argv[1]); text = path.read_text()
model_id = re.search(r"^\s*model_id:\s*(\S+)", text, re.M).group(1)
current = re.search(r"^(\s*revision:\s*)(\S+)", text, re.M)
if re.fullmatch(r"[0-9a-f]{40}", current.group(2)):
    print(f"{path}: already pinned to {current.group(2)}"); sys.exit(0)
sha = HfApi().model_info(model_id).sha
path.write_text(text[:current.start(2)] + sha + text[current.end(2):])
print(f"{path}: {model_id} pinned to {sha}")
EOF
  done
}

phase_smoke() {
  for study in $STUDIES; do
    for stage in zeroshot context icl; do
      run_stage "$study" "$stage" "$SMOKE_DATASET" "$SMOKE_OUTCOME" --limit 40
    done
  done
}

phase_classical() {  # once, into the base study's outputs dir
  pairs all | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$BASE_STUDY"'" classical "$0" "$1"'
  pairs full | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$BASE_STUDY"'" classical "$0" "$1" --full'
}

phase_gpu() {
  local study="$1"
  while read -r dataset outcome; do
    for stage in zeroshot context icl; do run_stage "$study" "$stage" "$dataset" "$outcome"; done
  done < <(pairs all)
}

phase_zeroshot() {
  local study="$1" subset="${2:-all}"
  while read -r dataset outcome; do
    run_stage "$study" zeroshot "$dataset" "$outcome" --subset "$subset"
  done < <(pairs all)
}

phase_holdout() {
  local study="$1"
  if [ "$study" = "$BASE_STUDY" ]; then
    pairs all | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$study"'" xgb_holdout "$0" "$1"'
    pairs all | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$study"'" xgb_holdout_sensitivity "$0" "$1"'
    pairs full | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$study"'" xgb_holdout_full "$0" "$1"'
  else
    export JEVTEST_HOLDOUT_SOURCE; JEVTEST_HOLDOUT_SOURCE="$(outputs_of "$BASE_STUDY")"
    pairs all | xargs -P "$CPU_WORKERS" -n 2 bash -c 'run_stage "'"$study"'" xgb_holdout_rescore "$0" "$1"'
  fi
}

phase_finish() {
  local study="$1" base_out this_out
  base_out="$(outputs_of "$BASE_STUDY")"; this_out="$(outputs_of "$study")"
  if [ "$this_out" != "$base_out" ]; then
    # M2 provenance is backbone-independent (config.Run._strip_noncausal_study), so the one classical fit is shared.
    while read -r dataset outcome; do
      for sub in "$outcome" "${outcome}__full"; do
        src="$base_out/$dataset/$sub/predictions"
        [ -f "$src/M2.csv" ] || continue
        mkdir -p "$this_out/$dataset/$sub/predictions"
        cp -p "$src/M2.csv" "$src/M2.meta.json" "$this_out/$dataset/$sub/predictions/"
      done
    done < <(pairs all)
    log "shared M2 predictions from $base_out into $this_out"
  fi
  while read -r dataset outcome; do
    for stage in ensemble report; do run_stage "$study" "$stage" "$dataset" "$outcome"; done
  done < <(pairs all)
}

phase_summarise() {
  local study="$1" tag; tag="$(basename "${study%.yaml}")"
  for script in status_benchmark_progress summarise_holdout analyse_uci_complexity_gaps analyse_codebook_quality plot_predictions; do
    log "SUMMARY $tag $script"
    JEVTEST_STUDY="$study" "$PY" "scripts/$script.py" >"outputs/logs/pod_${tag}_${script}.log" 2>&1 || log "FAIL  $tag $script"
  done
}

PHASE="${1:-all}"
case "$PHASE" in
  pin) phase_pin ;;
  smoke) phase_smoke ;;
  classical) phase_classical ;;
  zeroshot) phase_zeroshot "${2:?study file}" "${3:-all}" ;;
  gpu) phase_gpu "${2:?study file}" ;;
  holdout) phase_holdout "${2:?study file}" ;;
  finish) phase_finish "${2:?study file}" ;;
  summarise) phase_summarise "${2:?study file}" ;;
  all)
    log "QUEUE START studies=[$STUDIES]"
    phase_pin
    phase_smoke
    phase_classical &
    CLASSICAL_PID=$!
    for study in $STUDIES; do
      phase_gpu "$study"
      [ "$study" = "$BASE_STUDY" ] && wait "$CLASSICAL_PID"
      phase_holdout "$study"
      phase_finish "$study"
      phase_summarise "$study"
    done
    log "QUEUE END"
    ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
