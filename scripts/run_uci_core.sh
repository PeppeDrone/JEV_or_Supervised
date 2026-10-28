#!/bin/bash
# Run the UCI core benchmark (M2, ZS_raw) on all datasets.
# Stages: check -> classical (CPU) -> zeroshot (GPU).
# Does NOT run context/icl/ensemble (those launch ICL_full / Mode / Mean).
# Safe to re-run: completed prediction artefacts with matching provenance are reused.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
if [ ! -x "$VENV/bin/python" ]; then
  echo "Python environment unavailable at $VENV; create .venv as in README.md or set VENV." >&2
  exit 2
fi
unset PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}

mapfile -t PAIRS < <("$VENV/bin/python" - <<'PY'
from jevtest.benchmark import ALL_PAIRS
for ds, oc in ALL_PAIRS:
    print(ds, oc)
PY
)

mkdir -p outputs/logs
SUMMARY=outputs/logs/uci_core_run_status.txt
: > "$SUMMARY"

have_cuda() {
  "$VENV/bin/python" - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

run_stage() {
  local dataset=$1 outcome=$2 stage=$3
  local log="outputs/logs/${dataset}_${outcome}_${stage}.log"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${dataset}/${outcome} ${stage}" | tee -a "$SUMMARY"
  if "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$dataset" --outcome "$outcome" >"$log" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) OK    ${dataset}/${outcome} ${stage}" | tee -a "$SUMMARY"
    return 0
  fi
  local ec=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL  ${dataset}/${outcome} ${stage} exit=${ec}" | tee -a "$SUMMARY"
  echo "---- last 30 lines of ${log} ----" | tee -a "$SUMMARY"
  tail -30 "$log" | tee -a "$SUMMARY"
  return "$ec"
}

GPU_OK=0
if have_cuda; then
  GPU_OK=1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) CUDA available" | tee -a "$SUMMARY"
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) CUDA UNAVAILABLE: will complete CPU stages only; GPU stages marked blocked" | tee -a "$SUMMARY"
fi

OVERALL=0
for pair in "${PAIRS[@]}"; do
  # shellcheck disable=SC2086
  set -- $pair
  dataset=$1 outcome=$2
  echo "======== ${dataset} / ${outcome} ========"

  run_stage "$dataset" "$outcome" check || { OVERALL=1; continue; }
  run_stage "$dataset" "$outcome" classical || OVERALL=1

  if [ "$GPU_OK" -eq 1 ]; then
    run_stage "$dataset" "$outcome" zeroshot || OVERALL=1
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) BLOCKED ${dataset}/${outcome} zeroshot (no CUDA)" | tee -a "$SUMMARY"
    OVERALL=1
  fi
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finished overall_exit=${OVERALL}" | tee -a "$SUMMARY"
exit "$OVERALL"
