#!/bin/bash
# All stages for one dataset x outcome: bash scripts/run.sh DATASET OUTCOME [--limit N]
# Uses this project's .venv by default; VENV can select another installed environment.
# Safe to rerun: GPU units are cached and skipped. Run inside tmux, or detached with setsid nohup.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/run.sh DATASET OUTCOME [--limit N]" >&2
  exit 2
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "Python environment unavailable at $VENV; create .venv as described in README.md or set VENV." >&2
  exit 2
fi
# Pod presets break downloads (hf_transfer, xet); weights go to the pod's local disk.
unset PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p outputs/logs
for stage in check context classical zeroshot icl ensemble report; do
  echo "$(date -u +%H:%M) $stage ..."
  if ! "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$1" --outcome "$2" "${@:3}" > "outputs/logs/$1_$2_$stage.log" 2>&1; then
    echo "$(date -u +%H:%M) $stage FAILED:"; tail -20 "outputs/logs/$1_$2_$stage.log"; exit 1
  fi
done
echo "$(date -u +%H:%M) done: outputs/$1/$2/REPORT.md"
