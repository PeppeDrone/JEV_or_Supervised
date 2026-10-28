#!/bin/bash
# Bologna external validation with the coded admission working diagnosis, then manuscript rebuild.
# Inference only: the saved Careggi fold models are loaded and applied; nothing is trained.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=outputs/logs/pipeline_wdx_$STAMP.log
STATUS=outputs/logs/pipeline_status.txt
mkdir -p outputs/logs
ln -sfn "$(basename "$LOG")" outputs/logs/pipeline_latest.log
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
step() {
  echo "$(date '+%F %T') | $1" | tee "$STATUS"
  echo "================================================================"
}
trap 'echo "$(date "+%F %T") | FAILED at: $(cat "$STATUS")" | tee -a "$STATUS"' ERR

step "import Bologna (adds eminence_bologna_external_wdx)"
"$VENV/bin/python" scripts/import_eminence_bologna.py
step "Bologna validation, working diagnosis missing: saved Careggi fold models (no training)"
"$VENV/bin/python" scripts/external_validate_eminence.py eminence_bologna_external
step "Bologna validation WITH working diagnosis: saved Careggi fold models (no training)"
"$VENV/bin/python" scripts/external_validate_eminence.py eminence_bologna_external_wdx
step "manuscript: figures and tables"
"$VENV/bin/python" manuscript/make_manuscript.py
"$VENV/bin/python" manuscript/supplementary_reporting.py
step "manuscript: write and compile PDF"
"$VENV/bin/python" manuscript/write_manuscript.py
rm -f manuscript/manuscript_package.zip
(cd manuscript && zip -j manuscript_package.zip paper/manuscript.pdf figures/Figure*.pdf figures/Figure*.png \
  figures/figure_legends.md tables/*.csv)
step "DONE  (log: $LOG)"
