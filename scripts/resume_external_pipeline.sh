#!/bin/bash
# Resume scripts/run_external_pipeline.sh from the Bologna validation step (Careggi models already saved).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
LOG=outputs/logs/$(readlink outputs/logs/pipeline_latest.log)
STATUS=outputs/logs/pipeline_status.txt
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
step() {
  echo "$(date '+%F %T') | $1" | tee "$STATUS"
  echo "================================================================"
}
trap 'echo "$(date "+%F %T") | FAILED at: $(cat "$STATUS")" | tee -a "$STATUS"' ERR

step "RESUMED: Bologna external validation (cached Bologna zero-shot and embeddings, saved Careggi fold models)"
"$VENV/bin/python" scripts/external_validate_eminence.py
step "manuscript: figures and tables"
"$VENV/bin/python" manuscript/make_manuscript.py
"$VENV/bin/python" manuscript/supplementary_reporting.py
step "manuscript: write and compile PDF"
"$VENV/bin/python" manuscript/write_manuscript.py
rm -f manuscript/manuscript_package.zip
(cd manuscript && zip -j manuscript_package.zip paper/manuscript.pdf figures/Figure*.pdf figures/Figure*.png \
  figures/figure_legends.md tables/*.csv)
step "DONE  (log: $LOG)"
