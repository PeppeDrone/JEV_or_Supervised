#!/bin/bash
# Retrain the manuscript models on Careggi (saving every fold model), validate them on Bologna
# without refitting, and rebuild the manuscript. Binary outcome only.
#   tmux new -s jevpipe 'bash scripts/run_external_pipeline.sh'
# Follow it with: bash scripts/watch.sh   (or: tail -f outputs/logs/pipeline_latest.log)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
OUTCOME=epileptic_final_diagnosis
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=outputs/logs/pipeline_$STAMP.log
STATUS=outputs/logs/pipeline_status.txt
BACKUP=outputs/backup_before_$STAMP
mkdir -p outputs/logs
ln -sfn "pipeline_$STAMP.log" outputs/logs/pipeline_latest.log
exec > >(tee -a "$LOG") 2>&1

unset PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export HF_HOME=${JEVTEST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}

step() {
  echo "$(date '+%F %T') | $1" | tee "$STATUS"
  echo "================================================================"
}
trap 'echo "$(date "+%F %T") | FAILED at: $(cat "$STATUS")" | tee -a "$STATUS"' ERR

step "backup of current predictions -> $BACKUP"
for ds in diagnosis_ICUEEG diagnosis_ICUEEG_no_working_dx eminence_bologna_external; do
  [ -d "outputs/$ds/$OUTCOME" ] && mkdir -p "$BACKUP/$ds" && cp -r "outputs/$ds/$OUTCOME/." "$BACKUP/$ds/" \
    --no-preserve=all 2>/dev/null || true
done
rm -rf "$BACKUP"/*/cache   # caches are untouched; only predictions are compared
cp -r manuscript/tables "$BACKUP/manuscript_tables"
cp manuscript/paper/manuscript.pdf "$BACKUP/manuscript_before.pdf" 2>/dev/null || true

step "import Bologna with the corrected coding"
"$VENV/bin/python" scripts/import_eminence_bologna.py

for ds in diagnosis_ICUEEG diagnosis_ICUEEG_no_working_dx; do
  for stage in classical zeroshot; do
    step "Careggi training: $ds $stage (fold models saved to outputs/$ds/$OUTCOME/models)"
    "$VENV/bin/python" -m jevtest.cli "$stage" --dataset "$ds" --outcome "$OUTCOME"
  done
done
step "Careggi: Mean and Mode rebound to the regenerated members"
"$VENV/bin/python" -m jevtest.cli ensemble --dataset diagnosis_ICUEEG --outcome "$OUTCOME"

step "check: regenerated Careggi predictions equal the manuscript's previous ones"
"$VENV/bin/python" - "$BACKUP" "$OUTCOME" <<'PY'
import sys
from pathlib import Path
import pandas as pd
backup, outcome = Path(sys.argv[1]), sys.argv[2]
for ds in ("diagnosis_ICUEEG", "diagnosis_ICUEEG_no_working_dx"):
    for old in sorted((backup / ds / "predictions").glob("*.csv")):
        new = Path("outputs") / ds / outcome / "predictions" / old.name
        a, b = pd.read_csv(old), pd.read_csv(new)
        cols = [c for c in a if c.startswith("p__")]
        gap = (a[cols] - b[cols]).abs().to_numpy().max()
        print(f"{ds:32} {old.stem:9} max |difference| = {gap:.2e}  {'IDENTICAL' if gap < 1e-9 else 'CHANGED'}")
PY

step "Bologna external validation: saved Careggi fold models, no fitting (GPU: zero-shot)"
"$VENV/bin/python" scripts/external_validate_eminence.py

step "manuscript: figures and tables"
"$VENV/bin/python" manuscript/make_manuscript.py
"$VENV/bin/python" manuscript/supplementary_reporting.py
step "manuscript: write and compile PDF"
"$VENV/bin/python" manuscript/write_manuscript.py
rm -f manuscript/manuscript_package.zip
(cd manuscript && zip -j manuscript_package.zip paper/manuscript.pdf figures/Figure*.pdf figures/Figure*.png \
  figures/figure_legends.md tables/*.csv)

step "DONE  (log: $LOG; backup: $BACKUP)"
