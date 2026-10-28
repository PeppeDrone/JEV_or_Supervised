#!/usr/bin/env bash
# Derived v2 outputs once 27B inference is verified complete: analysis, figures, tests, artifact QA.
# The manuscript prose is regenerated afterwards from these outputs (not automated: interpretation follows results).
set -u
cd "$(dirname "$0")/.."
LOG=reports/scientific_revision_2026-10-26_v2/logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
step() { echo "$(date -Is) START $1" >> "$LOG/post_pipeline.log"; shift; "$@"; local rc=$?; echo "$(date -Is) END exit=$rc" >> "$LOG/post_pipeline.log"; return $rc; }
step analysis bash -c '.venv/bin/python scripts/run_scientific_revision.py --refresh > '"$LOG"'/analysis_27b.log 2>&1' || exit 1
step figures bash -c '.venv/bin/python scripts/plot_scientific_revision.py > '"$LOG"'/figures.log 2>&1'
step tests bash -c '.venv/bin/python -m pytest -q > '"$LOG"'/tests_post27b.log 2>&1'
step validation bash -c '.venv/bin/python scripts/validate_scientific_revision.py > '"$LOG"'/validation_post27b.log 2>&1'
echo "$(date -Is) POST PIPELINE FINISHED" >> "$LOG/post_pipeline.log"
