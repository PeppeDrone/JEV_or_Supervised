#!/usr/bin/env bash
# Additional 27B inference for the 2026-10-26 v2 revision, run stage by stage across all 45 tasks so that
# an interrupted queue still leaves complete higher-priority experiments:
#   1 development calibration pool (label-blind)       -> calibration / label-budget / complementarity
#   2 regenerated original hold-out subset             -> readout-stage ablation and paired control
#   3-7 semantic renderings on the same subset          -> anonymous first, then glosses, setting, units, concise
#   8 marginal direction and two-feature probes
# Caps are fixed here before any result is seen; they depend on cost only, never on outcomes.
# Launch persistently (survives SSH logout):
#   systemd-run --user --unit=jev-rev27b --collect -p WorkingDirectory=$PWD scripts/queue_revision_27b.sh
set -u
cd "$(dirname "$0")/.."
DEV_POOL=${DEV_POOL:-200}
HOLD_CAP=${HOLD_CAP:-100}
BATCH=${BATCH:-8}
LOG=reports/scientific_revision_2026-10-26_v2/logs
mkdir -p "$LOG"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for stage in development original anonymous no_category_gloss no_setting no_units concise direction; do
  echo "$(date -Is) START $stage" >> "$LOG/queue_27b.log"
  .venv/bin/python scripts/run_revision_inference.py --backbone 27b --stages "$stage" \
      --dev-pool "$DEV_POOL" --holdout-cap "$HOLD_CAP" --batch "$BATCH" --sort-length --single-device --resume \
      >> "$LOG/inference_27b_$stage.log" 2>&1
  echo "$(date -Is) END $stage exit=$?" >> "$LOG/queue_27b.log"
done
