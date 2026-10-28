#!/bin/bash
# Zero-shot readout on every task for the Qwen3.5 size ladder, one backbone after another on the local GPU.
# Default order: 0.8B, 2B, 4B, 9B on the local DGX Spark. Larger backbones (27B dense, 122B-A10B, 397B-A17B) run on a
# pod with scripts/queue_pod.sh zeroshot STUDY holdout. Each study writes to its own outputs dir; reruns resume from caches.
# SUBSET=holdout (default) scores only the XGB_optuna hold-out patients, the same 20% for every backbone;
# SUBSET=all scores every patient (only the legacy Qwen3-0.6B pass was run this way).
#   setsid nohup scripts/queue_zeroshot_ladder.sh > outputs/logs/zeroshot_ladder.out 2>&1 &
set -eu
cd "$(dirname "$0")/.."
LADDER="${LADDER:-configs/study_qwen3.5_0.8b.yaml configs/study_qwen3.5_2b.yaml configs/study_qwen3.5_4b.yaml configs/study_qwen3.5_9b.yaml}"
SUBSET="${SUBSET:-holdout}"
STUDIES="$LADDER" scripts/queue_pod.sh pin
for study in $LADDER; do
  echo "$(date -Is) LADDER zeroshot $study subset=$SUBSET"
  scripts/queue_pod.sh zeroshot "$study" "$SUBSET"
done
echo "$(date -Is) LADDER END"
