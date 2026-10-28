#!/usr/bin/env bash
# Supervises the v2 27B queue (unit jev-rev27b): restarts it when it stops before outputs are complete,
# kills it when no progress is logged for STALL_MIN minutes, and runs the post-inference pipeline once
# scripts/check_revision_inference.py verifies every planned output file. Gives up after MAX_RESTARTS.
set -u
cd "$(dirname "$0")/.."
LOG=reports/scientific_revision_2026-10-26_v2/logs
W="$LOG/watchdog.log"
STALL_MIN=${STALL_MIN:-60}
MAX_RESTARTS=${MAX_RESTARTS:-8}
restarts=0
say() { echo "$(date -Is) $*" >> "$W"; }
say "watchdog started (stall ${STALL_MIN} min, max restarts ${MAX_RESTARTS})"
while true; do
  if .venv/bin/python scripts/check_revision_inference.py > "$LOG/coverage_27b.txt" 2>&1; then
    if systemctl --user is-active -q jev-rev27b; then sleep 300; continue; fi
    say "inference verified complete; running post pipeline"
    scripts/post_inference_pipeline.sh; rc=$?
    say "post pipeline exit=$rc; watchdog done"
    exit $rc
  fi
  if systemctl --user is-active -q jev-rev27b; then
    newest=$(stat -c %Y "$LOG"/inference_27b_*.log 2>/dev/null | sort -n | tail -1)
    if [ -n "$newest" ] && [ $(( $(date +%s) - newest )) -gt $(( STALL_MIN * 60 )) ]; then
      say "no progress for >${STALL_MIN} min; stopping queue for restart"
      systemctl --user stop jev-rev27b
    fi
  else
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then say "GIVING UP after $restarts restarts"; exit 1; fi
    restarts=$((restarts + 1))
    say "queue not running and outputs incomplete; restart $restarts/$MAX_RESTARTS"
    say "$(tr '\n' ' ' < "$LOG/coverage_27b.txt")"
    systemd-run --user --unit=jev-rev27b --collect -p WorkingDirectory="$PWD" scripts/queue_revision_27b.sh >> "$W" 2>&1
  fi
  sleep 300
done
