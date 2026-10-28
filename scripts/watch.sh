#!/bin/bash
# Live view of the pipeline: current step, model progress, GPU, memory and the latest log lines.
# Usage: bash scripts/watch.sh [seconds]   (Ctrl-C to quit; the pipeline keeps running)
cd "$(dirname "$0")/.."
INTERVAL=${1:-5}
while true; do
  clear
  echo "== JEV pipeline watchdog  $(date '+%F %T')  (refresh ${INTERVAL}s, Ctrl-C quits) =="
  echo
  echo "-- current step"
  cat outputs/logs/pipeline_status.txt 2>/dev/null || echo "no pipeline started"
  if tmux has-session -t jevpipe 2>/dev/null; then echo "tmux session jevpipe: RUNNING"; else echo "tmux session jevpipe: not running"; fi
  echo
  echo "-- per-run progress"
  for f in outputs/*/epileptic_final_diagnosis/progress.json; do
    printf "%-34s %s\n" "$(echo "$f" | cut -d/ -f2)" "$(tr -d '\n ' < "$f")"
  done
  echo
  echo "-- GPU"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi unavailable"
  echo "-- memory / load"
  free -h | awk 'NR<=2'
  uptime
  echo
  echo "-- last log lines (outputs/logs/pipeline_latest.log)"
  tail -n 20 outputs/logs/pipeline_latest.log 2>/dev/null | cut -c1-200
  sleep "$INTERVAL"
done
