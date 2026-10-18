#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python verify_final_lock.py --lock final_lock/FINAL_LOCK.json
if [[ -e runs/final_fast/FINAL_COMPLETE.json ]]; then
  echo "Final fast run is already complete. Refusing to rerun."
  exit 2
fi
if [[ -e runs/final_fast/RUNNING.json ]]; then
  echo "A RUNNING marker exists. If the old process is dead, use scripts/05b_resume_final_fast.sh."
  exit 2
fi
mkdir -p logs
nohup python -u run_final_fast_suite.py \
  --lock final_lock/FINAL_LOCK.json \
  --out-dir runs/final_fast \
  > logs/final_fast.log 2>&1 &
echo $! > logs/final_fast.pid
echo "Started PID $(cat logs/final_fast.pid)"
echo "Live progress: tail -f logs/final_fast.log"
echo "Compact status: watch -n 3 'bash scripts/06_status.sh'"
