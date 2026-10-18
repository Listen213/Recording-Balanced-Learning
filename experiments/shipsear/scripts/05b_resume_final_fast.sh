#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python verify_final_lock.py --lock final_lock/FINAL_LOCK.json
if [[ -e runs/final_fast/FINAL_COMPLETE.json ]]; then
  echo "Already complete. Refusing to rerun."
  exit 2
fi
mkdir -p logs
nohup python -u run_final_fast_suite.py \
  --lock final_lock/FINAL_LOCK.json \
  --out-dir runs/final_fast \
  --resume \
  > logs/final_fast_resume.log 2>&1 &
echo $! > logs/final_fast_resume.pid
echo "Resume PID $(cat logs/final_fast_resume.pid)"
echo "Completed draw/bootstrap checkpoints will be skipped automatically."
echo "tail -f logs/final_fast_resume.log"
