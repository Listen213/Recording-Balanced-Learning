#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python verify_deepship_lock.py --lock deepship_lock/DEEPSHIP_LOCK.json

if [[ -e runs/deepship_final/FINAL_COMPLETE.json ]]; then
  echo "DeepShip external validation already complete."
  exit 2
fi

mkdir -p logs
nohup python run_deepship_fast_suite.py \
  --lock deepship_lock/DEEPSHIP_LOCK.json \
  --out-dir runs/deepship_final \
  --resume \
  > logs/deepship_final_resume.log 2>&1 &

echo $! > logs/deepship_final_resume.pid
echo "Resumed PID $(cat logs/deepship_final_resume.pid)"
echo "tail -f logs/deepship_final_resume.log"
