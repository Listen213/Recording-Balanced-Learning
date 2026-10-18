#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

if [[ ! -e runs/deepship_final/FINAL_COMPLETE.json ]]; then
  echo "DeepShip external validation is not complete."
  exit 2
fi

python make_deepship_tables.py \
  --run-dir runs/deepship_final \
  --out-dir paper_outputs_deepship
