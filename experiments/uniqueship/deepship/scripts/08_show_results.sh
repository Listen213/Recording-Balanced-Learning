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

echo "===== COMPLETE ====="
cat runs/deepship_final/FINAL_COMPLETE.json
echo
echo "===== SUMMARY ====="
cat runs/deepship_final/summary.json
echo
echo "===== HIERARCHICAL BOOTSTRAP ====="
cat runs/deepship_final/hierarchical_paired_bootstrap.json
