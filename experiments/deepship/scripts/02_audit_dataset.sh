#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

rm -rf audit
mkdir -p audit
python audit_deepship.py   --root /root/autodl-tmp/DeepShip   --out-dir audit
