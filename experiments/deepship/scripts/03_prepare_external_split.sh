#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

if [[ -e manifests_deepship_50/support_pool.json || -e manifests_deepship_50/test.json ]]; then
  echo "Refusing to overwrite existing DeepShip V3 50/class split."
  exit 2
fi

python prepare_deepship_50class.py \
  --root /root/autodl-tmp/DeepShip \
  --out-dir manifests_deepship_50 \
  --per-class 50 \
  --support-per-class 30 \
  --test-per-class 20 \
  --seed 20261015 \
  --target-sr 16000 \
  --segment-sec 5 \
  --hop-sec 5 \
  --diagnostic-threshold 0.997 \
  --fingerprint-sr 8000 \
  --fingerprint-window-sec 20 \
  --fingerprint-views 3
