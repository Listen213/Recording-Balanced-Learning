#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
mkdir -p cache/beats_deepship_50
if [[ -e cache/beats_deepship_50/support_pool.pt || -e cache/beats_deepship_50/test.pt ]]; then
  echo "Refusing to overwrite existing DeepShip 50/class cache."
  exit 2
fi
python build_deepship_cache.py \
  --manifest manifests_deepship_50/support_pool.json \
  --out cache/beats_deepship_50/support_pool.pt \
  --beats-source-dir third_party/beats \
  --beats-ckpt weights/BEATs_iter3_plus_AS2M.pt \
  --batch-size 24
python build_deepship_cache.py \
  --manifest manifests_deepship_50/test.json \
  --out cache/beats_deepship_50/test.pt \
  --beats-source-dir third_party/beats \
  --beats-ckpt weights/BEATs_iter3_plus_AS2M.pt \
  --batch-size 24
