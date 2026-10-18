#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
if [[ -e deepship_lock/DEEPSHIP_LOCK.json ]]; then
  echo "DEEPSHIP_LOCK.json already exists. Refusing to overwrite."
  exit 2
fi
python freeze_deepship_protocol.py \
  --config DEEPSHIP_CONFIG.json \
  --manifest-dir manifests_deepship_50 \
  --cache-dir cache/beats_deepship_50 \
  --out deepship_lock/DEEPSHIP_LOCK.json
python verify_deepship_lock.py --lock deepship_lock/DEEPSHIP_LOCK.json
