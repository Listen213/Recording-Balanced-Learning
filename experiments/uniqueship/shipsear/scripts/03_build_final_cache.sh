#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p cache/beats_final

if [[ -e cache/beats_final/support_pool.pt || -e cache/beats_final/test.pt ]]; then
  echo "Refusing to overwrite existing final cache."
  exit 2
fi

python build_cache.py \
  --manifest manifests_final/support_pool.json \
  --out cache/beats_final/support_pool.pt \
  --beats-source-dir third_party/beats \
  --beats-ckpt weights/BEATs_iter3_plus_AS2M.pt \
  --batch-size 24

python build_cache.py \
  --manifest manifests_final/test.json \
  --out cache/beats_final/test.pt \
  --beats-source-dir third_party/beats \
  --beats-ckpt weights/BEATs_iter3_plus_AS2M.pt \
  --batch-size 24
