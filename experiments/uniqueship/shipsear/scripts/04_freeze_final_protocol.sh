#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e final_lock/FINAL_LOCK.json ]]; then
  echo "FINAL_LOCK.json already exists. Refusing to overwrite."
  exit 2
fi

python freeze_final_protocol.py \
  --config FINAL_CONFIG.json \
  --manifest-dir manifests_final \
  --cache-dir cache/beats_final \
  --out final_lock/FINAL_LOCK.json

python verify_final_lock.py --lock final_lock/FINAL_LOCK.json
