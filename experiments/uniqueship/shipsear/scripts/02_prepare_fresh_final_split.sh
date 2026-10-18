#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e manifests_final/support_pool.json || -e manifests_final/test.json ]]; then
  echo "Refusing to overwrite an existing final split."
  echo "If this is only a failed pre-lock attempt, move manifests_final away manually and document why."
  exit 2
fi

python prepare_final_split.py \
  --audio-root /root/autodl-tmp/ShipsEar \
  --csv /root/autodl-tmp/ShipsEar/shipsEarMeta_90.csv \
  --out-dir manifests_final \
  --seed 20261014 \
  --test-ratio 0.30 \
  --near-dup-threshold 0.997 \
  --min-support-components-per-class 6 \
  --target-sr 16000 \
  --segment-sec 5 \
  --hop-sec 5 \
  --fingerprint-sr 8000 \
  --fingerprint-window-sec 20 \
  --fingerprint-windows 3
