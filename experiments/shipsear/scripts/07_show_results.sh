#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -e runs/final_fast/FINAL_COMPLETE.json ]]; then
  echo "Final run is not complete. Use: bash scripts/06_status.sh"
  exit 2
fi
echo "===== FINAL COMPLETE ====="
cat runs/final_fast/FINAL_COMPLETE.json
echo
echo "===== SUMMARY ====="
cat runs/final_fast/summary.json
echo
echo "===== HIERARCHICAL PAIRED BOOTSTRAP ====="
cat runs/final_fast/hierarchical_paired_bootstrap.json
