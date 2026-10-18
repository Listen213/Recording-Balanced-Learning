#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -e runs/final_fast/FINAL_COMPLETE.json ]] || { echo "Final run is not complete."; exit 2; }
python make_paper_tables.py --run-dir runs/final_fast --out-dir paper_outputs
