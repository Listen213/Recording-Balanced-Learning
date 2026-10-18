#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python status_final_fast.py --run-dir runs/final_fast
