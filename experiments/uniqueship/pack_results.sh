#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -f UniqueShip_8class_FINAL_results.zip
zip -qr UniqueShip_8class_FINAL_results.zip outputs data/subset_selection.csv data/protocols/fold_1_8class_subset.csv configs/config.yaml logs 2>/dev/null || zip -qr UniqueShip_8class_FINAL_results.zip outputs configs/config.yaml logs
echo "WROTE $(pwd)/UniqueShip_8class_FINAL_results.zip"
