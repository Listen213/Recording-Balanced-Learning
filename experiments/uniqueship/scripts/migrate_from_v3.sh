#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-/root/autodl-tmp/RBL_UNIQUESHIP_8CLASS_EXTERNAL_V3}"
if [ ! -d "$SRC" ]; then echo "Old V3 project not found: $SRC"; exit 2; fi
mkdir -p cache data/protocols outputs logs
[ -f "$SRC/cache/uniqueship_8class_beats_5s.npz" ] && cp -av "$SRC/cache/uniqueship_8class_beats_5s.npz" cache/
for f in subset_selection.csv selected_vessel_clips.csv; do [ -f "$SRC/data/$f" ] && cp -av "$SRC/data/$f" data/; done
[ -f "$SRC/data/protocols/fold_1_8class_subset.csv" ] && cp -av "$SRC/data/protocols/fold_1_8class_subset.csv" data/protocols/
for f in uniqueship_subset_audit.json subset_stats.csv cache_summary.json available_sources_fold1.csv; do [ -f "$SRC/outputs/$f" ] && cp -av "$SRC/outputs/$f" outputs/; done
for d in subset_fold1_primary subset_fold1_classbal; do [ -d "$SRC/outputs/$d" ] && cp -a "$SRC/outputs/$d" outputs/; done
[ -d "$SRC/logs" ] && cp -a "$SRC/logs/." logs/ || true
echo "Migration completed. You may now keep only this clean project after checking the copied files."
