#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PM="outputs/primary/subset_fold1_primary_100draws_metrics.csv"
PP="outputs/primary/subset_fold1_primary_100draws_predictions.csv"
CM="outputs/classbal/subset_fold1_classbal_100draws_metrics.csv"
CP="outputs/classbal/subset_fold1_classbal_100draws_predictions.csv"
# Accept migrated V3 folder names too.
[ -f "$PM" ] || PM="outputs/subset_fold1_primary/subset_fold1_primary_100draws_metrics.csv"
[ -f "$PP" ] || PP="outputs/subset_fold1_primary/subset_fold1_primary_100draws_predictions.csv"
[ -f "$CM" ] || CM="outputs/subset_fold1_classbal/subset_fold1_classbal_100draws_metrics.csv"
[ -f "$CP" ] || CP="outputs/subset_fold1_classbal/subset_fold1_classbal_100draws_predictions.csv"
for f in "$PM" "$PP" "$CM" "$CP"; do [ -f "$f" ] || { echo "Missing $f"; exit 3; }; done
python scripts/06_mechanism_summary.py --primary-metrics "$PM" --primary-predictions "$PP" --classbal-metrics "$CM" --classbal-predictions "$CP" --out-dir outputs/final_summary --hier-iters 5000 --draw-iters 10000 --batch-size 16
cat outputs/final_summary/method_summary.csv
cat outputs/final_summary/paired_draw_bootstrap.csv
cat outputs/final_summary/hierarchical_bootstrap_all_pairs.csv
