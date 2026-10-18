#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p logs outputs cache data/protocols
python scripts/00_setup_assets.py
python scripts/01_inspect_protocol.py --config configs/config.yaml
python scripts/02_prepare_protocol.py --config configs/config.yaml
[ -f cache/uniqueship_8class_beats_5s.npz ] || python scripts/03_build_beats_cache.py --config configs/config.yaml
python scripts/04_run_experiments.py --config configs/config.yaml --draws 100 --mode primary --out-dir outputs/primary
python scripts/04_run_experiments.py --config configs/config.yaml --draws 100 --mode classbal --out-dir outputs/classbal
bash run_analysis_only.sh
echo "DONE: UniqueShip final clean experiment."
