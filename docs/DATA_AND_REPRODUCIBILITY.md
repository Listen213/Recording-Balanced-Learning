# Data and reproducibility guide

## ShipsEar

See `experiments/shipsear/README_CN.md`, `FINAL_PROTOCOL_CN.md`, `FINAL_CONFIG.json`, and `scripts/`. The original package expects a frozen experiment on a separate server with a BEATs checkpoint and precomputed features. The provided configuration enumerates seeds 101–120 (20 per K), and an optional full-support reference. This is **not equivalent** to the paper's unified 100-draw primary comparison.

## DeepShip

See `experiments/deepship/README_CN.md`, `PROTOCOL_CN.md`, `DEEPSHIP_CONFIG.json`. The frozen subset selects 50 recording units per class and partitions 30 for support and 20 for test. The package uses 20 draws per K; high acoustic similarity is diagnostic rather than proof of duplicate identity.

## UniqueShip

See `experiments/uniqueship/README_CN.md`, `configs/config.yaml`. Runs use eight selected Fold-1 classes and K in {2, 3, 4}, 100 draws. Analysis may run on preserved output CSVs. For from-scratch training, edit `project.dataset_root` and `project.source_project` and acquire authorized upstream audio and official BEATs source/weights. The original `run_from_scratch.sh` uses these configured locations.

## Assets removed from the public-ready package

- Cached BEATs embeddings and intermediate chunks; regenerate them from appropriately licensed datasets.
- `vendor/beats_src/`: third-party source with upstream rights; obtain/install official BEATs with its license.
- `.pyc`, process PID files, old crash logs, local runtime logs and nested result ZIPs.

## Reproduction claim limitations

This repository was assembled by restructuring the three uploaded archives, without running GPU training. Python source files were syntax checked where possible; full dataset/feature/model checkpoints are absent. Correspondence to manuscript Tables 3–5 and Figures 3–6 requires additional source/result mapping and validation.
