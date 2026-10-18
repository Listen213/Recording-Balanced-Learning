# Recording-Balanced Learning: three-dataset research archive

This repository packages **three independently developed experiment projects** supplied by the authors for the manuscript *Recording-balanced learning for few-recording underwater acoustic target recognition*.

**Important scope:** The code packages are **not a single unified end-to-end reproduction** of all numbers in the manuscript. The ShipsEar and DeepShip archives primarily describe **20-draw S0/S1/PB/PT/F0 experiments**, whereas the UniqueShip archive contains a **100-draw S0/ClassBal/S1 experiment**. The unified 100-draw ShipsEar/DeepShip main-study runs, and some robustness / exact-duplication analyses described in the manuscript, are **not established as present in these three uploads**. Do not claim that the present archive fully reproduces the manuscript tables before these missing streams are added and checked.

## Structure

| Folder | Dataset | Archived scope |
| --- | --- | --- |
| `experiments/shipsear/` | ShipsEar | Frozen recording/component partition, S0/S1/PB/PT/F0; 2/3/4 recordings per class, 20 draws in the supplied configuration, supporting output checkpoints |
| `experiments/deepship/` | DeepShip | Fixed 50-recordings/class (30 support, 20 test), S0/S1/PB/PT/F0, 20 draws, source and duplicate diagnostics |
| `experiments/uniqueship/` | UniqueShip | Identity-aware Fold 1, eight classes, 100 draws for S0/ClassBal/S1, final summary tables and prediction archives |

## Quick start

The three projects keep their original scripts, local relative paths and project-specific requirements. Run commands **inside each dataset directory**; do not copy the three `requirements.txt` files over one another.

```bash
# Optional: create a fresh environment for one project
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/uniqueship/requirements.txt
cd experiments/uniqueship
bash run_analysis_only.sh
```

`run_analysis_only.sh` uses the archived results. For full data and BEATs feature extraction, consult `README_CN.md` and `docs/DATA_AND_REPRODUCIBILITY.md`; datasets, official BEATs weights and pretrained checkpoints are **not included**. Some author-local `/root/autodl-tmp/...` settings must be changed to match your environment.

For ShipsEar and DeepShip, read their own `README_CN.md` and run the available smoke-test scripts before invoking an expensive training run. Some scripts require cached features and a separately installed BEATs implementation.

## Reproducibility boundaries and research integrity

- The supplied archives contain frozen split manifests and saved outputs, but no guarantee that all published paper tables can be recreated from them alone.
- **Do not use external test labels to select hyperparameters.** Keep the original frozen source partitions and support seeds.
- The supplied UniqueShip CSV summaries may not agree exactly with the manually plotted values of the latest manuscript. Resolve provenance using the original experiment log rather than changing either set of numbers merely to match.
- Saved prediction CSVs are retained as numeric research outputs; raw acoustic waveform data and BEATs feature caches have been excluded from this public-ready package.
- No pretrained BEATs weights are distributed. Download assets from original authorized publishers and follow their licenses.
- Files in `vendor/` were excluded, including third-party BEATs source. Obtain it from the official upstream repository and preserve its license. Consult the original project README.

## Data and third-party licenses

ShipsEar, DeepShip and UniqueShip have separate rights and access processes. Do not upload their original waveforms, restricted metadata or pretrained weights without permission. The available metadata/manifest CSV and JSON are preserved for reproducibility; **review any path fields, file identifiers and upstream dataset terms before public release**.

## Related documents

- `docs/DATA_AND_REPRODUCIBILITY.md`: input assets and caveats
- `docs/RELEASE_AUDIT.md`: source archive inventory, excluded categories and validation results
- `experiments/*/README_CN.md`: authoritative original project instructions

## Citation

Please cite the associated manuscript when published. A final paper DOI has not been provided in these archives. This release does not include a verified open-source license grant; see `LICENSE_NOTICE.md`.
