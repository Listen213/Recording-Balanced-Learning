#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

if [[ ! -e runs/deepship_final/FINAL_COMPLETE.json ]]; then
  echo "DeepShip external validation is not complete."
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"
out="deepship_fewrec_results_${ts}.zip"

python - "$out" <<'PY'
import sys,zipfile
from pathlib import Path

out=Path(sys.argv[1])
items=[
    Path("DEEPSHIP_CONFIG.json"),
    Path("audit/inventory_summary.json"),
    Path("manifests_deepship_50/split_summary.json"),
    Path("manifests_deepship_50/leakage_audit.json"),
    Path("manifests_deepship_50/date_split_diagnostics.json"),
    Path("manifests_deepship_50/acoustic_similarity_diagnostics.json"),
    Path("manifests_deepship_50/exact_duplicate_edges.json"),
    Path("manifests_deepship_50/selected_exact_components.json"),
    Path("deepship_lock"),
    Path("runs/deepship_final"),
    Path("paper_outputs_deepship"),
    Path("logs/deepship_final.log"),
]
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for p in items:
        if p.is_file():
            z.write(p,p.as_posix())
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    z.write(f,f.as_posix())
print(out)
PY
