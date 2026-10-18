#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -e runs/final_fast/FINAL_COMPLETE.json ]] || { echo "Final run is not complete."; exit 2; }
ts="$(date +%Y%m%d_%H%M%S)"
out="fewrec_final_fast_results_${ts}.zip"
python - "$out" <<'PY'
import sys,zipfile
from pathlib import Path
out=Path(sys.argv[1])
items=[
    Path('FINAL_CONFIG.json'),Path('final_lock'),
    Path('manifests_final/split_summary.json'),Path('manifests_final/leakage_audit.json'),
    Path('manifests_final/components.json'),Path('runs/final_fast/FINAL_COMPLETE.json'),
    Path('runs/final_fast/summary.json'),Path('runs/final_fast/summary.csv'),
    Path('runs/final_fast/detail.json'),Path('runs/final_fast/support_draws.json'),
    Path('runs/final_fast/hierarchical_paired_bootstrap.json'),
    Path('runs/final_fast/predictions.npz'),Path('runs/final_fast/PROGRESS.json'),
    Path('paper_outputs'),Path('logs/final_fast.log'),Path('logs/final_fast_resume.log'),
]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in items:
        if p.is_file(): z.write(p,p.as_posix())
        elif p.is_dir():
            for f in p.rglob('*'):
                if f.is_file(): z.write(f,f.as_posix())
print(out)
PY
