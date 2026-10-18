from __future__ import annotations
import argparse,json
from pathlib import Path
from deepship_uatr.split import discover_recording_units,inventory_summary,CLASS_NAMES
from deepship_uatr.common import save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/root/autodl-tmp/DeepShip')
    ap.add_argument('--out-dir',default='audit')
    a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    units,empty_dirs,ignored=discover_recording_units(a.root)
    summary=inventory_summary(units)
    counts={c:sum(u['class_name']==c for u in units) for c in CLASS_NAMES}
    summary['raw_recording_unit_counts']=counts
    summary['raw_balanced_max_per_class']=min(counts.values()) if counts else 0
    summary['target_for_this_project']=50
    summary['raw_count_can_reach_50_each']=all(v>=50 for v in counts.values())
    save_json(summary,out/'inventory_summary.json')
    save_json(units,out/'recording_units.json')
    save_json(empty_dirs,out/'empty_dirs.json')
    save_json(ignored,out/'ignored_nonwav_files.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nThis audit is informational. Step 03 performs stricter same-day + near-duplicate independence checks.')
    if not summary['raw_count_can_reach_50_each']:
        print('WARNING: at least one class has <50 raw recording/session units. Step 03 will stop.')

if __name__=='__main__':
    main()
