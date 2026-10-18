from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from final_uatr.common import save_json, sha256_file
from final_uatr.split import (
    LABEL_TO_IDX, build_recordings, acoustic_fingerprint, build_components,
    split_components, audit_split, class_recording_counts, class_component_counts
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audio-root",required=True)
    ap.add_argument("--csv",required=True)
    ap.add_argument("--out-dir",default="manifests_final")
    ap.add_argument("--seed",type=int,default=20261014)
    ap.add_argument("--test-ratio",type=float,default=0.30)
    ap.add_argument("--near-dup-threshold",type=float,default=0.997)
    ap.add_argument("--min-support-components-per-class",type=int,default=6)
    ap.add_argument("--target-sr",type=int,default=16000)
    ap.add_argument("--segment-sec",type=float,default=5.0)
    ap.add_argument("--hop-sec",type=float,default=5.0)
    ap.add_argument("--fingerprint-sr",type=int,default=8000)
    ap.add_argument("--fingerprint-window-sec",type=float,default=20.0)
    ap.add_argument("--fingerprint-windows",type=int,default=3)
    args=ap.parse_args()

    if args.hop_sec < args.segment_sec:
        raise ValueError("Final protocol requires non-overlapping evaluation/support cache: hop_sec >= segment_sec")
    if not (0.05 <= args.test_ratio <= 0.5):
        raise ValueError("test_ratio should be between 0.05 and 0.5")

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    records=build_recordings(args.audio_root,args.csv)
    print(f"Found {len(records)} raw recordings. Computing leakage fingerprints...")

    fps=[]
    for i,r in enumerate(records,1):
        fps.append(acoustic_fingerprint(
            r["path"],
            target_sr=args.fingerprint_sr,
            window_sec=args.fingerprint_window_sec,
            n_windows=args.fingerprint_windows,
        ))
        if i%10==0 or i==len(records):
            print(f"  fingerprint {i}/{len(records)}")
    fps=np.stack(fps)

    comps,pairs,conflicts=build_components(records,fps,args.near_dup_threshold)
    if conflicts:
        save_json(conflicts,out/"LABEL_CONFLICT_COMPONENTS.json")
        raise RuntimeError(
            "Near-duplicate component contains conflicting coarse labels. "
            "Inspect LABEL_CONFLICT_COMPONENTS.json before proceeding."
        )

    support,test,test_comp=split_components(
        records,comps,args.seed,args.test_ratio,args.min_support_components_per_class
    )
    audit=audit_split(support,test,pairs)
    if audit["recording_id_overlap"] or audit["sha256_overlap"] or audit["cross_split_near_duplicate_count"]:
        save_json(audit,out/"FAILED_LEAKAGE_AUDIT.json")
        raise RuntimeError("Leakage audit failed; final manifests were not frozen.")

    base={
        "protocol":"fresh near-duplicate-component-aware ShipsEar final protocol",
        "split_seed":args.seed,
        "near_dup_threshold":args.near_dup_threshold,
        "target_sr":args.target_sr,
        "segment_sec":args.segment_sec,
        "hop_sec":args.hop_sec,
        "label_to_idx":LABEL_TO_IDX,
        "fingerprint":{
            "target_sr":args.fingerprint_sr,
            "window_sec":args.fingerprint_window_sec,
            "n_windows":args.fingerprint_windows,
        },
    }
    sup={**base,"split":"support_pool","recordings":support}
    tst={**base,"split":"test","recordings":test}
    save_json(sup,out/"support_pool.json")
    save_json(tst,out/"test.json")

    comp_dump=[]
    for i,c in enumerate(comps):
        comp_dump.append({
            "component_id":c["component_id"],
            "label_name":c["label_name"],
            "size":c["size"],
            "recording_ids":c["recording_ids"],
            "source_types":c["source_types"],
            "split":"test" if i in test_comp else "support_pool",
        })
    save_json(comp_dump,out/"components.json")
    save_json(pairs,out/"near_duplicate_edges.json")
    save_json(audit,out/"leakage_audit.json")

    np.savez_compressed(
        out/"fingerprints.npz",
        recording_ids=np.asarray([r["recording_id"] for r in records],dtype=object),
        fingerprints=fps.astype(np.float32),
    )

    summary={
        "raw_recordings":len(records),
        "near_duplicate_components":len(comps),
        "near_duplicate_edges":len(pairs),
        "support_pool_recordings":len(support),
        "test_recordings":len(test),
        "support_pool_recording_counts":class_recording_counts(support),
        "test_recording_counts":class_recording_counts(test),
        "support_pool_component_counts":class_component_counts(support),
        "test_component_counts":class_component_counts(test),
        "support_pool_manifest_sha256":sha256_file(out/"support_pool.json"),
        "test_manifest_sha256":sha256_file(out/"test.json"),
        "leakage_audit":audit,
    }
    save_json(summary,out/"split_summary.json")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("FRESH FINAL SPLIT PREPARED. Do not inspect final test outcomes before protocol lock.")

if __name__=="__main__":
    main()
