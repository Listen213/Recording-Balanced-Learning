from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from final_uatr.common import sha256_file,save_json,load_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="FINAL_CONFIG.json")
    ap.add_argument("--manifest-dir",default="manifests_final")
    ap.add_argument("--cache-dir",default="cache/beats_final")
    ap.add_argument("--out",default="final_lock/FINAL_LOCK.json")
    a=ap.parse_args()

    cfg=load_json(a.config)
    md=Path(a.manifest_dir); cd=Path(a.cache_dir)
    sup_m=md/"support_pool.json"; test_m=md/"test.json"
    sup_c=cd/"support_pool.pt"; test_c=cd/"test.pt"
    for p in [sup_m,test_m,sup_c,test_c,Path(a.config)]:
        if not p.exists():
            raise FileNotFoundError(p)

    sup_obj=torch.load(sup_c,map_location="cpu",weights_only=False)
    test_obj=torch.load(test_c,map_location="cpu",weights_only=False)
    sup_mh=sha256_file(sup_m); test_mh=sha256_file(test_m)
    if sup_obj.get("manifest_sha256")!=sup_mh:
        raise RuntimeError("support cache does not match support manifest")
    if test_obj.get("manifest_sha256")!=test_mh:
        raise RuntimeError("test cache does not match test manifest")

    audit=load_json(md/"leakage_audit.json")
    if audit.get("recording_id_overlap") or audit.get("sha256_overlap") or audit.get("cross_split_near_duplicate_count"):
        raise RuntimeError("leakage audit is not clean")

    lock={
        "lock_version":1,
        "config_sha256":sha256_file(a.config),
        "support_manifest_sha256":sup_mh,
        "test_manifest_sha256":test_mh,
        "support_cache_sha256":sha256_file(sup_c),
        "test_cache_sha256":sha256_file(test_c),
        "components_sha256":sha256_file(md/"components.json"),
        "leakage_audit_sha256":sha256_file(md/"leakage_audit.json"),
        "split_summary_sha256":sha256_file(md/"split_summary.json"),
        "config":cfg,
        "paths":{
            "config":str(Path(a.config)),
            "support_manifest":str(sup_m),
            "test_manifest":str(test_m),
            "support_cache":str(sup_c),
            "test_cache":str(test_c),
            "components":str(md/"components.json"),
            "leakage_audit":str(md/"leakage_audit.json"),
            "split_summary":str(md/"split_summary.json"),
        }
    }
    save_json(lock,a.out)
    print(json.dumps(lock,ensure_ascii=False,indent=2))
    print("FINAL PROTOCOL FROZEN.")
    print("From this point, do not change split, hyperparameters, methods, or support seeds.")

if __name__=="__main__":
    main()
