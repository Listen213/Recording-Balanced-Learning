from __future__ import annotations
import argparse
from pathlib import Path
import torch
from final_uatr.common import load_json,sha256_file

def verify(lock_path):
    lock=load_json(lock_path)
    p=lock["paths"]
    checks={
        "config_sha256":p["config"],
        "support_manifest_sha256":p["support_manifest"],
        "test_manifest_sha256":p["test_manifest"],
        "support_cache_sha256":p["support_cache"],
        "test_cache_sha256":p["test_cache"],
        "components_sha256":p["components"],
        "leakage_audit_sha256":p["leakage_audit"],
        "split_summary_sha256":p["split_summary"],
    }
    for key,path in checks.items():
        got=sha256_file(path)
        want=lock[key]
        if got!=want:
            raise RuntimeError(f"LOCK MISMATCH {key}: {path}\nwant={want}\ngot ={got}")

    sup=torch.load(p["support_cache"],map_location="cpu",weights_only=False)
    tst=torch.load(p["test_cache"],map_location="cpu",weights_only=False)
    if sup.get("manifest_sha256")!=lock["support_manifest_sha256"]:
        raise RuntimeError("support cache internal manifest hash mismatch")
    if tst.get("manifest_sha256")!=lock["test_manifest_sha256"]:
        raise RuntimeError("test cache internal manifest hash mismatch")
    return lock

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lock",default="final_lock/FINAL_LOCK.json")
    a=ap.parse_args()
    verify(a.lock)
    print("FINAL LOCK VERIFIED")

if __name__=="__main__":
    main()
