#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, canonical_mmsi


def read_meta(root: Path, fold: int, split: str) -> pd.DataFrame:
    fp = root / f"fold_{fold}" / f"metadata_5s_{split}.csv"
    df = pd.read_csv(fp, low_memory=False)
    need = {"label", "MMSI", "path"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise RuntimeError(f"{fp}: missing columns {missing}")
    out = pd.DataFrame({
        "label": df["label"].astype(str).str.strip().str.lower(),
        "mmsi": [canonical_mmsi(x) for x in df["MMSI"]],
        "path": df["path"].astype(str),
    })
    out = out[(out.mmsi != "") & (out.mmsi != "0")].copy()
    out["source_id"] = "MMSI:" + out.mmsi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/uniqueship_subset.yaml")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    root = Path(cfg["project"]["dataset_root"]).resolve()
    fold = int(cfg["project"]["official_fold"])
    exclude = {x.lower() for x in cfg["classes"]["exclude"]}
    min_train = int(cfg["classes"]["min_train_vessels"])
    min_test = int(cfg["classes"]["min_test_vessels"])
    expected = sorted(x.lower() for x in cfg["classes"]["expected_selected_classes"])

    all_frames = {}
    all_labels = set()
    for split in ["train", "validation", "test"]:
        df = read_meta(root, fold, split)
        df = df[~df.label.isin(exclude)].copy()
        all_frames[split] = df
        all_labels.update(df.label.unique())

    rows = []
    for lab in sorted(all_labels):
        for split, df in all_frames.items():
            g = df[df.label == lab]
            rows.append({
                "split": split,
                "label": lab,
                "sources": int(g.source_id.nunique()),
                "clips": int(len(g)),
            })
    out = pd.DataFrame(rows)
    proj = Path(__file__).resolve().parents[1]
    (proj / "outputs").mkdir(exist_ok=True)
    out.to_csv(proj / "outputs" / "available_sources_fold1.csv", index=False)

    piv = out.pivot(index="label", columns="split", values="sources").fillna(0).astype(int)
    for col in ["train", "validation", "test"]:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv[["test", "train", "validation"]]
    print(f"\nUnique vessel counts by class (official fold {fold}):")
    print(piv.to_string())

    eligible = sorted(piv.index[(piv["train"] >= min_train) & (piv["test"] >= min_test)].tolist())
    ineligible = piv.loc[[x for x in piv.index if x not in eligible]].copy()
    print("\nMetadata-only eligibility rule:")
    print(f"  train vessels >= {min_train}")
    print(f"  test vessels  >= {min_test}")
    print("Eligible classes:", ", ".join(eligible))
    if len(ineligible):
        print("\nExcluded vessel classes by the rule:")
        print(ineligible.to_string())

    if eligible != expected:
        print("\nELIGIBILITY MISMATCH")
        print("Expected for UniqueShip v1.0:", expected)
        print("Observed:", eligible)
        raise SystemExit(2)

    req_s = int(cfg["subset"]["support_pool_sources_per_class"])
    if req_s > min_train:
        raise RuntimeError(f"support_pool_sources_per_class={req_s} exceeds min_train_vessels={min_train}")
    selected = piv.loc[expected]
    bad_s = selected[selected["train"] < req_s]
    bad_t = selected[selected["test"] < min_test]
    if len(bad_s) or len(bad_t):
        print("\nINSUFFICIENT COUNTS for requested 8-class protocol.")
        if len(bad_s): print("Train shortage:\n", bad_s.to_string())
        if len(bad_t): print("Test shortage:\n", bad_t.to_string())
        raise SystemExit(2)

    print(f"\nRequested support subset: {req_s} vessel IDs/class from official train")
    print("Test policy: keep the ENTIRE official Fold-1 test set for the 8 eligible classes.")
    print("\nCOUNT CHECK OK: 8-class protocol is feasible.")


if __name__ == "__main__":
    main()
