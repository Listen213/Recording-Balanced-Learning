#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, save_json, canonical_mmsi, cv, gini


def canon_path(root: Path, fold_dir: Path, raw: str) -> str:
    p = (fold_dir / str(raw)).resolve()
    try:
        return str(p.relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        raise RuntimeError(f"Path escapes dataset root: {raw} -> {p}")


def read_split(root: Path, fold: int, split: str) -> pd.DataFrame:
    fd = root / f"fold_{fold}"
    fp = fd / f"metadata_5s_{split}.csv"
    df = pd.read_csv(fp, low_memory=False)
    required = ["label", "path", "MMSI"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"{fp}: missing {miss}; columns={list(df.columns)}")
    out = pd.DataFrame()
    out["canonical_path"] = [canon_path(root, fd, x) for x in df["path"]]
    out["label"] = df["label"].astype(str).str.strip().str.lower()
    out["mmsi"] = [canonical_mmsi(x) for x in df["MMSI"]]
    out["source_id"] = ["MMSI:" + x if x not in ("", "0") else "BACKGROUND" for x in out["mmsi"]]
    out["official_split"] = split
    for col in ["duration_sec", "sample_rate", "date", "hydrophone"]:
        if col in df.columns:
            out[col] = df[col]
    return out


def hash_rank(seed: int, fold: int, label: str, source_id: str) -> str:
    key = f"{seed}|{fold}|train|{label}|{source_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def determine_eligible(full: pd.DataFrame, exclude: set[str], min_train: int, min_test: int) -> list[str]:
    vessel = full[(~full.label.isin(exclude)) & (full.source_id != "BACKGROUND")].copy()
    counts = vessel.groupby(["label", "official_split"])["source_id"].nunique().unstack(fill_value=0)
    for col in ["train", "test"]:
        if col not in counts.columns:
            counts[col] = 0
    return sorted(counts.index[(counts["train"] >= min_train) & (counts["test"] >= min_test)].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/uniqueship_subset.yaml")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    proj = Path(__file__).resolve().parents[1]
    root = Path(cfg["project"]["dataset_root"]).resolve()
    fold = int(cfg["project"]["official_fold"])
    seed = int(cfg["subset"]["selection_seed"])
    ns = int(cfg["subset"]["support_pool_sources_per_class"])
    exclude = {x.lower() for x in cfg["classes"]["exclude"]}
    min_train = int(cfg["classes"]["min_train_vessels"])
    min_test = int(cfg["classes"]["min_test_vessels"])
    expected = sorted(x.lower() for x in cfg["classes"]["expected_selected_classes"])

    parts = [read_split(root, fold, sp) for sp in ["train", "validation", "test"]]
    full = pd.concat(parts, ignore_index=True)
    eligible = determine_eligible(full, exclude, min_train, min_test)
    if eligible != expected:
        raise RuntimeError(f"metadata eligibility produced {eligible}, expected {expected}")

    # From here onward only the eligible 8 classes enter the experiment.
    vessel = full[(full.label.isin(expected)) & (full.source_id != "BACKGROUND")].copy()
    classes = sorted(vessel.label.unique())
    if classes != expected:
        raise RuntimeError(f"selected vessel classes {classes} != expected {expected}")

    bad = vessel.groupby("source_id").label.nunique()
    if (bad > 1).any():
        raise RuntimeError(f"{int((bad>1).sum())} MMSI/source IDs have >1 selected label")
    split_sets = {sp: set(vessel.loc[vessel.official_split == sp, "source_id"]) for sp in ["train", "validation", "test"]}
    overlap = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"official fold is not MMSI-disjoint for selected classes: {overlap}")

    train = vessel[vessel.official_split == "train"].copy()
    test = vessel[vessel.official_split == "test"].copy()
    selected_support = []
    selection_rows = []
    for lab in expected:
        cand = sorted(train.loc[train.label == lab, "source_id"].unique())
        ranked = sorted(cand, key=lambda sid: hash_rank(seed, fold, lab, sid))
        if len(ranked) < ns:
            raise RuntimeError(f"{lab}/train: requested {ns} support-pool sources but only {len(ranked)} available")
        chosen = ranked[:ns]
        selected_support.append(train[(train.label == lab) & train.source_id.isin(chosen)].copy())
        for sid in chosen:
            selection_rows.append({
                "label": lab,
                "split": "support",
                "source_id": sid,
                "official_split": "train",
                "clip_count": int(((train.label == lab) & (train.source_id == sid)).sum()),
                "deterministic_rank": int(ranked.index(sid) + 1),
                "selection_seed": seed,
            })

    support = pd.concat(selected_support, ignore_index=True)
    support["split"] = "support"
    test = test[test.label.isin(expected)].copy()
    test["split"] = "test"
    for (lab, sid), g in test.groupby(["label", "source_id"]):
        selection_rows.append({
            "label": lab,
            "split": "test",
            "source_id": sid,
            "official_split": "test",
            "clip_count": int(len(g)),
            "deterministic_rank": None,
            "selection_seed": None,
        })

    missing_test_classes = sorted(set(expected) - set(test.label.unique()))
    if missing_test_classes:
        raise RuntimeError(f"official test missing selected vessel classes: {missing_test_classes}")

    protocol = pd.concat([support, test], ignore_index=True)
    protocol = protocol.sort_values(["split", "label", "source_id", "canonical_path"]).reset_index(drop=True)
    protocol["abs_path"] = [str(root / p) for p in protocol.canonical_path]
    missing = [p for p in protocol.abs_path if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"missing selected audio files={len(missing)}; first={missing[:3]}")
    if protocol.canonical_path.duplicated().any():
        raise RuntimeError("duplicate clip paths inside fixed 8-class protocol")
    ss = {sp: set(protocol.loc[protocol.split == sp, "source_id"]) for sp in ["support", "test"]}
    if ss["support"] & ss["test"]:
        raise RuntimeError("source overlap between selected support and official test")

    proto_path = proj / cfg["project"]["protocol_file"]
    manifest_path = proj / cfg["project"]["selected_manifest"]
    selection_path = proj / cfg["project"]["selection_table"]
    proto_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (proj / "outputs").mkdir(parents=True, exist_ok=True)
    protocol.to_csv(proto_path, index=False)
    protocol.drop_duplicates("canonical_path").to_csv(manifest_path, index=False)
    pd.DataFrame(selection_rows).sort_values(["split", "label", "source_id"]).to_csv(selection_path, index=False)

    stats = []
    for (sp, lab), g in protocol.groupby(["split", "label"]):
        arr = g.groupby("source_id").size().to_numpy(float)
        stats.append({
            "split": sp,
            "label": lab,
            "sources": int(g.source_id.nunique()),
            "clips": int(len(g)),
            "clips_per_source_min": int(arr.min()),
            "clips_per_source_median": float(np.median(arr)),
            "clips_per_source_max": int(arr.max()),
            "clip_count_cv": cv(arr),
            "clip_count_gini": gini(arr),
        })
    pd.DataFrame(stats).to_csv(proj / "outputs" / "subset_stats.csv", index=False)
    audit = {
        "dataset_root": str(root),
        "official_fold": fold,
        "class_eligibility_rule": {
            "min_train_vessels": min_train,
            "min_test_vessels": min_test,
            "uses_model_results": False,
        },
        "selection_seed": seed,
        "support_selection_rule": "official train source IDs ranked by SHA256(seed|fold|train|label|MMSI); take first N per eligible class",
        "support_pool_sources_per_class": ns,
        "test_policy": "entire official Fold-1 test set for metadata-eligible vessel classes; background excluded",
        "classes": expected,
        "n_classes": len(expected),
        "support_sources": int(support.source_id.nunique()),
        "test_sources": int(test.source_id.nunique()),
        "support_clips": int(len(support)),
        "test_clips": int(len(test)),
        "official_source_overlap": overlap,
        "selected_support_test_source_overlap": len(ss["support"] & ss["test"]),
        "missing_audio": len(missing),
    }
    save_json(audit, proj / "outputs" / "uniqueship_subset_audit.json")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\nFIXED 8-CLASS SUPPORT SUBSET + OFFICIAL TEST PREPARED OK")
    print("Protocol:", proto_path)
    print("Selected clips:", manifest_path)
    print("Selection table:", selection_path)


if __name__ == "__main__":
    main()
