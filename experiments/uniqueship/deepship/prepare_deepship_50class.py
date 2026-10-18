from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

from deepship_uatr.common import save_json, sha256_file
from deepship_uatr.split import (
    CLASS_NAMES, LABEL_TO_IDX, discover_recording_units, inventory_summary,
    recording_fingerprint, cosine,
)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def stable_rng(seed, class_index, offset=0):
    return np.random.default_rng(int(seed) + 10007 * int(class_index) + int(offset))


def build_exact_duplicate_components(units):
    """Hard components use only exact byte-level evidence.

    Two official recording/session units are joined iff:
      1) their complete unit signature is identical, or
      2) they share an identical WAV SHA256.

    Calendar date and acoustic similarity are deliberately NOT hard identity rules.
    """
    n = len(units)
    dsu = DSU(n)
    exact_edges = []

    by_unit_sha = defaultdict(list)
    for i, u in enumerate(units):
        by_unit_sha[u["unit_sha256"]].append(i)
    for ids in by_unit_sha.values():
        base = ids[0]
        for j in ids[1:]:
            dsu.union(base, j)
            exact_edges.append({
                "a": units[base]["recording_id"],
                "b": units[j]["recording_id"],
                "reason": "unit_sha256",
            })

    by_wav_sha = defaultdict(list)
    for i, u in enumerate(units):
        for af in u["audio_files"]:
            by_wav_sha[af["sha256"]].append(i)
    for ids0 in by_wav_sha.values():
        ids = sorted(set(ids0))
        if len(ids) <= 1:
            continue
        base = ids[0]
        for j in ids[1:]:
            dsu.union(base, j)
            exact_edges.append({
                "a": units[base]["recording_id"],
                "b": units[j]["recording_id"],
                "reason": "wav_sha256",
            })

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    comps = []
    conflicts = []
    for ids in groups.values():
        classes = sorted({units[i]["class_name"] for i in ids})
        rids = sorted(units[i]["recording_id"] for i in ids)
        cid = "exact_" + hashlib.sha1("|".join(rids).encode()).hexdigest()[:12]
        if len(classes) > 1:
            conflicts.append({
                "component_id": cid,
                "recording_ids": rids,
                "classes": classes,
            })
            continue
        comps.append({
            "component_id": cid,
            "class_name": classes[0],
            "label": int(units[ids[0]]["label"]),
            "indices": ids,
            "recording_ids": rids,
            "size": len(ids),
            "dates": sorted({units[i].get("session_date") for i in ids if units[i].get("session_date")}),
        })

    comps.sort(key=lambda c: (c["label"], c["component_id"]))
    return comps, exact_edges, conflicts


def exact_component_counts(comps):
    out = {c: 0 for c in CLASS_NAMES}
    for comp in comps:
        out[comp["class_name"]] += 1
    return out


def choose_balanced_recordings(units, comps, per_class, seed):
    """Choose exactly one representative from each selected exact component."""
    by = defaultdict(list)
    for ci, comp in enumerate(comps):
        by[comp["class_name"]].append(ci)

    selected = []
    counts = {}
    for class_index, cname in enumerate(CLASS_NAMES):
        ids = list(by[cname])
        counts[cname] = len(ids)
        if len(ids) < int(per_class):
            raise RuntimeError(
                f"{cname}: only {len(ids)} exact-duplicate components; cannot select {per_class}. "
                f"Balanced maximum is {min(len(by[c]) for c in CLASS_NAMES)}."
            )
        rng = stable_rng(seed, class_index, 31)
        picked = rng.choice(len(ids), int(per_class), replace=False)
        for jj in picked:
            comp = comps[ids[int(jj)]]
            # If an exact component has multiple byte-identical copies, choose one deterministically/randomly.
            ui = comp["indices"][int(rng.integers(len(comp["indices"])))]
            u = dict(units[ui])
            u["component_id"] = comp["component_id"]
            selected.append(u)

    return sorted(selected, key=lambda x: (x["label"], x["recording_id"])), counts


def _date_block_key(u):
    # Date is a soft correlation field, never a hard identity field.
    return u.get("session_date") or f"RID::{u['recording_id']}"


def _exact_block_subset(blocks, target, rng):
    order = np.arange(len(blocks))
    rng.shuffle(order)
    dp = {0: []}
    for ii in order:
        w = len(blocks[int(ii)][1])
        for s, chosen in list(dp.items()):
            ns = s + w
            if ns <= int(target) and ns not in dp:
                dp[ns] = chosen + [int(ii)]
        if int(target) in dp:
            return dp[int(target)]
    return None


def split_fixed_date_blocked(selected, support_per_class, test_per_class, seed):
    """30/20 split with date used only as a blocking preference.

    Same-day official sessions remain distinct recordings. When possible, all selected
    recordings from the same class/date are placed on the same side. If exact counts
    make that impossible, at most one date block is split and the exception is logged.
    """
    by = defaultdict(list)
    for u in selected:
        by[u["class_name"]].append(u)

    support, test = [], []
    diag = {"classes": {}}

    for ci, cname in enumerate(CLASS_NAMES):
        arr = list(by[cname])
        expected = int(support_per_class) + int(test_per_class)
        if len(arr) != expected:
            raise RuntimeError(f"{cname}: expected {expected} selected recordings, got {len(arr)}")

        groups = defaultdict(list)
        for u in arr:
            groups[_date_block_key(u)].append(u)
        blocks = sorted(groups.items(), key=lambda kv: kv[0])
        rng = stable_rng(seed, ci, 177)
        exact = _exact_block_subset(blocks, int(test_per_class), rng)
        split_block = None

        class_support, class_test = [], []
        if exact is not None:
            test_ids = set(exact)
            for bi, (_, rows) in enumerate(blocks):
                (class_test if bi in test_ids else class_support).extend(rows)
        else:
            # Fallback preserves exact 30/20 counts and splits at most one date block.
            order = np.arange(len(blocks))
            rng.shuffle(order)
            remaining = int(test_per_class)
            full_test_blocks = set()
            partial = None
            for ii in order:
                bi = int(ii)
                size = len(blocks[bi][1])
                if size <= remaining:
                    full_test_blocks.add(bi)
                    remaining -= size
                    if remaining == 0:
                        break
                elif partial is None and remaining > 0:
                    partial = (bi, remaining)
                    remaining = 0
                    break
            if remaining != 0:
                raise RuntimeError(f"{cname}: could not construct exact test size={test_per_class}")

            for bi, (key, rows0) in enumerate(blocks):
                rows = list(rows0)
                rng.shuffle(rows)
                if bi in full_test_blocks:
                    class_test.extend(rows)
                elif partial and bi == partial[0]:
                    k = int(partial[1])
                    class_test.extend(rows[:k])
                    class_support.extend(rows[k:])
                    split_block = key
                else:
                    class_support.extend(rows)

        if len(class_support) != int(support_per_class) or len(class_test) != int(test_per_class):
            raise RuntimeError(
                f"{cname}: split size mismatch support={len(class_support)} test={len(class_test)}"
            )

        sdates = {u["session_date"] for u in class_support if u.get("session_date")}
        tdates = {u["session_date"] for u in class_test if u.get("session_date")}
        overlap = sorted(sdates & tdates)
        diag["classes"][cname] = {
            "selected_recordings": len(arr),
            "selected_dated_recordings": sum(bool(u.get("session_date")) for u in arr),
            "selected_unique_dates": len({u["session_date"] for u in arr if u.get("session_date")}),
            "support_recordings": len(class_support),
            "test_recordings": len(class_test),
            "support_unique_dates": len(sdates),
            "test_unique_dates": len(tdates),
            "same_date_cross_split": overlap,
            "same_date_cross_split_count": len(overlap),
            "fallback_split_date_block": split_block,
        }
        support.extend(class_support)
        test.extend(class_test)

    return (
        sorted(support, key=lambda x: (x["label"], x["recording_id"])),
        sorted(test, key=lambda x: (x["label"], x["recording_id"])),
        diag,
    )


def hard_leakage_audit(support, test):
    sr = {u["recording_id"] for u in support}
    tr = {u["recording_id"] for u in test}
    sf = {af["sha256"] for u in support for af in u["audio_files"]}
    tf = {af["sha256"] for u in test for af in u["audio_files"]}
    sc = {u["component_id"] for u in support}
    tc = {u["component_id"] for u in test}
    return {
        "recording_id_overlap": len(sr & tr),
        "wav_sha256_overlap": len(sf & tf),
        "exact_component_overlap": len(sc & tc),
        "hard_leakage_clean": len(sr & tr) == 0 and len(sf & tf) == 0 and len(sc & tc) == 0,
    }


def class_counts(rows):
    return dict(Counter(u["class_name"] for u in rows))


def acoustic_similarity_diagnostics(selected, support, test, args):
    """Audio similarity is diagnostic only; it never merges recordings or changes split."""
    split_of = {u["recording_id"]: "support_pool" for u in support}
    split_of.update({u["recording_id"]: "test" for u in test})

    fps = {}
    total = len(selected)
    print(f"Computing acoustic diagnostics for selected {total} recordings (diagnostic only)...", flush=True)
    for i, u in enumerate(selected, 1):
        fps[u["recording_id"]] = recording_fingerprint(
            u,
            target_sr=args.fingerprint_sr,
            window_sec=args.fingerprint_window_sec,
            n_views=args.fingerprint_views,
        )
        if i % 20 == 0 or i == total:
            print(f"  diagnostic fingerprint {i}/{total}", flush=True)

    by = defaultdict(list)
    for u in selected:
        by[u["class_name"]].append(u)

    result = {
        "policy": "DIAGNOSTIC_ONLY_DO_NOT_MERGE_OR_RESPLIT",
        "threshold": float(args.diagnostic_threshold),
        "fingerprint": {
            "target_sr": int(args.fingerprint_sr),
            "window_sec": float(args.fingerprint_window_sec),
            "n_views": int(args.fingerprint_views),
        },
        "classes": {},
    }
    all_top = []

    for cname in CLASS_NAMES:
        arr = by[cname]
        sims = []
        above = []
        cross_above = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                a, b = arr[i], arr[j]
                sim = cosine(fps[a["recording_id"]], fps[b["recording_id"]])
                row = {
                    "recording_a": a["recording_id"],
                    "recording_b": b["recording_id"],
                    "similarity": float(sim),
                    "split_a": split_of[a["recording_id"]],
                    "split_b": split_of[b["recording_id"]],
                    "same_date": bool(a.get("session_date") and a.get("session_date") == b.get("session_date")),
                }
                sims.append(sim)
                all_top.append({"class_name": cname, **row})
                if sim >= float(args.diagnostic_threshold):
                    above.append(row)
                    if row["split_a"] != row["split_b"]:
                        cross_above.append(row)

        v = np.asarray(sims, dtype=float)
        above.sort(key=lambda r: -r["similarity"])
        cross_above.sort(key=lambda r: -r["similarity"])
        result["classes"][cname] = {
            "selected_recordings": len(arr),
            "pair_count": int(len(v)),
            "max_similarity": float(v.max()) if len(v) else None,
            "median_similarity": float(np.median(v)) if len(v) else None,
            "p95_similarity": float(np.quantile(v, 0.95)) if len(v) else None,
            "p99_similarity": float(np.quantile(v, 0.99)) if len(v) else None,
            "pairs_ge_threshold_count": len(above),
            "cross_split_pairs_ge_threshold_count": len(cross_above),
            "top_pairs_ge_threshold": above[:100],
            "top_cross_split_pairs_ge_threshold": cross_above[:100],
        }

    all_top.sort(key=lambda r: -r["similarity"])
    result["global_top_100_pairs"] = all_top[:100]
    result["note"] = (
        "High same-class cosine similarity is expected in DeepShip and is NOT treated as duplicate identity. "
        "Only exact SHA256 evidence creates hard components."
    )
    return result, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/DeepShip")
    ap.add_argument("--out-dir", default="manifests_deepship_50")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--support-per-class", type=int, default=30)
    ap.add_argument("--test-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--segment-sec", type=float, default=5.0)
    ap.add_argument("--hop-sec", type=float, default=5.0)
    ap.add_argument("--diagnostic-threshold", type=float, default=0.997)
    ap.add_argument("--fingerprint-sr", type=int, default=8000)
    ap.add_argument("--fingerprint-window-sec", type=float, default=20.0)
    ap.add_argument("--fingerprint-views", type=int, default=3)
    args = ap.parse_args()

    if args.support_per_class + args.test_per_class != args.per_class:
        raise ValueError("support-per-class + test-per-class must equal per-class")
    if args.hop_sec < args.segment_sec:
        raise ValueError("Use non-overlapping segments: hop_sec >= segment_sec")

    out = Path(args.out_dir)
    if (out / "support_pool.json").exists() or (out / "test.json").exists():
        raise RuntimeError("Refusing to overwrite an existing DeepShip split.")
    out.mkdir(parents=True, exist_ok=True)

    units, empty_dirs, ignored = discover_recording_units(args.root)
    inv = inventory_summary(units)
    save_json(inv, out / "inventory_summary.json")
    save_json(empty_dirs, out / "empty_dirs.json")
    save_json(ignored, out / "ignored_nonwav_files.json")

    raw_counts = {c: sum(u["class_name"] == c for u in units) for c in CLASS_NAMES}
    print("Raw recording/session units:", raw_counts, flush=True)
    print("Hard duplicate identity: exact SHA256 only.", flush=True)
    print("Calendar date: soft correlation control only.", flush=True)
    print("Acoustic similarity: diagnostic only; never used to merge/resplit.", flush=True)

    comps, exact_edges, conflicts = build_exact_duplicate_components(units)
    if conflicts:
        save_json(conflicts, out / "EXACT_DUPLICATE_LABEL_CONFLICTS.json")
        raise RuntimeError("Exact byte-identical data appears under multiple class labels. Inspect conflict file.")

    exact_counts = exact_component_counts(comps)
    print("Exact-deduplicated component counts:", exact_counts, flush=True)
    save_json(exact_edges, out / "exact_duplicate_edges.json")
    save_json([
        {
            "component_id": c["component_id"],
            "class_name": c["class_name"],
            "size": c["size"],
            "recording_ids": c["recording_ids"],
            "dates": c["dates"],
        }
        for c in comps
    ], out / "exact_duplicate_components_all.json")

    selected, independent_counts = choose_balanced_recordings(units, comps, args.per_class, args.seed)
    support, test, date_diag = split_fixed_date_blocked(
        selected, args.support_per_class, args.test_per_class, args.seed
    )

    hard_audit = hard_leakage_audit(support, test)
    if not hard_audit["hard_leakage_clean"]:
        save_json(hard_audit, out / "FAILED_HARD_LEAKAGE_AUDIT.json")
        raise RuntimeError(f"Hard leakage audit failed: {hard_audit}")

    base = {
        "dataset": "DeepShip",
        "protocol": "balanced 50-recordings/class external validation v3",
        "selection_seed": args.seed,
        "selected_recordings_per_class": args.per_class,
        "support_pool_per_class": args.support_per_class,
        "frozen_test_per_class": args.test_per_class,
        "hard_duplicate_rule": "exact SHA256 only",
        "same_day_hard_grouping": False,
        "date_blocked_support_test_split": True,
        "acoustic_similarity_role": "diagnostic_only",
        "acoustic_diagnostic_threshold": args.diagnostic_threshold,
        "target_sr": args.target_sr,
        "segment_sec": args.segment_sec,
        "hop_sec": args.hop_sec,
        "label_to_idx": LABEL_TO_IDX,
        "class_names": CLASS_NAMES,
    }

    save_json({**base, "split": "support_pool", "recordings": support}, out / "support_pool.json")
    save_json({**base, "split": "test", "recordings": test}, out / "test.json")
    save_json(selected, out / "selected_50_per_class.json")
    save_json(date_diag, out / "date_split_diagnostics.json")
    save_json(hard_audit, out / "leakage_audit.json")

    acoustic_diag, fps = acoustic_similarity_diagnostics(selected, support, test, args)
    save_json(acoustic_diag, out / "acoustic_similarity_diagnostics.json")
    np.savez_compressed(
        out / "selected_acoustic_fingerprints.npz",
        recording_ids=np.asarray([u["recording_id"] for u in selected], dtype=object),
        fingerprints=np.stack([fps[u["recording_id"]] for u in selected]).astype(np.float32),
    )

    selected_components = []
    selected_by_id = {u["recording_id"]: u for u in selected}
    selected_comp_ids = {u["component_id"] for u in selected}
    support_comp_ids = {u["component_id"] for u in support}
    for c in comps:
        if c["component_id"] in selected_comp_ids:
            selected_components.append({
                "component_id": c["component_id"],
                "class_name": c["class_name"],
                "size": c["size"],
                "recording_ids": c["recording_ids"],
                "dates": c["dates"],
                "split": "support_pool" if c["component_id"] in support_comp_ids else "test",
            })
    save_json(selected_components, out / "selected_exact_components.json")

    summary = {
        "raw_recording_unit_counts": raw_counts,
        "exact_deduplicated_component_counts": independent_counts,
        "exact_duplicate_edges_count": len(exact_edges),
        "selected_per_class": args.per_class,
        "support_pool_per_class": args.support_per_class,
        "frozen_test_per_class": args.test_per_class,
        "support_total": len(support),
        "test_total": len(test),
        "support_counts": class_counts(support),
        "test_counts": class_counts(test),
        "hard_duplicate_rule": "exact SHA256 only",
        "same_day_hard_grouping": False,
        "acoustic_similarity_role": "diagnostic_only",
        "acoustic_diagnostic_threshold": args.diagnostic_threshold,
        "date_split_diagnostics": date_diag,
        "hard_leakage_audit": hard_audit,
        "support_manifest_sha256": sha256_file(out / "support_pool.json"),
        "test_manifest_sha256": sha256_file(out / "test.json"),
    }
    save_json(summary, out / "split_summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("DeepShip V3 subset prepared.")
    print(f"{args.per_class} official recording/session units per class = "
          f"{args.support_per_class} support + {args.test_per_class} frozen test.")
    print("Only exact SHA256 duplicates are hard-merged.")
    print("Acoustic similarity has been saved as DIAGNOSTIC ONLY.")


if __name__ == "__main__":
    main()
