#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def macro_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """cm: [G, C, C], rows=true, cols=pred. Returns macro-F1 in [0, 1]."""
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(np.float64)
    true_n = cm.sum(axis=2, dtype=np.float64)
    pred_n = cm.sum(axis=1, dtype=np.float64)
    den = true_n + pred_n
    f1 = np.divide(2.0 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=1)


def _prepare_prediction_cube(pred: pd.DataFrame, fold: int, k: int):
    z = pred[
        (pred["fold"].astype(int) == int(fold))
        & (pred["k"].astype(str) == str(k))
        & pred["method"].isin(["s0", "s1"])
        & (pred["head"] == "lr")
        & (pred["scaler"] == "segment")
    ].copy()
    if z.empty:
        raise RuntimeError(f"No S0/S1 LR+segment predictions for fold={fold}, k={k}")

    draws = np.array(sorted(z["seed"].astype(int).unique()), dtype=np.int64)
    sources = np.array(sorted(z["source_id"].astype(str).unique()), dtype=object)
    labels = np.array(sorted(z["true"].astype(str).unique()), dtype=object)
    label_to_i = {lab: i for i, lab in enumerate(labels)}
    D, T, C = len(draws), len(sources), len(labels)
    draw_to_i = {int(x): i for i, x in enumerate(draws)}
    src_to_i = {str(x): i for i, x in enumerate(sources)}

    # One fixed ground-truth label per test source.
    truth_by_source = z[["source_id", "true"]].drop_duplicates()
    if truth_by_source.groupby("source_id")["true"].nunique().max() != 1:
        raise RuntimeError("A test source has inconsistent ground-truth labels")
    true_vec = np.full(T, -1, dtype=np.int16)
    for sid, tru in truth_by_source.itertuples(index=False, name=None):
        true_vec[src_to_i[str(sid)]] = label_to_i[str(tru)]
    if (true_vec < 0).any():
        raise RuntimeError("Missing ground truth for one or more test sources")

    cubes = {}
    for method in ("s0", "s1"):
        a = z[z["method"] == method]
        expected = D * T
        if len(a) != expected:
            raise RuntimeError(
                f"{method}: expected {expected} rows ({D} draws x {T} sources), got {len(a)}"
            )
        mat = np.full((D, T), -1, dtype=np.int16)
        for seed, sid, pred_lab in a[["seed", "source_id", "pred"]].itertuples(index=False, name=None):
            pi = label_to_i.get(str(pred_lab))
            if pi is None:
                raise RuntimeError(f"Prediction label {pred_lab!r} absent from true-label set")
            mat[draw_to_i[int(seed)], src_to_i[str(sid)]] = pi
        if (mat < 0).any():
            raise RuntimeError(f"{method}: incomplete draw x test-source prediction matrix")
        cubes[method] = mat

    return draws, sources, labels, true_vec, cubes


def hier_boot_fast(pred: pd.DataFrame, fold: int, k: int, iters: int, seed: int, batch_size: int):
    draws, sources, labels, true_vec, cubes = _prepare_prediction_cube(pred, fold, k)
    D, T, C = len(draws), len(sources), len(labels)
    rng = np.random.default_rng(seed)
    effects = np.empty(iters, dtype=np.float64)

    t0 = time.time()
    done = 0
    while done < iters:
        b = min(batch_size, iters - done)

        # Same hierarchy as the original implementation:
        # 1) resample support draws with replacement;
        # 2) independently within each sampled draw, resample test sources with replacement;
        # 3) compute paired S1-S0 Macro-F1 and average across sampled draws.
        draw_idx = rng.integers(0, D, size=(b, D), endpoint=False)
        src_idx = rng.integers(0, T, size=(b, D, T), endpoint=False)
        yt = true_vec[src_idx]  # [b, D, T]

        method_scores = {}
        groups = b * D
        offsets = (np.arange(groups, dtype=np.int64) * (C * C))[:, None]
        yt2 = yt.reshape(groups, T).astype(np.int64, copy=False)

        for method in ("s0", "s1"):
            selected_draws = cubes[method][draw_idx]  # [b, D, T]
            yp = np.take_along_axis(selected_draws, src_idx, axis=2)
            yp2 = yp.reshape(groups, T).astype(np.int64, copy=False)
            pair_code = yt2 * C + yp2
            bc = np.bincount(
                (offsets + pair_code).ravel(),
                minlength=groups * C * C,
            )
            cm = bc.reshape(groups, C, C)
            method_scores[method] = macro_f1_from_confusion(cm).reshape(b, D)

        effects[done : done + b] = 100.0 * (
            method_scores["s1"] - method_scores["s0"]
        ).mean(axis=1)
        done += b

        if done == iters or done % max(100, batch_size * 10) == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (iters - done) / max(rate, 1e-9)
            print(
                f"bootstrap fold={fold} k={k}: {done}/{iters} "
                f"({100*done/iters:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    return {
        "fold": int(fold),
        "k": int(k),
        "mean_effect": float(np.mean(effects)),
        "ci_low": float(np.quantile(effects, 0.025)),
        "ci_high": float(np.quantile(effects, 0.975)),
        "iterations": int(iters),
        "n_draws": int(D),
        "n_test_sources": int(T),
        "n_classes": int(C),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", default="outputs/summary")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261016)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # k is intentionally read as string because rows include 2/3/4 and 'full'.
    m = pd.read_csv(args.metrics, dtype={"k": str})
    p = pd.read_csv(args.predictions, dtype={"k": str, "source_id": str, "true": str, "pred": str})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    few = m[m["k"] != "full"].copy()
    grp = (
        few.groupby(["fold", "k", "method", "head", "scaler"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            n=("macro_f1", "size"),
        )
        .reset_index()
    )
    grp.to_csv(out / "summary.csv", index=False)

    # Paired S1-S0 draw table + imbalance correlations.
    base = few[
        few["method"].isin(["s0", "s1"])
        & (few["head"] == "lr")
        & (few["scaler"] == "segment")
    ]
    a0 = base[base["method"] == "s0"]
    a1 = base[base["method"] == "s1"]
    keys = ["fold", "k", "seed"]
    q = a0.merge(a1, on=keys, suffixes=("_s0", "_s1"))
    q["delta_macro_f1"] = q["macro_f1_s1"] - q["macro_f1_s0"]
    q.to_csv(out / "paired_s1_s0.csv", index=False)

    cors = []
    for (fold, k), g in q.groupby(["fold", "k"]):
        for col in [
            "source_clip_cv_s0",
            "source_clip_gini_s0",
            "source_clip_maxmin_s0",
            "class_clip_cv_s0",
        ]:
            x = np.asarray(g[col], dtype=float)
            y = np.asarray(g["delta_macro_f1"], dtype=float)
            if len(x) < 3 or pd.Series(x).dropna().nunique() <= 1 or pd.Series(y).dropna().nunique() <= 1:
                rho = np.nan
                pval = np.nan
            else:
                r = spearmanr(x, y, nan_policy="omit")
                rho = float(r.statistic)
                pval = float(r.pvalue)
            cors.append(
                {
                    "fold": int(fold),
                    "k": str(k),
                    "metric": col.replace("_s0", ""),
                    "rho": rho,
                    "p": pval,
                    "n": len(g),
                }
            )
    pd.DataFrame(cors).to_csv(out / "imbalance_gain_correlations.csv", index=False)

    boots = []
    folds = sorted(p["fold"].astype(int).unique())
    for fold in folds:
        for k in (2, 3, 4):
            boots.append(
                hier_boot_fast(
                    p,
                    int(fold),
                    k,
                    args.bootstrap_iters,
                    args.bootstrap_seed + int(fold) * 100 + k,
                    args.batch_size,
                )
            )
    pd.DataFrame(boots).to_csv(out / "hierarchical_bootstrap.csv", index=False)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
