from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .common import soft_clip_probs

def _l2(x, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), eps, None)

def recording_vector(s1, rec):
    # Equal-recording representation: average standardized segment embeddings,
    # then normalize to the unit sphere.
    z = s1.transform(rec.embeddings).mean(axis=0)
    return _l2(z)

def _class_prototypes(vecs, y, n_classes):
    ps = []
    for c in range(n_classes):
        v = vecs[y == c]
        if len(v) == 0:
            raise ValueError(f"class {c} absent from support")
        ps.append(_l2(v.mean(axis=0)))
    return np.stack(ps)

def _prototype_uncertainty(X, y, p0, n_classes):
    """
    Recording-level prototype uncertainty.

    It combines:
      1) within-class angular dispersion;
      2) leave-one-recording-out prototype movement.

    No segment is treated as an independent support item.
    """
    raw = np.zeros(n_classes, dtype=np.float64)
    within = np.zeros(n_classes, dtype=np.float64)
    loo = np.zeros(n_classes, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)

    for c in range(n_classes):
        V = _l2(X[y == c])
        counts[c] = len(V)
        if len(V) == 0:
            raise ValueError(f"class {c} absent")
        # 1-cos distance to full equal-recording prototype.
        within[c] = float(np.mean(1.0 - np.clip(V @ p0[c], -1.0, 1.0)))
        if len(V) >= 2:
            ds = []
            for i in range(len(V)):
                q = _l2(np.delete(V, i, axis=0).mean(axis=0))
                ds.append(1.0 - float(np.clip(q @ p0[c], -1.0, 1.0)))
            loo[c] = float(np.mean(ds))
        else:
            # A one-recording class has maximal epistemic ambiguity for LOO;
            # use within-class term plus a count penalty, not a fabricated zero.
            loo[c] = float(within[c] + 0.25)
        raw[c] = 0.5 * within[c] + 0.5 * loo[c]

    positive = raw[raw > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    norm = np.clip(raw / max(scale, 1e-8), 0.0, 4.0)
    return {
        "raw": raw,
        "normalized": norm,
        "within": within,
        "loo": loo,
        "counts": counts,
        "scale": scale,
    }

def _residual_basis(X, y, p0, rank):
    residual = X - p0[y]
    residual = residual - residual.mean(axis=0, keepdims=True)
    if len(residual) > 1 and np.linalg.norm(residual) > 1e-8:
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        rr = max(1, min(int(rank), vt.shape[0], X.shape[1]))
        U = vt[:rr].T.astype(np.float32)
    else:
        rr = 1
        U = np.zeros((X.shape[1], 1), np.float32)
        U[0, 0] = 1.0
    return U, rr

def _class_residual_variance(X, y, p0, U, n_classes, shrink=2.0):
    coeff = (X - p0[y]) @ U
    global_var = np.mean(coeff ** 2, axis=0) + 1e-6
    out = []
    for c in range(n_classes):
        z = coeff[y == c]
        if len(z) >= 2:
            local = np.var(z, axis=0, ddof=1)
        else:
            local = global_var.copy()
        # Small-sample shrinkage toward pooled residual variance.
        var = (len(z) * local + float(shrink) * global_var) / (len(z) + float(shrink))
        out.append(np.maximum(var, 1e-6))
    return np.stack(out).astype(np.float32)

@dataclass
class PrototypeModel:
    s1: object
    prototypes: np.ndarray
    temperature: float = 0.25
    info: dict | None = None

    @classmethod
    def fit_base(cls, s1, recordings, n_classes, temperature=0.25):
        X = np.stack([recording_vector(s1, r) for r in recordings])
        y = np.asarray([r.label for r in recordings], np.int64)
        p = _class_prototypes(X, y, n_classes)
        unc = _prototype_uncertainty(X, y, p, n_classes)
        info = {
            "type": "PB",
            "trainable_params": 0,
            "uncertainty_raw": unc["raw"].tolist(),
            "uncertainty_norm": unc["normalized"].tolist(),
            "class_counts": unc["counts"].tolist(),
        }
        return cls(s1, p, float(temperature), info)

    @classmethod
    def fit_tuned(
        cls, s1, recordings, n_classes, rank=4, steps=160, lr=0.05,
        l2=1.0, temperature=0.25, seed=0
    ):
        """Low-rank prototype tuning baseline (no uncertainty gate, no perturbation)."""
        return cls.fit_uncertainty_tuned(
            s1, recordings, n_classes, rank=rank, steps=steps, lr=lr, l2=l2,
            temperature=temperature, seed=seed, gamma=0.0, perturb_scale=0.0,
            mc_samples=1, method_name="PT"
        )

    @classmethod
    def fit_uncertainty_tuned(
        cls, s1, recordings, n_classes, rank=4, steps=160, lr=0.05,
        l2=1.0, temperature=0.25, seed=0, gamma=1.0,
        perturb_scale=0.0, mc_samples=4, method_name="UP"
    ):
        """
        Recording-Uncertainty-Calibrated Low-Rank Prototype Tuning.

        Base prototype:
            equal mass per independent recording.

        Uncertainty gate:
            g_c = exp(-gamma * u_c),
        where u_c is recording-level prototype instability.

        Prototype perturbation:
            draws plausible class centers from empirical low-rank residual variance.
            It never creates new independent recordings; it is an optimization
            regularizer around the support-derived class center.
        """
        X = np.stack([recording_vector(s1, r) for r in recordings]).astype(np.float32)
        y = np.asarray([r.label for r in recordings], np.int64)
        p0 = _class_prototypes(X, y, n_classes).astype(np.float32)
        unc = _prototype_uncertainty(X, y, p0, n_classes)
        U, rr = _residual_basis(X, y, p0, rank)
        class_var = _class_residual_variance(X, y, p0, U, n_classes)

        gate = np.exp(-float(gamma) * unc["normalized"]).astype(np.float32)
        # Do not let a very uncertain class become completely frozen.
        gate = np.clip(gate, 0.10, 1.0)

        torch.manual_seed(int(seed))
        Xt = torch.tensor(X)
        yt = torch.tensor(y, dtype=torch.long)
        p0t = torch.tensor(p0)
        Ut = torch.tensor(U)
        gatet = torch.tensor(gate[:, None])
        vart = torch.tensor(class_var)

        coeff = torch.nn.Parameter(torch.zeros(n_classes, rr))
        opt = torch.optim.Adam([coeff], lr=float(lr))
        best = None

        for st in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            center = F.normalize(p0t + gatet * (coeff @ Ut.T), dim=1)

            losses = []
            nmc = max(1, int(mc_samples if perturb_scale > 0 else 1))
            for _ in range(nmc):
                if perturb_scale > 0:
                    eps = torch.randn_like(coeff) * torch.sqrt(vart)
                    # Larger empirical uncertainty => stronger center perturbation.
                    uamp = torch.tensor(
                        np.sqrt(np.clip(unc["normalized"], 0.0, 4.0) + 0.25),
                        dtype=torch.float32
                    )[:, None]
                    eps = eps * float(perturb_scale) * uamp
                    p = F.normalize(center + eps @ Ut.T, dim=1)
                else:
                    p = center
                logits = (F.normalize(Xt, dim=1) @ p.T) / float(temperature)
                losses.append(F.cross_entropy(logits, yt))
            ce = torch.stack(losses).mean()
            reg = torch.mean(coeff ** 2)
            # Prototype-retention regularizer in angular space.
            retain = torch.mean(1.0 - torch.sum(center * F.normalize(p0t, dim=1), dim=1))
            loss = ce + float(l2) * reg + 0.25 * retain
            loss.backward()
            torch.nn.utils.clip_grad_norm_([coeff], 5.0)
            opt.step()

            val = float(loss.detach())
            if best is None or val < best[0]:
                best = (val, coeff.detach().clone())

        with torch.no_grad():
            coeff.copy_(best[1])
            p = F.normalize(p0t + gatet * (coeff @ Ut.T), dim=1).cpu().numpy().astype(np.float64)
            shift = float(torch.sqrt(torch.mean(coeff ** 2)).cpu())
            angular_shift = (
                1.0 - np.sum(_l2(p) * _l2(p0), axis=1)
            ).astype(float)

        info = {
            "type": method_name,
            "rank": int(rr),
            "steps": int(steps),
            "lr": float(lr),
            "l2": float(l2),
            "temperature": float(temperature),
            "gamma": float(gamma),
            "perturb_scale": float(perturb_scale),
            "mc_samples": int(mc_samples),
            "trainable_params": int(n_classes * rr),
            "coef_rms": shift,
            "prototype_angular_shift": angular_shift.tolist(),
            "uncertainty_raw": unc["raw"].tolist(),
            "uncertainty_norm": unc["normalized"].tolist(),
            "uncertainty_within": unc["within"].tolist(),
            "uncertainty_loo": unc["loo"].tolist(),
            "class_counts": unc["counts"].tolist(),
            "adaptation_gate": gate.astype(float).tolist(),
            "residual_var_mean": np.mean(class_var, axis=1).astype(float).tolist(),
        }
        return cls(s1, p, float(temperature), info)

    def logits_recording(self, rec):
        z = recording_vector(self.s1, rec)
        return (z @ self.prototypes.T) / float(self.temperature)

    def predict_recording(self, rec):
        z = self.logits_recording(rec)
        z = z - z.max()
        e = np.exp(z)
        return soft_clip_probs(e / e.sum())
