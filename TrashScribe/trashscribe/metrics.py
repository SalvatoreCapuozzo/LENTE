"""Evaluation metrics.

Segmentation : IoU / F1 / precision / recall of the waste class, boundary F1.
Description  : per-field mAP against the prevalence floor (the honest zero of a
               multi-label problem).
Naming       : generalised zero-shot top-1 on seen and held-out classes and their
               harmonic mean (Xian et al., 2019).
Retrieval    : AP of an atlas entry over a pool, lift = AP / prevalence, bootstrap
               CI and permutation p-value.
Bias         : how well the environment can be decoded from object embeddings.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score


class SegMeter:
    def __init__(self, thr=0.5, bf_tol=2):
        self.thr, self.tol = thr, bf_tol
        self.c = defaultdict(lambda: np.zeros(3))       # tp, fp, fn per dataset
        self.bf = defaultdict(list)

    @torch.no_grad()
    def update(self, prob, target, valid, datasets, boundary=True):
        """prob, target, valid: (B, 1, H, W) at the same resolution."""
        pred = (prob > self.thr) & valid
        tgt = (target > 0.5) & valid
        tp = (pred & tgt).sum((1, 2, 3)).cpu().numpy()
        fp = (pred & ~tgt).sum((1, 2, 3)).cpu().numpy()
        fn = (~pred & tgt).sum((1, 2, 3)).cpu().numpy()
        pn, tn = pred.cpu().numpy(), tgt.cpu().numpy()
        for i, d in enumerate(datasets):
            for key in (d, "_all"):
                self.c[key] += (tp[i], fp[i], fn[i])
            if boundary:
                f = boundary_f1(pn[i, 0], tn[i, 0], self.tol)
                if f is not None:
                    self.bf[d].append(f)
                    self.bf["_all"].append(f)

    def result(self) -> dict:
        out = {}
        for k, (tp, fp, fn) in self.c.items():
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            out[k] = {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * p * r / max(p + r, 1e-9), "precision": p,
                      "recall": r, "bf1": float(np.mean(self.bf[k])) if self.bf[k] else float("nan")}
        return out


def _edges(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol=2):
    pe, ge = _edges(pred), _edges(gt)
    if not pe.any() and not ge.any():
        return None
    if not pe.any() or not ge.any():
        return 0.0
    dg = cv2.distanceTransform((~ge).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~pe).astype(np.uint8), cv2.DIST_L2, 3)
    p = (dg[pe] <= tol).mean()
    r = (dp[ge] <= tol).mean()
    return float(2 * p * r / max(p + r, 1e-9))


def descriptor_map(probs: np.ndarray, tgt: np.ndarray, msk: np.ndarray, schema) -> dict:
    """Per-field mAP over values with at least one positive and one negative, and
    the prevalence floor (expected AP of a random ranking)."""
    out = {}
    for f in schema.fields:
        sl = schema.slices[f.name]
        rows = msk[:, sl].all(1)
        if rows.sum() < 2:
            continue
        P, T = probs[rows, sl], (tgt[rows, sl] > 0).astype(int)
        aps, prev = [], []
        for v in range(T.shape[1]):
            if 0 < T[:, v].sum() < len(T):
                aps.append(average_precision_score(T[:, v], P[:, v]))
                prev.append(T[:, v].mean())
        if aps:
            out[f.name] = {"map": float(np.mean(aps)), "floor": float(np.mean(prev)), "n": int(rows.sum()),
                           "scope": f.scope}
    intr = [v["map"] for v in out.values() if v["scope"] == "intrinsic"]
    flo = [v["floor"] for v in out.values() if v["scope"] == "intrinsic"]
    out["_macro_intrinsic"] = {"map": float(np.mean(intr)) if intr else float("nan"),
                               "floor": float(np.mean(flo)) if flo else float("nan")}
    return out


def gzsl(pred: list, true: list, seen: set, unseen: set, abstained=None) -> dict:
    """Per-class-averaged top-1 on seen and unseen classes and their harmonic mean.
    Abstentions count as errors here; the abstention rate is reported apart."""
    def acc(classes):
        per = []
        for c in classes:
            idx = [i for i, t in enumerate(true) if t == c]
            if idx:
                per.append(np.mean([pred[i] == c for i in idx]))
        return float(np.mean(per)) if per else float("nan")
    s, u = acc(seen), acc(unseen)
    h = 2 * s * u / (s + u) if (s + u) > 0 and not np.isnan(s + u) else float("nan")
    out = {"seen": s, "unseen": u, "H": h, "chance_unseen": 1.0 / max(len(seen | unseen), 1)}
    if abstained is not None:
        out["abstain_rate"] = float(np.mean(abstained))
    return out


def retrieval(scores: np.ndarray, is_pos: np.ndarray, n_boot=2000, n_perm=2000, seed=0) -> dict:
    """AP of one query over a pool, lift over prevalence, bootstrap CI, permutation p."""
    rng = np.random.default_rng(seed)
    if is_pos.sum() == 0:
        return {}
    ap = average_precision_score(is_pos, scores)
    prev = is_pos.mean()
    boots = []
    n = len(scores)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if 0 < is_pos[i].sum() < n:
            boots.append(average_precision_score(is_pos[i], scores[i]))
    perm = [average_precision_score(rng.permutation(is_pos), scores) for _ in range(n_perm)]
    p = (1 + sum(x >= ap for x in perm)) / (1 + n_perm)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"ap": float(ap), "prevalence": float(prev), "lift": float(ap / prev), "ci": [float(lo), float(hi)],
            "p": float(p), "n_pos": int(is_pos.sum())}


def environment_leakage(z: np.ndarray, env: np.ndarray, folds=5, seed=0) -> dict:
    """Cross-validated accuracy of a linear probe decoding the surroundings from
    object embeddings, against the majority-class rate. Lower is better: an
    embedding that describes the object should not tell where it lies."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    keep = env >= 0
    z, env = z[keep], env[keep]
    classes, counts = np.unique(env, return_counts=True)
    if len(classes) < 2 or counts.min() < folds:
        return {}
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(z, env):
        clf = LogisticRegression(max_iter=2000).fit(z[tr], env[tr])
        accs.append((clf.predict(z[te]) == env[te]).mean())
    return {"probe_acc": float(np.mean(accs)), "majority": float(counts.max() / counts.sum()),
            "n": int(len(env))}


def ausuc(fused: np.ndarray, true: list, entry_names: list, seen: set, unseen: set, n=200) -> dict:
    """Area Under the Seen-Unseen accuracy Curve (Chao et al., ECCV 2016).

    Calibrated stacking subtracts gamma from every seen-class score; sweeping
    gamma traces seen vs unseen per-class accuracy. The area is calibration-free,
    so no held-out class is needed to tune anything. Also returns the curve and
    the best harmonic mean along it (an ORACLE value: gamma chosen on test)."""
    idx = [i for i, t in enumerate(true) if t in seen or t in unseen]
    if not idx or not seen or not unseen:
        return {}
    F_ = fused[idx]
    T = [true[i] for i in idx]
    is_seen = np.array([e in seen for e in entry_names])
    span = float(np.nanmax(F_) - np.nanmin(F_)) if np.isfinite(F_).any() else 1.0
    pts = []
    for g in np.linspace(-span, span, n):
        pred = np.argmax(F_ - g * is_seen[None], 1)
        names = [entry_names[j] for j in pred]
        s = np.mean([np.mean([p == c for p, t in zip(names, T) if t == c]) for c in seen if c in T])
        u = np.mean([np.mean([p == c for p, t in zip(names, T) if t == c]) for c in unseen if c in T])
        pts.append((float(s), float(u)))
    pts = sorted(set(pts))
    s_, u_ = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    order = np.argsort(s_)
    area = float(np.trapezoid(u_[order], s_[order])) if len(pts) > 1 else 0.0
    h = [2 * a * b / (a + b) if a + b > 0 else 0.0 for a, b in pts]
    k = int(np.argmax(h))
    return {"ausuc": area, "best_H_oracle": float(h[k]), "at_seen": pts[k][0], "at_unseen": pts[k][1],
            "curve": pts}
