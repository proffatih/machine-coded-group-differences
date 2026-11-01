"""Estimators of a between-group difference in category prevalence from machine codes.

Everything works on K x K count tables n[true, predicted]:
  coded[g]  the coded sample of group g (true labels unknown to the researcher;
            only its column sums are used)
  valid[g]  the hand-coded validation sample of group g (true and predicted known)

Estimators of the prevalence vector of group g (the difference is A minus B):
  naive        share of machine codes
  bc_pooled    matrix back-calculation with one confusion matrix from both groups'
               validation items (Bachl & Scharkow, 2017), solved by NNLS
  bc_group     the same with each group's own confusion matrix
  dd_pooled    difference (prediction-powered) estimator with the residual averaged
               over both groups' validation items
  dd_group     difference estimator with each group's own validation residual
  dd_tuned     the same with the machine-code term weighted by a per-topic factor
               lambda in [0, 1] chosen to minimise variance (power tuning, as in PPI++);
               lambda = 0 is hand coding, lambda = 1 is dd_group
  hand         validation sample alone (hand coding, the coder unused)
"""
import numpy as np
from scipy.optimize import nnls

ESTIMATORS = ["naive", "bc_pooled", "bc_group", "dd_pooled", "dd_group", "dd_tuned", "hand"]


def row_norm(n):
    r = n.sum(1, keepdims=True)
    return np.divide(n, np.where(r == 0, 1, r))


def back_calc(q, conf_counts):
    """Solve C^T x = q on the simplex by NNLS; C estimated from counts."""
    x, _ = nnls(row_norm(conf_counts).T, q)
    s = x.sum()
    return x / s if s > 0 else q


def _moments(v):
    nv = v.sum()
    p_y, p_p, p_both = v.sum(1) / nv, v.sum(0) / nv, np.diag(v) / nv
    return nv, p_y, p_p, p_both


def estimate(coded, valid, finite=False):
    """coded, valid: dict g -> K x K counts, g in ('A','B'). Returns (diff vectors,
    standard errors). With finite=True the estimand is the prevalence among the coded
    items themselves and the validation items are a subsample of them, so the
    machine-code share carries no sampling error and a finite-population correction
    applies to the validation term."""
    q = {g: coded[g].sum(0) / coded[g].sum() for g in coded}
    vp = valid["A"] + valid["B"]
    prev = {e: {} for e in ESTIMATORS}
    K = coded["A"].shape[0]
    var = {e: np.zeros(K) for e in ("naive", "dd_group", "dd_tuned", "hand")}
    for g in ("A", "B"):
        v, N = valid[g], coded[g].sum()
        nv, p_y, p_p, p_both = _moments(v)
        fpc = (1 - nv / N) if finite else 1.0
        vy, vpred = p_y * (1 - p_y), p_p * (1 - p_p)
        cov = p_both - p_y * p_p
        ratio = 0.0 if finite else nv / N
        lam = np.clip(np.divide(cov, (1 + ratio) * vpred, out=np.zeros_like(cov),
                                where=vpred > 0), 0, 1)
        prev["naive"][g] = q[g]
        prev["bc_pooled"][g] = back_calc(q[g], vp)
        prev["bc_group"][g] = back_calc(q[g], v)
        prev["dd_pooled"][g] = q[g] + (vp.sum(1) - vp.sum(0)) / vp.sum()
        prev["dd_group"][g] = q[g] + (p_y - p_p)
        prev["dd_tuned"][g] = lam * q[g] + (p_y - lam * p_p)
        prev["hand"][g] = p_y
        c = nv / max(nv - 1, 1)
        var_q = 0.0 if finite else q[g] * (1 - q[g]) / N
        var_e1 = (vy + vpred - 2 * cov) * c            # Var(y - p)
        var_el = (vy + lam ** 2 * vpred - 2 * lam * cov) * c
        var["naive"] = var["naive"] + var_q
        var["dd_group"] = var["dd_group"] + var_q + fpc * var_e1 / nv
        var["dd_tuned"] = var["dd_tuned"] + lam ** 2 * var_q + fpc * var_el / nv
        var["hand"] = var["hand"] + fpc * vy * c / nv
    out = {e: prev[e]["A"] - prev[e]["B"] for e in ESTIMATORS}
    se = {e: np.sqrt(np.maximum(v_, 0)) for e, v_ in var.items()}
    return out, se


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())
