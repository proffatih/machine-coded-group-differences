"""Estimators of a between-group difference in category prevalence from machine codes.

Everything works on K x K count tables n[true, predicted]:
  coded[g]  the coded items of group g (true labels unknown to the researcher;
            only its column sums are used)
  valid[g]  the hand-coded validation items of group g, a simple random subsample of
            the coded items (true and predicted known)

Estimators of the prevalence vector of group g (the difference is A minus B):
  naive        share of machine codes
  bc_pooled    matrix back-calculation (Bachl & Scharkow, 2017) with one confusion
               matrix estimated from both groups' validation items, solved by NNLS
  bc_group     the same with each group's own confusion matrix
  rg_group     per-category Rogan-Gladen correction with each group's own sensitivity
               and false-positive rate; for two groups this is what a misclassification
               model with the group in the error model reduces to (Greenland, 1988;
               TeBlunthuis et al., 2024)
  cal_group    calibration ("inverse matrix") estimator: machine-code shares weighted by
               P(true | predicted) estimated in the group's own validation items, i.e.
               post-stratification on the predicted category (Morrissey & Spiegelman, 1999)
  dd_group     difference estimator: machine-code share plus the group's own mean
               validation residual (two-phase difference estimator; prediction-powered
               inference)
  dd_tuned     the same with the machine-code term weighted by the per-category
               regression coefficient lambda in [0, 1] (power tuning, as in PPI++);
               lambda = 0 is hand coding, lambda = 1 is dd_group
  hand         validation items alone (hand coding, the coder unused)

Confusion-matrix rows with no validation items are set to the identity row (the coder
is taken at its word for a category it was never checked on), and calibration columns
with no validation items likewise; a zero row would force that category's prevalence
to zero.

Variances follow two-phase sampling with the validation items drawn from the coded
items (Saerndal et al., 1992, ch. 9): for the tuned estimator
    Var = S_y^2 / N + (1/n - 1/N) S_{y - lambda p}^2,
whose minimiser is lambda = Cov(y, p) / Var(p). With finite=True the estimand is the
prevalence among the coded items themselves, so the first term drops.
"""
import numpy as np
from scipy.optimize import nnls

ESTIMATORS = ["naive", "bc_pooled", "bc_group", "rg_group", "cal_group",
              "dd_group", "dd_tuned", "hand"]
WITH_SE = ["naive", "rg_group", "cal_group", "dd_group", "dd_tuned", "hand"]


def row_norm(n):
    """Row-normalise counts; empty rows become identity rows."""
    r = n.sum(1, keepdims=True)
    out = np.divide(n, np.where(r == 0, 1, r)).astype(float)
    empty = np.where(r[:, 0] == 0)[0]
    out[empty, empty] = 1.0
    return out


def back_calc(q, conf_counts):
    """Solve C^T x = q on the simplex by NNLS; C estimated from counts."""
    x, _ = nnls(row_norm(conf_counts).T, q)
    s = x.sum()
    return x / s if s > 0 else q


def rogan_gladen(q, v, nN, finite, a=0.5):
    """Per-category binary correction; falls back to q where the coder is not
    informative in the validation items (sensitivity minus false-positive rate < .05).
    Delta-method variance (Greenland, 1988) from rates with a added to each cell."""
    n = v.sum()
    pos = v.sum(1)
    tp = np.diag(v).astype(float)
    fp = v.sum(0) - tp
    neg = n - pos
    sens = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    fpr = np.divide(fp, neg, out=np.zeros_like(tp), where=neg > 0)
    den = sens - fpr
    ok = (pos > 0) & (den > .05)
    est = np.where(ok, np.clip((q - fpr) / np.where(ok, den, 1), 0, 1), q)
    ss, fs = (tp + a) / (pos + 2 * a), (fp + a) / (neg + 2 * a)
    var_q = 0.0 if finite else q * (1 - q) / nN
    var = (var_q + est ** 2 * ss * (1 - ss) / (pos + 2 * a)
           + (1 - est) ** 2 * fs * (1 - fs) / (neg + 2 * a)) / np.where(ok, den, 1) ** 2
    return est, np.where(ok, var, var_q + (tp + a) / (n + 2 * a) / (n + 2 * a))


def calibration(q, v, nN, finite, a=0.1):
    """sum_j q_j P(true = k | pred = j); variance conditional on the validation counts
    per predicted category, from rates with a added to each cell."""
    col = v.sum(0)
    P = np.divide(v, np.where(col == 0, 1, col)[None, :]).astype(float)   # P[k, j]
    empty = np.where(col == 0)[0]
    P[empty, empty] = 1.0
    est = P @ q
    Ps = (v + a) / (col + 2 * a)[None, :]
    fpc = 1 - col / np.maximum(np.rint(nN * q), 1) if finite else 1.0
    var = (Ps * (1 - Ps) * fpc / (col + 2 * a)[None, :]) @ (q ** 2)
    if not finite:
        var = var + est * (1 - est) / nN
    return est, var


def _moments(v, a=0.5):
    """Validation moments per category from its 2 x 2 table (true k / predicted k).
    Variances and lambda use the table with a added to each cell, so that a category
    seldom or never seen in the validation items does not get a zero standard error
    (cf. Agresti & Caffo, 2000); point estimates use the raw shares."""
    n = v.sum()
    p_y, p_p, p_both = v.sum(1) / n, v.sum(0) / n, np.diag(v) / n
    t11 = np.diag(v) + a
    t10 = v.sum(1) - np.diag(v) + a
    t01 = v.sum(0) - np.diag(v) + a
    t00 = n - v.sum(1) - v.sum(0) + np.diag(v) + a
    m = t11 + t10 + t01 + t00
    sy, sp, sb = (t11 + t10) / m, (t11 + t01) / m, t11 / m
    c = n / max(n - 1, 1)
    return n, p_y, p_p, sy * (1 - sy) * c, sp * (1 - sp) * c, (sb - sy * sp) * c


def estimate(coded, valid, finite=False):
    """coded, valid: dict g -> K x K counts, g in ('A','B'). Returns (difference
    vectors, standard errors). finite=True: estimand is the prevalence among the coded
    items themselves (Study 2)."""
    q = {g: coded[g].sum(0) / coded[g].sum() for g in coded}
    vp = valid["A"] + valid["B"]
    prev = {e: {} for e in ESTIMATORS}
    K = coded["A"].shape[0]
    var = {e: np.zeros(K) for e in WITH_SE}
    for g in ("A", "B"):
        v, N = valid[g], coded[g].sum()
        n, p_y, p_p, vy, vpred, cov = _moments(v)
        lam = np.clip(np.divide(cov, vpred, out=np.zeros_like(cov), where=vpred > 0), 0, 1)
        f2 = 1 / n - 1 / N                       # second-phase factor
        f1 = 0.0 if finite else 1 / N            # first-phase factor
        prev["naive"][g] = q[g]
        prev["bc_pooled"][g] = back_calc(q[g], vp)
        prev["bc_group"][g] = back_calc(q[g], v)
        prev["rg_group"][g], v_rg = rogan_gladen(q[g], v, N, finite)
        prev["cal_group"][g], v_cal = calibration(q[g], v, N, finite)
        prev["dd_group"][g] = q[g] + (p_y - p_p)
        prev["dd_tuned"][g] = lam * q[g] + (p_y - lam * p_p)
        prev["hand"][g] = p_y
        var_e1 = vy + vpred - 2 * cov                       # S^2 of y - p
        # lambda is estimated from the same items: one more degree of freedom
        var_el = (vy + lam ** 2 * vpred - 2 * lam * cov) * (n - 1) / max(n - 2, 1)
        var["naive"] += q[g] * (1 - q[g]) / N              # the interval a researcher reports
        var["rg_group"] += v_rg
        var["cal_group"] += v_cal
        var["dd_group"] += f1 * vy + f2 * var_e1
        var["dd_tuned"] += f1 * vy + f2 * var_el
        var["hand"] += f1 * vy + f2 * vy
    out = {e: prev[e]["A"] - prev[e]["B"] for e in ESTIMATORS}
    se = {e: np.sqrt(np.maximum(v_, 0)) for e, v_ in var.items()}
    return out, se


def decompose_naive(cA, cB):
    """Exact split of the uncorrected error with population confusion matrices:
    q_A - q_B - d = (Cbar^T - I) d  +  sum_i omega_i (C_A[i] - C_B[i])^T,
    with Cbar the count-weighted pooled matrix. Returns (shared, differential)."""
    nA, nB = cA.sum(), cB.sum()
    piA, piB = cA.sum(1) / nA, cB.sum(1) / nB
    CA, CB, Cbar = row_norm(cA), row_norm(cB), row_norm(cA + cB)
    d = piA - piB
    naive_err = CA.T @ piA - CB.T @ piB - d
    shared = Cbar.T @ d - d
    return shared, naive_err - shared


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())
