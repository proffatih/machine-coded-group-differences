#!/usr/bin/env python3
"""Test the algebra behind the pooled-correction result against what actually happens.

For two groups A and B write C_g for each group's confusion matrix (row-stochastic,
estimated on the calibration half), pi_g for the true topic distribution and q_g for the
distribution of assigned codes (both on the test half). With Cbar = (C_A + C_B)/2,
Delta = (C_A - C_B)/2, d = pi_A - pi_B and s = pi_A + pi_B, the derivation gives

    error of the uncorrected difference        = (Cbar' - I) d + Delta' s          (1)
    error of the difference after correcting
      both groups with the pooled matrix Cbar  = (Cbar')^-1 Delta' s               (2)

The derivation assumes the confusion structure seen on the calibration half is the one
that generated the test-half codes. In real data it is only an estimate, so the
derivation can be wrong, and the test is whether it predicts the size of the error that
is actually observed -- not merely its direction. The direction alone is no test here:
equation (2) predicts the pooled correction to be worse in essentially every cell,
so agreement on direction is just the observed share of worse cells.

Three things are reported honestly:
  * cells where Cbar is numerically singular, and why (a topic that is never coded in the
    calibration half of either group in the pair leaves a zero column; one with no calibration items leaves a
    zero row). Equation (2) is undefined there and those cells are not used as evidence;
  * on the well-conditioned cells, the rank correlation between predicted and observed
    error, for the uncorrected difference and for the pooled correction;
  * the observed error both for the exact inverse (what the derivation describes) and for
    non-negative least squares renormalised to a distribution (the practical estimator the
    paper's correction tables use), so the gap between the two is visible.

Usage: python code/07_mechanism.py
Output: results/mechanism.csv, results/metrics/mechanism_summary.json
"""
import glob, itertools, json, os, sys
import numpy as np
from scipy import stats
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "results", "annotations")
OUT = os.path.join(ROOT, "results")
CODES = np.array(VALID_CODES); K = len(CODES)
SV_TOL = 1e-6          # below this Cbar is treated as singular


def conf(y, p):
    m = np.zeros((K, K)); np.add.at(m, (y, p), 1.0)
    r = m.sum(1, keepdims=True)
    return np.divide(m, np.where(r == 0, 1, r)), m


def prev(idx):
    c = np.bincount(idx, minlength=K).astype(float); return c / max(c.sum(), 1)


def tv(v):
    return float(np.abs(v).sum() / 2)


def nnls_correct(Cbar, q):
    x, _ = nnls(Cbar.T, q); s = x.sum()
    return x / s if s > 0 else q


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(ANN, "*.npz"))):
        z = np.load(p, allow_pickle=True); meta = json.loads(str(z["meta"])); n = len(z["gold"])
        out.append(dict(model=meta["model"], condition=meta["condition"], corpus=meta["corpus"],
                        y=np.searchsorted(CODES, z["gold"]), pred=z["probs"].argmax(1),
                        split=z["split"].astype(str), year=z["year"], n_words=z["n_words"],
                        country=np.full(n, CORPORA[meta["corpus"]][3]),
                        venue=np.full(n, CORPORA[meta["corpus"]][4])))
    return out


def main():
    recs = load()
    rows = []
    for model, cond in sorted({(r["model"], r["condition"]) for r in recs}):
        sel = [r for r in recs if r["model"] == model and r["condition"] == cond]
        cat = lambda k: np.concatenate([r[k] for r in sel])
        y, pr, sp = cat("y"), cat("pred"), cat("split")
        nw, yr = cat("n_words"), cat("year")
        med_yr = np.median(yr[yr > 0])
        groupings = {"country": cat("country"), "venue": cat("venue"),
                     "length": np.where(nw >= np.median(nw), "long", "short"),
                     "era": np.where(yr == 0, "unknown", np.where(yr >= med_yr, "later", "earlier"))}
        te, ca = sp == "test", sp == "calib"
        acc = float((y[te] == pr[te]).mean())
        for gname, g in groupings.items():
            levels = [v for v in np.unique(g[te]) if v != "unknown" and (g[te] == v).sum() >= 200]
            for ga, gb in itertools.combinations(levels, 2):
                mA, mB = ca & (g == ga), ca & (g == gb)
                CA, rawA = conf(y[mA], pr[mA]); CB, rawB = conf(y[mB], pr[mB])
                Cbar, Delta = (CA + CB) / 2, (CA - CB) / 2
                tA, tB = te & (g == ga), te & (g == gb)
                piA, piB, qA, qB = prev(y[tA]), prev(y[tB]), prev(pr[tA]), prev(pr[tB])
                d, s = piA - piB, piA + piB
                sv = np.linalg.svd(Cbar, compute_uv=False); smin = float(sv[-1])
                zero_rows = int(((rawA.sum(1) == 0) | (rawB.sum(1) == 0)).sum())
                zero_cols = int(((Cbar.sum(0)) == 0).sum())
                well = smin > SV_TOL
                rec = dict(model=model, condition=cond, grouping=gname, pair=f"{ga}|{gb}",
                           accuracy=acc, sigma_min=smin, well_conditioned=well,
                           zero_rows=zero_rows, zero_cols=zero_cols,
                           pred_naive=tv((Cbar.T - np.eye(K)) @ d + Delta.T @ s),
                           obs_naive=tv((qA - qB) - d))
                # practical estimator, defined in every cell
                rec["obs_pooled_nnls"] = tv(nnls_correct(Cbar, qA) - nnls_correct(Cbar, qB) - d)
                if well:
                    inv = np.linalg.solve
                    rec["amplification"] = 1.0 / smin
                    rec["pred_pooled"] = tv(inv(Cbar.T, Delta.T @ s))
                    rec["obs_pooled_exact"] = tv(inv(Cbar.T, qA) - inv(Cbar.T, qB) - d)
                rows.append(rec)

    well = [r for r in rows if r["well_conditioned"]]
    sing = [r for r in rows if not r["well_conditioned"]]

    def rho(a, b):
        r, p = stats.spearmanr(a, b); return float(r), float(p)

    pn = rho([r["pred_naive"] for r in well], [r["obs_naive"] for r in well])
    pp = rho([r["pred_pooled"] for r in well], [r["obs_pooled_exact"] for r in well])
    pe = rho([r["pred_pooled"] - r["pred_naive"] for r in well],
             [r["obs_pooled_exact"] - r["obs_naive"] for r in well])
    pnn = rho([r["pred_pooled"] for r in well], [r["obs_pooled_nnls"] for r in well])
    amp = [r["amplification"] for r in well]
    summary = {
        "n_cells": len(rows),
        "n_singular": len(sing),
        "singular_with_zero_columns": sum(1 for r in sing if r["zero_cols"] > 0),
        "singular_with_zero_rows": sum(1 for r in sing if r["zero_rows"] > 0),
        "n_well_conditioned": len(well),
        "sv_tolerance": SV_TOL,
        "share_predicted_pooled_worse_well": float(np.mean([r["pred_pooled"] > r["pred_naive"] for r in well])),
        "share_observed_pooled_worse_exact_well": float(np.mean([r["obs_pooled_exact"] > r["obs_naive"] for r in well])),
        "share_observed_pooled_worse_nnls_all": float(np.mean([r["obs_pooled_nnls"] > r["obs_naive"] for r in rows])),
        "spearman_naive_pred_vs_obs": pn,
        "spearman_pooled_pred_vs_obs_exact": pp,
        "spearman_excess_pred_vs_obs_exact": pe,
        "spearman_pooled_pred_vs_obs_nnls": pnn,
        "median_abs_rel_error_pooled": float(np.median([abs(r["pred_pooled"] - r["obs_pooled_exact"]) / max(r["obs_pooled_exact"], 1e-9) for r in well])),
        "median_amplification_well": float(np.median(amp)),
        "spearman_log_amplification_vs_accuracy": rho(np.log(amp), [r["accuracy"] for r in well]),
        "models_in_well": sorted({r["model"] for r in well}),
    }
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "mechanism.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "metrics", "mechanism_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
