#!/usr/bin/env python3
"""An agenda comparison worked through, and validation-sample planning.

Illustration: U.S. Congressional bill titles against New York Times front-page headlines,
coded by the best zero-shot language model (Qwen2.5 3B) and by the classifier trained on
the five corpora outside the pair (no labels from either corpus). One validation draw of 200 items per
corpus (fixed seed) is used, as a researcher would have. For the five topics most
prevalent across the two corpora the table reports the human-coded difference, the
uncorrected estimate, pooled back-calculation, the calibration estimator, the tuned
difference estimate, and the estimated bias of the uncorrected difference (the difference
between the groups' mean validation residuals), each tuned or bias estimate with a 95%
interval.

Planning: the validation size per group needed for a target standard error of a
difference, from the variance of the tuned estimator for a large coded corpus,
n_g = pi_g (1 - pi_g)(1 - rho_g^2) / SE_g^2, where rho_g is the correlation between the
human and the machine indicator of the category in group g. Two outputs: a grid over pi
and rho, and the values implied by the real coders' focal-topic correlations.

Output: results/illustration.csv, results/planning_grid.csv, results/planning_real.csv
"""
import csv, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import estimate
from s2_real_coders import load, counts, CODES, K, TRANSFER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
NAMES = {1: "Macroeconomics", 2: "Civil rights", 3: "Health", 4: "Agriculture", 5: "Labour", 6: "Education",
         7: "Environment", 8: "Energy", 9: "Immigration", 10: "Transportation", 12: "Law and crime",
         13: "Social welfare", 14: "Housing", 15: "Banking and commerce", 16: "Defence", 17: "Science and technology",
         18: "Foreign trade", 19: "International affairs", 20: "Government operations", 21: "Public lands", 23: "Culture"}
CODERS = (("qwen2.5-3b", "zs", "Qwen2.5 3B, zero-shot"),
          ("tfidf-logreg-transfer", "sup", "Classifier trained on the other five corpora"))


def phi(c, k):
    """Correlation of the human and machine indicators of category k, and its prevalence."""
    n = c.sum(); py = c.sum(1)[k] / n; pp = c.sum(0)[k] / n; pb = c[k, k] / n
    den = np.sqrt(py * (1 - py) * pp * (1 - pp))
    return (pb - py * pp) / den if den > 0 else 0.0, py, pp


def main():
    sh, pairs = load(with_pairs=True)
    shard = lambda m, c, corpus, pair: (pairs[pair][corpus] if m == TRANSFER else sh[(m, c, corpus)])
    rng = np.random.default_rng(20261102)
    rows = []
    for model, cond, label in CODERS:
        pair = tuple(sorted(("us_bills", "us_nyt")))
        a, b = shard(model, cond, "us_bills", pair), shard(model, cond, "us_nyt", pair)
        cA, cB = counts(a["y"], a["pred"]), counts(b["y"], b["pred"])
        truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
        valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), 200).reshape(K, K),
                 "B": rng.multivariate_hypergeometric(cB.ravel(), 200).reshape(K, K)}
        est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
        bias = est["dd_group"] - est["naive"]          # minus the estimated bias of naive
        top = np.argsort(-(cA.sum(1) / cA.sum() + cB.sum(1) / cB.sum()))[:5]
        for k in top:
            rows.append({"coder": label, "topic": NAMES[int(CODES[k])], "true": truth[k],
                         "naive": est["naive"][k], "bc_pooled": est["bc_pooled"][k],
                         "calibration": est["cal_group"][k],
                         "tuned": est["dd_tuned"][k], "tuned_se": se["dd_tuned"][k],
                         "naive_bias_est": -bias[k], "naive_bias_se": se["dd_group"][k],
                         "naive_bias_true": est["naive"][k] - truth[k],
                         "tuned_covers": bool(abs(est["dd_tuned"][k] - truth[k]) <= 1.96 * se["dd_tuned"][k])})
    with open(os.path.join(RES, "illustration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    grid = []
    for se_t in (0.03, 0.02):
        for p in (0.05, 0.10, 0.20, 0.30):
            for r in (0.0, 0.5, 0.7, 0.9):
                # equal prevalence and correlation in both groups, equal allocation:
                # SE^2 = 2 p(1-p)(1-r^2)/n
                grid.append({"target_se": se_t, "prevalence": p, "rho": r,
                             "n_per_group": int(np.ceil(2 * p * (1 - p) * (1 - r ** 2) / se_t ** 2))})
    with open(os.path.join(RES, "planning_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

    real = []
    corpora = sorted({k[2] for k in sh})
    coders = sorted({(m, c) for m, c, _ in sh})
    for m, cnd in coders:
        hand, tuned, rhos = [], [], []
        for ca, cb in itertools.combinations(corpora, 2):
            cs = [counts(shard(m, cnd, x, (ca, cb))["y"], shard(m, cnd, x, (ca, cb))["pred"]) for x in (ca, cb)]
            k = int(np.argmax(cs[0].sum(1) / cs[0].sum() + cs[1].sum(1) / cs[1].sum()))
            vh = vt = 0.0
            for c in cs:
                r, py, _ = phi(c, k)
                vh += py * (1 - py); vt += py * (1 - py) * (1 - max(r, 0) ** 2); rhos.append(r)
            hand.append(vh); tuned.append(vt)
        for se_t in (0.03, 0.02):
            real.append({"model": m, "condition": cnd, "target_se": se_t,
                         "median_rho": float(np.median(rhos)),
                         "n_hand_median": float(np.median(hand) / se_t ** 2),
                         "n_tuned_median": float(np.median(tuned) / se_t ** 2)})
    with open(os.path.join(RES, "planning_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(real[0])); w.writeheader(); w.writerows(real)
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for r in real:
        if r["target_se"] == 0.03: print(r)


if __name__ == "__main__":
    main()
