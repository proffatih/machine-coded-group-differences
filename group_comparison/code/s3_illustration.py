#!/usr/bin/env python3
"""An agenda comparison worked through: U.S. Congressional bills against New York Times
front-page headlines, coded by the best zero-shot language model and by a supervised
classifier. One validation draw of 200 items per corpus (fixed seed) is used, as a
researcher would have; the table reports, for the five topics most prevalent across the
two corpora, the human-coded difference, the uncorrected estimate, and the tuned
difference estimate with 95% intervals.

Also computes the validation size per group needed for a target standard error of the
difference for one topic, from the closed-form variance, for hand coding alone and for the
tuned estimator, using each coder's observed residual variances (planning rule).

Output: results/illustration.csv, results/planning.csv
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import estimate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ANN = os.path.join(os.path.dirname(ROOT), "results", "annotations")
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "code"))
from cap_codebook import VALID_CODES
CODES = np.array(VALID_CODES); K = len(CODES)
NAMES = {1: "Macroeconomics", 2: "Civil rights", 3: "Health", 4: "Agriculture", 5: "Labour", 6: "Education",
         7: "Environment", 8: "Energy", 9: "Immigration", 10: "Transportation", 12: "Law and crime",
         13: "Social welfare", 14: "Housing", 15: "Banking and commerce", 16: "Defence", 17: "Science and technology",
         18: "Foreign trade", 19: "International affairs", 20: "Government operations", 21: "Public lands", 23: "Culture"}


def counts(model, cond, corpus):
    z = np.load(os.path.join(ANN, f"{model}__{cond}__{corpus}.npz"), allow_pickle=True)
    y = np.searchsorted(CODES, z["gold"]); p = z["probs"].argmax(1)
    c = np.zeros((K, K), dtype=np.int64); np.add.at(c, (y, p), 1)
    return c


def main():
    rng = np.random.default_rng(20261102)
    rows = []
    for model, cond, label in (("qwen2.5-3b", "zs", "Qwen2.5 3B, zero-shot"),
                               ("tfidf-logreg", "sup", "TF-IDF + logistic regression")):
        cA, cB = counts(model, cond, "us_bills"), counts(model, cond, "us_nyt")
        truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
        valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), 200).reshape(K, K),
                 "B": rng.multivariate_hypergeometric(cB.ravel(), 200).reshape(K, K)}
        est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
        top = np.argsort(-(cA.sum(1) / cA.sum() + cB.sum(1) / cB.sum()))[:5]
        for k in top:
            rows.append({"coder": label, "topic": NAMES[int(CODES[k])], "true": truth[k],
                         "naive": est["naive"][k], "tuned": est["dd_tuned"][k], "tuned_se": se["dd_tuned"][k],
                         "naive_covers": False,
                         "tuned_covers": abs(est["dd_tuned"][k] - truth[k]) <= 1.96 * se["dd_tuned"][k]})
    with open(os.path.join(RES, "illustration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # planning: n_v per group so that SE(difference) <= target, for the most prevalent topic of each
    # between-corpus pair, using population residual variances (all items known here)
    plan = []
    for target in (0.05, 0.03, 0.02):
        for model, cond in (("qwen2.5-3b", "zs"), ("phi3.5-mini", "zs"), ("tfidf-logreg", "sup"), ("tfidf-svm", "sup")):
            ratios, hands = [], []
            corp = sorted({os.path.basename(p).split("__")[2][:-4] for p in glob.glob(os.path.join(ANN, f"{model}__{cond}__*.npz"))})
            for i in range(len(corp)):
                for j in range(i + 1, len(corp)):
                    var_h, var_t = 0.0, 0.0
                    cs = [counts(model, cond, corp[i]), counts(model, cond, corp[j])]
                    k = int(np.argmax(cs[0].sum(1) / cs[0].sum() + cs[1].sum(1) / cs[1].sum()))
                    for c in cs:
                        n = c.sum(); py = c.sum(1)[k] / n; pp = c.sum(0)[k] / n; pb = c[k, k] / n
                        vy, vp, cov = py * (1 - py), pp * (1 - pp), pb - py * pp
                        lam = np.clip(cov / vp, 0, 1) if vp > 0 else 0.0
                        var_h += vy; var_t += vy + lam ** 2 * vp - 2 * lam * cov
                    hands.append(var_h / target ** 2); ratios.append(var_t / target ** 2)
            plan.append({"target_se": target, "coder": f"{model}/{cond}",
                         "n_hand_median": float(np.median(hands)), "n_tuned_median": float(np.median(ratios))})
    with open(os.path.join(RES, "planning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0])); w.writeheader(); w.writerows(plan)
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for p in plan: print(p)


if __name__ == "__main__":
    main()
