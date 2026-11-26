#!/usr/bin/env python3
"""Do the results depend on how the language models were scored, or on their weakness?
(Online Appendix G)

(1) Prior-corrected scores. The language models' topic probabilities come from the first
    token of each topic name, renormalised over the 21 topics, and some topics are almost
    never ranked first. Dividing each topic's probability by its mean over the corpus (a
    label-free correction in the spirit of contextual calibration) and taking the largest
    gives a recalibrated code for every item.
(2) Stronger coders. Synthetic coders that return the human code with probability s and
    otherwise the in-corpus logistic regression's code, with s = .5 and .8 (accuracy of about
    .8 and .9), stand in for coders far better than any used here.

For each new coder: accuracy, topics never assigned, and the Study 2 comparisons (period
halves and pairs of corpora) with 100 validation items per group, R draws.

Output: results/coders_ext.csv (coder x comparison x estimator), results/coders_ext_quality.csv
"""
import csv, itertools, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import estimate, tv, ESTIMATORS, WITH_SE
from s2_real_coders import load, counts, halves, SEED, K
from s4_validity import load_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
R = 100


def build():
    sh = load_probs()
    corpora = sorted({k[2] for k in sh})
    rng = np.random.default_rng(SEED + 909)
    new = {}
    for (m, cond, c), d in sh.items():
        if m.startswith("tfidf"):
            continue
        p = d["probs"] / np.maximum(d["probs"].mean(0, keepdims=True), 1e-12)
        new[(m + "+prior", cond, c)] = {"y": d["y"], "pred": p.argmax(1), "year": d["year"],
                                        "orig_pred": d["pred"]}
    for c in corpora:
        base = sh[("tfidf-logreg", "sup", c)]
        for s in (.5, .8):
            keep = rng.random(len(base["y"])) < s
            new[(f"synthetic-{s}", "sup", c)] = {"y": base["y"], "pred": np.where(keep, base["y"], base["pred"]),
                                                 "year": base["year"], "orig_pred": base["pred"]}
    return new, corpora


def job(args):
    (m, cond), seed = args
    new, corpora = build()
    rng = np.random.default_rng(seed)
    rows, qrows = [], []
    split = {}
    for ci, corpus in enumerate(corpora):
        d = new[next(k for k in new if k[2] == corpus)]
        split[corpus] = halves(d["year"].astype(float), SEED + ci)
    comps = []
    for c in corpora:
        d = new[(m, cond, c)]; g = split[c]
        unassigned = int(((np.bincount(d["pred"], minlength=K) == 0) & (np.bincount(d["y"], minlength=K) > 0)).sum())
        unassigned_o = int(((np.bincount(d["orig_pred"], minlength=K) == 0) & (np.bincount(d["y"], minlength=K) > 0)).sum())
        qrows.append({"model": m, "condition": cond, "corpus": c, "accuracy": float((d["pred"] == d["y"]).mean()),
                      "accuracy_original": float((d["orig_pred"] == d["y"]).mean()),
                      "unassigned": unassigned, "unassigned_original": unassigned_o})
        comps.append(("within", f"{c}:period", counts(d["y"][~g], d["pred"][~g]), counts(d["y"][g], d["pred"][g])))
    for ca, cb in itertools.combinations(corpora, 2):
        a, b = new[(m, cond, ca)], new[(m, cond, cb)]
        comps.append(("between", f"{ca}|{cb}", counts(a["y"], a["pred"]), counts(b["y"], b["pred"])))
    for design, comp, cA, cB in comps:
        truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
        prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
        focal = int(np.argmax(prev))
        res = {e: {"tv": [], "fe": [], "cf": []} for e in ESTIMATORS}
        for _ in range(R):
            valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), 100).reshape(K, K),
                     "B": rng.multivariate_hypergeometric(cB.ravel(), 100).reshape(K, K)}
            est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
            for e in ESTIMATORS:
                res[e]["tv"].append(tv(est[e], truth)); res[e]["fe"].append(est[e][focal] - truth[focal])
                if e in WITH_SE:
                    res[e]["cf"].append(abs(est[e][focal] - truth[focal]) <= 1.96 * se[e][focal])
        for e in ESTIMATORS:
            fe = np.array(res[e]["fe"])
            rows.append({"model": m, "condition": cond, "design": design, "comparison": comp, "estimator": e,
                         "mean_tv_error": float(np.mean(res[e]["tv"])), "focal_rmse": float(np.sqrt((fe ** 2).mean())),
                         "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else ""})
    print(f"{m}/{cond}", flush=True)
    return rows, qrows


def main():
    t0 = time.time()
    new, _ = build()
    coders = sorted({(m, c) for m, c, _ in new})
    jobs = [(mc, SEED + 909 + 1000 * i) for i, mc in enumerate(coders)]
    with Pool(min(14, len(jobs))) as p:
        parts = p.map(job, jobs)
    for name, i in (("coders_ext.csv", 0), ("coders_ext_quality.csv", 1)):
        rows = [r for part in parts for r in part[i]]
        with open(os.path.join(RES, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("bitti", round(time.time() - t0))


if __name__ == "__main__":
    main()
