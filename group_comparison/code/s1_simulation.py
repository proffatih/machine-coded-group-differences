#!/usr/bin/env python3
"""Study 1: the estimators against a known truth.

Each replication draws two groups' true topic prevalences, a coder whose confusion
matrix is the same in both groups except in a controlled set of rows, a coded sample
of N items per group, and a hand-coded validation sample of n_v items per group drawn
from it. The estimand is the difference in true prevalence between the groups.

Non-invariance is partial by design: it touches no rows, the three rarest topics, the
three most frequent topics, or every row, with strength delta (the share of a row's
probability that group B redirects to other topics). This is what lets the study ask
which departures matter for a comparison rather than whether any departure exists.

Output: ~/.cache/run_out/cmm_v2_sim_raw.csv (one row per replication x estimator),
        results/sim_summary.csv (condition means)
"""
import csv, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, estimate, tv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
# the per-replication file is large; it stays outside the repository
RAW = os.path.join(os.path.expanduser("~"), ".cache", "run_out", "cmm_v2_sim_raw.csv")
os.makedirs(os.path.dirname(RAW), exist_ok=True)
K, N, R, SEED = 21, 5000, 500, 20261102
BASE = 1 / np.arange(1, K + 1) ** 1.1
BASE /= BASE.sum()                     # CAP-like skew: top topic ~.25, bottom ~.01
ACC = [0.35, 0.55, 0.75]
DELTA = [0.10, 0.20, 0.30]
WHERE = ["rare", "frequent", "all"]
NV = [100, 250, 500, 1000]


def base_confusion(acc, rng):
    C = np.zeros((K, K))
    for i in range(K):
        a = float(np.clip(acc + rng.normal(0, .08), .05, .97))
        off = rng.dirichlet(np.full(K - 1, .3))          # errors go to a few confusers
        C[i, np.arange(K) != i] = (1 - a) * off
        C[i, i] = a
    return C


def perturb(C, rows, delta, rng):
    Cb = C.copy()
    for i in rows:
        w = rng.dirichlet(np.full(K - 1, .5))
        red = np.zeros(K); red[np.arange(K) != i] = w
        Cb[i] = (1 - delta) * C[i] + delta * red
    return Cb


def draw_counts(pi, C, n, rng):
    return rng.multinomial(n, (pi[:, None] * C).ravel()).reshape(K, K)


def subsample(counts, n, rng):
    return rng.multivariate_hypergeometric(counts.ravel(), n).reshape(K, K)


def main():
    rng = np.random.default_rng(SEED)
    conds = [(a, 0.0, "none", nv) for a in ACC for nv in NV]
    conds += [(a, d, w, nv) for a in ACC for d in DELTA for w in WHERE for nv in NV]
    raw = open(RAW, "w", newline="")
    wr = csv.writer(raw)
    wr.writerow(["acc", "delta", "where", "n_v", "rep", "estimator", "tv_error",
                 "focal_error", "focal_covered", "realised_acc"])
    t0 = time.time()
    for ci, (a, d, where, nv) in enumerate(conds):
        for rep in range(R):
            pA = rng.dirichlet(30 * BASE); pB = rng.dirichlet(30 * BASE)
            C = base_confusion(a, rng)
            rows = {"none": [], "rare": list(range(K - 3, K)), "frequent": [0, 1, 2],
                    "all": list(range(K))}[where]
            CA, CB = C, perturb(C, rows, d, rng)
            coded = {"A": draw_counts(pA, CA, N, rng), "B": draw_counts(pB, CB, N, rng)}
            valid = {g: subsample(coded[g], nv, rng) for g in coded}
            truth = pA - pB
            est, se = estimate(coded, valid)
            racc = (np.trace(coded["A"]) + np.trace(coded["B"])) / (2 * N)
            for e in ESTIMATORS:
                err = est[e][0] - truth[0]                # focal topic = most frequent
                cov = ""
                if e in se:
                    cov = int(abs(err) <= 1.96 * se[e][0])
                wr.writerow([a, d, where, nv, rep, e, f"{tv(est[e], truth):.6f}",
                             f"{err:.6f}", cov, f"{racc:.4f}"])
        if ci % 10 == 0:
            print(f"{ci + 1}/{len(conds)} kosul, {time.time() - t0:.0f}s", flush=True)
    raw.close()

    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(RAW)):
        k = (r["acc"], r["delta"], r["where"], r["n_v"], r["estimator"])
        agg[k]["tv"].append(float(r["tv_error"]))
        agg[k]["fe"].append(float(r["focal_error"]))
        if r["focal_covered"] != "":
            agg[k]["cov"].append(int(r["focal_covered"]))
        agg[k]["acc"].append(float(r["realised_acc"]))
    with open(os.path.join(RES, "sim_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acc", "delta", "where", "n_v", "estimator", "mean_tv_error",
                    "focal_bias", "focal_rmse", "focal_coverage", "realised_acc", "reps"])
        for k, v in sorted(agg.items()):
            fe = np.array(v["fe"])
            w.writerow(list(k) + [f"{np.mean(v['tv']):.5f}", f"{fe.mean():.5f}",
                                  f"{np.sqrt((fe ** 2).mean()):.5f}",
                                  f"{np.mean(v['cov']):.4f}" if v["cov"] else "",
                                  f"{np.mean(v['acc']):.4f}", len(fe)])
    print("bitti", time.time() - t0)


if __name__ == "__main__":
    main()
