#!/usr/bin/env python3
"""Study 1: the estimators against a known truth.

Each replication draws two groups' true topic prevalences, a coder whose confusion
matrix is the same in both groups except in a controlled set of rows, a coded sample
of N items per group, and a hand-coded validation sample of n_v items per group drawn
from it. The estimand is the difference in true prevalence between the groups.

A second validation design draws 2 n_v items from both groups' coded items together
(pooled) and analyses them by group, as a researcher with one random validation sample
would.

Differential error is partial by design: it touches no rows, the three rarest topics, the
three most frequent topics, or every row, with strength delta (the share of a row's
probability that group B redirects to other topics). This is what lets the study ask
which departures matter for a comparison rather than whether any departure exists.

Output: ~/.cache/run_out/cmm_v2_sim_raw.csv (one row per replication x estimator),
        results/sim_summary.csv (condition means)
"""
import csv, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, WITH_SE, estimate, tv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
# the per-replication file is large; it stays outside the repository
RAW = os.path.join(os.path.expanduser("~"), ".cache", "run_out", "cmm_v2_sim_raw.csv")
os.makedirs(os.path.dirname(RAW), exist_ok=True)
K, N, R, SEED = 21, 5000, 500, 20261102
BASE = 1 / np.arange(1, K + 1) ** 1.1
BASE /= BASE.sum()                     # CAP-like skew: top topic ~.31, bottom ~.01
ACC = [0.35, 0.55, 0.75]
DELTA = [0.10, 0.20, 0.30]
WHERE = ["rare", "frequent", "all"]
NV = [50, 100, 250, 500, 1000]
RARE = [K - 3, K - 2, K - 1]


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
    wr.writerow(["acc", "delta", "where", "n_v", "rep", "sampling", "estimator", "tv_error",
                 "focal_error", "focal_truth", "focal_covered", "allcat_covered",
                 "rare_covered", "rare_error", "rare_truth", "realised_acc"])
    t0 = time.time()
    for ci, (a, d, where, nv) in enumerate(conds):
        for rep in range(R):
            pA = rng.dirichlet(30 * BASE); pB = rng.dirichlet(30 * BASE)
            C = base_confusion(a, rng)
            rows = {"none": [], "rare": RARE, "frequent": [0, 1, 2],
                    "all": list(range(K))}[where]
            CA, CB = C, perturb(C, rows, d, rng)
            coded = {"A": draw_counts(pA, CA, N, rng), "B": draw_counts(pB, CB, N, rng)}
            truth = pA - pB
            racc = (np.trace(coded["A"]) + np.trace(coded["B"])) / (2 * N)
            strat = {g: subsample(coded[g], nv, rng) for g in coded}
            both = rng.multivariate_hypergeometric(
                np.concatenate([coded["A"], coded["B"]]).ravel(), 2 * nv).reshape(2 * K, K)
            pooled = {"A": both[:K], "B": both[K:]}
            for sampling, valid, ests in (("stratified", strat, ESTIMATORS),
                                          ("pooled", pooled, ["cal_group", "dd_group", "dd_tuned", "hand"])):
                est, se = estimate(coded, valid)
                for e in ests:
                    err = est[e] - truth
                    cf = ca = cr = ""
                    if e in se:
                        hit = np.abs(err) <= 1.96 * se[e]
                        cf, ca, cr = int(hit[0]), f"{hit.mean():.4f}", f"{hit[RARE].mean():.4f}"
                    wr.writerow([a, d, where, nv, rep, sampling, e, f"{tv(est[e], truth):.6f}",
                                 f"{err[0]:.6f}", f"{truth[0]:.6f}", cf, ca, cr,
                                 f"{err[K - 1]:.6f}", f"{truth[K - 1]:.6f}", f"{racc:.4f}"])
        if ci % 10 == 0:
            print(f"{ci + 1}/{len(conds)} kosul, {time.time() - t0:.0f}s", flush=True)
    raw.close()

    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(RAW)):
        k = (r["acc"], r["delta"], r["where"], r["n_v"], r["sampling"], r["estimator"])
        for f in ("tv_error", "focal_error", "focal_truth", "rare_error", "realised_acc"):
            agg[k][f].append(float(r[f]))
        if r["focal_covered"] != "":
            agg[k]["cf"].append(int(r["focal_covered"]))
            agg[k]["ca"].append(float(r["allcat_covered"]))
            agg[k]["cr"].append(float(r["rare_covered"]))
    with open(os.path.join(RES, "sim_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acc", "delta", "where", "n_v", "sampling", "estimator", "mean_tv_error",
                    "focal_bias", "focal_rmse", "focal_slope", "rare_rmse", "focal_coverage",
                    "focal_coverage_mcse", "allcat_coverage", "rare_coverage",
                    "realised_acc", "reps"])
        for k, v in sorted(agg.items()):
            fe, ft = np.array(v["focal_error"]), np.array(v["focal_truth"])
            slope = np.polyfit(ft, fe + ft, 1)[0]          # estimate on truth; 1 = no attenuation
            cf = np.mean(v["cf"]) if v["cf"] else None
            w.writerow(list(k) + [f"{np.mean(v['tv_error']):.5f}", f"{fe.mean():.5f}",
                                  f"{np.sqrt((fe ** 2).mean()):.5f}", f"{slope:.4f}",
                                  f"{np.sqrt(np.mean(np.square(v['rare_error']))):.5f}",
                                  f"{cf:.4f}" if cf is not None else "",
                                  f"{np.sqrt(cf * (1 - cf) / len(v['cf'])):.4f}" if cf is not None else "",
                                  f"{np.mean(v['ca']):.4f}" if v["ca"] else "",
                                  f"{np.mean(v['cr']):.4f}" if v["cr"] else "",
                                  f"{np.mean(v['realised_acc']):.4f}", len(fe)])
    print("bitti", time.time() - t0)


if __name__ == "__main__":
    main()
