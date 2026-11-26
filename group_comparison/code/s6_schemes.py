#!/usr/bin/env python3
"""Study 3: does anything depend on the 21-category scheme?

(a) Simulation. The design of Study 1 repeated with K = 2, 5, 10 and 21 categories:
    Zipf-like prevalences (exponent 1.1), coder quality set by chance-corrected agreement
    kappa = .3/.5/.7 (accuracy 1/K + kappa (1 - 1/K), so that a coder means the same thing for
    every K; at K = 21 this is close to Study 1's .35/.55/.75), differential error of
    strength delta = .20 in no rows, the rarest rows, the most frequent rows, or all rows
    (one row each when K < 10, three otherwise), n_v = 100 and 250 per group, N = 5,000.

(b) Real coders, collapsed schemes. The 21 CAP topics are merged into K' = 2, 5 or 10
    categories by random partitions (P partitions per K', the same for every coder), and
    into a binary scheme that keeps only the comparison's most prevalent topic against all
    others, the single-category use common in practice. Human and machine codes are merged
    by the same map. Comparisons: period halves of each corpus and all pairs of corpora,
    every coder; stratified validation, n_v = 100 per group.

Output: results/sim_schemes.csv, results/real_schemes.csv
"""
import csv, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, WITH_SE, estimate, tv
from s2_real_coders import load, halves, SEED, K as K21

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TRANSFER = "tfidf-logreg-transfer"


# ---------- (a) simulation over K ----------
def base_confusion(K, acc, rng):
    C = np.zeros((K, K))
    for i in range(K):
        a = float(np.clip(acc + rng.normal(0, .08), .05, .97))
        C[i, i] = a
        if K > 1:
            C[i, np.arange(K) != i] = (1 - a) * rng.dirichlet(np.full(K - 1, .3))
    return C


def perturb(C, rows, delta, rng):
    K = len(C); Cb = C.copy()
    for i in rows:
        red = np.zeros(K); red[np.arange(K) != i] = rng.dirichlet(np.full(K - 1, .5))
        Cb[i] = (1 - delta) * C[i] + delta * red
    return Cb


def simulate(R=300, N=5000):
    rng = np.random.default_rng(SEED + 303)
    rows = []
    for K in (2, 5, 10, 21):
        base = 1 / np.arange(1, K + 1) ** 1.1; base /= base.sum()
        nrow = 3 if K >= 10 else 1
        where_rows = {"none": [], "rare": list(range(K - nrow, K)), "frequent": list(range(nrow)),
                      "all": list(range(K))}
        for kappa in (.3, .5, .7):
            acc = 1 / K + kappa * (1 - 1 / K)
            for where, rws in where_rows.items():
                for nv in (100, 250):
                    res = {e: {"tv": [], "fe": [], "cf": [], "ca": []} for e in ESTIMATORS}
                    for _ in range(R):
                        pA, pB = rng.dirichlet(30 * base), rng.dirichlet(30 * base)
                        C = base_confusion(K, acc, rng)
                        CB = perturb(C, rws, .20, rng)
                        coded = {"A": rng.multinomial(N, (pA[:, None] * C).ravel()).reshape(K, K),
                                 "B": rng.multinomial(N, (pB[:, None] * CB).ravel()).reshape(K, K)}
                        valid = {g: rng.multivariate_hypergeometric(coded[g].ravel(), nv).reshape(K, K)
                                 for g in coded}
                        est, se = estimate(coded, valid)
                        truth = pA - pB
                        for e in ESTIMATORS:
                            err = est[e] - truth
                            res[e]["tv"].append(tv(est[e], truth)); res[e]["fe"].append(err[0])
                            if e in WITH_SE:
                                hit = np.abs(err) <= 1.96 * se[e]
                                res[e]["cf"].append(hit[0]); res[e]["ca"].append(hit.mean())
                    naive_tv = np.array(res["naive"]["tv"])
                    for e in ESTIMATORS:
                        fe = np.array(res[e]["fe"])
                        rows.append({"K": K, "kappa": kappa, "acc": round(acc, 3), "where": where, "n_v": nv, "estimator": e,
                                     "mean_tv_error": float(np.mean(res[e]["tv"])),
                                     "share_worse_than_naive": float(np.mean(np.array(res[e]["tv"]) > naive_tv)),
                                     "focal_bias": float(fe.mean()), "focal_rmse": float(np.sqrt((fe ** 2).mean())),
                                     "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else "",
                                     "allcat_coverage": float(np.mean(res[e]["ca"])) if res[e]["ca"] else ""})
            print("sim K", K, "kappa", kappa, flush=True)
    write("sim_schemes.csv", rows)


# ---------- (b) real coders, collapsed schemes ----------
def partitions(P=10):
    rng = np.random.default_rng(SEED + 404)
    out = []
    for k in (2, 5, 10):
        for p in range(P):
            while True:
                lab = rng.permutation(np.arange(K21) % k)      # every merged category non-empty
                if len(set(lab)) == k:
                    break
            out.append((f"random{k}", p, k, lab))
    return out


def counts_k(y, pred, k):
    c = np.zeros((k, k), dtype=np.int64); np.add.at(c, (y, pred), 1)
    return c


def run(cA, cB, rng, nv=100, R=100):
    k = len(cA)
    truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
    prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
    focal = int(np.argmax(prev))
    res = {e: {"tv": [], "fe": [], "cf": [], "ca": []} for e in ESTIMATORS}
    for _ in range(R):
        valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), nv).reshape(k, k),
                 "B": rng.multivariate_hypergeometric(cB.ravel(), nv).reshape(k, k)}
        est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
        for e in ESTIMATORS:
            err = est[e] - truth
            res[e]["tv"].append(tv(est[e], truth)); res[e]["fe"].append(err[focal])
            if e in WITH_SE:
                hit = np.abs(err) <= 1.96 * se[e]
                res[e]["cf"].append(hit[focal]); res[e]["ca"].append(hit.mean())
    naive_tv = np.array(res["naive"]["tv"])
    return {e: {"mean_tv_error": float(np.mean(res[e]["tv"])),
                "share_worse_than_naive": float(np.mean(np.array(res[e]["tv"]) > naive_tv)),
                "focal_bias": float(np.mean(res[e]["fe"])),
                "focal_rmse": float(np.sqrt(np.mean(np.square(res[e]["fe"])))),
                "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else "",
                "allcat_coverage": float(np.mean(res[e]["ca"])) if res[e]["ca"] else ""}
            for e in ESTIMATORS}, tv(truth, 0 * truth)


def _coder_job(args):
    (m, cond), seed = args
    rng = np.random.default_rng(seed)
    sh = load()
    corpora = sorted({k[2] for k in sh})
    split = {}
    for ci, corpus in enumerate(corpora):
        d = sh[next(k for k in sh if k[2] == corpus)]
        split[corpus] = halves(d["year"].astype(float), SEED + ci)
    schemes = partitions()
    rows = []
    pairs = [("within", f"{c}:period", sh[(m, cond, c)], split[c]) for c in corpora]
    if m != TRANSFER:
        pairs += [("between", f"{a}|{b}", (sh[(m, cond, a)], sh[(m, cond, b)]), None)
                  for a, b in itertools.combinations(corpora, 2)]
    for design, comp, d, g in pairs:
        if design == "within":
            yA, pA, yB, pB = d["y"][~g], d["pred"][~g], d["y"][g], d["pred"][g]
        else:
            yA, pA, yB, pB = d[0]["y"], d[0]["pred"], d[1]["y"], d[1]["pred"]
        top = int(np.argmax(np.bincount(np.concatenate([yA, yB]), minlength=K21)))
        focal_map = (np.arange(K21) != top).astype(int)        # 0 = focal topic, 1 = rest
        for name, p, k, lab in schemes + [("focal_binary", 0, 2, focal_map)]:
            cA, cB = counts_k(lab[yA], lab[pA], k), counts_k(lab[yB], lab[pB], k)
            out, true_tv = run(cA, cB, rng)
            for e, r in out.items():
                rows.append({"model": m, "condition": cond, "design": design, "comparison": comp,
                             "scheme": name, "partition": p, "K": k, "estimator": e,
                             "true_tv_difference": true_tv, **r})
    print(f"{m}/{cond}", flush=True)
    return rows


def real():
    from multiprocessing import Pool
    coders = sorted({(m, c) for m, c, _ in load()})
    jobs = [(mc, SEED + 505 + 1000 * i) for i, mc in enumerate(coders)]   # one stream per coder
    with Pool(min(12, len(jobs))) as pool:
        rows = [r for part in pool.map(_coder_job, jobs) for r in part]
    write("real_schemes.csv", rows)


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    if "--sim" in sys.argv or len(sys.argv) == 1:
        simulate()
    if "--real" in sys.argv or len(sys.argv) == 1:
        real()
