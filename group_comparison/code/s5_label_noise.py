#!/usr/bin/env python3
"""Study 2b: what if the human reference codes are themselves wrong?

Study 2 treats the CAP major-topic codes as the truth. Here they play the role of the
latent truth instead, and the researcher's hand codes of the validation items are drawn
from them with error, as a second human coder with imperfect agreement would produce:

  symmetric     with probability eps the code is replaced by another topic at random;
  confusable    with probability eps it is replaced by a topic the coders of this corpus
                tend to confuse it with (the off-diagonal rows of the pooled confusion of
                the in-corpus supervised classifiers, a proxy for which topics are close);
  differential  symmetric error at eps = .05 in one group and .15 in the other.

The machine codes are unchanged. Every estimator is then evaluated against two targets:
the latent difference d (the CAP codes), and the difference a full hand coding by the
same imperfect coders would give in expectation, d_h = M_A' pi_A - M_B' pi_B, where M is
the human error matrix. With non-differential human error the estimators that use the
validation items are unbiased for d_h, and d_h is an attenuated version of d.

Comparisons: the period halves of each corpus (within) and all pairs of corpora (between),
every coder, stratified validation of n_v = 100 per group, R replications.

Output: results/label_noise.csv (coder x comparison x noise x estimator)
"""
import csv, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, WITH_SE, estimate, tv, row_norm
from s2_real_coders import load, counts, halves, SEED, K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EPS = [0.0, 0.05, 0.10, 0.20]
NV, R = 100, 100
TRANSFER = "tfidf-logreg-transfer"


def noise_matrix(kind, eps, confuse):
    if kind == "symmetric":
        M = np.full((K, K), eps / (K - 1)); np.fill_diagonal(M, 1 - eps)
        return M
    M = eps * confuse; M[np.diag_indices(K)] = 1 - eps
    return M


def corrupt(y, M, rng):
    cum = M[y].cumsum(1)
    return (rng.random(len(y))[:, None] > cum).sum(1).clip(0, K - 1)


def main():
    rng = np.random.default_rng(SEED + 55)
    sh = load()
    corpora = sorted({k[2] for k in sh})
    coders = sorted({(m, c) for m, c, _ in sh})
    # which topics are close: pooled confusion of the in-corpus supervised classifiers
    confuse = {}
    for corpus in corpora:
        c = sum(counts(sh[(m, "sup", corpus)]["y"], sh[(m, "sup", corpus)]["pred"])
                for m in ("tfidf-logreg", "tfidf-svm")).astype(float) + 0.01
        np.fill_diagonal(c, 0)
        confuse[corpus] = c / c.sum(1, keepdims=True)
    split = {}
    for ci, corpus in enumerate(corpora):
        d = sh[next(k for k in sh if k[2] == corpus)]
        split[corpus] = halves(d["year"].astype(float), SEED + ci)
    settings = [("symmetric", e) for e in EPS] + [("confusable", e) for e in EPS[1:]] + [("differential", None)]
    rows, t0 = [], time.time()
    for (m, cond) in coders:
        jobs = []
        for corpus in corpora:
            d = sh[(m, cond, corpus)]; g = split[corpus]
            jobs.append(("within", f"{corpus}:period", corpus, corpus,
                         (d["y"][~g], d["pred"][~g]), (d["y"][g], d["pred"][g])))
        # the transfer classifier's between-corpus codes come from pair-specific models
        # (Study 2); here it enters the within-corpus comparisons only
        for ca, cb in ([] if m == TRANSFER else itertools.combinations(corpora, 2)):
            a, b = sh[(m, cond, ca)], sh[(m, cond, cb)]
            jobs.append(("between", f"{ca}|{cb}", ca, cb, (a["y"], a["pred"]), (b["y"], b["pred"])))
        for design, comp, corpA, corpB, (yA, pA), (yB, pB) in jobs:
            coded = {"A": counts(yA, pA), "B": counts(yB, pB)}
            piA, piB = coded["A"].sum(1) / len(yA), coded["B"].sum(1) / len(yB)
            d_lat = piA - piB
            focal = int(np.argmax((coded["A"].sum(1) + coded["B"].sum(1))))
            for kind, eps in settings:
                if kind == "differential":
                    MA, MB = noise_matrix("symmetric", .05, None), noise_matrix("symmetric", .15, None)
                else:
                    MA = noise_matrix(kind, eps, confuse[corpA]); MB = noise_matrix(kind, eps, confuse[corpB])
                d_h = MA.T @ piA - MB.T @ piB
                res = {e: {"lat": [], "hum": [], "fl": [], "cl": [], "ch": []} for e in ESTIMATORS}
                for _ in range(R):
                    valid = {}
                    for gname, y, p, M in (("A", yA, pA, MA), ("B", yB, pB, MB)):
                        idx = rng.choice(len(y), NV, replace=False)
                        valid[gname] = counts(corrupt(y[idx], M, rng), p[idx])
                    est, se = estimate(coded, valid, finite=True)
                    for e in ESTIMATORS:
                        res[e]["lat"].append(tv(est[e], d_lat)); res[e]["hum"].append(tv(est[e], d_h))
                        res[e]["fl"].append(est[e][focal] - d_lat[focal])
                        if e in WITH_SE:
                            res[e]["cl"].append(abs(est[e][focal] - d_lat[focal]) <= 1.96 * se[e][focal])
                            res[e]["ch"].append(abs(est[e][focal] - d_h[focal]) <= 1.96 * se[e][focal])
                for e in ESTIMATORS:
                    rows.append({"model": m, "condition": cond, "design": design, "comparison": comp,
                                 "noise": kind, "eps": "" if eps is None else eps, "estimator": e,
                                 "true_tv_difference": tv(d_lat, 0 * d_lat),
                                 "human_target_tv_gap": tv(d_h, d_lat),
                                 "tv_error_latent": float(np.mean(res[e]["lat"])),
                                 "tv_error_human_target": float(np.mean(res[e]["hum"])),
                                 "focal_bias_latent": float(np.mean(res[e]["fl"])),
                                 "focal_attenuation": float(d_h[focal] / d_lat[focal]) if abs(d_lat[focal]) > 1e-9 else "",
                                 "focal_coverage_latent": float(np.mean(res[e]["cl"])) if res[e]["cl"] else "",
                                 "focal_coverage_human_target": float(np.mean(res[e]["ch"])) if res[e]["ch"] else ""})
        print(f"{m}/{cond} {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(RES, "label_noise.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("bitti", time.time() - t0)


if __name__ == "__main__":
    main()
