#!/usr/bin/env python3
"""Extensions of Studies 1-2 with the real coders (Online Appendix G).

(a) Simulation with the real coders' confusion matrices. For every coder-condition
    combination and every comparison (period halves, pairs of corpora) the population
    confusion matrices C_A, C_B and prevalences pi_A, pi_B are taken from the real data, and
    N = 5,000 items per group are drawn from pi_g (x) C_g; n_v = 100 or 200 per group are
    hand-coded. Two versions: the matrices as they are, and the matrices mixed with the
    identity (C' = .9 C + .1 I), which makes every topic assignable and the matrices well
    conditioned. This places Study 1's design on the real coders' error structure.

(b) Regularised back-calculation, the strongest form of the matrix method we could build:
    each group's matrix is estimated with its rows shrunk towards the pooled matrix,
    C_g[i] = (v_g[i] + tau * Cbar[i]) / (n_gi + tau), tau = 10, which keeps rows with few
    validation items close to the pooled estimate and never leaves a row empty; the system is
    solved by NNLS. Its standard error, and that of per-group back-calculation, comes from a
    bootstrap of the validation items within each group (B = 100), for the first R_BOOT draws.

(c) Intervals in detail, from the same real-data draws (n_v = 100 per group, stratified):
    coverage and mean half-width by topic prevalence (< .02, .02-.05, .05-.15, >= .15), and
    joint coverage of all topics of a comparison with Bonferroni-adjusted intervals.

Output: results/ext_empirical_sim.csv, results/ext_real_draws.csv (comparison x estimator),
        results/ext_bins.csv (comparison x estimator x prevalence bin)
"""
import csv, itertools, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import estimate, tv, row_norm, back_calc, WITH_SE
from s2_real_coders import load, counts, halves, SEED, K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TRANSFER = "tfidf-logreg-transfer"
TAU, B, R_REAL, R_BOOT, R_SIM, N_SIM = 10, 100, 100, 30, 100, 5000
BINS = [(0, .02, "<.02"), (.02, .05, ".02-.05"), (.05, .15, ".05-.15"), (.15, 1.01, ">=.15")]
EXT = ["naive", "bc_pooled", "bc_group", "bc_shrink", "cal_group", "dd_tuned", "hand"]
SE_EST = ["naive", "cal_group", "dd_tuned", "hand"]


def bc_shrink(q, v, vpool, tau=TAU):
    Cbar = row_norm(vpool)
    n_i = v.sum(1, keepdims=True)
    C = (v + tau * Cbar) / (n_i + tau)
    from scipy.optimize import nnls
    try:
        x, _ = nnls(C.T, q)
    except RuntimeError:
        x = np.clip(np.linalg.lstsq(C.T, q, rcond=None)[0], 0, None)
    s = x.sum()
    return x / s if s > 0 else q


def boot_se(q, v, vother, rng, kind):
    """Bootstrap SE of one group's back-calculated prevalence: resample the group's validation
    items (and, for the shrunken version, the other group's for the pooled target)."""
    cells = np.repeat(np.arange(K * K), v.ravel())
    cells_o = np.repeat(np.arange(K * K), vother.ravel())
    out = np.empty((B, K))
    for b in range(B):
        vb = np.bincount(rng.choice(cells, len(cells)), minlength=K * K).reshape(K, K)
        if kind == "bc_group":
            out[b] = back_calc(q, vb)
        else:
            vo = np.bincount(rng.choice(cells_o, len(cells_o)), minlength=K * K).reshape(K, K)
            out[b] = bc_shrink(q, vb, vb + vo)
    return out.std(0, ddof=1)


def real_job(args):
    (m, cond), seed = args
    rng = np.random.default_rng(seed)
    sh, pairs = load(with_pairs=True)
    corpora = sorted({k[2] for k in sh})
    split = {}
    for ci, corpus in enumerate(corpora):
        d = sh[next(k for k in sh if k[2] == corpus)]
        split[corpus] = halves(d["year"].astype(float), SEED + ci)
    jobs = []
    for c in corpora:
        d = sh[(m, cond, c)]; g = split[c]
        jobs.append(("within", f"{c}:period", counts(d["y"][~g], d["pred"][~g]), counts(d["y"][g], d["pred"][g])))
    for ca, cb in itertools.combinations(corpora, 2):
        a, b = (pairs[(ca, cb)][ca], pairs[(ca, cb)][cb]) if m == TRANSFER else (sh[(m, cond, ca)], sh[(m, cond, cb)])
        jobs.append(("between", f"{ca}|{cb}", counts(a["y"], a["pred"]), counts(b["y"], b["pred"])))
    rows, brows, srows = [], [], []
    for design, comp, cA, cB in jobs:
        truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
        prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
        present = (cA.sum(1) + cB.sum(1)) > 0
        focal = int(np.argmax(prev))
        zb = norm.ppf(1 - .05 / (2 * present.sum()))
        acc = {e: {"tv": [], "fe": []} for e in EXT}
        cov = {e: [] for e in SE_EST + ["bc_group", "bc_shrink"]}
        joint = {e: [] for e in SE_EST}
        binc = {(e, bn): [] for e in SE_EST for *_, bn in BINS}
        binw = {(e, bn): [] for e in SE_EST for *_, bn in BINS}
        for r in range(R_REAL):
            valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), 100).reshape(K, K),
                     "B": rng.multivariate_hypergeometric(cB.ravel(), 100).reshape(K, K)}
            est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
            qA, qB = cA.sum(0) / cA.sum(), cB.sum(0) / cB.sum()
            vp = valid["A"] + valid["B"]
            est["bc_shrink"] = bc_shrink(qA, valid["A"], vp) - bc_shrink(qB, valid["B"], vp)
            for e in EXT:
                acc[e]["tv"].append(tv(est[e], truth)); acc[e]["fe"].append(est[e][focal] - truth[focal])
            for e in SE_EST:
                err = np.abs(est[e] - truth)
                cov[e].append(err[focal] <= 1.96 * se[e][focal])
                joint[e].append(bool(np.all(err[present] <= zb * se[e][present])))
                for lo, hi, bn in BINS:
                    sel = present & (prev >= lo) & (prev < hi)
                    if sel.any():
                        binc[(e, bn)].append(float(np.mean(err[sel] <= 1.96 * se[e][sel])))
                        binw[(e, bn)].append(float(np.mean(1.96 * se[e][sel])))
            if r < R_BOOT:
                for kind in ("bc_group", "bc_shrink"):
                    s2 = boot_se(qA, valid["A"], valid["B"], rng, kind) ** 2 + boot_se(qB, valid["B"], valid["A"], rng, kind) ** 2
                    cov[kind].append(abs(est[kind][focal] - truth[focal]) <= 1.96 * np.sqrt(s2[focal]))
        naive_tv = np.array(acc["naive"]["tv"])
        for e in EXT:
            fe = np.array(acc[e]["fe"])
            rows.append({"model": m, "condition": cond, "design": design, "comparison": comp, "estimator": e,
                         "mean_tv_error": float(np.mean(acc[e]["tv"])),
                         "share_draws_worse_than_naive": float(np.mean(np.array(acc[e]["tv"]) > naive_tv)),
                         "focal_bias": float(fe.mean()), "focal_rmse": float(np.sqrt((fe ** 2).mean())),
                         "focal_coverage": float(np.mean(cov[e])) if e in cov else "",
                         "joint_coverage_bonferroni": float(np.mean(joint[e])) if e in joint else ""})
        for (e, bn), v in binc.items():
            if v:
                brows.append({"model": m, "condition": cond, "design": design, "comparison": comp, "estimator": e,
                              "bin": bn, "coverage": float(np.mean(v)), "half_width": float(np.mean(binw[(e, bn)]))})
        # (a) simulation on this comparison's matrices
        for version in ("empirical", "repaired"):
            CA, CB = row_norm(cA), row_norm(cB)
            if version == "repaired":
                CA, CB = .9 * CA + .1 * np.eye(K), .9 * CB + .1 * np.eye(K)
            piA, piB = cA.sum(1) / cA.sum(), cB.sum(1) / cB.sum()
            sm = float(np.linalg.svd(.5 * (CA + CB), compute_uv=False)[-1])
            for nv in (100, 200):
                res = {e: {"tv": [], "cf": []} for e in ("naive", "bc_pooled", "bc_group", "cal_group", "dd_tuned")}
                for _ in range(R_SIM):
                    coded = {"A": rng.multinomial(N_SIM, (piA[:, None] * CA).ravel()).reshape(K, K),
                             "B": rng.multinomial(N_SIM, (piB[:, None] * CB).ravel()).reshape(K, K)}
                    tr = coded["A"].sum(1) / N_SIM - coded["B"].sum(1) / N_SIM
                    valid = {g_: rng.multivariate_hypergeometric(coded[g_].ravel(), nv).reshape(K, K) for g_ in coded}
                    est, se = estimate(coded, valid, finite=True)
                    for e in res:
                        res[e]["tv"].append(tv(est[e], tr))
                        if e in se:
                            res[e]["cf"].append(abs(est[e][focal] - tr[focal]) <= 1.96 * se[e][focal])
                nt = np.array(res["naive"]["tv"])
                for e in res:
                    srows.append({"model": m, "condition": cond, "design": design, "comparison": comp,
                                  "version": version, "n_v": nv, "estimator": e, "sigma_min": sm,
                                  "mean_tv_error": float(np.mean(res[e]["tv"])),
                                  "share_draws_worse_than_naive": float(np.mean(np.array(res[e]["tv"]) > nt)),
                                  "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else ""})
    print(f"{m}/{cond}", flush=True)
    return rows, brows, srows


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    t0 = time.time()
    coders = sorted({(m, c) for m, c, _ in load()})
    jobs = [(mc, SEED + 707 + 1000 * i) for i, mc in enumerate(coders)]
    with Pool(min(14, len(jobs))) as pool:
        parts = pool.map(real_job, jobs)
    write("ext_real_draws.csv", [r for p in parts for r in p[0]])
    write("ext_bins.csv", [r for p in parts for r in p[1]])
    write("ext_empirical_sim.csv", [r for p in parts for r in p[2]])
    print("bitti", round(time.time() - t0))


if __name__ == "__main__":
    main()
