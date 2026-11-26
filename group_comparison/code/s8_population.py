#!/usr/bin/env python3
"""Study 2 with the full CAP corpus as the population (Online Appendix G).

In Study 2 the estimand is the prevalence among the 1,500 coded items of a corpus, and the
standard errors carry a finite-population correction that credits validation samples of up
to a quarter of each half. Here the estimand is the prevalence in the full CAP corpus from
which the 1,500 were drawn (all items that pass the same filters: one of the 21 topics,
at least three words, deduplicated), and the standard errors are the two-phase ones for a
sample from a population (finite=False): the 1,500 coded items are the first phase, the
validation items the second.

Groups: within a corpus, items before and after the sample's median year (items in the median
year are left out of both the sample groups and the population groups, so that the groups are
defined by the same rule in both); between corpora, the corpora themselves. The 1,500 items
were drawn with proportional allocation across topics, so the first phase is at least as
efficient as simple random sampling and the intervals are, if anything, conservative.

Output: results/population.csv (coder x comparison x n_v x estimator)
"""
import collections, csv, hashlib, itertools, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import estimate, tv, ESTIMATORS, WITH_SE
from s2_real_coders import load, counts, SEED, K, CODES, SSCR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(SSCR, "code"))
from cap_codebook import CORPORA, VALID_CODES
csv.field_size_limit(10 ** 9)
TRANSFER = "tfidf-logreg-transfer"
NV, R = [50, 100, 200], 100
MIN_WORDS = 3


def population():
    """Human-code counts and years of every item that passes the filters of 01_prepare_corpora.py."""
    valid = set(VALID_CODES)
    out = {}
    for key, (fname, tcol, lcol, *_rest) in CORPORA.items():
        seen, codes, years = set(), [], []
        with open(os.path.join(SSCR, "data_raw", fname), encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                text = " ".join((row.get(tcol) or "").split())
                if not text:
                    continue
                try:
                    code = int(float((row.get(lcol) or "").strip()))
                except ValueError:
                    continue
                if code not in valid or len(text.split()) < MIN_WORDS:
                    continue
                h = hashlib.md5(text.lower().encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                try:
                    year = int(float(row.get("year") or 0))
                except ValueError:
                    year = 0
                codes.append(int(np.searchsorted(CODES, code))); years.append(year)
        out[key] = (np.array(codes), np.array(years))
    return out


POP = None


def job(args):
    (m, cond), seed = args
    global POP
    rng = np.random.default_rng(seed)
    sh, pairs = load(with_pairs=True)
    corpora = sorted({k[2] for k in sh})
    rows = []
    comps = []
    for c in corpora:
        d = sh[(m, cond, c)]
        T = np.median(d["year"])
        gA, gB = d["year"] < T, d["year"] > T
        py, pyr = POP[c]
        piA = np.bincount(py[pyr < T], minlength=K) / (pyr < T).sum()
        piB = np.bincount(py[pyr > T], minlength=K) / (pyr > T).sum()
        comps.append(("within", f"{c}:period", counts(d["y"][gA], d["pred"][gA]), counts(d["y"][gB], d["pred"][gB]), piA - piB))
    for ca, cb in itertools.combinations(corpora, 2):
        a, b = (pairs[(ca, cb)][ca], pairs[(ca, cb)][cb]) if m == TRANSFER else (sh[(m, cond, ca)], sh[(m, cond, cb)])
        piA = np.bincount(POP[ca][0], minlength=K) / len(POP[ca][0])
        piB = np.bincount(POP[cb][0], minlength=K) / len(POP[cb][0])
        comps.append(("between", f"{ca}|{cb}", counts(a["y"], a["pred"]), counts(b["y"], b["pred"]), piA - piB))
    for design, comp, cA, cB, truth in comps:
        prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
        focal = int(np.argmax(prev))
        sample_truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
        for nv in NV:
            res = {e: {"tv": [], "fe": [], "cf": [], "ca": []} for e in ESTIMATORS}
            for _ in range(R):
                valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), nv).reshape(K, K),
                         "B": rng.multivariate_hypergeometric(cB.ravel(), nv).reshape(K, K)}
                est, se = estimate({"A": cA, "B": cB}, valid, finite=False)
                for e in ESTIMATORS:
                    err = est[e] - truth
                    res[e]["tv"].append(tv(est[e], truth)); res[e]["fe"].append(err[focal])
                    if e in WITH_SE:
                        hit = np.abs(err) <= 1.96 * se[e]
                        res[e]["cf"].append(hit[focal]); res[e]["ca"].append(hit.mean())
            nt = np.array(res["naive"]["tv"])
            for e in ESTIMATORS:
                fe = np.array(res[e]["fe"])
                rows.append({"model": m, "condition": cond, "design": design, "comparison": comp, "n_v": nv,
                             "estimator": e, "n_A": int(cA.sum()), "n_B": int(cB.sum()),
                             "sample_vs_population_tv": tv(sample_truth, truth),
                             "mean_tv_error": float(np.mean(res[e]["tv"])),
                             "share_draws_worse_than_naive": float(np.mean(np.array(res[e]["tv"]) > nt)),
                             "focal_bias": float(fe.mean()), "focal_rmse": float(np.sqrt((fe ** 2).mean())),
                             "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else "",
                             "allcat_coverage": float(np.mean(res[e]["ca"])) if res[e]["ca"] else ""})
    print(f"{m}/{cond}", flush=True)
    return rows


def init(pop):
    global POP
    POP = pop


def main():
    t0 = time.time()
    pop = population()
    print({k: len(v[0]) for k, v in pop.items()}, flush=True)
    coders = sorted({(m, c) for m, c, _ in load()})
    jobs = [(mc, SEED + 808 + 1000 * i) for i, mc in enumerate(coders)]
    with Pool(min(14, len(jobs)), initializer=init, initargs=(pop,)) as p:
        rows = [r for part in p.map(job, jobs) for r in part]
    with open(os.path.join(RES, "population.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("bitti", round(time.time() - t0))


if __name__ == "__main__":
    main()
