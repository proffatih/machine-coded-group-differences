#!/usr/bin/env python3
"""Study 2: the estimators with real coders, where the truth is the human code.

Every item of the seven CAP corpora (1,500 each) carries a human code and a code from
each of 20 coder-condition combinations (six open-weight language models under three
prompt conditions, two supervised classifiers predicted out-of-fold). A comparison's
estimand is the difference in human-coded prevalence between the two groups of items;
the researcher sees every machine code but hand-codes only a validation sample of n_v
items per group, drawn at random from those items. Repeating the draw R times gives
each estimator's error distribution for that comparison.

  within   two halves of one corpus: split at the corpus's own median length or
           median year (ties broken at random so the halves are equal). The prompt,
           the examples and the classifier are identical for both halves.
  between  two corpora, i.e. two bodies of text such as a legislative and a media
           agenda -- the comparison agenda research makes. Nothing is inferred here
           about why a coder behaves differently on the two; only how well each
           estimator recovers the difference.

Output: results/real_summary.csv (one row per comparison x n_v x estimator),
        results/real_coder_accuracy.csv
"""
import csv, glob, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, estimate, tv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(os.path.dirname(ROOT), "results", "annotations")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "code"))
from cap_codebook import VALID_CODES
CODES = np.array(VALID_CODES); K = len(CODES)
NV = [50, 100, 200]
R_WITHIN, R_BETWEEN, SEED = 200, 100, 20261102


def load():
    shards = {}
    for p in sorted(glob.glob(os.path.join(ANN, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        y = np.searchsorted(CODES, z["gold"])
        pred = z["probs"].argmax(1)
        shards[(m["model"], m["condition"], m["corpus"])] = {
            "y": y, "pred": pred, "year": z["year"], "n_words": z["n_words"]}
    return shards


def counts(y, pred):
    c = np.zeros((K, K), dtype=np.int64)
    np.add.at(c, (y, pred), 1)
    return c


def halves(x, rng):
    """Equal halves by rank, ties broken at random."""
    order = np.lexsort((rng.random(len(x)), x))
    g = np.zeros(len(x), dtype=bool); g[order[len(x) // 2:]] = True
    return g


def run_comparison(cA, cB, reps, rng):
    truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
    focal = int(np.argmax(cA.sum(1) + cB.sum(1)))
    rows = []
    for nv in NV:
        tvs = {e: [] for e in ESTIMATORS}; fe = {e: [] for e in ESTIMATORS}
        cov = {e: [] for e in ("naive", "dd_group", "dd_tuned", "hand")}
        for _ in range(reps):
            valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), nv).reshape(K, K),
                     "B": rng.multivariate_hypergeometric(cB.ravel(), nv).reshape(K, K)}
            est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
            for e in ESTIMATORS:
                tvs[e].append(tv(est[e], truth)); fe[e].append(est[e][focal] - truth[focal])
            for e in cov:
                s = se[e][focal]
                cov[e].append(int(abs(est[e][focal] - truth[focal]) <= 1.96 * s) if s > 0
                              else int(abs(est[e][focal] - truth[focal]) < 1e-12))
        for e in ESTIMATORS:
            f = np.array(fe[e])
            rows.append({"n_v": nv, "estimator": e, "mean_tv_error": float(np.mean(tvs[e])),
                         "focal_bias": float(f.mean()), "focal_rmse": float(np.sqrt((f ** 2).mean())),
                         "focal_coverage": float(np.mean(cov[e])) if e in cov else ""})
    return rows, float(tv(truth, 0 * truth))


def main():
    rng = np.random.default_rng(SEED)
    sh = load()
    coders = sorted({(m, c) for m, c, _ in sh})
    corpora = sorted({k[2] for k in sh})
    out, acc_rows = [], []
    t0 = time.time()
    for (m, c) in coders:
        for corpus in corpora:
            d = sh[(m, c, corpus)]
            acc_rows.append({"model": m, "condition": c, "corpus": corpus,
                             "accuracy": float((d["y"] == d["pred"]).mean())})
            for gname, x in (("length", d["n_words"]), ("period", d["year"])):
                if gname == "period" and (x <= 0).mean() > .1:
                    continue
                g = halves(x.astype(float), rng)
                cA, cB = counts(d["y"][g], d["pred"][g]), counts(d["y"][~g], d["pred"][~g])
                rows, size = run_comparison(cA, cB, R_WITHIN, rng)
                for r in rows:
                    r.update({"design": "within", "model": m, "condition": c,
                              "comparison": f"{corpus}:{gname}", "true_tv_difference": size})
                    out.append(r)
        for ca, cb in itertools.combinations(corpora, 2):
            a, b = sh[(m, c, ca)], sh[(m, c, cb)]
            rows, size = run_comparison(counts(a["y"], a["pred"]), counts(b["y"], b["pred"]),
                                        R_BETWEEN, rng)
            for r in rows:
                r.update({"design": "between", "model": m, "condition": c,
                          "comparison": f"{ca}|{cb}", "true_tv_difference": size})
                out.append(r)
        print(f"{m}/{c} {time.time() - t0:.0f}s", flush=True)
    keys = ["design", "model", "condition", "comparison", "true_tv_difference", "n_v",
            "estimator", "mean_tv_error", "focal_bias", "focal_rmse", "focal_coverage"]
    with open(os.path.join(RES, "real_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    with open(os.path.join(RES, "real_coder_accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0])); w.writeheader(); w.writerows(acc_rows)
    print("bitti", time.time() - t0)


if __name__ == "__main__":
    main()
