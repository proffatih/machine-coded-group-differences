#!/usr/bin/env python3
"""Study 2: the estimators with real coders, where the truth is the human code.

Every item of the seven CAP corpora (1,500 each) carries a human code and a code from
each of 30 coder-condition combinations: nine open-weight language models under three
prompt conditions, two supervised classifiers cross-fitted within the corpus, and one
classifier trained on the other six corpora. The five few-shot exemplars of each corpus
are removed for every coder, so that no coder is evaluated on an item it was shown.
A comparison's estimand is the difference in human-coded prevalence between the two
groups of items; the researcher sees every machine code but hand-codes a validation
sample of n_v items per group. Repeating the draw gives each estimator's error
distribution for that comparison.

  within   two halves of one corpus, split at its median year (earlier vs later agenda,
           the comparison agenda research makes over time) or, as a stress test that
           changes the text a coder sees, at its median length. The halves are fixed
           per corpus (same items for every coder); ties are broken by a seeded draw.
  between  two corpora, e.g. a legislative and a media agenda. Nothing is inferred about
           why a coder behaves differently on the two; only how well each estimator
           recovers the difference.

Two validation designs are drawn for every replication: n_v items within each group
(stratified), and 2 n_v items from both groups together (pooled), each analysed by group.

Per comparison, the population confusion matrices (all items) also give
  - back-calculation with the true matrices, pooled and per group (no sampling error),
  - the exact split of the uncorrected error into a part shared by the groups and a
    part due to their differing matrices (estimators.decompose_naive).

With --oracle-only, only results/real_oracle.csv is rewritten (no validation draws).

Output: results/real_summary.csv      comparison x design x n_v x estimator
        results/real_oracle.csv       comparison-level population quantities
        results/real_coder_accuracy.csv
"""
import csv, glob, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ESTIMATORS, WITH_SE, estimate, tv, back_calc, decompose_naive, row_norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the annotation study: a sibling folder in the working tree, the parent folder in the public repository
SSCR = os.path.join(os.path.dirname(ROOT), "Social_Science_Computer_Review")
if not os.path.isdir(SSCR):
    SSCR = os.path.dirname(ROOT)
ANN = [os.path.join(SSCR, "results", "annotations"),
       os.path.join(ROOT, "results", "annotations_transfer")]
PAIRS = os.path.join(ROOT, "results", "annotations_transfer_pairs")
TRANSFER = "tfidf-logreg-transfer"
N_RANDOM = 20
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(SSCR, "code"))
from cap_codebook import CORPORA, VALID_CODES, SEED as ANNOT_SEED
CODES = np.array(VALID_CODES); K = len(CODES)
NV = [50, 100, 200]
R_WITHIN, R_BETWEEN, SEED = 200, 100, 20261102
K_SHOT = 5


def fewshot_ids(corpus):
    """The exemplars 02_annotate.py put into the few-shot prompt (same rule, same seed)."""
    with open(os.path.join(SSCR, "data", f"{corpus}.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = np.random.default_rng(ANNOT_SEED)
    by_lab = {}
    for r in rows:
        if r["split"] == "calib":
            by_lab.setdefault(int(r["label"]), []).append(r)
    return {by_lab[int(l)][0]["id"] for l in rng.permutation(sorted(by_lab))[:K_SHOT]}


def _read(p):
    z = np.load(p, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    return m, {"ids": z["ids"], "y": np.searchsorted(CODES, z["gold"]),
               "pred": z["probs"].argmax(1), "year": z["year"], "n_words": z["n_words"]}


def load(with_pairs=False):
    """Coder outputs per (model, condition, corpus), few-shot exemplars removed. With
    with_pairs, also the transfer classifier's codes for each pair of corpora from the
    model trained outside the pair: {(a, b): {corpus: shard}}."""
    shards = {}
    for d in ANN:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            m, v = _read(p)
            shards[(m["model"], m["condition"], m["corpus"])] = v
    keep = {c: ~np.isin(next(v["ids"] for k, v in shards.items() if k[2] == c),
                        list(fewshot_ids(c))) for c in CORPORA}
    def trim(v, c):
        for f in ("ids", "y", "pred", "year", "n_words"):
            v[f] = v[f][keep[c]]
    for (m, cond, c), v in shards.items():
        trim(v, c)
    if not with_pairs:
        return shards
    pairs = {}
    for p in sorted(glob.glob(os.path.join(PAIRS, "*.npz"))):
        m, v = _read(p)
        trim(v, m["corpus"])
        pairs.setdefault(tuple(m["pair"]), {})[m["corpus"]] = v
    return shards, pairs


def counts(y, pred):
    c = np.zeros((K, K), dtype=np.int64)
    np.add.at(c, (y, pred), 1)
    return c


def halves(x, seed):
    """Equal halves by rank, ties broken by a seeded draw (same for every coder)."""
    r = np.random.default_rng(seed).random(len(x))
    order = np.lexsort((r, x))
    g = np.zeros(len(x), dtype=bool); g[order[len(x) // 2:]] = True
    return g


def pooled_draw(cA, cB, n, rng):
    both = np.concatenate([cA.ravel(), cB.ravel()])
    s = rng.multivariate_hypergeometric(both, n)
    return {"A": s[:K * K].reshape(K, K), "B": s[K * K:].reshape(K, K)}


def run_comparison(cA, cB, reps, rng):
    truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
    prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
    focal = int(np.argmax(prev))
    rare = prev < .02
    rows = []
    for nv in NV:
        acc = {}
        for design in ("stratified", "pooled"):
            ests = ESTIMATORS if design == "stratified" else ["cal_group", "dd_group", "dd_tuned", "hand"]
            res = {e: {"tv": [], "fe": [], "cf": [], "ca": [], "cr": []} for e in ests}
            for _ in range(reps):
                if design == "stratified":
                    valid = {"A": rng.multivariate_hypergeometric(cA.ravel(), nv).reshape(K, K),
                             "B": rng.multivariate_hypergeometric(cB.ravel(), nv).reshape(K, K)}
                else:
                    valid = pooled_draw(cA, cB, 2 * nv, rng)
                    if min(valid["A"].sum(), valid["B"].sum()) < 3:
                        continue
                est, se = estimate({"A": cA, "B": cB}, valid, finite=True)
                for e in ests:
                    err = est[e] - truth
                    res[e]["tv"].append(tv(est[e], truth)); res[e]["fe"].append(err[focal])
                    if e in WITH_SE:
                        hit = np.abs(err) <= 1.96 * se[e]
                        res[e]["cf"].append(hit[focal]); res[e]["ca"].append(hit.mean())
                        res[e]["cr"].append(hit[rare].mean() if rare.any() else np.nan)
            for e in ests:
                f = np.array(res[e]["fe"])
                rows.append({"n_v": nv, "sampling": design, "estimator": e,
                             "mean_tv_error": float(np.mean(res[e]["tv"])),
                             "focal_bias": float(f.mean()),
                             "focal_rmse": float(np.sqrt((f ** 2).mean())),
                             "focal_coverage": float(np.mean(res[e]["cf"])) if res[e]["cf"] else "",
                             "allcat_coverage": float(np.mean(res[e]["ca"])) if res[e]["ca"] else "",
                             "rare_coverage": float(np.nanmean(res[e]["cr"])) if res[e]["cr"] and rare.any() else ""})
    return rows, oracle_quantities(cA, cB)


def oracle_quantities(cA, cB):
    """Population-level quantities of one comparison (no sampling)."""
    truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
    prev = (cA.sum(1) + cB.sum(1)) / (cA.sum() + cB.sum())
    focal = int(np.argmax(prev))
    shared, diff = decompose_naive(cA, cB)
    qA, qB = cA.sum(0) / cA.sum(), cB.sum(0) / cB.sum()
    CA, CB = row_norm(cA), row_norm(cB)
    oracle = {
        "true_tv_difference": tv(truth, 0 * truth),
        "naive_tv_error": tv(qA - qB, truth),
        "shared_tv": tv(shared, 0 * shared), "differential_tv": tv(diff, 0 * diff),
        "naive_focal_error": float((qA - qB - truth)[focal]),
        "shared_focal": float(shared[focal]), "differential_focal": float(diff[focal]),
        "bc_oracle_pooled_tv": tv(back_calc(qA, cA + cB) - back_calc(qB, cA + cB), truth),
        "bc_oracle_group_tv": tv(back_calc(qA, cA) - back_calc(qB, cB), truth),
        "acc_A": float(np.trace(cA) / cA.sum()), "acc_B": float(np.trace(cB) / cB.sum()),
        "row_l1_weighted": float(0.5 * (np.abs(CA - CB).sum(1) * prev).sum()),
        "sigma_min_pooled": float(np.linalg.svd(row_norm(cA + cB), compute_uv=False)[-1]),
        "sigma_min_group": float(min(np.linalg.svd(CA, compute_uv=False)[-1],
                                     np.linalg.svd(CB, compute_uv=False)[-1])),
        "unassigned_topics": int(max(((cA.sum(0) == 0) & (cA.sum(1) > 0)).sum(),
                                     ((cB.sum(0) == 0) & (cB.sum(1) > 0)).sum())),
        "focal_prevalence": float(prev[focal]), "n_A": int(cA.sum()), "n_B": int(cB.sum()),
    }
    return oracle


def random_splits(sh, seed):
    """The decomposition for random halves of each corpus: the differential part that
    arises by chance when the groups do not differ systematically."""
    out = []
    for (m, c, corpus), d in sorted(sh.items()):
        for r in range(N_RANDOM):
            g = halves(np.zeros(len(d["y"])), seed + 1000 * r + 10 * sorted(CORPORA).index(corpus) + 7)
            cA, cB = counts(d["y"][g], d["pred"][g]), counts(d["y"][~g], d["pred"][~g])
            shared, diff = decompose_naive(cA, cB)
            truth = cA.sum(1) / cA.sum() - cB.sum(1) / cB.sum()
            out.append({"model": m, "condition": c, "corpus": corpus, "rep": r,
                        "true_tv_difference": tv(truth, 0 * truth),
                        "shared_tv": tv(shared, 0 * shared), "differential_tv": tv(diff, 0 * diff)})
    return out


def main(oracle_only=False):
    rng = np.random.default_rng(SEED)
    sh, pairs = load(with_pairs=True)
    coders = sorted({(m, c) for m, c, _ in sh})
    corpora = sorted({k[2] for k in sh})
    split = {}
    for ci, corpus in enumerate(corpora):
        d = sh[next(k for k in sh if k[2] == corpus)]
        split[(corpus, "period")] = halves(d["year"].astype(float), SEED + ci)
        split[(corpus, "length")] = halves(d["n_words"].astype(float), SEED + 100 + ci)
    out, orc, acc_rows = [], [], []
    t0 = time.time()
    for (m, c) in coders:
        jobs = []
        for corpus in corpora:
            d = sh[(m, c, corpus)]
            acc_rows.append({"model": m, "condition": c, "corpus": corpus,
                             "accuracy": float((d["y"] == d["pred"]).mean()),
                             "majority_baseline": float(np.bincount(d["y"], minlength=K).max() / len(d["y"]))})
            for gname in ("period", "length"):
                g = split[(corpus, gname)]
                jobs.append(("within", f"{corpus}:{gname}", counts(d["y"][g], d["pred"][g]),
                             counts(d["y"][~g], d["pred"][~g]), R_WITHIN))
        for ca, cb in itertools.combinations(corpora, 2):
            if m == TRANSFER:      # trained outside the pair: no label of either group
                a, b = pairs[(ca, cb)][ca], pairs[(ca, cb)][cb]
            else:
                a, b = sh[(m, c, ca)], sh[(m, c, cb)]
            jobs.append(("between", f"{ca}|{cb}", counts(a["y"], a["pred"]),
                         counts(b["y"], b["pred"]), R_BETWEEN))
        for design, comp, cA, cB, reps in jobs:
            if oracle_only:
                rows, oracle = [], oracle_quantities(cA, cB)
            else:
                rows, oracle = run_comparison(cA, cB, reps, rng)
            tag = {"design": design, "model": m, "condition": c, "comparison": comp,
                   "true_tv_difference": oracle["true_tv_difference"]}
            for r in rows:
                r.update(tag); out.append(r)
            orc.append({**tag, **oracle})
        print(f"{m}/{c} {time.time() - t0:.0f}s", flush=True)
    rs = random_splits(sh, SEED)
    with open(os.path.join(RES, "real_random_split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    if oracle_only:
        with open(os.path.join(RES, "real_oracle.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(orc[0])); w.writeheader(); w.writerows(orc)
        return
    keys = ["design", "model", "condition", "comparison", "true_tv_difference", "n_v",
            "sampling", "estimator", "mean_tv_error", "focal_bias", "focal_rmse",
            "focal_coverage", "allcat_coverage", "rare_coverage"]
    with open(os.path.join(RES, "real_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    with open(os.path.join(RES, "real_oracle.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orc[0])); w.writeheader(); w.writerows(orc)
    with open(os.path.join(RES, "real_coder_accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0])); w.writeheader(); w.writerows(acc_rows)
    print("bitti", time.time() - t0)


if __name__ == "__main__":
    main(oracle_only="--oracle-only" in sys.argv)
