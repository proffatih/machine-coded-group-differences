#!/usr/bin/env python3
"""Study 2a: measurement quality of every coder within each group, before any comparison.

For each coder-condition combination, corpus and within-corpus split (period, length) and
for each of the two groups separately:
  - accuracy, macro-averaged precision, recall and F1 over the topics present in the group,
    prevalence-weighted F1, and per-topic precision, recall and F1 with their support;
  - calibration of the coder's probabilities: top-label expected calibration error
    (ECE, 10 equal-width bins) and the multi-class Brier score.
The between-corpus comparisons use each corpus as a group; their within-group figures are
the corpus-level rows (split = "all").

Within-group error patterns: for every topic with at least five items in each group, the
difference between the two groups' recall is compared with a permutation distribution that
keeps the topic's number of items in each group (its items are reassigned to the groups at
random, N_PERM draws). Reported are the share of topics with a two-sided p < .05 and the share
that survives Benjamini-Hochberg control of the false discovery rate at .05 within the
comparison.

Validity and the comparison: each within-corpus comparison is placed in a tier by the
macro-F1 of its weaker group, and the tiers are joined to the estimator results of Study 2
(results/real_summary.csv, stratified validation, n_v = 100 per group).

Can the validation sample itself establish within-group validity? For n_v = 50, 100 and 200
hand-coded items per group, macro-F1 and accuracy are re-estimated from repeated validation
draws; reported are their bias, standard deviation and how often the weaker group's tier is
classified correctly.

Output: results/validity_group.csv, validity_category.csv, validity_pattern.csv,
        validity_tiers.csv, validity_sampling.csv
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_real_coders import ANN, CODES, K, SEED, CORPORA, fewshot_ids, halves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
N_PERM, NV, R_SAMPLE = 1000, [50, 100, 200], 200
TIERS = [(0.0, 0.20, "< .20"), (0.20, 0.35, ".20-.35"), (0.35, 0.50, ".35-.50"), (0.50, 1.01, ">= .50")]


def load_probs():
    shards = {}
    for d in ANN:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(p, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            shards[(m["model"], m["condition"], m["corpus"])] = {
                "ids": z["ids"], "y": np.searchsorted(CODES, z["gold"]), "probs": z["probs"].astype(float),
                "year": z["year"], "n_words": z["n_words"]}
    keep = {}
    for c in CORPORA:
        ids = next(v["ids"] for k, v in shards.items() if k[2] == c)
        keep[c] = ~np.isin(ids, list(fewshot_ids(c)))
    for (m, cond, c), v in shards.items():
        for f in ("ids", "y", "probs", "year", "n_words"):
            v[f] = v[f][keep[c]]
        v["pred"] = v["probs"].argmax(1)
    return shards


def prf(y, pred):
    """Per-topic precision, recall, F1 and support (topics never predicted get precision 0)."""
    c = np.zeros((K, K)); np.add.at(c, (y, pred), 1)
    tp, sup, npred = np.diag(c), c.sum(1), c.sum(0)
    p = np.divide(tp, npred, out=np.zeros(K), where=npred > 0)
    r = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(K), where=(p + r) > 0)
    return p, r, f, sup


def summary(y, probs):
    pred = probs.argmax(1)
    p, r, f, sup = prf(y, pred)
    present = sup > 0
    conf = probs.max(1); hit = (pred == y).astype(float)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    onehot = np.eye(K)[y]
    return {"n": int(len(y)), "accuracy": float(hit.mean()),
            "macro_precision": float(p[present].mean()), "macro_recall": float(r[present].mean()),
            "macro_f1": float(f[present].mean()), "weighted_f1": float((f * sup).sum() / sup.sum()),
            "ece": float(ece), "brier": float(((probs - onehot) ** 2).sum(1).mean())}, (p, r, f, sup)


def recall_test(hitA, hitB, rng, n_perm):
    """Two-sided permutation p-value for a difference in recall between two groups, keeping
    each group's number of items of the topic: hitA, hitB are 0/1 arrays (code == topic)."""
    allh = np.concatenate([hitA, hitB]).astype(float)
    n, nA = len(allh), len(hitA)
    idx = np.argsort(rng.random((n_perm, n)), axis=1)
    sA = allh[idx[:, :nA]].mean(1); sB = allh[idx[:, nA:]].mean(1)
    obs = hitA.mean() - hitB.mean()
    return (1 + (np.abs(sA - sB) >= abs(obs) - 1e-12).sum()) / (n_perm + 1)


def bh(p, q=.05):
    p = np.asarray(p); m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p); thr = (np.arange(1, m + 1) / m) * q
    ok = np.where(p[order] <= thr)[0]
    out = np.zeros(m, dtype=bool)
    if len(ok):
        out[order[:ok.max() + 1]] = True
    return out


def tier(x):
    return next(t for lo, hi, t in TIERS if lo <= x < hi)


def main():
    sh = load_probs()
    corpora = sorted({k[2] for k in sh})
    split = {}
    for ci, corpus in enumerate(corpora):
        d = sh[next(k for k in sh if k[2] == corpus)]
        split[(corpus, "period")] = halves(d["year"].astype(float), SEED + ci)
        split[(corpus, "length")] = halves(d["n_words"].astype(float), SEED + 100 + ci)
    rng_perm = np.random.default_rng(SEED + 9)
    grows, crows, prows = [], [], []
    rng = np.random.default_rng(SEED + 7)
    srows = []
    for (m, cond, corpus), d in sorted(sh.items()):
        s, _ = summary(d["y"], d["probs"])
        grows.append({"model": m, "condition": cond, "corpus": corpus, "split": "all", "group": "all", **s})
        for sp in ("period", "length"):
            g = split[(corpus, sp)]
            per = {}
            for gname, mask in (("A", ~g), ("B", g)):
                s, (p, r, f, sup) = summary(d["y"][mask], d["probs"][mask])
                per[gname] = (s, r, sup)
                grows.append({"model": m, "condition": cond, "corpus": corpus, "split": sp, "group": gname, **s})
                for k in range(K):
                    if sup[k] > 0 or (d["pred"][mask] == k).any():
                        crows.append({"model": m, "condition": cond, "corpus": corpus, "split": sp, "group": gname,
                                      "topic": int(CODES[k]), "support": int(sup[k]), "precision": float(p[k]),
                                      "recall": float(r[k]), "f1": float(f[k])})
            # recall differences against a reference that keeps each topic's count in each half:
            # the items of topic k are reassigned to the halves at random, nA_k to A and nB_k to B,
            # and recall is recomputed (N_PERM draws); two-sided p-values, Benjamini-Hochberg at .05
            # over the topics tested in this comparison
            both = (per["A"][2] >= 5) & (per["B"][2] >= 5)
            obs = per["A"][1] - per["B"][1]
            pvals = np.ones(K)
            for k in np.where(both)[0]:
                items = np.where(d["y"] == k)[0]
                hit = d["pred"][items] == k
                pvals[k] = recall_test(hit[~g[items]], hit[g[items]], rng_perm, N_PERM)
            tested = np.where(both)[0]
            out = np.zeros(K, dtype=bool); out[tested] = pvals[tested] < .05
            out_bh = np.zeros(K, dtype=bool); out_bh[tested] = bh(pvals[tested])
            prows.append({"model": m, "condition": cond, "corpus": corpus, "split": sp,
                          "topics_tested": int(both.sum()), "topics_systematic": int(out.sum()),
                          "share_systematic": float(out.sum() / both.sum()) if both.any() else "",
                          "topics_systematic_bh": int(out_bh.sum()),
                          "share_systematic_bh": float(out_bh.sum() / both.sum()) if both.any() else "",
                          "max_abs_recall_diff": float(np.abs(obs[both]).max()) if both.any() else "",
                          "weaker_macro_f1": min(per["A"][0]["macro_f1"], per["B"][0]["macro_f1"]),
                          "macro_f1_gap": abs(per["A"][0]["macro_f1"] - per["B"][0]["macro_f1"])})
            # can a validation sample of n_v per group establish the weaker group's validity?
            if sp == "period":
                true_w = min(per["A"][0]["macro_f1"], per["B"][0]["macro_f1"])
                for nv in NV:
                    est_f1, est_acc, ok = [], [], []
                    for _ in range(R_SAMPLE):
                        vals = []
                        for mask in (~g, g):
                            idx = rng.choice(np.where(mask)[0], nv, replace=False)
                            s_, _ = summary(d["y"][idx], d["probs"][idx])
                            vals.append(s_)
                        w = min(v["macro_f1"] for v in vals)
                        est_f1.append(w); est_acc.append(min(v["accuracy"] for v in vals))
                        ok.append(tier(w) == tier(true_w))
                    srows.append({"model": m, "condition": cond, "corpus": corpus, "n_v": nv,
                                  "true_weaker_macro_f1": true_w, "est_mean": float(np.mean(est_f1)),
                                  "est_bias": float(np.mean(est_f1) - true_w), "est_sd": float(np.std(est_f1)),
                                  "tier_correct": float(np.mean(ok))})
        print(m, cond, corpus, flush=True)
    # the same test between corpora (each corpus a group); the transfer classifier is left out,
    # because between corpora it codes each pair with a model trained outside the pair
    import itertools
    for (m, cond) in sorted({(k[0], k[1]) for k in sh}):
        if m == "tfidf-logreg-transfer":
            continue
        for ca, cb in itertools.combinations(corpora, 2):
            a, b = sh[(m, cond, ca)], sh[(m, cond, cb)]
            pv, tested = [], 0
            for k in range(K):
                ia, ib = a["y"] == k, b["y"] == k
                if ia.sum() >= 5 and ib.sum() >= 5:
                    pv.append(recall_test(a["pred"][ia] == k, b["pred"][ib] == k, rng_perm, N_PERM))
            pv = np.array(pv)
            prows.append({"model": m, "condition": cond, "corpus": f"{ca}|{cb}", "split": "between",
                          "topics_tested": int(len(pv)), "topics_systematic": int((pv < .05).sum()),
                          "share_systematic": float((pv < .05).mean()) if len(pv) else "",
                          "topics_systematic_bh": int(bh(pv).sum()),
                          "share_systematic_bh": float(bh(pv).mean()) if len(pv) else "",
                          "max_abs_recall_diff": "", "weaker_macro_f1": "", "macro_f1_gap": ""})
    write("validity_group.csv", grows); write("validity_category.csv", crows)
    write("validity_pattern.csv", prows); write("validity_sampling.csv", srows)

    # tiers joined to the estimator results of Study 2
    summ = list(csv.DictReader(open(os.path.join(RES, "real_summary.csv"))))
    pat = {(r["model"], r["condition"], f'{r["corpus"]}:{r["split"]}'): r for r in prows}
    trows = []
    for r in summ:
        if r["design"] != "within" or r["sampling"] != "stratified" or r["n_v"] != "100":
            continue
        k = (r["model"], r["condition"], r["comparison"])
        if k not in pat:
            continue
        trows.append({**{f: r[f] for f in ("model", "condition", "comparison", "estimator", "mean_tv_error",
                                           "focal_bias", "focal_rmse", "focal_coverage", "allcat_coverage")},
                      "weaker_macro_f1": pat[k]["weaker_macro_f1"], "tier": tier(pat[k]["weaker_macro_f1"]),
                      "share_systematic": pat[k]["share_systematic"]})
    write("validity_tiers.csv", trows)


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
