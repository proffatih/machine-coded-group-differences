#!/usr/bin/env python3
"""Robustness: invariance tested inside each corpus, where the instrument is identical.

In the main analysis the length and period groupings are cut over the pooled corpora,
so the two groups differ in corpus mix, and with it in the source line of the
language-model prompt and in the corpus-specific supervised classifier. Here each
corpus is split at its own median length and its own median year, so both groups see
the same prompt, the same few-shot examples and the same trained classifier. The
per-corpus likelihood-ratio statistics are summed into one test per coder, condition
and grouping; its null distribution is the sum of the corpora's independent
within-topic permutation draws.

Output: results/within_corpus_invariance.csv, results/metrics/within_corpus_summary.json
"""
import importlib.util, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("inv", os.path.join(HERE, "03_invariance.py"))
inv = importlib.util.module_from_spec(spec); spec.loader.exec_module(inv)

MIN_N = 100


def split_at_median(x):
    """Two halves at the median; ties go to the smaller half so neither group is empty."""
    med = np.median(x)
    hi = x > med if (x > med).sum() >= (x >= med).sum() / 2 else x >= med
    return np.where(hi, "upper", "lower")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    if a.ann_dir:
        inv.ANN = os.path.abspath(a.ann_dir)
    if a.out_dir:
        inv.OUT = os.path.abspath(a.out_dir)
        os.makedirs(os.path.join(inv.OUT, "metrics"), exist_ok=True)
    recs = inv.load_all()
    rows = []
    for r in recs:
        te = r["split"] == "test"
        y, pred = r["y"][te], r["pred"][te]
        correct = (y == pred).astype(float)
        topic = inv.CODES[y]
        cuts = {"length": r["n_words"][te]}
        yrs = r["year"][te]
        if (yrs > 0).mean() > .9:
            cuts["era"] = yrs
        for gname, x in cuts.items():
            ok = x > 0 if gname == "era" else np.ones(len(x), bool)
            g = split_at_median(x[ok])
            if min((g == v).sum() for v in ("lower", "upper")) < MIN_N:
                continue
            lad = inv.invariance_ladder(y[ok], pred[ok], g, keep_null=True)
            dif = inv.dif_lr_tests(correct[ok], topic[ok], g)
            rows.append({"model": r["model"], "condition": r["condition"], "corpus": r["corpus"],
                         "grouping": gname, "n": int(ok.sum()),
                         "lr_chi2": lad["lr_chi2"], "df": lad["df"], "p": lad["p"],
                         "null_mean": lad["null_mean"], "_null": lad["_null"],
                         "uniform_p": dif["uniform_p"], "nonuniform_p": dif["nonuniform_p"]})
    comb = {}
    for r in rows:
        k = (r["model"], r["condition"], r["grouping"])
        c = comb.setdefault(k, {"stat": 0.0, "null": 0.0, "k": 0, "n": 0})
        c["stat"] += r["lr_chi2"]; c["null"] = c["null"] + r.pop("_null"); c["k"] += 1
        c["n"] += r["n"]
    for c in comb.values():
        c["p"] = float((1 + (c["null"] >= c["stat"]).sum()) / (len(c["null"]) + 1))
        c["ratio"] = float(c["stat"] / c["null"].mean())
        c["cmi"] = float((c["stat"] - c["null"].mean()) / (2 * c["n"]))

    out = os.path.join(inv.OUT, "within_corpus_invariance.csv")
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {"min_group_n": MIN_N, "n_corpus_tests": len(rows), "by_family": {}}
    for fam in ("llm", "sup"):
        cs = {k: v for k, v in comb.items() if k[0].startswith("tfidf") == (fam == "sup")}
        if not cs:
            continue
        ps = [v["p"] for v in cs.values()]
        rs = [r for r in rows if r["model"].startswith("tfidf") == (fam == "sup")]
        summ["by_family"][fam] = {
            "combined_tests": len(cs),
            "combined_rejected_share": float(np.mean([p < .05 for p in ps])),
            "combined_max_p": float(max(ps)),
            "combined_stat_over_null_range": [min(v["ratio"] for v in cs.values()),
                                              max(v["ratio"] for v in cs.values())],
            "combined_stat_over_null_mean": float(np.mean([v["ratio"] for v in cs.values()])),
            "combined_cmi_by_grouping": {g: float(np.mean([v["cmi"] for k, v in cs.items() if k[2] == g]))
                                         for g in sorted({k[2] for k in cs})},
            "corpus_tests": len(rs),
            "corpus_rejected_share": float(np.mean([r["p"] < .05 for r in rs])),
            "corpus_nonuniform_share": float(np.mean([r["nonuniform_p"] < .05 for r in rs])),
            "corpus_uniform_share": float(np.mean([r["uniform_p"] < .05 for r in rs])),
            "groupings": sorted({k[2] for k in cs}),
            "corpora_per_grouping": {g: sorted({r["corpus"] for r in rs if r["grouping"] == g})
                                     for g in sorted({k[2] for k in cs})},
        }
    json.dump(summ, open(os.path.join(inv.OUT, "metrics", "within_corpus_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
