#!/usr/bin/env python3
"""Interval estimates for every headline quantity, with the right clustering.

The first version of this study reported bare point estimates, and the share of cells
in which the pooled correction did worse than no correction was quoted as a raw count.
Neither is adequate. The cells are not independent: each model contributes cells to
every grouping, each corpus appears under several models, and the same calibration
split is reused throughout. A count over such cells is a description of this design,
not evidence about instruments in general.

Two devices are used.

  Item bootstrap -- for quantities defined within a corpus (agreement, subgroup gaps),
  resample items with replacement inside each corpus and subgroup, preserving the
  calibration/test split.

  Cluster bootstrap over models -- for claims about instruments as a class, resample
  whole models with replacement. The model is the unit that generalises; cells within a
  model are not independent replications of anything.

Usage: python code/08_uncertainty.py [--boot 2000]
Output: results/metrics/intervals.json
"""
import argparse, csv, collections, glob, json, os, sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

LLM = {"olmo2-1b", "qwen2.5-1.5b", "smollm2-1.7b",
       "granite3.1-2b", "qwen2.5-3b", "phi3.5-mini"}
SUP = {"tfidf-logreg", "tfidf-svm"}


def rd(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def ci(samples, lo=2.5, hi=97.5):
    a = np.asarray([s for s in samples if np.isfinite(s)])
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if len(a) else (None, None)


def cluster_boot(by_cluster, stat, rng, B):
    """Resample whole clusters (models) with replacement and recompute `stat`."""
    keys = list(by_cluster)
    out = []
    for _ in range(B):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        pooled = [v for k in draw for v in by_cluster[keys[k]]]
        if pooled:
            out.append(stat(pooled))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {"n_bootstrap": args.boot, "clustering": "models resampled as whole units"}

    # ---- agreement, by instrument family -------------------------------------
    cacc = rd("corpus_accuracy.csv")
    for fam, members in (("llm", LLM), ("supervised", SUP)):
        by = collections.defaultdict(list)
        for r in cacc:
            if r["model"] in members and r["condition"] in ("zs", "sup"):
                by[r["model"]].append(float(r["accuracy"]))
        if not by:
            continue
        point = float(np.mean([v for vs in by.values() for v in vs]))
        boot = cluster_boot(by, lambda xs: float(np.mean(xs)), rng, args.boot)
        res[f"accuracy_{fam}"] = {"point": point, "ci95": ci(boot), "n_models": len(by)}

    # ---- invariance rejection, by family --------------------------------------
    lad = rd("invariance_ladder.csv")
    for fam, members in (("llm", LLM), ("supervised", SUP)):
        sub = [r for r in lad if r["model"] in members]
        if sub:
            res[f"invariance_rejected_{fam}"] = {
                "point": float(np.mean([float(r["p"]) < .05 for r in sub])),
                "n_tests": len(sub),
                "max_p": float(max(float(r["p"]) for r in sub)),
            }

    # ---- pooled correction on between-group differences -----------------------
    prev = rd("prevalence_bias.csv")
    for fam, members in (("llm", LLM), ("supervised", SUP)):
        by_worse, by_delta = collections.defaultdict(list), collections.defaultdict(list)
        for r in prev:
            if r["model"] not in members or not r["group"].startswith("DIFF:"):
                continue
            n, p = float(r["tv_naive"]), float(r["tv_pooled_correction"])
            by_worse[r["model"]].append(float(p > n))
            by_delta[r["model"]].append(p - n)
        if not by_worse:
            continue
        pt_share = float(np.mean([v for vs in by_worse.values() for v in vs]))
        pt_delta = float(np.mean([v for vs in by_delta.values() for v in vs]))
        res[f"pooled_worse_share_{fam}"] = {
            "point": pt_share,
            "ci95": ci(cluster_boot(by_worse, lambda xs: float(np.mean(xs)), rng, args.boot)),
            "n_models": len(by_worse)}
        res[f"pooled_excess_error_{fam}"] = {
            "point": pt_delta,
            "ci95": ci(cluster_boot(by_delta, lambda xs: float(np.mean(xs)), rng, args.boot)),
            "note": "mean increase in total-variation error from applying the pooled "
                    "correction to a between-subgroup difference; positive means harmful"}

    # ---- subgroup-specific correction ----------------------------------------
    for fam, members in (("llm", LLM), ("supervised", SUP)):
        by = collections.defaultdict(list)
        for r in prev:
            if r["model"] in members and r["group"].startswith("DIFF:"):
                by[r["model"]].append(float(r["tv_group_correction"]) - float(r["tv_naive"]))
        if by:
            res[f"subgroup_excess_error_{fam}"] = {
                "point": float(np.mean([v for vs in by.values() for v in vs])),
                "ci95": ci(cluster_boot(by, lambda xs: float(np.mean(xs)), rng, args.boot)),
                "n_models": len(by)}

    # ---- subgroup accuracy gaps, by grouping ---------------------------------
    acc = rd("subgroup_accuracy.csv")
    cells = collections.defaultdict(dict)
    for r in acc:
        cells[(r["model"], r["condition"], r["grouping"])][r["group"]] = float(r["accuracy"])
    gaps = collections.defaultdict(lambda: collections.defaultdict(list))
    for (m, c, g), v in cells.items():
        if len(v) > 1:
            gaps[g][m].append(max(v.values()) - min(v.values()))
    res["accuracy_gap_by_grouping"] = {}
    for g, by in gaps.items():
        res["accuracy_gap_by_grouping"][g] = {
            "point": float(np.mean([v for vs in by.values() for v in vs])),
            "ci95": ci(cluster_boot(by, lambda xs: float(np.mean(xs)), rng, args.boot))}

    json.dump(res, open(os.path.join(RES, "metrics", "intervals.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
