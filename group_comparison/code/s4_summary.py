#!/usr/bin/env python3
"""Numbers quoted in the text about within-group validity (results/validity_summary.json)."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def fam(m):
    if m in ("tfidf-logreg", "tfidf-svm"):
        return "incorpus"
    if m == "tfidf-logreg-transfer":
        return "transfer"
    return "llm_large" if any(s in m for s in ("7b", "8b")) else "llm_small"


def main():
    g = pd.read_csv(os.path.join(RES, "validity_group.csv")); g["fam"] = g.model.map(fam)
    p = pd.read_csv(os.path.join(RES, "validity_pattern.csv")); p["fam"] = p.model.map(fam)
    t = pd.read_csv(os.path.join(RES, "validity_tiers.csv")); t["fam"] = t.model.map(fam)
    s = pd.read_csv(os.path.join(RES, "validity_sampling.csv"))
    out = {}
    a = g[g.split == "all"]
    for f, d in a.groupby("fam"):
        out[f"corpus_{f}"] = {k: [round(float(d[k].min()), 3), round(float(d[k].mean()), 3), round(float(d[k].max()), 3)]
                              for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "ece", "brier")}
    w = g[g.split != "all"]
    out["group_macro_f1_range"] = [round(float(w.macro_f1.min()), 3), round(float(w.macro_f1.max()), 3)]
    pw = p[p.split != "between"]
    out["macro_f1_gap_mean"] = round(float(pw.macro_f1_gap.mean()), 3)
    out["macro_f1_gap_max"] = round(float(pw.macro_f1_gap.max()), 3)
    out["share_systematic"] = {sp: round(float(d.share_systematic.mean()), 3) for sp, d in p.groupby("split")}
    out["share_systematic_bh"] = {sp: round(float(d.share_systematic_bh.mean()), 3) for sp, d in p.groupby("split")}
    out["any_systematic_bh"] = {sp: round(float((d.topics_systematic_bh > 0).mean()), 3) for sp, d in p.groupby("split")}
    out["share_systematic_by_fam"] = {sp: {f: round(float(e.share_systematic.mean()), 3) for f, e in d.groupby("fam")}
                                      for sp, d in p.groupby("split")}
    out["topics_tested_mean"] = {sp: round(float(d.topics_tested.mean()), 1) for sp, d in p.groupby("split")}
    comp = t.drop_duplicates(["model", "condition", "comparison"])
    out["n_comparisons"] = int(len(comp))
    out["tier_counts"] = comp.tier.value_counts().to_dict()
    cov = t[t.estimator.isin(["naive", "rg_group", "cal_group", "dd_group", "dd_tuned", "hand"])]
    out["coverage_by_tier"] = {e: {k: round(float(v), 3) for k, v in d.groupby("tier").focal_coverage.mean().items()}
                               for e, d in cov.groupby("estimator")}
    piv = t.pivot_table(index=["model", "condition", "comparison", "tier"], columns="estimator", values="focal_rmse")
    ratio = (piv.dd_tuned / piv.hand).groupby(level="tier")
    out["tuned_over_hand_by_tier"] = {k: [round(float(v.quantile(.25)), 3), round(float(v.median()), 3),
                                          round(float(v.quantile(.75)), 3)] for k, v in ratio}
    calr = (piv.cal_group / piv.hand).groupby(level="tier")
    out["cal_over_hand_median_by_tier"] = {k: round(float(v.median()), 3) for k, v in calr}
    tv = t.pivot_table(index="tier", columns="estimator", values="mean_tv_error", aggfunc="mean")
    out["tv_by_tier"] = {e: {k: round(float(v), 3) for k, v in tv[e].items()} for e in ("naive", "bc_pooled", "cal_group", "dd_tuned", "hand")}
    out["sampling"] = {int(n): {"bias": round(float(d.est_bias.mean()), 3), "sd": round(float(d.est_sd.mean()), 3),
                                "tier_correct": round(float(d.tier_correct.mean()), 3)} for n, d in s.groupby("n_v")}
    json.dump(out, open(os.path.join(RES, "validity_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
