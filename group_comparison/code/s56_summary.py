#!/usr/bin/env python3
"""Numbers quoted in the text from Study 2b (human-code error) and Study 3 (other schemes).
Output: results/label_noise_summary.json, results/schemes_summary.json"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
R3 = lambda x: round(float(x), 3)


def noise():
    d = pd.read_csv(os.path.join(RES, "label_noise.csv"))
    d["setting"] = d.noise + d.eps.fillna("").astype(str).str.replace("nan", "")
    out = {}
    for (design, setting), g in d.groupby(["design", "setting"]):
        o = {}
        for e, h in g.groupby("estimator"):
            o[e] = {"tv_latent": R3(h.tv_error_latent.mean()), "tv_human": R3(h.tv_error_human_target.mean()),
                    "cov_latent": R3(h.focal_coverage_latent.mean()) if h.focal_coverage_latent.notna().any() else None,
                    "cov_human": R3(h.focal_coverage_human_target.mean()) if h.focal_coverage_human_target.notna().any() else None,
                    "focal_bias_latent": R3(h.focal_bias_latent.mean())}
        a = g[g.estimator == "hand"]
        o["_target"] = {"human_target_tv_gap": R3(a.human_target_tv_gap.mean()),
                        "true_tv_difference": R3(a.true_tv_difference.mean()),
                        "focal_attenuation_median": R3(pd.to_numeric(a.focal_attenuation, errors="coerce").median())}
        out.setdefault(design, {})[setting] = o
    json.dump(out, open(os.path.join(RES, "label_noise_summary.json"), "w"), indent=1)
    return out


def schemes():
    out = {}
    s = pd.read_csv(os.path.join(RES, "sim_schemes.csv"))
    sim = {}
    for K, g in s.groupby("K"):
        cov = g[g.focal_coverage.notna()]
        r = g.pivot_table(index=["kappa", "where", "n_v"], columns="estimator", values="focal_rmse")
        sim[int(K)] = {
            "max_abs_bias_valid": R3(g[g.estimator.isin(["cal_group", "dd_group", "dd_tuned"])].focal_bias.abs().max()),
            "min_focal_cov": {e: R3(h.focal_coverage.min()) for e, h in cov.groupby("estimator")},
            "mean_focal_cov": {e: R3(h.focal_coverage.mean()) for e, h in cov.groupby("estimator")},
            "n_cond": int(len(r)),
            "n_cond_below_925": {e: int((h.focal_coverage < .925).sum()) for e, h in cov.groupby("estimator")
                                 if e in ("cal_group", "dd_group", "dd_tuned", "hand")},
            "bc_pooled_worse_than_naive": {w: R3(h.share_worse_than_naive.mean())
                                           for w, h in g[g.estimator == "bc_pooled"].groupby("where")},
            "tv": {e: R3(h.mean_tv_error.mean()) for e, h in g.groupby("estimator")},
            "tuned_over_hand": {str(k): R3(v) for k, v in (r.dd_tuned / r.hand).groupby(level="kappa").mean().items()},
            "cal_over_hand": {str(k): R3(v) for k, v in (r.cal_group / r.hand).groupby(level="kappa").mean().items()}}
    out["sim"] = sim
    if os.path.exists(os.path.join(RES, "real_schemes.csv")):
        d = pd.read_csv(os.path.join(RES, "real_schemes.csv"))
        d["kind"] = d.scheme.str.replace(r"\d+$", "", regex=True).where(d.scheme == "focal_binary", d.scheme)
        real = {}
        for (scheme, design), g in d.groupby(["scheme", "design"]):
            piv = g.pivot_table(index=["model", "condition", "comparison", "partition"], columns="estimator",
                                values="focal_rmse")
            cov = g[g.focal_coverage.notna()]
            real.setdefault(scheme, {})[design] = {
                "n": int(len(piv)),
                "bc_pooled_worse_than_naive": R3(g[g.estimator == "bc_pooled"].share_worse_than_naive.mean()),
                "bc_group_worse_than_naive": R3(g[g.estimator == "bc_group"].share_worse_than_naive.mean()),
                "focal_cov": {e: R3(h.focal_coverage.mean()) for e, h in cov.groupby("estimator")},
                "allcat_cov": {e: R3(h.allcat_coverage.mean()) for e, h in cov.groupby("estimator")},
                "tv": {e: R3(h.mean_tv_error.mean()) for e, h in g.groupby("estimator")},
                "tuned_over_hand_median": R3((piv.dd_tuned / piv.hand).median()),
                "tuned_better_than_naive_tv": R3((g[g.estimator == "dd_tuned"].mean_tv_error.values
                                                  < g[g.estimator == "naive"].mean_tv_error.values).mean())}
        out["real"] = real
    json.dump(out, open(os.path.join(RES, "schemes_summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    if os.path.exists(os.path.join(RES, "label_noise.csv")):
        print(json.dumps(noise(), indent=1)[:6000])
    print(json.dumps(schemes(), indent=1)[:6000])
