#!/usr/bin/env python3
"""Numbers quoted from the extensions (Online Appendix G): results/ext_summary.json."""
import json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
R3 = lambda x: round(float(x), 4)
VALID = ["dd_group", "dd_tuned", "hand"]


def worse(df, a, b, by):
    p = df.pivot_table(index=["model", "condition", "design", "comparison"] + [c for c in by if c not in ("design",)],
                       columns="estimator", values="mean_tv_error")
    return (p[a] > p[b]).groupby(level=by).mean()


def main():
    o = {}
    # population estimand
    d = pd.read_csv(os.path.join(RES, "population.csv"))
    c = d[d.focal_coverage.notna()].groupby(["design", "n_v", "estimator"]).focal_coverage.mean()
    o["pop_valid_cov"] = [R3(c.loc[:, :, VALID].min()), R3(c.loc[:, :, VALID].max())]
    o["pop_cal_cov"] = [R3(c.loc[:, :, "cal_group"].min()), R3(c.loc[:, :, "cal_group"].max())]
    w = worse(d, "bc_pooled", "naive", ["design", "n_v"])
    o["pop_bcp_worse"] = [R3(w.min()), R3(w.max())]
    o["pop_naive_cov"] = {k: R3(v) for k, v in c.loc[:, :, "naive"].groupby(level="design").mean().items()}
    # real draws: regularised back-calculation, bootstrap coverage, joint coverage
    r = pd.read_csv(os.path.join(RES, "ext_real_draws.csv"))
    for e in ("bc_shrink", "bc_group", "bc_pooled"):
        o[f"{e}_worse_than_naive"] = {k: R3(v) for k, v in worse(r, e, "naive", ["design"]).items()}
        o[f"{e}_worse_than_tuned"] = {k: R3(v) for k, v in worse(r, e, "dd_tuned", ["design"]).items()}
    cv = r[r.focal_coverage.notna()].groupby(["design", "estimator"]).focal_coverage.mean()
    o["boot_cov"] = {e: {dz: R3(cv[(dz, e)]) for dz in ("within", "between")} for e in ("bc_shrink", "bc_group")}
    jc = r[r.joint_coverage_bonferroni.notna()].groupby(["design", "estimator"]).joint_coverage_bonferroni.mean()
    o["joint_cov"] = {e: {dz: R3(jc[(dz, e)]) for dz in ("within", "between")} for e in ("dd_tuned", "cal_group", "hand", "naive")}
    # prevalence bins
    b = pd.read_csv(os.path.join(RES, "ext_bins.csv")).groupby(["estimator", "bin"])[["coverage", "half_width"]].mean()
    o["bins"] = {e: {bn: {"cov": R3(b.loc[(e, bn), "coverage"]), "hw": R3(b.loc[(e, bn), "half_width"])}
                     for bn in ("<.02", ".02-.05", ".05-.15", ">=.15")} for e in ("dd_tuned", "cal_group", "hand", "naive")}
    # simulation with the real matrices
    s = pd.read_csv(os.path.join(RES, "ext_empirical_sim.csv"))
    p = s.pivot_table(index=["model", "condition", "design", "comparison", "version", "n_v"], columns="estimator",
                      values="mean_tv_error")
    w = (p.bc_pooled > p.naive).groupby(level=["version"]).agg(["min", "max"]) if False else None
    wv = (p.bc_pooled > p.naive).groupby(level=["version", "design", "n_v"]).mean()
    o["empsim_bcp_worse"] = {v: [R3(wv.loc[v].min()), R3(wv.loc[v].max())] for v in ("empirical", "repaired")}
    sc = s[s.focal_coverage.notna()].groupby(["version", "design", "n_v", "estimator"]).focal_coverage.mean()
    o["empsim_valid_cov"] = [R3(sc.loc[:, :, :, ["cal_group", "dd_tuned"]].min()), R3(sc.loc[:, :, :, ["cal_group", "dd_tuned"]].max())]
    sm = s.drop_duplicates(["model", "condition", "comparison", "version"]).groupby("version").sigma_min.median()
    o["empsim_sigma_min_median"] = {k: R3(v) for k, v in sm.items()}
    # other coders
    q = pd.read_csv(os.path.join(RES, "coders_ext_quality.csv"))
    q["kind"] = q.model.where(q.model.str.startswith("synthetic"), "llm+prior")
    qq = q.groupby("kind")[["accuracy", "accuracy_original", "unassigned", "unassigned_original"]].mean()
    o["coders_quality"] = {k: {c_: R3(v[c_]) for c_ in qq.columns} for k, v in qq.iterrows()}
    x = pd.read_csv(os.path.join(RES, "coders_ext.csv"))
    x["kind"] = x.model.where(x.model.str.startswith("synthetic"), "llm+prior")
    px = x.pivot_table(index=["kind", "model", "condition", "design", "comparison"], columns="estimator", values="mean_tv_error")
    o["coders_bcp_worse"] = {f"{k}|{dz}": R3(v) for (k, dz), v in (px.bc_pooled > px.naive).groupby(level=["kind", "design"]).mean().items()}
    o["coders_tuned_beats_naive"] = {f"{k}|{dz}": R3(v) for (k, dz), v in (px.dd_tuned < px.naive).groupby(level=["kind", "design"]).mean().items()}
    xc = x[x.focal_coverage.notna()].groupby(["kind", "design", "estimator"]).focal_coverage.mean()
    o["coders_cov"] = {f"{k}|{dz}|{e}": R3(v) for (k, dz, e), v in xc.items()}
    rx = x.pivot_table(index=["kind", "model", "condition", "design", "comparison"], columns="estimator", values="focal_rmse")
    o["coders_tuned_over_hand"] = {f"{k}|{dz}": R3(v) for (k, dz), v in (rx.dd_tuned / rx.hand).groupby(level=["kind", "design"]).median().items()}
    json.dump(o, open(os.path.join(RES, "ext_summary.json"), "w"), indent=1)
    print(json.dumps(o, indent=1))


if __name__ == "__main__":
    main()
