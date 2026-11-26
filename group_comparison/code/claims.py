#!/usr/bin/env python3
"""Every number the manuscript states, computed from the result files (results/claims.json)."""
import csv, json, os, collections, itertools
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rd = lambda n: list(csv.DictReader(open(os.path.join(RES, n))))
C = {}
ACC = ("0.35", "0.55", "0.75")

# ---------------- simulation
sim = rd("sim_summary.csv")
S = [r for r in sim if r["sampling"] == "stratified"]
P = [r for r in sim if r["sampling"] == "pooled"]
cond = lambda r: (r["acc"], r["delta"], r["where"], r["n_v"])
get = lambda rows, **k: [r for r in rows if all(r[a] == str(b) for a, b in k.items())]
num = lambda r, f: float(r[f])
C["sim_conditions"] = len({cond(r) for r in S})
C["sim_reps"] = int(S[0]["reps"])
C["sim_mcse_95"] = float(np.sqrt(.95 * .05 / C["sim_reps"]))
for a in ACC:
    for w, d in (("none", "0.0"), ("rare", "0.2"), ("frequent", "0.2"), ("all", "0.2")):
        for e in ("bc_pooled", "bc_group", "naive", "dd_tuned", "cal_group"):
            C[f"sim_{e}_{a}_{w}_nv1000"] = num(get(S, acc=a, delta=d, where=w, n_v=1000, estimator=e)[0], "mean_tv_error")
    # attenuation slope of the uncorrected estimate (focal topic), averaged over n_v
    for w, d in (("none", "0.0"), ("frequent", "0.2"), ("all", "0.2")):
        C[f"sim_naive_slope_{a}_{w}"] = float(np.mean([num(r, "focal_slope") for r in get(S, acc=a, delta=d, where=w, estimator="naive")]))
    for e in ("dd_tuned", "cal_group", "dd_group", "rg_group", "bc_group"):
        t = np.mean([num(r, "mean_tv_error") for r in S if r["acc"] == a and r["estimator"] == e])
        h = np.mean([num(r, "mean_tv_error") for r in S if r["acc"] == a and r["estimator"] == "hand"])
        C[f"sim_{e}_over_hand_{a}"] = float(t / h)
for e in ("naive", "dd_group", "dd_tuned", "hand", "cal_group", "rg_group"):
    v = [num(r, "focal_coverage") for r in S if r["estimator"] == e]
    C[f"sim_cov_{e}_min"], C[f"sim_cov_{e}_max"] = min(v), max(v)
    C[f"sim_allcov_{e}_min"] = min(num(r, "allcat_coverage") for r in S if r["estimator"] == e)
    C[f"sim_rarecov_{e}_min"] = min(num(r, "rare_coverage") for r in S if r["estimator"] == e)
C["sim_tuned_below_nominal"] = sum(num(r, "focal_coverage") < .95 - 1.96 * C["sim_mcse_95"]
                                   for r in S if r["estimator"] == "dd_tuned")
best = collections.Counter()
for k in {cond(r) for r in S}:
    rows = [r for r in S if cond(r) == k and r["estimator"] != "rg_group"]
    best[min(rows, key=lambda r: num(r, "mean_tv_error"))["estimator"]] += 1
C["sim_best_counts"] = dict(best)
C["sim_ddtuned_absbias_max"] = max(abs(num(r, "focal_bias")) for r in S if r["estimator"] == "dd_tuned")
C["sim_mean_tv"] = {e: float(np.mean([num(r, "mean_tv_error") for r in S if r["estimator"] == e]))
                    for e in sorted({r["estimator"] for r in S})}
# pooled vs stratified validation sample (all validation-based estimators)
diff, pcov, pall = [], [], []
for r in P:
    st = get(S, acc=r["acc"], delta=r["delta"], where=r["where"], n_v=r["n_v"], estimator=r["estimator"])[0]
    diff.append(num(r, "mean_tv_error") / num(st, "mean_tv_error") - 1)
    pcov.append(num(r, "focal_coverage")); pall.append(num(r, "allcat_coverage"))
C["sim_pooled_vs_strat_maxreldiff"] = float(max(np.abs(diff)))
C["sim_pooled_cov_min"], C["sim_pooled_cov_max"] = min(pcov), max(pcov)
C["sim_pooled_allcov_min"] = min(pall)
C["sim_pooled_estimators"] = sorted({r["estimator"] for r in P})

# ---------------- real coders
real = rd("real_summary.csv")
FAM = lambda m: "supin" if m in ("tfidf-logreg", "tfidf-svm") else ("xfer" if m == "tfidf-logreg-transfer" else "llm")
SPLIT = lambda r: "between" if r["design"] == "between" else r["comparison"].split(":")[1]
for r in real:
    r["fam"], r["split"] = FAM(r["model"]), SPLIT(r)
RS = [r for r in real if r["sampling"] == "stratified"]
C["real_coders"] = len({(r["model"], r["condition"]) for r in real})
C["real_within_comparisons"] = len({(r["model"], r["condition"], r["comparison"]) for r in real if r["design"] == "within"})
C["real_between_comparisons"] = len({(r["model"], r["condition"], r["comparison"]) for r in real if r["design"] == "between"})
ESTS = ("naive", "bc_pooled", "bc_group", "rg_group", "cal_group", "dd_group", "dd_tuned", "hand")
for sp in ("period", "length", "between"):
    for fm in ("llm", "xfer", "supin"):
        for nv in ("50", "100", "200"):
            s = [r for r in RS if r["split"] == sp and r["fam"] == fm and r["n_v"] == nv]
            for e in ESTS:
                v = [r for r in s if r["estimator"] == e]
                p = f"real_{sp}_{fm}_{nv}_{e}"
                C[p + "_tv"] = float(np.mean([num(r, "mean_tv_error") for r in v]))
                C[p + "_rmse"] = float(np.sqrt(np.mean([num(r, "focal_rmse") ** 2 for r in v])))
                if v[0]["focal_coverage"] != "":
                    C[p + "_cov"] = float(np.mean([num(r, "focal_coverage") for r in v]))
                    C[p + "_allcov"] = float(np.mean([num(r, "allcat_coverage") for r in v]))
# per-comparison win/loss shares
for d in ("within", "between"):
    for nv in ("50", "200"):
        k = collections.defaultdict(dict)
        for r in RS:
            if r["design"] == d and r["n_v"] == nv:
                k[(r["model"], r["condition"], r["comparison"])][r["estimator"]] = num(r, "mean_tv_error")
        C[f"real_{d}_{nv}_bcpooled_worse_than_naive"] = float(np.mean([v["bc_pooled"] > v["naive"] for v in k.values()]))
        C[f"real_{d}_{nv}_bcgroup_worse_than_naive"] = float(np.mean([v["bc_group"] > v["naive"] for v in k.values()]))
        C[f"real_{d}_{nv}_bcgroup_worse_than_tuned"] = float(np.mean([v["bc_group"] > v["dd_tuned"] for v in k.values()]))
        for fm in ("llm", "xfer", "supin"):
            vv = [v for (m, c, cmp), v in k.items() if FAM(m) == fm]
            C[f"real_{d}_{nv}_{fm}_tuned_beats_naive"] = float(np.mean([v["dd_tuned"] < v["naive"] for v in vv]))
# validity: coverage of the design-based estimators across all cells
cells = [r for r in RS if r["estimator"] in ("dd_group", "dd_tuned", "hand")]
cellmeans = collections.defaultdict(list)
for r in cells:
    cellmeans[(r["split"], r["fam"], r["n_v"], r["estimator"])].append(num(r, "focal_coverage"))
cm = {k: np.mean(v) for k, v in cellmeans.items()}
C["real_valid_cov_min"], C["real_valid_cov_max"] = float(min(cm.values())), float(max(cm.values()))
C["real_valid_allcov_min"] = float(min(np.mean([num(r, "allcat_coverage") for r in cells
                                               if (r["split"], r["fam"], r["n_v"], r["estimator"]) == k]) for k in cm))
calc = collections.defaultdict(list)
for r in RS:
    if r["estimator"] == "cal_group":
        calc[(r["split"], r["fam"], r["n_v"])].append(num(r, "focal_coverage"))
C["real_cal_cov_min"] = float(min(np.mean(v) for v in calc.values()))
C["real_cal_cov_max"] = float(max(np.mean(v) for v in calc.values()))
# naive: the binomial interval a researcher would report, per split x family
for sp in ("period", "length", "between"):
    for fm in ("llm", "xfer", "supin"):
        C[f"real_naivecov_{sp}_{fm}"] = float(np.mean([num(r, "focal_coverage") for r in RS
                                                       if r["split"] == sp and r["fam"] == fm and r["estimator"] == "naive"]))
# pooled validation sample vs stratified (all validation-based estimators)
st = {(r["model"], r["condition"], r["comparison"], r["n_v"], r["estimator"]): num(r, "mean_tv_error") for r in RS}
rel = [abs(num(r, "mean_tv_error") / st[(r["model"], r["condition"], r["comparison"], r["n_v"], r["estimator"])] - 1)
       for r in real if r["sampling"] == "pooled"]
C["real_pooled_vs_strat_mean_reldiff"] = float(np.mean(rel))
pc = collections.defaultdict(list)
for r in real:
    if r["sampling"] == "pooled":
        pc[r["estimator"]].append(num(r, "focal_coverage"))
C["real_pooled_cov_min"] = float(min(np.mean(v) for k, v in pc.items() if k != "cal_group"))
C["real_pooled_cov_max"] = float(max(np.mean(v) for k, v in pc.items() if k != "cal_group"))
C["real_pooled_cal_cov"] = float(np.mean(pc["cal_group"]))
# per-category binary correction in real data
rgc = collections.defaultdict(list)
for r in RS:
    if r["estimator"] == "rg_group":
        rgc[(r["split"], r["fam"], r["n_v"])].append(num(r, "focal_coverage"))
C["real_rg_cov_min"] = float(min(np.mean(v) for v in rgc.values()))
C["real_rg_cov_max"] = float(max(np.mean(v) for v in rgc.values()))
for d in ("within", "between"):
    for nv in ("50", "200"):
        k = collections.defaultdict(dict)
        for r in RS:
            if r["design"] == d and r["n_v"] == nv:
                k[(r["model"], r["condition"], r["comparison"])][r["estimator"]] = num(r, "mean_tv_error")
        C[f"real_{d}_{nv}_rg_worse_than_tuned"] = float(np.mean([v["rg_group"] > v["dd_tuned"] for v in k.values()]))
        for fm in ("llm", "xfer", "supin"):
            vv = [v for (m, c, cmp), v in k.items() if FAM(m) == fm]
            C[f"real_{d}_{nv}_{fm}_bcpooled_worse_than_naive"] = float(np.mean([v["bc_pooled"] > v["naive"] for v in vv]))
            C[f"real_{d}_{nv}_{fm}_bcgroup_worse_than_naive"] = float(np.mean([v["bc_group"] > v["naive"] for v in vv]))
C["real_true_tv_within"] = float(np.mean([num(r, "true_tv_difference") for r in RS if r["design"] == "within"]))
C["real_true_tv_between"] = float(np.mean([num(r, "true_tv_difference") for r in RS if r["design"] == "between"]))

# ---------------- population-level (oracle) quantities
orc = rd("real_oracle.csv")
for r in orc:
    r["fam"], r["split"] = FAM(r["model"]), SPLIT(r)
for grp in ("within", "between"):
    o = [r for r in orc if r["design"] == grp]
    for f in ("naive_tv_error", "shared_tv", "differential_tv", "bc_oracle_pooled_tv", "bc_oracle_group_tv"):
        C[f"orc_{grp}_{f}"] = float(np.mean([num(r, f) for r in o]))
    C[f"orc_{grp}_pooled_worse_than_group"] = float(np.mean([num(r, "bc_oracle_pooled_tv") > num(r, "bc_oracle_group_tv") for r in o]))
for fm in ("llm", "xfer", "supin"):
    o = [r for r in orc if r["fam"] == fm]
    C[f"orc_{fm}_bc_group_tv"] = float(np.mean([num(r, "bc_oracle_group_tv") for r in o]))
    C[f"orc_{fm}_singular_share"] = float(np.mean([num(r, "sigma_min_group") < 1e-4 for r in o]))
    C[f"orc_{fm}_unassigned_share"] = float(np.mean([num(r, "unassigned_topics") > 0 for r in o]))
    C[f"orc_{fm}_differential_share"] = float(np.mean([num(r, "differential_tv") / (num(r, "differential_tv") + num(r, "shared_tv")) for r in o]))
C["orc_rho_pooled_vs_differential"] = float(spearmanr([num(r, "bc_oracle_pooled_tv") - num(r, "bc_oracle_group_tv") for r in orc],
                                                      [num(r, "differential_tv") for r in orc])[0])
C["orc_rho_group_vs_sigma"] = float(spearmanr([num(r, "bc_oracle_group_tv") for r in orc],
                                              [num(r, "sigma_min_group") for r in orc])[0])
C["orc_n"] = len(orc)
inv = [r for r in orc if num(r, "sigma_min_group") >= 1e-4]
C["orc_invertible_n"] = len(inv)
C["orc_singular_share"] = 1 - len(inv) / len(orc)
C["orc_invertible_bc_group_tv_max"] = float(max(num(r, "bc_oracle_group_tv") for r in inv))
C["orc_singular_bc_group_tv"] = float(np.mean([num(r, "bc_oracle_group_tv") for r in orc if num(r, "sigma_min_group") < 1e-4]))
pinv = [r for r in orc if num(r, "sigma_min_pooled") >= 1e-4]
C["orc_pooled_invertible_n"] = len(pinv)
C["orc_rho_pooled_vs_differential_invertible"] = float(spearmanr([num(r, "bc_oracle_pooled_tv") - num(r, "bc_oracle_group_tv") for r in pinv],
                                                                 [num(r, "differential_tv") for r in pinv])[0])
C["orc_pooled_worse_than_group_invertible"] = float(np.mean([num(r, "bc_oracle_pooled_tv") > num(r, "bc_oracle_group_tv") for r in pinv]))
for sp in ("period", "length"):
    o = [r for r in orc if r["split"] == sp]
    C[f"orc_{sp}_differential_tv"] = float(np.mean([num(r, "differential_tv") for r in o]))
    C[f"orc_{sp}_shared_tv"] = float(np.mean([num(r, "shared_tv") for r in o]))
rsp = rd("real_random_split.csv")
C["rand_differential_tv"] = float(np.mean([num(r, "differential_tv") for r in rsp]))
C["rand_shared_tv"] = float(np.mean([num(r, "shared_tv") for r in rsp]))
C["rand_true_tv"] = float(np.mean([num(r, "true_tv_difference") for r in rsp]))
C["rand_n"] = len(rsp)

# topics a coder never assigns in a corpus, and the share of items they hold
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_real_coders import load as _load, counts as _counts
_un = collections.defaultdict(list)
for (m, c, corp), d in _load().items():
    cc = _counts(d["y"], d["pred"]); u = (cc.sum(0) == 0) & (cc.sum(1) > 0)
    _un[FAM(m)].append((u.sum(), cc.sum(1)[u].sum() / cc.sum()))
for fm, v in _un.items():
    v = np.array(v, dtype=float)
    C[f"unassigned_{fm}_n"], C[f"unassigned_{fm}_mass"] = float(v[:, 0].mean()), float(v[:, 1].mean())

# ---------------- accuracy
ca = rd("real_coder_accuracy.csv")
for m in sorted({r["model"] for r in ca}):
    for c in sorted({r["condition"] for r in ca if r["model"] == m}):
        C[f"acc_{m}_{c}"] = float(np.mean([num(r, "accuracy") for r in ca if r["model"] == m and r["condition"] == c]))
C["acc_min"] = min(C[k] for k in C if k.startswith("acc_") and k != "acc_min")
C["acc_max"] = max(C[k] for k in C if k.startswith("acc_") and k not in ("acc_min",))
C["majority_baseline_mean"] = float(np.mean([num(r, "majority_baseline") for r in ca if r["model"] == "tfidf-logreg"]))

# ---------------- precision gain (Figure 4) and planning
pg = rd("precision_gain.csv")
acc = np.array([num(r, "accuracy") for r in pg]); tr = np.array([num(r, "tuned_over_hand") for r in pg])
C["gain_spearman_tuned_acc"] = float(spearmanr(acc, tr)[0])
C["gain_n_points"] = len(pg)
for r in pg:
    C[f"gain_{r['model']}_{r['condition']}"] = num(r, "tuned_over_hand")
llm = [r for r in pg if FAM(r["model"]) == "llm"]
C["gain_llm_tuned_range"] = [min(num(r, "tuned_over_hand") for r in llm), max(num(r, "tuned_over_hand") for r in llm)]
C["gain_plain_worse_than_hand_llm"] = sum(num(r, "plain_over_hand") > 1 for r in llm)
C["gain_llm_n"] = len(llm)
pl = rd("planning_real.csv")
for r in pl:
    k = f"plan_{r['model']}_{r['condition']}_{r['target_se']}"
    C[k + "_hand"], C[k + "_tuned"], C[k + "_rho"] = num(r, "n_hand_median"), num(r, "n_tuned_median"), num(r, "median_rho")
pg_ = rd("planning_grid.csv")
for r in pg_:
    C[f"grid_{r['target_se']}_{r['prevalence']}_{r['rho']}"] = int(r["n_per_group"])

# ---------------- illustration
ill = rd("illustration.csv")
for r in ill:
    k = "ill_" + ("qwen" if r["coder"].startswith("Qwen") else "xfer") + "_" + r["topic"].replace(" ", "_")
    for f in ("true", "naive", "bc_pooled", "calibration", "tuned", "tuned_se", "naive_bias_est", "naive_bias_se", "naive_bias_true"):
        C[f"{k}_{f}"] = num(r, f)
    C[f"{k}_covers"] = r["tuned_covers"] == "True"
C["ill_tuned_misses"] = sum(r["tuned_covers"] != "True" for r in ill)
C["ill_bias_se_mean"] = float(np.mean([num(r, "naive_bias_se") for r in ill]))
C["ill_bias_mdb"] = 2.8 * C["ill_bias_se_mean"]      # 5% two-sided test, 80% power
C["ill_n"] = len(ill)

# ---------------- language models by size: 1-3.8B vs the 7-8B member of three families
LARGE = ("olmo2-7b", "qwen2.5-7b", "granite3.1-8b")
SIZE = lambda m: "large" if m in LARGE else "small"
C["llm_large_n_models"] = len(LARGE)
for sz in ("small", "large"):
    o = [r for r in orc if r["fam"] == "llm" and SIZE(r["model"]) == sz]
    C[f"orc_llm{sz}_n"] = len(o)
    C[f"orc_llm{sz}_bc_group_tv"] = float(np.mean([num(r, "bc_oracle_group_tv") for r in o]))
    C[f"orc_llm{sz}_bc_pooled_tv"] = float(np.mean([num(r, "bc_oracle_pooled_tv") for r in o]))
    C[f"orc_llm{sz}_naive_tv"] = float(np.mean([num(r, "naive_tv_error") for r in o]))
    C[f"orc_llm{sz}_singular_share"] = float(np.mean([num(r, "sigma_min_group") < 1e-4 for r in o]))
    v = np.array([(u.sum(), cc.sum(1)[u].sum() / cc.sum())
                  for (m, c, corp), d in _load().items() if FAM(m) == "llm" and SIZE(m) == sz
                  for cc in [_counts(d["y"], d["pred"])] for u in [(cc.sum(0) == 0) & (cc.sum(1) > 0)]], dtype=float)
    C[f"unassigned_llm{sz}_n"], C[f"unassigned_llm{sz}_mass"] = float(v[:, 0].mean()), float(v[:, 1].mean())
    for d in ("within", "between"):
        for nv in ("50", "200"):
            k = collections.defaultdict(dict)
            for r in RS:
                if r["design"] == d and r["n_v"] == nv and r["fam"] == "llm" and SIZE(r["model"]) == sz:
                    k[(r["model"], r["condition"], r["comparison"])][r["estimator"]] = num(r, "mean_tv_error")
            C[f"real_{d}_{nv}_llm{sz}_bcpooled_worse_than_naive"] = float(np.mean([v["bc_pooled"] > v["naive"] for v in k.values()]))
    C[f"acc_llm{sz}_mean"] = float(np.mean([num(r, "accuracy") for r in ca if r["model"] not in ("tfidf-logreg", "tfidf-svm", "tfidf-logreg-transfer") and SIZE(r["model"]) == sz]))

json.dump(C, open(os.path.join(RES, "claims.json"), "w"), indent=1)
if __name__ == "__main__":
    for k, v in C.items():
        if isinstance(v, float):
            v = round(v, 4)
        print(k, v)
