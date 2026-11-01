#!/usr/bin/env python3
"""Every number the manuscript states, computed from the result files (results/claims.json)."""
import csv, json, os, collections
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rd = lambda n: list(csv.DictReader(open(os.path.join(RES, n))))
C = {}

# ---------------- simulation
sim = rd("sim_summary.csv")
f = lambda **k: [r for r in sim if all(r[a] == str(b) for a, b in k.items())]
cov = lambda e: [float(r["focal_coverage"]) for r in sim if r["estimator"] == e]
for e in ("naive", "dd_group", "dd_tuned", "hand"):
    C[f"sim_cov_{e}_min"], C[f"sim_cov_{e}_max"] = min(cov(e)), max(cov(e))
C["sim_conditions"] = len({(r["acc"], r["delta"], r["where"], r["n_v"]) for r in sim})
C["sim_reps"] = int(sim[0]["reps"])
for a in ("0.35", "0.55", "0.75"):
    for w, d in (("none", "0.0"), ("rare", "0.2"), ("frequent", "0.2"), ("all", "0.2")):
        C[f"sim_bcpooled_{a}_{w}_nv1000"] = float(f(acc=a, delta=d, where=w, n_v=1000, estimator="bc_pooled")[0]["mean_tv_error"])
        C[f"sim_naive_{a}_{w}_nv1000"] = float(f(acc=a, delta=d, where=w, n_v=1000, estimator="naive")[0]["mean_tv_error"])
    for nv in (100, 250, 1000):
        for e in ("naive", "bc_pooled", "bc_group", "dd_group", "dd_tuned", "hand"):
            C[f"sim_{e}_{a}_none_nv{nv}"] = float(f(acc=a, delta="0.0", where="none", n_v=nv, estimator=e)[0]["mean_tv_error"])
            C[f"sim_{e}_{a}_frequent_nv{nv}"] = float(f(acc=a, delta="0.2", where="frequent", n_v=nv, estimator=e)[0]["mean_tv_error"])
# tuned vs hand, averaged over all conditions per accuracy
for a in ("0.35", "0.55", "0.75"):
    t = np.mean([float(r["mean_tv_error"]) for r in sim if r["acc"] == a and r["estimator"] == "dd_tuned"])
    h = np.mean([float(r["mean_tv_error"]) for r in sim if r["acc"] == a and r["estimator"] == "hand"])
    C[f"sim_tuned_over_hand_{a}"] = t / h
# tuned best share of conditions
best = collections.Counter()
for k in {(r["acc"], r["delta"], r["where"], r["n_v"]) for r in sim}:
    rows = [r for r in sim if (r["acc"], r["delta"], r["where"], r["n_v"]) == k]
    best[min(rows, key=lambda r: float(r["mean_tv_error"]))["estimator"]] += 1
C["sim_best_counts"] = dict(best)
# naive focal bias grows with non-invariance in frequent rows
C["sim_naive_absbias_frequent_0.2"] = float(np.mean([abs(float(r["focal_bias"])) for r in sim if r["estimator"] == "naive" and r["where"] == "frequent" and r["delta"] == "0.2"]))
C["sim_naive_absbias_none"] = float(np.mean([abs(float(r["focal_bias"])) for r in sim if r["estimator"] == "naive" and r["where"] == "none"]))
C["sim_ddtuned_absbias_max"] = float(max(abs(float(r["focal_bias"])) for r in sim if r["estimator"] == "dd_tuned"))

# ---------------- real coders
real = rd("real_summary.csv")
fam = lambda m: "sup" if m.startswith("tfidf") else "llm"
C["real_within_comparisons"] = len({(r["model"], r["condition"], r["comparison"]) for r in real if r["design"] == "within"})
C["real_between_comparisons"] = len({(r["model"], r["condition"], r["comparison"]) for r in real if r["design"] == "between"})
for d in ("within", "between"):
    for fm in ("llm", "sup"):
        for nv in ("50", "100", "200"):
            s = [r for r in real if r["design"] == d and fam(r["model"]) == fm and r["n_v"] == nv]
            for e in ("naive", "bc_pooled", "bc_group", "dd_group", "dd_tuned", "hand"):
                v = [r for r in s if r["estimator"] == e]
                C[f"real_{d}_{fm}_{nv}_{e}_tv"] = float(np.mean([float(r["mean_tv_error"]) for r in v]))
                C[f"real_{d}_{fm}_{nv}_{e}_rmse"] = float(np.sqrt(np.mean([float(r["focal_rmse"]) ** 2 for r in v])))
                C[f"real_{d}_{fm}_{nv}_{e}_absbias"] = float(np.mean([abs(float(r["focal_bias"])) for r in v]))
                if e in ("naive", "dd_group", "dd_tuned", "hand"):
                    C[f"real_{d}_{fm}_{nv}_{e}_cov"] = float(np.mean([float(r["focal_coverage"]) for r in v]))
    for nv in ("50", "100", "200"):
        k = {}
        for r in real:
            if r["design"] == d and r["n_v"] == nv and r["estimator"] in ("naive", "dd_tuned", "bc_pooled", "bc_group"):
                k.setdefault((r["model"], r["condition"], r["comparison"]), {})[r["estimator"]] = float(r["mean_tv_error"])
        C[f"real_{d}_{nv}_tuned_beats_naive"] = float(np.mean([v["dd_tuned"] < v["naive"] for v in k.values()]))
        C[f"real_{d}_{nv}_bcpooled_worse_than_naive"] = float(np.mean([v["bc_pooled"] > v["naive"] for v in k.values()]))
        C[f"real_{d}_{nv}_bcgroup_worse_than_naive"] = float(np.mean([v["bc_group"] > v["naive"] for v in k.values()]))
cov_all = [float(r["focal_coverage"]) for r in real if r["estimator"] in ("dd_group", "dd_tuned", "hand")]
C["real_valid_cov_min"], C["real_valid_cov_max"] = float(np.mean([min(cov_all)])), max(cov_all)
C["real_true_tv_within"] = float(np.mean([float(r["true_tv_difference"]) for r in real if r["design"] == "within"]))
C["real_true_tv_between"] = float(np.mean([float(r["true_tv_difference"]) for r in real if r["design"] == "between"]))

# ---------------- precision gain
pg = rd("precision_gain.csv")
acc = np.array([float(r["accuracy"]) for r in pg]); tr = np.array([float(r["tuned_over_hand"]) for r in pg])
pr = np.array([float(r["plain_over_hand"]) for r in pg])
C["gain_spearman_tuned_acc"] = float(spearmanr(acc, tr)[0])
C["gain_tuned_max"] = float(tr.max()); C["gain_tuned_min"] = float(tr.min())
sup = [r for r in pg if r["model"].startswith("tfidf")]
C["gain_sup_tuned"] = [float(r["tuned_over_hand"]) for r in sup]
C["gain_sup_effective_n"] = [1 / float(r["tuned_over_hand"]) ** 2 for r in sup]
llm = [r for r in pg if not r["model"].startswith("tfidf")]
C["gain_llm_tuned_range"] = [min(float(r["tuned_over_hand"]) for r in llm), max(float(r["tuned_over_hand"]) for r in llm)]
C["gain_plain_worse_than_hand_llm"] = sum(float(r["plain_over_hand"]) > 1 for r in llm)
C["gain_llm_n"] = len(llm)
C["gain_best_llm_effective_n"] = 1 / min(float(r["tuned_over_hand"]) for r in llm) ** 2
ca = rd("real_coder_accuracy.csv")
for m in sorted({r["model"] for r in ca}):
    for c in sorted({r["condition"] for r in ca if r["model"] == m}):
        C[f"acc_{m}_{c}"] = float(np.mean([float(r["accuracy"]) for r in ca if r["model"] == m and r["condition"] == c]))
json.dump(C, open(os.path.join(RES, "claims.json"), "w"), indent=1)
if __name__ == "__main__":
    for k in ("sim_conditions", "sim_reps", "sim_best_counts", "sim_cov_naive_min", "sim_cov_naive_max", "sim_cov_dd_group_min", "sim_cov_dd_group_max", "sim_cov_dd_tuned_min", "sim_cov_dd_tuned_max", "sim_cov_hand_min", "sim_cov_hand_max",
              "sim_tuned_over_hand_0.35", "sim_tuned_over_hand_0.55", "sim_tuned_over_hand_0.75", "sim_naive_absbias_none", "sim_naive_absbias_frequent_0.2", "sim_ddtuned_absbias_max",
              "real_within_comparisons", "real_between_comparisons", "real_valid_cov_min", "real_valid_cov_max", "real_true_tv_within", "real_true_tv_between",
              "gain_spearman_tuned_acc", "gain_sup_tuned", "gain_sup_effective_n", "gain_llm_tuned_range", "gain_plain_worse_than_hand_llm", "gain_best_llm_effective_n"):
        print(k, C[k])
    for d in ("within", "between"):
        for nv in ("50", "100", "200"):
            print(d, nv, "tuned<naive", round(C[f"real_{d}_{nv}_tuned_beats_naive"], 3), "bcpool>naive", round(C[f"real_{d}_{nv}_bcpooled_worse_than_naive"], 3), "bcgroup>naive", round(C[f"real_{d}_{nv}_bcgroup_worse_than_naive"], 3))
    for a in ("0.35", "0.55", "0.75"):
        print(a, {w: (round(C[f"sim_bcpooled_{a}_{w}_nv1000"], 3), round(C[f"sim_naive_{a}_{w}_nv1000"], 3)) for w in ("none", "rare", "frequent", "all")})
