#!/usr/bin/env python3
"""Figures for the paper, in greyscale (markers and line styles carry the estimator)."""
import csv, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Papers/_templates"))
import pro_style; pro_style.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 600, "axes.prop_cycle": plt.cycler(color=["black"]),
                     # Political Analysis: Arial or Verdana where possible (Liberation Sans is metric-identical to Arial)
                     "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})
UNC = {}   # largest half-width of the error bars drawn, quoted in the captions (results/figure_uncertainty.json)
RAW_SIM = os.path.join(os.path.expanduser("~"), ".cache", "run_out", "study1_sim_raw.csv")

STY = {  # label, marker, linestyle, grey
    "naive":     ("no correction",                  "o", "-",  "0.55"),
    "bc_pooled": ("back-calculation, pooled",       "s", "--", "0.35"),
    "bc_group":  ("back-calculation, per group",    "^", ":",  "0.35"),
    "cal_group": ("calibration, per group",         "D", "-",  "0.3"),
    "dd_group":  ("difference estimator",           "D", "-",  "0.0"),
    "dd_tuned":  ("difference estimator, tuned",    "v", "-.", "0.0"),
    "hand":      ("hand coding only",               "x", ":",  "0.6"),
}
ORDER = ["naive", "bc_pooled", "bc_group", "cal_group", "dd_tuned", "hand"]
FAM = lambda m: ("in-corpus classifiers" if m in ("tfidf-logreg", "tfidf-svm")
                 else "transfer classifier" if m == "tfidf-logreg-transfer" else "language models")


def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


def fig_sim():
    """Study 1, delta = .20 regimes; error bars are 1.96 Monte Carlo standard errors over the
    500 replications of each condition (from the per-replication file of s1_simulation.py)."""
    import pandas as pd
    raw = pd.read_csv(RAW_SIM, usecols=["acc", "delta", "where", "n_v", "sampling", "estimator", "tv_error"])
    raw = raw[raw.sampling == "stratified"]
    agg = raw.groupby(["acc", "delta", "where", "n_v", "estimator"]).tv_error.agg(["mean", "std", "count"])
    regimes = [(0.0, "none", "same errors in both groups"),
               (0.2, "rare", "errors differ: three rare topics"),
               (0.2, "frequent", "errors differ: three frequent topics")]
    accs = [0.35, 0.55, 0.75]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
    hw = []
    for i, (d, w, title) in enumerate(regimes):
        for j, a in enumerate(accs):
            ax = axes[i, j]
            for e in ORDER:
                g = agg.loc[(a, d, w)].xs(e, level="estimator").sort_index()
                se = 1.96 * g["std"] / np.sqrt(g["count"]); hw += list(se)
                lab, mk, ls, gr = STY[e]
                ax.errorbar(g.index, g["mean"], yerr=se, marker=mk, ls=ls, color=gr, lw=1.2, ms=3.5,
                            elinewidth=0.8, capsize=0, label=lab)
            ax.set_xscale("log"); ax.set_xticks([50, 100, 250, 500, 1000]); ax.set_xticklabels(["50", "100", "250", "500", "1000"])
            ax.minorticks_off()
            if i == 0: ax.set_title(f"coder accuracy {a}", fontsize=9)
            if j == 0: ax.set_ylabel(title.replace(": ", ":\n"), fontsize=8)
            if i == 2: ax.set_xlabel("hand-coded items per group", fontsize=8)
            ax.tick_params(labelsize=7)
    UNC["fig1_max_halfwidth"] = float(max(hw))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.supylabel("error of the estimated difference (total variation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sim_error.png")); plt.close(fig)


LLM_LABEL = {"olmo2-1b": "OLMo-2 1B", "qwen2.5-1.5b": "Qwen2.5 1.5B", "smollm2-1.7b": "SmolLM2 1.7B",
             "granite3.1-2b": "Granite 3.1 2B", "qwen2.5-3b": "Qwen2.5 3B", "phi3.5-mini": "Phi-3.5-mini",
             "olmo2-7b": "OLMo-2 7B", "qwen2.5-7b": "Qwen2.5 7B", "granite3.1-8b": "Granite 3.1 8B",
             "tfidf-logreg": "TF-IDF + logistic regression", "tfidf-svm": "TF-IDF + SVM"}


def fig_real():
    rows = [r for r in rd("real_summary.csv") if r["sampling"] == "stratified"]
    split = lambda r: "between" if r["design"] == "between" else r["comparison"].split(":")[1]
    fams = ["language models", "transfer classifier", "in-corpus classifiers"]
    designs = [("period", "earlier vs later half\nof one corpus"), ("between", "two corpora")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True, sharey=True)
    hw, top = [], []
    for i, (dk, dl) in enumerate(designs):
        for j, f in enumerate(fams):
            ax = axes[i, j]
            for e in ORDER:
                pts = []
                for nv in (50, 100, 200):
                    v = [float(r["mean_tv_error"]) for r in rows if split(r) == dk and FAM(r["model"]) == f
                         and r["estimator"] == e and int(r["n_v"]) == nv]
                    pts.append((nv, np.mean(v), 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))))
                lab, mk, ls, gr = STY[e]
                hw += [p[2] for p in pts]; top.append(max(p[1] + p[2] for p in pts))
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker=mk, ls=ls,
                            color=gr, lw=1.2, ms=3.5, elinewidth=0.8, capsize=0, label=lab)
            if i == 0: ax.set_title(f, fontsize=8.5)
            if j == 0: ax.set_ylabel(dl + "\n(total variation error)", fontsize=7.5)
            ax.set_xticks([50, 100, 200]); ax.tick_params(labelsize=7)
    for ax in axes[1]: ax.set_xlabel("hand-coded items per group", fontsize=8)
    UNC["fig3_max_halfwidth"] = float(max(hw))
    axes[0, 0].set_ylim(0, 1.04 * max(top))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_real_error.png")); plt.close(fig)


def fig_oracle():
    """Back-calculation with the population confusion matrices: (a) the extra error of the
    pooled matrix against the differential part of the uncorrected error (Equation 2);
    (b) the error with each group's own matrix against the conditioning of the matrix."""
    o = rd("real_oracle.csv")
    mk = {"language models": ("o", "0.55"), "transfer classifier": ("s", "0.0"), "in-corpus classifiers": ("^", "0.3")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for f, (m, col) in mk.items():
        v = [r for r in o if FAM(r["model"]) == f]
        x = [float(r["differential_tv"]) for r in v]
        y = [float(r["bc_oracle_pooled_tv"]) - float(r["bc_oracle_group_tv"]) for r in v]
        axes[0].scatter(x, y, marker=m, s=9, facecolor="none", edgecolor=col, linewidth=0.6, label=f)
        x = [max(float(r["sigma_min_group"]), 1e-4) for r in v]
        y = [float(r["bc_oracle_group_tv"]) for r in v]
        axes[1].scatter(x, y, marker=m, s=9, facecolor="none", edgecolor=col, linewidth=0.6, label=f)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xlabel("differential part of the uncorrected error\n(total variation)", fontsize=8)
    axes[0].set_ylabel("pooled minus per-group\nback-calculation error", fontsize=8)
    axes[0].set_title("(a) pooling the confusion matrix", fontsize=8.5)
    axes[1].set_xscale("log"); axes[1].set_xlim(6e-5, 0.3)
    axes[1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    axes[1].set_xticklabels(["$\\leq 10^{-4}$\n(singular)", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    axes[1].set_xlabel("smallest singular value\nof the two groups' matrices", fontsize=8)
    axes[1].set_ylabel("per-group back-calculation error\n(true matrices)", fontsize=8)
    axes[1].set_title("(b) inverting the confusion matrix", fontsize=8.5)
    for ax in axes: ax.tick_params(labelsize=7)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_oracle.png")); plt.close(fig)


def precision_gain():
    """Per coder-condition: focal RMSE of the tuned and plain difference estimators
    relative to hand coding alone, at 200 items per group, against accuracy."""
    rows = [r for r in rd("real_summary.csv") if int(r["n_v"]) == 200 and r["sampling"] == "stratified"]
    acc = collections.defaultdict(list)
    for r in rd("real_coder_accuracy.csv"):
        acc[(r["model"], r["condition"])].append(float(r["accuracy"]))
    out = []
    for (m, c) in sorted(acc):
        def rmse(e):
            v = [float(r["focal_rmse"]) for r in rows if r["model"] == m and r["condition"] == c and r["estimator"] == e]
            return float(np.sqrt(np.mean(np.square(v))))
        out.append({"model": m, "condition": c, "accuracy": float(np.mean(acc[(m, c)])),
                    "tuned_over_hand": rmse("dd_tuned") / rmse("hand"),
                    "plain_over_hand": rmse("dd_group") / rmse("hand")})
    with open(os.path.join(RES, "precision_gain.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def fig_gain():
    out = precision_gain()
    mk = {"language models": ("o", "0.55"), "transfer classifier": ("s", "0.0"), "in-corpus classifiers": ("^", "0.3")}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for f, (m, col) in mk.items():
        v = [o for o in out if FAM(o["model"]) == f]
        ax.scatter([o["accuracy"] for o in v], [o["tuned_over_hand"] for o in v], marker=m, s=24,
                   facecolor=col if f != "language models" else "none", edgecolor=col, linewidth=0.8,
                   label=f + (" (trained on 1,200 labels of the same corpus)" if f.startswith("in-corpus") else ""))
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.text(0.99, 1.0, "hand coding alone", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", ha="right", color="0.25")
    ax.set_xlabel("coder agreement with the human code")
    ax.set_ylabel("tuned difference estimator,\nerror relative to hand coding alone", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_precision_gain.png")); plt.close(fig)


def fig_validity():
    """Within-group measurement quality and what it implies for the comparison."""
    grp = rd("validity_group.csv")
    pat = {(r["model"], r["condition"], r["corpus"]): r for r in rd("validity_pattern.csv") if r["split"] == "period"}
    tiers = rd("validity_tiers.csv")
    mk = {"language models": ("o", "0.55"), "transfer classifier": ("s", "0.0"), "in-corpus classifiers": ("^", "0.3")}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axes[0]
    f1 = {}
    for r in grp:
        if r["split"] == "period":
            f1[(r["model"], r["condition"], r["corpus"], r["group"])] = float(r["macro_f1"])
    for f, (m, col) in mk.items():
        keys = [k for k in pat if FAM(k[0]) == f]
        ax.scatter([f1[k + ("A",)] for k in keys], [f1[k + ("B",)] for k in keys], marker=m, s=12,
                   facecolor="none" if f == "language models" else col, edgecolor=col, linewidth=0.6, label=f)
    ax.plot([0, .7], [0, .7], color="black", lw=0.6, ls="--")
    ax.set_xlabel("macro-F1, earlier half", fontsize=8); ax.set_ylabel("macro-F1, later half", fontsize=8)
    ax.set_title("(a) quality within each group", fontsize=8.5); ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax = axes[1]
    by = collections.defaultdict(dict)
    for r in tiers:
        by[(r["model"], r["condition"], r["comparison"])][r["estimator"]] = r
    xs, ys, cols = [], [], []
    for k, d in by.items():
        xs.append(float(d["hand"]["weaker_macro_f1"]))
        ys.append(float(d["dd_tuned"]["focal_rmse"]) / float(d["hand"]["focal_rmse"]))
        cols.append(mk[FAM(k[0])])
    for (m, col) in mk.values():
        ii = [i for i, c in enumerate(cols) if c == (m, col)]
        ax.scatter([xs[i] for i in ii], [ys[i] for i in ii], marker=m, s=10,
                   facecolor="none" if col == "0.55" else col, edgecolor=col, linewidth=0.6)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xlabel("macro-F1 of the weaker group", fontsize=8)
    ax.set_ylabel("tuned estimator error\nrelative to hand coding", fontsize=8)
    ax.set_title("(b) what quality buys", fontsize=8.5)
    ax = axes[2]
    order = ["< .20", ".20-.35", ".35-.50", ">= .50"]
    hw = []
    for e in ("naive", "rg_group", "cal_group", "dd_tuned"):
        vals = [[float(r["focal_coverage"]) for r in tiers if r["estimator"] == e and r["tier"] == t] for t in order]
        v = [np.mean(x) for x in vals]
        se = [1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) for x in vals]; hw += se
        lab, m_, ls, gr = STY[e] if e in STY else ("per-category correction", "P", "--", "0.4")
        ax.errorbar(range(4), v, yerr=se, marker=m_, ls=ls, color=gr, lw=1.0, ms=3.5, elinewidth=0.8, capsize=0, label=lab)
    UNC["fig2c_max_halfwidth"] = float(max(hw))
    UNC["fig2c_max_halfwidth_valid"] = float(max(hw[-8:]))
    ax.axhline(.95, color="black", lw=0.5, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([o.replace("-", "–") for o in order], fontsize=6.5)
    ax.set_xlabel("macro-F1 of the weaker group", fontsize=8); ax.set_ylabel("coverage, most prevalent topic", fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_title("(c) coverage by quality", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    for a in axes:
        a.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_validity.png")); plt.close(fig)


def export():
    """Numbered copies for the manuscript and the portal: PNG, and greyscale TIFF at 600 dpi.
    Main text: 1 simulation, 2 within-group validity, 3 real-coder error, 4 true matrices;
    Online Appendix: A1 precision gain."""
    from PIL import Image
    sub = os.path.join(ROOT, "SUBMISSION", "figures"); os.makedirs(sub, exist_ok=True)
    names = [("1", "fig_sim_error"), ("2", "fig_validity"), ("3", "fig_real_error"), ("4", "fig_oracle"),
             ("A1", "fig_precision_gain")]
    for i, n in names:
        im = Image.open(os.path.join(FIG, n + ".png")).convert("L")
        im.save(os.path.join(FIG, f"Figure{i}.png"), dpi=(600, 600))
        for d in (FIG, sub):
            im.save(os.path.join(d, f"Figure{i}.tif"), dpi=(600, 600), compression="tiff_lzw")


if __name__ == "__main__":
    fig_sim(); fig_validity(); fig_real(); fig_oracle(); fig_gain(); export()
    import json
    json.dump(UNC, open(os.path.join(RES, "figure_uncertainty.json"), "w"), indent=1)
    print("ok")
