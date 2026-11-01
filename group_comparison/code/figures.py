#!/usr/bin/env python3
"""Figures for the paper, in greyscale (markers and line styles carry the estimator)."""
import csv, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_style; pro_style.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 600, "axes.prop_cycle": plt.cycler(color=["black"])})

STY = {  # label, marker, linestyle, grey
    "naive":     ("no correction",                  "o", "-",  "0.55"),
    "bc_pooled": ("back-calculation, pooled",       "s", "--", "0.35"),
    "bc_group":  ("back-calculation, per group",    "^", ":",  "0.35"),
    "dd_group":  ("difference estimator",           "D", "-",  "0.0"),
    "dd_tuned":  ("difference estimator, tuned",    "v", "-.", "0.0"),
    "hand":      ("hand coding only",               "x", ":",  "0.6"),
}
ORDER = ["naive", "bc_pooled", "bc_group", "dd_group", "dd_tuned", "hand"]


def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


def fig_sim():
    rows = rd("sim_summary.csv")
    regimes = [("0.0", "none", "invariant coder"),
               ("0.2", "rare", "non-invariant: three rare topics"),
               ("0.2", "frequent", "non-invariant: three frequent topics")]
    accs = ["0.35", "0.55", "0.75"]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
    for i, (d, w, title) in enumerate(regimes):
        for j, a in enumerate(accs):
            ax = axes[i, j]
            for e in ORDER:
                pts = sorted((int(r["n_v"]), float(r["mean_tv_error"])) for r in rows
                             if r["acc"] == a and r["delta"] == d and r["where"] == w and r["estimator"] == e)
                lab, mk, ls, gr = STY[e]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls, color=gr,
                        lw=1.2, ms=3.5, label=lab)
            ax.set_xscale("log"); ax.set_xticks([100, 250, 500, 1000]); ax.set_xticklabels(["100", "250", "500", "1000"])
            if i == 0: ax.set_title(f"coder accuracy {a}", fontsize=9)
            if j == 0: ax.set_ylabel(title.replace(": ", ":\n"), fontsize=8)
            if i == 2: ax.set_xlabel("hand-coded items per group", fontsize=8)
            ax.tick_params(labelsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.supylabel("error of the estimated difference (total variation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sim_error.png")); plt.close(fig)


LLM_LABEL = {"olmo2-1b": "OLMo-2 1B", "qwen2.5-1.5b": "Qwen2.5 1.5B", "smollm2-1.7b": "SmolLM2 1.7B",
             "granite3.1-2b": "Granite 3.1 2B", "qwen2.5-3b": "Qwen2.5 3B", "phi3.5-mini": "Phi-3.5-mini",
             "tfidf-logreg": "TF-IDF + logistic regression", "tfidf-svm": "TF-IDF + SVM"}


def fig_real():
    rows = rd("real_summary.csv")
    fam = lambda m: "supervised classifiers" if m.startswith("tfidf") else "language models"
    panels = [("within", "language models"), ("within", "supervised classifiers"),
              ("between", "language models"), ("between", "supervised classifiers")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), sharex=True)
    for ax, (design, f) in zip(axes.ravel(), panels):
        for e in ORDER:
            pts = []
            for nv in (50, 100, 200):
                v = [float(r["mean_tv_error"]) for r in rows if r["design"] == design and fam(r["model"]) == f
                     and r["estimator"] == e and int(r["n_v"]) == nv]
                pts.append((nv, np.mean(v)))
            lab, mk, ls, gr = STY[e]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls, color=gr, lw=1.2, ms=3.5, label=lab)
        where = "two halves of one corpus" if design == "within" else "two corpora"
        ax.set_title(f"{f}: {where}", fontsize=8.5)
        ax.set_xticks([50, 100, 200]); ax.tick_params(labelsize=7)
        ax.set_ylim(0, None)
    for ax in axes[1]: ax.set_xlabel("hand-coded items per group", fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel("error of the estimated difference\n(total variation)", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_real_error.png")); plt.close(fig)


def precision_gain():
    """Per coder-condition: focal RMSE of the tuned and plain difference estimators
    relative to hand coding alone, at 200 items per group, against accuracy."""
    rows = [r for r in rd("real_summary.csv") if int(r["n_v"]) == 200]
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
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for key, (lab, mk, col) in {"plain_over_hand": ("difference estimator", "D", "0.45"),
                                "tuned_over_hand": ("difference estimator, tuned", "v", "0.0")}.items():
        xs = [o["accuracy"] for o in out]; ys = [o[key] for o in out]
        ax.scatter(xs, ys, marker=mk, s=22, facecolor="none" if key == "plain_over_hand" else col,
                   edgecolor=col, linewidth=0.8, label=lab)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.text(0.99, 1.0, "hand coding alone", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", ha="right", color="0.25")
    ax.set_xlabel("coder agreement with the human code")
    ax.set_ylabel("error relative to hand coding alone\n(RMSE ratio, 200 items per group)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_precision_gain.png")); plt.close(fig)


if __name__ == "__main__":
    fig_sim(); fig_real(); fig_gain()
    print("ok")
