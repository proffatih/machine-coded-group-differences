#!/usr/bin/env python3
"""Figures for the manuscript. Greyscale-legible by construction.

SSCR charges $800 for the first colour figure in print and $200 for each additional
one; online colour is free. So every figure has to survive being printed in grey:
series are separated by marker shape and line style, heatmaps use a single-hue
sequential ramp, and no claim in a caption depends on hue.

Usage: python code/04_figures.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CAP_MAJOR_TOPICS, VALID_CODES

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    # CMM kilavuzu: line art 1200 dpi, GRI TON 600 dpi, renkli 300 dpi.
    # Tum sekiller gri ton -> 600 dpi.
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MODEL_LABEL = {
    "smollm2-1.7b": "SmolLM2 1.7B", "olmo2-1b": "OLMo-2 1B",
    "granite3.1-2b": "Granite 3.1 2B", "qwen2.5-1.5b": "Qwen2.5 1.5B",
    "qwen2.5-3b": "Qwen2.5 3B", "phi3.5-mini": "Phi-3.5-mini",
}
GROUP_LABEL = {"country": "country", "venue": "institutional venue",
               "length": "text length", "era": "period"}
# short but unambiguous topic names for axis ticks
TOPIC_SHORT = {
    1: "Macroeconomics", 2: "Civil rights", 3: "Health", 4: "Agriculture",
    5: "Labour", 6: "Education", 7: "Environment", 8: "Energy", 9: "Immigration",
    10: "Transport", 12: "Law and crime", 13: "Social welfare", 14: "Housing",
    15: "Banking", 16: "Defence", 17: "Science and tech", 18: "Foreign trade",
    19: "Intl. affairs", 20: "Govt. operations", 21: "Public lands",
    23: "Culture",
}
LINES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
GREYS = plt.get_cmap("Greys")


def read_csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split(",")
        rows = []
        for line in fh:
            vals = line.rstrip("\n").split(",")
            if len(vals) != len(head):
                continue
            rec = {}
            for k, v in zip(head, vals):
                if v == "":
                    rec[k] = None; continue
                try:
                    rec[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    rec[k] = v
            rows.append(rec)
    return rows


def fig2_accuracy_by_subgroup(acc):
    """Raw vs topic-standardised accuracy: how much of a subgroup gap is just topic mix.

    The legend sits above the panels rather than inside an axis, because inside any
    panel it lands on top of bars; venue labels are written out in full and the venue
    panel is given more width than the two-level panels.
    """
    if not acc:
        return
    pretty = {"legislative_oral": "legislative\n(oral)", "legislative": "legislative",
              "media": "media", "executive": "executive", "judicial": "judicial",
              "US": "US", "UK": "UK", "long": "long", "short": "short",
              "earlier": "earlier", "later": "later"}
    groupings = sorted({r["grouping"] for r in acc})
    widths = [2.6 if g == "venue" else 1.0 for g in groupings]
    fig, axes = plt.subplots(1, len(groupings), figsize=(7.4, 3.0), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, gn in zip(axes, groupings):
        sub = [r for r in acc if r["grouping"] == gn]
        labs = sorted({r["group"] for r in sub})
        x = np.arange(len(labs)); w = 0.38
        raw = [np.mean([r["accuracy"] for r in sub if r["group"] == g]) for g in labs]
        std = [np.mean([r["topic_standardised_accuracy"] for r in sub if r["group"] == g])
               for g in labs]
        ax.bar(x - w / 2, raw, w, label="raw", color="0.35", edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, std, w, label="topic-standardised", color="0.80",
               edgecolor="black", linewidth=0.4, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty.get(g, g) for g in labs], rotation=0, fontsize=6.5)
        ax.set_title(GROUP_LABEL.get(gn, gn))
        ax.set_xlim(-0.6, len(labs) - 0.4)
    axes[0].set_ylabel("agreement with human code")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig2_subgroup_accuracy.png")); plt.close(fig)

def fig3_topic_gap_heatmap(topic_rows):
    """The main figure: where in the codebook the instrument stops behaving the same."""
    if not topic_rows:
        return
    groupings = sorted({r["grouping"] for r in topic_rows})
    models = sorted({r["model"] for r in topic_rows})
    for gn in groupings:
        sub = [r for r in topic_rows if r["grouping"] == gn]
        topics = sorted({r["topic"] for r in sub})
        M = np.full((len(models), len(topics)), np.nan)
        for i, m in enumerate(models):
            for j, t in enumerate(topics):
                vals = [r["gap"] for r in sub if r["model"] == m and r["topic"] == t]
                if vals:
                    M[i, j] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(0.34 * len(topics) + 2.4, 0.34 * len(models) + 2.4))
        im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0,
                       vmax=np.nanmax(M) if np.isfinite(np.nanmax(M)) else 1)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels([TOPIC_SHORT.get(t, str(t)) for t in topics], rotation=90)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("between-group recall gap", rotation=270, labelpad=10)
        ax.set_title(f"Between-subgroup recall gap by topic ({GROUP_LABEL.get(gn, gn)})")
        fig.savefig(os.path.join(FIG, f"fig3_topic_gap_{gn}.png")); plt.close(fig)


def fig4_dif_tests(dif):
    """Uniform vs non-uniform DIF: is the instrument merely harder, or differently shaped?"""
    if not dif:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    groupings = sorted({r["grouping"] for r in dif})
    for i, gn in enumerate(groupings):
        sub = [r for r in dif if r["grouping"] == gn]
        u = [-np.log10(max(r["uniform_p"], 1e-300)) for r in sub]
        n = [-np.log10(max(r["nonuniform_p"], 1e-300)) for r in sub]
        ax.scatter(u, n, s=22, marker=MARKERS[i % len(MARKERS)],
                   facecolor="none", edgecolor="black", linewidth=0.7,
                   label=GROUP_LABEL.get(gn, gn))
    lim = ax.get_xlim()[1]
    thr = -np.log10(0.05)
    ax.axhline(thr, color="0.4", lw=0.7, ls="--")
    ax.axvline(thr, color="0.4", lw=0.7, ls="--")
    ax.text(thr, ax.get_ylim()[1], " p = .05", fontsize=6, va="top", color="0.3")
    ax.set_xlabel(r"uniform DIF, $-\log_{10} p$")
    ax.set_ylabel(r"non-uniform DIF, $-\log_{10} p$")
    ax.set_title("Differential item functioning by subgrouping")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig4_dif_tests.png")); plt.close(fig)


def fig6_inference_cost(prev):
    """The payoff figure: what the bias does to a between-group comparison."""
    if not prev:
        return
    diffs = [r for r in prev if str(r["group"]).startswith("DIFF:")]
    singles = [r for r in prev if not str(r["group"]).startswith("DIFF:")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, rows, title in ((axes[0], singles, "single-subgroup prevalence"),
                            (axes[1], diffs, "between-subgroup difference")):
        if not rows:
            continue
        series = [("naive", "tv_naive"), ("pooled correction", "tv_pooled_correction"),
                  ("subgroup-specific", "tv_group_correction")]
        vals = [[r[k] for r in rows if r[k] is not None] for _, k in series]
        bp = ax.boxplot(vals, tick_labels=[s for s, _ in series], widths=0.55,
                        patch_artist=True, medianprops=dict(color="black"))
        for patch, shade in zip(bp["boxes"], ("0.35", "0.6", "0.85")):
            patch.set_facecolor(shade); patch.set_edgecolor("black"); patch.set_linewidth(0.5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("total-variation error vs human coding")
    fig.suptitle("Cost of non-invariance for the quantity social scientists report", y=1.03)
    fig.savefig(os.path.join(FIG, "fig6_inference_cost.png")); plt.close(fig)


def fig5_ladder(ladder):
    """Where on the invariance ladder each subgrouping fails."""
    if not ladder:
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    groupings = sorted({r["grouping"] for r in ladder})
    models = sorted({r["model"] for r in ladder})
    x = np.arange(len(groupings)); w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [np.mean([r["scaled_chi2_df"] for r in ladder
                         if r["grouping"] == g and r["model"] == m] or [np.nan])
                for g in groupings]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=MODEL_LABEL.get(m, m),
               color=str(0.25 + 0.6 * i / max(len(models) - 1, 1)),
               edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.text(len(groupings) - 0.5, 1.0, " invariance", fontsize=6, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(groupings)
    ax.set_ylabel(r"$\chi^2$ / df")
    ax.set_title("Invariance test of the confusion structure")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.savefig(os.path.join(FIG, "fig5_invariance_ladder.png")); plt.close(fig)


def fig7_capability_threshold(prev, cacc):
    """Does the correction pay off? Plot its benefit against the instrument's accuracy.

    This is the model comparison made visible. Each model contributes one point per
    grouping; the vertical axis is how much the correction improves (positive) or
    damages (negative) the between-subgroup difference relative to leaving it alone.
    The pooled correction sits below zero at every accuracy level. The
    subgroup-specific one is less damaging as accuracy rises but only reaches
    break-even at the top -- and not monotonically, which is the point: a
    practitioner cannot read off from an accuracy figure whether correcting is safe.
    """
    if not (prev and cacc):
        return
    acc = {}
    for r in cacc:
        acc.setdefault(r["model"], []).append(float(r["accuracy"]))
    acc = {k: float(np.mean(v)) for k, v in acc.items()}

    pts = {"pooled": [], "subgroup": []}
    for r in prev:
        if not str(r["group"]).startswith("DIFF:"):
            continue
        a = acc.get(r["model"])
        if a is None or r["tv_naive"] is None:
            continue
        n = float(r["tv_naive"])
        pts["pooled"].append((a, n - float(r["tv_pooled_correction"])))
        pts["subgroup"].append((a, n - float(r["tv_group_correction"])))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    style = {"pooled": ("pooled correction", "s", "0.55"),
             "subgroup": ("subgroup-specific correction", "o", "0.15")}
    for key, (lab, mk, col) in style.items():
        xs = sorted({round(x, 4) for x, _ in pts[key]})
        med = [np.median([y for a, y in pts[key] if round(a, 4) == x]) for x in xs]
        ax.scatter([x for x, _ in pts[key]], [y for _, y in pts[key]],
                   s=12, marker=mk, facecolor="none", edgecolor=col,
                   linewidth=0.6, alpha=.55)
        ax.plot(xs, med, marker=mk, color=col, linewidth=1.4, markersize=5, label=lab)
    ax.axhline(0, color="black", lw=0.9)
    ax.text(0.01, 0, "correction breaks even", fontsize=6, va="center", ha="left",
            color="0.25", transform=ax.get_yaxis_transform(), zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    ax.set_xlabel("agreement with the human code (mean over corpora)")
    ax.set_ylabel("improvement over no correction\n(between-subgroup difference)")
    ax.set_title("Benefit of correcting, against instrument accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig7_capability_threshold.png")); plt.close(fig)


def main():
    acc = read_csv("subgroup_accuracy.csv")
    dif = read_csv("dif_tests.csv")
    ladder = read_csv("invariance_ladder.csv")
    topic_rows = read_csv("per_topic_recall_gaps.csv")
    prev = read_csv("prevalence_bias.csv")
    if acc is None:
        sys.exit("results/ bos - once 03_invariance.py calistir")
    fig2_accuracy_by_subgroup(acc)
    fig3_topic_gap_heatmap(topic_rows)
    fig4_dif_tests(dif)
    fig5_ladder(ladder)
    fig6_inference_cost(prev)
    fig7_capability_threshold(prev, read_csv("corpus_accuracy.csv"))
    made = sorted(f for f in os.listdir(FIG) if f.endswith(".png"))
    print(f"{len(made)} sekil yazildi -> figures/")
    for f in made:
        print("  ", f)


if __name__ == "__main__":
    main()
