#!/usr/bin/env python3
"""Tables added in v3 (within-group validity, human-code error, other schemes)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript", "tables"); os.makedirs(OUT, exist_ok=True)


def f2(x):
    s = f"{x:.2f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def validity():
    V = json.load(open(os.path.join(RES, "validity_summary.json")))
    rows = [("Language models, 1--4B (18)", "corpus_llm_small"), ("Language models, 7--8B (9)", "corpus_llm_large"),
            ("Transfer classifier (1)", "corpus_transfer"), ("In-corpus classifiers (2)", "corpus_incorpus")]
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "brier"]
    body = "\n".join(
        f"{lab} & " + " & ".join(f"{f2(V[k][c][1])} {{\\scriptsize({f2(V[k][c][0])}--{f2(V[k][c][2])})}}" for c in cols) + r" \\"
        for lab, k in rows)
    t = r"""\begin{table}[htbp]
\centering
\caption{Measurement quality of the coders at the corpus level: mean over coder--condition
combinations and corpora, with the range in parentheses. Precision, recall, and F1 are
macro-averaged over the topics present; weighted F1 weights each topic by its prevalence.
ECE is the expected calibration error of the top-label probability (ten bins); Brier is the
multi-class Brier score (0 is perfect, 2 the maximum). The same figures for each half of each
corpus, and per topic, are in the supplementary material.}
\label{tab:validity}
\scriptsize
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Coders (combinations) & Accuracy & Precision & Recall & F1 & Weighted F1 & ECE & Brier \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab_validity.tex"), "w").write(t)


def schemes():
    """Real coders under other schemes, at the comparison level (the definition used for
    the CAP-21 results): coverage is the mean over comparisons of each comparison's share of
    covering draws; 'worse than none' is the share of comparisons whose mean total-variation
    error exceeds that of the uncorrected difference. n_v = 100 per group, stratified.
    Without results/real_schemes.csv (review archive) the table is rebuilt from the stored
    results/schemes_table.json."""
    order = [("cap21", "21 CAP topics"), ("random10", "10 merged categories"), ("random5", "5 merged categories"),
             ("random2", "2 merged categories"), ("focal_binary", "Most prevalent topic vs.\\ rest")]
    cols = ("worse", "naive", "rg_group", "cal_group", "dd_tuned", "gain")
    if os.path.exists(os.path.join(RES, "real_schemes.csv")):
        import pandas as pd
        d = pd.read_csv(os.path.join(RES, "real_schemes.csv"))
        base = pd.read_csv(os.path.join(RES, "real_summary.csv"))
        base = base[(base.n_v == 100) & (base.sampling == "stratified")].copy()
        base["scheme"] = "cap21"; base["partition"] = 0
        base = base[~((base.design == "within") & base.comparison.str.endswith("length"))]
        keep = ["model", "condition", "design", "comparison", "scheme", "partition", "estimator",
                "mean_tv_error", "focal_coverage", "focal_rmse"]
        d = pd.concat([d[keep], base[keep]])
        stats = {}
        for sc, _ in order:
            for design in ("within", "between"):
                g = d[(d.scheme == sc) & (d.design == design)]
                piv = g.pivot_table(index=["model", "condition", "comparison", "partition"], columns="estimator",
                                    values=["mean_tv_error", "focal_coverage", "focal_rmse"])
                for col in cols:
                    if col == "worse":
                        v = float((piv["mean_tv_error"]["bc_pooled"] > piv["mean_tv_error"]["naive"]).mean())
                    elif col == "gain":
                        v = float((piv["focal_rmse"]["dd_tuned"] / piv["focal_rmse"]["hand"]).median())
                    else:
                        v = float(piv["focal_coverage"][col].mean())
                    stats.setdefault(sc, {}).setdefault(design, {})[col] = round(v, 4)
            stats[sc]["n"] = [int(d[(d.scheme == sc) & (d.design == dz) & (d.estimator == "naive")].shape[0])
                              for dz in ("within", "between")]
        json.dump(stats, open(os.path.join(RES, "schemes_table.json"), "w"), indent=1)
    else:
        stats = json.load(open(os.path.join(RES, "schemes_table.json")))
    lines = []
    for sc, lab in order:
        cells = [" / ".join(f"{100 * stats[sc][dz][col]:.0f}" if col != "gain" else f2(stats[sc][dz][col])
                            for dz in ("within", "between")) for col in cols]
        n_w, n_b = stats[sc]["n"]
        lines.append(f"{lab} & {n_w:,} / {n_b:,} & " + " & ".join(cells) + r" \\")
    t = r"""\begin{table}[htbp]
\centering
\caption{Real coders under other coding schemes. The 21 CAP topics are merged into 10, 5, or 2
categories by random partitions (ten per size, the same for every coder), or reduced to the
comparison's most prevalent topic against all others; human and machine codes are merged by
the same map. Each cell gives the value for comparisons within corpora (earlier vs.\ later
half) / between corpora, with 100 validation items per group. ``Worse'': share of comparisons
in which pooled back-calculation had a larger error than no correction. Coverage (\%) of
nominal 95\% intervals for the most prevalent category's difference. ``Tuned/hand'': median
ratio of the tuned difference estimator's root mean squared error to that of hand coding.
The transfer classifier, whose between-corpus codes come from models trained outside each
pair, enters the merged schemes within corpora only.}
\label{tab:schemes}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & & Back-calc., & \multicolumn{4}{c}{Coverage (\%)} & \\
\cmidrule(lr){4-7}
Scheme & Comparisons & pooled worse (\%) & None & Per-category & Calibration & Tuned diff. & Tuned/hand \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab_schemes.tex"), "w").write(t)


def noise():
    N = json.load(open(os.path.join(RES, "label_noise_summary.json")))
    sets = [("symmetric0.0", "None"), ("symmetric0.05", "Symmetric, .05"), ("symmetric0.1", "Symmetric, .10"),
            ("symmetric0.2", "Symmetric, .20"), ("confusable0.05", "Confusable topics, .05"),
            ("confusable0.1", "Confusable topics, .10"), ("confusable0.2", "Confusable topics, .20"),
            ("differential", "Differential, .05 / .15")]
    lines = []
    for key, lab in sets:
        cells = []
        for design in ("within", "between"):
            v = N[design][key]
            cells += [f2(v["_target"]["focal_attenuation_median"]), f"{100 * v['naive']['cov_latent']:.0f}",
                      f"{100 * v['dd_tuned']['cov_latent']:.1f}", f"{100 * v['dd_tuned']['cov_human']:.1f}",
                      f"{100 * v['cal_group']['cov_latent']:.1f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    t = r"""\begin{table}[htbp]
\centering
\caption{Study 2b: error in the hand codes. The CAP codes are the latent truth; the validation
items' hand codes are drawn from them with error $\epsilon$ (symmetric: replaced by a random
other topic; confusable: by a topic the in-corpus classifiers confuse it with; differential:
symmetric with $\epsilon = .05$ in one group and .15 in the other). Ratio: median ratio of the
hand-coded to the CAP difference for the most prevalent topic. Coverage (\%) of nominal 95\%
intervals for that topic against the CAP difference (CAP) or the difference a complete hand
coding would give (hand). 100 validation items per group, 100 draws per comparison.}
\label{tab:noise}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
 & \multicolumn{5}{c}{Within corpora} & \multicolumn{5}{c}{Between corpora} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Hand-coding error & Ratio & None & Tuned (CAP) & Tuned (hand) & Calib. & Ratio & None & Tuned (CAP) & Tuned (hand) & Calib. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab_noise.tex"), "w").write(t)


if __name__ == "__main__":
    validity()
    for f in ("noise", "schemes"):
        g = globals().get(f)
        if g and os.path.exists(os.path.join(RES, {"noise": "label_noise_summary.json",
                                                   "schemes": "schemes_summary.json"}[f])):
            g()
