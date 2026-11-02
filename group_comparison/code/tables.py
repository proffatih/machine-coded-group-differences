#!/usr/bin/env python3
"""Tables for the manuscript, written to manuscript/tables/."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = json.load(open(os.path.join(ROOT, "results", "claims.json")))
OUT = os.path.join(ROOT, "manuscript", "tables"); os.makedirs(OUT, exist_ok=True)
f3 = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x < 1 else f"{x:.3f}"

T1 = r"""\begin{table}[htbp]
\centering
\caption{Estimators of a between-group difference in topic prevalence. $q_g$ is the
share of machine codes for each topic in group $g$; the validation items are a random
sample of the coded items whose human code is known. ``Valid'' means consistent for every
pair of confusion matrices, including when the coder errs differently in the two groups,
given validation items from each group. Formulas and standard errors are in the Appendix.}
\label{tab:estimators}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{3.3cm}p{6.6cm}cc@{}}
\toprule
Estimator & Estimate for group $g$ & Valid & Interval \\
\midrule
No correction & $q_g$ & no & binomial \\
Back-calculation, pooled & solve $\hat{C}^{\top}x = q_g$, $\hat{C}$ = P(code $|$ true) from both groups' validation items & no & none \\
Back-calculation, per group & the same with group $g$'s own $\hat{C}_g$ & if invertible & none \\
Rogan--Gladen, per group & per topic, $(q_{gk} - \widehat{\mathit{fpr}}_{gk})/(\widehat{\mathit{sens}}_{gk} - \widehat{\mathit{fpr}}_{gk})$ & if informative & delta \\
Calibration, per group & $\sum_j q_{gj}\,\hat{P}_g(\text{true}=k \mid \text{code}=j)$ & yes & closed form \\
Difference estimator & $q_g + (\bar{y}_g - \bar{p}_g)$, the group's own mean validation residual & yes & closed form \\
Difference estimator, tuned & $\lambda q_g + (\bar{y}_g - \lambda\bar{p}_g)$, $\lambda\in[0,1]$ per group and topic & yes & closed form \\
Hand coding only & $\bar{y}_g$ & yes & closed form \\
\bottomrule
\end{tabular}
\end{table}
"""

T2 = r"""\begin{table}[htbp]
\centering
\caption{The seven Comparative Agendas Project corpora. All share the CAP master codebook
of 21 major policy topics, and every item carries a human code. ``Off-scheme'' is the
share of records dropped because they carried a media-specific extension code or the
residual ``other'' category rather than one of the 21 shared topics. Words is the mean
length of an item; Supreme Court summaries were truncated at 1,800 characters. The five
items of each corpus used as few-shot examples are excluded from all analyses, leaving
1,495 per corpus.}
\label{tab:corpora}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llcrrrr@{}}
\toprule
Corpus & Venue & Country & $n$ & Words & Topics & Off-sch.\ (\%) \\
\midrule
Congressional bills & Legislative & US & 1,500 & 27.5 & 20 & 24.4 \\
NYT front page & Media & US & 1,500 & 8.6 & 21 & 11.4 \\
Executive orders & Executive & US & 1,500 & 13.4 & 21 & 0.0 \\
Supreme Court cases & Judicial & US & 1,500 & 181.1 & 21 & 0.0 \\
Acts of Parliament & Legislative & UK & 1,500 & 39.2 & 21 & 0.0 \\
PM's Questions & Legisl. (oral) & UK & 1,500 & 79.7 & 21 & 0.0 \\
Newspaper front page & Media & UK & 1,500 & 6.5 & 21 & 20.1 \\
\bottomrule
\end{tabular}
\end{table}
"""

ROWS = [("OLMo-2 1B", "olmo2-1b", "allenai"), ("Qwen2.5 1.5B", "qwen2.5-1.5b", "Qwen"),
        ("SmolLM2 1.7B", "smollm2-1.7b", "HuggingFaceTB"), ("Granite 3.1 2B", "granite3.1-2b", "ibm-granite"),
        ("Qwen2.5 3B", "qwen2.5-3b", "Qwen"), ("Phi-3.5-mini 3.8B", "phi3.5-mini", "microsoft")]
body = "\n".join(f"{n} & {d} & {f3(C['acc_'+k+'_zs'])} & {f3(C['acc_'+k+'_cb'])} & {f3(C['acc_'+k+'_fs'])} \\\\"
                 for n, k, d in ROWS)
T3 = r"""\begin{table}[htbp]
\centering
\caption{The coders and their agreement with the human code, averaged over the seven
corpora. Chance is .048; always predicting a corpus's most frequent topic would score
""" + f3(C["majority_baseline_mean"]) + r""" on average. The six language models are open-weight instruction-tuned models
loaded in 4-bit NF4 quantisation and prompted zero-shot, zero-shot with codebook
definitions, or with five examples from the same corpus; each topic's score is read from
the logits of its name. The two supervised classifiers use TF-IDF word and character
$n$-grams; the first two predict every item out of fold by five-fold cross-fitting within
each corpus, and so were trained on about 1,200 human codes of that corpus, while the
transfer classifier was trained on the 9,000 items of the other six corpora and has seen
no label of the corpus it codes.}
\label{tab:coders}
\footnotesize
\begin{tabular}{@{}llrrr@{}}
\toprule
Coder & Developer & Zero-shot & Codebook & Five-shot \\
\midrule
""" + body + r"""
\midrule
TF-IDF + logistic regression & scikit-learn & \multicolumn{3}{c}{""" + f3(C["acc_tfidf-logreg_sup"]) + r"""} \\
TF-IDF + calibrated linear SVM & scikit-learn & \multicolumn{3}{c}{""" + f3(C["acc_tfidf-svm_sup"]) + r"""} \\
TF-IDF + logistic regression, transfer & scikit-learn & \multicolumn{3}{c}{""" + f3(C["acc_tfidf-logreg-transfer_sup"]) + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""

lines = []
for sp, sl in (("period", "Earlier vs later half of one corpus"), ("length", "Shorter vs longer half of one corpus (stress test)"),
               ("between", "Two corpora")):
    lines.append(r"\multicolumn{11}{@{}l}{\emph{" + sl + r"}} \\")
    for fm, fl in (("llm", "Language models"), ("xfer", "Transfer classifier"), ("supin", "In-corpus classifiers")):
        for nv in ("50", "200"):
            p = f"real_{sp}_{fm}_{nv}_"
            lines.append(f"{fl if nv == '50' else ''} & {nv} & " + " & ".join(
                f3(C[p + e + "_tv"]) for e in ("naive", "bc_pooled", "bc_group", "rg_group", "cal_group", "dd_tuned", "hand"))
                + (f" & {C[f'real_naivecov_{sp}_{fm}']*100:.0f}" if nv == "50" else " & ")
                + f" & {C[p + 'dd_tuned_cov']*100:.0f} \\\\")
T4 = r"""\begin{table}[p]
\centering
\caption{Real coders: error of the estimated difference in topic prevalence (total
variation distance from the human-coded difference, averaged over comparisons and 200
(within a corpus) or 100 (between corpora) validation draws), and coverage of nominal 95\%
intervals for the most prevalent topic. For the uncorrected difference the interval is the
binomial one a researcher would report, and does not depend on $n_v$. Within-corpus
comparisons: 21 coder--condition combinations $\times$ seven corpora for each split;
between-corpus comparisons: 21 $\times$ 21 pairs. BC = back-calculation; RG = per-category
(Rogan--Gladen) correction. The transfer classifier codes each pair of corpora with a model
trained on the five corpora outside the pair.}
\label{tab:real}
\scriptsize
\renewcommand{\arraystretch}{0.95}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & & \multicolumn{7}{c}{Error of the estimated difference} & \multicolumn{2}{c}{Coverage (\%)} \\
\cmidrule(lr){3-9}\cmidrule(lr){10-11}
Coders & $n_v$ & None & BC pooled & BC group & RG & Calibr. & Tuned & Hand & None & Tuned \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
for n, t in (("tab1_estimators", T1), ("tab2_corpora", T2), ("tab3_coders", T3), ("tab4_real", T4)):
    open(os.path.join(OUT, n + ".tex"), "w").write(t)
print("4 tablo")

# ---- Table 5: worked agenda comparison
import csv as _csv
ill = list(_csv.DictReader(open(os.path.join(ROOT, "results", "illustration.csv"))))
def sg(x):
    x = float(x); s = f"{abs(x):.3f}".lstrip("0")
    return ("$-$" if x < -0.0005 else "") + s
lines5 = []
iv = lambda m, se: f"[{sg(m - 1.96 * se)}, {sg(m + 1.96 * se)}]"
for coder in dict.fromkeys(r["coder"] for r in ill):
    lines5.append(r"\multicolumn{8}{@{}l}{\emph{" + coder + r"}} \\")
    for r in [x for x in ill if x["coder"] == coder]:
        t, se, b, bse = (float(r[k]) for k in ("tuned", "tuned_se", "naive_bias_est", "naive_bias_se"))
        lines5.append(f"{r['topic']} & {sg(r['true'])} & {sg(r['naive'])} & {sg(r['bc_pooled'])} & {sg(r['calibration'])} & "
                      f"{sg(t)} {iv(t, se)} & {sg(r['naive_bias_true'])} & {sg(b)} {iv(b, bse)} \\\\")
T5 = r"""\begin{table}[htbp]
\centering
\caption{A legislative and a media agenda compared: the difference in the share of
U.S. Congressional bill titles and New York Times front-page headlines on each of the five
topics most prevalent across the two corpora (bills minus headlines). ``Human'' is the
difference in the human codes of all items. Estimates use one random validation sample of
200 hand-coded items per corpus. The last two columns give the bias of the uncorrected
difference, as it is and as estimated from the validation residuals, with 95\% intervals.}
\label{tab:illustration}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrlrl@{}}
\toprule
 & & & BC & & & \multicolumn{2}{c}{Bias of uncorrected} \\
\cmidrule(lr){7-8}
Topic & Human & Uncorr. & pooled & Calibr. & Tuned [95\% CI] & True & Estimated [95\% CI] \\
\midrule
""" + "\n".join(lines5) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(OUT, "tab5_illustration.tex"), "w").write(T5)
print("tab5")

# ---- Table 6: planning grid
g = list(_csv.DictReader(open(os.path.join(ROOT, "results", "planning_grid.csv"))))
rows6 = []
for se_t in ("0.03", "0.02"):
    rows6.append(r"\multicolumn{5}{@{}l}{\emph{Standard error of the difference " + se_t.lstrip("0") + r"}} \\")
    for p_ in ("0.05", "0.1", "0.2", "0.3"):
        v = [next(int(r["n_per_group"]) for r in g if r["target_se"] == se_t and r["prevalence"] == p_ and r["rho"] == rho)
             for rho in ("0.0", "0.5", "0.7", "0.9")]
        rows6.append(f"{float(p_):.2f}".lstrip("0") + " & " + " & ".join(f"{x:,}" for x in v) + r" \\")
T6 = r"""\begin{table}[htbp]
\centering
\caption{Hand-coded items needed per group for a target standard error of the difference
in one topic's prevalence, with the tuned difference estimator and a large coded corpus,
when the topic has the same prevalence $\pi$ in both groups and the human and machine
indicators of the topic correlate at $\rho$ in each group:
$n_g = 2\pi(1-\pi)(1-\rho^2)/\mathrm{SE}^2$. $\rho = 0$ is hand coding alone. In Study 2 the
median $\rho$ for the most prevalent topic was """ + f"{C['plan_qwen2.5-3b_zs_0.03_rho']:.2f}".lstrip("0") + r""" for Qwen2.5 3B zero-shot, """ + f"{C['plan_tfidf-logreg-transfer_sup_0.03_rho']:.2f}".lstrip("0") + r""" for the
transfer classifier, and """ + f"{C['plan_tfidf-logreg_sup_0.03_rho']:.2f}".lstrip("0") + r""" for the in-corpus logistic regression.}
\label{tab:planning}
\footnotesize
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{4}{c}{Correlation of human and machine codes ($\rho$)} \\
\cmidrule(lr){2-5}
Prevalence $\pi$ & 0 & .5 & .7 & .9 \\
\midrule
""" + "\n".join(rows6) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(OUT, "tab6_planning.tex"), "w").write(T6)
print("tab6")
