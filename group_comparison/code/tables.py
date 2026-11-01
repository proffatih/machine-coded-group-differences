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
share of machine codes for each topic in group $g$; the validation sample is a random
sample of $n_v$ items per group whose human code is known. ``Unbiased'' means unbiased
for every pair of confusion matrices, including when the coder errs differently in the
two groups.}
\label{tab:estimators}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{3.2cm}p{6.2cm}cc@{}}
\toprule
Estimator & Estimate for group $g$ & Unbiased & Standard error \\
\midrule
No correction & $q_g$ & no & invalid \\
Back-calculation, pooled & solve $\hat{C}^{\top}x = q_g$, $\hat{C}$ from both groups' validation items & no & none \\
Back-calculation, per group & solve $\hat{C}_g^{\top}x = q_g$, $\hat{C}_g$ from group $g$'s validation items & approx. & none \\
Difference estimator & $q_g + (\bar{y}_g - \bar{p}_g)$, the group's own validation residual & yes & closed form \\
Difference estimator, tuned & $\lambda q_g + (\bar{y}_g - \lambda\bar{p}_g)$, $\lambda\in[0,1]$ per topic & yes & closed form \\
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
length of an item.}
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
corpora. Chance is .048. The six language models are open-weight instruction-tuned models
loaded in 4-bit NF4 quantisation and prompted zero-shot, zero-shot with codebook
definitions, or with five examples from the same corpus; each topic's score is read from
the logits of its name. The two supervised classifiers use TF-IDF word and character
$n$-grams and predict every item out of fold (five-fold cross-fitting within each corpus).}
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
\bottomrule
\end{tabular}
\end{table}
"""

lines = []
for d, dl in (("within", "Two halves of one corpus"), ("between", "Two corpora")):
    lines.append(r"\multicolumn{9}{@{}l}{\emph{" + dl + r"}} \\")
    for fm, fl in (("llm", "Language models"), ("sup", "Supervised")):
        for nv in ("50", "100", "200"):
            p = f"real_{d}_{fm}_{nv}_"
            lines.append(f"{fl if nv == '50' else ''} & {nv} & " + " & ".join(
                f3(C[p + e + "_tv"]) for e in ("naive", "bc_pooled", "bc_group", "dd_tuned", "hand"))
                + f" & {C[p + 'naive_cov']*100:.0f} & {C[p + 'dd_tuned_cov']*100:.0f} \\\\")
T4 = r"""\begin{table}[htbp]
\centering
\caption{Real coders: error of the estimated difference in topic prevalence (total
variation distance from the human-coded difference, averaged over comparisons and 200
or 100 validation draws) and coverage of nominal 95\% intervals for the most prevalent
topic. Within-corpus comparisons split a corpus at its own median length or year (280
comparisons: 20 coder--condition combinations, seven corpora, two splits); between-corpus
comparisons set two corpora against each other (420 comparisons). Back-calculation has no
standard error; the untuned difference estimator is shown in Figure~2.}
\label{tab:real}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & & \multicolumn{5}{c}{Error of the estimated difference} & \multicolumn{2}{c}{Coverage (\%)} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}
Coders & $n_v$ & None & BC pooled & BC group & Tuned & Hand & None & Tuned \\
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
for coder in dict.fromkeys(r["coder"] for r in ill):
    lines5.append(r"\multicolumn{5}{@{}l}{\emph{" + coder + r"}} \\")
    for r in [x for x in ill if x["coder"] == coder]:
        lo, hi = float(r["tuned"]) - 1.96 * float(r["tuned_se"]), float(r["tuned"]) + 1.96 * float(r["tuned_se"])
        lines5.append(f"{r['topic']} & {sg(r['true'])} & {sg(r['naive'])} & {sg(r['tuned'])} & [{sg(lo)}, {sg(hi)}] \\\\")
T5 = r"""\begin{table}[htbp]
\centering
\caption{A legislative and a media agenda compared: the difference in the share of
U.S. Congressional bill titles and New York Times front-page headlines on each of the five
topics most prevalent across the two corpora (bills minus headlines). ``Human'' is the
difference in the human codes of all 1,500 items per corpus. The tuned difference estimate
uses one random validation sample of 200 hand-coded items per corpus. Under this estimand
the uncorrected estimate has no sampling error, so no interval can signal its bias.}
\label{tab:illustration}
\footnotesize
\begin{tabular}{@{}lrrrc@{}}
\toprule
Topic & Human & Uncorrected & Tuned & 95\% interval \\
\midrule
""" + "\n".join(lines5) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(OUT, "tab5_illustration.tex"), "w").write(T5)
print("tab5")
