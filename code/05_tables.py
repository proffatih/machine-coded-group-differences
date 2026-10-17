#!/usr/bin/env python3
"""Emit the manuscript tables as LaTeX, straight from the artefacts.

No number in the manuscript is typed by hand. Table 1 and Table 2 describe the design
and can be built as soon as the corpora exist; the result tables require the analysis
to have run. Each file is \\input{} by manuscript.tex.

Usage: python code/05_tables.py
Output: manuscript/tables/*.tex
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(OUT, exist_ok=True)

VENUE = {"legislative": "Legislative", "legislative_oral": "Legisl. (oral)",
         "media": "Media", "executive": "Executive", "judicial": "Judicial"}

# apa7 `man` modunda metin blogu dar; uzun korpus adlariyla tablo saga 102pt tasiyordu.
# Kisa etiket tabloda, tam tanim caption'da.
SHORT = {
    "us_bills":  "Congressional bills",
    "us_nyt":    "NYT front page",
    "us_eo":     "Executive orders",
    "us_scotus": "Supreme Court cases",
    "uk_acts":   "Acts of Parliament",
    "uk_pmqs":   "PM's Questions",
    "uk_media":  "Newspaper front page",
}


def table1_corpora():
    path = os.path.join(ROOT, "data", "prep_report.json")
    if not os.path.exists(path):
        return None
    rep = json.load(open(path, encoding="utf-8"))
    rows = []
    for key in CORPORA:
        c = rep["corpora"].get(key)
        if not c:
            continue
        drop_pct = 100 * c["off_scheme_total"] / max(c["n_raw_kept"] + c["off_scheme_total"], 1)
        rows.append((SHORT.get(key, c["pretty"]), c["country"],
                     VENUE.get(c["venue"], c["venue"]),
                     c["n_sampled"], c["mean_words"], c["n_classes_present"], drop_pct))
    body = "\n".join(
        f"{p} & {v} & {co} & {n:,} & {w:.1f} & {k} & {d:.1f} \\\\" for p, co, v, n, w, k, d in rows)
    tex = r"""\begin{table}[htbp]
\centering
\caption{The seven Comparative Agendas Project corpora: U.S. Congressional bill
titles, New York Times front-page headlines, U.S. executive orders, U.S. Supreme
Court case summaries, UK Acts of Parliament long titles, UK Prime Minister's
Questions, and UK newspaper front-page headlines. All share the CAP master codebook
of 21 major policy topics; each is stratified-subsampled and split into disjoint
calibration and test halves of equal size. ``Off-scheme'' is the share of records
dropped because they carried a media-specific extension code or the residual
``other'' category rather than one of the 21 shared topics.}
\label{tab:corpora}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llcrrrr@{}}
\toprule
Corpus & Venue & Country & $n$ & Words & Topics & Off-sch.\ (\%) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def table2_models():
    sys.path.insert(0, os.path.join(ROOT, "code"))
    # 02_annotate.py torch cektigi icin MODELS sozlugunu metinden okuyoruz
    src = open(os.path.join(ROOT, "code", "02_annotate.py"), encoding="utf-8").read()
    block = src.split("MODELS = {", 1)[1].split("}", 1)[0]
    rows = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith('"'):
            continue
        key, rest = line.split(":", 1)
        repo, prec = [p.strip().strip('"()').strip() for p in rest.strip(" ()").split(",")]
        developer = repo.split("/")[0]
        rows.append((repo.split("/")[1], developer,
                     "4-bit NF4" if prec == "nf4" else "fp16"))
    rows.append(("TF-IDF + logistic regression", "scikit-learn", "supervised, fp64"))
    rows.append(("TF-IDF + calibrated linear SVM", "scikit-learn", "supervised, fp64"))
    body = "\n".join(
        (f"\\texttt{{{m}}} & {d} & {p} \\\\" if "TF-IDF" not in m
         else f"{m} & {d} & {p} \\\\") for m, d, p in rows)
    tex = r"""\begin{table}[htbp]
\centering
\caption{The eight coders. Six are open-weight instruction-tuned language models from
five independent developers; two are supervised classifiers trained on the
researcher's own labels. Gated releases (Llama, Gemma) were excluded so that the full pipeline can
be reproduced without credentials. All six language models are loaded in 4-bit NF4
quantisation, held constant across the roster so that a difference between models cannot
be an artefact of one being quantised and another not; it is also the realistic
deployment condition on the single consumer GPU used here.}
\label{tab:models}
\small
\begin{tabular}{lll}
\toprule
Model & Developer & Precision \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def main():
    made = []
    for name, fn in (("tab1_corpora.tex", table1_corpora), ("tab2_models.tex", table2_models)):
        tex = fn()
        if tex:
            with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
                fh.write(tex)
            made.append(name)
    print(f"{len(made)} tablo yazildi -> manuscript/tables/")
    for m in made:
        print("  ", m)
    if os.path.exists(os.path.join(ROOT, "results", "metrics", "headline.json")):
        for m in main_results():
            print("  ", m)
    else:
        print("\n(sonuc tablolari icin once 03_invariance.py calismali)")




# ---------------------------------------------------------------- result tables
def _read(name):
    import csv as _csv
    p = os.path.join(ROOT, "results", name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return list(_csv.DictReader(fh))


def _mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return sum(xs) / len(xs) if xs else float("nan")


MODEL_LABEL = {
    "smollm2-1.7b": "SmolLM2 1.7B", "olmo2-1b": "OLMo-2 1B",
    "granite3.1-2b": "Granite 3.1 2B", "qwen2.5-1.5b": "Qwen2.5 1.5B",
    "qwen2.5-3b": "Qwen2.5 3B", "phi3.5-mini": "Phi-3.5-mini 3.8B",
    "tfidf-logreg": "TF-IDF + logistic regression",
    "tfidf-svm": "TF-IDF + calibrated SVM",
}
GROUP_LABEL = {"country": "Country", "venue": "Institutional venue",
               "length": "Text length", "era": "Period"}


def table3_accuracy():
    rows = _read("corpus_accuracy.csv")
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["condition"]), []).append(r["accuracy"])
    # language models first, ordered by zero-shot agreement; supervised coders after,
    # since they are a different family rather than a further rung of the same ladder
    lms = sorted({m for m, c in by if c != "sup"}, key=lambda m: _mean(by.get((m, "zs"), [0])))
    sup = sorted({m for m, c in by if c == "sup"}, key=lambda m: _mean(by[(m, "sup")]))
    models = lms + sup
    body = []
    for m in models:
        if (m, "sup") in by:          # supervised coders have a single condition
            cells = f"\\multicolumn{{3}}{{c}}{{{_mean(by[(m, 'sup')]):.3f}}}"
        else:
            cells = " & ".join(f"{_mean(by.get((m, c), [])):.3f}" for c in ("zs", "cb", "fs"))
        if m == sup[0] if sup else False:
            body.append("\\midrule")
        body.append(f"{MODEL_LABEL.get(m, m)} & {cells} \\\\")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Agreement with the human CAP code, averaged over the seven corpora, by coder and
prompt condition. Chance is .048. The three prompt conditions apply to the language
models only: zero-shot, zero-shot with codebook definitions appended to each topic,
and five-shot with examples drawn from the corpus's own calibration half. The
supervised coders are trained by five-fold cross-fitting within each corpus, so every
item is predicted by a model that did not see it; they have a single condition.}
\label{tab:accuracy}
\small
\begin{tabular}{lrrr}
\toprule
Model & Zero-shot & Codebook & Five-shot \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def table4_dif():
    dif = _read("dif_tests.csv"); acc = _read("subgroup_accuracy.csv")
    lad = _read("invariance_ladder.csv")
    if not (dif and acc and lad):
        return None
    body = []
    for g in ("country", "venue", "length", "era"):
        d = [r for r in dif if r["grouping"] == g]
        l = [r for r in lad if r["grouping"] == g]
        a = {}
        for r in acc:
            if r["grouping"] == g:
                a.setdefault((r["model"], r["condition"]), {})[r["group"]] = float(r["accuracy"])
        gaps = [max(v.values()) - min(v.values()) for v in a.values() if len(v) > 1]
        uni = _mean([1.0 if float(r["uniform_p"]) < .05 else 0.0 for r in d])
        non = _mean([1.0 if float(r["nonuniform_p"]) < .05 else 0.0 for r in d])
        rej = _mean([1.0 if float(r["p"]) < .05 else 0.0 for r in l])
        body.append(f"{GROUP_LABEL[g]} & {_mean(gaps):.3f} & {uni:.2f} & {non:.2f} & "
                    f"{_mean([r['scaled_chi2_df'] for r in l]):.1f} & {rej:.2f} \\\\")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Differential item functioning and invariance of the response structure, by
grouping variable, pooled over models and prompt conditions. The accuracy gap is the
largest between-subgroup difference in raw agreement. The two DIF columns give the
share of pairwise likelihood-ratio tests significant at $p<.05$ after stratifying on
the true topic. Non-uniform DIF is near-universal while uniform DIF is not: the
instrument is not simply less accurate in one subgroup, it behaves differently.}
\label{tab:dif}
\small
\begin{tabular}{lrrrrr}
\toprule
 & Accuracy & \multicolumn{2}{c}{DIF significant} & \multicolumn{2}{c}{Invariance test} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Grouping & gap & Uniform & Non-uniform & $\chi^2$/df & Rejected \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def table5_correction():
    prev = _read("prevalence_bias.csv"); acc = _read("corpus_accuracy.csv")
    if not (prev and acc):
        return None
    a = {}
    for r in acc:
        a.setdefault(r["model"], []).append(r["accuracy"])
    a = {k: _mean(v) for k, v in a.items()}
    lvl, dif = {}, {}
    for r in prev:
        tgt = dif if r["group"].startswith("DIFF:") else lvl
        tgt.setdefault(r["model"], {"n": [], "p": [], "g": []})
        tgt[r["model"]]["n"].append(r["tv_naive"])
        tgt[r["model"]]["p"].append(r["tv_pooled_correction"])
        tgt[r["model"]]["g"].append(r["tv_group_correction"])
    body = []
    for m in sorted(a, key=lambda x: a[x]):
        L, D = lvl.get(m, {}), dif.get(m, {})
        body.append(
            f"{MODEL_LABEL.get(m, m)} & {a[m]:.3f} & "
            f"{_mean(L['n']):.3f} & {_mean(L['p']):.3f} & {_mean(L['g']):.3f} & "
            f"{_mean(D['n']):.3f} & {_mean(D['p']):.3f} & {_mean(D['g']):.3f} \\\\")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Total-variation error of estimated topic prevalence against the human coding,
uncorrected and after correcting with a confusion matrix estimated on the held-out
calibration half (non-negative least-squares inversion). Acc.\ is agreement with the
human code averaged over corpora and prompt conditions. Left: the prevalence of a single
subgroup. Right: the \emph{difference} in prevalence between two subgroups, which is the
quantity agenda-setting research reports. On the difference, a pooled correction is worse
than no correction for every coder. A subgroup-specific correction usually beats the
pooled one but is not reliably better than leaving the estimate alone, and for the
supervised coders it is clearly worse.}
\label{tab:correction}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{Single subgroup} & \multicolumn{3}{c}{Between-subgroup difference} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Coder & Acc. & None & Pooled & Subgr. & None & Pooled & Subgr. \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def main_results():
    made = []
    for name, fn in (("tab3_accuracy.tex", table3_accuracy),
                     ("tab4_dif.tex", table4_dif),
                     ("tab5_correction.tex", table5_correction)):
        tex = fn()
        if tex:
            with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
                fh.write(tex)
            made.append(name)
    return made


if __name__ == "__main__":
    main()
