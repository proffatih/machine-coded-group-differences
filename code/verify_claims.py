#!/usr/bin/env python3
"""Recompute every numeric claim in the manuscript from the analysis artefacts.

Numbers drift when a figure is regenerated or a table changes and the prose is not
updated with it. This recomputes each claim and reports the computed and stated values
side by side so a mismatch is fixed rather than argued about. It is run until it prints
zero mismatches.

Usage: python code/verify_claims.py
"""
import csv, json, os, sys, collections, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
if not os.path.isdir(RES):  # author working copy keeps results beside the paper folder
    RES = os.path.join(os.path.dirname(ROOT), "Social_Science_Computer_Review", "results")

LLM = {"olmo2-1b", "qwen2.5-1.5b", "smollm2-1.7b",
       "granite3.1-2b", "qwen2.5-3b", "phi3.5-mini"}
SUP = {"tfidf-logreg", "tfidf-svm"}


def rd(n):
    return list(csv.DictReader(open(os.path.join(RES, n), encoding="utf-8")))


def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return float(np.mean(xs)) if xs else float("nan")


cacc, lad, dif = rd("corpus_accuracy.csv"), rd("invariance_ladder.csv"), rd("dif_tests.csv")
sg, top, prev = rd("subgroup_accuracy.csv"), rd("per_topic_recall_gaps.csv"), rd("prevalence_bias.csv")
iv = json.load(open(os.path.join(RES, "metrics", "intervals.json")))
mech = json.load(open(os.path.join(RES, "metrics", "mechanism_summary.json")))

by_mc = collections.defaultdict(list)
for r in cacc:
    by_mc[(r["model"], r["condition"])].append(r["accuracy"])

fam_acc = {}
for fam, mem in (("llm", LLM), ("sup", SUP)):
    fam_acc[fam] = mean([r["accuracy"] for r in cacc
                         if r["model"] in mem and r["condition"] in ("zs", "sup")])

std = collections.defaultdict(list)
for r in sg:
    std[r["grouping"]].append(abs(float(r["raw_minus_standardised"])))

cells = collections.defaultdict(dict)
for r in sg:
    cells[(r["model"], r["condition"], r["grouping"])][r["group"]] = float(r["accuracy"])
gap = collections.defaultdict(list)
for (m, c, g), v in cells.items():
    if len(v) > 1:
        gap[g].append(max(v.values()) - min(v.values()))

tg = collections.defaultdict(list)
for r in top:
    if r["grouping"] == "venue":
        tg[int(r["topic"])].append(float(r["gap"]))

dsh = collections.defaultdict(lambda: collections.defaultdict(list))
for r in dif:
    fam = "sup" if r["model"] in SUP else "llm"
    dsh[fam]["u"].append(float(r["uniform_p"]) < .05)
    dsh[fam]["n"].append(float(r["nonuniform_p"]) < .05)
    if r["uniform_or"]:
        dsh[fam]["or"].append(float(r["uniform_or"]))

ladf = collections.defaultdict(list)
for r in lad:
    ladf["sup" if r["model"] in SUP else "llm"].append(r)

chi = collections.defaultdict(list)
for r in lad:
    chi[("sup" if r["model"] in SUP else "llm", r["grouping"])].append(float(r["cmi_pairwise_mean"]))
cmi_m = collections.defaultdict(list)
for r in lad:
    cmi_m[r["model"]].append(float(r["cmi_pairwise_mean"]))
wcr = list(csv.DictReader(open(os.path.join(RES, "within_corpus_invariance.csv"), encoding="utf-8")))
from scipy import stats as _st
GR = ("country", "venue", "length", "era")
wc = json.load(open(os.path.join(RES, "metrics", "within_corpus_summary.json")))["by_family"]
plc = json.load(open(os.path.join(RES, "metrics", "placebo_calibration.json")))
PS = os.path.join(RES, "pooled_supervised")
ps_lad = list(csv.DictReader(open(os.path.join(PS, "invariance_ladder.csv"), encoding="utf-8")))
ps_dif = list(csv.DictReader(open(os.path.join(PS, "dif_tests.csv"), encoding="utf-8")))
ps_acc = list(csv.DictReader(open(os.path.join(PS, "corpus_accuracy.csv"), encoding="utf-8")))
ps_prev = [r for r in csv.DictReader(open(os.path.join(PS, "prevalence_bias.csv"), encoding="utf-8")) if r["group"].startswith("DIFF:")]
ps_wc = json.load(open(os.path.join(PS, "metrics", "within_corpus_summary.json")))["by_family"]["sup"]
tgap = collections.defaultdict(list)
for r in top:
    tgap[("sup" if r["model"] in SUP else "llm", r["grouping"])].append(float(r["gap"]))

lvl = collections.defaultdict(lambda: collections.defaultdict(list))
for r in prev:
    k = "DIFF" if r["group"].startswith("DIFF:") else "LEVEL"
    lvl[(k, r["model"])]["n"].append(float(r["tv_naive"]))
    lvl[(k, r["model"])]["g"].append(float(r["tv_group_correction"]))

lvl2 = collections.defaultdict(lambda: collections.defaultdict(list))
for r in prev:
    if r["group"].startswith("DIFF:"):
        lvl2[r["model"]]["n"].append(float(r["tv_naive"])); lvl2[r["model"]]["p"].append(float(r["tv_pooled_correction"]))

pil = list(csv.DictReader(open(os.path.join(RES, "pilot", "response_format_pilot.csv"), encoding="utf-8")))

C = [
    ("pilot letter mean .130",           mean([r["letter_accuracy"] for r in pil]), .130, .001),
    ("pilot name mean .435",             mean([r["name_accuracy"] for r in pil]), .435, .001),
    ("pilot min ratio 2.5",              min(float(r["ratio"]) for r in pil), 2.5, .05),
    ("pilot max ratio 4.6",              max(float(r["ratio"]) for r in pil), 4.6, .05),
    ("pilot items per corpus 100",       float(pil[0]["n"]), 100, 0),
    ("chance rate .048",                 1 / 21, .048, .001),
    ("supervised accuracy .612",         fam_acc["sup"], .612, .001),
    ("llm accuracy .323",                fam_acc["llm"], .323, .002),
    ("supervised 89% higher",            100 * (fam_acc["sup"] / fam_acc["llm"] - 1), 89, .5),
    ("logreg .610",                      mean(by_mc[("tfidf-logreg", "sup")]), .610, .001),
    ("svm .615",                         mean(by_mc[("tfidf-svm", "sup")]), .615, .001),
    ("qwen3b zs .470",                   mean(by_mc[("qwen2.5-3b", "zs")]), .470, .002),
    ("qwen3b cb .502",                   mean(by_mc[("qwen2.5-3b", "cb")]), .502, .002),
    ("phi3.5 zs .474",                   mean(by_mc[("phi3.5-mini", "zs")]), .474, .002),
    ("olmo zs .223 (five-shot baseline)", mean(by_mc[("olmo2-1b", "zs")]), .223, .002),
    ("olmo fs .105",                     mean(by_mc[("olmo2-1b", "fs")]), .105, .002),
    ("std diff country .017",            mean(std["country"]), .017, .002),
    ("std diff era .013",                mean(std["era"]), .013, .002),
    ("std diff length .021",             mean(std["length"]), .021, .002),
    ("std diff venue .042",              mean(std["venue"]), .042, .001),
    ("accuracy gap venue .142",          mean(gap["venue"]), .142, .001),
    ("accuracy gap country .032",        mean(gap["country"]), .032, .001),
    ("accuracy gap era .028",            mean(gap["era"]), .028, .002),
    ("venue gap CI lo .112",             iv["accuracy_gap_by_grouping"]["venue"]["ci95"][0], .112, .001),
    ("venue gap CI hi .187",             iv["accuracy_gap_by_grouping"]["venue"]["ci95"][1], .187, .001),
    ("country gap CI lo .023",           iv["accuracy_gap_by_grouping"]["country"]["ci95"][0], .023, .001),
    ("country gap CI hi .044",           iv["accuracy_gap_by_grouping"]["country"]["ci95"][1], .044, .001),
    ("era gap CI lo .021",               iv["accuracy_gap_by_grouping"]["era"]["ci95"][0], .021, .002),
    ("era gap CI hi .035",               iv["accuracy_gap_by_grouping"]["era"]["ci95"][1], .035, .002),
    ("llm acc CI lo .206",               iv["accuracy_llm"]["ci95"][0], .206, .003),
    ("llm acc CI hi .431",               iv["accuracy_llm"]["ci95"][1], .431, .003),
    ("sup acc CI lo .610",               iv["accuracy_supervised"]["ci95"][0], .610, .001),
    ("sup acc CI hi .615",               iv["accuracy_supervised"]["ci95"][1], .615, .001),
    ("topic 20 venue gap .450",          mean(tg[20]), .450, .001),
    ("topic 10 venue gap .444",          mean(tg[10]), .444, .001),
    ("topic 12 venue gap .379",          mean(tg[12]), .379, .001),
    ("topic 1 venue gap .349",           mean(tg[1]), .349, .002),
    ("llm invariance tests 72",          len(ladf["llm"]), 72, 0),
    ("sup invariance tests 8",           len(ladf["sup"]), 8, 0),
    ("llm nonuniform .90",               mean([1.0 * x for x in dsh["llm"]["n"]]), .90, .01),
    ("sup nonuniform 1.00",              mean([1.0 * x for x in dsh["sup"]["n"]]), 1.00, .001),
    ("llm uniform .61",                  mean([1.0 * x for x in dsh["llm"]["u"]]), .61, .01),
    ("sup uniform .65",                  mean([1.0 * x for x in dsh["sup"]["u"]]), .65, .005),
    ("phi3.5 level naive .297",          mean(lvl[("LEVEL", "phi3.5-mini")]["n"]), .297, .002),
    ("phi3.5 level group .146",          mean(lvl[("LEVEL", "phi3.5-mini")]["g"]), .146, .002),
    ("logreg level naive .066",          mean(lvl[("LEVEL", "tfidf-logreg")]["n"]), .066, .001),
    ("svm level naive .101",             mean(lvl[("LEVEL", "tfidf-svm")]["n"]), .101, .001),
    ("logreg level subgroup .126",       mean(lvl[("LEVEL", "tfidf-logreg")]["g"]), .126, .001),
    ("svm level subgroup .116",          mean(lvl[("LEVEL", "tfidf-svm")]["g"]), .116, .001),
    ("all LLMs: subgroup corr helps level", float(all(mean(lvl[("LEVEL", m)]["g"]) < mean(lvl[("LEVEL", m)]["n"]) for m in LLM)), 1.0, 0),
    ("both SUP: subgroup corr hurts level", float(all(mean(lvl[("LEVEL", m)]["g"]) > mean(lvl[("LEVEL", m)]["n"]) for m in SUP)), 1.0, 0),
    ("five venue levels",                 float(len({r["group"] for r in sg if r["grouping"] == "venue"})), 5.0, 0),
    ("pooled excess llm .285",           iv["pooled_excess_error_llm"]["point"], .285, .002),
    ("pooled excess llm CI lo .215",     iv["pooled_excess_error_llm"]["ci95"][0], .215, .004),
    ("pooled excess llm CI hi .356",     iv["pooled_excess_error_llm"]["ci95"][1], .356, .004),
    ("pooled excess sup .165",           iv["pooled_excess_error_supervised"]["point"], .165, .001),
    ("pooled excess sup CI lo .159",     iv["pooled_excess_error_supervised"]["ci95"][0], .159, .001),
    ("pooled excess sup CI hi .171",     iv["pooled_excess_error_supervised"]["ci95"][1], .171, .001),
    ("pooled worse llm 97.9%",           iv["pooled_worse_share_llm"]["point"] * 100, 97.9, .1),
    ("pooled worse llm CI lo 94.4",      iv["pooled_worse_share_llm"]["ci95"][0] * 100, 94.4, .6),
    ("pooled worse sup 100%",            iv["pooled_worse_share_supervised"]["point"] * 100, 100, .01),
    ("subgroup excess llm +.058",        iv["subgroup_excess_error_llm"]["point"], .058, .002),
    ("subgroup excess llm CI lo -.035",  iv["subgroup_excess_error_llm"]["ci95"][0], -.035, .004),
    ("subgroup excess llm CI hi .186",   iv["subgroup_excess_error_llm"]["ci95"][1], .186, .004),
    ("subgroup excess sup +.114",        iv["subgroup_excess_error_supervised"]["point"], .114, .001),
    ("subgroup excess sup CI lo .098",   iv["subgroup_excess_error_supervised"]["ci95"][0], .098, .001),
    ("subgroup excess sup CI hi .129",   iv["subgroup_excess_error_supervised"]["ci95"][1], .129, .001),
    ("mechanism cells 260",              mech["n_cells"], 260, 0),
    ("mechanism singular cells 152",     mech["n_singular"], 152, 0),
    ("mechanism well-conditioned 108",   mech["n_well_conditioned"], 108, 0),
    ("pooled NNLS worse in all 260",     mech["share_observed_pooled_worse_nnls_all"] * 100, 100, .01),
    ("naive pred-vs-obs rho .99",        mech["spearman_naive_pred_vs_obs"][0], .99, .005),
    ("pooled pred-vs-obs rho .94",       mech["spearman_pooled_pred_vs_obs_exact"][0], .94, .005),
    ("excess pred-vs-obs rho .94",       mech["spearman_excess_pred_vs_obs_exact"][0], .94, .005),
    ("median relative error 11%",        mech["median_abs_rel_error_pooled"] * 100, 11, .5),
    ("median amplification 60",          mech["median_amplification_well"], 60, .5),
    ("log amplification-accuracy -.43",  mech["spearman_log_amplification_vs_accuracy"][0], -.43, .005),
    ("well-conditioned: 4 LLMs",         float(len(set(mech["models_in_well"]) & LLM)), 4, 0),
    ("well-conditioned: both sup",       float(len(set(mech["models_in_well"]) & SUP)), 2, 0),
    ("pilot first-3 letter share nyt .76", float(pil[0].get("letter_share_first3", "nan") or "nan"), .76, .005),
    ("llm subgroup beats pooled 86%",    100*np.mean([float(r["tv_group_correction"]) < float(r["tv_pooled_correction"]) for r in prev if r["group"].startswith("DIFF:") and r["model"] in LLM]), 86, .5),
    ("sup subgroup beats pooled 81%",    100*np.mean([float(r["tv_group_correction"]) < float(r["tv_pooled_correction"]) for r in prev if r["group"].startswith("DIFF:") and r["model"] in SUP]), 81, .5),
    ("pooled worse than none on DIFF for all 8 coders", float(all(mean(lvl2[m]["p"]) > mean(lvl2[m]["n"]) for m in LLM | SUP)), 1.0, 0),
    ("Gov Ops venue gap ~.45",           mean(tg[20]), .45, .005),
    ("country gap .032 x4 < venue .142", float(4 * mean(gap["country"]) < mean(gap["venue"])), 1.0, 0),
    ("dif n min 1500",                   min(float(r["n"]) for r in dif), 1500, 0),
    ("dif n max 5250",                   max(float(r["n"]) for r in dif), 5250, 0),
    ("perm floor p .0005 all 80 tests",  max(float(r["p"]) for r in lad), .0005, 1e-9),
    ("placebo asymptotic corpus 65%",    plc["single_corpus"]["reject_asymptotic_chi2"] * 100, 65, .5),
    ("placebo asymptotic pooled 71%",    plc["pooled_corpora"]["reject_asymptotic_chi2"] * 100, 71, .5),
    ("placebo perm corpus 4.5%",         plc["single_corpus"]["reject_permutation"] * 100, 4.5, .05),
    ("placebo perm pooled 8%",           plc["pooled_corpora"]["reject_permutation"] * 100, 8, .5),
    ("placebo DIF min 4%",               100 * min(plc[k][f] for k in ("single_corpus", "pooled_corpora") for f in ("reject_dif_uniform", "reject_dif_nonuniform")), 4, .5),
    ("placebo DIF max 7%",               100 * max(plc[k][f] for k in ("single_corpus", "pooled_corpora") for f in ("reject_dif_uniform", "reject_dif_nonuniform")), 7, .5),
    ("within llm 36 tests",              wc["llm"]["combined_tests"], 36, 0),
    ("within llm 35 rejected",           wc["llm"]["combined_rejected_share"] * 36, 35, .01),
    ("within llm max p .056",            wc["llm"]["combined_max_p"], .056, .001),
    ("within sup 2 of 4 rejected",       wc["sup"]["combined_rejected_share"] * 4, 2, .01),
    ("pooled-sup acc logreg .651",       mean([r["accuracy"] for r in ps_acc if r["model"] == "tfidf-logreg-pooled"]), .651, .001),
    ("pooled-sup acc svm .650",          mean([r["accuracy"] for r in ps_acc if r["model"] == "tfidf-svm-pooled"]), .650, .001),
    ("pooled-sup 8 tests",               len(ps_lad), 8, 0),
    ("pooled-sup all p .0005",           max(float(r["p"]) for r in ps_lad), .0005, 1e-9),
    ("pooled-sup nonuniform 92%",        100 * np.mean([float(r["nonuniform_p"]) < .05 for r in ps_dif]), 92, .5),
    ("pooled-sup within 4/4",            ps_wc["combined_rejected_share"] * 4, 4, .01),
    ("pooled-sup within max p .045",     ps_wc["combined_max_p"], .045, .001),
    ("pooled-sup pooled corr worse 100%", 100 * np.mean([float(r["tv_pooled_correction"]) > float(r["tv_naive"]) for r in ps_prev]), 100, .01),
    ("pooled-sup pooled excess .140",    np.mean([float(r["tv_pooled_correction"]) - float(r["tv_naive"]) for r in ps_prev]), .140, .001),
    ("sup topic gap min .12",            min(mean(tgap[("sup", g)]) for g in GR), .12, .005),
    ("sup topic gap max .34",            max(mean(tgap[("sup", g)]) for g in GR), .34, .005),
    ("llm topic gap min .07",            min(mean(tgap[("llm", g)]) for g in GR), .07, .005),
    ("llm topic gap max .23",            max(mean(tgap[("llm", g)]) for g in GR), .23, .005),
    ("sup topic gap > llm every grouping", float(all(mean(tgap[("sup", g)]) > mean(tgap[("llm", g)]) for g in GR)), 1, 0),
    ("five-shot lowers 5 of 6 LLMs",     float(sum(mean(by_mc[(m, "fs")]) < mean(by_mc[(m, "zs")]) for m in LLM)), 5, 0),
    ("smollm zs .074",                   mean(by_mc[("smollm2-1.7b", "zs")]), .074, .001),
    ("smollm fs .101",                   mean(by_mc[("smollm2-1.7b", "fs")]), .101, .001),
    ("phi most accurate LLM (zs)",       float(max(LLM, key=lambda m: mean(by_mc[(m, "zs")])) == "phi3.5-mini"), 1, 0),
    ("llm I min .036",                   min(mean(chi[("llm", g)]) for g in GR), .036, .0005),
    ("llm I max .117 (venue pairwise)",  max(mean(chi[("llm", g)]) for g in GR), .117, .0005),
    ("sup I min .016",                   min(mean(chi[("sup", g)]) for g in GR), .016, .0005),
    ("sup I max .069 (venue pairwise)",  max(mean(chi[("sup", g)]) for g in GR), .069, .0005),
    ("llm length I .054",                mean(chi[("llm", "length")]), .054, .0005),
    ("llm era I .036",                   mean(chi[("llm", "era")]), .036, .0005),
    ("sup length I .038",                mean(chi[("sup", "length")]), .038, .0005),
    ("sup era I .016",                   mean(chi[("sup", "era")]), .016, .0005),
    ("phi I .044",                       mean(cmi_m["phi3.5-mini"]), .044, .0005),
    ("qwen3b I .039",                    mean(cmi_m["qwen2.5-3b"]), .039, .0005),
    ("logreg I .030",                    mean(cmi_m["tfidf-logreg"]), .030, .0005),
    ("svm I .041",                       mean(cmi_m["tfidf-svm"]), .041, .0005),
    ("sup I < llm every grouping",       float(all(mean(chi[("sup", g)]) < mean(chi[("llm", g)]) for g in GR)), 1, 0),
    ("within llm length I .016",         wc["llm"]["combined_cmi_by_grouping"]["length"], .016, .0005),
    ("within llm era I .012",            wc["llm"]["combined_cmi_by_grouping"]["era"], .012, .0005),
    ("within sup length I .014",         wc["sup"]["combined_cmi_by_grouping"]["length"], .014, .0005),
    ("within sup era I .003",            wc["sup"]["combined_cmi_by_grouping"]["era"], .003, .0005),
    ("within/pooled llm length ~1/3",    wc["llm"]["combined_cmi_by_grouping"]["length"] / mean(chi[("llm", "length")]), .33, .05),
    ("within/pooled llm era ~1/3",       wc["llm"]["combined_cmi_by_grouping"]["era"] / mean(chi[("llm", "era")]), .33, .05),
    ("within llm DIF tests 252",         len([r for r in wcr if r["model"] not in SUP]), 252, 0),
    ("within llm nonuniform 10%",        100 * np.mean([float(r["nonuniform_p"]) < .05 for r in wcr if r["model"] not in SUP]), 10, .5),
    ("within llm uniform 12%",           100 * np.mean([float(r["uniform_p"]) < .05 for r in wcr if r["model"] not in SUP]), 12, .5),
    ("placebo corpus splits 420",        plc["single_corpus"]["n_splits"], 420, 0),
    ("placebo pooled splits 100",        plc["pooled_corpora"]["n_splits"], 100, 0),
    ("placebo perm draws 499",           plc["n_perm"], 499, 0),
    ("placebo 8/100 CI lo 3.5%",         100 * _st.beta.ppf(.025, 8, 93), 3.5, .05),
    ("placebo 8/100 CI hi 15%",          100 * _st.beta.ppf(.975, 9, 92), 15, .5),
    ("abs uniform OR llm 1.43",          float(np.median([np.exp(abs(np.log(float(r["uniform_or"])))) for r in dif if r["model"] not in SUP and r["uniform_or"]])), 1.43, .005),
    ("abs uniform OR sup 1.29",          float(np.median([np.exp(abs(np.log(float(r["uniform_or"])))) for r in dif if r["model"] in SUP and r["uniform_or"]])), 1.29, .005),
    ("venue mean pairwise acc diff .070", float(np.mean([np.mean([abs(a - b) for a, b in itertools.combinations(v.values(), 2)]) for (m, c, g), v in cells.items() if g == "venue"])), .070, .0005),
    ("venue pairwise ~2x country",       float(np.mean([np.mean([abs(a - b) for a, b in itertools.combinations(v.values(), 2)]) for (m, c, g), v in cells.items() if g == "venue"])) / mean(gap["country"]), 2.2, .15),
    ("pooled-sup subgroup excess .137",  np.mean([float(r["tv_group_correction"]) - float(r["tv_naive"]) for r in ps_prev]), .137, .001),
    ("pooled-sup I min .017",            min(np.mean([float(r["cmi_pairwise_mean"]) for r in ps_lad if r["grouping"] == g]) for g in GR), .017, .0005),
    ("pooled-sup I max .058",            max(np.mean([float(r["cmi_pairwise_mean"]) for r in ps_lad if r["grouping"] == g]) for g in GR), .058, .0005),
    ("pooled-sup recall gap > llm every grouping", float(all(np.mean([float(r["gap"]) for r in csv.DictReader(open(os.path.join(PS, "per_topic_recall_gaps.csv"))) if r["grouping"] == g]) > mean(tgap[("llm", g)]) for g in GR)), 1, 0),
    ("venue largest I llm",              float(max(GR, key=lambda g: mean(chi[("llm", g)])) == "venue"), 1, 0),
    ("venue largest I sup",              float(max(GR, key=lambda g: mean(chi[("sup", g)])) == "venue"), 1, 0),
    ("venue 5-group I all .205",         mean([r["cmi"] for r in lad if r["grouping"] == "venue"]), .205, .0005),
    ("pooled-sup subgroup<pooled 14/26", float(sum(float(r["tv_group_correction"]) < float(r["tv_pooled_correction"]) for r in ps_prev)), 14, 0),
    ("pooled-sup more accurate than any", float(min(mean([r["accuracy"] for r in ps_acc if r["model"] == m]) for m in ("tfidf-logreg-pooled", "tfidf-svm-pooled")) > max(mean(by_mc[k]) for k in by_mc)), 1, 0),
    ("within rejected 41 of 44",         wc["llm"]["combined_rejected_share"] * 36 + wc["sup"]["combined_rejected_share"] * 4 + ps_wc["combined_rejected_share"] * 4, 41, .01),
]

bad = 0
print(f"{'iddia':36s} {'hesap':>10s} {'metin':>10s}  durum")
for name, got, said, tol in C:
    ok = abs(float(got) - float(said)) <= tol
    bad += (not ok)
    print(f"{name:36s} {float(got):10.3f} {float(said):10.3f}  {'OK' if ok else '<<< UYUSMUYOR'}")
print(f"\ntoplam {len(C)} iddia, {bad} uyusmazlik")
sys.exit(1 if bad else 0)
