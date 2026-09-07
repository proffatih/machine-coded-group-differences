#!/usr/bin/env python3
"""Check every numeric claim in the manuscript against the analysis artefacts.

Numbers in a paper drift: a figure is regenerated, a table changes, the prose does not.
This recomputes each claim from results/ and compares it to what the manuscript says.
A mismatch is reported with both values so it can be fixed rather than argued about.

Usage: python code/verify_claims.py
"""
import csv, json, os, re, sys, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rd(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return float(np.mean(xs)) if xs else float("nan")


acc = rd("subgroup_accuracy.csv")
dif = rd("dif_tests.csv")
lad = rd("invariance_ladder.csv")
top = rd("per_topic_recall_gaps.csv")
prev = rd("prevalence_bias.csv")
head = json.load(open(os.path.join(ROOT, "results", "metrics", "headline.json"),
                     encoding="utf-8"))

by_mc = collections.defaultdict(list)
for r in acc:
    by_mc[(r["model"], r["condition"])].append(r["accuracy"])
by_m = collections.defaultdict(list)
for r in acc:
    by_m[r["model"]].append(r["accuracy"])

# raw - standardised, by grouping
std = collections.defaultdict(list)
for r in acc:
    std[r["grouping"]].append(abs(float(r["raw_minus_standardised"])))

# accuracy gap by grouping
gap = collections.defaultdict(list)
tmp = collections.defaultdict(dict)
for r in acc:
    tmp[(r["model"], r["condition"], r["grouping"])][r["group"]] = float(r["accuracy"])
for (m, c, g), v in tmp.items():
    if len(v) > 1:
        gap[g].append(max(v.values()) - min(v.values()))

# DIF shares
dsh = collections.defaultdict(lambda: collections.defaultdict(list))
for r in dif:
    dsh[r["grouping"]]["u"].append(float(r["uniform_p"]) < .05)
    dsh[r["grouping"]]["n"].append(float(r["nonuniform_p"]) < .05)

# scaled chi2
chi = collections.defaultdict(list)
for r in lad:
    chi[r["grouping"]].append(float(r["scaled_chi2_df"]))

# per-topic venue gaps
tg = collections.defaultdict(list)
for r in top:
    if r["grouping"] == "venue":
        tg[int(r["topic"])].append(float(r["gap"]))

# prevalence
pv = collections.defaultdict(lambda: collections.defaultdict(list))
for r in prev:
    k = "DIFF" if r["group"].startswith("DIFF:") else "LEVEL"
    pv[(k, r["model"])]["n"].append(float(r["tv_naive"]))
    pv[(k, r["model"])]["p"].append(float(r["tv_pooled_correction"]))
    pv[(k, r["model"])]["g"].append(float(r["tv_group_correction"]))
worse = collections.defaultdict(list)
for r in prev:
    k = "DIFF" if r["group"].startswith("DIFF:") else "LEVEL"
    worse[k].append(float(r["tv_pooled_correction"]) > float(r["tv_naive"]))

CLAIMS = [
    ("chance rate .048", 1 / 21, .048, .001),
    ("SmolLM2 zero-shot", mean(by_mc[("smollm2-1.7b", "zs")]), .074, .002),
    ("OLMo-2 zero-shot ~.23", mean(by_mc[("olmo2-1b", "zs")]), .227, .002),
    ("OLMo-2 codebook .230", mean(by_mc[("olmo2-1b", "cb")]), .230, .002),
    ("OLMo-2 five-shot .106", mean(by_mc[("olmo2-1b", "fs")]), .106, .002),
    ("Qwen2.5-3B zero-shot .469", mean(by_mc[("qwen2.5-3b", "zs")]), .469, .002),
    ("Qwen2.5-3B codebook .498", mean(by_mc[("qwen2.5-3b", "cb")]), .498, .002),
    ("std diff country .018", mean(std["country"]), .018, .002),
    ("std diff era .014", mean(std["era"]), .014, .002),
    ("std diff length .023", mean(std["length"]), .023, .002),
    ("std diff venue .041", mean(std["venue"]), .041, .002),
    ("accuracy gap venue .137", mean(gap["venue"]), .137, .002),
    ("accuracy gap country .032", mean(gap["country"]), .032, .002),
    ("accuracy gap period .026", mean(gap["era"]), .026, .002),
    ("venue worst gap .265", max(gap["venue"]), .265, .002),
    ("chi2/df period 3.4", mean(chi["era"]), 3.4, .05),
    ("chi2/df country 5.6", mean(chi["country"]), 5.6, .05),
    ("invariance rejected 100%", head["invariance_rejected_share"], 1.0, 1e-9),
    ("topic 20 venue gap .458", mean(tg[20]), .458, .002),
    ("topic 10 venue gap .452", mean(tg[10]), .452, .002),
    ("topic 12 venue gap .367", mean(tg[12]), .367, .002),
    ("topic 1 venue gap .354", mean(tg[1]), .354, .002),
    ("phi3.5 level naive .297", mean(pv[("LEVEL", "phi3.5-mini")]["n"]), .297, .002),
    ("phi3.5 level group .146", mean(pv[("LEVEL", "phi3.5-mini")]["g"]), .146, .002),
    ("phi3.5 diff naive .308", mean(pv[("DIFF", "phi3.5-mini")]["n"]), .308, .002),
    ("phi3.5 diff pooled .530", mean(pv[("DIFF", "phi3.5-mini")]["p"]), .530, .002),
    ("phi3.5 diff group .233", mean(pv[("DIFF", "phi3.5-mini")]["g"]), .233, .002),
    ("qwen3b diff naive .304", mean(pv[("DIFF", "qwen2.5-3b")]["n"]), .304, .002),
    ("qwen3b diff pooled .466", mean(pv[("DIFF", "qwen2.5-3b")]["p"]), .466, .002),
    ("smollm2 diff naive .645", mean(pv[("DIFF", "smollm2-1.7b")]["n"]), .645, .002),
    ("smollm2 diff pooled 1.067", mean(pv[("DIFF", "smollm2-1.7b")]["p"]), 1.067, .002),
    ("pooled worse on DIFF 97.9%", float(np.mean(worse["DIFF"])) * 100, 97.9, .1),
    ("pooled worse on LEVEL 42.4%", float(np.mean(worse["LEVEL"])) * 100, 42.4, .1),
]

# DIF shares stated as ranges in the text
dif_u = [mean([1.0 if x else 0.0 for x in dsh[g]["u"]]) for g in dsh]
dif_n = [mean([1.0 if x else 0.0 for x in dsh[g]["n"]]) for g in dsh]
CLAIMS += [
    ("non-uniform DIF min 89%", min(dif_n) * 100, 89, 1.0),
    ("non-uniform DIF max 94%", max(dif_n) * 100, 94, 1.0),
    ("uniform DIF min 44%", min(dif_u) * 100, 44, 1.0),
    ("uniform DIF max 63%", max(dif_u) * 100, 63, 1.0),
]

bad = 0
print(f"{'iddia':34s} {'hesap':>9s} {'metin':>9s}  durum")
for name, computed, stated, tol in CLAIMS:
    ok = abs(computed - stated) <= tol
    bad += (not ok)
    print(f"{name:34s} {computed:9.3f} {stated:9.3f}  {'OK' if ok else '<<< UYUSMUYOR'}")
print(f"\ntoplam {len(CLAIMS)} iddia, {bad} uyusmazlik")
sys.exit(1 if bad else 0)
