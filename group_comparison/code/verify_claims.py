#!/usr/bin/env python3
"""Check every number stated in the manuscript against results/claims.json.

Each check recomputes a value, formats it the way the text states it, and requires that
exact string to appear in the manuscript (author and anonymous versions). Run until it
reports zero mismatches.
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT, "manuscript", "manuscript.tex")):
    sys.exit("verify_claims: the article source (manuscript/) is not in this folder; "
             "the text checks need it (supplied with the submission)")
C = json.load(open(os.path.join(ROOT, "results", "claims.json")))
d3 = lambda x: f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.3f}"
d2 = lambda x: f"{x:.2f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}"
pct = lambda x: f"{round(100 * x)}"
pct1 = lambda x: "100" if round(100 * x, 1) == 100 else f"{100 * x:.1f}"
sg = lambda x: ("$-$" if x < 0 else "") + d3(abs(x))
chk = []
add = lambda name, text: chk.append((name, text))

# ---- simulation
add("conditions", f"{C['sim_conditions']} conditions with {C['sim_reps']} replications")
add("mcse", "Monte Carlo standard error of about one percentage")
assert abs(C["sim_mcse_95"] - .01) < .001
add("slope none", f"was {d2(C['sim_naive_slope_0.35_none'])}, {d2(C['sim_naive_slope_0.55_none'])}, and {d2(C['sim_naive_slope_0.75_none'])} at")
add("slope frequent", f"lowered the slope to {d2(C['sim_naive_slope_0.35_frequent'])}, {d2(C['sim_naive_slope_0.55_frequent'])}, and {d2(C['sim_naive_slope_0.75_frequent'])}")
add("naive cov", f"covered the truth in {pct(C['sim_cov_naive_min'])}\\% to {pct(C['sim_cov_naive_max'])}\\% of replications")
add("bcp .35 none/rare", f"its error was {d3(C['sim_bc_pooled_0.35_none_nv1000'])} when the coder erred identically in the two groups and {d3(C['sim_bc_pooled_0.35_rare_nv1000'])} when")
add("bcp .35 freq/all", f"but {d3(C['sim_bc_pooled_0.35_frequent_nv1000'])} when they differed on the three most frequent and {d3(C['sim_bc_pooled_0.35_all_nv1000'])} when they")
add("bcp .55", "(" + ", ".join(d3(C[f"sim_bc_pooled_0.55_{w}_nv1000"]) for w in ("none", "rare", "frequent", "all")) + ")")
add("bcp .75", "(" + ", ".join(d3(C[f"sim_bc_pooled_0.75_{w}_nv1000"]) for w in ("none", "rare", "frequent", "all")) + ")")
add("bcg/hand", f"{C['sim_bc_group_over_hand_0.35']:.2f}, {C['sim_bc_group_over_hand_0.55']:.2f}, and {C['sim_bc_group_over_hand_0.75']:.2f} times that of hand coding")
add("rg/hand", f"({C['sim_rg_group_over_hand_0.35']:.2f}, {C['sim_rg_group_over_hand_0.55']:.2f}, and {C['sim_rg_group_over_hand_0.75']:.2f}")
add("rg rare cov", f"covered as few as {pct(C['sim_rarecov_rg_group_min'])}\\% of the rare")
add("tuned max bias", f"in any condition was {d3(C['sim_ddtuned_absbias_max'])})")
lo = min(C[f"sim_cov_{e}_min"] for e in ("dd_group", "dd_tuned", "hand")); hi = max(C[f"sim_cov_{e}_max"] for e in ("dd_group", "dd_tuned", "hand"))
add("valid cov", f"difference in {pct(lo)}\\% to {pct(hi)}\\% of replications")
add("allcov", f"at least {pct1(min(C[f'sim_allcov_{e}_min'] for e in ('dd_group', 'dd_tuned', 'hand')))}\\% of")
add("below nominal", f"by more than Monte Carlo error in {C['sim_tuned_below_nominal']} of the 150 conditions")
add("tuned min cov", f"with a minimum of {pct(C['sim_cov_dd_tuned_min'])}\\%")
add("cal cov", f"covered the truth in {pct(C['sim_cov_cal_group_min'])}\\% to\n{pct(C['sim_cov_cal_group_max'])}\\% of replications, mostly above nominal")
add("tuned/hand", f"error averaged {pct(C['sim_dd_tuned_over_hand_0.35'])}\\%, {pct(C['sim_dd_tuned_over_hand_0.55'])}\\%, and {pct(C['sim_dd_tuned_over_hand_0.75'])}\\% at")
add("cal/hand", f"the calibration estimator's {pct(C['sim_cal_group_over_hand_0.35'])}\\%, {pct(C['sim_cal_group_over_hand_0.55'])}\\%, and {pct(C['sim_cal_group_over_hand_0.75'])}\\%")
b = C["sim_best_counts"]
add("best", f"calibration had the lowest error in {b['cal_group']} conditions, the tuned difference estimator in {b['dd_tuned']}, the uncorrected difference in {b['naive']}, and pooled back-calculation in {b['bc_pooled']}")
add("pooled design", f"estimators by at most\n{pct1(C['sim_pooled_vs_strat_maxreldiff'])}\\% and left their coverage between {pct(C['sim_pooled_cov_min'])}\\% and {pct(C['sim_pooled_cov_max'])}\\%")

# ---- real coders: design
add("coders", "Thirty coder--condition combinations")
assert C["real_coders"] == 30
add("within n", f"This gives {C['real_within_comparisons']} within-corpus comparisons")
add("between n", f"giving {C['real_between_comparisons']} comparisons")
add("acc range", f"ranges from {d3(C['acc_min'])} to {d3(C['acc_max'])}, against")
add("majority", f"{d3(C['majority_baseline_mean'])} for always predicting")
# decomposition
add("within decomposition", f"error of {d3(C['orc_within_naive_tv_error'])} on average, of which the shared part alone would have\nproduced {d3(C['orc_within_shared_tv'])} and the differential part {d3(C['orc_within_differential_tv'])}")
add("between decomposition", f"the error was {d3(C['orc_between_naive_tv_error'])}, with\nparts of {d3(C['orc_between_shared_tv'])} and {d3(C['orc_between_differential_tv'])}")
add("random split", f"the differential part averaged {d3(C['rand_differential_tv'])}. Between earlier and later\nhalves it averaged {d3(C['orc_period_differential_tv'])}, between shorter and longer halves {d3(C['orc_length_differential_tv'])}, and between corpora {d3(C['orc_between_differential_tv'])}")
# back-calculation
w = [C[f"real_within_{n}_bcpooled_worse_than_naive"] for n in ("50", "200")]
bt = [C[f"real_between_{n}_bcpooled_worse_than_naive"] for n in ("50", "200")]
add("bcp worse within", f"correction in {pct1(min(w))}\\% to {pct1(max(w))}\\% of within-corpus comparisons")
add("bcp worse between", f"and {pct1(min(bt))}\\% to {pct1(max(bt))}\\% of\nbetween-corpus ones")
assert min(w + bt) > .96
assert all(C[f"real_within_{n}_bcgroup_worse_than_naive"] == 1 for n in ("50", "200"))
assert all(C[f"real_between_{n}_{f}_bcgroup_worse_than_naive"] == 1 for n in ("50", "200") for f in ("xfer", "supin"))
add("bcg llm between", f"in {pct1(C['real_between_200_llm_bcgroup_worse_than_naive'])}\\% to {pct1(C['real_between_50_llm_bcgroup_worse_than_naive'])}\\% of the language models'")
gt = [C[f"real_{d}_{n}_bcgroup_worse_than_tuned"] for d in ("within", "between") for n in ("50", "200")]
add("bcg worse than tuned", f"tuned difference estimator in {pct1(min(gt))}\\% to 100\\% of all comparisons")
rg = [C[f"real_{d}_{n}_rg_worse_than_tuned"] for d in ("within", "between") for n in ("50", "200")]
assert max(rg) == 1
add("rg worse", f"{pct1(min(rg))}\\% to 100\\% of comparisons")
add("rg cov min", f"in as few as {pct1(C['real_rg_cov_min'])}\\% of draws")
add("unassigned llm small", f"average {C['unassigned_llmsmall_n']:.1f} of the 21 topics unassigned in a corpus, topics that held {pct(C['unassigned_llmsmall_mass'])}\\% of its items")
add("singular llm small", f"singular in {pct(C['orc_llmsmall_singular_share'])}\\% of their comparisons")
add("invertible", f"invertible ({C['orc_invertible_n']} of the {C['orc_n']:,} comparisons)")
assert C["llm_large_n_models"] == 3
assert C["orc_invertible_bc_group_tv_max"] < 1e-9
add("singular err", f"its error averaged {d3(C['orc_singular_bc_group_tv'])}")
add("unassigned large+cls", f"unassigned ({C['unassigned_llmlarge_n']:.1f} per corpus for the larger models, {C['unassigned_xfer_n']:.1f} and {C['unassigned_supin_n']:.1f} for the classifiers) held at most 1\\% of the items")
assert round(100 * max(C["unassigned_llmlarge_mass"], C["unassigned_xfer_mass"], C["unassigned_supin_mass"]), 1) <= 1.0
add("oracle large+cls", f"({d3(C['orc_llmlarge_bc_group_tv'])} for the larger language models, {d3(C['orc_xfer_bc_group_tv'])} for the transfer classifier, and {d3(C['orc_supin_bc_group_tv'])} for the in-corpus classifiers)")
lw = [C[f"real_{d}_{n}_llmlarge_bcpooled_worse_than_naive"] for d in ("within", "between") for n in ("50", "200")]
add("bcp worse large", f"no correction in {pct1(min(lw))}\\% to {pct1(max(lw))}\\% of the larger language models' comparisons")
add("limitations oracle", f"true matrices had an error of {d3(C['orc_llmlarge_bc_group_tv'])}, against {d3(C['orc_llmsmall_bc_group_tv'])} for the smaller models")
assert all(C[f"real_{d}_{n}_{f}_bcpooled_worse_than_naive"] == 1 for d in ("within", "between") for n in ("50", "200") for f in ("xfer", "supin"))
add("pooled invertible", f"Among the {C['orc_pooled_invertible_n']} comparisons whose pooled true\nmatrix was invertible, pooling made back-calculation worse than with per-group matrices in\n{pct1(C['orc_pooled_worse_than_group_invertible'])}\\%")
add("rho differential", f"(rank correlation {d2(C['orc_rho_pooled_vs_differential_invertible'])}, SE {d2(((1 - C['orc_rho_pooled_vs_differential_invertible'] ** 2) / (C['orc_pooled_invertible_n'] - 2)) ** .5)}, $n = {C['orc_pooled_invertible_n']}$)")
# naive coverage
add("naive cov llm", f"in {pct(C['real_naivecov_period_llm'])}\\%\nof comparisons between earlier and later halves, {pct(C['real_naivecov_length_llm'])}\\% between shorter and longer halves,\nand {pct(C['real_naivecov_between_llm'])}\\% between corpora")
add("naive cov xfer", f"the figures were {pct(C['real_naivecov_period_xfer'])}\\%, {pct(C['real_naivecov_length_xfer'])}\\%, and\n{pct(C['real_naivecov_between_xfer'])}\\%")
add("naive cov supin", f"nearly safe ({pct(C['real_naivecov_period_supin'])}\\% and {pct(C['real_naivecov_length_supin'])}\\%)")
add("naive cov supin between", f"covered the truth in {pct(C['real_naivecov_between_supin'])}\\% of comparisons")
# valid estimators
add("valid cov", f"{pct1(C['real_valid_cov_min'])}\\% to {pct1(C['real_valid_cov_max'])}\\% of draws, and over all topics in at least {pct1(C['real_valid_allcov_min'])}\\%")
add("cal cov real", f"covered it in {pct1(C['real_cal_cov_min'])}\\% to {pct1(C['real_cal_cov_max'])}\\%")
add("pooled real", f"errors by\n{pct1(C['real_pooled_vs_strat_mean_reldiff'])}\\% on average; the difference estimators and hand coding then covered the truth in\n{pct1(C['real_pooled_cov_min'])}\\% to {pct1(C['real_pooled_cov_max'])}\\% of draws and calibration in {pct1(C['real_pooled_cal_cov'])}\\%")
add("beats between", f"in {pct(C['real_between_50_llm_tuned_beats_naive'])}\\% of the language models' comparisons with 50 validation items")
n_llm_between = sum(1 for k in C if k.startswith("acc_") and k.endswith(("_zs", "_cb", "_fs"))) * 21
assert round((1 - C["real_between_200_llm_tuned_beats_naive"]) * n_llm_between) == 1 and C["real_between_200_xfer_tuned_beats_naive"] == 1
add("beats all but one", "and in all but one of them with 200")
add("beats between xfer", f"and in {pct(C['real_between_50_xfer_tuned_beats_naive'])}\\% and all of the transfer classifier's")
add("beats within", f"closer in {pct(C['real_within_50_llm_tuned_beats_naive'])}\\% and {pct(C['real_within_200_llm_tuned_beats_naive'])}\\% of\nthe language models' comparisons and {pct(C['real_within_50_xfer_tuned_beats_naive'])}\\% and {pct(C['real_within_200_xfer_tuned_beats_naive'])}\\% of the transfer classifier's")
add("supin naive vs tuned", f"(a total variation error of {d3(C['real_period_supin_50_naive_tv'])} between halves of\ndifferent years, against {d3(C['real_period_supin_200_dd_tuned_tv'])} for the tuned estimator")
# precision
add("gain rho", f"(rank correlation $-{d2(abs(C['gain_spearman_tuned_acc']))}$, SE {d2(((1 - C['gain_spearman_tuned_acc'] ** 2) / 28) ** .5)}, $n = 30$)")
add("gain values", f"to {d2(C['gain_llm_tuned_range'][0])} for the best language model (Granite 3.1 8B), {d2(C['gain_tfidf-logreg-transfer_sup'])} for the transfer classifier, and {d2(C['gain_tfidf-logreg_sup'])} and {d2(C['gain_tfidf-svm_sup'])} for the in-corpus")
assert C["gain_llm_tuned_range"][0] == C["gain_granite3.1-8b_zs"]
add("plan rho", f"was {d2(C['plan_qwen2.5-3b_zs_0.03_rho'])} for Qwen2.5 3B zero-shot, {d2(C['plan_granite3.1-8b_zs_0.03_rho'])} for Granite 3.1 8B zero-shot, and {d2(C['plan_tfidf-logreg-transfer_sup_0.03_rho'])} for the transfer classifier")
add("plan n", f"needed a median of {round(C['plan_qwen2.5-3b_zs_0.03_tuned'])}, {round(C['plan_granite3.1-8b_zs_0.03_tuned'])}, and {round(C['plan_tfidf-logreg-transfer_sup_0.03_tuned'])} hand-coded items per group with the tuned estimator, against {round(C['plan_qwen2.5-3b_zs_0.03_hand'])} for hand coding alone")
add("untuned worse", f"worse than hand coding for {C['gain_plain_worse_than_hand_llm']} of the {C['gain_llm_n']}")
# illustration
q = lambda t, f: C[f"ill_qwen_{t}_{f}"]
add("ill intl naive", f"(a difference of {sg(q('International_affairs', 'naive'))})")
add("ill intl true", f"{100 * abs(q('International_affairs', 'true')):.1f} points ahead")
add("ill intl tuned", f"The tuned estimate is {sg(q('International_affairs', 'tuned'))}, with an interval of [{sg(q('International_affairs', 'tuned') - 1.96 * q('International_affairs', 'tuned_se'))}, {sg(q('International_affairs', 'tuned') + 1.96 * q('International_affairs', 'tuned_se'))}]")
b_, s_ = q("International_affairs", "naive_bias_est"), q("International_affairs", "naive_bias_se")
add("ill intl bias", f"bias at {d3(b_)} [{d3(b_ - 1.96 * s_)},\n{d3(b_ + 1.96 * s_)}], against a true bias of {d3(q('International_affairs', 'naive_bias_true'))}")
add("ill law", f"({sg(q('Law_and_crime', 'naive'))} against {sg(q('Law_and_crime', 'true'))} in the human codes)")
b_, s_ = q("Law_and_crime", "naive_bias_est"), q("Law_and_crime", "naive_bias_se")
add("ill law bias", f"estimated bias of {sg(b_)} [{sg(b_ - 1.96 * s_)}, {sg(b_ + 1.96 * s_)}]")
add("ill bcp", f"(international affairs {sg(q('International_affairs', 'bc_pooled'))})")
x = lambda t, f: C[f"ill_xfer_{t}_{f}"]
for t in ("International_affairs", "Government_operations"):
    b_, s_ = x(t, "naive_bias_est"), x(t, "naive_bias_se")
    add(f"ill xfer {t}", f"{sg(b_)} [{sg(b_ - 1.96 * s_)}, {sg(b_ + 1.96 * s_)}]")
add("ill xfer true", f"against a true bias of {sg(x('International_affairs', 'naive_bias_true'))}")
add("ill xfer gov true", f"largest bias, {d3(x('Government_operations', 'naive_bias_true'))} for government operations")
add("mdb", f"standard errors of about {d3(C['ill_bias_se_mean'])}, so only biases of\nroughly {f'{C[chr(105)+"ll_bias_mdb"]:.2f}'.lstrip('0')} or more")
assert C["ill_tuned_misses"] == 1 and not C["ill_qwen_Government_operations_covers"]
add("ill misses", "missed the true difference for one of the ten topic--coder pairs")
# planning grid statements
g = lambda p_, r: C[f"grid_0.03_{p_}_{r}"]
assert all(.74 <= 1 - r ** 2 <= .84 for r in (.4, .5)) and abs((1 - .7 ** 2) - .5) < .02


# ---- v3: within-group validity (results/validity_summary.json)
V = json.load(open(os.path.join(ROOT, "results", "validity_summary.json")))
mf = lambda k: V[f"corpus_{k}"]["macro_f1"]
add("macro-F1 ranges", f"ranged from {d2(mf('incorpus')[0])} to {d2(mf('incorpus')[2])} for the in-corpus classifiers (mean {d2(mf('incorpus')[1])}), from {d2(mf('transfer')[0])} to {d2(mf('transfer')[2])} for the transfer classifier ({d2(mf('transfer')[1])}), and from {d2(mf('llm_large')[0])} to {d2(mf('llm_large')[2])} for the larger and {d2(mf('llm_small')[0])} to {d2(mf('llm_small')[2])} for the smaller language models ({d2(mf('llm_large')[1])} and {d2(mf('llm_small')[1])})")
ece = lambda k: V[f"corpus_{k}"]["ece"][1]
add("ece", f"error of {d2(ece('llm_small'))} and {d2(ece('llm_large'))} on average for the smaller and larger models, against {d2(ece('incorpus'))} and {d2(ece('transfer'))} for the in-corpus and transfer classifiers")
add("f1 gap", f"differed by {d3(V['macro_f1_gap_mean'])} on average and by at most {d3(V['macro_f1_gap_max'])}")
sh_, bh_, an_ = V["share_systematic"], V["share_systematic_bh"], V["any_systematic_bh"]
add("recall within", f"{pct1(sh_['period'])}\\% of the topics had a recall difference with $p < .05$, about what chance produces, and {pct1(bh_['period'])}\\% survived false-discovery control; between shorter and longer halves, {pct1(sh_['length'])}\\% and {pct1(bh_['length'])}\\%")
add("recall between", f"Between corpora, {pct1(sh_['between'])}\\% and {pct1(bh_['between'])}\\% did, and {pct1(an_['between'])}\\% of the between-corpus comparisons")
add("abstract recall", f"differed between corpora for {pct(bh_['between'])}\\% of topics after false-discovery control")
assert sh_["period"] < .07 and sh_["length"] < .07
cv = V["coverage_by_tier"]; T = ["< .20", ".20-.35", ".35-.50", ">= .50"]
add("naive by tier", f"difference in {pct(cv['naive'][T[0]])}\\% of the comparisons whose weaker group had a macro-F1 below .20, {pct(cv['naive'][T[1]])}\\% between .20 and .35, {pct(cv['naive'][T[2]])}\\% between .35 and .50, and {pct(cv['naive'][T[3]])}\\% above .50")
add("rg by tier", f"covering {pct(cv['rg_group'][T[0]])}\\% below .20")
add("valid by tier", f"in {pct1(min(cv['dd_tuned'].values()))}\\% to {pct1(max(cv['dd_tuned'].values()))}\\% (tuned difference estimator) and {pct1(min(cv['cal_group'].values()))}\\% to {pct1(max(cv['cal_group'].values()))}\\% (calibration)")
g = V["tuned_over_hand_by_tier"]
add("gain by tier", f"median of {g[T[0]][1]:.2f} below .20, {d2(g[T[1]][1])} between .20 and .35, {d2(g[T[2]][1])} between .35 and .50, and {d2(g[T[3]][1])} above .50")
sp = V["sampling"]
add("sampling", f"bias of $-${d3(abs(sp['100']['bias']))} and a standard deviation of {d3(sp['100']['sd'])}")
add("sampling tier", f"in {pct(sp['100']['tier_correct'])}\\% of draws ({pct(sp['200']['tier_correct'])}\\% with 200 items)")
assert V["n_comparisons"] == 420
add("tier best", f"and {pct(cv['naive'][T[3]])}\\% above .50, so not fully even in the best tier")

# ---- v3: error in the hand codes (results/label_noise_summary.json)
N = json.load(open(os.path.join(ROOT, "results", "label_noise_summary.json")))
ch = [N[dz][k]["dd_tuned"]["cov_human"] for dz in N for k in N[dz]]
add("noise human target", f"in {pct1(min(ch))}\\% to {pct1(max(ch))}\\% of draws in every setting")
att = [N["within"][f"symmetric{e}"]["_target"]["focal_attenuation_median"] for e in ("0.05", "0.1", "0.2")]
assert [d2(a) for a in att] == [d2(N["between"][f"symmetric{e}"]["_target"]["focal_attenuation_median"]) for e in ("0.05", "0.1", "0.2")]   # same within and between at the stated precision
add("noise attenuation", "by a median factor of " + ", ".join(d2(a) for a in att[:2]) + f", and {d2(att[2])}")
cw = [N["within"][f"symmetric{e}"]["dd_tuned"]["cov_latent"] for e in ("0.0", "0.05", "0.1", "0.2")]
cb = [N["between"][f"symmetric{e}"]["dd_tuned"]["cov_latent"] for e in ("0.05", "0.1", "0.2")]
add("noise within", f"within corpora from {pct1(cw[0])}\\% to {pct1(cw[1])}\\%, {pct1(cw[2])}\\%, and {pct1(cw[3])}\\%")
add("noise between", f"to {pct1(cb[0])}\\%, {pct1(cb[1])}\\%, and {pct1(cb[2])}\\%")
add("noise differential", f"gave {pct1(N['within']['differential']['dd_tuned']['cov_latent'])}\\% within and {pct1(N['between']['differential']['dd_tuned']['cov_latent'])}\\% between")
add("noise naive", f"covered the CAP difference in {pct(N['within']['symmetric0.0']['naive']['cov_latent'])}\\% and {pct(N['between']['symmetric0.0']['naive']['cov_latent'])}\\% of draws")
assert all(len({pct(N[dz][k]["naive"]["cov_latent"]) for k in N[dz]}) == 1 for dz in N)   # "regardless" of the hand codes
conf = [N[dz][f"confusable{e}"]["dd_tuned"]["cov_latent"] for dz in N for e in ("0.05", "0.1", "0.2")]
add("noise confusable", f"(no lower than {pct1(min(conf))}\\%)")

# ---- v3: other schemes (results/schemes_summary.json, results/schemes_table.json)
SS = json.load(open(os.path.join(ROOT, "results", "schemes_summary.json")))["sim"]
add("sim K bias", f"(largest absolute bias {d3(max(SS[k]['max_abs_bias_valid'] for k in SS))})")
add("sim K cov", f"in at least {pct(min(SS[k]['min_focal_cov'][e] for k in SS for e in ('cal_group', 'dd_group', 'dd_tuned', 'hand')))}\\% of replications in every condition")
import pandas as _pd
_n = _pd.read_csv(os.path.join(ROOT, "results", "sim_schemes.csv")); _n = _n[_n.estimator == "naive"].focal_coverage
add("sim K naive", f"covered it in {pct(_n.min())}\\% to {pct(_n.max())}\\% ({pct(_n.mean())}\\% on average)")
add("sim K2 bcp", f"with two, in {pct(SS['2']['bc_pooled_worse_than_naive']['frequent'])}\\% of replications")
add("sim K2 rg", f"per-category correction covered as few as {pct(SS['2']['min_focal_cov']['rg_group'])}\\% in some conditions")
TB = json.load(open(os.path.join(ROOT, "results", "schemes_table.json")))
merged = ["random10", "random5", "random2", "focal_binary"]
vv = [TB[s][d][e] for s in merged for d in ("within", "between") for e in ("cal_group", "dd_tuned")]
add("real schemes valid", f"difference estimators covered the truth in {pct(min(vv))}\\% to {pct(max(vv))}\\% of draws")
ww = [TB[s][d]["worse"] for s in merged for d in ("within", "between")]
add("real schemes bcp", f"worse than no correction in {pct(min(ww))}\\% to {pct(max(ww))}\\% of comparisons")
nn = [TB[s][d]["naive"] for s in merged for d in ("within", "between")]
add("real schemes naive", f"the uncorrected difference covered the truth in {pct(min(nn))}\\% to {pct(max(nn))}\\%")

fails = 0
tex = "\n".join(open(os.path.join(ROOT, "manuscript", fn), encoding="utf-8").read()
                for fn in ("manuscript.tex", "appendix.tex"))
texn = re.sub(r"\s+", " ", tex)
for name, text in chk:
    if re.sub(r"\s+", " ", text) not in texn:
        fails += 1
        print(f"UYUSMAZLIK {name}: {re.sub(chr(10), ' ', text)}")

# ---- v3: intro example, rare topics, cross-national pairs, wins against the uncorrected difference
import pandas as _pd2
_o = _pd2.read_csv(os.path.join(ROOT, "results", "real_oracle.csv"))
_llm = lambda df: df[~df.model.str.startswith("tfidf")]
_ob = _llm(_o[_o.comparison == "uk_acts|us_bills"])
add("intro bills-acts", f"(a total variation distance of {d3(_ob.naive_tv_error.mean())}) exceeded the size of the difference itself ({d3(_ob.true_tv_difference.mean())})")
assert _ob.naive_tv_error.mean() > _ob.true_tv_difference.mean()
_r = _pd2.read_csv(os.path.join(ROOT, "results", "real_summary.csv"))
_rs = _r[_r.sampling == "stratified"]
_x = lambda pair, e: _llm(_rs[(_rs.comparison == pair) & (_rs.n_v == 100) & (_rs.estimator == e)]).focal_coverage.mean()
add("cross-national", f"uncorrected intervals covered the most prevalent topic's difference in {pct(_x('uk_acts|us_bills', 'naive'))}\\% and {pct(_x('uk_media|us_nyt', 'naive'))}\\% of draws and the tuned estimator's in {pct1(_x('uk_acts|us_bills', 'dd_tuned'))}\\% and {pct1(_x('uk_media|us_nyt', 'dd_tuned'))}\\%")
add("wins main", f"in only {pct(C['real_within_50_llm_tuned_beats_naive'])}\\% of the language models' comparisons with 50 validation items per group and {pct(C['real_within_200_llm_tuned_beats_naive'])}\\% with 200")
add("wins between main", f"between corpora the tuned estimator was closer in {pct(C['real_between_50_llm_tuned_beats_naive'])}\\% and nearly all")
_g = _pd2.read_csv(os.path.join(ROOT, "results", "precision_gain.csv"))
assert 1 - _g.tuned_over_hand.min() <= .2 + 1e-9          # "saved at most about a fifth"
_hi = [N[dz][k]["dd_tuned"]["cov_latent"] for dz in N for k in ("symmetric0.05", "symmetric0.1", "confusable0.05", "confusable0.1")]
add("noise 90%", f"At 90\\% agreement or more the tuned intervals covered the CAP difference in at least {pct1(min(_hi))}\\% of draws")
add("noise 80% between", f"between-corpus intervals were too narrow ({pct1(N['between']['symmetric0.2']['dd_tuned']['cov_latent'])}\\%)")

# ---- v3: robustness (results/ext_summary.json)
X = json.load(open(os.path.join(ROOT, "results", "ext_summary.json")))
add("rob pop", f"in {pct1(X['pop_valid_cov'][0])}\\% to {pct1(X['pop_valid_cov'][1])}\\% of draws and those of calibration in {pct1(X['pop_cal_cov'][0])}\\% to {pct1(X['pop_cal_cov'][1])}\\%, and pooled back-calculation was worse than no correction in {pct1(X['pop_bcp_worse'][0])}\\% to {pct1(X['pop_bcp_worse'][1])}\\%")
add("rob shrink", f"was worse than no correction in {pct1(X['bc_shrink_worse_than_naive']['between'])}\\% of the between-corpus and in all within-corpus comparisons")
assert X["bc_shrink_worse_than_naive"]["within"] == 1 and X["bc_shrink_worse_than_tuned"]["within"] == 1 and X["bc_shrink_worse_than_tuned"]["between"] == 1
add("rob shrink boot", f"intervals covered the truth in {pct1(X['boot_cov']['bc_shrink']['between'])}\\% and {pct1(X['boot_cov']['bc_shrink']['within'])}\\% of draws")
add("rob empsim", f"(worse than no correction in {pct1(X['empsim_bcp_worse']['empirical'][0])}\\% to {pct1(X['empsim_bcp_worse']['empirical'][1])}\\% of comparisons, and in {pct1(X['empsim_bcp_worse']['repaired'][0])}\\% to {pct1(X['empsim_bcp_worse']['repaired'][1])}\\% when the matrices were made well conditioned), while calibration and the tuned estimator covered {pct1(X['empsim_valid_cov'][0])}\\% to {pct1(X['empsim_valid_cov'][1])}\\%")
add("rob bins", f"covered {pct1(X['bins']['dd_tuned']['>=.15']['cov'])}\\% of the differences of topics with a prevalence of .15 or more and {pct1(X['bins']['dd_tuned']['<.02']['cov'])}\\% below .02")
add("rob joint", f"in {pct1(min(X['joint_cov']['dd_tuned'].values()))}\\% to {pct1(max(X['joint_cov']['dd_tuned'].values()))}\\% of draws")
cq = X["coders_quality"]
add("rob prior", f"raised their accuracy from {d2(cq['llm+prior']['accuracy_original'])} to {d2(cq['llm+prior']['accuracy'])} without changing any conclusion")
assert cq["llm+prior"]["unassigned"] == 0
add("rob synth", f"synthetic coders with an accuracy of {d2(cq['synthetic-0.5']['accuracy'])} and {d2(cq['synthetic-0.8']['accuracy'])} made pooled back-calculation worse than no correction in every comparison")
assert all(v == 1 for k, v in X["coders_bcp_worse"].items() if k.startswith("synthetic"))
_toh = [v for k, v in X["coders_tuned_over_hand"].items() if k.startswith("synthetic")]
add("rob synth gain", f"cut the error of hand coding alone by {pct(1 - max(_toh))}\\% to {pct(1 - min(_toh))}\\%")
assert all(v == 0 for k, v in X["coders_tuned_beats_naive"].items() if k.startswith("synthetic"))

# ---- figure captions: uncertainty actually drawn (results/figure_uncertainty.json)
UNCJ = json.load(open(os.path.join(ROOT, "results", "figure_uncertainty.json")))
add("fig1 mcse", f"Monte Carlo standard errors (at most {d3(UNCJ['fig1_max_halfwidth'])})")
add("fig2 tiers", f"({V['tier_counts']['< .20']}, {V['tier_counts']['.20-.35']}, {V['tier_counts']['.35-.50']}, and {V['tier_counts']['>= .50']} comparisons)")

# ---- cover letter (SUBMISSION/cover_letter.md)
_clp = os.path.join(ROOT, "SUBMISSION", "cover_letter.md")   # not part of the replication archive
cl = re.sub(r"\s+", " ", open(_clp, encoding="utf-8").read()) if os.path.exists(_clp) else None
import subprocess
_wc = subprocess.run([sys.executable, os.path.join(ROOT, "code", "wordcount.py")], capture_output=True, text=True).stdout
_tot = int(re.search(r"TOTAL (\d+)", _wc).group(1))
for name, text in (("cover >96", "more than 96%"), ("cover combos", "30 real coder–condition combinations"),
                   ("cover words", f"is about {round(_tot, -2):,}")):
    if cl is not None and text not in cl:
        fails += 1; print(f"UYUSMAZLIK [cover] {name}: {text}")
assert _tot <= 6000
if f"figure legends; references excluded): {_tot:,}" not in texn:
    fails += 1; print(f"UYUSMAZLIK title-page word count {_tot}")
print(f"toplam {len(chk)} iddia (ana metin + ek), {fails} uyusmazlik")
sys.exit(1 if fails else 0)
