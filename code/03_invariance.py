#!/usr/bin/env python3
"""Measurement invariance of LLM annotation across text subgroups.

The premise is psychometric rather than machine-learning. An LLM coder is treated
as a measurement instrument: the text is the item, the human CAP code is the latent
construct, and a subgroup of texts (country, institutional venue, length band, era)
is the grouping variable. The instrument is *invariant* if two texts carrying the
same true topic receive the same response distribution regardless of subgroup. When
it is not, the instrument exhibits differential item functioning, and any comparison
a social scientist draws between those subgroups is biased even when headline
accuracy looks acceptable.

Four analyses, in increasing order of what a reader has to accept:

  1. Conditional accuracy by subgroup, stratified on the true topic
  2. Logistic DIF as nested LR tests (Swaminathan & Rogers): uniform = topic + group
     against topic only; non-uniform = topic x group against topic + group
  3. Invariance of the confusion structure: one pooled confusion matrix against one
     per group, LR statistic referred to a within-topic permutation distribution
  4. The substantive cost: bias in estimated topic prevalence, and in the
     *between-subgroup difference* in prevalence, which is the quantity actually
     reported in agenda-setting research; pooled vs subgroup-specific confusion
     correction (Hopkins & King 2010 style). Confusion matrices come from the
     held-out calibration half and are applied to the test half, so the correction
     is the one a researcher could really run on a labelled validation sample.

Output: results/*.csv and results/metrics/headline.json
"""
import argparse, glob, json, os, sys, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES, SEED

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "results", "annotations")
OUT = os.path.join(ROOT, "results")
os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)

CODES = np.array(VALID_CODES)
K = len(CODES)
EPS = 1e-12


# ------------------------------------------------------------------ helpers
def load_all():
    """shard -> flat record arrays, with subgroup covariates attached."""
    recs = []
    for p in sorted(glob.glob(os.path.join(ANN, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        ckey = meta["corpus"]
        n = len(z["gold"])
        recs.append({
            "model": meta["model"], "condition": meta["condition"], "corpus": ckey,
            "probs": z["probs"], "y": np.searchsorted(CODES, z["gold"]),
            "pred": np.searchsorted(CODES, CODES[z["probs"].argmax(1)]),
            "split": z["split"].astype(str),
            "year": z["year"] if "year" in z else np.zeros(n, dtype=int),
            "n_words": z["n_words"] if "n_words" in z else np.zeros(n, dtype=int),
            "country": np.full(n, CORPORA[ckey][3]),
            "venue": np.full(n, CORPORA[ckey][4]),
        })
    return recs


def pool(recs, model, cond):
    """Stack every corpus for one (model, condition) into a single item pool."""
    sel = [r for r in recs if r["model"] == model and r["condition"] == cond]
    if not sel:
        return None
    out = {}
    for k in ("y", "pred", "split", "year", "n_words", "country", "venue"):
        out[k] = np.concatenate([r[k] for r in sel])
    out["probs"] = np.concatenate([r["probs"] for r in sel])
    out["corpus"] = np.concatenate([np.full(len(r["y"]), r["corpus"]) for r in sel])
    return out


def subgroupings(d):
    """Named grouping variables over the pooled items."""
    med_len = np.median(d["n_words"])
    yrs = d["year"][d["year"] > 0]
    med_yr = np.median(yrs) if len(yrs) else 0
    g = {
        "country": d["country"],
        "venue": d["venue"],
        "length": np.where(d["n_words"] >= med_len, "long", "short"),
    }
    if med_yr:
        era = np.where(d["year"] == 0, "unknown",
                       np.where(d["year"] >= med_yr, "later", "earlier"))
        g["era"] = era
    return g


# ------------------------------------------------- 2. logistic DIF by LR test
def _mnl_loglik_saturated(correct, strata, groups=None):
    """Log-likelihood of the saturated Bernoulli model over the given cells."""
    keys = strata if groups is None else np.char.add(
        np.char.add(strata.astype(str), "|"), groups.astype(str))
    ll = 0.0; df = 0
    for k in np.unique(keys):
        m = keys == k
        n = m.sum(); s = correct[m].sum()
        if n == 0:
            continue
        df += 1
        p = s / n
        if 0 < p < 1:
            ll += s * np.log(p) + (n - s) * np.log(1 - p)
    return ll, df


def dif_lr_tests(correct, topic, group):
    """Uniform and non-uniform DIF as nested LR tests, stratified on true topic.

    M0: P(correct) depends on topic only
    M1: M0 plus a constant group shift on the log-odds scale (uniform DIF)
    M2: P(correct) free in every topic x group cell (non-uniform DIF)
    """
    topic = topic.astype(str)
    group = group.astype(str)
    ll0, df0 = _mnl_loglik_saturated(correct, topic)
    ll2, df2 = _mnl_loglik_saturated(correct, topic, group)

    # M1: topic intercepts plus one group shift on the log-odds scale (2 groups only).
    # The likelihood depends on the data only through successes and trials per
    # topic-by-group cell, so the fit runs on those counts. For a given shift beta each
    # topic intercept is profiled out by a vectorised bisection (same [-12, 12] bounds
    # as before), and the profile likelihood is maximised over beta in [-6, 6]. This is
    # the same estimator as the earlier per-topic loop, computed on arrays instead.
    gvals = np.unique(group)
    if len(gvals) == 2:
        from scipy.optimize import minimize_scalar
        ref = gvals[0]
        tvals = np.unique(topic)
        g1 = (group != ref)
        n0 = np.array([np.sum((topic == t) & ~g1) for t in tvals], float)
        n1 = np.array([np.sum((topic == t) & g1) for t in tvals], float)
        s0 = np.array([correct[(topic == t) & ~g1].sum() for t in tvals], float)
        s1 = np.array([correct[(topic == t) & g1].sum() for t in tvals], float)

        def profile_a(beta):
            lo, hi = np.full(len(tvals), -12.0), np.full(len(tvals), 12.0)
            for _ in range(60):
                a = (lo + hi) / 2
                score = (s0 + s1) - n0 / (1 + np.exp(-a)) - n1 / (1 + np.exp(-(a + beta)))
                up = score > 0
                lo = np.where(up, a, lo); hi = np.where(up, hi, a)
            return (lo + hi) / 2

        def nll(beta):
            a = profile_a(beta)
            p0 = np.clip(1 / (1 + np.exp(-a)), EPS, 1 - EPS)
            p1 = np.clip(1 / (1 + np.exp(-(a + beta))), EPS, 1 - EPS)
            return -float((s0 * np.log(p0) + (n0 - s0) * np.log(1 - p0)
                           + s1 * np.log(p1) + (n1 - s1) * np.log(1 - p1)).sum())

        opt = minimize_scalar(nll, bounds=(-6.0, 6.0), method="bounded",
                              options={"xatol": 1e-9})
        beta = float(opt.x)
        ll1 = -nll(beta)
        df1 = df0 + 1
    else:
        beta, ll1, df1 = np.nan, ll0, df0

    def lr(lla, dfa, llb, dfb):
        stat = max(2 * (llb - lla), 0.0)
        ddf = max(dfb - dfa, 1)
        return stat, ddf, float(stats.chi2.sf(stat, ddf))

    uni = lr(ll0, df0, ll1, df1)
    non = lr(ll1, df1, ll2, df2)
    return {"uniform_beta": float(beta),
            "uniform_or": float(np.exp(beta)) if np.isfinite(beta) else None,
            "uniform_chi2": uni[0], "uniform_df": uni[1], "uniform_p": uni[2],
            "nonuniform_chi2": non[0], "nonuniform_df": non[1], "nonuniform_p": non[2]}


# ---------------------------------------- 3. invariance ladder over confusion
def confusion(y, pred, k=K):
    m = np.zeros((k, k), dtype=float)
    np.add.at(m, (y, pred), 1.0)
    return m


def multinom_loglik(counts):
    """Log-likelihood of a saturated row-wise multinomial fit to `counts`."""
    ll = 0.0
    for row in counts:
        n = row.sum()
        if n == 0:
            continue
        p = row / n
        nz = row > 0
        ll += float((row[nz] * np.log(p[nz])).sum())
    return ll


N_PERM = 1999
PERM_RNG = np.random.default_rng(20260905)


def _g2(y, pred, gi, n_groups):
    """Likelihood-ratio statistic for pred independent of group given the true topic.

    Equals 2 x (log-likelihood with a confusion matrix per group - log-likelihood with
    one pooled confusion matrix), i.e. the scalar-versus-configural comparison.
    """
    n = np.bincount((y * K + pred) * n_groups + gi,
                    minlength=K * K * n_groups).reshape(K, K, n_groups).astype(float)
    n_ig = n.sum(1, keepdims=True)
    n_ij = n.sum(2, keepdims=True)
    n_i = n.sum((1, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = n * np.log(n * n_i / (n_ig * n_ij))
    return float(2 * np.nansum(t))


def invariance_ladder(y, pred, group, n_perm=N_PERM, rng=None, keep_null=False):
    """Scalar (one pooled confusion matrix) against configural (one per group).

    The p-value is from a conditional permutation test: group labels are shuffled
    within each true topic, which holds the group-by-topic composition fixed and
    realises exactly the null that the coder's response profile does not depend on
    the group. The asymptotic chi-square reference is also reported, but it is not
    used for inference: with 21 x 21 sparse confusion tables it rejected 65% of random
    half-splits of a single corpus and 71% of pooled-corpus splits
    (results/metrics/placebo_calibration.json), while the permutation test stayed
    close to its nominal level.
    """
    rng = PERM_RNG if rng is None else rng
    gvals, gi = np.unique(group, return_inverse=True)
    G = len(gvals)
    stat = _g2(y, pred, gi, G)
    order = np.argsort(y, kind="stable")
    ys, g_sorted = y[order], gi[order]
    null = np.empty(n_perm)
    for b in range(n_perm):
        k = np.lexsort((rng.random(len(y)), ys))      # shuffle within topic blocks
        gp = np.empty_like(gi)
        gp[order] = g_sorted[k]
        null[b] = _g2(y, pred, gp, G)
    pooled = confusion(y, pred)
    df = 0
    for i in range(K):
        if pooled[i].sum() == 0:
            continue
        df += max(int((pooled[i] > 0).sum()) - 1, 0)
    df = max(df * (G - 1), 1)
    return {"n_groups": int(G), "lr_chi2": stat, "df": int(df),
            "p": float((1 + (null >= stat).sum()) / (n_perm + 1)),
            "p_asymptotic": float(stats.chi2.sf(stat, df)),
            "null_mean": float(null.mean()),
            "stat_over_null": float(stat / null.mean()),
            # bias-corrected conditional mutual information I(code; group | topic), nats:
            # the excess of the statistic over its null mean per item, halved. Unlike the
            # ratio, it does not depend on how many cells the tables have.
            "n": int(len(y)),
            "cmi": float((stat - null.mean()) / (2 * len(y))),
            "scaled_chi2_df": float(stat / df),
            **({"_null": null} if keep_null else {})}


def per_topic_recall_gap(y, pred, group):
    """Largest between-group gap in recall for any single topic (metric invariance)."""
    gvals = np.unique(group)
    worst, worst_t = 0.0, None
    rows = []
    for t in range(K):
        rec = {}
        for g in gvals:
            m = (group == g) & (y == t)
            if m.sum() >= 20:
                rec[g] = float((pred[m] == t).mean())
        if len(rec) >= 2:
            gap = max(rec.values()) - min(rec.values())
            rows.append({"topic": int(CODES[t]), "gap": gap,
                         **{f"recall_{g}": v for g, v in rec.items()}})
            if gap > worst:
                worst, worst_t = gap, int(CODES[t])
    return worst, worst_t, rows


# --------------------------------- 4. prevalence bias and confusion correction
def prevalence(idx, k=K):
    c = np.bincount(idx, minlength=k).astype(float)
    return c / max(c.sum(), 1)


def correct_prevalence(pred_idx, conf_mat):
    """Invert the confusion matrix to de-bias an estimated prevalence vector.

    conf_mat[i, j] = P(predicted j | true i). Solving p_obs = C^T p_true for p_true
    is the classifier-correction step social scientists know from Hopkins & King.
    Non-negative least squares keeps the answer on the simplex.
    """
    obs = prevalence(pred_idx)
    rows = conf_mat.sum(1, keepdims=True)
    C = np.divide(conf_mat, np.where(rows == 0, 1, rows))
    A = C.T
    try:
        from scipy.optimize import nnls
        x, _ = nnls(A, obs)
    except Exception:
        x = np.linalg.lstsq(A, obs, rcond=None)[0]
        x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else obs


def total_variation(a, b):
    return float(0.5 * np.abs(a - b).sum())


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_group_n", type=int, default=200)
    ap.add_argument("--ann_dir", default=None, help="read shards from here (robustness runs)")
    ap.add_argument("--out_dir", default=None, help="write results here (robustness runs)")
    args = ap.parse_args()
    global ANN, OUT
    if args.ann_dir:
        ANN = os.path.abspath(args.ann_dir)
    if args.out_dir:
        OUT = os.path.abspath(args.out_dir)
        os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)

    recs = load_all()
    if not recs:
        sys.exit("results/annotations/ bos - once 02_annotate.py calistir")

    pairs = sorted({(r["model"], r["condition"]) for r in recs})
    acc_rows, dif_rows, ladder_rows, topic_rows, prev_rows = [], [], [], [], []

    for model, cond in pairs:
        d = pool(recs, model, cond)
        te = d["split"] == "test"
        ca = d["split"] == "calib"
        y, pred = d["y"][te], d["pred"][te]
        y_c, pred_c = d["y"][ca], d["pred"][ca]
        correct = (y == pred).astype(float)
        topic = CODES[y]
        all_groups = subgroupings(d)
        groups = {k: v[te] for k, v in all_groups.items()}
        groups_c = {k: v[ca] for k, v in all_groups.items()}
        overall_acc = float(correct.mean())

        for gname, gvec in groups.items():
            gvals = [g for g in np.unique(gvec)
                     if (gvec == g).sum() >= args.min_group_n and g != "unknown"]
            if len(gvals) < 2:
                continue
            keep = np.isin(gvec, gvals)
            yk, pk, ck, tk, gk = y[keep], pred[keep], correct[keep], topic[keep], gvec[keep]
            # confusion matrices are estimated on the held-out CALIBRATION half and
            # applied to the test half -- the correction a practitioner can actually
            # perform, and the only version that is not circular
            gvec_c = groups_c[gname]
            keep_c = np.isin(gvec_c, gvals)
            yc, pc, gc = y_c[keep_c], pred_c[keep_c], gvec_c[keep_c]

            # 1. marginal and topic-conditional accuracy
            for g in gvals:
                m = gk == g
                # topic-standardised accuracy: reweight to the pooled topic mix
                w = np.zeros(len(CODES))
                acc_t = np.zeros(len(CODES))
                for t in range(K):
                    mt = m & (yk == CODES[t])
                    w[t] = (yk == CODES[t]).mean()
                    acc_t[t] = ck[mt].mean() if mt.sum() >= 10 else np.nan
                ok = ~np.isnan(acc_t)
                std_acc = float((acc_t[ok] * w[ok]).sum() / max(w[ok].sum(), EPS))
                acc_rows.append({
                    "model": model, "condition": cond, "grouping": gname, "group": str(g),
                    "n": int(m.sum()), "accuracy": float(ck[m].mean()),
                    "topic_standardised_accuracy": std_acc,
                    "raw_minus_standardised": float(ck[m].mean() - std_acc),
                })

            # 2. DIF (pairwise so the LR tests stay two-group)
            for ga, gb in itertools.combinations(gvals, 2):
                m = np.isin(gk, [ga, gb])
                res = dif_lr_tests(ck[m], tk[m], gk[m])
                res.update({"model": model, "condition": cond, "grouping": gname,
                            "group_a": str(ga), "group_b": str(gb),
                            "n": int(m.sum()),
                            "acc_a": float(ck[gk == ga].mean()),
                            "acc_b": float(ck[gk == gb].mean())})
                dif_rows.append(res)

            # 3. invariance ladder + worst per-topic recall gap
            lad = invariance_ladder(yk, pk, gk)
            gap, gap_topic, per_t = per_topic_recall_gap(yk, pk, gk)
            lad.update({"model": model, "condition": cond, "grouping": gname,
                        "overall_accuracy": overall_acc,
                        "worst_topic_recall_gap": gap, "worst_topic": gap_topic})
            ladder_rows.append(lad)
            for row in per_t:
                row.update({"model": model, "condition": cond, "grouping": gname})
                topic_rows.append(row)

            # 4. prevalence bias, and whether correction has to be group-specific
            pooled_conf = confusion(yc, pc)                       # calib, all groups
            group_conf = {g: confusion(yc[gc == g], pc[gc == g])  # calib, per group
                          for g in gvals}
            for g in gvals:
                m = gk == g
                true_p = prevalence(yk[m])
                naive_p = prevalence(pk[m])
                own_conf = group_conf[g]
                prev_rows.append({
                    "model": model, "condition": cond, "grouping": gname, "group": str(g),
                    "n": int(m.sum()),
                    "tv_naive": total_variation(naive_p, true_p),
                    "tv_pooled_correction": total_variation(
                        correct_prevalence(pk[m], pooled_conf), true_p),
                    "tv_group_correction": total_variation(
                        correct_prevalence(pk[m], own_conf), true_p),
                })
            # the quantity social scientists actually report: the DIFFERENCE
            for ga, gb in itertools.combinations(gvals, 2):
                ma, mb = gk == ga, gk == gb
                d_true = prevalence(yk[ma]) - prevalence(yk[mb])
                d_naive = prevalence(pk[ma]) - prevalence(pk[mb])
                d_pool = (correct_prevalence(pk[ma], pooled_conf)
                          - correct_prevalence(pk[mb], pooled_conf))
                d_grp = (correct_prevalence(pk[ma], group_conf[ga])
                         - correct_prevalence(pk[mb], group_conf[gb]))
                prev_rows.append({
                    "model": model, "condition": cond, "grouping": gname,
                    "group": f"DIFF:{ga}-{gb}", "n": int(ma.sum() + mb.sum()),
                    "tv_naive": float(np.abs(d_naive - d_true).sum() / 2),
                    "tv_pooled_correction": float(np.abs(d_pool - d_true).sum() / 2),
                    "tv_group_correction": float(np.abs(d_grp - d_true).sum() / 2),
                })

    def dump(rows, name):
        if not rows:
            return
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(c) is None else str(r.get(c, ""))
                                  for c in cols) + "\n")

    # One accuracy per (model, condition), averaged over the seven corpora on the test
    # half. Subgroup rows cannot be averaged for this purpose: they weight groupings
    # by how many subgroups each has, which is not a meaningful quantity. The
    # manuscript and Table 3 both read this file so they cannot drift apart.
    corpus_rows = []
    for model, cond in pairs:
        for r in recs:
            if r["model"] != model or r["condition"] != cond:
                continue
            te = r["split"] == "test"
            corpus_rows.append({
                "model": model, "condition": cond, "corpus": r["corpus"],
                "n_test": int(te.sum()),
                "accuracy": float((r["y"][te] == r["pred"][te]).mean()),
            })
    dump(corpus_rows, "corpus_accuracy.csv")

    dump(acc_rows, "subgroup_accuracy.csv")
    dump(dif_rows, "dif_tests.csv")
    dump(ladder_rows, "invariance_ladder.csv")
    dump(topic_rows, "per_topic_recall_gaps.csv")
    dump(prev_rows, "prevalence_bias.csv")

    diffs = [r for r in prev_rows if r["group"].startswith("DIFF:")]
    head = {
        "n_model_condition": len(pairs),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in acc_rows])),
        "dif_uniform_share_p05": float(np.mean([r["uniform_p"] < 0.05 for r in dif_rows])),
        "dif_nonuniform_share_p05": float(np.mean([r["nonuniform_p"] < 0.05 for r in dif_rows])),
        "invariance_rejected_share": float(np.mean([r["p"] < 0.05 for r in ladder_rows])),
        "mean_worst_topic_recall_gap": float(np.mean([r["worst_topic_recall_gap"]
                                                      for r in ladder_rows])),
        "prevalence_tv_naive": float(np.mean([r["tv_naive"] for r in prev_rows
                                              if not r["group"].startswith("DIFF:")])),
        "prevalence_tv_pooled": float(np.mean([r["tv_pooled_correction"] for r in prev_rows
                                               if not r["group"].startswith("DIFF:")])),
        "prevalence_tv_group": float(np.mean([r["tv_group_correction"] for r in prev_rows
                                              if not r["group"].startswith("DIFF:")])),
        "difference_tv_naive": float(np.mean([r["tv_naive"] for r in diffs])) if diffs else None,
        "difference_tv_pooled": float(np.mean([r["tv_pooled_correction"] for r in diffs])) if diffs else None,
        "difference_tv_group": float(np.mean([r["tv_group_correction"] for r in diffs])) if diffs else None,
    }
    with open(os.path.join(OUT, "metrics", "headline.json"), "w", encoding="utf-8") as fh:
        json.dump(head, fh, indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
