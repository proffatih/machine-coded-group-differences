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
  2. Logistic DIF: correct ~ topic + group (uniform) and + topic x group (non-uniform),
     tested by likelihood-ratio against the topic-only model (Swaminathan & Rogers)
  3. An invariance ladder over the confusion matrix -- configural / metric (equal
     per-topic recall) / scalar (equal recall and precision) -- as nested multinomial
     models compared by LR test
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

    # M1 by one-dimensional search over the group log-odds offset (2 groups only)
    gvals = np.unique(group)
    if len(gvals) == 2:
        ref = gvals[0]
        tvals = np.unique(topic)

        def nll(beta):
            tot = 0.0
            for t in tvals:
                mt = topic == t
                # profile the topic intercept out numerically
                a_lo, a_hi = -12.0, 12.0
                for _ in range(60):
                    a = (a_lo + a_hi) / 2
                    eta = a + beta * (group[mt] != ref)
                    p = 1 / (1 + np.exp(-eta))
                    if (correct[mt] - p).sum() > 0:
                        a_lo = a
                    else:
                        a_hi = a
                a = (a_lo + a_hi) / 2
                eta = a + beta * (group[mt] != ref)
                p = np.clip(1 / (1 + np.exp(-eta)), EPS, 1 - EPS)
                tot += (correct[mt] * np.log(p) + (1 - correct[mt]) * np.log(1 - p)).sum()
            return -tot

        b_lo, b_hi = -6.0, 6.0
        for _ in range(80):
            m1, m2 = b_lo + (b_hi - b_lo) / 3, b_hi - (b_hi - b_lo) / 3
            if nll(m1) < nll(m2):
                b_hi = m2
            else:
                b_lo = m1
        beta = (b_lo + b_hi) / 2
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


def invariance_ladder(y, pred, group):
    """Nested tests: pooled confusion (scalar) vs per-group confusion (configural)."""
    gvals = np.unique(group)
    pooled = confusion(y, pred)
    ll_pooled = 0.0
    for g in gvals:
        m = group == g
        cg = confusion(y[m], pred[m])
        # scalar model: each group's counts scored under the POOLED row profiles
        for i in range(K):
            n = pooled[i].sum()
            if n == 0 or cg[i].sum() == 0:
                continue
            p = np.clip(pooled[i] / n, EPS, 1)
            ll_pooled += float((cg[i] * np.log(p)).sum())
    ll_free = sum(multinom_loglik(confusion(y[group == g], pred[group == g]))
                  for g in gvals)
    # df counts only cells the data can actually inform: for each true topic that
    # occurs, the number of distinct predicted labels it ever receives, less one for
    # the simplex constraint, times the number of extra groups. Counting all K*K
    # cells would inflate df by the (many) structurally empty ones and make the test
    # anticonservative in the wrong direction -- it would never reject.
    df = 0
    for i in range(K):
        if pooled[i].sum() == 0:
            continue
        used = int((pooled[i] > 0).sum())
        df += max(used - 1, 0)
    df = max(df * (len(gvals) - 1), 1)
    stat = max(2 * (ll_free - ll_pooled), 0.0)
    return {"n_groups": int(len(gvals)), "lr_chi2": float(stat), "df": int(df),
            "p": float(stats.chi2.sf(stat, df)),
            "scaled_chi2_df": float(stat / df)}


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
    args = ap.parse_args()

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
