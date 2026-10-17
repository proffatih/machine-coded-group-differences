#!/usr/bin/env python3
"""Placebo check of the tests' false-positive rate.

Items are split into two groups at random, so invariance holds by construction and a
calibrated 5% test should reject about 5% of the time. Run on every shard (one corpus,
750 test items) and on every pooled coder-condition sample, for the invariance test
with its asymptotic chi-square reference and with the within-topic permutation
reference used in the paper, and for both DIF tests.

Output: results/metrics/placebo_calibration.json
"""
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("inv", os.path.join(HERE, "03_invariance.py"))
inv = importlib.util.module_from_spec(spec); spec.loader.exec_module(inv)
REPS_SHARD, REPS_POOLED, N_PERM = 3, 5, 499


def one(y, pred, rng):
    g = rng.permutation(np.r_[np.zeros(len(y) // 2), np.ones(len(y) - len(y) // 2)]).astype(str)
    lad = inv.invariance_ladder(y, pred, g, n_perm=N_PERM, rng=rng)
    dif = inv.dif_lr_tests((y == pred).astype(float), inv.CODES[y], g)
    return lad["p_asymptotic"], lad["p"], dif["uniform_p"], dif["nonuniform_p"]


def main():
    rng = np.random.default_rng(20260905)
    recs = inv.load_all()
    out = {"n_perm": N_PERM}
    shard = []
    for r in recs:
        te = r["split"] == "test"
        for _ in range(REPS_SHARD):
            shard.append(one(r["y"][te], r["pred"][te], rng))
    pooled = []
    for m, c in sorted({(r["model"], r["condition"]) for r in recs}):
        d = inv.pool(recs, m, c); te = d["split"] == "test"
        for _ in range(REPS_POOLED):
            pooled.append(one(d["y"][te], d["pred"][te], rng))
    for name, res in (("single_corpus", shard), ("pooled_corpora", pooled)):
        a = np.array(res) < .05
        out[name] = {"n_splits": len(res),
                     "reject_asymptotic_chi2": float(a[:, 0].mean()),
                     "reject_permutation": float(a[:, 1].mean()),
                     "reject_dif_uniform": float(a[:, 2].mean()),
                     "reject_dif_nonuniform": float(a[:, 3].mean())}
    json.dump(out, open(os.path.join(inv.OUT, "metrics", "placebo_calibration.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
