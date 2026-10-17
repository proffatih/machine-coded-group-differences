#!/usr/bin/env python3
"""Robustness: one supervised classifier for all seven corpora.

The main supervised coders are trained within each corpus, so a country or venue
comparison for them compares separately trained classifiers. Here each learner is
trained once on the pooled corpora by five-fold cross-fitting (folds stratified on
topic and corpus), so every subgroup is coded by the same fitted procedure, without a
corpus label among its inputs. The shards are written to a separate directory and
analysed with the unchanged 03_invariance.py and 10_within_corpus.py:

  python code/11_pooled_supervised.py
  python code/03_invariance.py   --ann_dir results/annotations_pooled_sup --out_dir results/pooled_supervised
  python code/10_within_corpus.py --ann_dir results/annotations_pooled_sup --out_dir results/pooled_supervised
"""
import csv, importlib.util, json, os, sys, time, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cap_codebook import CORPORA, SEED
spec = importlib.util.spec_from_file_location("sup", os.path.join(HERE, "06_supervised_baseline.py"))
sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)

OUTD = os.path.join(sup.ROOT, "results", "annotations_pooled_sup")
os.makedirs(OUTD, exist_ok=True)
CODES = sup.CODES


def main():
    rows, corpus = [], []
    for ckey in CORPORA:
        with open(os.path.join(sup.DATA, f"{ckey}.csv"), encoding="utf-8") as fh:
            r = list(csv.DictReader(fh))
        rows += r; corpus += [ckey] * len(r)
    corpus = np.array(corpus)
    text = np.array([r["text"] for r in rows])
    gold = np.array([int(r["label"]) for r in rows], dtype=np.int16)
    split = np.array([r["split"] for r in rows])
    strat = np.array([f"{c}|{g}" for c, g in zip(corpus, gold)])
    # strata with fewer members than folds fall back to the topic alone
    u, cnt = np.unique(strat, return_counts=True)
    small = set(u[cnt < sup.K_FOLDS])
    strat = np.array([str(g) if s in small else s for s, g in zip(strat, gold)])

    for mkey, make_clf in sup.LEARNERS.items():
        name = f"{mkey}-pooled"
        if all(os.path.exists(os.path.join(OUTD, f"{name}__sup__{c}.npz")) for c in CORPORA):
            print(f"[{name}] hazir, atlaniyor"); continue
        t0 = time.time()
        probs = np.zeros((len(rows), len(CODES)), dtype=np.float32)
        skf = StratifiedKFold(n_splits=sup.K_FOLDS, shuffle=True, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, te_idx in skf.split(text, strat):
                tr = np.zeros(len(rows), bool); tr[tr_idx] = True
                vals, c = np.unique(gold[tr], return_counts=True)
                tr &= np.isin(gold, vals[c >= 2])
                pipe = make_pipeline(sup.features(), make_clf(gold[tr]))
                pipe.fit(text[tr], gold[tr])
                p = pipe.predict_proba(text[te_idx])
                col = {cc: j for j, cc in enumerate(pipe.classes_)}
                for k, code in enumerate(CODES):
                    if code in col:
                        probs[te_idx, k] = p[:, col[code]]
        for ckey in CORPORA:
            m = corpus == ckey
            rr = [rows[i] for i in np.flatnonzero(m)]
            np.savez_compressed(
                os.path.join(OUTD, f"{name}__sup__{ckey}.npz"),
                probs=probs[m], gold=gold[m], split=split[m],
                ids=np.array([r["id"] for r in rr]),
                year=np.array([int(r["year"]) for r in rr], dtype=np.int32),
                n_words=np.array([int(r["n_words"]) for r in rr], dtype=np.int32),
                codes=CODES.astype(np.int16),
                meta=json.dumps({"model": name, "repo": "scikit-learn", "precision": "fp64",
                                 "condition": "sup", "corpus": ckey, "k_shot": 0,
                                 "response_format": "supervised probability",
                                 "trained_on": f"one classifier for all corpora, {sup.K_FOLDS}-fold cross-fitting",
                                 "seed": SEED, "n": int(m.sum())}))
            te = m & (split == "test")
            print(f"  {name}|{ckey}: acc={(CODES[probs[te].argmax(1)] == gold[te]).mean():.3f}")
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
