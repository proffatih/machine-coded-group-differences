#!/usr/bin/env python3
"""A supervised coder that has seen no labels from the corpus it codes.

The two supervised classifiers of the annotation study were cross-fitted within each
corpus, i.e. trained on about 1,200 of that corpus's own human codes. A researcher who
had those labels would not need a validation sample, so for the question here -- how
much a coder lowers the hand coding a valid comparison needs -- a classifier trained
elsewhere is the realistic case. Each corpus is coded by a TF-IDF logistic regression
trained on the other six corpora (the transfer setting of Burscher et al., 2015). For a
comparison between two corpora, both are coded by a model trained on the remaining five,
so that the classifier has seen no label of either group.

Output: results/annotations_transfer/tfidf-logreg-transfer__sup__<corpus>.npz and
results/annotations_transfer_pairs/<corpus A>__<corpus B>__<corpus>.npz, in the shard
format of the annotation study.
"""
import csv, itertools, json, os, sys, time, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the annotation study: a sibling folder in the working tree, the parent folder in the public repository
SSCR = os.path.join(os.path.dirname(ROOT), "Social_Science_Computer_Review")
if not os.path.isdir(SSCR):
    SSCR = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(SSCR, "code"))
from cap_codebook import CORPORA, VALID_CODES, SEED
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

OUTD = os.path.join(ROOT, "results", "annotations_transfer")
PAIRD = os.path.join(ROOT, "results", "annotations_transfer_pairs")
os.makedirs(OUTD, exist_ok=True); os.makedirs(PAIRD, exist_ok=True)
CODES = np.array(VALID_CODES)


def features():
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, strip_accents="unicode"))


def read(c):
    with open(os.path.join(SSCR, "data", f"{c}.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fit(rows):
    pipe = make_pipeline(features(), LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit([r["text"] for r in rows], [int(r["label"]) for r in rows])
    return pipe


def code(pipe, rows, path, meta):
    p = pipe.predict_proba([r["text"] for r in rows])
    probs = np.zeros((len(rows), len(CODES)), dtype=np.float32)
    col = {k: j for j, k in enumerate(pipe.classes_)}
    for k, c in enumerate(CODES):
        if c in col:
            probs[:, k] = p[:, col[c]]
    gold = np.array([int(r["label"]) for r in rows], dtype=np.int16)
    np.savez_compressed(
        path, probs=probs, gold=gold, split=np.array([r["split"] for r in rows]),
        ids=np.array([r["id"] for r in rows]),
        year=np.array([int(r["year"]) for r in rows], dtype=np.int32),
        n_words=np.array([int(r["n_words"]) for r in rows], dtype=np.int32),
        codes=CODES.astype(np.int16), meta=json.dumps(meta))
    return float((CODES[probs.argmax(1)] == gold).mean())


def main():
    data = {c: read(c) for c in CORPORA}
    for c in CORPORA:
        t0 = time.time()
        train = [r for o in CORPORA if o != c for r in data[o]]
        acc = code(fit(train), data[c], os.path.join(OUTD, f"tfidf-logreg-transfer__sup__{c}.npz"),
                   {"model": "tfidf-logreg-transfer", "condition": "sup", "corpus": c,
                    "trained_on": "the six other corpora", "n_train": len(train), "seed": SEED, "n": len(data[c])})
        print(f"{c}: acc={acc:.3f} n_train={len(train)} ({time.time() - t0:.0f}s)", flush=True)
    for a, b in itertools.combinations(sorted(CORPORA), 2):
        t0 = time.time()
        train = [r for o in CORPORA if o not in (a, b) for r in data[o]]
        pipe = fit(train)
        for c in (a, b):
            code(pipe, data[c], os.path.join(PAIRD, f"{a}__{b}__{c}.npz"),
                 {"model": "tfidf-logreg-transfer", "condition": "sup", "corpus": c, "pair": [a, b],
                  "trained_on": "the five corpora outside the pair", "n_train": len(train), "seed": SEED, "n": len(data[c])})
        print(f"{a}|{b} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
