#!/usr/bin/env python3
"""Supervised baselines, scored exactly like the language models.

The study's claim is about an instrument, and the obvious question an editor asks is
whether the instrument has to be a language model for the claim to hold. If a plain
supervised classifier trained on the researcher's own labels is invariant, the problem
is specific to LLM annotation. If it is not, the problem belongs to machine-coded text
in general, which is a larger and more useful finding -- but a different paper from the
one that only tested language models.

Every prediction is out-of-fold. Each corpus is split into five label-stratified folds;
for each fold a classifier is trained on the other four and predicts the held-out fold.
Every item is therefore predicted by a model that never saw it, and calibration-half and
test-half predictions come from the same procedure. That matters for the correction
analysis: a confusion matrix estimated on calibration-half predictions is then applied to
test-half predictions, so both must come from the same instrument. (An earlier
construction fitted one classifier per half and estimated the confusion matrix from the
other one's predictions; that was a different instrument from the one being corrected.)

Two learners:

  tfidf-logreg   TF-IDF word and character n-grams into multinomial logistic regression.
                 The standard supervised comparison in this literature.
  tfidf-svm      The same features into a calibrated linear SVM, as a check that the
                 result is not an artefact of one learner.

Output is written in the same .npz shard format as the language-model runs so that
the analyses in group_comparison/ read both without modification.

Usage: python code/06_supervised_baseline.py
"""
import csv, json, os, sys, time

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES, SEED

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTD = os.path.join(ROOT, "results", "annotations")
os.makedirs(OUTD, exist_ok=True)
CODES = np.array(VALID_CODES)
K_FOLDS = 5


def features():
    """Word and character n-grams. Character n-grams matter because several corpora are
    very short -- newspaper headlines average six words -- where word features alone are
    sparse."""
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, strip_accents="unicode"),
    )


def _logreg(y):
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")


def _svm(y):
    """Calibrated linear SVM. The number of folds has to adapt: stratified corpora leave
    some CAP topics with a single training example, and CalibratedClassifierCV cannot
    cross-validate a class it cannot split."""
    import collections
    counts = collections.Counter(y.tolist())
    folds = max(2, min(3, min(counts.values())))
    return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"),
                                  method="sigmoid", cv=folds)


LEARNERS = {"tfidf-logreg": _logreg, "tfidf-svm": _svm}


def main():
    for mkey, make_clf in LEARNERS.items():
        for ckey in CORPORA:
            out = os.path.join(OUTD, f"{mkey}__sup__{ckey}.npz")
            if os.path.exists(out):
                print(f"[{mkey}|{ckey}] hazir, atlaniyor", flush=True); continue
            with open(os.path.join(DATA, f"{ckey}.csv"), encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            text = np.array([r["text"] for r in rows])
            gold = np.array([int(r["label"]) for r in rows], dtype=np.int16)
            split = np.array([r["split"] for r in rows])
            ca, te = split == "calib", split == "test"

            t0 = time.time()
            # a topic with a single labelled example cannot be learned or cross-validated;
            # it is dropped from training and simply never predicted
            def _trainable(mask):
                import collections
                cnt = collections.Counter(gold[mask].tolist())
                keep = {c for c, n in cnt.items() if n >= 2}
                return mask & np.isin(gold, list(keep))

            from sklearn.model_selection import StratifiedKFold
            import warnings
            probs = np.zeros((len(rows), len(CODES)), dtype=np.float32)
            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # rare classes < K_FOLDS members
                for tr_idx, te_idx in skf.split(text, gold):
                    tr = np.zeros(len(rows), bool); tr[tr_idx] = True
                    tr = _trainable(tr)
                    pipe = make_pipeline(features(), make_clf(gold[tr]))
                    pipe.fit(text[tr], gold[tr])
                    p = pipe.predict_proba(text[te_idx])
                    col = {c: j for j, c in enumerate(pipe.classes_)}
                    for k, code in enumerate(CODES):
                        if code in col:
                            probs[te_idx, k] = p[:, col[code]]
            np.savez_compressed(
                out, probs=probs, gold=gold, split=split,
                ids=np.array([r["id"] for r in rows]),
                year=np.array([int(r["year"]) for r in rows], dtype=np.int32),
                n_words=np.array([int(r["n_words"]) for r in rows], dtype=np.int32),
                codes=CODES.astype(np.int16),
                meta=json.dumps({"model": mkey, "repo": "scikit-learn", "precision": "fp64",
                                 "condition": "sup", "corpus": ckey, "k_shot": 0,
                                 "response_format": "supervised probability",
                                 "trained_on": f"{K_FOLDS}-fold cross-fitting, out-of-fold predictions",
                                 "seed": SEED, "n": len(rows),
                                 "elapsed_s": round(time.time() - t0, 1)}))
            acc = float((CODES[probs[te].argmax(1)] == gold[te]).mean())
            print(f"  -> {mkey}|{ckey}: acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
