#!/usr/bin/env python3
"""Smoke test: inject a known measurement-invariance violation and check we find it.

Synthetic annotations are built from the real corpora so the label structure and the
subgroup sizes are the ones the study will actually face. UK texts whose true topic
is Defense (16) or International Affairs (19) are given a much lower probability of
being coded correctly, with the mass moved to the other member of that pair -- a
textbook non-uniform DIF pattern confined to one country and two topics.

The analysis must (a) reject invariance, (b) locate a large per-topic recall gap,
and (c) show that a pooled confusion correction fails on between-group differences
where a subgroup-specific one succeeds. If any of those stop holding, the analysis
code has regressed.

Usage: python code/test_invariance_smoke.py
"""
import csv, json, os, subprocess, sys, tempfile, shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES = np.array(VALID_CODES)
K = len(CODES)
IDX = {c: i for i, c in enumerate(VALID_CODES)}
P_OK = 0.75
P_OK_DIF = 0.30
DIF_TOPICS = (16, 19)


def build(out_dir, seed=7):
    rng = np.random.default_rng(seed)
    for ckey, (_f, _tc, _lc, country, _venue, _pretty) in CORPORA.items():
        path = os.path.join(ROOT, "data", f"{ckey}.csv")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        gold = np.array([int(r["label"]) for r in rows])
        y = np.searchsorted(CODES, gold)
        probs = np.full((len(rows), K), 0.01, dtype=np.float32)
        for i, t in enumerate(y):
            code = int(CODES[t])
            p_ok = P_OK
            if code == 16:
                rival = IDX[19]
            elif code == 19:
                rival = IDX[16]
            else:
                rival = int(rng.integers(K))
            if country == "UK" and code in DIF_TOPICS:
                p_ok = P_OK_DIF
            probs[i][t] += p_ok
            probs[i][rival] += (1.0 - p_ok) * 0.9
            probs[i] /= probs[i].sum()
        np.savez_compressed(
            os.path.join(out_dir, f"smoke__zs__{ckey}.npz"),
            probs=probs, gold=gold.astype(np.int16),
            split=np.array([r["split"] for r in rows]),
            ids=np.array([r["id"] for r in rows]),
            year=np.array([int(r["year"]) for r in rows], dtype=np.int32),
            n_words=np.array([int(r["n_words"]) for r in rows], dtype=np.int32),
            codes=CODES.astype(np.int16),
            meta=json.dumps({"model": "smoke", "repo": "synthetic", "condition": "zs",
                             "corpus": ckey, "k_shot": 0, "seed": seed,
                             "n": len(rows), "elapsed_s": 0}))


def main():
    real = os.path.join(ROOT, "results", "annotations")
    stash = os.path.join(ROOT, "results", "annotations.realstash")
    tmp = tempfile.mkdtemp(prefix="inv_smoke_")
    build(tmp)
    moved = False
    try:
        if os.path.isdir(real) and os.listdir(real):
            shutil.move(real, stash); moved = True
        elif os.path.isdir(real):
            os.rmdir(real)
        shutil.copytree(tmp, real)
        subprocess.run([sys.executable, os.path.join(ROOT, "code", "03_invariance.py")],
                       check=True, stdout=subprocess.DEVNULL)
        head = json.load(open(os.path.join(ROOT, "results", "metrics", "headline.json")))
    finally:
        shutil.rmtree(real, ignore_errors=True)
        if moved:
            shutil.move(stash, real)
        else:
            os.makedirs(real, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)

    checks = [
        ("degismezlik reddedildi", head["invariance_rejected_share"] == 1.0),
        ("tekduze DIF yakalandi", head["dif_uniform_share_p05"] > 0.5),
        ("buyuk konu-duyarlilik farki bulundu", head["mean_worst_topic_recall_gap"] > 0.4),
        ("havuzlanmis duzeltme FARKLARDA basarisiz",
         head["difference_tv_pooled"] > head["difference_tv_naive"]),
        ("gruba ozgu duzeltme havuzlanmistan iyi",
         head["difference_tv_group"] < head["difference_tv_pooled"]),
    ]
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    print(json.dumps(head, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
