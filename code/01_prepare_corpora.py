#!/usr/bin/env python3
"""Build the seven analysis corpora from the raw CAP downloads.

Each corpus is restricted to the 21 canonical CAP major topics, deduplicated, and
stratified-subsampled to N_PER_CORPUS items, then split into disjoint calibration
and test halves. Everything is driven by a fixed SEED so the split is reproducible.

Output: data/<key>.csv with columns id, text, label, split
        data/prep_report.json  (drop accounting, class balance)
"""
import csv, json, os, sys, hashlib, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES, CAP_MAJOR_TOPICS, SEED

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(ROOT, "data")
N_PER_CORPUS = 1500
MIN_WORDS = 3
MAX_CHARS = 1800          # keeps SCOTUS summaries inside a sane prompt budget

os.makedirs(OUT, exist_ok=True)
VALID = set(VALID_CODES)
report = {"seed": SEED, "n_per_corpus": N_PER_CORPUS, "corpora": {}}


def clean(text):
    return " ".join(text.split())


for key, (fname, tcol, lcol, country, venue, pretty) in CORPORA.items():
    path = os.path.join(RAW, fname)
    kept, drop = [], collections.Counter()
    seen = set()

    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_t, raw_l = (row.get(tcol) or ""), (row.get(lcol) or "").strip()
            text = clean(raw_t)
            if not text:
                drop["empty_text"] += 1; continue
            try:
                code = int(float(raw_l))
            except ValueError:
                drop["unparsable_label"] += 1; continue
            if code not in VALID:
                drop[f"off_scheme_{code}"] += 1; continue
            if len(text.split()) < MIN_WORDS:
                drop["too_short"] += 1; continue
            h = hashlib.md5(text.lower().encode()).hexdigest()
            if h in seen:
                drop["duplicate"] += 1; continue
            seen.add(h)
            try:
                year = int(float(row.get("year") or 0))
            except ValueError:
                year = 0
            kept.append((text[:MAX_CHARS], code, year))

    # stratified subsample: proportional allocation, at least 1 per present class
    by_label = collections.defaultdict(list)
    for t, c, y in kept:
        by_label[c].append((t, y))
    rng = random.Random(SEED)
    for c in by_label:
        rng.shuffle(by_label[c])

    total = len(kept)
    target = min(N_PER_CORPUS, total)
    alloc, remainders = {}, []
    for c, items in by_label.items():
        exact = len(items) * target / total
        alloc[c] = min(len(items), max(1, int(exact)))
        remainders.append((exact - int(exact), c))
    # trim/grow to hit target exactly
    while sum(alloc.values()) > target:
        c = max((c for c in alloc if alloc[c] > 1), key=lambda c: alloc[c])
        alloc[c] -= 1
    for _, c in sorted(remainders, reverse=True):
        if sum(alloc.values()) >= target:
            break
        if alloc[c] < len(by_label[c]):
            alloc[c] += 1

    sample = [(t, c, y) for c, n in alloc.items() for t, y in by_label[c][:n]]
    rng.shuffle(sample)

    # disjoint calibration / test halves, stratified by construction via interleave
    ordered = sorted(sample, key=lambda x: (x[1], hashlib.md5(x[0].encode()).hexdigest()))
    rows = []
    for i, (t, c, y) in enumerate(ordered):
        rows.append({"id": f"{key}_{i:05d}", "text": t, "label": c, "year": y,
                     "n_words": len(t.split()),
                     "split": "calib" if i % 2 == 0 else "test"})
    rng.shuffle(rows)

    with open(os.path.join(OUT, f"{key}.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "label", "year", "n_words", "split"])
        w.writeheader(); w.writerows(rows)

    dist = collections.Counter(r["label"] for r in rows)
    report["corpora"][key] = {
        "pretty": pretty, "country": country, "venue": venue,
        "n_raw_kept": total, "n_sampled": len(rows),
        "n_calib": sum(1 for r in rows if r["split"] == "calib"),
        "n_test": sum(1 for r in rows if r["split"] == "test"),
        "n_classes_present": len(dist),
        "mean_words": round(sum(len(r["text"].split()) for r in rows) / len(rows), 1),
        "class_counts": {str(k): v for k, v in sorted(dist.items())},
        "drops": dict(drop.most_common()),
        "off_scheme_total": sum(v for k, v in drop.items() if k.startswith("off_scheme")),
    }
    print(f"{key:10s} kept={total:>7,} -> sampled={len(rows):>5,} "
          f"classes={len(dist):>2} mean_words={report['corpora'][key]['mean_words']:>6.1f} "
          f"off-scheme dropped={report['corpora'][key]['off_scheme_total']:>7,}")

with open(os.path.join(OUT, "prep_report.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print("\nyazildi ->", os.path.join(OUT, "prep_report.json"))
