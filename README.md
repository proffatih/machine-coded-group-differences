# Measurement invariance in LLM text annotation

Replication package for a study that treats a large language model annotator as a
**measurement instrument** and asks whether it measures the same construct the same
way across subgroups of text — country, institutional venue, text length, period.

Submitted to *Social Science Computer Review*. This repository will be finalised at
submission; the DOI and the article reference will be added here.

## What is here

```
code/     data preparation, annotation runner, invariance analysis, figures, tables
data/     the derived analysis corpora (1,500 items each, calibration/test split)
results/  model outputs (.npz), analysis tables (.csv), headline metrics
figures/  manuscript figures, greyscale-legible
```

## Data

Seven corpora from the **Comparative Agendas Project**, all coded by trained human
coders against the same CAP master codebook of 21 major policy topics:

| Corpus | Country | Venue |
|---|---|---|
| U.S. Congressional bill titles | US | Legislative |
| New York Times front-page headlines | US | Media |
| U.S. executive orders | US | Executive |
| U.S. Supreme Court case summaries | US | Judicial |
| UK Acts of Parliament long titles | UK | Legislative |
| UK Prime Minister's Questions | UK | Legislative (oral) |
| UK newspaper front-page headlines | UK | Media |

A single codebook applied across institutions and countries is what makes the design
work: domain shift varies while the coding scheme is held fixed.

The raw CAP files are **not redistributed here**. `code/01_prepare_corpora.py` downloads
them from the project's own servers and rebuilds the derived corpora deterministically
(`SEED = 20260905`). Source: <https://www.comparativeagendas.net/datasets_codebooks>.

## Models

Six open-weight instruction-tuned models from five independent developers, all loaded
in 4-bit NF4 so that quantisation is constant across the roster:
OLMo-2 1B · Qwen2.5 1.5B · SmolLM2 1.7B · Granite 3.1 2B · Qwen2.5 3B · Phi-3.5-mini.

Gated releases were deliberately excluded so that anyone can reproduce the run without
credentials, and closed API models were excluded because they are silently versioned
and cannot be re-run.

Each item is scored by restricting the logits at the answer position to the 21
single-token option letters. Every item therefore receives a response — no refusals and
no unparsable output, which matters because that attrition would itself differ across
subgroups and contaminate the invariance test.

## Reproducing

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
python code/01_prepare_corpora.py      # downloads CAP, rebuilds data/
bash   code/run_all.sh                 # 6 models x 3 conditions x 7 corpora, resumable
python code/06_supervised_baseline.py # TF-IDF coders, 5-fold cross-fitting -> results/annotations/
python code/03_invariance.py           # DIF, invariance ladder, corrections -> results/
python code/07_mechanism.py            # derivation check -> results/mechanism.csv
python code/08_uncertainty.py --boot 3000  # cluster bootstrap -> results/metrics/intervals.json
python code/09_pilot_response_format.py    # letters vs names pilot -> results/pilot/
python code/10_within_corpus.py        # invariance inside each corpus (same prompt, same classifier)
python code/11_pooled_supervised.py    # one supervised classifier for all corpora, then:
python code/03_invariance.py   --ann_dir results/annotations_pooled_sup --out_dir results/pooled_supervised
python code/10_within_corpus.py --ann_dir results/annotations_pooled_sup --out_dir results/pooled_supervised
python code/12_placebo_calibration.py  # false-positive rate of every test under random splits
python code/04_figures.py              # figures/
python code/05_tables.py               # manuscript tables
python code/verify_claims.py           # recomputes every number stated in the paper
```

`code/test_invariance_smoke.py` injects a known invariance violation into synthetic
annotations and checks that the analysis recovers it. Run it before trusting a change
to the analysis code.

## Licence

Code: MIT (see `LICENSE`). The CAP data are distributed by the Comparative Agendas
Project under its own terms and are not included here.
