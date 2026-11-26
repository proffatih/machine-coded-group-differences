# Estimating group differences in topic prevalence from machine-coded text

Replication material for

> Gül, Fatih. "Estimating Group Differences in Topic Prevalence From Machine-Coded Text:
> Back-Calculation, Calibration, and Difference Estimators." Manuscript under review.

The article asks how a coder's errors (a supervised classifier or a large language model)
carry into a comparison of topic prevalence between two groups of texts, and which
estimators recover the difference from a hand-coded validation sample. It evaluates
confusion-matrix back-calculation, a per-category (Rogan–Gladen) correction, and
calibration and difference estimators in simulations and with 30 real coder–condition
combinations on seven Comparative Agendas Project corpora.

## Layout

```
code/                     building the corpora and coding them
  cap_codebook.py           the 21 CAP major topics and the corpus definitions
  01_prepare_corpora.py     the seven analysis corpora from the public CAP files
  02_annotate.py            the language models (prompts and scoring are in the script)
  06_supervised_baseline.py the in-corpus TF-IDF classifiers (5-fold cross-fitting)
data/                     the derived corpora (1,500 human-coded items each)
results/annotations/      one .npz per coder x condition x corpus (scores for the 21 topics)
run_large_models.sh       the 7-8B runs on an 8 GB GPU

group_comparison/         the analyses reported in the article
  code/estimators.py        every estimator and its standard error (Online Appendix A)
  code/s0_transfer_classifier.py  classifier trained outside the corpus (or the pair of corpora)
  code/s1_simulation.py     Study 1
  code/s2_real_coders.py    Study 2
  code/s3_illustration.py   the worked example and the planning tables
  code/s4_validity.py       measurement quality within each group (Section 5.3)
  code/s5_label_noise.py    error in the hand codes (Section 5.5)
  code/s6_schemes.py        other coding schemes (Section 6; --sim, --real)
  code/s7_extensions.py     regularised back-calculation, simulation with the real matrices,
                            intervals by prevalence and joint coverage (Section 5.6)
  code/s8_population.py     the full CAP corpora as the population (needs data_raw/, below)
  code/s9_coders.py         prior-corrected language-model codes and synthetic strong coders
  code/*_summary.py, claims.py   the numbers quoted in the text
  code/tables.py, tables_v3.py, figures.py   the tables and figures
  code/verify_claims.py     checks every number in the article against the results
                            (needs the article source, which is not in this repository)
  results/                  every result file the article reports
  figures/                  Figures 1-4 and A1
```

## Reproducing

Run from `group_comparison/`, in this order (times on a 16-core CPU):

```
python code/s0_transfer_classifier.py            # ~6 min
python code/s1_simulation.py                     # ~5 min; per-replication file in ~/.cache/run_out/
python code/s2_real_coders.py                    # ~10 min
python code/s3_illustration.py
python code/s4_validity.py                       # ~1 min
python code/s5_label_noise.py                    # ~15 min
python code/s6_schemes.py --sim && python code/s6_schemes.py --real   # ~10 min, 12 processes
python code/s7_extensions.py                     # ~11 min, 14 processes
python code/s8_population.py                     # ~1 min, needs data_raw/
python code/s9_coders.py                         # <1 min
python code/s4_summary.py && python code/s56_summary.py && python code/s789_summary.py
python code/claims.py && python code/tables.py && python code/tables_v3.py && python code/figures.py
```

Every script fixes its random seeds (corpora 20260905; analyses 20261102 and offsets).
The coder outputs in `results/annotations/` and `group_comparison/results/annotations_*`
let every analysis run without a GPU. To recode the corpora, run `code/02_annotate.py`
(1-4B models on a 4 GB GPU) and `run_large_models.sh` (7-8B models on an 8 GB GPU).

`s8_population.py` needs the original CAP downloads in `data_raw/` (file names in
`code/cap_codebook.py`); they are public at the Comparative Agendas Project and are not
redistributed here.

## Software

Analyses: Python 3.13.5, numpy 2.2.4, scipy 1.15.3, scikit-learn 1.9.0, pandas 2.2.3,
matplotlib 3.10.1. Language-model coding: torch 2.6.0 (CUDA 12.4), transformers 5.16.1,
bitsandbytes 0.50.2, accelerate 1.14.0; models in 4-bit NF4 from the Hugging Face
repositories named in `code/02_annotate.py`. The 7-8B models were run with the same
script on a second machine. See `requirements.txt`.

## Data

Seven corpora from the Comparative Agendas Project (https://www.comparativeagendas.net),
each restricted to the 21 shared major topics, deduplicated, and subsampled to 1,500
items: U.S. Congressional bill titles, New York Times front-page headlines, U.S. executive
orders, U.S. Supreme Court case summaries, UK Acts of Parliament, UK Prime Minister's
Questions, and UK newspaper front pages. Please cite the Comparative Agendas Project when
using the derived corpora.

## License

Code: MIT (see `LICENSE`). Derived corpora: subject to the terms of the Comparative
Agendas Project.
