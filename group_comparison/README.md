# Group comparisons with machine-coded text

Estimators of a between-group difference in topic prevalence from machine codes: no
correction, matrix back-calculation (pooled and per group), a per-category (Rogan–Gladen)
correction, calibration (post-stratification on the machine code), the difference
(prediction-powered) estimator and its tuned version, and hand coding alone.

```
python code/s0_transfer_classifier.py  # classifier trained outside the coded corpus (and outside each pair of corpora)
python code/s1_simulation.py           # Study 1: 150 conditions x 500 replications, stratified and pooled validation samples
python code/s2_real_coders.py          # Study 2: 21 coder-condition combinations, 294 within- and 441 between-corpus comparisons
python code/s2_real_coders.py --oracle-only   # population quantities only (true confusion matrices, error decomposition, random splits)
python code/s3_illustration.py         # worked agenda comparison and validation-size planning
python code/figures.py; python code/claims.py; python code/tables.py
```
Study 2 reads the coder outputs in `../results/annotations/` and the corpora in `../data/`. The
per-replication simulation file is written outside the repository (`~/.cache/run_out/`);
`results/sim_summary.csv` holds the condition means.
