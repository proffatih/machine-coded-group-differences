# Group comparisons with machine-coded text

Estimators of a between-group difference in topic prevalence from machine codes:
no correction, matrix back-calculation (pooled and per group), the difference
(prediction-powered) estimator and its tuned version, and hand coding alone.

```
python code/s1_simulation.py      # Study 1: simulation with known truth (120 conditions x 500 replications)
python code/s2_real_coders.py     # Study 2: 20 real coders, 280 within-corpus and 420 between-corpus comparisons
python code/s3_illustration.py    # worked agenda comparison and validation-size planning
python code/claims.py             # every number stated in the paper -> results/claims.json
python code/tables.py; python code/figures.py
```
Study 2 reads the coder outputs in `../results/annotations/`. The per-replication simulation file is
written outside the repository (`~/.cache/run_out/`); `results/sim_summary.csv` holds the condition means.
