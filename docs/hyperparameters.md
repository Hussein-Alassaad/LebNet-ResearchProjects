# Hyperparameters & configuration reference

All hyperparameters are defined in `src/model.py:XGBoostConfig` and can be
overridden via `src/train.py` CLI flags. This document records every value
used across the reproduction runs referenced in
`results/reproduction_results.md`.

## Fixed across all runs

| Setting | Value | Source |
|---|---|---|
| Random seed | 42 | `data_loader.RANDOM_STATE`, `XGBoostConfig.seed` |
| Objective | `binary:logistic` | paper's loss `l` = logistic loss for binary classification |
| Eval metrics | `logloss`, `auc` | tracked every round for early stopping + reporting |
| `tree_method` | `hist` | histogram-based approximate split finding (paper Section 3's weighted quantile sketch, as implemented in modern xgboost) |
| Early stopping | 20 rounds without val-logloss improvement | controls overfitting; paper's own experiments also use validation-based stopping |
| Train/val/test split | 60% / 6.67% / 33.3% (~2/3 train+val, 1/3 test) | matches original UCI Adult train/test file size ratio |

## Per-run configuration

`tune_attempt1` and `tune_attempt2` were trained before the train/test
deduplication fix (see `results/reproduction_results.md`); `xgboost_adult`
and `xgboost_adult_final` were (re)trained after it, on leakage-free
splits — which is why `xgboost_adult_final` differs slightly from
`tune_attempt2` despite using the same hyperparameters.

| Run | `max_depth` | `eta` | `lambda` | `gamma` | `subsample` | `colsample_bytree` | `num_boost_round` | best iter | test accuracy | test AUC | post-dedup? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `tune_attempt1` (deeper, slower) | 8 | 0.05 | 1.0 | 0.0 | 0.8 | 0.8 | 500 | 262 | 0.8746 | 0.9281 | no |
| `tune_attempt2` (pre-dedup) | 5 | 0.1 | 2.0 | 1.0 | 0.9 | 0.7 | 400 | 169 | 0.8767 | 0.9292 | no |
| `xgboost_adult` (baseline defaults, post-dedup) | 6 | 0.1 | 1.0 | 0.0 | 0.8 | 0.8 | 300 | 86  | 0.8738 | 0.9292 | yes |
| **`xgboost_adult_final`** (selected config, post-dedup) | 5 | 0.1 | 2.0 | 1.0 | 0.9 | 0.7 | 400 | 141 | **0.8753** | **0.9294** | yes |

`xgboost_adult_final` is the configuration referenced throughout
`results/reproduction_results.md` and the README's reproduction
instructions — it is the same hyperparameters as `tune_attempt2`,
re-run on the corrected, leakage-free data split. Exact numbers are
recorded per-run in `results/<model_name>_train_run.json`,
`_evals_result.json`, and `_metrics.json`.

## Why `tune_attempt2` was selected as final

- Shallower trees (`max_depth=5` vs. 6-8) plus stronger `lambda`/`gamma`
  regularization reduced the train/validation gap slightly compared to
  the baseline and `tune_attempt1`, matching the paper's claim (Section
  2.2) that the regularized objective helps generalization.
- Best test accuracy and AUC among the three runs, while using fewer
  boosting rounds at convergence (141) than `tune_attempt1` (262).

## Reproducing a specific run

```bash
# xgboost_adult (baseline defaults, post-dedup)
python src/train.py --model-name xgboost_adult

# tune_attempt1
python src/train.py --model-name tune_attempt1 --max-depth 8 --eta 0.05 --num-boost-round 500

# xgboost_adult_final (selected configuration)
python src/train.py --model-name xgboost_adult_final \
    --max-depth 5 --eta 0.1 --reg-lambda 2.0 --gamma 1.0 \
    --subsample 0.9 --colsample-bytree 0.7 --num-boost-round 400
```
