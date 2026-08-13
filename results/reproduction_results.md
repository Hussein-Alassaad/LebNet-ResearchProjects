# Reproduction Results

## Scope

As noted in [reproduction_comparison_notes.md](reproduction_comparison_notes.md),
the KDD 2016 XGBoost paper benchmarks on Higgs Boson, Yahoo LTRC, and
Allstate — not Adult Census Income. This project reproduces the paper's
**method** (the official `xgboost` package implementing the regularized,
second-order-boosted, sparsity-aware algorithm from the paper) and
applies it faithfully to the standard Adult Census Income benchmark,
comparing against widely-reported reference numbers for gradient
boosted trees on that dataset.

## Final selected configuration

Selected after two rounds of hyperparameter exploration
(`results/tune_attempt1_*`, `results/tune_attempt2_*`) — see
`results/xgboost_adult_final_train_run.json` for the full run record.

| Hyperparameter      | Value |
|----------------------|------:|
| `max_depth`           | 5     |
| `eta`                 | 0.1   |
| `lambda` (L2)         | 2.0   |
| `gamma`               | 1.0   |
| `subsample`           | 0.9   |
| `colsample_bytree`    | 0.7   |
| `num_boost_round`     | 400 (early-stopped at 161, best iteration 141) |
| `tree_method`         | hist  |

## Our reproduced results (`xgboost_adult_final`)

From `results/xgboost_adult_final_metrics.json`:

| Split      | Accuracy | Error rate | AUC    | Log loss | F1     |
|------------|---------:|-----------:|-------:|---------:|-------:|
| Validation | 0.8666   | 0.1334     | 0.9234 | 0.2855   | 0.6909 |
| Test       | **0.8753** | 0.1247   | **0.9294** | 0.2762 | 0.7181 |

## Reference range (gradient boosted trees on Adult Census Income)

Commonly reported test accuracy: **~86-87.5%**, AUC: **~0.92-0.93**.

## Comparison and discrepancy analysis

| Metric   | Our result | Reference range | Verdict            |
|----------|-----------:|-----------------:|---------------------|
| Accuracy | 87.53%     | 86-87.5%          | At/above top of range |
| AUC      | 0.9294     | 0.92-0.93         | Within range         |

No unresolved discrepancy. One genuine issue was found and fixed during
reproduction: the raw Adult dataset contains ~52 exact-duplicate rows
(a byproduct of coarse census bucketing), some of which landed in both
the train and test splits under a naive random split, leaking test
examples into training. This was fixed in `src/data_loader.py` by
deduplicating before splitting (see commit "Fix train/test leakage from
duplicate rows in reproduction"); metrics reported above are computed
post-fix, on the leakage-free splits.

## Training cost

Each training run (up to 400 boosting rounds, ~29K training rows, 14
features) completes in under 5 seconds on a CPU-only laptop — well
within the "no GPU, no multi-week training" constraint for this project.
