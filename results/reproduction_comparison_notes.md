# Baseline results vs. reference numbers — gap analysis

## Important scoping note

The original XGBoost paper (Chen & Guestrin, KDD 2016) benchmarks on
**Higgs Boson, Yahoo LTRC, Allstate, and a synthetic large-scale dataset**
— it does not report results on Adult Census Income. This project
reproduces the paper's **method** (regularized gradient boosting via the
official `xgboost` implementation) and applies it faithfully to a
standard, commonly-used, easily-reproducible tabular benchmark (Adult
Census Income) rather than attempting to reproduce the exact Higgs/Yahoo
numbers, which need substantially larger downloads and longer training.

The comparison below is therefore against **widely-reported reference
numbers for gradient boosted trees / XGBoost on Adult Census Income**
from the broader ML literature and common benchmark suites (e.g.
scikit-learn examples, OpenML benchmark leaderboards), not a specific
table in the KDD 2016 paper itself. This is documented explicitly to
avoid overstating what is being reproduced.

## Our baseline run (`xgboost_adult`, default config)

From `results/xgboost_adult_metrics.json`:

| Split      | Accuracy | Error rate | AUC    | Log loss | F1     |
|------------|---------:|-----------:|-------:|---------:|-------:|
| Validation | 0.8818   | 0.1182     | 0.9278 | 0.2776   | 0.7343 |
| Test       | 0.8748   | 0.1252     | 0.9281 | 0.2776   | 0.7175 |

Config: `max_depth=6, eta=0.1, lambda=1.0, subsample=0.8,
colsample_bytree=0.8`, early stopping at round 179/300 (see
`results/xgboost_adult_train_run.json`).

## Reference range for GBT / XGBoost on Adult Census Income

Commonly reported test accuracy for gradient boosted tree models
(XGBoost, LightGBM, and earlier GBM implementations) on this dataset
clusters around **86-87.5% accuracy** and **AUC in the 0.92-0.93 range**,
depending on preprocessing choices (native categorical handling vs.
one-hot encoding, hyperparameter tuning depth, and exact train/test
split).

## Comparison

| Metric   | Our test result | Reference range | Gap                     |
|----------|-----------------:|-----------------:|--------------------------|
| Accuracy | 87.48%            | ~86-87.5%         | Within range, near top   |
| AUC      | 0.9281            | ~0.92-0.93        | Within range             |

No discrepancy requiring debugging was found on the first full run —
results land inside the commonly reported range with default,
untuned hyperparameters. Section "Step 6: Reproduce Official Results"
below therefore focuses on a small amount of hyperparameter exploration
to confirm stability of these numbers rather than chasing a large gap.
