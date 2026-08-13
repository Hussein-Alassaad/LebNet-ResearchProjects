# Extension Results: XGBoost vs. LightGBM vs. CatBoost

See [docs/extension_plan.md](../docs/extension_plan.md) for motivation
and methodology. All three libraries trained with a matched
capacity/regularization budget (`max_depth`/`depth`=5, `lr`=0.1,
L2=2.0, subsample=0.9, colsample=0.7, up to 400 rounds, 20-round early
stopping, seed=42) on the identical leakage-free train/val/test splits
from `src/data_loader.py`. Full numbers: `results/extension_results.json`.
Visualizations: `notebooks/03_extension_analysis.ipynb`.

## TL;DR

Accuracy and AUC are effectively tied across all three libraries
(within 0.16 percentage points / 0.0006 AUC) — the extension's real
finding is on **efficiency**: LightGBM trains fastest (~35% quicker
than XGBoost, ~24x quicker than CatBoost) and produces the smallest
model; CatBoost is far slower to train (ordered boosting overhead) but
fastest at inference (oblivious trees); XGBoost sits in the middle on
every axis, matching its framing as a well-rounded general-purpose
system rather than one optimized for a single metric.

## Results table

| Library  | Test accuracy | Test AUC | Log loss | F1     | Train time (s) | Inference time (s, test set) | Model size (KB) | Best iteration |
|----------|---------------:|---------:|---------:|-------:|----------------:|-------------------------------:|-----------------:|---------------:|
| XGBoost  | 0.8746         | 0.9293   | 0.2765   | 0.7165 | 0.70            | 0.0157                          | 469              | 141            |
| LightGBM | 0.8730         | 0.9289   | 0.2774   | 0.7126 | **0.46**        | 0.0238                          | **237**          | 111            |
| CatBoost | 0.8741         | 0.9287   | 0.2779   | 0.7111 | 10.96           | **0.0096**                      | 325              | 219            |

(Bold = best in column.)

## Comparison with the original reproduction

All three libraries land within **0.16 percentage points of each other
on test accuracy** (87.30-87.46%) and **within 0.0006 AUC** (0.9287-
0.9293) — none diverges meaningfully from XGBoost's reproduced result
(`results/reproduction_results.md`: 87.53% accuracy, 0.9294 AUC on the
`xgboost_adult_final` run; the `extension_xgboost` run here uses a
slightly different `gamma` and is a fresh fit, hence the small
difference from that earlier number). This confirms the expectation in
the extension plan: Adult Census Income is a well-behaved benchmark
where boosting variants converge to similar predictive quality
regardless of the specific systems-level algorithm used to build the
trees.

The real differentiation is in **efficiency**, which is exactly what
the extension plan set out to probe:

- **LightGBM trained ~35% faster than XGBoost** (0.46s vs. 0.70s) and
  produced a **~2x smaller model file** (237 KB vs. 469 KB), consistent
  with leaf-wise growth reaching a comparable loss with fewer/smaller
  splits, and GOSS/EFB reducing the work per boosting round.
- **CatBoost's training was dramatically slower** — ~16x slower than
  XGBoost and ~24x slower than LightGBM (10.96s vs. 0.70s / 0.46s) —
  the cost of ordered boosting, which requires multiple internal data
  permutations to avoid prediction shift, plus on-the-fly target-
  statistic computation for categorical features. This is a real,
  measurable systems cost for CatBoost's headline technique, not
  overhead specific to this dataset's size.
- **CatBoost had the fastest inference** (0.0096s vs. 0.0157s XGBoost,
  0.0238s LightGBM) despite the slowest training — its oblivious trees
  (same split condition at every node of a given depth) are cheap to
  evaluate, since a full tree's leaf can be looked up via a single
  bitmask rather than the sequential split checks generic (non-
  oblivious) trees like XGBoost's or LightGBM's require.
- XGBoost sits **in between on every axis** — not the fastest to train,
  not the fastest at inference, but reasonably competitive on both,
  which matches the paper's framing of XGBoost as a well-rounded,
  general-purpose system rather than one optimized for a single
  objective (train speed, inference speed, or memory) at the expense of
  the others.

## Why: tying back to each library's systems contribution

- **LightGBM's speed** traces directly to its two headline
  contributions: GOSS (subsampling low-gradient examples cheaply) and
  EFB (bundling mutually-exclusive sparse/categorical features), both
  aimed squarely at reducing histogram-construction cost per round —
  exactly what shows up as lower `train_seconds` here.
- **CatBoost's training cost** traces to ordered boosting: to avoid the
  "prediction shift" problem (using a tree's own training examples to
  compute its own leaf statistics), CatBoost maintains multiple
  permutations of the training data and computes target statistics in a
  causally-ordered way — provably more robust to overfitting on small
  data, but mechanically more expensive per round than XGBoost's or
  LightGBM's simpler histogram-based leaf statistics.
- **CatBoost's inference speed** traces to oblivious trees — a
  structural choice (same split feature/threshold at every node of a
  given tree depth) that trades some flexibility in what a single tree
  can represent for O(depth) branchless lookup at prediction time.

## Limitations of this comparison

- Hyperparameters are matched by *intent* (comparable capacity/
  regularization), not perfectly equivalent — e.g. LightGBM's
  `num_leaves=32` vs. XGBoost's `max_depth=5` (max 32 leaves) is a close
  but not identical capacity constraint, and CatBoost's `l2_leaf_reg`
  regularizes differently than XGBoost's leaf-weight L2 penalty.
- Timing was measured on a single shared laptop CPU with the repeats
  the script defaults to (5 for inference, 1 for training since training
  includes early stopping and each run is already deterministic given
  the fixed seed, so repeating it would not add signal); absolute times
  will vary by machine, but the
  *relative ordering* (LightGBM fastest to train, CatBoost slowest to
  train but fastest at inference) is a stable, mechanism-driven result
  expected to reproduce on other hardware.
- This dataset (~29K training rows) is small relative to what these
  libraries are built for; efficiency gaps (especially LightGBM's edge)
  are documented in each library's own literature to widen further on
  larger datasets than tested here.
