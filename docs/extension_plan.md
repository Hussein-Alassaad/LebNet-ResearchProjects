# Extension Plan: Efficiency Comparison — XGBoost vs. LightGBM vs. CatBoost

## What extension I'm doing, and why

The XGBoost paper's central claim is not just "boosted trees can be
accurate" — it is that a *systems-aware* implementation (sparsity-aware
splits, weighted quantile sketch, cache-aware block structure) makes
gradient boosting simultaneously accurate **and** fast/scalable
(Section 1: "the impact of XGBoost has been widely recognized in a
number of machine learning and data mining challenges... [for both]
scalability... and accuracy").

Two major libraries were built explicitly in response to XGBoost's
design, each making a different systems trade-off:

- **LightGBM** (Ke et al., NeurIPS 2017) replaces XGBoost's
  level-wise tree growth with **leaf-wise** growth and uses
  **gradient-based one-side sampling (GOSS)** + **exclusive feature
  bundling (EFB)** to speed up histogram-based split finding.
- **CatBoost** (Prokhorenkova et al., NeurIPS 2018) introduces
  **ordered boosting** and native **target-statistic encoding** for
  categorical features, aimed at reducing prediction shift / overfitting
  on categorical-heavy data — directly relevant to Adult Census Income,
  which is ~57% categorical features.

This extension directly tests the paper's own efficiency framing: given
the *same* dataset, splits, and a comparable regularized-objective
config, how do XGBoost's accuracy and training/inference speed compare
to the two libraries that followed it? This is a natural, scoped
extension — it reuses the existing data pipeline and evaluation metrics
unchanged, adding only new model wrappers and a benchmarking script.

## How I'll measure success

For each library (XGBoost, LightGBM, CatBoost), on the identical
train/val/test splits from `src/data_loader.py`:

1. **Accuracy metrics** — accuracy, AUC, log loss, F1 (same as
   `src/evaluate.py`), to check no library sacrifices predictive quality.
2. **Training wall-clock time** — seconds to fit with early stopping,
   using comparable hyperparameters (same `max_depth`/tree-count budget,
   learning rate) across libraries where the concept transfers directly.
3. **Inference wall-clock time** — seconds to score the full test set.
4. **Model size on disk** — a rough proxy for memory footprint.

Success is not "one library wins" — it's producing an honest,
apples-to-aples efficiency/accuracy comparison and explaining *why* any
differences appear, tying back to each library's specific systems
contribution (leaf-wise growth, GOSS/EFB, ordered boosting/categorical
handling).

## Expected outcomes

- All three libraries should land in a similar accuracy/AUC range on
  this dataset (Adult Census Income is a well-behaved benchmark where
  boosting variants tend to converge to similar quality).
- LightGBM is expected to train fastest due to leaf-wise growth + GOSS,
  especially as data volume grows (though at ~30K rows the effect may be
  small).
- CatBoost's native categorical handling may show a modest accuracy or
  robustness edge given Adult's high categorical feature share, at the
  cost of somewhat slower training (ordered boosting has extra overhead).
- XGBoost, the paper's own system, should remain competitive and
  well-balanced across both axes rather than dominating either.

## Implementation timeline

1. Add `LightGBMModel` / `CatBoostModel` wrappers in `src/model.py` (or a
   new `src/extension_models.py`) mirroring `XGBoostModel`'s interface.
2. Add `src/benchmark_extension.py` to train/evaluate/time all three
   models on the same splits and write results to
   `results/extension_results.json`.
3. Add `notebooks/03_extension_analysis.ipynb` to visualize the
   accuracy-vs-speed comparison.
4. Write up findings in `results/extension_results.md`.

## Risks & fallbacks

- **Risk:** hyperparameters aren't perfectly equivalent across libraries
  (e.g. LightGBM's `num_leaves` vs. XGBoost's `max_depth` don't map
  1:1). **Fallback:** document the mapping choices explicitly rather
  than claiming a perfectly controlled comparison; report both
  "matched-budget" and each library's own reasonable defaults.
- **Risk:** timing noise on a shared/laptop CPU. **Fallback:** run each
  training/inference step multiple times and report mean ± std.
- **Risk:** a library's categorical handling requires different
  preprocessing (e.g. CatBoost wants raw category indices/strings, not
  pandas `category` dtype in all code paths). **Fallback:** keep a
  clearly separated adapter per library so `data_loader.py`'s output is
  unchanged and each model wrapper handles its own conversion.
