# Findings

## What I learned from the paper

XGBoost's headline contribution isn't a new model family — gradient
boosted trees predate the paper by well over a decade — it's that a
*regularized objective* (Eq. 2: training loss plus an explicit penalty
on leaf count and leaf weight magnitude) combined with *systems-level
engineering* (sparsity-aware splits, a weighted quantile sketch for
approximate split finding, cache-aware data layout) turns an already-
good algorithm into one that is simultaneously more accurate, more
robust to missing/sparse data, and dramatically faster at scale. Reading
the paper closely made it clear how much of "XGBoost is good" is really
"XGBoost is *engineered* well" — the second-order Taylor approximation
and closed-form leaf weights (Eq. 5) are elegant, but the paper spends
at least as much space on block-based storage and out-of-core computation
as on the objective itself. That systems framing directly motivated this
project's extension (see below).

## How reproduction went

Reproduction was smoother than expected on the metrics side — the
official `xgboost` package already implements the paper's algorithm
faithfully, so "reproducing" the paper mostly meant applying it
correctly to a standard benchmark rather than debugging a from-scratch
implementation. The default configuration landed within the commonly
reported reference range for Adult Census Income on the very first full
training run (`results/reproduction_comparison_notes.md`), which was
almost anticlimactic compared to the "attempt after attempt of
hyperparameter chasing" I expected going in.

The one genuine bug I hit was in `src/data_loader.py`, in two parts:

1. A pandas-version compatibility issue — pandas 3.0's new default
   string dtype (`str`) isn't recognized by the categorical-detection
   logic that checked `dtype == object`, which silently skipped
   converting any column to `category`, causing a downstream XGBoost
   error about invalid dtypes. Small fix, but a good reminder that
   "works on my machine" reproducibility claims are fragile across
   library versions unless explicitly tested.
2. A more interesting data-quality issue: the Adult dataset's coarse
   census bucketing produces ~52 rows that are exact duplicates of each
   other. Under a naive random train/test split, some of these
   duplicates land on both sides of the split — meaning the model
   effectively "sees" a handful of test examples during training. This
   is a genuine, if small, leakage bug, and I only found it because I
   went looking for train/test overlap as a sanity check rather than
   trusting that a `sklearn.train_test_split` call is automatically
   leakage-free. The fix (deduplicate before splitting) was simple, but
   the discovery process was the more valuable part.

## What surprised me

- How little tuning mattered. Two rounds of hyperparameter search moved
  test accuracy by well under half a percentage point. For a dataset
  this size and this "easy" (Adult Census Income is a widely-used
  teaching benchmark precisely because it's well-behaved), the paper's
  regularized objective with sane defaults is already close to whatever
  ceiling exists for a single boosted-tree model.
- How large the *training-time* gap was between CatBoost and the other
  two libraries in the extension (~16-24x slower), despite achieving
  essentially the same accuracy. I expected CatBoost's ordered boosting
  to cost *something*, but not an order of magnitude — it's a vivid,
  concrete illustration of a systems design choice (avoiding prediction
  shift) having a real, measurable price, exactly the kind of trade-off
  the original paper's own systems-vs-accuracy framing is about.
- That the fastest-to-train library (LightGBM) and the fastest-at-
  inference library (CatBoost) were different libraries — there wasn't
  a single winner across every efficiency axis, which made "efficiency"
  a less one-dimensional story than I initially expected when writing
  the extension plan.

## Extension insights

The extension's real finding isn't about accuracy (all three libraries
tie) — it's that a "systems paper" like XGBoost's invites exactly this
kind of comparison, and the two libraries built in direct response to
it (LightGBM, CatBoost) each made a different, legible trade-off:
LightGBM optimized training throughput (GOSS/EFB), CatBoost optimized
robustness-to-overfitting and inference latency (ordered boosting,
oblivious trees) at the cost of training time. None of this shows up if
you only look at accuracy tables, which is exactly why the original
paper itself reports wall-clock benchmarks alongside its accuracy
numbers — the extension is, in that sense, a small-scale repeat of the
paper's own methodology, applied to its successors.

## Challenges faced

- Balancing "matched" hyperparameters across three libraries with
  genuinely different tree-growth strategies (level-wise vs. leaf-wise
  vs. oblivious) — there is no perfect 1:1 mapping between
  `max_depth`, `num_leaves`, and CatBoost's `depth`, so the comparison
  is honest but not perfectly controlled, and I documented that
  limitation explicitly rather than overstating the comparison's rigor.
- CatBoost's categorical-feature handling requires missing values to be
  an explicit string category rather than `NaN` (unlike XGBoost's native
  sparsity-aware handling of `NaN`), which needed a small adapter layer
  in `src/extension_models.py` to keep `data_loader.py`'s output format
  unchanged across all three model wrappers.
- Windows-specific friction (PowerShell vs. Git Bash path/quoting
  differences, `nbconvert` Proactor event loop warnings when executing
  notebooks) that didn't affect correctness but slowed down the
  verification process.

## Key takeaways

1. A well-engineered, regularized algorithm on a well-behaved dataset
   doesn't need much tuning to hit reference-quality results — the
   paper's regularization is doing real work, not just adding a
   diminishing-returns knob.
2. Reproducibility bugs are often not in the model at all — they're in
   the unglamorous data-splitting code, and finding them requires
   actively checking assumptions (e.g. "is my split actually leakage-
   free?") rather than trusting a library call by default.
3. "Systems" claims in a paper (speed, scalability, memory) are testable
   and falsifiable in the same way accuracy claims are, and doing so —
   as the extension did — surfaces real, mechanism-grounded differences
   between libraries that an accuracy-only comparison would completely
   miss.
