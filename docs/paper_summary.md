# Paper Summary

## Reference

- **Title:** XGBoost: A Scalable Tree Boosting System
- **Authors:** Tianqi Chen, Carlos Guestrin
- **Venue:** KDD 2016 (22nd ACM SIGKDD International Conference on Knowledge
  Discovery and Data Mining)
- **Year:** 2016
- **Paper:** https://arxiv.org/abs/1603.02754
- **Official code:** https://github.com/dmlc/xgboost
- **Papers with Code:** https://paperswithcode.com/paper/xgboost-a-scalable-tree-boosting-system

## Core problem

Gradient boosted trees are among the strongest general-purpose models for
structured/tabular data, but existing implementations at the time did not
scale well to large datasets and did not systematically handle sparse data
(missing values, one-hot encoded categoricals) or make efficient use of
memory and CPU cache. The paper asks: how do you build a tree boosting
system that is simultaneously more accurate (via better regularization),
algorithmically scalable to huge data, and fast in practice on real
hardware?

## Main methodology

XGBoost optimizes a regularized objective (training loss + a penalty on
tree complexity — number of leaves and leaf weight magnitudes) using a
second-order Taylor expansion of the loss, which gives a closed-form
per-leaf optimal weight and a "gain" formula used to greedily choose splits.
On top of this objective, the paper introduces systems-level contributions:
a **sparsity-aware split-finding algorithm** that learns a default direction
for missing/sparse values, a **weighted quantile sketch** for approximate
split finding on weighted data, and cache-aware / out-of-core / block-based
data structures that make the algorithm fast and scalable on a single
machine or cluster.

## Key results reported in the paper

- State-of-the-art or better accuracy compared to prior boosting
  implementations (e.g. scikit-learn GBM, R gbm, pGBRT) across several
  benchmark datasets, while training substantially faster.
- Scales to datasets with billions of examples using the same code,
  via out-of-core computation and distributed/parallel execution.
- Ablations show the regularized objective reduces overfitting relative to
  standard (unregularized) gradient boosting, and the sparsity-aware
  algorithm gives large speedups (50x+ in the paper's Allstate experiment)
  over a naive approach to missing data.

## Why this paper

- **CPU-feasible:** tree boosting on tabular data trains in minutes on a
  laptop CPU — no GPU or multi-day training run required, which fits the
  timeline and hardware available for this project.
- **Reproducible:** the official `xgboost` Python package implements
  exactly the algorithm described in the paper, so "reproduction" means
  faithfully applying it to a public dataset and reporting standard
  metrics (accuracy, AUC, log loss) rather than re-implementing a research
  codebase from scratch.
- **Publicly available data:** the UCI **Adult Census Income** dataset
  (a binary income-classification task, ~48,842 rows, 14 features, mixed
  categorical/numeric) is a standard, widely-used tabular benchmark with
  no licensing friction, easily downloaded via `sklearn.datasets` /
  OpenML or the UCI repository mirror.
- **Understandable core idea:** gradient boosting with second-order
  loss approximation and a regularized objective is a well-scoped,
  well-documented idea — deep enough to be a real reproduction, not so
  complex that debugging becomes intractable.
- **Natural extension:** because XGBoost's own paper is partly about
  *systems efficiency* (not just accuracy), a follow-up efficiency
  comparison against later boosting libraries (LightGBM, CatBoost) that
  explicitly responded to XGBoost's design is a direct, meaningful
  extension of the paper's own central claim — see
  [docs/extension_plan.md](extension_plan.md).
