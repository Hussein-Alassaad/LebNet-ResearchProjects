"""Model definition matching XGBoost (Chen & Guestrin, KDD 2016).

The paper's core contribution is not a fixed "architecture" in the neural
network sense, but a regularized gradient boosting objective:

    L(phi) = sum_i l(y_i, y_hat_i) + sum_k Omega(f_k)
    Omega(f) = gamma * T + 0.5 * lambda * ||w||^2

where each f_k is a regression tree with T leaves and leaf weights w,
l is a differentiable convex loss (log loss for binary classification),
gamma penalizes the number of leaves, and lambda is L2 regularization on
leaf weights. Trees are added greedily and each split is scored with the
paper's exact gain formula (Eq. 7):

    Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                   - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

This module wraps the official `xgboost` package (which implements this
objective, second-order Newton boosting, and the sparsity-aware split
algorithm from the paper) behind a small, explicit config class so all
hyperparameters referenced in the paper are visible in one place rather
than scattered through training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class CheckpointCallback(xgb.callback.TrainingCallback):
    """Save the booster to disk every `interval` boosting rounds.

    Lets a long training run be resumed or inspected mid-flight instead
    of only having the final model, at the cost of a small amount of
    extra disk I/O every `interval` rounds.
    """

    def __init__(self, checkpoint_dir: Path, interval: int = 50, prefix: str = "checkpoint"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval = interval
        self.prefix = prefix

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log) -> bool:
        round_number = epoch + 1
        if round_number % self.interval == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(self.checkpoint_dir / f"{self.prefix}_round{round_number}.json"))
        return False


@dataclass
class XGBoostConfig:
    """Hyperparameters for the boosted tree ensemble.

    Names follow the paper's notation where applicable:
    - ``num_boost_round``: number of boosting rounds (trees), i.e. K in
      the additive model f(x) = sum_{k=1}^{K} f_k(x).
    - ``max_depth``: maximum depth of each regression tree.
    - ``eta``: shrinkage/learning rate applied to new tree weights
      (Section 2.3 "Shrinkage and Column Subsampling").
    - ``reg_lambda`` / ``reg_alpha``: L2 / L1 regularization terms in
      Omega(f) (Eq. 2).
    - ``gamma``: minimum loss reduction required to make a further split
      (the gamma * T term in Omega(f)).
    - ``subsample`` / ``colsample_bytree``: row and column subsampling
      ratios (Section 2.3), a regularization technique borrowed from
      random forests.
    - ``tree_method``: split-finding algorithm; "hist" is the
      histogram-based approximation of the paper's weighted quantile
      sketch (Section 3), used for speed on CPU.
    """

    num_boost_round: int = 300
    max_depth: int = 6
    eta: float = 0.1
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    objective: str = "binary:logistic"
    eval_metric: tuple[str, ...] = ("logloss", "auc")
    early_stopping_rounds: int = 20
    seed: int = 42

    def to_xgb_params(self) -> dict:
        """Translate to the flat dict `xgboost.train` expects."""
        return {
            "max_depth": self.max_depth,
            "eta": self.eta,
            "lambda": self.reg_lambda,
            "alpha": self.reg_alpha,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "objective": self.objective,
            "eval_metric": list(self.eval_metric),
            "seed": self.seed,
        }


@dataclass
class XGBoostModel:
    """Thin wrapper around `xgboost.Booster` for the Adult income task."""

    config: XGBoostConfig = field(default_factory=XGBoostConfig)
    booster: xgb.Booster | None = None
    best_iteration: int | None = None

    def _to_dmatrix(self, x: pd.DataFrame, y: pd.Series | None = None) -> xgb.DMatrix:
        """Wrap a pandas frame as XGBoost's internal DMatrix format.

        `enable_categorical=True` lets XGBoost consume the `category`
        dtype columns produced by `data_loader.load_adult_dataset`
        directly, using the paper's sparsity-aware split algorithm to
        route each category (and missing values) without manual
        one-hot encoding.
        """
        return xgb.DMatrix(x, label=y, enable_categorical=True)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        verbose_eval: bool | int = False,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 50,
    ) -> "XGBoostModel":
        """Train via Newton boosting on the regularized objective.

        Uses a held-out validation set for early stopping, matching the
        paper's use of validation-based stopping to control overfitting
        instead of a fixed round count. If `checkpoint_dir` is given, the
        booster is additionally saved every `checkpoint_interval` rounds.
        """
        dtrain = self._to_dmatrix(x_train, y_train)
        dval = self._to_dmatrix(x_val, y_val)

        callbacks = []
        if checkpoint_dir is not None:
            callbacks.append(CheckpointCallback(checkpoint_dir, interval=checkpoint_interval))

        evals_result: dict = {}
        self.booster = xgb.train(
            params=self.config.to_xgb_params(),
            dtrain=dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration
        self.evals_result_ = evals_result
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return P(income > 50K) for each row."""
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        dmatrix = self._to_dmatrix(x)
        iteration_range = (0, self.best_iteration + 1) if self.best_iteration is not None else (0, 0)
        return self.booster.predict(dmatrix, iteration_range=iteration_range)

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Threshold `predict_proba` into a hard {0, 1} income class."""
        return (self.predict_proba(x) >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Persist the boosted ensemble (all trees f_1..f_K) to disk."""
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        self.booster.save_model(path)

    def load(self, path: str) -> "XGBoostModel":
        """Load a previously saved ensemble, replacing any current booster."""
        self.booster = xgb.Booster()
        self.booster.load_model(path)
        return self
