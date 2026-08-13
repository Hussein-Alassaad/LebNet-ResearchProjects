"""Extension: LightGBM and CatBoost wrappers for the efficiency comparison.

See docs/extension_plan.md for the motivation. These wrappers mirror
`model.XGBoostModel`'s interface (`fit`, `predict_proba`, `predict`,
`save`, `load`) so `src/benchmark_extension.py` can drive all three
libraries identically. Kept in a separate module from `model.py` so the
original reproduction code is untouched by the extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters, chosen to match `XGBoostConfig`'s budget.

    `num_leaves` is LightGBM's primary complexity control under its
    leaf-wise (best-first) tree growth, unlike XGBoost's depth-wise
    growth controlled by `max_depth`. We set `num_leaves` to roughly
    `2**max_depth` to give both libraries a comparable per-tree capacity
    budget, while keeping `max_depth` as a secondary cap to avoid
    LightGBM growing unusually deep, unbalanced trees on this small
    dataset.
    """

    num_boost_round: int = 400
    max_depth: int = 5
    num_leaves: int = 31  # ~2**5
    learning_rate: float = 0.1
    reg_lambda: float = 2.0
    min_split_gain: float = 1.0  # analogous to xgboost's gamma
    subsample: float = 0.9
    colsample_bytree: float = 0.7
    early_stopping_rounds: int = 20
    seed: int = 42

    def to_lgb_params(self) -> dict:
        return {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "lambda_l2": self.reg_lambda,
            "min_split_gain": self.min_split_gain,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1,
            "feature_fraction": self.colsample_bytree,
            "seed": self.seed,
            "verbosity": -1,
        }


@dataclass
class LightGBMModel:
    """Thin wrapper around `lightgbm.Booster`, mirroring `model.XGBoostModel`'s interface."""

    config: LightGBMConfig = field(default_factory=LightGBMConfig)
    booster: lgb.Booster | None = None

    def fit(self, x_train, y_train, x_val, y_val) -> "LightGBMModel":
        """Train via leaf-wise boosting with GOSS/EFB, early-stopped on validation logloss."""
        dtrain = lgb.Dataset(x_train, label=y_train)
        dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
        self.booster = lgb.train(
            params=self.config.to_lgb_params(),
            train_set=dtrain,
            num_boost_round=self.config.num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.booster.predict(x, num_iteration=self.booster.best_iteration)

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        self.booster.save_model(path)

    def load(self, path: str) -> "LightGBMModel":
        self.booster = lgb.Booster(model_file=path)
        return self


@dataclass
class CatBoostConfigWrapper:
    """CatBoost hyperparameters, matched to the same capacity/regularization budget."""

    iterations: int = 400
    depth: int = 5
    learning_rate: float = 0.1
    l2_leaf_reg: float = 2.0
    subsample: float = 0.9
    colsample_bylevel: float = 0.7
    early_stopping_rounds: int = 20
    seed: int = 42


_CATBOOST_MISSING_SENTINEL = "__missing__"


@dataclass
class CatBoostModel:
    """Wraps CatBoostClassifier, using native categorical feature support.

    Unlike XGBoost/LightGBM (which need explicit category encoding),
    CatBoost consumes raw categorical columns directly via `cat_features`
    and uses ordered target statistics internally -- this is CatBoost's
    key claimed advantage on categorical-heavy tabular data like Adult
    Census Income (~57% categorical features).

    CatBoost's categorical handling requires missing values to be an
    explicit string category rather than NaN (unlike XGBoost's
    sparsity-aware splits, which treat NaN as a native "go missing"
    signal) -- we fill NaN with a sentinel string so CatBoost's own
    target-statistic encoding can treat "missing" as just another
    category value, its documented approach for handling missing
    categoricals.
    """

    config: CatBoostConfigWrapper = field(default_factory=CatBoostConfigWrapper)
    booster: cb.CatBoostClassifier | None = None
    categorical_columns: list[str] = field(default_factory=list)

    def _fill_missing_categoricals(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for col in self.categorical_columns:
            if str(x[col].dtype) == "category" and _CATBOOST_MISSING_SENTINEL not in x[col].cat.categories:
                x[col] = x[col].cat.add_categories([_CATBOOST_MISSING_SENTINEL])
            x[col] = x[col].fillna(_CATBOOST_MISSING_SENTINEL)
        return x

    def fit(self, x_train, y_train, x_val, y_val, categorical_columns: list[str]) -> "CatBoostModel":
        self.categorical_columns = categorical_columns
        x_train = self._fill_missing_categoricals(x_train)
        x_val = self._fill_missing_categoricals(x_val)

        self.booster = cb.CatBoostClassifier(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            l2_leaf_reg=self.config.l2_leaf_reg,
            subsample=self.config.subsample,
            colsample_bylevel=self.config.colsample_bylevel,
            random_seed=self.config.seed,
            eval_metric="AUC",
            bootstrap_type="Bernoulli",
            early_stopping_rounds=self.config.early_stopping_rounds,
            cat_features=categorical_columns,
            verbose=False,
        )
        self.booster.fit(x_train, y_train, eval_set=(x_val, y_val))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        x = self._fill_missing_categoricals(x)
        return self.booster.predict_proba(x)[:, 1]

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("Model has not been fit yet.")
        self.booster.save_model(path)

    def load(self, path: str, categorical_columns: list[str]) -> "CatBoostModel":
        self.categorical_columns = categorical_columns
        self.booster = cb.CatBoostClassifier()
        self.booster.load_model(path)
        return self
