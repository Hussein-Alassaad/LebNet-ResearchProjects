"""Sanity test: model can fit a few rounds and produce valid predictions.

Not a full unit test suite (see src/test_reproducibility.py for that,
added in Phase 3) -- this just verifies the forward pass works end-to-end
on synthetic data shaped like the Adult dataset, so model.py bugs are
caught before running the full training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import XGBoostConfig, XGBoostModel


def _make_synthetic_frame(n_rows: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n_rows),
            "hours_per_week": rng.integers(1, 99, n_rows),
            "education_num": rng.integers(1, 16, n_rows),
            "workclass": pd.Categorical(rng.choice(["Private", "Self-emp", "Gov"], n_rows)),
            "sex": pd.Categorical(rng.choice(["Male", "Female"], n_rows)),
        }
    )
    y = pd.Series((x["age"] + x["hours_per_week"] > 100).astype(int), name="income")
    return x, y


def test_forward_pass_runs_and_returns_valid_probabilities():
    x, y = _make_synthetic_frame()
    x_train, x_val = x.iloc[:150], x.iloc[150:]
    y_train, y_val = y.iloc[:150], y.iloc[150:]

    model = XGBoostModel(config=XGBoostConfig(num_boost_round=10, early_stopping_rounds=5))
    model.fit(x_train, y_train, x_val, y_val)

    proba = model.predict_proba(x_val)
    assert proba.shape == (len(x_val),)
    assert np.all((proba >= 0) & (proba <= 1))

    preds = model.predict(x_val)
    assert set(np.unique(preds)).issubset({0, 1})


if __name__ == "__main__":
    test_forward_pass_runs_and_returns_valid_probabilities()
    print("forward pass test passed")
