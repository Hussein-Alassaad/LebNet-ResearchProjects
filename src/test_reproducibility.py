"""Unit tests covering the core reproduction pipeline: data loading,
a training step, and evaluation metrics. Run with:

    pytest src/test_reproducibility.py

Uses the real cached Adult dataset (downloads on first run via
data_loader.load_adult_dataset) rather than synthetic data, so these
tests double as an end-to-end reproducibility check.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_loader import load_adult_dataset
from evaluate import compute_metrics
from model import XGBoostConfig, XGBoostModel


@pytest.fixture(scope="module")
def dataset():
    return load_adult_dataset()


class TestDataLoading:
    def test_splits_are_non_empty_and_disjoint_sizes_match_expected_ratio(self, dataset):
        assert len(dataset.x_train) > 0
        assert len(dataset.x_val) > 0
        assert len(dataset.x_test) > 0
        total = len(dataset.x_train) + len(dataset.x_val) + len(dataset.x_test)
        # Test split should be roughly 1/3 of the full (deduplicated) dataset.
        assert abs(len(dataset.x_test) / total - 1 / 3) < 0.02

    def test_no_duplicate_rows_leak_across_train_and_test(self, dataset):
        train = dataset.x_train.copy()
        train["income"] = dataset.y_train.values
        test = dataset.x_test.copy()
        test["income"] = dataset.y_test.values

        train_keys = set(map(tuple, train.astype(str).values))
        test_keys = set(map(tuple, test.astype(str).values))
        assert train_keys.isdisjoint(test_keys)

    def test_categorical_columns_have_category_dtype(self, dataset):
        for col in dataset.categorical_columns:
            assert str(dataset.x_train[col].dtype) == "category"

    def test_target_is_binary(self, dataset):
        assert set(dataset.y_train.unique()).issubset({0, 1})


class TestTrainingStep:
    def test_single_training_run_improves_over_random_baseline(self, dataset):
        config = XGBoostConfig(num_boost_round=20, early_stopping_rounds=10)
        model = XGBoostModel(config=config)
        model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)

        proba = model.predict_proba(dataset.x_val)
        assert proba.shape == (len(dataset.x_val),)
        assert np.all((proba >= 0) & (proba <= 1))

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(dataset.y_val, proba)
        assert auc > 0.7  # well above the 0.5 random baseline

    def test_model_save_and_load_roundtrip_produces_identical_predictions(self, dataset, tmp_path):
        config = XGBoostConfig(num_boost_round=15, early_stopping_rounds=10)
        model = XGBoostModel(config=config)
        model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)

        proba_before = model.predict_proba(dataset.x_test)

        save_path = tmp_path / "model.json"
        model.save(str(save_path))

        reloaded = XGBoostModel().load(str(save_path))
        # A freshly loaded booster has no best_iteration tracked; force full range.
        reloaded.best_iteration = model.best_iteration
        proba_after = reloaded.predict_proba(dataset.x_test)

        np.testing.assert_allclose(proba_before, proba_after)


class TestEvaluationMetrics:
    def test_compute_metrics_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.01, 0.99, 0.02, 0.98, 0.97])
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_proba, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["error_rate"] == 0.0
        assert metrics["f1"] == 1.0
        assert metrics["auc"] == 1.0

    def test_compute_metrics_returns_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.7, 0.6, 0.4])
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_proba, y_pred)
        assert set(metrics.keys()) == {"accuracy", "error_rate", "auc", "log_loss", "f1"}
