"""Extension: benchmark XGBoost vs. LightGBM vs. CatBoost on Adult Census Income.

Trains each library with a matched capacity/regularization budget (see
docs/extension_plan.md for the hyperparameter mapping rationale),
measures training time, inference time, model size, and the standard
accuracy/AUC/log loss/F1 metrics, and writes a combined comparison to
results/extension_results.json.

Usage:
    python src/benchmark_extension.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data_loader import load_adult_dataset
from evaluate import compute_metrics
from extension_models import CatBoostConfigWrapper, CatBoostModel, LightGBMConfig, LightGBMModel
from model import XGBoostConfig, XGBoostModel
from utils import MODELS_DIR, RESULTS_DIR, get_logger, save_json, set_seed

logger = get_logger("benchmark_extension")

# Matched capacity/regularization budget across all three libraries,
# mirroring the selected xgboost_adult_final configuration.
NUM_ROUNDS = 400
MAX_DEPTH = 5
LEARNING_RATE = 0.1
L2_REG = 2.0
SUBSAMPLE = 0.9
COLSAMPLE = 0.7
EARLY_STOPPING_ROUNDS = 20
SEED = 42


def _time_call(fn, repeats: int) -> tuple[float, float]:
    """Run `fn` `repeats` times, returning (mean_seconds, std_seconds)."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return float(np.mean(durations)), float(np.std(durations))


def benchmark_xgboost(dataset, repeats: int) -> dict:
    config = XGBoostConfig(
        num_boost_round=NUM_ROUNDS,
        max_depth=MAX_DEPTH,
        eta=LEARNING_RATE,
        reg_lambda=L2_REG,
        gamma=1.0,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=SEED,
    )
    model = XGBoostModel(config=config)
    train_mean, train_std = _time_call(
        lambda: model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val), repeats=1
    )

    infer_mean, infer_std = _time_call(lambda: model.predict_proba(dataset.x_test), repeats=repeats)

    model_path = MODELS_DIR / "extension_xgboost.json"
    model.save(str(model_path))

    metrics = compute_metrics(dataset.y_test, model.predict_proba(dataset.x_test), model.predict(dataset.x_test))
    return {
        "library": "xgboost",
        "train_seconds": train_mean,
        "train_seconds_std": train_std,
        "inference_seconds": infer_mean,
        "inference_seconds_std": infer_std,
        "model_size_bytes": model_path.stat().st_size,
        "best_iteration": model.best_iteration,
        **metrics,
    }


def benchmark_lightgbm(dataset, repeats: int) -> dict:
    config = LightGBMConfig(
        num_boost_round=NUM_ROUNDS,
        max_depth=MAX_DEPTH,
        num_leaves=2**MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        reg_lambda=L2_REG,
        min_split_gain=1.0,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=SEED,
    )
    model = LightGBMModel(config=config)
    train_mean, train_std = _time_call(
        lambda: model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val), repeats=1
    )

    infer_mean, infer_std = _time_call(lambda: model.predict_proba(dataset.x_test), repeats=repeats)

    model_path = MODELS_DIR / "extension_lightgbm.txt"
    model.save(str(model_path))

    metrics = compute_metrics(dataset.y_test, model.predict_proba(dataset.x_test), model.predict(dataset.x_test))
    return {
        "library": "lightgbm",
        "train_seconds": train_mean,
        "train_seconds_std": train_std,
        "inference_seconds": infer_mean,
        "inference_seconds_std": infer_std,
        "model_size_bytes": model_path.stat().st_size,
        "best_iteration": model.booster.best_iteration,
        **metrics,
    }


def benchmark_catboost(dataset, repeats: int) -> dict:
    config = CatBoostConfigWrapper(
        iterations=NUM_ROUNDS,
        depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2_REG,
        subsample=SUBSAMPLE,
        colsample_bylevel=COLSAMPLE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=SEED,
    )
    model = CatBoostModel(config=config)
    train_mean, train_std = _time_call(
        lambda: model.fit(
            dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val, dataset.categorical_columns
        ),
        repeats=1,
    )

    infer_mean, infer_std = _time_call(lambda: model.predict_proba(dataset.x_test), repeats=repeats)

    model_path = MODELS_DIR / "extension_catboost.cbm"
    model.save(str(model_path))

    metrics = compute_metrics(dataset.y_test, model.predict_proba(dataset.x_test), model.predict(dataset.x_test))
    return {
        "library": "catboost",
        "train_seconds": train_mean,
        "train_seconds_std": train_std,
        "inference_seconds": infer_mean,
        "inference_seconds_std": infer_std,
        "model_size_bytes": model_path.stat().st_size,
        "best_iteration": model.booster.get_best_iteration(),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark XGBoost vs LightGBM vs CatBoost")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats for inference timing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    logger.info("Loading dataset...")
    dataset = load_adult_dataset()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name, fn in [
        ("xgboost", benchmark_xgboost),
        ("lightgbm", benchmark_lightgbm),
        ("catboost", benchmark_catboost),
    ]:
        logger.info("Benchmarking %s...", name)
        result = fn(dataset, repeats=args.repeats)
        logger.info(
            "%s: train=%.2fs, infer=%.4fs (+/-%.4f), acc=%.4f, auc=%.4f",
            name,
            result["train_seconds"],
            result["inference_seconds"],
            result["inference_seconds_std"],
            result["accuracy"],
            result["auc"],
        )
        results.append(result)

    save_json({"results": results, "matched_config": {
        "num_rounds": NUM_ROUNDS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "l2_reg": L2_REG,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "seed": SEED,
    }}, RESULTS_DIR / "extension_results.json")
    logger.info("Wrote results to %s", RESULTS_DIR / "extension_results.json")


if __name__ == "__main__":
    main()
