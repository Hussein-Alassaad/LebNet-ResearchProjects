"""Evaluate a trained model against the metrics used in the XGBoost paper.

The paper reports test error rate and AUC for its classification
benchmarks (e.g. the Higgs Boson experiment in Table 3). We report the
same two metrics here, plus log loss and F1 since the Adult task is
class-imbalanced (~24% positive) and accuracy alone can be misleading.

Usage:
    python src/evaluate.py [--model-name xgboost_adult]
"""

from __future__ import annotations

import argparse

from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from data_loader import load_adult_dataset
from model import XGBoostModel
from utils import MODELS_DIR, RESULTS_DIR, get_logger, save_json

logger = get_logger("evaluate")


def compute_metrics(y_true, y_proba, y_pred) -> dict:
    """Compute the paper-aligned metric set for a binary classifier.

    ``error_rate`` mirrors the paper's "test-error" column directly
    (1 - accuracy), reported alongside accuracy for easy cross-reference.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "error_rate": float(1 - accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def evaluate_model(model: XGBoostModel, x, y) -> dict:
    y_proba = model.predict_proba(x)
    y_pred = model.predict(x)
    return compute_metrics(y, y_proba, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost model")
    parser.add_argument("--model-name", type=str, default="xgboost_adult")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = MODELS_DIR / f"{args.model_name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run src/train.py first.")

    logger.info("Loading dataset and model %s...", model_path)
    dataset = load_adult_dataset()
    model = XGBoostModel().load(str(model_path))

    val_metrics = evaluate_model(model, dataset.x_val, dataset.y_val)
    test_metrics = evaluate_model(model, dataset.x_test, dataset.y_test)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"validation": val_metrics, "test": test_metrics},
        RESULTS_DIR / f"{args.model_name}_metrics.json",
    )
    logger.info("Wrote metrics to %s", RESULTS_DIR / f"{args.model_name}_metrics.json")


if __name__ == "__main__":
    main()
