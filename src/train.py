"""Train the XGBoost reproduction model on the Adult Census Income dataset.

Usage:
    python src/train.py [--num-boost-round N] [--max-depth N] [--eta F]

Loads data via `data_loader`, fits `model.XGBoostModel` with early
stopping on the validation split, saves the trained booster to
`models/`, and writes training curves + config to `results/`.
"""

from __future__ import annotations

import argparse
import time

from data_loader import load_adult_dataset
from model import XGBoostConfig, XGBoostModel
from utils import MODELS_DIR, RESULTS_DIR, get_logger, save_json, set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on Adult Census Income")
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="xgboost_adult")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading Adult Census Income dataset...")
    dataset = load_adult_dataset()
    logger.info("Loaded: %s", dataset.stats())

    config = XGBoostConfig(
        num_boost_round=args.num_boost_round,
        max_depth=args.max_depth,
        eta=args.eta,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )
    logger.info("Config: %s", config)

    model = XGBoostModel(config=config)

    checkpoint_dir = MODELS_DIR / "checkpoints" / args.model_name

    start = time.time()
    model.fit(
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        verbose_eval=25,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    train_seconds = time.time() - start
    logger.info(
        "Training complete in %.1fs, best_iteration=%s (of %d requested)",
        train_seconds,
        model.best_iteration,
        config.num_boost_round,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.model_name}.json"
    model.save(str(model_path))
    logger.info("Saved model to %s", model_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "config": config.to_xgb_params()
            | {"num_boost_round": config.num_boost_round, "early_stopping_rounds": config.early_stopping_rounds},
            "best_iteration": model.best_iteration,
            "train_seconds": train_seconds,
            "data_stats": dataset.stats(),
        },
        RESULTS_DIR / f"{args.model_name}_train_run.json",
    )
    save_json(model.evals_result_, RESULTS_DIR / f"{args.model_name}_evals_result.json")
    logger.info("Wrote training run metadata to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
