"""Load and preprocess the UCI Adult Census Income dataset.

The Adult dataset is a standard tabular benchmark: predict whether a
person's income exceeds $50K/year from census attributes. It is fetched
from OpenML (mirrors the UCI repository copy) so no manual download step
is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "processed_data"

TARGET_COLUMN = "income"
RANDOM_STATE = 42

# Adult has a canonical 2/3 - 1/3 train/test split in the original UCI
# files; we replicate that ratio, carving validation out of the train
# portion so downstream code can early-stop without touching the test set.
TRAIN_FRACTION = 2 / 3
VAL_FRACTION_OF_TRAIN = 0.1


@dataclass
class AdultDataset:
    """Container for the processed Adult Census Income splits."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    categorical_columns: list[str]
    numeric_columns: list[str]

    def stats(self) -> dict:
        return {
            "n_train": len(self.x_train),
            "n_val": len(self.x_val),
            "n_test": len(self.x_test),
            "n_features": self.x_train.shape[1],
            "n_categorical": len(self.categorical_columns),
            "n_numeric": len(self.numeric_columns),
            "positive_rate_train": float(self.y_train.mean()),
            "positive_rate_val": float(self.y_val.mean()),
            "positive_rate_test": float(self.y_test.mean()),
        }


def _download_raw(cache_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Fetch the raw Adult dataset from OpenML and cache it locally as CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "adult_raw.csv"

    if raw_path.exists():
        return pd.read_csv(raw_path)

    bunch = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df.to_csv(raw_path, index=False)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names/values and encode the binary target."""
    df = df.copy()
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

    if "class" in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={"class": TARGET_COLUMN})

    # OpenML's "adult" encodes the target as a category like '>50K' / '<=50K'.
    target_raw = df[TARGET_COLUMN].astype(str).str.strip()
    df[TARGET_COLUMN] = target_raw.str.startswith(">50K").astype(int)

    # Treat '?' (the UCI missing-value sentinel) as a true missing value so
    # XGBoost's native sparsity-aware split finding can learn a default
    # direction for it, per the paper.
    df = df.replace("?", np.nan)

    return df


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[TARGET_COLUMN])
    # pandas >= 3.0 reports plain text columns as dtype "str" rather than
    # the legacy "object", so check both to stay compatible across
    # versions instead of relying on `dtype == object`.
    categorical_columns = [
        c for c in feature_df.columns if str(feature_df[c].dtype) in ("object", "str", "category")
    ]
    numeric_columns = [c for c in feature_df.columns if c not in categorical_columns]
    return categorical_columns, numeric_columns


def load_adult_dataset(force_redownload: bool = False) -> AdultDataset:
    """Download (if needed), clean, split, and return the Adult dataset.

    Splits: ~2/3 train, with 10% of that carved out as validation, and the
    remaining ~1/3 held out as test — matching the original UCI train/test
    file sizes (roughly 32,561 / 16,281 rows).
    """
    if force_redownload:
        raw_path = RAW_DATA_DIR / "adult_raw.csv"
        if raw_path.exists():
            raw_path.unlink()

    df = _clean(_download_raw())
    categorical_columns, numeric_columns = _split_columns(df)

    # Categorical dtype lets XGBoost consume categoricals natively
    # (enable_categorical=True) without manual one-hot encoding.
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, train_size=TRAIN_FRACTION, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VAL_FRACTION_OF_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return AdultDataset(
        x_train=x_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        x_val=x_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def save_processed(dataset: AdultDataset, out_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Persist processed splits and summary stats to processed_data/."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, x_split, y_split in (
        ("train", dataset.x_train, dataset.y_train),
        ("val", dataset.x_val, dataset.y_val),
        ("test", dataset.x_test, dataset.y_test),
    ):
        combined = x_split.copy()
        combined[TARGET_COLUMN] = y_split.values
        combined.to_csv(out_dir / f"adult_{split_name}.csv", index=False)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(dataset.stats(), f, indent=2)


if __name__ == "__main__":
    dataset = load_adult_dataset()
    save_processed(dataset)
    print(json.dumps(dataset.stats(), indent=2))
