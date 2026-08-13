"""Small shared helpers used across the training/evaluation scripts."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int) -> None:
    """Seed Python and numpy RNGs for reproducible splits/sampling."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a single stream handler attached.

    Guards against duplicate handlers so repeated calls (e.g. re-running
    a cell in a notebook) don't produce duplicated log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: dict, path: Path) -> None:
    """Write `obj` as pretty-printed JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    """Read a JSON file previously written by `save_json`."""
    with open(path) as f:
        return json.load(f)
