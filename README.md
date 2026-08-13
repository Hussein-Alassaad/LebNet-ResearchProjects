# Tech Fellows Project: XGBoost Reproduction + Extension

Reproduction and extension of **"XGBoost: A Scalable Tree Boosting System"**
(Chen & Guestrin, KDD 2016) on the UCI Adult Census Income dataset.

> Status: Complete — reproduction (Phase 1) and extension (Phase 2)
> both finished. See Results and Extension below.

## Project layout

```
/
├── README.md
├── requirements.txt
├── data/                     # Raw data (downloaded, not committed)
├── processed_data/           # Cleaned/processed data
├── models/                   # Saved model checkpoints
├── src/
│   ├── data_loader.py        # Data loading, cleaning, dedup, splitting
│   ├── model.py               # XGBoostConfig/XGBoostModel (reproduction)
│   ├── extension_models.py    # LightGBM/CatBoost wrappers (extension)
│   ├── train.py                # Training script (reproduction)
│   ├── evaluate.py             # Evaluation & metrics
│   ├── benchmark_extension.py  # Accuracy + efficiency benchmark (extension)
│   ├── utils.py                 # Logging, seeding, JSON helpers
│   ├── test_model_forward_pass.py   # Sanity test for model.py
│   └── test_reproducibility.py      # End-to-end pipeline tests
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset exploration
│   ├── 02_baseline_results.ipynb    # Reproduction training curves/metrics
│   └── 03_extension_analysis.ipynb  # Extension accuracy/efficiency plots
├── results/                  # Metrics, plots, written analysis
└── docs/
    ├── paper_summary.md      # Understanding of the paper
    ├── hyperparameters.md    # Full hyperparameter/config reference
    ├── extension_plan.md     # Extension proposal
    └── findings.md           # Final learnings & reflection
```

## Paper

See [docs/paper_summary.md](docs/paper_summary.md) for the paper reference,
core idea, and why it was chosen.

## Setup & reproduction

### Requirements

- Python 3.11+ (no GPU required — trains in seconds on CPU)
- ~200 MB disk for dependencies, a few MB for the cached dataset

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download & preprocess the data

```bash
python src/data_loader.py
```

Downloads the UCI Adult Census Income dataset (via OpenML) to
`data/adult_raw.csv`, cleans it, deduplicates it (see note below), and
writes train/val/test CSVs + summary stats to `processed_data/`.

### 3. Train the model

```bash
python src/train.py --model-name xgboost_adult_final \
    --max-depth 5 --eta 0.1 --reg-lambda 2.0 --gamma 1.0 \
    --subsample 0.9 --colsample-bytree 0.7 --num-boost-round 400
```

This is the exact configuration used for the final reported results (see
`results/reproduction_results.md`). Run `python src/train.py --help` for
all options; omitting flags uses `model.XGBoostConfig`'s defaults.
Training completes in a few seconds on a laptop CPU. The trained model
is saved to `models/<model-name>.json`, periodic checkpoints to
`models/checkpoints/<model-name>/`, and a run summary + training curves
to `results/<model-name>_train_run.json` / `_evals_result.json`.

### 4. Evaluate

```bash
python src/evaluate.py --model-name xgboost_adult_final
```

Prints validation/test accuracy, error rate, AUC, log loss, and F1, and
writes them to `results/<model-name>_metrics.json`.

### 5. Explore the notebooks

```bash
python -m ipykernel install --user --name lebnet-venv
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` — dataset shape, missingness, class
  balance, feature distributions.
- `02_baseline_results.ipynb` — training curves and metrics across the
  baseline and tuning attempts.
- `03_extension_analysis.ipynb` — extension results (Phase 2/3).

### Notes on reproducibility

- All random splits use a fixed seed (`RANDOM_STATE = 42` in
  `src/data_loader.py`), so re-running `data_loader.py` from a clean
  cache produces identical splits.
- The raw dataset contains ~52 exact-duplicate rows; `data_loader.py`
  deduplicates before splitting to prevent train/test leakage (see
  `results/reproduction_results.md` for details).
- Re-downloading: delete `data/adult_raw.csv` (or pass
  `force_redownload=True` to `load_adult_dataset`) to refetch from
  OpenML.

### Running tests

```bash
pytest src/ -v
```

Covers data loading (split sizes, no train/test leakage, dtypes),
a training step (AUC beats random baseline, save/load round-trip is
exact), and evaluation metric correctness. All 9 tests pass in ~2
seconds on CPU.

## Extension

**Efficiency comparison: XGBoost vs. LightGBM vs. CatBoost.** The
original paper frames XGBoost as both accurate *and* systems-efficient;
this extension tests that framing directly by running the two major
libraries that followed XGBoost (LightGBM, CatBoost) on the identical
data/splits with a matched capacity/regularization budget, and comparing
accuracy, training time, inference time, and model size. See
[docs/extension_plan.md](docs/extension_plan.md) for the full proposal
and [results/extension_results.md](results/extension_results.md) for
results and analysis.

```bash
python src/benchmark_extension.py
```

**TL;DR:** accuracy/AUC are effectively tied across all three libraries;
LightGBM trains fastest, CatBoost is far slower to train (ordered
boosting overhead) but fastest at inference (oblivious trees), and
XGBoost is the balanced generalist — never fastest on either axis, but
never far behind.

## Results

See [results/reproduction_results.md](results/reproduction_results.md)
for the full reproduction write-up,
[results/reproduction_comparison_notes.md](results/reproduction_comparison_notes.md)
for the gap analysis against reference numbers, and
[docs/hyperparameters.md](docs/hyperparameters.md) for every
hyperparameter used across all training runs.

**Reproduction (final test set):** 87.53% accuracy, 0.9294 AUC — within
the commonly reported reference range for gradient boosted trees on
Adult Census Income (~86-87.5% accuracy, ~0.92-0.93 AUC).

**Extension:** see
[results/extension_results.md](results/extension_results.md) for the
full accuracy/efficiency comparison table and analysis.

See [docs/findings.md](docs/findings.md) for reflections on the project
as a whole.

## Future work

- Extend the efficiency comparison to a larger dataset (Adult's ~29K
  training rows is small relative to what these libraries are built
  for; LightGBM's edge in particular is expected to widen at scale).
- Try a fairness/bias audit across the `sex` and `race` columns in
  Adult Census Income, a natural follow-up given the dataset's
  demographic features and the extension's finding that all three
  libraries reach similar aggregate accuracy (aggregate parity does not
  imply subgroup parity).
- Tune LightGBM's and CatBoost's hyperparameters independently (rather
  than a matched budget) to see each library's best achievable
  accuracy, not just its accuracy under a shared constraint.
