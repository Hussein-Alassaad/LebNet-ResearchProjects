# Tech Fellows Project: XGBoost Reproduction + Extension

Reproduction and extension of **"XGBoost: A Scalable Tree Boosting System"**
(Chen & Guestrin, KDD 2016) on the UCI Adult Census Income dataset.

> Status: Phase 0 (setup). This README will be filled in with setup
> instructions, results, and an extension summary as the project progresses.

## Project layout

```
/
├── README.md
├── requirements.txt
├── data/               # Raw data (downloaded, not committed)
├── processed_data/     # Cleaned/processed data
├── models/             # Saved model checkpoints
├── src/
│   ├── data_loader.py  # Data loading & preprocessing
│   ├── model.py        # Model architecture/config
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation & metrics
│   └── utils.py        # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_results.ipynb
│   └── 03_extension_analysis.ipynb
├── results/             # Outputs, plots, metrics
└── docs/
    ├── paper_summary.md    # Understanding of the paper
    ├── extension_plan.md   # Planned extension
    └── findings.md         # Final results & analysis
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

## Extension

_Coming in Phase 2._

## Results

See [results/reproduction_results.md](results/reproduction_results.md)
for the full write-up, and
[results/reproduction_comparison_notes.md](results/reproduction_comparison_notes.md)
for the gap analysis against reference numbers.

**Final test set:** 87.53% accuracy, 0.9294 AUC — within the commonly
reported reference range for gradient boosted trees on Adult Census
Income (~86-87.5% accuracy, ~0.92-0.93 AUC).
