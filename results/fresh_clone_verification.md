# Fresh clone reproducibility test

To confirm the project is reproducible by someone else from scratch, the
repository was cloned to a separate temp directory (simulating a fresh
`git clone`), and the full pipeline was run with no manual intervention:

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python src/data_loader.py
python src/train.py --model-name xgboost_adult_final \
    --max-depth 5 --eta 0.1 --reg-lambda 2.0 --gamma 1.0 \
    --subsample 0.9 --colsample-bytree 0.7 --num-boost-round 400
python src/evaluate.py --model-name xgboost_adult_final
```

## Timing

| Stage        | Time |
|--------------|-----:|
| `pip install -r requirements.txt` | ~4 min (dominated by downloading xgboost/lightgbm/catboost wheels) |
| `data_loader.py` (download + preprocess) | 2s |
| `train.py` (400 rounds, early-stopped at 141) | 3s |
| `evaluate.py` | 1s |
| **Total (excluding one-time dependency install)** | **6s** |

## Result match

| Metric        | Original run | Fresh clone run | Match |
|----------------|-------------:|-----------------:|:-----:|
| Test accuracy  | 0.8753074274471225 | 0.8753074274471225 | Exact |
| Test AUC       | 0.9294437808057419 | 0.9294437808057419 | Exact |
| Best iteration | 141                 | 141                 | Exact |

Results are bit-for-bit identical between the original run and the fresh
clone, confirming the fixed random seed (`RANDOM_STATE = 42` in
`src/data_loader.py`, `seed=42` in `model.XGBoostConfig`) and
deterministic `tree_method="hist"` produce a fully reproducible pipeline
end-to-end, with no manual data download or hidden state required.
