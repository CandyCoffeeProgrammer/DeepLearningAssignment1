# Santa Fe Laser — Recursive 200-step Forecasting

Utrecht University Deep Learning assignment. Team project.

## Goal
Predict 200 future points of the Santa Fe laser dataset (chaotic far-infrared NH3 laser, dataset A from the 1991 prediction competition) given the 1000 training points in `data/Xtrain.mat`. Lowest MAE/MSE on the held-out test set wins.

## Setup
```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

CUDA 12.8 wheels are required for Blackwell GPUs (RTX 50-series). For CPU-only, drop the `--extra-index-url`.

## Layout
```
data/                Xtrain.mat lives here
src/                 library code (data, models, losses, train, predict, search, evaluate, utils)
configs/             YAML configs; `base.yaml` is the default
experiments/         all run output: checkpoints, logs, plots, reports
notebooks/           EDA
scripts/             entry-point scripts (run_search.sh, run_final.sh, make_test_evaluation.py)
```

## Usage (planned)
| Command | What it does |
| --- | --- |
| `python -m src.search --config configs/base.yaml` | Run the Optuna sweep across the three synthetic holdouts |
| `python -m src.predict --final` | Retrain best ensemble on full Xtrain and dump predictions |
| `python scripts/make_test_evaluation.py --xtest data/Xtest.mat` | Score saved predictions once the real test arrives |

## Validation protocol
We mirror the real test by holding out three 200-pt segments from Xtrain:

| Holdout | Test segment | Trained on | Inner validation |
| --- | --- | --- | --- |
| A | 801–1000 | 1–800 | 701–800 |
| B | 701–900 | 1–700 | 601–700 |
| C | 601–800 | 1–600 | 501–600 |

For each, the model receives the last `window` points of its training portion and recursively predicts 200 steps. Final ranking metric: mean recursive 200-step MAE across A/B/C in the original integer scale (2–255).

After the winning configuration / ensemble is picked, every member is retrained on the full `Xtrain` (1–1000) before generating the real-test prediction.
