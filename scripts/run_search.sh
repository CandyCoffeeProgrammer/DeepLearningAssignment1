#!/usr/bin/env bash
# Run the full Optuna sweep across all six families.
# Defaults: 4 priority x 15 trials + 2 baseline x 10 trials = 80 trials, 60 epochs each.
# Uses LITE_SEARCH_CAPS in src/search.py (capped hidden_dim, num_layers, etc.) so trials
# stay under ~30 s on the RTX 5070 Ti, total sweep ~20 min. For the spec-faithful budget
# (100/40 trials, 200 epochs, full caps) override the env vars and edit src/search.py
# to use FULL_SEARCH_CAPS.

set -e
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-60}"
N_PRIORITY="${N_PRIORITY:-15}"
N_BASELINE="${N_BASELINE:-10}"

echo "[search] epochs=$EPOCHS  priority=$N_PRIORITY  baseline=$N_BASELINE"

for f in lstm gru tcn cnn_lstm; do
    echo "=== $f (priority, $N_PRIORITY trials) ==="
    python scripts/run_optuna_search.py --family "$f" --n-trials "$N_PRIORITY" --epochs "$EPOCHS"
done

for f in mlp cnn1d; do
    echo "=== $f (baseline, $N_BASELINE trials) ==="
    python scripts/run_optuna_search.py --family "$f" --n-trials "$N_BASELINE" --epochs "$EPOCHS"
done

echo "[search] all families complete"
