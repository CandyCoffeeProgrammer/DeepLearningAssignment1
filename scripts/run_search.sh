#!/usr/bin/env bash
# Run the full Optuna sweep across all six families.
# 4 priority families x 30 trials + 2 baseline families x 20 trials = 160 trials total.
# At ~100 epochs and ~12-15s per trial on the RTX 5070 Ti, ~35-45 min total.
# Adjust EPOCHS / N_PRIORITY / N_BASELINE for shorter or longer sweeps.

set -e
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-100}"
N_PRIORITY="${N_PRIORITY:-30}"
N_BASELINE="${N_BASELINE:-20}"

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
