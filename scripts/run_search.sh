#!/usr/bin/env bash
# Run the full Optuna sweep across all six families.
# Spec-faithful defaults: 200 epochs per trial, FULL search caps, 100 trials for
# the priority families (LSTM, GRU, TCN, CNN+LSTM) and 40 for the baselines
# (MLP, CNN1D). Total: 480 trials, ~8-14 hours on the RTX 5070 Ti.
#
# Override via env vars for shorter sweeps, e.g.:
#   EPOCHS=60 N_PRIORITY=15 N_BASELINE=10 CAPS=lite bash scripts/run_search.sh
#
# Live progress is written to experiments/search/STATUS.json after every eval
# cycle (typically every 5 epochs). `cat experiments/search/STATUS.json` to see
# current trial / family / holdout / epoch / loss / elapsed time.

set -e
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-200}"
N_PRIORITY="${N_PRIORITY:-100}"
N_BASELINE="${N_BASELINE:-40}"
CAPS="${CAPS:-full}"           # "full" or "lite"

if [[ "$CAPS" == "full" ]]; then
    CAPS_FLAG="--full-caps"
else
    CAPS_FLAG=""
fi

START_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[search] start=$START_TS  epochs=$EPOCHS  priority=$N_PRIORITY  baseline=$N_BASELINE  caps=$CAPS"

for f in lstm gru tcn cnn_lstm; do
    echo "=== $f (priority, $N_PRIORITY trials) ==="
    python scripts/run_optuna_search.py --family "$f" --n-trials "$N_PRIORITY" --epochs "$EPOCHS" $CAPS_FLAG
done

for f in mlp cnn1d; do
    echo "=== $f (baseline, $N_BASELINE trials) ==="
    python scripts/run_optuna_search.py --family "$f" --n-trials "$N_BASELINE" --epochs "$EPOCHS" $CAPS_FLAG
done

END_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[search] all families complete  start=$START_TS  end=$END_TS"
