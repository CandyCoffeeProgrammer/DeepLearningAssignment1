# Milestone 2 — Holdout A baseline (one-step training, no tricks)

End-to-end pipeline check: `BaseForecaster` interface working, `MLP` and `LSTM`
implementations trained with plain one-step MSE, recursive 200-step prediction
on Holdout A.

## Setup
- Holdout A: train = points 1–700 · inner-val = 701–800 · test = 801–1000
- Window 30 · MinMax to [0, 1] · seed 42 · 200 epochs (no early stopping triggered)
- AdamW lr=1e-3 wd=1e-5 · OneCycleLR · grad_clip=1.0
- One-step MSE only — no multi-step loss, no scheduled sampling, no input noise

## Numbers (recursive 200-step on the test segment, original 2–255 scale)
| model | params | best epoch | inner-val MAE (100-step) | test MAE | test MSE | NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP   |  6209 | 115 | 2.56  | **14.65**  | 1060.3 | 0.534 |
| LSTM  | 50497 |  75 | 4.85  | **50.65**  | 4967.8 | 2.502 |

## Reading the result

The most important takeaway is the **gap between inner-val MAE and test MAE**:

- MLP: 2.56 → 14.65   (×5.7 worse over 200 steps vs 100)
- LSTM: 4.85 → 50.65  (×10.4 worse, NMSE > 1 = worse than predicting the mean)

That gap is the one-step-training / recursive-rollout mismatch in pure form.
Inner-val (100 steps) sits inside the radius where compounding errors haven't
yet pushed the trajectory off-manifold. By 200 steps, the LSTM's predictions
have drifted into a region of the state space it never saw during training, so
the rollout collapses.

This is exactly what milestone 3 is meant to cure:

- **Multi-step training loss** (the single biggest lever): unroll the model
  during training and put loss on the unrolled predictions.
- **Scheduled sampling**: feed the model's own outputs back during training so
  it learns to recover from its own mistakes.
- **Input noise**: small Gaussian noise on inputs improves robustness.

So this baseline is the floor we're trying to beat. Expected ranges with
milestone-3 tricks (per literature on this dataset): MLP ~8–12 MAE, LSTM ~6–10
MAE, with NMSE well below 1.

## Files
- `mlp_A_prediction.png`, `lstm_A_prediction.png` — overlay + residual plots
- `*_pred.npy` — 200-step predictions in original scale
- `*_history.json` — per-epoch train loss + (every 5 epochs) inner-val metrics
- `summary.json` — machine-readable metrics
- `config.yaml` — exact config used
- `run.log` — full training log

## How to reproduce
```bash
python scripts/run_holdout_a.py --out-dir experiments/baselines/milestone2_holdout_a
```
