# Milestone 4 — All 6 model families on Holdout A (multistep only)

Seeds [0, 1, 2] • 200 epochs • anneal_epochs=150 (75% of training) • window 30 •
all defaults: hidden 64, num_layers 2 (3 for cnn1d), dropout 0.1, kernel_size 3.

Multistep variant only (k_max=5, p_max=0, input_noise=0). The goal here is
"do the new families train and produce sensible numbers" — not to crown a
winner. Multistep was chosen because milestone 3 showed it's the safest
single trick; the search later will tune everything per family.

Test MAE in original 2–255 scale, mean ± std across seeds.

| family | inner-val MAE (mean) | test MAE (mean ± std) | best | worst | NMSE (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp      | 2.75  | **19.86 ± 5.94** | 11.94 | 26.23 | 0.603 |
| tcn      | 3.43  | 27.73 ± 6.79     | 18.80 | 35.25 | 0.963 |
| cnn1d    | 14.31 | 28.09 ± 1.47     | 26.20 | 29.78 | 0.746 |
| lstm     | 4.95  | 38.56 ± 2.23     | 35.44 | 40.49 | 1.781 |
| cnn_lstm | 3.31  | 39.04 ± 3.48     | 36.16 | 43.95 | 1.633 |
| gru      | 5.62  | 43.21 ± 9.89     | 29.58 | 52.76 | 2.091 |

## Reading the numbers

- **All six families train and roll out cleanly** end-to-end. No NaNs, no
  silent shape bugs, no gradient explosions. That's the milestone-4 bar.
- **MLP wins this slice** (best mean and best single-seed run for multistep).
  This is misleading in isolation: milestone 3 showed the LSTM benefits much
  more from the additional tricks (sched/noise), reaching MAE 5.09 single-seed
  with `all_tricks`. Don't conclude "MLP is the strongest model" — conclude
  "MLP responds best to multistep alone".
- **TCN is the standout new family**: 18.80 best single-seed and a
  reasonably tight std. Worth pushing through the full trick stack and
  search.
- **CNN1D has a high inner-val MAE (14.31) but a normal test MAE (28.09)**.
  The GAP head produces a smoothed prediction that's bad at short horizons
  and fine at long horizons. Single-seed std is the lowest in the table —
  this model is *consistent* but not strong.
- **GRU lands very close to LSTM in mean** but with much larger variance.
  Effectively the same architecture for our purposes; we'll keep both as
  ensemble candidates.

## Decisions

- All six families pass into milestone 5 (three-holdout protocol) with no
  changes to the default hyperparameters.
- We do not declare any single family a winner here — that's the search's job.
  The current best single configuration overall is still **LSTM all_tricks
  seed 2 (MAE 5.09)** from milestone 3.

## How to reproduce
```bash
python scripts/ablation_holdout_a.py \
    --variants multistep \
    --families mlp lstm gru cnn1d cnn_lstm tcn \
    --out-dir experiments/baselines/milestone4_all_families_holdout_a
```
