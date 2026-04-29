# Three-holdout evaluation — LSTM / all_tricks

Seeds [0, 1, 2] • 200 epochs • anneal=150 • window 30 • hidden 64 × 2 layers.

**Overall mean test MAE: 35.946** (worst holdout: C @ 53.871)

Per-holdout (recursive 200-step on the test segment, original 2–255 scale):

| holdout | mean MAE | std | best | worst | mean MSE | mean NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 28.17 | 16.54 | 5.09 | 43.00 | 2559.7 | 1.289 |
| B | 25.79 | 0.24 | 25.47 | 26.04 | 989.2 | 1.041 |
| C | 53.87 | 1.32 | 52.01 | 54.92 | 4549.9 | 6.691 |

Per-run breakdown:

| holdout | seed | best_epoch | inner-val MAE | test MAE | test MSE | test NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 85 | 3.91 | 43.00 | 3963.8 | 1.9965 |
| A | 1 | 145 | 7.45 | 36.42 | 3614.6 | 1.8206 |
| A | 2 | 175 | 1.75 | 5.09 | 100.5 | 0.0506 |
| B | 0 | 15 | 17.21 | 25.88 | 966.3 | 1.0164 |
| B | 1 | 5 | 16.97 | 25.47 | 1044.3 | 1.0985 |
| B | 2 | 20 | 17.49 | 26.04 | 957.2 | 1.0068 |
| C | 0 | 95 | 33.15 | 54.68 | 4777.3 | 7.0251 |
| C | 1 | 65 | 48.02 | 54.92 | 4630.6 | 6.8092 |
| C | 2 | 90 | 20.83 | 52.01 | 4241.9 | 6.2377 |

## Reading the per-holdout split

This is the first time we're seeing the three holdouts side by side, and the
per-holdout structure is much more interesting than the overall mean.

- **Holdout A (test 801..1000)** — mostly stable oscillations. Variance
  dominates: seed 2 finds an excellent rollout (MAE 5.09) while seeds 0/1
  collapse early (36–43). This is the holdout we already characterised in
  milestone 3 as high-seed-noise.
- **Holdout B (test 701..900)** — train 1..700 contains the second collapse
  (~600), and the test segment is post-collapse stable oscillation. The
  model converges to nearly identical quality across all three seeds
  (std 0.24). It is also reliably the *easy* holdout, and unrepresentative
  of the worst case.
- **Holdout C (test 601..800)** — train 1..600 is everything pre-collapse.
  The test segment opens with the collapse at ~600 and recovers afterwards.
  All three seeds get stuck at ~52–55 MAE; there is no "lucky" seed here.
  The model has no way to predict a regime change it has never seen.

The mean across holdouts is dragged up by C. The **worst-case holdout is
the right thing to track**, since Holdout A on its own can be misleading
(highest variance, sometimes very good). For the search objective we should
report mean *and* worst-case.

## What this tells us for the next milestones

- **Ensembling matters disproportionately on Holdout A**: averaging seed 2
  (5.09) with seeds 0/1 (~40) immediately beats any single seed. We should
  use median, not mean, to be robust to the seed-0/1-style collapses.
- **Holdout C is the bottleneck**. We need configurations that can handle
  the collapse — likely larger windows (so the model has seen the
  pre-collapse behaviour for several cycles), or model families with a
  longer effective receptive field (TCN). Optuna search will help here.
- **Inner-val MAE is misleading on C** — values of 21–48 still corresponded
  to ~52 test MAE. The early-stopping signal on C is unreliable; the rollout
  collapses regardless. This is worth noting but not actionable yet.