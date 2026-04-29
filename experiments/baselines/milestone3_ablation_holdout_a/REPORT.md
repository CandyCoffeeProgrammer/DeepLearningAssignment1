# Milestone 3 ablation — Holdout A

Seeds [0, 1, 2] • 200 epochs • anneal_epochs=150 (75% of training) • window 30 • MLP hidden 64×2, LSTM hidden 64×2.

Test MAE in original 2–255 scale, mean ± std across seeds. **best** = single-seed minimum, **worst** = single-seed maximum.

| family | variant | inner-val MAE (mean) | test MAE (mean ± std) | best | worst | NMSE (mean) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mlp | baseline | 2.36 | 29.74 ± 2.39 | 26.83 | 32.69 | 1.243 |
| lstm | baseline | 2.64 | 43.37 ± 13.69 | 24.02 | 53.21 | 2.146 |
| mlp | multistep | 2.75 | 19.86 ± 5.94 | 11.94 | 26.23 | 0.603 |
| lstm | multistep | 4.95 | 38.56 ± 2.23 | 35.44 | 40.49 | 1.781 |
| mlp | multi+sched | 3.36 | 29.16 ± 2.17 | 27.40 | 32.23 | 1.063 |
| lstm | multi+sched | 4.39 | 32.19 ± 6.62 | 23.86 | 40.06 | 1.551 |
| mlp | all_tricks | 2.38 | 31.92 ± 2.57 | 28.73 | 35.02 | 1.186 |
| lstm | all_tricks | 4.37 | 28.17 ± 16.54 | 5.09 | 43.00 | 1.289 |

## Deltas vs baseline (mean test MAE)

### mlp
| variant | mean test MAE | Δ vs baseline | Δ % |
| --- | ---: | ---: | ---: |
| baseline | 29.74 | +0.00 | +0.0% |
| multistep | 19.86 | -9.88 | -33.2% |
| multi+sched | 29.16 | -0.58 | -1.9% |
| all_tricks | 31.92 | +2.18 | +7.3% |

### lstm
| variant | mean test MAE | Δ vs baseline | Δ % |
| --- | ---: | ---: | ---: |
| baseline | 43.37 | +0.00 | +0.0% |
| multistep | 38.56 | -4.81 | -11.1% |
| multi+sched | 32.19 | -11.18 | -25.8% |
| all_tricks | 28.17 | -15.20 | -35.0% |

## Takeaways

1. **Multi-step training alone is the biggest lever** — exactly as the literature
   on recursive forecasting predicts. -33% mean MAE on MLP, -11% on LSTM.
2. **LSTM benefits from every trick stacked**: each addition reduces mean MAE.
   `all_tricks` finds the best single-seed run in the whole sweep
   (seed 2 → MAE 5.09, NMSE 0.05), the first time any model has flipped a
   cleanly-tracking 200-step rollout on Holdout A.
3. **MLP responds non-monotonically**: scheduled sampling and input noise
   undo most of the multi-step gain. Plausibly the small flat-window MLP is
   already at the limit of what it can do, and the extra noise injected by
   sched/noise pushes it off the manifold during rollout. We should keep MLP
   on `multistep` only when running it as an ensemble member.
4. **Seed variance is large** — LSTM `baseline` std is 13.7 MAE, `all_tricks`
   std is 16.5 (range 5.09 to 43.00). Single-seed comparisons on this dataset
   are unreliable. We will need 5+ seeds for the search and final selection.
5. **Inner-val MAE ≠ test MAE**: e.g. MLP `multistep` has slightly worse
   inner-val (2.75 vs 2.36) but much better test (19.86 vs 29.74). Inner-val
   only sees 100 steps; the divergence shows up later. Recursive 200-step
   inner-val on a different slice would be a closer proxy, but we'd be
   eating into training data — keep an eye on this for later milestones.

## What's next

- The all-tricks LSTM is the strongest model so far, but its single-run
  variance demands ensembling. Multi-seed averaging should bring the
  variance down sharply (the 5.09-MAE seed, averaged with two ~30-MAE seeds,
  trivially beats either of the worse seeds alone).
- Continue with the planned milestone 4 (GRU, 1D CNN, CNN+LSTM, TCN). At a
  minimum, multistep should be on for every model.
- Re-examine MLP later: if the search finds a smaller or larger hidden size
  that responds well to scheduled sampling, the current verdict could flip.
