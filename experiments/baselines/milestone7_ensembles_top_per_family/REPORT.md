# Milestone 7 — ensembling across the three holdouts

Members (6): `gru_best`, `tcn_best`, `mlp_best`, `cnn1d_best`, `cnn_lstm_best`, `lstm_best`
Seeds: [0, 1, 2] (so 18 ensemble members per holdout).

**Winning strategy by mean MAE across A/B/C: `mean` (24.914)**

| holdout | strategy | MAE | MSE | NMSE | n_members |
| --- | --- | ---: | ---: | ---: | ---: |
| A | mean | 27.62 | 1518.8 | 0.765 | 18 |
| A | median | 24.92 | 1498.9 | 0.755 | 18 |
| A | weighted | 26.89 | 1586.9 | 0.799 | 18 |
| A | best_individual (mlp_best/s0) | 11.93 | 640.7 | — | 1 |
| B | mean | 22.65 | 824.3 | 0.867 | 18 |
| B | median | 24.11 | 901.4 | 0.948 | 18 |
| B | weighted | 22.85 | 829.7 | 0.873 | 18 |
| B | best_individual (mlp_best/s1) | 11.38 | 287.4 | — | 1 |
| C | mean | 24.47 | 970.7 | 1.427 | 18 |
| C | median | 28.51 | 1331.6 | 1.958 | 18 |
| C | weighted | 26.24 | 1135.2 | 1.669 | 18 |
| C | best_individual (cnn1d_best/s1) | 22.04 | 972.8 | — | 1 |

## Strategy mean across holdouts

| strategy | mean test MAE |
| --- | ---: |
| mean | 24.914 |
| weighted | 25.327 |
| median | 25.844 |