# Milestone 7 — ensembling across the three holdouts

Members (6): `mlp_best`, `cnn_lstm_best`, `lstm_best`, `tcn_best`, `gru_best`, `cnn1d_best`
Seeds: [0, 1, 2, 3, 4] (so 30 ensemble members per holdout).

**Winning strategy by mean MAE across A/B/C: `weighted` (22.170)**

| holdout | strategy | MAE | MSE | NMSE | n_members |
| --- | --- | ---: | ---: | ---: | ---: |
| A | mean | 23.17 | 1191.4 | 0.600 | 30 |
| A | median | 20.98 | 1294.0 | 0.652 | 30 |
| A | weighted | 20.53 | 986.7 | 0.497 | 30 |
| A | best_individual (gru_best/s2) | 6.06 | 108.4 | — | 1 |
| B | mean | 22.62 | 845.5 | 0.889 | 30 |
| B | median | 23.51 | 879.7 | 0.925 | 30 |
| B | weighted | 22.66 | 841.0 | 0.885 | 30 |
| B | best_individual (cnn_lstm_best/s1) | 15.92 | 440.8 | — | 1 |
| C | mean | 21.51 | 770.7 | 1.133 | 30 |
| C | median | 29.49 | 1315.9 | 1.935 | 30 |
| C | weighted | 23.32 | 896.1 | 1.318 | 30 |
| C | best_individual (tcn_best/s4) | 20.78 | 961.1 | — | 1 |

## Strategy mean across holdouts

| strategy | mean test MAE |
| --- | ---: |
| weighted | 22.170 |
| mean | 22.434 |
| median | 24.663 |