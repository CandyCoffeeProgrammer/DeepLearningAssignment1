# Final summary — Santa Fe laser recursive forecast

Lowest MAE / MSE on the held-out 200-pt test set wins. This file collects
the search leaderboard, the chosen ensemble, and the expected performance
based on synthetic holdouts. Once `Xtest.mat` arrives, run
`python scripts/make_test_evaluation.py --xtest data/Xtest.mat` to score.

## Leaderboard — top 20 single models across all families

| rank | family | trial | mean MAE | worst MAE | A | B | C | window | hidden | k_max | p_max | noise |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mlp | 25 | 15.747 | 21.372 | 10.68 | 15.19 | 21.37 | 20 | 32 | 5 | 0.5 | 0.01 |
| 2 | cnn_lstm | 63 | 17.723 | 29.821 | 8.68 | 14.67 | 29.82 | 50 | 128 | 3 | 0.0 | 0.01 |
| 3 | lstm | 130 | 18.402 | 27.617 | 2.82 | 24.77 | 27.62 | 75 | 256 | 10 | 0.5 | 0.02 |
| 4 | tcn | 54 | 19.041 | 25.074 | 13.78 | 25.07 | 18.27 | 40 | 64 | 5 | 0.25 | 0.0 |
| 5 | gru | 88 | 19.334 | 25.410 | 19.74 | 25.41 | 12.86 | 20 | 256 | 7 | 0.5 | 0.005 |
| 6 | cnn_lstm | 71 | 19.582 | 31.311 | 9.01 | 18.42 | 31.31 | 30 | 256 | 3 | 0.5 | 0.0 |
| 7 | lstm | 73 | 19.976 | 25.548 | 13.67 | 25.55 | 20.71 | 25 | 256 | 10 | 0.25 | 0.01 |
| 8 | lstm | 92 | 20.011 | 31.422 | 2.61 | 26.00 | 31.42 | 75 | 256 | 1 | 0.5 | 0.02 |
| 9 | gru | 48 | 20.220 | 35.343 | 2.19 | 23.12 | 35.34 | 25 | 128 | 10 | 0.5 | 0.02 |
| 10 | mlp | 20 | 20.363 | 23.308 | 21.75 | 23.31 | 16.04 | 30 | 64 | 5 | 0.5 | 0.005 |
| 11 | cnn_lstm | 98 | 20.462 | 25.521 | 11.34 | 25.52 | 24.53 | 50 | 64 | 1 | 0.0 | 0.0 |
| 12 | tcn | 93 | 20.494 | 22.601 | 20.07 | 22.60 | 18.81 | 25 | 128 | 3 | 0.5 | 0.005 |
| 13 | tcn | 92 | 20.605 | 22.505 | 21.40 | 22.51 | 17.91 | 50 | 128 | 5 | 0.5 | 0.0 |
| 14 | cnn_lstm | 76 | 20.738 | 48.195 | 4.95 | 9.07 | 48.19 | 50 | 64 | 10 | 0.25 | 0.01 |
| 15 | mlp | 37 | 20.925 | 25.455 | 13.20 | 25.46 | 24.12 | 15 | 256 | 3 | 0.5 | 0.005 |
| 16 | lstm | 5 | 20.948 | 25.957 | 12.49 | 25.96 | 24.40 | 75 | 256 | 10 | 0.25 | 0.0 |
| 17 | mlp | 17 | 20.973 | 24.738 | 23.11 | 24.74 | 15.07 | 15 | 256 | 5 | 0.5 | 0.005 |
| 18 | mlp | 21 | 21.007 | 26.072 | 26.07 | 23.62 | 13.33 | 30 | 64 | 5 | 0.5 | 0.01 |
| 19 | lstm | 91 | 21.026 | 36.329 | 2.50 | 24.25 | 36.33 | 75 | 256 | 1 | 0.0 | 0.02 |
| 20 | gru | 111 | 21.039 | 31.091 | 5.24 | 26.78 | 31.09 | 50 | 256 | 7 | 0.5 | 0.02 |

## Best per family

| family | best mean MAE | worst MAE | per-holdout |
| --- | ---: | ---: | --- |
| mlp | 15.747 | 21.372 | A=10.68, B=15.19, C=21.37 |
| cnn_lstm | 17.723 | 29.821 | A=8.68, B=14.67, C=29.82 |
| lstm | 18.402 | 27.617 | A=2.82, B=24.77, C=27.62 |
| tcn | 19.041 | 25.074 | A=13.78, B=25.07, C=18.27 |
| gru | 19.334 | 25.410 | A=19.74, B=25.41, C=12.86 |
| cnn1d | 25.110 | 30.544 | A=30.54, B=26.67, C=18.11 |

## Ensemble strategy comparison

Source: `experiments\baselines\full_sweep_ensemble`

| strategy | mean test MAE across A/B/C |
| --- | ---: |
| weighted | 22.170 |
| mean | 22.434 |
| median | 24.663 |

**Winning strategy: `weighted`**


## Final ensemble (retrained on full Xtrain)

Generated at: `2026-05-01T17:17:13.928133Z`
Strategy: **weighted**, members: **30**, horizon: 200
Forecast range: min=13.43  max=173.62  mean=60.10  std=24.77

| tag | family | window | epochs | scaler |
| --- | --- | ---: | ---: | --- |
| mlp_best_s0 | mlp | 20 | 115 | minmax_neg1_1 |
| mlp_best_s1 | mlp | 20 | 115 | minmax_neg1_1 |
| mlp_best_s2 | mlp | 20 | 115 | minmax_neg1_1 |
| mlp_best_s3 | mlp | 20 | 115 | minmax_neg1_1 |
| mlp_best_s4 | mlp | 20 | 115 | minmax_neg1_1 |
| cnn_lstm_best_s0 | cnn_lstm | 50 | 115 | standard |
| cnn_lstm_best_s1 | cnn_lstm | 50 | 115 | standard |
| cnn_lstm_best_s2 | cnn_lstm | 50 | 115 | standard |
| cnn_lstm_best_s3 | cnn_lstm | 50 | 115 | standard |
| cnn_lstm_best_s4 | cnn_lstm | 50 | 115 | standard |
| lstm_best_s0 | lstm | 75 | 145 | standard |
| lstm_best_s1 | lstm | 75 | 145 | standard |
| lstm_best_s2 | lstm | 75 | 145 | standard |
| lstm_best_s3 | lstm | 75 | 145 | standard |
| lstm_best_s4 | lstm | 75 | 145 | standard |
| tcn_best_s0 | tcn | 40 | 70 | minmax_0_1 |
| tcn_best_s1 | tcn | 40 | 70 | minmax_0_1 |
| tcn_best_s2 | tcn | 40 | 70 | minmax_0_1 |
| tcn_best_s3 | tcn | 40 | 70 | minmax_0_1 |
| tcn_best_s4 | tcn | 40 | 70 | minmax_0_1 |
| gru_best_s0 | gru | 20 | 60 | standard |
| gru_best_s1 | gru | 20 | 60 | standard |
| gru_best_s2 | gru | 20 | 60 | standard |
| gru_best_s3 | gru | 20 | 60 | standard |
| gru_best_s4 | gru | 20 | 60 | standard |
| cnn1d_best_s0 | cnn1d | 25 | 150 | minmax_0_1 |
| cnn1d_best_s1 | cnn1d | 25 | 150 | minmax_0_1 |
| cnn1d_best_s2 | cnn1d | 25 | 150 | minmax_0_1 |
| cnn1d_best_s3 | cnn1d | 25 | 150 | minmax_0_1 |
| cnn1d_best_s4 | cnn1d | 25 | 150 | minmax_0_1 |

## Expected test-set performance

Based on the synthetic holdouts under the winning ensemble (`weighted`), the expected test MAE band is **[20.53, 23.32]** (mean across A/B/C: 22.17).

Per holdout:

| holdout | ensemble MAE |
| --- | ---: |
| A | 20.53 |
| B | 22.66 |
| C | 23.32 |

## Plots

- Final 200-step forecast: `experiments\final\ensemble_preview.png`
