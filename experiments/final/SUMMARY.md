# Final summary — Santa Fe laser recursive forecast

Lowest MAE / MSE on the held-out 200-pt test set wins. This file collects
the search leaderboard, the chosen ensemble, and the expected performance
based on synthetic holdouts. Once `Xtest.mat` arrives, run
`python scripts/make_test_evaluation.py --xtest data/Xtest.mat` to score.

## Leaderboard — top 20 single models across all families

| rank | family | trial | mean MAE | worst MAE | A | B | C | window | hidden | k_max | p_max | noise |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | gru | 13 | 18.592 | 27.357 | 3.13 | 27.36 | 25.29 | 10 | 64 | 3 | 0.25 | 0.02 |
| 2 | tcn | 1 | 20.595 | 25.296 | 24.30 | 25.30 | 12.19 | 10 | 32 | 5 | 0.5 | 0.02 |
| 3 | mlp | 2 | 21.678 | 36.730 | 21.28 | 7.03 | 36.73 | 25 | 128 | 5 | 0.5 | 0.005 |
| 4 | tcn | 8 | 21.953 | 28.514 | 20.87 | 16.47 | 28.51 | 10 | 64 | 1 | 0.0 | 0.01 |
| 5 | tcn | 6 | 22.237 | 34.833 | 34.83 | 14.75 | 17.13 | 60 | 64 | 1 | 0.5 | 0.0 |
| 6 | mlp | 9 | 24.181 | 29.095 | 29.10 | 25.01 | 18.44 | 60 | 32 | 3 | 0.25 | 0.005 |
| 7 | mlp | 3 | 24.577 | 30.040 | 30.04 | 22.88 | 20.81 | 40 | 32 | 1 | 0.5 | 0.005 |
| 8 | mlp | 7 | 24.838 | 38.616 | 38.62 | 25.65 | 10.25 | 30 | 128 | 5 | 0.0 | 0.005 |
| 9 | tcn | 3 | 25.039 | 31.961 | 31.96 | 25.06 | 18.10 | 60 | 128 | 1 | 0.25 | 0.02 |
| 10 | tcn | 12 | 25.205 | 31.498 | 31.50 | 25.25 | 18.86 | 10 | 64 | 1 | 0.0 | 0.01 |
| 11 | mlp | 1 | 25.383 | 33.597 | 33.60 | 23.89 | 18.66 | 15 | 128 | 3 | 0.25 | 0.0 |
| 12 | gru | 11 | 25.988 | 31.261 | 31.26 | 27.17 | 19.53 | 60 | 128 | 3 | 0.25 | 0.0 |
| 13 | tcn | 5 | 26.085 | 32.637 | 32.64 | 27.82 | 17.79 | 25 | 32 | 1 | 0.0 | 0.005 |
| 14 | tcn | 4 | 26.585 | 34.932 | 34.93 | 25.99 | 18.83 | 10 | 128 | 1 | 0.5 | 0.02 |
| 15 | mlp | 5 | 26.605 | 35.006 | 35.01 | 25.80 | 19.01 | 10 | 64 | 1 | 0.25 | 0.02 |
| 16 | mlp | 4 | 26.645 | 35.777 | 35.78 | 24.79 | 19.36 | 60 | 32 | 5 | 0.25 | 0.005 |
| 17 | cnn1d | 9 | 27.159 | 36.285 | 36.28 | 26.14 | 19.05 | 50 | 64 | 1 | 0.5 | 0.0 |
| 18 | cnn1d | 2 | 27.351 | 37.743 | 37.74 | 25.99 | 18.32 | 25 | 128 | 3 | 0.25 | 0.02 |
| 19 | tcn | 13 | 27.822 | 37.186 | 37.19 | 24.86 | 21.42 | 60 | 64 | 5 | 0.5 | 0.02 |
| 20 | tcn | 7 | 28.222 | 33.226 | 33.23 | 25.64 | 25.80 | 20 | 128 | 5 | 0.0 | 0.0 |

## Best per family

| family | best mean MAE | worst MAE | per-holdout |
| --- | ---: | ---: | --- |
| gru | 18.592 | 27.357 | A=3.13, B=27.36, C=25.29 |
| tcn | 20.595 | 25.296 | A=24.30, B=25.30, C=12.19 |
| mlp | 21.678 | 36.730 | A=21.28, B=7.03, C=36.73 |
| cnn1d | 27.159 | 36.285 | A=36.28, B=26.14, C=19.05 |
| cnn_lstm | 29.474 | 43.905 | A=13.39, B=31.13, C=43.91 |
| lstm | 29.783 | 48.271 | A=14.86, B=26.22, C=48.27 |

## Ensemble strategy comparison

Source: `experiments\baselines\milestone7_ensembles_top_per_family`

| strategy | mean test MAE across A/B/C |
| --- | ---: |
| mean | 24.914 |
| weighted | 25.327 |
| median | 25.844 |

**Winning strategy: `mean`**


## Final ensemble (retrained on full Xtrain)

Generated at: `2026-04-29T11:01:52.310618Z`
Strategy: **mean**, members: **30**, horizon: 200
Forecast range: min=19.20  max=161.98  mean=54.77  std=23.62

| tag | family | window | epochs | scaler |
| --- | --- | ---: | ---: | --- |
| gru_best_s0 | gru | 10 | 100 | minmax_neg1_1 |
| gru_best_s1 | gru | 10 | 100 | minmax_neg1_1 |
| gru_best_s2 | gru | 10 | 100 | minmax_neg1_1 |
| gru_best_s3 | gru | 10 | 100 | minmax_neg1_1 |
| gru_best_s4 | gru | 10 | 100 | minmax_neg1_1 |
| tcn_best_s0 | tcn | 10 | 100 | minmax_neg1_1 |
| tcn_best_s1 | tcn | 10 | 100 | minmax_neg1_1 |
| tcn_best_s2 | tcn | 10 | 100 | minmax_neg1_1 |
| tcn_best_s3 | tcn | 10 | 100 | minmax_neg1_1 |
| tcn_best_s4 | tcn | 10 | 100 | minmax_neg1_1 |
| mlp_best_s0 | mlp | 25 | 100 | minmax_0_1 |
| mlp_best_s1 | mlp | 25 | 100 | minmax_0_1 |
| mlp_best_s2 | mlp | 25 | 100 | minmax_0_1 |
| mlp_best_s3 | mlp | 25 | 100 | minmax_0_1 |
| mlp_best_s4 | mlp | 25 | 100 | minmax_0_1 |
| cnn1d_best_s0 | cnn1d | 50 | 100 | minmax_0_1 |
| cnn1d_best_s1 | cnn1d | 50 | 100 | minmax_0_1 |
| cnn1d_best_s2 | cnn1d | 50 | 100 | minmax_0_1 |
| cnn1d_best_s3 | cnn1d | 50 | 100 | minmax_0_1 |
| cnn1d_best_s4 | cnn1d | 50 | 100 | minmax_0_1 |
| cnn_lstm_best_s0 | cnn_lstm | 40 | 100 | minmax_0_1 |
| cnn_lstm_best_s1 | cnn_lstm | 40 | 100 | minmax_0_1 |
| cnn_lstm_best_s2 | cnn_lstm | 40 | 100 | minmax_0_1 |
| cnn_lstm_best_s3 | cnn_lstm | 40 | 100 | minmax_0_1 |
| cnn_lstm_best_s4 | cnn_lstm | 40 | 100 | minmax_0_1 |
| lstm_best_s0 | lstm | 30 | 100 | standard |
| lstm_best_s1 | lstm | 30 | 100 | standard |
| lstm_best_s2 | lstm | 30 | 100 | standard |
| lstm_best_s3 | lstm | 30 | 100 | standard |
| lstm_best_s4 | lstm | 30 | 100 | standard |

## Expected test-set performance

Based on the synthetic holdouts under the winning ensemble (`mean`), the expected test MAE band is **[22.65, 27.62]** (mean across A/B/C: 24.91).

Per holdout:

| holdout | ensemble MAE |
| --- | ---: |
| A | 27.62 |
| B | 22.65 |
| C | 24.47 |

## Plots

- Final 200-step forecast: `experiments\final\ensemble_preview.png`
