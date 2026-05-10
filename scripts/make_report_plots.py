"""Generate the report-style plots for our baseline predictions.

Mirrors the plot set the friend produced for his Forward Lateral Causal CNN
(real-vs-predicted line plot, residuals over time, error histogram,
predicted-vs-actual scatter, binned regression confusion matrix) so the
two approaches can be compared visually with the same conventions.

Inputs (auto-discovered):
    experiments/final/predictions.npy   — our 200-step recursive forecast
    data/Xtest.mat                      — released test set

Outputs:
    experiments/final/report_plots/baseline/baseline_real_vs_predicted.png
    experiments/final/report_plots/baseline/baseline_residuals.png
    experiments/final/report_plots/baseline/baseline_error_histogram.png
    experiments/final/report_plots/baseline/baseline_predicted_vs_actual.png
    experiments/final/report_plots/baseline/baseline_regression_confusion_graph.png
    experiments/final/report_plots/baseline/baseline_metrics.{csv,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import (
    confusion_matrix,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]


def align_true_pred(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(yt), len(yp))
    return yt[:n], yp[:n]


def compute_forecast_metrics(y_true, y_pred) -> dict:
    yt, yp = align_true_pred(y_true, y_pred)
    err = yp - yt
    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(yt, yp)
    var_t = float(np.var(yt))
    nmse = mse / var_t if var_t > 0 else float("nan")
    return {
        "n_steps": int(len(yt)),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
        "NMSE": nmse,
        "MedianAE": float(median_absolute_error(yt, yp)),
        "MaxError": float(max_error(yt, yp)),
        "MeanResidual": float(err.mean()),
        "StdResidual": float(err.std(ddof=0)),
    }


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_real_vs_predicted(yt, yp, title, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(yt))
    ax.plot(x, yt, label="True", marker="o", markersize=3)
    ax.plot(x, yp, label="Predicted", marker="x", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_residuals(yt, yp, title, out):
    res = yt - yp
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(res))
    ax.plot(x, res, label="Residuals", marker="o", markersize=3)
    ax.axhline(0, color="red", linestyle="--", label="Zero Error")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("Residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_error_histogram(yt, yp, title, out):
    err = yt - yp
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_predicted_vs_actual(yt, yp, title, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, alpha=0.7, color="green", edgecolor="black")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.grid(True, alpha=0.3)
    save(fig, out)


def make_regression_bins(values, n_bins=5, strategy="quantile"):
    if strategy == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(values.min(), values.max(), n_bins + 1)
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def plot_regression_confusion(yt, yp, n_bins, title, out):
    all_vals = np.concatenate([yt, yp])
    edges = make_regression_bins(all_vals, n_bins=n_bins, strategy="quantile")
    yt_b = np.digitize(yt, edges[1:-1], right=False)
    yp_b = np.digitize(yp, edges[1:-1], right=False)
    labels = np.arange(len(edges) - 1)
    cm = confusion_matrix(yt_b, yp_b, labels=labels)

    bin_labels = []
    for i in range(len(edges) - 1):
        l, r = edges[i], edges[i + 1]
        if np.isneginf(l):
            bin_labels.append(f"<= {r:.3g}")
        elif np.isposinf(r):
            bin_labels.append(f"> {l:.3g}")
        else:
            bin_labels.append(f"{l:.3g} to {r:.3g}")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("Actual bin")
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_yticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_yticklabels(bin_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    save(fig, out)


def main() -> int:
    pred_path = ROOT / "experiments" / "final" / "predictions.npy"
    xtest_path = ROOT / "data" / "Xtest.mat"
    out_dir = ROOT / "experiments" / "final" / "report_plots" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    yp = np.load(pred_path).astype(np.float64).reshape(-1)
    yt = sio.loadmat(str(xtest_path))["Xtest"].astype(np.float64).reshape(-1)
    yt, yp = align_true_pred(yt, yp)

    metrics = compute_forecast_metrics(yt, yp)
    metrics["model_name"] = "Baseline"
    print(json.dumps(metrics, indent=2))

    pd.DataFrame([metrics]).to_csv(out_dir / "baseline_metrics.csv", index=False)
    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    name = "Baseline"
    plot_real_vs_predicted(yt, yp, f"{name}: Real vs Predicted Values",
                            out_dir / "baseline_real_vs_predicted.png")
    plot_residuals(yt, yp, f"{name}: Residuals Over Time",
                   out_dir / "baseline_residuals.png")
    plot_error_histogram(yt, yp, f"{name}: Prediction Error Distribution",
                         out_dir / "baseline_error_histogram.png")
    plot_predicted_vs_actual(yt, yp, f"{name}: Predicted vs Actual",
                              out_dir / "baseline_predicted_vs_actual.png")
    plot_regression_confusion(yt, yp, n_bins=5,
                               title=f"{name}: Binned Regression Confusion Graph",
                               out=out_dir / "baseline_regression_confusion_graph.png")

    print(f"\nplots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
