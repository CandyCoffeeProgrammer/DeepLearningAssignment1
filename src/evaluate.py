"""Metrics + plots, all in the original integer scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE, NMSE in the original scale."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    var_true = float(np.var(true)) if np.var(true) > 0 else 1.0
    nmse = mse / var_true
    return {"mae": mae, "mse": mse, "rmse": rmse, "nmse": nmse}


def plot_prediction(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    title: str,
    out_path: str | Path | None = None,
    test_start: int | None = None,
) -> plt.Figure:
    """Overlay prediction on ground truth, with residuals on a second axis."""
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    n = len(true)
    if test_start is None:
        x = np.arange(n)
        xlabel = "step"
    else:
        x = np.arange(test_start, test_start + n)
        xlabel = "sample index"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(x, true, color="black", lw=0.9, label="ground truth")
    axes[0].plot(x, pred, color="crimson", lw=0.9, alpha=0.85, label="prediction")
    axes[0].set_ylabel("intensity")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, pred - true, color="steelblue", lw=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig
