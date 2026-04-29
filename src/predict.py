"""Recursive prediction utilities.

For one model on one holdout, the standard flow is:

    pred_scaled = recursive_rollout(model, seed_window_scaled, n_steps)
    pred_orig   = scaler.inverse_transform(pred_scaled)
    metrics     = compute_metrics(pred_orig, holdout.test)

Ensembling lives separately: each member rolls out independently for the full
horizon using its OWN predictions, then we aggregate the trajectories.
Aggregating per step and feeding the aggregate back is a known anti-pattern.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data import Holdout
from src.models import BaseForecaster


@torch.no_grad()
def recursive_rollout(
    model: BaseForecaster,
    seed_window_scaled: np.ndarray | torch.Tensor,
    n_steps: int,
) -> np.ndarray:
    """Run the model recursively for `n_steps` and return predictions in scaled space."""
    if isinstance(seed_window_scaled, np.ndarray):
        seed = torch.from_numpy(np.asarray(seed_window_scaled, dtype=np.float32))
    else:
        seed = seed_window_scaled.float()
    out = model.rollout(seed, n_steps)
    return out.detach().cpu().numpy().astype(np.float64)


def predict_holdout(
    model: BaseForecaster,
    holdout: Holdout,
) -> dict:
    """Run a 200-step recursive prediction on a holdout's test segment.

    Returns a dict with 'pred_scaled', 'pred_orig', 'true_orig'. Metrics live
    in src.evaluate to keep this module thin.
    """
    pred_scaled = recursive_rollout(model, holdout.seed_window_scaled, n_steps=holdout.horizon)
    pred_orig = holdout.scaler.inverse_transform(pred_scaled)
    return {
        "pred_scaled": pred_scaled,
        "pred_orig": pred_orig,
        "true_orig": holdout.test,
    }


def predict_inner_val(
    model: BaseForecaster,
    holdout: Holdout,
) -> dict:
    """Run a recursive prediction for `len(inner_val)` steps starting from the
    last `window` points of the training portion. Used for early stopping."""
    seed = holdout.train_scaled[-holdout.window:]
    n_steps = holdout.inner_val.shape[0]
    pred_scaled = recursive_rollout(model, seed, n_steps=n_steps)
    pred_orig = holdout.scaler.inverse_transform(pred_scaled)
    return {
        "pred_scaled": pred_scaled,
        "pred_orig": pred_orig,
        "true_orig": holdout.inner_val,
    }


# ============================================================================
# Ensembling
# ============================================================================
#
# Critical rule: each ensemble member rolls out independently for the full
# horizon using its OWN predictions. Then we aggregate the trajectories.
# Aggregating per step and feeding the aggregate back destroys diversity and
# consistently underperforms in the time-series literature.

def ensemble_trajectories(
    trajectories: list[np.ndarray],
    *,
    method: str = "mean",
    weights: list[float] | None = None,
) -> np.ndarray:
    """Aggregate independent recursive trajectories.

    Args:
        trajectories: list of 1-D arrays each of length `n_steps`.
        method:       'mean', 'median', or 'weighted'.
        weights:      required for 'weighted'; will be normalised to sum to 1.
                      Use weights inversely proportional to inner-val MAE.
    Returns:
        aggregated trajectory, shape (n_steps,).
    """
    if not trajectories:
        raise ValueError("no trajectories provided")
    arr = np.stack(
        [np.asarray(t, dtype=np.float64).reshape(-1) for t in trajectories],
        axis=0,
    )  # (n_members, n_steps)

    method = method.lower()
    if method == "mean":
        return arr.mean(axis=0)
    if method == "median":
        return np.median(arr, axis=0)
    if method == "weighted":
        if weights is None:
            raise ValueError("weighted ensemble needs weights")
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (arr.shape[0],):
            raise ValueError(
                f"len(weights)={w.shape[0]} != n_members={arr.shape[0]}"
            )
        if (w <= 0).any():
            raise ValueError("weights must be strictly positive")
        w = w / w.sum()
        return (w[:, None] * arr).sum(axis=0)
    raise ValueError(f"unknown ensemble method {method!r}")


def inverse_inner_val_weights(inner_val_maes: list[float], eps: float = 1e-3) -> list[float]:
    """Return weights inversely proportional to inner-val MAE (lower is better)."""
    arr = np.asarray(inner_val_maes, dtype=np.float64) + eps
    w = 1.0 / arr
    return list(w)
