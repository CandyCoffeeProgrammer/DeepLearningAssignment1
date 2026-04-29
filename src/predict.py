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
