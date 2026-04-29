"""Multi-step training loss + scheduled sampling + input noise.

Drop-in replacement for the one-step MSE step used in milestone 2. With
``k_max=1`` and ``p_max=0`` and ``input_noise=0`` this collapses exactly to
that one-step behaviour, so it is safe to plug into the existing training
loop unconditionally.

Why these tricks exist
----------------------
The Santa Fe rollout fails because a one-step-trained model never sees the
state space its own predictions wander into during a long recursive rollout.
Multi-step loss puts the model inside its own error distribution at training
time. Scheduled sampling lets us interpolate between teacher-forcing
(stable, biased) and self-forcing (high-variance, on-distribution).
Input noise gives a cheap regulariser that nudges the model toward locally
contractive dynamics. All three are anneal-based so early training is stable.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.models import BaseForecaster


def linear_anneal(epoch: int, anneal_epochs: int, start: float, end: float) -> float:
    """Linear interpolation from `start` (epoch <= 0) to `end` (epoch >= anneal_epochs).

    Use ``epoch`` zero-indexed (i.e., epoch 0 returns ``start``, epoch
    ``anneal_epochs`` returns ``end``).
    """
    if anneal_epochs <= 0:
        return end
    frac = min(1.0, max(0.0, epoch / anneal_epochs))
    return start + frac * (end - start)


def anneal_k(epoch: int, anneal_epochs: int, k_max: int) -> int:
    """Integer multi-step depth that climbs 1 -> k_max linearly across `anneal_epochs`."""
    if k_max <= 1:
        return 1
    raw = linear_anneal(epoch, anneal_epochs, start=1.0, end=float(k_max))
    return max(1, min(k_max, int(round(raw))))


def multistep_step(
    model: BaseForecaster,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k_t: int,
    p_t: float = 0.0,
    input_noise: float = 0.0,
) -> torch.Tensor:
    """Compute the mean MSE over a `k_t`-step unrolled prediction.

    Args:
        model: forecaster with `forward(x: (B, W, 1)) -> (B, 1)`.
        x:     input window, shape (B, W, 1) in scaled space.
        y:     next ground-truth values, shape (B, H) with H >= k_t (in scaled space).
        k_t:   current unroll depth (>= 1).
        p_t:   scheduled-sampling probability — at unroll step t<k_t-1 each batch
               element independently feeds the model's own prediction (prob p_t)
               or the ground-truth value (prob 1-p_t). 0 disables it.
        input_noise: stddev of zero-mean Gaussian noise added to `x` (training only).

    Returns:
        scalar tensor — mean MSE over the k_t step predictions.
    """
    if k_t < 1:
        raise ValueError(f"k_t must be >= 1, got {k_t}")
    if y.shape[1] < k_t:
        raise ValueError(f"target horizon {y.shape[1]} < unroll depth {k_t}")

    if input_noise > 0.0:
        x = x + input_noise * torch.randn_like(x)

    cur = x                                    # (B, W, 1)
    total = 0.0
    for t in range(k_t):
        pred_t = model(cur)                    # (B, 1)
        target_t = y[:, t:t + 1]               # (B, 1)
        total = total + F.mse_loss(pred_t, target_t)

        if t < k_t - 1:
            if p_t > 0.0:
                # per-element coin flip: 1 = feed own prediction, 0 = feed ground truth
                use_own = (torch.rand_like(target_t) < p_t).float()
                feed = use_own * pred_t + (1.0 - use_own) * target_t
            else:
                feed = target_t
            cur = torch.cat([cur[:, 1:, :], feed.unsqueeze(-1)], dim=1)

    return total / k_t
