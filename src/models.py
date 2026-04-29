"""Forecasting models.

All models share `BaseForecaster`:
    forward(x: (B, window, 1)) -> (B, 1)        — one-step-ahead prediction
    rollout(seed: (window,) or (1, window, 1),
            n_steps: int) -> (n_steps,)        — recursive prediction in scaled space

Causality is preserved end-to-end (no bidirectional layers, no peeking at the
future). The rollout default re-runs the full window each step; that's
cheap for our (window <= 75, n_steps = 200) setting and keeps the rollout
identical for every model family. Stateful overrides can come later if speed
ever matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseForecaster(nn.Module, ABC):
    """Common interface for every forecaster in this project."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window, 1) -> y: (B, 1) one-step-ahead prediction."""

    @torch.no_grad()
    def rollout(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Recursive prediction in scaled space.

        Args:
            seed:     1-D tensor of length `window`, or (1, window, 1).
            n_steps:  number of future steps to predict.
        Returns:
            (n_steps,) tensor on the same device as the model parameters.
        """
        self.eval()
        device = next(self.parameters()).device

        if seed.ndim == 1:
            x = seed.to(device).view(1, -1, 1).float()
        elif seed.ndim == 3:
            x = seed.to(device).float()
        else:
            raise ValueError(f"seed must be 1-D or 3-D, got shape {tuple(seed.shape)}")

        preds = torch.empty(n_steps, device=device, dtype=x.dtype)
        for t in range(n_steps):
            y = self(x)                       # (1, 1)
            preds[t] = y.squeeze()
            # slide: drop oldest sample, append the new prediction
            x = torch.cat([x[:, 1:, :], y.view(1, 1, 1)], dim=1)
        return preds


# ---------------- MLP ----------------

class MLP(BaseForecaster):
    """Flattened-window MLP. 2..4 hidden layers, ReLU, dropout."""

    def __init__(self, window: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = window
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, 1) -> (B, W)
        return self.net(x.squeeze(-1))


# ---------------- LSTM ----------------

class LSTM(BaseForecaster):
    """nn.LSTM + linear head on the final timestep."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, **_unused):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        # PyTorch only applies LSTM dropout between stacked layers (>= 2)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (B, W, H)
        return self.head(out[:, -1, :])  # (B, 1)


# ---------------- factory ----------------

def build_model(family: str, *, window: int, hidden_dim: int, num_layers: int, dropout: float) -> BaseForecaster:
    family = family.lower()
    if family == "mlp":
        return MLP(window=window, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if family == "lstm":
        return LSTM(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"unknown model family {family!r} (milestone 2 supports: mlp, lstm)")
