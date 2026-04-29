"""Forecasting models.

Common interface (`BaseForecaster`):
    forward(x: (B, window, 1)) -> (B, 1)        — one-step-ahead prediction
    rollout(seed: (window,) or (1, window, 1),
            n_steps: int) -> (n_steps,)        — recursive prediction in scaled space

Causality is preserved end-to-end. No bidirectional layers. The default
`rollout` re-runs the full window each step; that's cheap for our
(window <= 75, n_steps = 200) setting and keeps every model interchangeable.

Families:
    - MLP        (flat-window dense stack)
    - LSTM, GRU  (recurrent + linear head on last timestep)
    - CNN1D      (causal conv stack + global average pool)
    - CNN_LSTM   (causal conv front-end feeding an LSTM)
    - TCN        (stacked dilated causal residual blocks)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class BaseForecaster(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window, 1) -> y: (B, 1) one-step-ahead prediction."""

    @torch.no_grad()
    def rollout(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Recursive prediction in scaled space.

        Args:
            seed:    1-D tensor of length `window`, or (1, window, 1).
            n_steps: number of future steps to predict.
        Returns:
            (n_steps,) tensor on the model's device.
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
            x = torch.cat([x[:, 1:, :], y.view(1, 1, 1)], dim=1)
        return preds


# ---------------- causal conv helper ----------------

class CausalConv1d(nn.Module):
    """1D conv that pads only on the left so output[t] depends on input[<=t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


# ---------------- MLP ----------------

class MLP(BaseForecaster):
    def __init__(self, window: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = window
        for _i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.squeeze(-1))


# ---------------- recurrent ----------------

class _RecurrentForecaster(BaseForecaster):
    """Shared shell for LSTM/GRU (single-step output via last hidden state)."""

    rnn_cls: type[nn.RNNBase]  # set by subclasses

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        # PyTorch only applies RNN dropout between stacked layers (>= 2)
        self.rnn = self.rnn_cls(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)            # (B, W, H)
        return self.head(out[:, -1, :])  # (B, 1)


class LSTM(_RecurrentForecaster):
    rnn_cls = nn.LSTM


class GRU(_RecurrentForecaster):
    rnn_cls = nn.GRU


# ---------------- 1D CNN ----------------

class CNN1D(BaseForecaster):
    """Stack of causal 1D conv blocks + global average pool + linear head."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1, **_):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        blocks: list[nn.Module] = []
        in_ch = 1
        for _i in range(num_layers):
            blocks += [
                CausalConv1d(in_ch, hidden_dim, kernel_size, dilation=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 1, W)
        x = self.blocks(x)             # (B, hidden, W)
        x = x.mean(dim=2)              # global average pool: (B, hidden)
        return self.head(x)            # (B, 1)


# ---------------- CNN + LSTM ----------------

class CNNLSTM(BaseForecaster):
    """Causal conv feature extractor + LSTM + linear head on last timestep."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        conv_layers: int = 2,
        **_,
    ):
        super().__init__()
        if num_layers < 1 or conv_layers < 1:
            raise ValueError("num_layers and conv_layers must be >= 1")
        conv_blocks: list[nn.Module] = []
        in_ch = 1
        for _i in range(conv_layers):
            conv_blocks += [
                CausalConv1d(in_ch, hidden_dim, kernel_size, dilation=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            in_ch = hidden_dim
        self.conv = nn.Sequential(*conv_blocks)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 1, W)
        x = self.conv(x)               # (B, hidden, W)
        x = x.transpose(1, 2)          # (B, W, hidden)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (B, 1)


# ---------------- Dilated causal TCN ----------------

class _TCNBlock(nn.Module):
    """Two stacked causal convs with residual + ReLU. Channels stay constant."""

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return self.act(x + res)


class TCN(BaseForecaster):
    """Stack of dilated causal residual blocks. Dilations 1, 2, 4, 8, ... .

    Receptive field at depth d (kernel K, two convs per block):
        1 + 2 * (2**d - 1) * (K - 1)
    For K=3, d=4 -> RF=61, d=5 -> 125 — covers any window in our search space.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 1, W)
        x = self.input_proj(x)         # (B, hidden, W)
        for block in self.blocks:
            x = block(x)               # (B, hidden, W)
        return self.head(x[:, :, -1])  # last-timestep features -> (B, 1)


# ---------------- factory ----------------

def build_model(
    family: str,
    *,
    window: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    kernel_size: int = 3,
    conv_layers: int = 2,
) -> BaseForecaster:
    family = family.lower()
    common = dict(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if family == "mlp":
        return MLP(window=window, **common)
    if family == "lstm":
        return LSTM(**common)
    if family == "gru":
        return GRU(**common)
    if family in ("cnn1d", "cnn", "conv1d"):
        return CNN1D(kernel_size=kernel_size, **common)
    if family in ("cnn_lstm", "cnnlstm"):
        return CNNLSTM(kernel_size=kernel_size, conv_layers=conv_layers, **common)
    if family == "tcn":
        return TCN(kernel_size=kernel_size, **common)
    raise ValueError(f"unknown model family {family!r}")
