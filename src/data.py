"""Data loading, scaling, windowing, holdout splits.

The Santa Fe laser series lives in `data/Xtrain.mat` under key `Xtrain`,
shape (1000, 1), uint8, values in [2, 255].

The real test set is points 1001..1200, generated recursively from a model
trained on 1..1000. Locally we mirror this with three synthetic holdouts:

    holdout A: predict 801..1000  | model has 1..800   | inner val 701..800
    holdout B: predict 701..900   | model has 1..700   | inner val 601..700
    holdout C: predict 601..800   | model has 1..600   | inner val 501..600

In each case the scaler is fit on the *training* portion only (everything before
the inner-validation start). The seed window for recursive prediction is the
last `window` points of the training+inner-val combined region — i.e., the last
`window` points before the held-out segment, exactly as in the real test
scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ---------- loading ----------

def load_xtrain(path: str | Path, key: str = "Xtrain") -> np.ndarray:
    """Load the laser series and return a flat float64 array of shape (N,)."""
    raw = sio.loadmat(str(path))
    if key not in raw:
        keys = [k for k in raw if not k.startswith("_")]
        raise KeyError(f"key {key!r} not found in {path}; available: {keys}")
    arr = np.asarray(raw[key]).astype(np.float64).reshape(-1)
    return arr


# ---------- scaling ----------

class Scaler(Protocol):
    """Minimal scaler protocol; works on 1-D float arrays."""

    def fit(self, x: np.ndarray) -> "Scaler": ...
    def transform(self, x: np.ndarray) -> np.ndarray: ...
    def inverse_transform(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class MinMaxScaler:
    """MinMax to [low, high]. Defaults to [0, 1]."""

    low: float = 0.0
    high: float = 1.0
    data_min_: float = 0.0
    data_max_: float = 1.0

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.data_min_ = float(x.min())
        self.data_max_ = float(x.max())
        if self.data_max_ == self.data_min_:
            raise ValueError("constant series cannot be MinMax scaled")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scale = (self.high - self.low) / (self.data_max_ - self.data_min_)
        return (x - self.data_min_) * scale + self.low

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scale = (self.data_max_ - self.data_min_) / (self.high - self.low)
        return (x - self.low) * scale + self.data_min_


@dataclass
class StandardScaler:
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.mean_ = float(x.mean())
        self.std_ = float(x.std())
        if self.std_ == 0:
            raise ValueError("constant series cannot be standard-scaled")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std_ + self.mean_


def make_scaler(name: str) -> Scaler:
    name = name.lower()
    if name in ("minmax_0_1", "minmax", "minmax01"):
        return MinMaxScaler(low=0.0, high=1.0)
    if name in ("minmax_neg1_1", "minmax-1-1", "minmaxneg11"):
        return MinMaxScaler(low=-1.0, high=1.0)
    if name in ("standard", "zscore"):
        return StandardScaler()
    raise ValueError(f"unknown scaler {name!r}")


# ---------- holdouts ----------

@dataclass
class Holdout:
    """One synthetic holdout. All slices are indexed into the original series.

    - `train`         : points the model is allowed to use for parameter updates
    - `inner_val`     : last 100 points of the training portion, held out for
                        early stopping on recursive 200-step MAE
    - `test`          : the 200-pt segment the model must recursively predict
    - `seed_window`   : last `window` points of (train + inner_val), used as
                        the input that triggers recursive prediction
    All `_scaled` variants use a scaler fit on `train` only.
    """

    name: str
    test_start: int            # 1-indexed, inclusive
    test_end: int              # 1-indexed, inclusive
    window: int

    train: np.ndarray
    inner_val: np.ndarray
    test: np.ndarray
    seed_window: np.ndarray

    scaler: Scaler
    train_scaled: np.ndarray
    inner_val_scaled: np.ndarray
    test_scaled: np.ndarray
    seed_window_scaled: np.ndarray

    @property
    def horizon(self) -> int:
        return self.test_end - self.test_start + 1


def make_holdout(
    series: np.ndarray,
    name: str,
    test_start: int,
    test_end: int,
    *,
    window: int,
    inner_val_len: int,
    scaler_name: str,
) -> Holdout:
    """Build one holdout.

    `series` is 1-indexed by convention here for clarity (matches the spec) but
    we work in 0-indexed numpy slicing internally.
    """
    n = series.shape[0]
    if not (1 <= test_start <= test_end <= n):
        raise ValueError(f"holdout {name}: bad range [{test_start}, {test_end}] for series of length {n}")
    if window < 1:
        raise ValueError("window must be >= 1")
    if inner_val_len < 1:
        raise ValueError("inner_val_len must be >= 1")

    # 0-indexed conversions
    ts = test_start - 1
    te = test_end                     # python slice end (exclusive)

    pre_test = series[:ts]            # everything available before the holdout test
    if pre_test.size <= inner_val_len + window:
        raise ValueError(
            f"holdout {name}: not enough pre-test data ({pre_test.size}) for "
            f"inner_val_len={inner_val_len} and window={window}"
        )

    train = pre_test[:-inner_val_len]
    inner_val = pre_test[-inner_val_len:]
    test = series[ts:te]
    seed_window = pre_test[-window:]

    scaler = make_scaler(scaler_name).fit(train)
    return Holdout(
        name=name,
        test_start=test_start,
        test_end=test_end,
        window=window,
        train=train,
        inner_val=inner_val,
        test=test,
        seed_window=seed_window,
        scaler=scaler,
        train_scaled=scaler.transform(train),
        inner_val_scaled=scaler.transform(inner_val),
        test_scaled=scaler.transform(test),
        seed_window_scaled=scaler.transform(seed_window),
    )


def make_holdouts(
    series: np.ndarray,
    holdouts_cfg: dict[str, list[int]],
    *,
    window: int,
    inner_val_len: int,
    scaler_name: str,
) -> dict[str, Holdout]:
    """Build all holdouts described in config (e.g. {'A': [801, 1000], ...})."""
    return {
        name: make_holdout(
            series,
            name,
            int(rng[0]),
            int(rng[1]),
            window=window,
            inner_val_len=inner_val_len,
            scaler_name=scaler_name,
        )
        for name, rng in holdouts_cfg.items()
    }


# ---------- windowing ----------

def make_windows(series: np.ndarray, window: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window construction.

    Returns:
        X: (n_samples, window)   - input windows
        Y: (n_samples, horizon)  - the next `horizon` ground-truth values
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    n = series.shape[0]
    n_samples = n - window - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"series length {n} too short for window={window} horizon={horizon}"
        )
    X = np.empty((n_samples, window), dtype=np.float32)
    Y = np.empty((n_samples, horizon), dtype=np.float32)
    for i in range(n_samples):
        X[i] = series[i : i + window]
        Y[i] = series[i + window : i + window + horizon]
    return X, Y


class LaserWindowDataset(Dataset):
    """Sliding-window dataset for one-step or multi-step training.

    Returns (x, y) per sample where:
      x: (window, 1) float32
      y: (horizon,)  float32  — next `horizon` ground-truth values
    """

    def __init__(self, series: np.ndarray, window: int, horizon: int = 1):
        X, Y = make_windows(series, window=window, horizon=horizon)
        self.X = torch.from_numpy(X).unsqueeze(-1)   # (N, window, 1)
        self.Y = torch.from_numpy(Y)                 # (N, horizon)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
