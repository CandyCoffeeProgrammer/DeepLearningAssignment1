"""Hyperparameter search machinery.

This module is the home of:
    - `train_eval_one`     : train one model on one holdout with one seed,
                              return test metrics.
    - `evaluate_config`    : aggregate `train_eval_one` across the three
                              holdouts and N seeds, returning the mean test
                              MAE used as the primary objective.
    - (later) Optuna study driver that samples configs and minimises
                              `evaluate_config(...)['mean_mae']`.

Final ranking is mean recursive 200-step MAE across A, B, C in the original
integer scale. This module computes that and also tracks per-holdout means
and the worst-case holdout for diagnostic use.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from src.data import Holdout
from src.evaluate import compute_metrics
from src.models import build_model
from src.predict import predict_holdout
from src.train import train_model
from src.utils import set_seed


def train_eval_one(
    cfg: dict,
    holdout: Holdout,
    seed: int,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict:
    """Train a fresh model with `seed` on `holdout`'s training portion, return
    recursive 200-step test metrics on its held-out segment.

    `cfg` is the full config dict (with `model` and `training` blocks).
    """
    set_seed(seed)
    model = build_model(
        family=cfg["model"]["family"],
        window=cfg["window"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        kernel_size=cfg["model"].get("kernel_size", 3),
        conv_layers=cfg["model"].get("conv_layers", 2),
    )
    result = train_model(model, holdout, cfg["training"], device=device, logger=logger)
    model.load_state_dict(result["best_state_dict"])
    model.to(device)
    pred = predict_holdout(model, holdout)
    metrics = compute_metrics(pred["pred_orig"], pred["true_orig"])
    return {
        "best_epoch": result["best_epoch"],
        "best_inner_val_mae": result["best_inner_val_mae"],
        "test_mae": metrics["mae"],
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_nmse": metrics["nmse"],
        "pred_orig": pred["pred_orig"],
    }


def evaluate_config(
    cfg: dict,
    holdouts: dict[str, Holdout],
    seeds: list[int],
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict:
    """Evaluate one configuration across `holdouts × seeds`.

    Aggregations:
        - `mean_mae`             : mean test MAE across all (holdout, seed) runs
        - `per_holdout_mean_mae` : mean across seeds, per holdout
        - `per_holdout_std_mae`  : seed std, per holdout
        - `worst_holdout`        : holdout with the largest mean MAE
        - `mean_mse`, `per_holdout_mean_mse`, ... — same family
    """
    all_runs: list[dict] = []
    by_holdout: dict[str, list[dict]] = {name: [] for name in holdouts}

    for name, h in holdouts.items():
        for seed in seeds:
            if logger is not None:
                logger.info(f"--- holdout {name}, seed {seed} ---")
            res = train_eval_one(cfg, h, seed, device, logger)
            res["holdout"] = name
            res["seed"] = seed
            # drop the prediction trajectory from `all_runs` to keep memory low;
            # callers wanting predictions should call `train_eval_one` themselves
            all_runs.append({k: v for k, v in res.items() if k != "pred_orig"})
            by_holdout[name].append(res)

    per_h_mean_mae = {n: float(np.mean([r["test_mae"] for r in rs])) for n, rs in by_holdout.items()}
    per_h_std_mae  = {n: float(np.std([r["test_mae"] for r in rs], ddof=0)) for n, rs in by_holdout.items()}
    per_h_mean_mse = {n: float(np.mean([r["test_mse"] for r in rs])) for n, rs in by_holdout.items()}

    worst_name = max(per_h_mean_mae.items(), key=lambda kv: kv[1])[0]

    overall_maes = [r["test_mae"] for r in all_runs]
    overall_mses = [r["test_mse"] for r in all_runs]
    return {
        "mean_mae": float(np.mean(overall_maes)),
        "std_mae":  float(np.std(overall_maes, ddof=0)),
        "max_mae":  float(np.max(overall_maes)),
        "min_mae":  float(np.min(overall_maes)),
        "mean_mse": float(np.mean(overall_mses)),
        "per_holdout_mean_mae": per_h_mean_mae,
        "per_holdout_std_mae":  per_h_std_mae,
        "per_holdout_mean_mse": per_h_mean_mse,
        "worst_holdout":        worst_name,
        "worst_holdout_mae":    per_h_mean_mae[worst_name],
        "all_runs":             all_runs,
    }
