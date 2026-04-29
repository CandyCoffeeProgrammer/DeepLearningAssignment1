"""Hyperparameter search machinery.

Pieces:
    - `train_eval_one`        : train one model on one holdout with one seed,
                                 return test metrics.
    - `evaluate_config`       : aggregate across (holdouts × seeds), return
                                 the mean test MAE used as the primary
                                 objective + per-holdout diagnostics.
    - `sample_config`         : Optuna sampler — derives a full config dict
                                 from a base config plus a trial.
    - `make_objective`        : produces the objective Optuna will minimise.
    - `run_search`            : run one Optuna study end-to-end with the
                                 conventional TPE sampler + leaderboard CSV.

Final ranking is mean recursive 200-step MAE across A, B, C in the
original integer scale.
"""

from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import torch

from src.data import Holdout, load_xtrain, make_holdouts
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


# ============================================================================
# Optuna search
# ============================================================================

# What hyperparameters are relevant to which family
_CONV_FAMILIES = {"cnn1d", "cnn_lstm", "tcn"}


def sample_config(trial: optuna.Trial, family: str, base_cfg: dict) -> dict:
    """Build a full config dict from the search space for a given family.

    Spec'd search space (see configs/base.yaml + the assignment brief):
        window         : {10, 15, 20, 25, 30, 40, 50, 60, 75}
        hidden_dim     : {32, 64, 128, 256}
        num_layers     : {1, 2, 3}
        dropout        : 0.0..0.4 uniform
        lr             : 5e-4..5e-3 log uniform
        batch_size     : {16, 32, 64}
        weight_decay   : 1e-6..1e-3 log uniform
        k_max          : {1, 3, 5, 7, 10}
        p_max          : {0.0, 0.25, 0.5}
        noise_sigma    : {0.0, 0.005, 0.01, 0.02}
        scaler         : {minmax_0_1, minmax_neg1_1, standard}
        lr_scheduler   : {OneCycleLR, ReduceLROnPlateau}
        kernel_size    : {3, 5}        (CNN-flavoured families only)
    """
    cfg = copy.deepcopy(base_cfg)

    cfg["window"] = trial.suggest_categorical("window", [10, 15, 20, 25, 30, 40, 50, 60, 75])

    cfg["model"]["family"] = family
    cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    cfg["model"]["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])
    cfg["model"]["dropout"]    = trial.suggest_float("dropout", 0.0, 0.4)
    if family in _CONV_FAMILIES:
        cfg["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
    if family == "cnn_lstm":
        cfg["model"]["conv_layers"] = trial.suggest_categorical("conv_layers", [1, 2])

    cfg["data"]["scaler"] = trial.suggest_categorical(
        "scaler", ["minmax_0_1", "minmax_neg1_1", "standard"]
    )

    cfg["training"]["lr"]            = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    cfg["training"]["batch_size"]    = trial.suggest_categorical("batch_size", [16, 32, 64])
    cfg["training"]["weight_decay"]  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg["training"]["lr_scheduler"]  = trial.suggest_categorical(
        "lr_scheduler", ["onecycle", "plateau"]
    )

    k_max = trial.suggest_categorical("k_max", [1, 3, 5, 7, 10])
    p_max = trial.suggest_categorical("p_max", [0.0, 0.25, 0.5])
    noise = trial.suggest_categorical("input_noise_sigma", [0.0, 0.005, 0.01, 0.02])
    epochs = int(cfg["training"]["epochs"])
    anneal = int(round(epochs * 0.75))   # fixed schedule shape — anneal over 75% of training
    cfg["training"]["multistep"] = {"k_max": k_max, "anneal_epochs": anneal if k_max > 1 else 0}
    cfg["training"]["scheduled_sampling"] = {"p_max": p_max, "anneal_epochs": anneal if p_max > 0 else 0}
    cfg["training"]["input_noise_sigma"]  = noise

    return cfg


def make_objective(
    family: str,
    base_cfg: dict,
    series: np.ndarray,
    device: torch.device,
    seed: int,
    logger: logging.Logger | None = None,
    leaderboard_path: Path | None = None,
) -> Callable[[optuna.Trial], float]:
    """Produce an Optuna objective that minimises mean test MAE across A/B/C.

    Each trial:
      1. samples a config,
      2. rebuilds holdouts with the sampled window + scaler,
      3. trains one model per holdout with seed `seed`,
      4. returns the mean recursive 200-step test MAE.

    Per-trial details are appended to `leaderboard_path` (CSV) if provided.
    """
    log = logger or logging.getLogger("search")

    def objective(trial: optuna.Trial) -> float:
        cfg = sample_config(trial, family, base_cfg)
        holdouts = make_holdouts(
            series,
            cfg["data"]["holdouts"],
            window=cfg["window"],
            inner_val_len=cfg["data"]["inner_val_len"],
            scaler_name=cfg["data"]["scaler"],
        )

        per_holdout: dict[str, dict] = {}
        try:
            for name, h in holdouts.items():
                res = train_eval_one(cfg, h, seed=seed, device=device, logger=None)
                per_holdout[name] = res
        except Exception as exc:
            log.warning(f"trial {trial.number} crashed: {exc!r}")
            raise optuna.TrialPruned() from exc

        maes = [r["test_mae"] for r in per_holdout.values()]
        mses = [r["test_mse"] for r in per_holdout.values()]
        mean_mae = float(np.mean(maes))
        worst_mae = float(np.max(maes))

        # Useful per-trial diagnostics in the dashboard
        trial.set_user_attr("per_holdout_mae", {n: float(r["test_mae"]) for n, r in per_holdout.items()})
        trial.set_user_attr("per_holdout_mse", {n: float(r["test_mse"]) for n, r in per_holdout.items()})
        trial.set_user_attr("worst_mae", worst_mae)
        trial.set_user_attr("mean_mse", float(np.mean(mses)))
        trial.set_user_attr("family", family)

        log.info(
            f"trial {trial.number:3d}  family={family}  mean_mae={mean_mae:7.3f}  "
            f"worst={worst_mae:7.3f}  per_holdout="
            + ", ".join(f"{n}={r['test_mae']:.2f}" for n, r in per_holdout.items())
        )

        if leaderboard_path is not None:
            row = {
                "trial": trial.number,
                "family": family,
                "mean_mae": mean_mae,
                "worst_mae": worst_mae,
                "mean_mse": float(np.mean(mses)),
                **{f"mae_{n}": r["test_mae"] for n, r in per_holdout.items()},
                **{f"mse_{n}": r["test_mse"] for n, r in per_holdout.items()},
                **{f"best_epoch_{n}": r["best_epoch"] for n, r in per_holdout.items()},
                **trial.params,
            }
            _append_csv_row(leaderboard_path, row)

        return mean_mae

    return objective


def _append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_search(
    family: str,
    base_cfg: dict,
    *,
    n_trials: int,
    series: np.ndarray,
    device: torch.device,
    seed: int = 42,
    storage: str | None = None,
    study_name: str | None = None,
    leaderboard_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> optuna.Study:
    """Run an Optuna TPE study for one family, return the completed study."""
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        study_name=study_name or f"santa_fe_{family}",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
        direction="minimize",
    )
    objective = make_objective(
        family=family,
        base_cfg=base_cfg,
        series=series,
        device=device,
        seed=seed,
        logger=logger,
        leaderboard_path=leaderboard_path,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study
