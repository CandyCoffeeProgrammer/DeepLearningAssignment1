"""Training loop with multi-step loss + scheduled sampling + input noise +
recursive inner-val evaluation.

Public entry point: `train_model(model, holdout, cfg, ...)` returns a dict with
the best state-dict and the full training history.

The training step is delegated to `src.losses.multistep_step`; with `k_max=1`,
`p_max=0`, and `input_noise=0` it collapses exactly to plain one-step MSE
(milestone 2 behaviour), so the same call works for ablations.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import Holdout, LaserWindowDataset
from src.evaluate import compute_metrics
from src.losses import anneal_k, linear_anneal, multistep_step
from src.models import BaseForecaster
from src.predict import predict_inner_val


def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    *,
    epochs: int,
    steps_per_epoch: int,
):
    name = cfg.get("lr_scheduler", "onecycle").lower()
    if name in ("onecycle", "onecyclelr"):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["lr"]),
            total_steps=epochs * steps_per_epoch,
        )
    if name in ("plateau", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    if name in ("none", "off", ""):
        return None
    raise ValueError(f"unknown lr_scheduler {name!r}")


def train_model(
    model: BaseForecaster,
    holdout: Holdout,
    cfg: dict,
    *,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Train one model on one holdout's training portion.

    cfg is the `training` block of the YAML (see configs/base.yaml).
    Inner-val recursive eval drives early stopping.
    """
    log = logger or logging.getLogger("train")

    # ---- trick configs ----
    ms_cfg = cfg.get("multistep", {}) or {}
    k_max = int(ms_cfg.get("k_max", 1))
    k_anneal = int(ms_cfg.get("anneal_epochs", 50))

    ss_cfg = cfg.get("scheduled_sampling", {}) or {}
    p_max = float(ss_cfg.get("p_max", 0.0))
    p_anneal = int(ss_cfg.get("anneal_epochs", 50))

    input_noise = float(cfg.get("input_noise_sigma", 0.0))

    # ---- data ----
    window = int(holdout.window)
    horizon = max(1, k_max)              # need k_max future targets for multi-step unroll
    train_ds = LaserWindowDataset(holdout.train_scaled, window=window, horizon=horizon)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    # ---- model + optim ----
    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    epochs = int(cfg["epochs"])
    scheduler = _build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    is_onecycle = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    grad_clip = float(cfg.get("grad_clip", 0.0))

    # ---- early-stopping state ----
    eval_every = int(cfg.get("recursive_eval_every", 5))
    patience = int(cfg.get("early_stop_patience", 30))
    best_mae = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    bad_evals = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        # zero-indexed for anneal helpers
        k_t = anneal_k(epoch - 1, k_anneal, k_max)
        p_t = linear_anneal(epoch - 1, p_anneal, start=0.0, end=p_max)

        model.train()
        running = 0.0
        n_obs = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)              # (B, W, 1)
            y = y.to(device, non_blocking=True)              # (B, horizon)
            loss = multistep_step(
                model, x, y,
                k_t=k_t, p_t=p_t, input_noise=input_noise,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if is_onecycle:
                scheduler.step()
            running += loss.item() * x.size(0)
            n_obs += x.size(0)
        epoch_loss = running / max(1, n_obs)
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": epoch_loss, "k_t": k_t, "p_t": p_t}

        # ---- recursive inner-val evaluation ----
        if epoch % eval_every == 0 or epoch == epochs:
            res = predict_inner_val(model, holdout)
            metrics = compute_metrics(res["pred_orig"], res["true_orig"])
            entry.update({f"val_{k}": v for k, v in metrics.items()})

            if is_plateau:
                scheduler.step(metrics["mae"])

            improved = metrics["mae"] < best_mae - 1e-6
            if improved:
                best_mae = metrics["mae"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
            else:
                bad_evals += 1

            log.info(
                f"epoch {epoch:4d}  train_loss={epoch_loss:.5f}  "
                f"val_mae={metrics['mae']:7.3f}  val_mse={metrics['mse']:8.2f}  "
                f"best_mae={best_mae:7.3f}@e{best_epoch}  bad={bad_evals}/{patience}"
            )

            if bad_evals >= patience:
                log.info(f"early stopping at epoch {epoch} (no inner-val improvement for {patience} evals)")
                history.append(entry)
                break

        history.append(entry)

    return {
        "history": history,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_inner_val_mae": best_mae,
    }


def train_full(
    model: BaseForecaster,
    series_scaled,
    cfg: dict,
    *,
    window: int,
    epochs: int,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Final-stage training: fit on the full Xtrain (1..1000) for a fixed
    number of epochs with no validation and no early stopping.

    Used by milestone 8 (full-data retrain). The epoch count is normally the
    median best-epoch across seeds from the search/holdout phase, since there
    is no validation curve to early-stop against.

    Returns final state dict + history.
    """
    log = logger or logging.getLogger("train_full")

    ms_cfg = cfg.get("multistep", {}) or {}
    k_max = int(ms_cfg.get("k_max", 1))
    k_anneal = int(ms_cfg.get("anneal_epochs", 0))

    ss_cfg = cfg.get("scheduled_sampling", {}) or {}
    p_max = float(ss_cfg.get("p_max", 0.0))
    p_anneal = int(ss_cfg.get("anneal_epochs", 0))

    input_noise = float(cfg.get("input_noise_sigma", 0.0))

    horizon = max(1, k_max)
    train_ds = LaserWindowDataset(series_scaled, window=window, horizon=horizon)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    is_onecycle = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    grad_clip = float(cfg.get("grad_clip", 0.0))

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        k_t = anneal_k(epoch - 1, k_anneal, k_max)
        p_t = linear_anneal(epoch - 1, p_anneal, start=0.0, end=p_max)

        model.train()
        running = 0.0
        n_obs = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = multistep_step(model, x, y, k_t=k_t, p_t=p_t, input_noise=input_noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if is_onecycle:
                scheduler.step()
            running += loss.item() * x.size(0)
            n_obs += x.size(0)
        epoch_loss = running / max(1, n_obs)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "k_t": k_t, "p_t": p_t})

        if epoch % 25 == 0 or epoch == epochs:
            log.info(f"epoch {epoch:4d}  train_loss={epoch_loss:.5f}  k_t={k_t}  p_t={p_t:.3f}")

    return {
        "history": history,
        "final_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "epochs": epochs,
    }
