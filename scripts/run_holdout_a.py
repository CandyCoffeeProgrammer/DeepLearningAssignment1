"""Milestone 2 driver: end-to-end MLP + LSTM on Holdout A.

Trains both models with one seed and the default config, runs the recursive
200-step prediction on the held-out test segment, prints metrics, and writes a
comparison plot per model under experiments/<run_id>/.

Run from project root:
    python scripts/run_holdout_a.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain, make_holdouts                 # noqa: E402
from src.evaluate import compute_metrics, plot_prediction       # noqa: E402
from src.models import build_model                              # noqa: E402
from src.predict import predict_holdout                         # noqa: E402
from src.train import train_model                               # noqa: E402
from src.utils import (                                          # noqa: E402
    get_device, load_yaml, make_run_dir, save_yaml, set_seed, setup_logger,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    p.add_argument("--epochs", type=int, default=None, help="override cfg.training.epochs")
    p.add_argument("--families", nargs="+", default=["mlp", "lstm"], help="model families to run")
    p.add_argument("--holdout", default="A", help="which holdout to evaluate on (A/B/C)")
    p.add_argument("--output-name", default="holdout_a", help="suffix for the run directory")
    p.add_argument("--out-dir", default=None, help="explicit output dir; overrides timestamp+output-name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.out_dir is not None:
        run_dir = (ROOT / args.out_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(ROOT / cfg["logging"]["output_root"], name=args.output_name)
    save_yaml(cfg, run_dir / "config.yaml")
    log = setup_logger("milestone2", log_file=run_dir / "run.log")

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    log.info(f"device = {device}, seed = {cfg['seed']}, run_dir = {run_dir}")

    # ---- data ----
    series = load_xtrain(ROOT / cfg["data"]["path"], key=cfg["data"]["key"])
    holdouts = make_holdouts(
        series,
        cfg["data"]["holdouts"],
        window=cfg["window"],
        inner_val_len=cfg["data"]["inner_val_len"],
        scaler_name=cfg["data"]["scaler"],
    )
    holdout = holdouts[args.holdout]
    log.info(
        f"holdout {holdout.name}: train={len(holdout.train)}  "
        f"inner_val={len(holdout.inner_val)}  test={len(holdout.test)}  "
        f"seed_window={len(holdout.seed_window)}"
    )

    summary: dict[str, dict] = {}

    for family in args.families:
        log.info(f"\n=== training {family.upper()} on holdout {holdout.name} ===")
        set_seed(cfg["seed"])
        model = build_model(
            family=family,
            window=cfg["window"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            kernel_size=cfg["model"].get("kernel_size", 3),
            conv_layers=cfg["model"].get("conv_layers", 2),
        )
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"model = {family}, params = {n_params}")

        result = train_model(model, holdout, cfg["training"], device=device, logger=log)
        log.info(
            f"finished {family}: best_inner_val_mae={result['best_inner_val_mae']:.3f} "
            f"@epoch {result['best_epoch']}"
        )

        # ---- recursive 200-step prediction with the BEST checkpoint ----
        model.load_state_dict(result["best_state_dict"])
        model.to(device)
        pred = predict_holdout(model, holdout)
        metrics = compute_metrics(pred["pred_orig"], pred["true_orig"])
        log.info(
            f"holdout {holdout.name} recursive 200-step: "
            f"MAE={metrics['mae']:.3f}  MSE={metrics['mse']:.2f}  "
            f"RMSE={metrics['rmse']:.3f}  NMSE={metrics['nmse']:.4f}"
        )

        plot_path = run_dir / f"{family}_{holdout.name}_prediction.png"
        plot_prediction(
            pred["true_orig"], pred["pred_orig"],
            title=f"{family.upper()} on holdout {holdout.name} — MAE={metrics['mae']:.2f}",
            out_path=plot_path,
            test_start=holdout.test_start,
        )
        log.info(f"saved {plot_path}")

        # ---- save weights + history ----
        torch.save(result["best_state_dict"], run_dir / f"{family}_{holdout.name}.pt")
        with open(run_dir / f"{family}_{holdout.name}_history.json", "w", encoding="utf-8") as f:
            json.dump(result["history"], f, indent=2)
        np.save(run_dir / f"{family}_{holdout.name}_pred.npy", pred["pred_orig"])

        summary[family] = {
            "best_epoch": result["best_epoch"],
            "best_inner_val_mae": result["best_inner_val_mae"],
            "test_metrics": metrics,
            "n_params": n_params,
        }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nsummary written to {summary_path}")
    log.info(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
