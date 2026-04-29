"""Milestone 8: full-data retrain + final 200-step prediction.

For every member in the ensemble manifest:
  1. fit a scaler on the full Xtrain (1..1000),
  2. retrain the member from scratch on the full Xtrain for the configured
     number of epochs (no validation; spec says use median best epoch from
     the search/holdout phase),
  3. recursively predict 200 steps starting from the last `window` points
     of Xtrain,
  4. inverse-transform to the original 2..255 scale.

All member trajectories are aggregated by the chosen ensemble strategy
(default: median, robust to single-seed collapses) and the result is
written to:

    experiments/final/predictions.npy        # shape (200,)
    experiments/final/predictions.csv        # one column, 200 rows
    experiments/final/metadata.json          # ensemble composition + per-member epochs
    experiments/final/ensemble_preview.png   # plot of last 100 of Xtrain + the 200-step forecast

Manifest format (YAML):
    members:
      - name: lstm_all_tricks
        epochs: 175                         # required: total epochs to train on full Xtrain
        seeds: [0, 1, 2, 3, 4]              # one final model per seed
        model:
          family: lstm
          hidden_dim: 64
          num_layers: 2
          dropout: 0.1
        training_overrides:
          multistep:           {k_max: 5, anneal_epochs: 130}
          scheduled_sampling:  {p_max: 0.25, anneal_epochs: 130}
          input_noise_sigma:   0.01
        window: 30                          # optional override; defaults to base config window
        scaler: minmax_0_1                  # optional override

Usage:
    python scripts/run_final_predictions.py --manifest configs/final_ensemble.yaml \
        --strategy median
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain, make_scaler                    # noqa: E402
from src.models import build_model                                # noqa: E402
from src.predict import (                                          # noqa: E402
    ensemble_trajectories, inverse_inner_val_weights, recursive_rollout,
)
from src.train import train_full                                  # noqa: E402
from src.utils import (                                            # noqa: E402
    get_device, load_yaml, save_yaml, set_seed, setup_logger,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--manifest", required=True,
                   help="YAML describing ensemble members and per-member epochs")
    p.add_argument("--strategy", default="median",
                   choices=["mean", "median", "weighted"])
    p.add_argument("--out-dir", default="experiments/final")
    p.add_argument("--horizon", type=int, default=200)
    return p.parse_args()


def _build_member_cfg(base_cfg: dict, member: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"].update(member.get("model", {}))
    cfg["training"].update(member.get("training_overrides", {}))
    if "window" in member:
        cfg["window"] = int(member["window"])
    if "scaler" in member:
        cfg["data"]["scaler"] = member["scaler"]
    return cfg


def main() -> int:
    args = parse_args()
    base_cfg = load_yaml(ROOT / args.config)
    manifest = load_yaml(ROOT / args.manifest)
    members = manifest["members"]

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml({"members": members}, out_dir / "members.yaml")
    log = setup_logger("final", log_file=out_dir / "run.log")

    device = get_device(base_cfg.get("device", "auto"))
    series = load_xtrain(ROOT / base_cfg["data"]["path"], key=base_cfg["data"]["key"])
    log.info(f"loaded Xtrain shape={series.shape}, range=[{series.min():.0f},{series.max():.0f}]")

    # ---- collect (member × seed) trajectories ----
    trajectories: list[np.ndarray] = []
    inner_val_proxies: list[float] = []   # for weighted ensemble: prefer using
                                          # search-time inner-val if provided in
                                          # the manifest, otherwise weight all 1
    member_records: list[dict] = []

    for member in members:
        m_cfg = _build_member_cfg(base_cfg, member)
        window = int(m_cfg["window"])
        epochs = int(member["epochs"])
        seeds = list(member.get("seeds", [0]))
        scaler_name = m_cfg["data"]["scaler"]

        # scaler fit on the FULL Xtrain (no holdout, no inner val)
        scaler = make_scaler(scaler_name).fit(series)
        series_scaled = scaler.transform(series).astype(np.float32)
        seed_window_scaled = series_scaled[-window:]

        for seed in seeds:
            tag = f"{member['name']}_s{seed}"
            log.info(
                f"\n--- training {tag} on full Xtrain  "
                f"(family={m_cfg['model']['family']}, window={window}, epochs={epochs}) ---"
            )
            set_seed(seed)
            model = build_model(
                family=m_cfg["model"]["family"],
                window=window,
                hidden_dim=m_cfg["model"]["hidden_dim"],
                num_layers=m_cfg["model"]["num_layers"],
                dropout=m_cfg["model"]["dropout"],
                kernel_size=m_cfg["model"].get("kernel_size", 3),
                conv_layers=m_cfg["model"].get("conv_layers", 2),
            )
            train_full(
                model, series_scaled, m_cfg["training"],
                window=window, epochs=epochs, device=device, logger=log,
            )

            # recursive rollout in scaled space, then inverse_transform
            pred_scaled = recursive_rollout(model, seed_window_scaled, n_steps=args.horizon)
            pred_orig = scaler.inverse_transform(pred_scaled)
            trajectories.append(pred_orig)
            inner_val_proxies.append(float(member.get("inner_val_mae_proxy", 1.0)))
            member_records.append({
                "tag": tag, "member": member["name"], "seed": seed,
                "epochs": epochs, "window": window, "scaler": scaler_name,
                "family": m_cfg["model"]["family"],
                "model": m_cfg["model"], "training": m_cfg["training"],
            })

    # ---- aggregate ----
    if args.strategy == "weighted":
        weights = inverse_inner_val_weights(inner_val_proxies)
    else:
        weights = None
    final_pred = ensemble_trajectories(trajectories, method=args.strategy, weights=weights)

    # ---- save ----
    np.save(out_dir / "predictions.npy", final_pred)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction"])
        for v in final_pred:
            writer.writerow([f"{v:.6f}"])

    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "strategy": args.strategy,
        "n_members": len(trajectories),
        "horizon": args.horizon,
        "members": member_records,
        "summary": {
            "min": float(final_pred.min()),
            "max": float(final_pred.max()),
            "mean": float(final_pred.mean()),
            "std": float(final_pred.std()),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ---- preview plot: last 100 of Xtrain + the forecast ----
    fig, ax = plt.subplots(figsize=(13, 5))
    n_train = len(series)
    history_idx = np.arange(n_train - 100 + 1, n_train + 1)
    forecast_idx = np.arange(n_train + 1, n_train + 1 + args.horizon)
    ax.plot(history_idx, series[-100:], color="black", lw=0.9, label="Xtrain (last 100)")
    # plot every individual trajectory faintly so spread is visible
    for traj in trajectories:
        ax.plot(forecast_idx, traj, color="grey", lw=0.5, alpha=0.4)
    ax.plot(forecast_idx, final_pred, color="crimson", lw=1.2, label=f"{args.strategy} ensemble")
    ax.axvline(n_train + 0.5, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("sample index")
    ax.set_ylabel("intensity")
    ax.set_title(f"Final ensemble forecast — {len(trajectories)} members, {args.strategy}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ensemble_preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\nwrote {out_dir / 'predictions.npy'}  ({final_pred.shape}, "
             f"min={final_pred.min():.2f}, max={final_pred.max():.2f})")
    log.info(f"wrote {out_dir / 'predictions.csv'}")
    log.info(f"wrote {out_dir / 'metadata.json'}")
    log.info(f"wrote {out_dir / 'ensemble_preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
