"""Milestone 3 ablation: quantify the effect of each training trick on Holdout A.

Variants compared (all use the same base config — only the training-trick knobs change):

    baseline    : k_max=1, p_max=0,    noise=0       (= milestone 2)
    +multistep  : k_max=5, p_max=0,    noise=0
    +scheduled  : k_max=5, p_max=0.25, noise=0
    +inputnoise : k_max=5, p_max=0.25, noise=0.01     (= "all tricks")

Anneal schedule: k_max and p_max climb over 75% of training (150 of 200
epochs); the user spec is "anneal from 1 to k_max by the end of training".

Runs every (variant, family) with multiple seeds and reports mean + worst test
MAE per cell, since chaotic recursive rollouts have high seed variance.

Run from project root:
    python scripts/ablation_holdout_a.py [--seeds 0 1 2] [--epochs 200]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain, make_holdouts                  # noqa: E402
from src.evaluate import compute_metrics, plot_prediction        # noqa: E402
from src.models import build_model                               # noqa: E402
from src.predict import predict_holdout                          # noqa: E402
from src.train import train_model                                # noqa: E402
from src.utils import (                                           # noqa: E402
    get_device, load_yaml, save_yaml, set_seed, setup_logger,
)


def make_variants(anneal: int) -> list[tuple[str, dict]]:
    return [
        ("baseline",    {"multistep": {"k_max": 1, "anneal_epochs": 0},
                         "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
                         "input_noise_sigma": 0.0}),
        ("multistep",   {"multistep": {"k_max": 5, "anneal_epochs": anneal},
                         "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
                         "input_noise_sigma": 0.0}),
        ("multi+sched", {"multistep": {"k_max": 5, "anneal_epochs": anneal},
                         "scheduled_sampling": {"p_max": 0.25, "anneal_epochs": anneal},
                         "input_noise_sigma": 0.0}),
        ("all_tricks",  {"multistep": {"k_max": 5, "anneal_epochs": anneal},
                         "scheduled_sampling": {"p_max": 0.25, "anneal_epochs": anneal},
                         "input_noise_sigma": 0.01}),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--anneal-frac", type=float, default=0.75,
                   help="fraction of training over which k and p anneal to their max")
    p.add_argument("--families", nargs="+", default=["mlp", "lstm"])
    p.add_argument("--holdout", default="A")
    p.add_argument("--out-dir", default="experiments/baselines/milestone3_ablation_holdout_a")
    return p.parse_args()


def _format_mean_std(values: list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 1:
        return f"{arr[0]:.2f}"
    return f"{arr.mean():.2f} ± {arr.std(ddof=0):.2f}"


def main() -> int:
    args = parse_args()
    base_cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        base_cfg["training"]["epochs"] = args.epochs

    epochs = int(base_cfg["training"]["epochs"])
    anneal = int(round(epochs * args.anneal_frac))
    variants = make_variants(anneal)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(base_cfg, out_dir / "base_config.yaml")
    log = setup_logger("ablation", log_file=out_dir / "run.log")

    device = get_device(base_cfg.get("device", "auto"))
    log.info(
        f"device={device}  seeds={args.seeds}  epochs={epochs}  "
        f"anneal_epochs={anneal}  out_dir={out_dir}"
    )

    # ---- data ----
    series = load_xtrain(ROOT / base_cfg["data"]["path"], key=base_cfg["data"]["key"])
    holdouts = make_holdouts(
        series,
        base_cfg["data"]["holdouts"],
        window=base_cfg["window"],
        inner_val_len=base_cfg["data"]["inner_val_len"],
        scaler_name=base_cfg["data"]["scaler"],
    )
    holdout = holdouts[args.holdout]
    log.info(
        f"holdout {holdout.name}: train={len(holdout.train)}  "
        f"inner_val={len(holdout.inner_val)}  test={len(holdout.test)}"
    )

    # ---- run grid ----
    rows: list[dict] = []
    for variant_name, overrides in variants:
        for family in args.families:
            cfg = copy.deepcopy(base_cfg)
            cfg["training"].update(overrides)

            seed_metrics: list[dict] = []
            best_pred = None
            best_mae_so_far = float("inf")
            for seed in args.seeds:
                tag = f"{family}_{variant_name}_s{seed}"
                log.info(
                    f"\n=== {tag} === multistep={cfg['training']['multistep']}  "
                    f"sched={cfg['training']['scheduled_sampling']}  "
                    f"noise={cfg['training']['input_noise_sigma']}"
                )

                set_seed(seed)
                model = build_model(
                    family=family,
                    window=cfg["window"],
                    hidden_dim=cfg["model"]["hidden_dim"],
                    num_layers=cfg["model"]["num_layers"],
                    dropout=cfg["model"]["dropout"],
                )
                result = train_model(model, holdout, cfg["training"], device=device, logger=log)
                model.load_state_dict(result["best_state_dict"])
                model.to(device)
                pred = predict_holdout(model, holdout)
                metrics = compute_metrics(pred["pred_orig"], pred["true_orig"])
                log.info(
                    f"  seed {seed}: best_inner_val_mae={result['best_inner_val_mae']:.3f}@e{result['best_epoch']}  "
                    f"test MAE={metrics['mae']:.3f}  MSE={metrics['mse']:.2f}  NMSE={metrics['nmse']:.4f}"
                )
                seed_metrics.append({
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "best_inner_val_mae": result["best_inner_val_mae"],
                    "test_mae": metrics["mae"],
                    "test_mse": metrics["mse"],
                    "test_rmse": metrics["rmse"],
                    "test_nmse": metrics["nmse"],
                })
                if metrics["mae"] < best_mae_so_far:
                    best_mae_so_far = metrics["mae"]
                    best_pred = pred

            # save best-of-seeds plot per (variant, family)
            tag = f"{family}_{variant_name}"
            plot_prediction(
                best_pred["true_orig"], best_pred["pred_orig"],
                title=f"{family.upper()} {variant_name} on holdout {holdout.name}  "
                      f"best-seed MAE={best_mae_so_far:.2f}",
                out_path=out_dir / f"{tag}_bestseed.png",
                test_start=holdout.test_start,
            )

            mae_vals = [s["test_mae"] for s in seed_metrics]
            mse_vals = [s["test_mse"] for s in seed_metrics]
            nmse_vals = [s["test_nmse"] for s in seed_metrics]
            rows.append({
                "family": family,
                "variant": variant_name,
                "seeds": list(args.seeds),
                "per_seed": seed_metrics,
                "test_mae_mean": float(np.mean(mae_vals)),
                "test_mae_std":  float(np.std(mae_vals, ddof=0)),
                "test_mae_worst": float(np.max(mae_vals)),
                "test_mae_best": float(np.min(mae_vals)),
                "test_mse_mean": float(np.mean(mse_vals)),
                "test_nmse_mean": float(np.mean(nmse_vals)),
            })

    # ---- summary ----
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md = [
        "# Milestone 3 ablation — Holdout A",
        "",
        f"Seeds {list(args.seeds)} • {epochs} epochs • anneal_epochs={anneal} "
        f"({int(args.anneal_frac*100)}% of training) • window {base_cfg['window']} "
        f"• MLP hidden 64×2, LSTM hidden 64×2.",
        "",
        "Test MAE in original 2–255 scale, mean ± std across seeds. "
        "**best** = single-seed minimum, **worst** = single-seed maximum.",
        "",
        "| family | variant | inner-val MAE (mean) | test MAE (mean ± std) | best | worst | NMSE (mean) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        inner_mean = np.mean([s["best_inner_val_mae"] for s in r["per_seed"]])
        md.append(
            f"| {r['family']} | {r['variant']} | {inner_mean:.2f} | "
            f"{r['test_mae_mean']:.2f} ± {r['test_mae_std']:.2f} | "
            f"{r['test_mae_best']:.2f} | {r['test_mae_worst']:.2f} | "
            f"{r['test_nmse_mean']:.3f} |"
        )

    md += [
        "",
        "## Deltas vs baseline (mean test MAE)",
        "",
    ]
    by_family: dict[str, dict[str, float]] = {}
    for r in rows:
        by_family.setdefault(r["family"], {})[r["variant"]] = r["test_mae_mean"]
    for fam, vals in by_family.items():
        base = vals.get("baseline")
        if base is None:
            continue
        md.append(f"### {fam}")
        md.append("| variant | mean test MAE | Δ vs baseline | Δ % |")
        md.append("| --- | ---: | ---: | ---: |")
        for vn, _ in variants:
            if vn in vals:
                v = vals[vn]
                d = v - base
                p = 100.0 * d / base if base > 0 else 0.0
                md.append(f"| {vn} | {v:.2f} | {d:+.2f} | {p:+.1f}% |")
        md.append("")

    md_path = out_dir / "REPORT.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info(f"\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
