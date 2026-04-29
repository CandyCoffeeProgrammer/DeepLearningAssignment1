"""Milestone 5 driver: run one configuration across the three synthetic holdouts.

Confirms the three-holdout protocol works end-to-end and produces the
official ranking metric (mean recursive 200-step MAE across A, B, C).

Run from project root:
    python scripts/run_three_holdouts.py --family lstm --variant all_tricks --seeds 0 1 2
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
from src.evaluate import plot_prediction                          # noqa: E402
from src.search import evaluate_config, train_eval_one            # noqa: E402
from src.utils import (                                            # noqa: E402
    get_device, load_yaml, save_yaml, setup_logger,
)


# Same trick variants as the Holdout-A ablation, kept here for convenience.
def make_variant_overrides(name: str, anneal: int) -> dict:
    table = {
        "baseline":   {"multistep": {"k_max": 1, "anneal_epochs": 0},
                       "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
                       "input_noise_sigma": 0.0},
        "multistep":  {"multistep": {"k_max": 5, "anneal_epochs": anneal},
                       "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
                       "input_noise_sigma": 0.0},
        "multi+sched":{"multistep": {"k_max": 5, "anneal_epochs": anneal},
                       "scheduled_sampling": {"p_max": 0.25, "anneal_epochs": anneal},
                       "input_noise_sigma": 0.0},
        "all_tricks": {"multistep": {"k_max": 5, "anneal_epochs": anneal},
                       "scheduled_sampling": {"p_max": 0.25, "anneal_epochs": anneal},
                       "input_noise_sigma": 0.01},
    }
    if name not in table:
        raise SystemExit(f"unknown variant {name!r}; available: {sorted(table)}")
    return table[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--family", required=True,
                   choices=["mlp", "lstm", "gru", "cnn1d", "cnn_lstm", "tcn"])
    p.add_argument("--variant", default="multistep",
                   choices=["baseline", "multistep", "multi+sched", "all_tricks"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--anneal-frac", type=float, default=0.75)
    p.add_argument("--out-dir", default=None,
                   help="default: experiments/baselines/milestone5_<family>_<variant>")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    epochs = int(cfg["training"]["epochs"])
    anneal = int(round(epochs * args.anneal_frac))
    cfg["training"].update(make_variant_overrides(args.variant, anneal))
    cfg["model"]["family"] = args.family

    out_dir = (
        ROOT / (args.out_dir or f"experiments/baselines/milestone5_{args.family}_{args.variant}")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config.yaml")
    log = setup_logger("m5", log_file=out_dir / "run.log")

    device = get_device(cfg.get("device", "auto"))
    log.info(
        f"device={device}  family={args.family}  variant={args.variant}  "
        f"seeds={args.seeds}  epochs={epochs}  anneal={anneal}  out={out_dir}"
    )

    # ---- holdouts ----
    series = load_xtrain(ROOT / cfg["data"]["path"], key=cfg["data"]["key"])
    holdouts = make_holdouts(
        series,
        cfg["data"]["holdouts"],
        window=cfg["window"],
        inner_val_len=cfg["data"]["inner_val_len"],
        scaler_name=cfg["data"]["scaler"],
    )
    for name, h in holdouts.items():
        log.info(f"holdout {name}: train={len(h.train)} inner_val={len(h.inner_val)} test={len(h.test)}")

    # ---- evaluate config across (holdouts × seeds), separately to keep predictions ----
    all_predictions: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict] = []
    for name, h in holdouts.items():
        all_predictions[name] = {}
        for seed in args.seeds:
            log.info(f"\n--- holdout {name}  seed {seed} ---")
            res = train_eval_one(cfg, h, seed, device, logger=log)
            log.info(
                f"  test MAE={res['test_mae']:.3f}  MSE={res['test_mse']:.2f}  "
                f"NMSE={res['test_nmse']:.4f}  best@e{res['best_epoch']}"
            )
            all_predictions[name][seed] = res["pred_orig"]
            rows.append({
                "holdout": name, "seed": seed,
                "best_epoch": res["best_epoch"],
                "best_inner_val_mae": res["best_inner_val_mae"],
                "test_mae": res["test_mae"],
                "test_mse": res["test_mse"],
                "test_nmse": res["test_nmse"],
            })

    # ---- aggregate ----
    by_h: dict[str, dict[str, float]] = {}
    for name in holdouts:
        maes = [r["test_mae"] for r in rows if r["holdout"] == name]
        mses = [r["test_mse"] for r in rows if r["holdout"] == name]
        nmses = [r["test_nmse"] for r in rows if r["holdout"] == name]
        by_h[name] = {
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes, ddof=0)),
            "mae_best": float(np.min(maes)), "mae_worst": float(np.max(maes)),
            "mse_mean": float(np.mean(mses)),
            "nmse_mean": float(np.mean(nmses)),
        }
    overall_mae = float(np.mean([r["test_mae"] for r in rows]))
    overall_worst = max(by_h.items(), key=lambda kv: kv[1]["mae_mean"])

    # ---- save best-seed prediction plot per holdout ----
    for name, h in holdouts.items():
        seed_maes = {s: float(np.mean(np.abs(p - h.test))) for s, p in all_predictions[name].items()}
        best_seed = min(seed_maes, key=seed_maes.get)
        plot_prediction(
            h.test, all_predictions[name][best_seed],
            title=f"{args.family.upper()} {args.variant}  Holdout {name}  "
                  f"best-seed (s{best_seed}) MAE={seed_maes[best_seed]:.2f}",
            out_path=out_dir / f"{name}_bestseed.png",
            test_start=h.test_start,
        )

    # ---- summary ----
    summary = {
        "family": args.family, "variant": args.variant,
        "seeds": list(args.seeds), "epochs": epochs, "anneal": anneal,
        "overall_mean_mae": overall_mae,
        "worst_holdout": overall_worst[0],
        "worst_holdout_mean_mae": overall_worst[1]["mae_mean"],
        "per_holdout": by_h,
        "per_run": rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [
        f"# Three-holdout evaluation — {args.family.upper()} / {args.variant}",
        "",
        f"Seeds {list(args.seeds)} • {epochs} epochs • anneal={anneal} • "
        f"window {cfg['window']} • hidden {cfg['model']['hidden_dim']} × "
        f"{cfg['model']['num_layers']} layers.",
        "",
        f"**Overall mean test MAE: {overall_mae:.3f}** "
        f"(worst holdout: {overall_worst[0]} @ {overall_worst[1]['mae_mean']:.3f})",
        "",
        "Per-holdout (recursive 200-step on the test segment, original 2–255 scale):",
        "",
        "| holdout | mean MAE | std | best | worst | mean MSE | mean NMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in holdouts:
        s = by_h[name]
        md.append(
            f"| {name} | {s['mae_mean']:.2f} | {s['mae_std']:.2f} | "
            f"{s['mae_best']:.2f} | {s['mae_worst']:.2f} | "
            f"{s['mse_mean']:.1f} | {s['nmse_mean']:.3f} |"
        )
    md += [
        "",
        "Per-run breakdown:",
        "",
        "| holdout | seed | best_epoch | inner-val MAE | test MAE | test MSE | test NMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['holdout']} | {r['seed']} | {r['best_epoch']} | "
            f"{r['best_inner_val_mae']:.2f} | {r['test_mae']:.2f} | "
            f"{r['test_mse']:.1f} | {r['test_nmse']:.4f} |"
        )

    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"\nwrote {out_dir / 'REPORT.md'}")
    log.info(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
