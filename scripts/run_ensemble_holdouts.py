"""Milestone 7: ensemble across the three synthetic holdouts.

Each ensemble member is a (model config, seed) pair. For every holdout we:

  1. train each member on the holdout's training portion,
  2. recursively roll out 200 steps INDEPENDENTLY for each member
     (each member uses its OWN predictions — never the ensemble's),
  3. aggregate the per-member trajectories via mean / median / weighted,
  4. compute MAE / MSE on the holdout test segment.

The script then picks the winning ensemble strategy by mean test MAE across
A, B, C.

Members come from:
  - a YAML manifest passed via --manifest (see configs/ensemble_known_good.yaml),
    or
  - a default hardcoded list of the strongest configurations from milestones
    3 and 5 if no manifest is supplied.

Run from project root:
    python scripts/run_ensemble_holdouts.py --seeds 0 1 2 --epochs 200
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
from src.predict import (                                          # noqa: E402
    ensemble_trajectories, inverse_inner_val_weights,
)
from src.search import train_eval_one                             # noqa: E402
from src.utils import (                                            # noqa: E402
    get_device, load_yaml, save_yaml, setup_logger,
)


# Default ensemble: strongest configurations validated through milestones 3-5.
# Each entry is (display_name, model_overrides, training_overrides).
DEFAULT_MEMBERS: list[dict] = [
    {
        "name": "lstm_all_tricks",
        "model": {"family": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
        "training_overrides": {
            "multistep": {"k_max": 5, "anneal_epochs": 150},
            "scheduled_sampling": {"p_max": 0.25, "anneal_epochs": 150},
            "input_noise_sigma": 0.01,
        },
    },
    {
        "name": "mlp_multistep",
        "model": {"family": "mlp", "hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
        "training_overrides": {
            "multistep": {"k_max": 5, "anneal_epochs": 150},
            "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
            "input_noise_sigma": 0.0,
        },
    },
    {
        "name": "tcn_multistep",
        "model": {"family": "tcn", "hidden_dim": 64, "num_layers": 4, "dropout": 0.1,
                  "kernel_size": 3},
        "training_overrides": {
            "multistep": {"k_max": 5, "anneal_epochs": 150},
            "scheduled_sampling": {"p_max": 0.0, "anneal_epochs": 0},
            "input_noise_sigma": 0.0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--manifest", default=None,
                   help="optional YAML with ensemble members; defaults to the hardcoded list")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", default="experiments/baselines/milestone7_ensembles_known_good")
    return p.parse_args()


def _build_member_cfg(base_cfg: dict, member: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"].update(member["model"])
    cfg["training"].update(member.get("training_overrides", {}))
    return cfg


def main() -> int:
    args = parse_args()
    base_cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        base_cfg["training"]["epochs"] = args.epochs

    if args.manifest is not None:
        manifest = load_yaml(ROOT / args.manifest)
        members = manifest["members"]
    else:
        members = DEFAULT_MEMBERS

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml({"members": members}, out_dir / "members.yaml")
    save_yaml(base_cfg, out_dir / "base_config.yaml")
    log = setup_logger("ensemble", log_file=out_dir / "run.log")

    device = get_device(base_cfg.get("device", "auto"))
    log.info(
        f"device={device}  members={[m['name'] for m in members]}  "
        f"seeds={args.seeds}  out={out_dir}"
    )

    series = load_xtrain(ROOT / base_cfg["data"]["path"], key=base_cfg["data"]["key"])
    holdouts = make_holdouts(
        series,
        base_cfg["data"]["holdouts"],
        window=base_cfg["window"],
        inner_val_len=base_cfg["data"]["inner_val_len"],
        scaler_name=base_cfg["data"]["scaler"],
    )

    # ---- run every (member, seed) on every holdout, store predictions ----
    # per_holdout[h] = list of dicts {member, seed, pred, inner_val_mae, individual_mae}
    per_holdout: dict[str, list[dict]] = {name: [] for name in holdouts}

    for member in members:
        m_cfg = _build_member_cfg(base_cfg, member)
        for seed in args.seeds:
            for name, h in holdouts.items():
                log.info(f"\n--- member={member['name']}  seed={seed}  holdout={name} ---")
                res = train_eval_one(m_cfg, h, seed=seed, device=device, logger=log)
                log.info(
                    f"  test MAE={res['test_mae']:.3f}  "
                    f"inner-val MAE={res['best_inner_val_mae']:.3f}  "
                    f"best@e{res['best_epoch']}"
                )
                per_holdout[name].append({
                    "member": member["name"],
                    "seed": seed,
                    "pred": res["pred_orig"],
                    "individual_mae": res["test_mae"],
                    "individual_mse": res["test_mse"],
                    "inner_val_mae": res["best_inner_val_mae"],
                    "best_epoch": res["best_epoch"],
                })

    # ---- aggregate per holdout ----
    rows: list[dict] = []
    for name, h in holdouts.items():
        members_runs = per_holdout[name]
        trajectories = [r["pred"] for r in members_runs]
        weights = inverse_inner_val_weights([r["inner_val_mae"] for r in members_runs])

        agg_results: dict[str, dict] = {}
        for method in ("mean", "median", "weighted"):
            kwargs = {"weights": weights} if method == "weighted" else {}
            pred_ens = ensemble_trajectories(trajectories, method=method, **kwargs)
            metrics = compute_metrics(pred_ens, h.test)
            agg_results[method] = {**metrics, "pred": pred_ens}
            rows.append({
                "holdout": name, "strategy": method,
                "mae": metrics["mae"], "mse": metrics["mse"], "nmse": metrics["nmse"],
                "n_members": len(trajectories),
            })

        # also: best-individual for reference
        best_run = min(members_runs, key=lambda r: r["individual_mae"])
        rows.append({
            "holdout": name, "strategy": f"best_individual ({best_run['member']}/s{best_run['seed']})",
            "mae": best_run["individual_mae"], "mse": best_run["individual_mse"],
            "nmse": float("nan"), "n_members": 1,
        })

        # plot mean ensemble prediction
        plot_prediction(
            h.test, agg_results["mean"]["pred"],
            title=f"Mean ensemble  Holdout {name}  MAE={agg_results['mean']['mae']:.2f}",
            out_path=out_dir / f"{name}_mean.png",
            test_start=h.test_start,
        )
        plot_prediction(
            h.test, agg_results["median"]["pred"],
            title=f"Median ensemble  Holdout {name}  MAE={agg_results['median']['mae']:.2f}",
            out_path=out_dir / f"{name}_median.png",
            test_start=h.test_start,
        )

    # ---- summary ----
    by_strategy: dict[str, list[float]] = {}
    for r in rows:
        if r["strategy"] in ("mean", "median", "weighted"):
            by_strategy.setdefault(r["strategy"], []).append(r["mae"])

    overall = {s: float(np.mean(v)) for s, v in by_strategy.items()}
    winning = min(overall, key=overall.get)

    log.info("\n=== ensemble summary ===")
    log.info(json.dumps(overall, indent=2))
    log.info(f"winning strategy by mean MAE: {winning} ({overall[winning]:.3f})")

    md = [
        "# Milestone 7 — ensembling across the three holdouts",
        "",
        f"Members ({len(members)}): "
        + ", ".join(f"`{m['name']}`" for m in members),
        f"Seeds: {list(args.seeds)} (so {len(members) * len(args.seeds)} ensemble members per holdout).",
        "",
        f"**Winning strategy by mean MAE across A/B/C: `{winning}` "
        f"({overall[winning]:.3f})**",
        "",
        "| holdout | strategy | MAE | MSE | NMSE | n_members |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        nmse = "—" if (isinstance(r["nmse"], float) and np.isnan(r["nmse"])) else f"{r['nmse']:.3f}"
        md.append(
            f"| {r['holdout']} | {r['strategy']} | {r['mae']:.2f} | "
            f"{r['mse']:.1f} | {nmse} | {r['n_members']} |"
        )
    md += [
        "",
        "## Strategy mean across holdouts",
        "",
        "| strategy | mean test MAE |",
        "| --- | ---: |",
    ]
    for s, v in sorted(overall.items(), key=lambda kv: kv[1]):
        md.append(f"| {s} | {v:.3f} |")

    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "overall": overall, "winning": winning}, f, indent=2)
    log.info(f"\nwrote {out_dir / 'REPORT.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
