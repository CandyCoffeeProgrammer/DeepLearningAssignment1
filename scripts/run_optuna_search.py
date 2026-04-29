"""Optuna search driver — milestone 6.

Examples:
    # 5-trial validation sweep for one family
    python scripts/run_optuna_search.py --family lstm --n-trials 5 --epochs 60

    # Per-spec budgets
    python scripts/run_optuna_search.py --family lstm     --n-trials 100
    python scripts/run_optuna_search.py --family gru      --n-trials 100
    python scripts/run_optuna_search.py --family tcn      --n-trials 100
    python scripts/run_optuna_search.py --family cnn_lstm --n-trials 100
    python scripts/run_optuna_search.py --family mlp      --n-trials  40
    python scripts/run_optuna_search.py --family cnn1d    --n-trials  40

The Optuna study is persisted to a SQLite DB so it can resume after a crash.
A per-trial CSV is written for downstream leaderboard inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain                                  # noqa: E402
from src.search import run_search                                  # noqa: E402
from src.utils import get_device, load_yaml, setup_logger          # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--family", required=True,
                   choices=["mlp", "lstm", "gru", "cnn1d", "cnn_lstm", "tcn"])
    p.add_argument("--n-trials", type=int, required=True)
    p.add_argument("--seed", type=int, default=42, help="trial seed (single seed per trial)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override training.epochs for the search (default 200)")
    p.add_argument("--out-root", default="experiments/search",
                   help="search artefacts go here")
    p.add_argument("--storage", default=None,
                   help="Optuna SQLite storage URL (default: sqlite:///experiments/search/optuna.db)")
    p.add_argument("--study-name", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    out_root = (ROOT / args.out_root).resolve()
    family_dir = out_root / args.family
    family_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(f"search.{args.family}", log_file=family_dir / "search.log")

    storage = args.storage or f"sqlite:///{(out_root / 'optuna.db').as_posix()}"
    study_name = args.study_name or f"santa_fe_{args.family}"
    leaderboard = family_dir / "trials.csv"

    device = get_device(cfg.get("device", "auto"))
    series = load_xtrain(ROOT / cfg["data"]["path"], key=cfg["data"]["key"])

    log.info(
        f"family={args.family}  n_trials={args.n_trials}  seed={args.seed}  "
        f"epochs={cfg['training']['epochs']}  device={device}"
    )
    log.info(f"storage={storage}  study={study_name}  leaderboard={leaderboard}")

    study = run_search(
        family=args.family,
        base_cfg=cfg,
        n_trials=args.n_trials,
        series=series,
        device=device,
        seed=args.seed,
        storage=storage,
        study_name=study_name,
        leaderboard_path=leaderboard,
        logger=log,
    )

    # ---- summary ----
    log.info(f"\nfinished {args.family}: {len(study.trials)} trials")
    if len(study.trials):
        best = study.best_trial
        log.info(f"best trial #{best.number}  mean_mae={best.value:.4f}")
        log.info(f"best params:\n{json.dumps(best.params, indent=2)}")
        log.info(f"best per-holdout:\n{json.dumps(best.user_attrs.get('per_holdout_mae', {}), indent=2)}")

        # write best.json for downstream consumption
        best_path = family_dir / "best.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({
                "family": args.family,
                "trial": best.number,
                "mean_mae": best.value,
                "worst_mae": best.user_attrs.get("worst_mae"),
                "per_holdout_mae": best.user_attrs.get("per_holdout_mae"),
                "params": best.params,
            }, f, indent=2)
        log.info(f"wrote {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
