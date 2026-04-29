"""Run the full Optuna sweep across all six families in a single Python process.

This is the long-running version of `scripts/run_search.sh`. The benefit of
keeping everything in one Python process: if the parent shell dies (e.g. the
agent's Bash background task hits its 10-minute timeout), this Python process
becomes orphaned but **keeps running** until completion — so the 8–14 hour
sweep doesn't depend on the launcher staying alive.

Spec defaults: 200 epochs, FULL search caps, 100 trials for the priority
families and 40 for the baselines. Override via env vars or CLI for shorter
sweeps.

Live progress: every eval cycle (~5 epochs) the trainer atomically rewrites
`experiments/search/STATUS.json` with current trial / family / holdout / epoch
/ loss / best so far / elapsed time. `cat` it from another shell to track.

Usage:
    python scripts/run_full_search.py
    python scripts/run_full_search.py --epochs 60 --priority 15 --baseline 10 --lite
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain                                  # noqa: E402
from src.search import (                                            # noqa: E402
    FULL_SEARCH_CAPS, LITE_SEARCH_CAPS, run_search,
)
from src.utils import get_device, load_yaml, setup_logger          # noqa: E402


PRIORITY_FAMILIES = ["lstm", "gru", "tcn", "cnn_lstm"]
BASELINE_FAMILIES = ["mlp", "cnn1d"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 200)))
    p.add_argument("--priority", type=int, default=int(os.environ.get("N_PRIORITY", 100)),
                   help="trials per priority family (lstm/gru/tcn/cnn_lstm)")
    p.add_argument("--baseline", type=int, default=int(os.environ.get("N_BASELINE", 40)),
                   help="trials per baseline family (mlp/cnn1d)")
    p.add_argument("--lite", action="store_true",
                   help="use LITE_SEARCH_CAPS (default is FULL)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="experiments/search")
    p.add_argument("--families", nargs="+", default=None,
                   help="optional: restrict to a subset of families")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    cfg["training"]["epochs"] = args.epochs

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logger("search.full", log_file=out_root / "search.log")
    storage = f"sqlite:///{(out_root / 'optuna.db').as_posix()}"
    status_path = out_root / "STATUS.json"
    caps = LITE_SEARCH_CAPS if args.lite else FULL_SEARCH_CAPS
    caps_label = "LITE" if args.lite else "FULL"

    families = args.families or (PRIORITY_FAMILIES + BASELINE_FAMILIES)
    counts = {**{f: args.priority for f in PRIORITY_FAMILIES},
              **{f: args.baseline for f in BASELINE_FAMILIES}}

    device = get_device(cfg.get("device", "auto"))
    series = load_xtrain(ROOT / cfg["data"]["path"], key=cfg["data"]["key"])

    start_ts = datetime.now()
    log.info(
        f"=== full search start  {start_ts.isoformat()}  caps={caps_label}  "
        f"epochs={args.epochs}  priority={args.priority}  baseline={args.baseline}  "
        f"families={families}  device={device} ==="
    )

    for family in families:
        n_trials = counts.get(family, 30)
        family_dir = out_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        leaderboard = family_dir / "trials.csv"

        log.info(f"\n=== {family.upper()} ({n_trials} trials, {caps_label} caps) ===")
        try:
            run_search(
                family=family,
                base_cfg=cfg,
                n_trials=n_trials,
                series=series,
                device=device,
                seed=args.seed,
                storage=storage,
                study_name=f"santa_fe_{family}",
                leaderboard_path=leaderboard,
                logger=log,
                caps=caps,
                status_path=status_path,
            )
        except Exception:
            log.error(f"{family} crashed; continuing to next family\n{traceback.format_exc()}")
            continue

        # Persist per-family best.json (mirrors run_optuna_search.py)
        try:
            import optuna, json
            study = optuna.load_study(study_name=f"santa_fe_{family}", storage=storage)
            if len(study.trials):
                best = study.best_trial
                with open(family_dir / "best.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "family": family,
                        "trial": best.number,
                        "mean_mae": best.value,
                        "worst_mae": best.user_attrs.get("worst_mae"),
                        "per_holdout_mae": best.user_attrs.get("per_holdout_mae"),
                        "params": best.params,
                    }, f, indent=2)
                log.info(f"{family}: best trial #{best.number}  mean_mae={best.value:.4f}")
        except Exception:
            log.warning(f"{family}: failed to write best.json:\n{traceback.format_exc()}")

    end_ts = datetime.now()
    log.info(f"\n=== full search complete  start={start_ts}  end={end_ts}  "
             f"elapsed={end_ts - start_ts} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
