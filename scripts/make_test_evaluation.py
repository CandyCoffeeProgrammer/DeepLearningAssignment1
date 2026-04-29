"""Milestone 9: score the final ensemble against the real test set.

Once `Xtest.mat` arrives, run:

    python scripts/make_test_evaluation.py --xtest data/Xtest.mat

This script does NO training and NO inference. It loads:
  - the saved final predictions (`experiments/final/predictions.npy`),
  - the released `Xtest.mat`,
and computes MAE / MSE / RMSE / NMSE in the original integer scale, plus a
comparison plot at `experiments/final/test_comparison.png`. Everything is
written into the output directory.

Args:
    --xtest        path to Xtest.mat (required)
    --predictions  path to predictions.npy (default: experiments/final/predictions.npy)
    --xtest-key    .mat dict key (default: Xtest)
    --out-dir      where to save results (default: experiments/final)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics, plot_prediction        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xtest", required=True, help="path to Xtest.mat")
    p.add_argument("--predictions", default="experiments/final/predictions.npy")
    p.add_argument("--xtest-key", default="Xtest")
    p.add_argument("--out-dir", default="experiments/final")
    p.add_argument("--xtrain", default="data/Xtrain.mat",
                   help="optional — when present, the comparison plot extends the x-axis "
                        "from the end of Xtrain")
    p.add_argument("--xtrain-key", default="Xtrain")
    return p.parse_args()


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    raw = sio.loadmat(str(path))
    if key not in raw:
        keys = [k for k in raw if not k.startswith("_")]
        raise KeyError(f"key {key!r} not found in {path}; available: {keys}")
    return np.asarray(raw[key]).astype(np.float64).reshape(-1)


def main() -> int:
    args = parse_args()
    xtest_path = (ROOT / args.xtest).resolve() if not Path(args.xtest).is_absolute() else Path(args.xtest)
    pred_path = (ROOT / args.predictions).resolve() if not Path(args.predictions).is_absolute() else Path(args.predictions)
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise SystemExit(f"predictions not found: {pred_path}")
    if not xtest_path.exists():
        raise SystemExit(f"Xtest not found: {xtest_path}")

    pred = np.load(pred_path).reshape(-1)
    truth = _load_mat_array(xtest_path, args.xtest_key)

    if pred.shape != truth.shape:
        raise SystemExit(
            f"prediction length {pred.shape[0]} != Xtest length {truth.shape[0]}"
        )

    metrics = compute_metrics(pred, truth)
    print("=" * 60)
    print("Final test evaluation")
    print("=" * 60)
    print(f"predictions: {pred_path}")
    print(f"Xtest      : {xtest_path}  (length {truth.shape[0]})")
    print()
    print(f"  MAE   = {metrics['mae']:.4f}")
    print(f"  MSE   = {metrics['mse']:.4f}")
    print(f"  RMSE  = {metrics['rmse']:.4f}")
    print(f"  NMSE  = {metrics['nmse']:.6f}")
    print("=" * 60)

    metadata = {
        "scored_at": datetime.utcnow().isoformat() + "Z",
        "predictions_path": str(pred_path),
        "xtest_path": str(xtest_path),
        "n_steps": int(truth.shape[0]),
        "metrics": metrics,
    }
    metadata_path = out_dir / "test_metrics.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {metadata_path}")

    # ---- plot ----
    xtrain_path = ROOT / args.xtrain
    if xtrain_path.exists():
        xtrain = _load_mat_array(xtrain_path, args.xtrain_key)
        n_train = len(xtrain)
        idx = np.arange(n_train + 1, n_train + 1 + len(truth))
        # zoomed comparison + extended-context plot
        fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                                 gridspec_kw={"height_ratios": [2, 1]})
        history_show = 200
        history_idx = np.arange(n_train - history_show + 1, n_train + 1)
        axes[0].plot(history_idx, xtrain[-history_show:], color="black", lw=0.7, label="Xtrain (tail)")
        axes[0].plot(idx, truth, color="black", lw=0.9, label="Xtest (truth)")
        axes[0].plot(idx, pred,  color="crimson", lw=0.9, alpha=0.85, label="ensemble prediction")
        axes[0].axvline(n_train + 0.5, color="black", lw=0.5, ls="--", alpha=0.5)
        axes[0].set_ylabel("intensity")
        axes[0].set_title(
            f"Final test  MAE={metrics['mae']:.3f}  MSE={metrics['mse']:.3f}  "
            f"NMSE={metrics['nmse']:.4f}"
        )
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        axes[1].plot(idx, pred - truth, color="steelblue", lw=0.7)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_ylabel("residual")
        axes[1].set_xlabel("sample index")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        out_plot = out_dir / "test_comparison.png"
        fig.savefig(out_plot, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        out_plot = out_dir / "test_comparison.png"
        plot_prediction(
            truth, pred,
            title=f"Final test  MAE={metrics['mae']:.3f}",
            out_path=out_plot,
        )
    print(f"wrote {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
