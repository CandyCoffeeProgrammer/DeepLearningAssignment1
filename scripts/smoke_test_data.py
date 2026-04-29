"""Quick CLI sanity test for the data module.

Loads Xtrain.mat, builds the three holdouts, and prints a one-liner per holdout
plus a MinMax round-trip check. Useful to verify the pipeline without firing up
Jupyter.

Run from project root:
    python scripts/smoke_test_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_xtrain, make_holdouts                 # noqa: E402
from src.utils import load_yaml                                  # noqa: E402


def main() -> int:
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    x = load_xtrain(ROOT / cfg["data"]["path"], key=cfg["data"]["key"])
    print(f"loaded series: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.0f},{x.max():.0f}]")

    holdouts = make_holdouts(
        x,
        cfg["data"]["holdouts"],
        window=cfg["window"],
        inner_val_len=cfg["data"]["inner_val_len"],
        scaler_name=cfg["data"]["scaler"],
    )

    print("\nholdouts:")
    print(f"  {'name':<6} {'train':>6} {'inner_val':>10} {'test':>5} {'seed_win':>9} {'test_range':>14}")
    for name, h in holdouts.items():
        print(
            f"  {name:<6} {len(h.train):>6} {len(h.inner_val):>10} {len(h.test):>5} "
            f"{len(h.seed_window):>9} {f'[{h.test_start},{h.test_end}]':>14}"
        )

    # Scaler round-trip on each holdout test
    print("\nscaler round-trip max-abs-err per holdout:")
    for name, h in holdouts.items():
        err = float(np.max(np.abs(h.scaler.inverse_transform(h.scaler.transform(h.test)) - h.test)))
        print(f"  {name}: {err:.3e}")

    # Windowing smoke test
    from src.data import LaserWindowDataset
    ds = LaserWindowDataset(holdouts["A"].train_scaled, window=cfg["window"], horizon=5)
    x0, y0 = ds[0]
    print(f"\nLaserWindowDataset[holdout A, window={cfg['window']}, horizon=5]: "
          f"len={len(ds)}, x.shape={tuple(x0.shape)}, y.shape={tuple(y0.shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
