"""Self-contained smoke test for scripts/make_test_evaluation.py.

Pretends the last 200 points of Xtrain are the released Xtest, and pretends
a constant-mean baseline is the saved predictions. Runs make_test_evaluation
end-to-end against those fakes and asserts that the metrics file is written
with the expected MAE.

Run from project root:
    python scripts/test_make_test_evaluation.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.data import load_xtrain   # local import after path mangling
    from src.evaluate import compute_metrics

    xtrain = load_xtrain(ROOT / "data" / "Xtrain.mat")
    fake_truth = xtrain[-200:].astype(np.float64)
    fake_pred = np.full_like(fake_truth, fill_value=fake_truth.mean())
    expected = compute_metrics(fake_pred, fake_truth)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        xtest_path = tmp_dir / "Xtest.mat"
        sio.savemat(str(xtest_path), {"Xtest": fake_truth.reshape(-1, 1)})

        pred_path = tmp_dir / "predictions.npy"
        np.save(pred_path, fake_pred)

        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "make_test_evaluation.py"),
            "--xtest", str(xtest_path),
            "--predictions", str(pred_path),
            "--out-dir", str(out_dir),
        ]
        print("running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"make_test_evaluation.py exited {proc.returncode}")

        metrics_path = out_dir / "test_metrics.json"
        plot_path = out_dir / "test_comparison.png"
        assert metrics_path.exists(), f"missing {metrics_path}"
        assert plot_path.exists(), f"missing {plot_path}"

        with open(metrics_path) as f:
            written = json.load(f)["metrics"]

        for k in ("mae", "mse", "rmse", "nmse"):
            if not abs(written[k] - expected[k]) < 1e-6:
                raise SystemExit(
                    f"{k} mismatch: written={written[k]} expected={expected[k]}"
                )

        print(f"OK — MAE={written['mae']:.4f}, MSE={written['mse']:.4f}, "
              f"RMSE={written['rmse']:.4f}, NMSE={written['nmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
