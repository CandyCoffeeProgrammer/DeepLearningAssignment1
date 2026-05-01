"""Run the remaining four family searches sequentially.

Used after GRU + LSTM completed (their best.json files are on disk). This
launches TCN, CNN+LSTM, MLP, CNN1D each with the spec-faithful trial count
in a single long-running Python process — robust against shell timeouts
because Python orchestrates the subprocess chain itself.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_REMAINING = [
    ("tcn",      100),
    ("cnn_lstm", 100),
    ("mlp",       40),
    ("cnn1d",     40),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", default=None,
                   help="optional subset; default runs DEFAULT_REMAINING in order")
    args = p.parse_args()
    plan = (
        [(f, dict(DEFAULT_REMAINING).get(f, 30)) for f in args.families]
        if args.families else DEFAULT_REMAINING
    )
    for family, n_trials in plan:
        cmd = [
            sys.executable, "scripts/run_optuna_search.py",
            "--family", family,
            "--n-trials", str(n_trials),
            "--epochs", "200",
            "--full-caps",
        ]
        print(f"\n=== {family.upper()} ({n_trials} trials, FULL caps) ===", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! {family} exited with code {result.returncode}; stopping chain", flush=True)
            return result.returncode
    print("\n=== ALL REMAINING FAMILIES DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
