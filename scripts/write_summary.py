"""Generate `experiments/final/SUMMARY.md` from search + ensemble + final-prediction outputs.

Produces the required artefact:
  - leaderboard of top 20 single models from the search (across families)
  - chosen ensemble composition + rationale
  - synthetic-holdout final ensemble metrics (mean / worst / per-segment)
  - per-holdout best-seed plot links
  - expected test performance range based on holdout statistics

Inputs (auto-discovered from the project layout):
  - experiments/search/<family>/trials.csv     (per-trial leaderboard)
  - experiments/search/<family>/best.json      (best config per family)
  - experiments/<ensemble_run>/summary.json    (ensemble strategy comparison)
  - experiments/final/metadata.json            (final ensemble composition)

Usage:
    python scripts/write_summary.py \
        --ensemble-run experiments/baselines/milestone7_ensembles_known_good \
        --out experiments/final/SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_trials_csv(family_dir: Path) -> list[dict]:
    path = family_dir / "trials.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--search-root", default="experiments/search")
    p.add_argument("--ensemble-run", default=None,
                   help="run dir from run_ensemble_holdouts.py")
    p.add_argument("--final-dir", default="experiments/final")
    p.add_argument("--out", default="experiments/final/SUMMARY.md")
    args = p.parse_args()

    search_root = (ROOT / args.search_root).resolve()
    final_dir = (ROOT / args.final_dir).resolve()
    out_path = (ROOT / args.out).resolve()

    md: list[str] = []
    md += [
        "# Final summary — Santa Fe laser recursive forecast",
        "",
        "Lowest MAE / MSE on the held-out 200-pt test set wins. This file collects",
        "the search leaderboard, the chosen ensemble, and the expected performance",
        "based on synthetic holdouts. Once `Xtest.mat` arrives, run",
        "`python scripts/make_test_evaluation.py --xtest data/Xtest.mat` to score.",
        "",
    ]

    # ---- leaderboard across families ----
    all_trials: list[dict] = []
    if search_root.exists():
        for family_dir in sorted(p for p in search_root.iterdir() if p.is_dir()):
            for row in load_trials_csv(family_dir):
                row.setdefault("family", family_dir.name)
                all_trials.append(row)

    md.append("## Leaderboard — top 20 single models across all families")
    md.append("")
    if all_trials:
        try:
            ranked = sorted(all_trials, key=lambda r: float(r["mean_mae"]))[:20]
        except (KeyError, ValueError):
            ranked = []
        if ranked:
            md.append("| rank | family | trial | mean MAE | worst MAE | A | B | C | window | hidden | k_max | p_max | noise |")
            md.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for i, r in enumerate(ranked, start=1):
                def g(k: str, fmt: str = "{}") -> str:
                    v = r.get(k, "")
                    if v == "":
                        return "—"
                    try:
                        return fmt.format(float(v))
                    except (ValueError, TypeError):
                        return str(v)
                md.append(
                    f"| {i} | {r.get('family','')} | {r.get('trial','')} | "
                    f"{g('mean_mae','{:.3f}')} | {g('worst_mae','{:.3f}')} | "
                    f"{g('mae_A','{:.2f}')} | {g('mae_B','{:.2f}')} | {g('mae_C','{:.2f}')} | "
                    f"{r.get('window','')} | {r.get('hidden_dim','')} | "
                    f"{r.get('k_max','')} | {r.get('p_max','')} | {r.get('input_noise_sigma','')} |"
                )
        else:
            md.append("_no completed search trials found_")
    else:
        md.append("_search has not been run yet — see `bash scripts/run_search.sh`_")
    md.append("")

    # ---- best per family ----
    md.append("## Best per family")
    md.append("")
    bests: list[dict] = []
    if search_root.exists():
        for family_dir in sorted(p for p in search_root.iterdir() if p.is_dir()):
            best_path = family_dir / "best.json"
            if best_path.exists():
                with open(best_path, "r", encoding="utf-8") as f:
                    bests.append(json.load(f))
    if bests:
        bests.sort(key=lambda b: b.get("mean_mae", float("inf")))
        md.append("| family | best mean MAE | worst MAE | per-holdout |")
        md.append("| --- | ---: | ---: | --- |")
        for b in bests:
            ph = b.get("per_holdout_mae") or {}
            md.append(
                f"| {b.get('family','')} | {b.get('mean_mae', float('nan')):.3f} | "
                f"{b.get('worst_mae', float('nan')):.3f} | "
                + ", ".join(f"{n}={v:.2f}" for n, v in sorted(ph.items())) + " |"
            )
    else:
        md.append("_no `best.json` files found yet_")
    md.append("")

    # ---- ensemble comparison ----
    md.append("## Ensemble strategy comparison")
    md.append("")
    if args.ensemble_run is not None:
        ens_dir = (ROOT / args.ensemble_run).resolve()
        ens_summary = ens_dir / "summary.json"
        if ens_summary.exists():
            with open(ens_summary, "r", encoding="utf-8") as f:
                ens = json.load(f)
            overall = ens.get("overall", {})
            md.append(f"Source: `{ens_dir.relative_to(ROOT)}`")
            md.append("")
            md.append("| strategy | mean test MAE across A/B/C |")
            md.append("| --- | ---: |")
            for s, v in sorted(overall.items(), key=lambda kv: kv[1]):
                md.append(f"| {s} | {v:.3f} |")
            md.append("")
            md.append(f"**Winning strategy: `{ens.get('winning','?')}`**")
            md.append("")
        else:
            md.append(f"_ensemble summary missing: {ens_summary}_")
    else:
        md.append("_pass --ensemble-run pointing at a run_ensemble_holdouts.py output_")
    md.append("")

    # ---- final ensemble composition ----
    md.append("## Final ensemble (retrained on full Xtrain)")
    md.append("")
    final_meta_path = final_dir / "metadata.json"
    if final_meta_path.exists():
        with open(final_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        md.append(f"Generated at: `{meta.get('generated_at','?')}`")
        md.append(f"Strategy: **{meta.get('strategy','?')}**, members: **{meta.get('n_members','?')}**, "
                  f"horizon: {meta.get('horizon','?')}")
        s = meta.get("summary", {})
        md.append(
            f"Forecast range: min={s.get('min',float('nan')):.2f}  max={s.get('max',float('nan')):.2f}  "
            f"mean={s.get('mean',float('nan')):.2f}  std={s.get('std',float('nan')):.2f}"
        )
        md.append("")
        md.append("| tag | family | window | epochs | scaler |")
        md.append("| --- | --- | ---: | ---: | --- |")
        for m in meta.get("members", []):
            md.append(
                f"| {m.get('tag','')} | {m.get('family','')} | {m.get('window','')} | "
                f"{m.get('epochs','')} | {m.get('scaler','')} |"
            )
    else:
        md.append("_no final predictions yet — run `scripts/run_final_predictions.py`_")
    md.append("")

    # ---- expected performance ----
    md.append("## Expected test-set performance")
    md.append("")
    if args.ensemble_run is not None:
        ens_summary = (ROOT / args.ensemble_run / "summary.json").resolve()
        if ens_summary.exists():
            with open(ens_summary) as f:
                ens = json.load(f)
            rows = ens.get("rows", [])
            mean_strat = ens.get("winning", "median")
            holdout_mae = {r["holdout"]: r["mae"] for r in rows if r.get("strategy") == mean_strat}
            if holdout_mae:
                lo = min(holdout_mae.values())
                hi = max(holdout_mae.values())
                avg = sum(holdout_mae.values()) / len(holdout_mae)
                md.append(
                    f"Based on the synthetic holdouts under the winning ensemble (`{mean_strat}`), "
                    f"the expected test MAE band is **[{lo:.2f}, {hi:.2f}]** "
                    f"(mean across A/B/C: {avg:.2f})."
                )
                md.append("")
                md.append("Per holdout:")
                md.append("")
                md.append("| holdout | ensemble MAE |")
                md.append("| --- | ---: |")
                for n, v in holdout_mae.items():
                    md.append(f"| {n} | {v:.2f} |")
            else:
                md.append("_no per-holdout ensemble data_")
        else:
            md.append("_no ensemble summary to derive expectations from_")
    else:
        md.append("_run the ensemble step first_")
    md.append("")

    md.append("## Plots")
    md.append("")
    if final_dir.exists():
        plot = final_dir / "ensemble_preview.png"
        if plot.exists():
            md.append(f"- Final 200-step forecast: `{plot.relative_to(ROOT)}`")
        test_plot = final_dir / "test_comparison.png"
        if test_plot.exists():
            md.append(f"- Test comparison (after Xtest released): `{test_plot.relative_to(ROOT)}`")
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
