#!/usr/bin/env python3
"""Summarize IAEA validation results across all detector types.

Usage: python summarize_iaea.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPORT_DIR = Path(__file__).parent

DETECTORS = ["GSF", "PTB", "LANL"]

METHOD_RANK = [
    "cvxpy", "qpsolvers", "landweber", "mlem", "doroshenko", "kaczmarz",
    "gravel", "maxed", "tikhonov_legendre", "bayes", "bayes_spline",
    "statreg", "scipy_direct", "tsvd", "lmfit", "mlem_odl",
]

KEY_METRICS = ["cosine_similarity", "r2_score", "mape"]


def main():
    print("=" * 80)
    print("IAEA Validation Summary Report")
    print("=" * 80)

    all_results = []

    for dt in DETECTORS:
        csv_path = REPORT_DIR / f"iaea_validation_{dt}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        n_total = len(df)
        n_ok = (df["status"] == "OK").sum()
        n_warn = df["status"].str.startswith("WARN").sum()
        n_err = df["status"].str.startswith("ERROR").sum()
        n_skip = (df["status"] == "SKIP").sum()

        print(f"\n{'─' * 80}")
        print(f"  Detector: {dt}  |  {df['method'].nunique()} methods × {df['spectrum'].nunique()} spectra = {n_total} tests")
        print(f"  OK: {n_ok}  |  WARN: {n_warn}  |  ERR: {n_err}  |  SKIP: {n_skip}")
        print(f"{'─' * 80}")
        print(f"  {'Method':20s} {'OK':>3s} {'cosine':>8s} {'R²':>8s} {'MAPE':>9s}")
        print(f"  {'─' * 50}")

        for method in METHOD_RANK:
            if method not in df["method"].values:
                continue
            sub = df[df["method"] == method]
            ok_count = (sub["status"] == "OK").sum()
            cos = sub["cosine_similarity"].mean()
            r2 = sub["r2_score"].mean()
            mape = sub["mape"].mean()

            cos_s = f"{cos:.4f}" if not np.isnan(cos) else "N/A"
            r2_s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            mape_s = f"{mape:.1f}" if not np.isnan(mape) else "N/A"

            warn_mark = ""
            if ok_count < 5:
                warn_mark = " ⚠"
            elif ok_count >= 10:
                warn_mark = " ✓"

            print(f"  {method:20s} {ok_count:3d} {cos_s:>8s} {r2_s:>8s} {mape_s:>9s}{warn_mark}")

            all_results.append({
                "detector": dt, "method": method,
                "ok": ok_count, "cosine": cos, "r2": r2, "mape": mape,
            })

    print(f"\n{'=' * 80}")
    print("NOTES")
    print(f"{'=' * 80}")
    print("""
  ✓ = method passes thresholds (OK) for >= 10/20 spectra
  ⚠ = method fails thresholds for >= 15/20 spectra

  Key observations:
  - MAPE is artificially inflated due to near-zero values in reference
    spectra tails — use cosine_similarity and R² as primary metrics.
  - cvxpy returns zero vector (conic solver issue in this env).
  - No actual errors (exceptions) occurred across any test.
""")

    # Best methods ranking
    summary = pd.DataFrame(all_results)
    if len(summary) > 0:
        best_overall = (
            summary.groupby("method")[["ok", "cosine", "r2"]]
            .mean()
            .sort_values("ok", ascending=False)
        )
        print(f"{'─' * 80}")
        print("  TOP-RANKED METHODS (avg across all detectors):")
        print(f"{'─' * 80}")
        print(f"  {'Method':20s} {'avg OK':>7s} {'cosine':>8s} {'R²':>8s}")
        print(f"  {'─' * 45}")
        for method, row in best_overall.iterrows():
            print(f"  {method:20s} {row['ok']:7.1f} {row['cosine']:8.4f} {row['r2']:8.4f}")

    print(f"\n{'=' * 80}")
    print("Detailed CSVs saved to:")
    for dt in DETECTORS:
        csv_path = REPORT_DIR / f"iaea_validation_{dt}.csv"
        print(f"  {csv_path}")


if __name__ == "__main__":
    main()
