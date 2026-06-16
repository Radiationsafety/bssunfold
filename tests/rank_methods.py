#!/usr/bin/env python3
"""Rank unfolding methods by reconstruction quality across all IAEA spectra.

Reads iaea_validation_*.csv files, normalizes all metrics to 0-1 scale,
and outputs a ranked list from best to worst method.

Usage:
    python tests/rank_methods.py
    python tests/rank_methods.py --csv-dir tests/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Metric classification ─────────────────────────────────────────
# "high":  higher value = better reconstruction
# "low":   lower value = better reconstruction
# "one":   closer to 1.0 = better (total_flux_ratio)
# "zero":  closer to 0.0 = better

METRIC_DIRECTION = {
    # high-is-better
    "cosine_similarity": "high",
    "r2_score": "high",
    "pearson_r": "high",
    "spearman_r": "high",
    "spectral_shape_similarity": "high",
    "log_lethargy_correlation": "high",
    # target is 1.0
    "total_flux_ratio": "one",
    # target is 0.0
    "fluence_difference_percent": "zero",
    "dose_difference_percent": "zero",
    "fluence_averaged_energy_diff": "zero",
    "dose_averaged_energy_diff": "zero",
    "energy_group_fluence_diff_thermal": "zero",
    "energy_group_fluence_diff_epithermal": "zero",
    "energy_group_fluence_diff_fast": "zero",
    "peak_location_error": "zero",
    "peak_width_error": "zero",
    "dose_weighted_error": "zero",
    # low-is-better (everything else)
    "kl_divergence": "low",
    "cross_entropy": "low",
    "entropy_difference_percent": "low",
    "wasserstein_dist": "low",
    "energy_dist": "low",
    "kolmogorov_smirnov_stat": "low",
    "mean_squared_error": "low",
    "root_mean_squared_error": "low",
    "mean_absolute_error": "low",
    "mape": "low",
    "max_error": "low",
    "median_absolute_error": "low",
    "mmd_rbf": "low",
    "chi_squared": "low",
    "g_test": "low",
    "freeman_tukey": "low",
    "cressie_read": "low",
    "anderson_darling": "low",
    "standardized_mean_difference": "low",
    "wilcoxon_test": "low",
    "mannwhitneyu_test": "low",
}

DETECTORS = ["GSF", "PTB", "LANL"]


def load_data(csv_dir: Path) -> pd.DataFrame:
    """Load and combine all detector CSVs."""
    frames = []
    for det in DETECTORS:
        path = csv_dir / f"iaea_validation_{det}.csv"
        if not path.exists():
            print(f"  [SKIP] {path} not found", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["detector"] = det
        frames.append(df)
    if not frames:
        sys.exit("No CSV files found")
    return pd.concat(frames, ignore_index=True)


def normalize_metric(values: pd.Series, direction: str) -> pd.Series:
    """Normalize a metric column to 0-1 where 1 = best.

    Handles NaN by filling with the worst possible value before normalizing.
    """
    v = values.copy()
    nan_mask = v.isna()

    vmin, vmax = v.min(), v.max()
    vrange = vmax - vmin

    if vrange == 0 or np.isnan(vrange):
        # All identical or all NaN → neutral score 1.0
        return pd.Series(1.0, index=v.index)

    if direction == "high":
        normed = (v - vmin) / vrange
    elif direction == "low":
        normed = (vmax - v) / vrange
    elif direction == "one":
        # |value - 1.0|: 0 is best, max deviation is worst
        deviation = (v - 1.0).abs()
        max_dev = deviation.max()
        if max_dev == 0:
            normed = pd.Series(1.0, index=v.index)
        else:
            normed = 1.0 - deviation / max_dev
    elif direction == "zero":
        # Absolute value: 0 is best, max value is worst
        abs_v = v.abs()
        max_abs = abs_v.max()
        if max_abs == 0:
            normed = pd.Series(1.0, index=v.index)
        else:
            normed = 1.0 - abs_v / max_abs
    else:
        normed = pd.Series(0.5, index=v.index)

    # NaN rows get worst score (0.0)
    normed[nan_mask] = 0.0
    return normed


def rank_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method scores and rank them."""
    # Identify metric columns
    metric_cols = [c for c in df.columns if c in METRIC_DIRECTION]

    # Only use metrics that actually exist in the data
    available = [c for c in metric_cols if c in df.columns]
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Metrics not in data: {missing}", file=sys.stderr)

    # Normalize each metric
    normed = df[["method", "detector", "spectrum", "status"]].copy()
    for col in available:
        direction = METRIC_DIRECTION[col]
        normed[col] = normalize_metric(df[col], direction)

    # Compute per-method scores
    results = []
    methods = sorted(normed["method"].unique())

    for method in methods:
        mdf = normed[normed["method"] == method]
        ok_count = (mdf["status"] == "OK").sum()
        total = len(mdf)
        ok_rate = ok_count / total * 100 if total > 0 else 0

        # Overall average score
        overall = mdf[available].mean().mean()

        # Per-detector scores
        det_scores = {}
        for det in DETECTORS:
            ddf = mdf[mdf["detector"] == det]
            if len(ddf) > 0:
                det_scores[det] = ddf[available].mean().mean()
            else:
                det_scores[det] = np.nan

        # Per-metric average (across all detectors and spectra)
        metric_avgs = {}
        for col in available:
            metric_avgs[col] = mdf[col].mean()

        results.append({
            "method": method,
            "overall_score": overall,
            "ok_rate": ok_rate,
            **{f"score_{d}": det_scores[d] for d in DETECTORS if d in det_scores},
        })

    ranking = pd.DataFrame(results)
    ranking = ranking.sort_values("overall_score", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1  # 1-based rank
    ranking.index.name = "rank"
    return ranking, normed, available


def print_ranking(ranking: pd.DataFrame, metric_avgs_df: pd.DataFrame = None):
    """Print formatted ranking table."""
    n = len(ranking)

    print("=" * 100)
    print("METHOD RANKING — IAEA 20-spectrum validation")
    print("=" * 100)
    print()

    # Main table
    header = f"{'Rank':>4s}  {'Method':<30s}  {'Score':>6s}  {'OK%':>5s}"
    for det in DETECTORS:
        col = f"score_{det}"
        if col in ranking.columns:
            header += f"  {det:>6s}"
    print(header)
    print("-" * len(header))

    for rank, (_, row) in enumerate(ranking.iterrows(), 1):
        line = f"{rank:>4d}  {row['method']:<30s}  {row['overall_score']:>6.3f}  {row['ok_rate']:>5.1f}%"
        for det in DETECTORS:
            col = f"score_{det}"
            if col in ranking.columns:
                val = row[col]
                line += f"  {val:>6.3f}" if not np.isnan(val) else "    N/A"
        print(line)

    print("-" * len(header))
    print()

    # Top-5 summary
    top5 = ranking.head(5)
    print("TOP-5 METHODS:")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {rank}. {row['method']:<30s}  score={row['overall_score']:.3f}  ok_rate={row['ok_rate']:.1f}%")
    print()

    # Bottom-5 summary
    bot5 = ranking.tail(5)
    print("BOTTOM-5 METHODS:")
    for rank, (_, row) in enumerate(bot5.iterrows(), n - 4):
        print(f"  {rank}. {row['method']:<30s}  score={row['overall_score']:.3f}  ok_rate={row['ok_rate']:.1f}%")
    print()

    # Per-metric breakdown for top method
    if metric_avgs_df is not None and len(metric_avgs_df) > 0:
        best_method = ranking.iloc[0]["method"]
        print(f"METRIC BREAKDOWN for best method ({best_method}):")
        print(f"  {'Metric':<45s}  {'Score':>6s}")
        print(f"  {'-'*55}")
        row = metric_avgs_df[metric_avgs_df["method"] == best_method]
        if len(row) > 0:
            for col in metric_avgs_df.columns:
                if col in METRIC_DIRECTION:
                    val = row[col].values[0]
                    print(f"  {col:<45s}  {val:>6.3f}")
    print()

    print("=" * 100)
    print("Scoring: min-max normalization per metric (0=worst, 1=best)")
    print("  'high' = higher is better,  'low' = lower is better")
    print("  'one' = closer to 1.0 is better,  'zero' = closer to 0.0 is better")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Rank unfolding methods by IAEA validation results")
    parser.add_argument("--csv-dir", type=Path, default=Path(__file__).parent,
                        help="Directory containing iaea_validation_*.csv files")
    args = parser.parse_args()

    print(f"Loading data from: {args.csv_dir}")
    df = load_data(args.csv_dir)
    print(f"  Loaded {len(df)} rows: {df['method'].nunique()} methods × "
          f"{df['spectrum'].nunique()} spectra × {df['detector'].nunique()} detectors")
    print()

    ranking, normed, available = rank_methods(df)

    # Build metric averages per method for breakdown
    metric_avgs = normed.groupby("method")[available].mean().reset_index()

    print_ranking(ranking, metric_avgs)

    # Save to CSV
    out_path = args.csv_dir / "method_ranking.csv"
    ranking.to_csv(out_path)
    print(f"\nRanking saved to: {out_path}")


if __name__ == "__main__":
    main()
