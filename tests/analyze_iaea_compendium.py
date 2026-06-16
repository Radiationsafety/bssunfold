#!/usr/bin/env python3
"""Analyze IAEA Compendium spectra: reconstruct with bssunfold, compare, find poor reconstructions.

Reads papers/IAEA_Compendium_dataset.csv (251 spectra × 60 energy bins),
computes effective readings for GSF/PTB/LANL detectors, unfolds with multiple
methods, compares with reference, calculates dose rates, and outputs a markdown report.

Usage:
    python tests/analyze_iaea_compendium.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bssunfold import Detector, RF_PTB, RF_LANL
from bssunfold.utils.comparison import compare_spectra
from bssunfold.core.dose_calculation import calculate_dose_rates

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────

CSV_ENERGIES = np.array([
    1e-9, 2.15e-9, 4.64e-9, 1e-8, 2.15e-8, 4.64e-8,
    1e-7, 2.15e-7, 4.64e-7, 1e-6, 2.15e-6, 4.64e-6,
    1e-5, 2.15e-5, 4.64e-5, 0.0001, 0.000215, 0.000464,
    0.001, 0.00215, 0.00464, 0.01, 0.0125, 0.0158,
    0.0199, 0.0251, 0.0316, 0.0398, 0.0501, 0.063,
    0.0794, 0.1, 0.125, 0.158, 0.199, 0.251,
    0.316, 0.398, 0.501, 0.63, 0.794, 1.0,
    1.25, 1.58, 1.99, 2.51, 3.16, 3.98,
    5.01, 6.3, 7.94, 10.0, 15.8, 25.1,
    39.8, 63.0, 100.0, 158.0, 251.0, 398.0,
])

DETECTOR_CONFIGS = {
    "GSF": lambda: Detector(),
    "PTB": lambda: Detector(pd.DataFrame(RF_PTB)),
    "LANL": lambda: Detector(pd.DataFrame(RF_LANL)),
}

METHODS = {
    "landweber": lambda d, r: d.unfold_landweber(r, max_iterations=500, save_result=False),
    "mlem": lambda d, r: d.unfold_mlem(r, max_iterations=500, save_result=False),
    "kaczmarz": lambda d, r: d.unfold_kaczmarz(r, max_iterations=500, save_result=False),
    "cvxpy": lambda d, r: d.unfold_cvxpy(r, regularization=1e-3, save_result=False),
    "qpsolvers": lambda d, r: d.unfold_qpsolvers(r, regularization=1e-3, save_result=False),
    "statreg": lambda d, r: d.unfold_statreg(r, save_result=False),
    "tsvd": lambda d, r: d.unfold_tsvd(r, method="discrepancy", save_result=False),
    "bayes": lambda d, r: d.unfold_bayes(r, max_iterations=200, save_result=False),
}

DOSE_GEOMETRIES = ["ISO", "ROT", "AP", "PA", "LLAT", "RLAT"]

OUTPUT_MD = Path(__file__).parent.parent / "papers" / "IAEA_compendium_analysis.md"


# ── Helpers ───────────────────────────────────────────────────────

def build_ref_dict(csv_spectrum: np.ndarray) -> dict:
    return {"E_MeV": CSV_ENERGIES, "Phi": csv_spectrum}


def unfold_one(detector, readings: dict, method_name: str, method_fn) -> np.ndarray | None:
    try:
        result = method_fn(detector, readings)
        if result and "spectrum" in result:
            return result["spectrum"]
    except Exception:
        pass
    return None


def compute_dose(spectrum: np.ndarray) -> dict:
    try:
        return calculate_dose_rates(spectrum)
    except Exception:
        return {g: 0.0 for g in DOSE_GEOMETRIES}


def cosine(s1: np.ndarray, s2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(s1), np.linalg.norm(s2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(s1, s2) / (n1 * n2))


def flux_ratio(ref: np.ndarray, test: np.ndarray) -> float:
    s_ref = np.sum(ref)
    s_test = np.sum(test)
    if s_ref == 0:
        return 0.0
    return float(s_test / s_ref)


# ── Main analysis ─────────────────────────────────────────────────

def run_analysis():
    csv_path = Path(__file__).parent.parent / "papers" / "IAEA_Compendium_dataset.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    energy_cols = [c for c in df.columns if c.startswith("Energy_bin")]
    places = df["Place"].values
    n_spectra = len(df)
    n_methods = len(METHODS)
    n_detectors = len(DETECTOR_CONFIGS)

    print(f"Loaded {n_spectra} spectra × {len(energy_cols)} bins")
    print(f"Methods: {n_methods}  |  Detectors: {n_detectors}")
    print(f"Total unfold calls: {n_spectra * n_detectors * n_methods}")
    print()

    # Pre-create detectors
    detectors = {name: fn() for name, fn in DETECTOR_CONFIGS.items()}

    # Result storage
    rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        place = row["Place"]
        csv_spectrum = np.array([row[c] for c in energy_cols], dtype=float)
        ref_dict = build_ref_dict(csv_spectrum)

        for det_name, detector in detectors.items():
            # Discretize reference to detector grid
            interp_df = detector.discretize_spectra(ref_dict)
            ref_on_grid = interp_df["Phi"].values

            # Get effective readings
            readings = detector.get_effective_readings_for_spectra(ref_dict)

            # Dose for reference
            dose_ref = compute_dose(ref_on_grid)

            for method_name, method_fn in METHODS.items():
                unfolded = unfold_one(detector, readings, method_name, method_fn)
                if unfolded is None:
                    rows.append({
                        "place": place, "detector": det_name, "method": method_name,
                        "status": "ERROR",
                        "cosine": 0.0, "flux_ratio": 0.0,
                        "dose_ratio_ISO": 0.0, "dose_ratio_ROT": 0.0,
                        "dose_ratio_AP": 0.0,
                        "dose_ref_ISO": dose_ref.get("ISO", 0.0),
                        "dose_unfolded_ISO": 0.0,
                    })
                    continue

                cos = cosine(ref_on_grid, unfolded)
                fr = flux_ratio(ref_on_grid, unfolded)
                dose_unfolded = compute_dose(unfolded)

                dose_ratio_iso = (dose_unfolded["ISO"] / dose_ref["ISO"]
                                  if dose_ref["ISO"] > 0 else 0.0)
                dose_ratio_rot = (dose_unfolded["ROT"] / dose_ref["ROT"]
                                  if dose_ref["ROT"] > 0 else 0.0)
                dose_ratio_ap = (dose_unfolded["AP"] / dose_ref["AP"]
                                 if dose_ref["AP"] > 0 else 0.0)

                rows.append({
                    "place": place, "detector": det_name, "method": method_name,
                    "status": "OK",
                    "cosine": cos, "flux_ratio": fr,
                    "dose_ratio_ISO": dose_ratio_iso,
                    "dose_ratio_ROT": dose_ratio_rot,
                    "dose_ratio_AP": dose_ratio_ap,
                    "dose_ref_ISO": dose_ref.get("ISO", 0.0),
                    "dose_unfolded_ISO": dose_unfolded.get("ISO", 0.0),
                })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_spectra} spectra...")

    print(f"  Done: {len(rows)} results")
    return pd.DataFrame(rows)


# ── Report generation ─────────────────────────────────────────────

def generate_report(results: pd.DataFrame):
    ok = results[results["status"] == "OK"].copy()
    n_total = len(results)
    n_ok = len(ok)
    n_err = n_total - n_ok

    print(f"\nResults: {n_ok} OK / {n_err} ERROR out of {n_total}")

    # ── Per-spectrum aggregate (avg across methods and detectors) ──
    spec_agg = ok.groupby("place").agg(
        avg_cosine=("cosine", "mean"),
        min_cosine=("cosine", "min"),
        avg_flux_ratio=("flux_ratio", "mean"),
        avg_dose_ratio_ISO=("dose_ratio_ISO", "mean"),
        avg_dose_ratio_AP=("dose_ratio_AP", "mean"),
    ).reset_index()
    spec_agg = spec_agg.sort_values("avg_cosine").reset_index(drop=True)

    # ── Per-method aggregate ──
    method_agg = ok.groupby("method").agg(
        avg_cosine=("cosine", "mean"),
        median_cosine=("cosine", "median"),
        min_cosine=("cosine", "min"),
        avg_flux_ratio=("flux_ratio", "mean"),
        avg_dose_ratio_ISO=("dose_ratio_ISO", "mean"),
        avg_dose_ratio_AP=("dose_ratio_AP", "mean"),
        n_ok=("cosine", "count"),
    ).reset_index()
    method_agg = method_agg.sort_values("avg_cosine", ascending=False).reset_index(drop=True)

    # ── Per-detector aggregate ──
    det_agg = ok.groupby("detector").agg(
        avg_cosine=("cosine", "mean"),
        median_cosine=("cosine", "median"),
        avg_flux_ratio=("flux_ratio", "mean"),
        avg_dose_ratio_ISO=("dose_ratio_ISO", "mean"),
    ).reset_index()
    det_agg = det_agg.sort_values("avg_cosine", ascending=False).reset_index(drop=True)

    # ── Per-method-per-detector ──
    md_agg = ok.groupby(["method", "detector"]).agg(
        avg_cosine=("cosine", "mean"),
        avg_flux_ratio=("flux_ratio", "mean"),
        avg_dose_ratio_ISO=("dose_ratio_ISO", "mean"),
    ).reset_index()

    # ── Poor reconstructions ──
    poor = spec_agg[spec_agg["avg_cosine"] < 0.95].head(30)
    very_poor = spec_agg[spec_agg["avg_cosine"] < 0.90]

    # ── Build markdown ──
    lines = []
    a = lines.append

    a("# IAEA Compendium Dataset — bssunfold Reconstruction Analysis\n")
    a(f"**Date**: generated by `tests/analyze_iaea_compendium.py`  ")
    a(f"**Spectra**: {len(spec_agg)}  ")
    a(f"**Detectors**: {', '.join(DETECTOR_CONFIGS.keys())}  ")
    a(f"**Methods**: {', '.join(METHODS.keys())}  ")
    a(f"**Total evaluations**: {n_total} (OK: {n_ok}, ERROR: {n_err})\n")

    # ── Method ranking ──
    a("## Method Ranking (avg across all spectra and detectors)\n")
    a("| Rank | Method | Avg Cosine | Median Cosine | Min Cosine | Avg Flux Ratio | Avg Dose ISO Ratio |")
    a("|------|--------|-----------|---------------|-----------|----------------|-------------------|")
    for rank, (_, r) in enumerate(method_agg.iterrows(), 1):
        a(f"| {rank} | {r['method']} | {r['avg_cosine']:.4f} | {r['median_cosine']:.4f} "
          f"| {r['min_cosine']:.4f} | {r['avg_flux_ratio']:.4f} | {r['avg_dose_ratio_ISO']:.4f} |")
    a("")

    # ── Detector comparison ──
    a("## Detector Comparison (avg across all methods and spectra)\n")
    a("| Detector | Avg Cosine | Median Cosine | Avg Flux Ratio | Avg Dose ISO Ratio |")
    a("|----------|-----------|---------------|----------------|-------------------|")
    for _, r in det_agg.iterrows():
        a(f"| {r['detector']} | {r['avg_cosine']:.4f} | {r['median_cosine']:.4f} "
          f"| {r['avg_flux_ratio']:.4f} | {r['avg_dose_ratio_ISO']:.4f} |")
    a("")

    # ── Method × Detector matrix ──
    a("## Method × Detector Matrix (Avg Cosine Similarity)\n")
    methods_list = method_agg["method"].tolist()
    detectors_list = det_agg["detector"].tolist()
    header = "| Method | " + " | ".join(detectors_list) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(detectors_list)) + "|"
    a(header)
    a(sep)
    for m in methods_list:
        row_data = md_agg[md_agg["method"] == m]
        vals = []
        for d in detectors_list:
            match = row_data[row_data["detector"] == d]
            if len(match) > 0:
                vals.append(f"{match.iloc[0]['avg_cosine']:.4f}")
            else:
                vals.append("N/A")
        a(f"| {m} | " + " | ".join(vals) + " |")
    a("")

    # ── Worst reconstructed spectra ──
    a("## Spectra with Poorest Reconstruction (avg cosine < 0.95)\n")
    if len(poor) > 0:
        a("| # | Place | Avg Cosine | Min Cosine | Avg Flux Ratio | Avg Dose ISO Ratio |")
        a("|---|-------|-----------|-----------|----------------|-------------------|")
        for rank, (_, r) in enumerate(poor.iterrows(), 1):
            a(f"| {rank} | {r['place']} | {r['avg_cosine']:.4f} | {r['min_cosine']:.4f} "
              f"| {r['avg_flux_ratio']:.4f} | {r['avg_dose_ratio_ISO']:.4f} |")
    else:
        a("No spectra with avg cosine < 0.95 found.\n")
    a("")

    a(f"**Very poor (cosine < 0.90)**: {len(very_poor)} spectra\n")

    # ── Best reconstructed spectra ──
    a("## Spectra with Best Reconstruction (top 20)\n")
    best = spec_agg.tail(20).iloc[::-1]
    a("| # | Place | Avg Cosine | Min Cosine | Avg Flux Ratio | Avg Dose ISO Ratio |")
    a("|---|-------|-----------|-----------|----------------|-------------------|")
    for rank, (_, r) in enumerate(best.iterrows(), 1):
        a(f"| {rank} | {r['place']} | {r['avg_cosine']:.4f} | {r['min_cosine']:.4f} "
          f"| {r['avg_flux_ratio']:.4f} | {r['avg_dose_ratio_ISO']:.4f} |")
    a("")

    # ── Dose rate analysis ──
    a("## Dose Rate Analysis\n")
    a("Ratio of unfolded dose to reference dose (1.0 = perfect). Values > 1.0 mean overestimation.\n")
    dose_stats = ok.groupby("method").agg(
        ISO_mean=("dose_ratio_ISO", "mean"),
        ISO_std=("dose_ratio_ISO", "std"),
        ROT_mean=("dose_ratio_ROT", "mean"),
        AP_mean=("dose_ratio_AP", "mean"),
    ).reset_index()
    dose_stats = dose_stats.sort_values("ISO_mean", key=abs, ascending=False)

    a("| Method | ISO Ratio (mean ± std) | ROT Ratio | AP Ratio |")
    a("|--------|----------------------|-----------|----------|")
    for _, r in dose_stats.iterrows():
        a(f"| {r['method']} | {r['ISO_mean']:.4f} ± {r['ISO_std']:.4f} "
          f"| {r['ROT_mean']:.4f} | {r['AP_mean']:.4f} |")
    a("")

    # ── Summary statistics ──
    a("## Summary Statistics\n")
    a(f"- **Total spectra analyzed**: {len(spec_agg)}")
    a(f"- **Total unfold calls**: {n_total} ({n_ok} OK, {n_err} ERROR)")
    a(f"- **Overall avg cosine**: {ok['cosine'].mean():.4f}")
    a(f"- **Overall median cosine**: {ok['cosine'].median():.4f}")
    a(f"- **Spectra with avg cosine ≥ 0.99**: {len(spec_agg[spec_agg['avg_cosine'] >= 0.99])}")
    a(f"- **Spectra with avg cosine ≥ 0.95**: {len(spec_agg[spec_agg['avg_cosine'] >= 0.95])}")
    a(f"- **Spectra with avg cosine < 0.95**: {len(spec_agg[spec_agg['avg_cosine'] < 0.95])}")
    a(f"- **Spectra with avg cosine < 0.90**: {len(spec_agg[spec_agg['avg_cosine'] < 0.90])}")
    a(f"- **Best method**: {method_agg.iloc[0]['method']} (cosine={method_agg.iloc[0]['avg_cosine']:.4f})")
    a(f"- **Worst method**: {method_agg.iloc[-1]['method']} (cosine={method_agg.iloc[-1]['avg_cosine']:.4f})")
    a("")

    a("---\n")
    a("*Metrics: cosine similarity (shape), flux ratio (total fluence), dose ratio (H*(10) ISO/ROT/AP). "
      "All computed on bssunfold detector energy grid (60 bins, 1e-9 to 631 MeV).*\n")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_analysis()

    # Save raw results
    results_path = Path(__file__).parent.parent / "papers" / "IAEA_compendium_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nRaw results saved to: {results_path}")

    # Generate report
    report = generate_report(results)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"Report saved to: {OUTPUT_MD}")
