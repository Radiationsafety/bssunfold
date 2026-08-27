#!/usr/bin/env python3
"""Regenerate the legacy BSSUnfold paper figures that were lost.

Produces into the target directory (default benchmark/paper_latex/figs):
  - fig_response_functions.png      GSF response functions (from RF_GSF)
  - fig_cf252_combined.png          combined unfolding pipeline (LANL)
  - fig_tikhonov_convergence.png    Tikhonov-Legendre convergence (LANL)
  - fig_cosine_heatmap.png          cosine-similarity heatmap from benchmark results
  - fig_method_ranking.png          per-detector method ranking from cross-detector table
  - fig_noise_sensitivity.png       dose-rate angle vs noise (from tbl:noise)
  - fig_reference_spectra.png       2x2 composite of representative spectra

fig_dose_rate_scatter.png is regenerated separately by tests/dose_rate_evaluation.py
and copied in by the caller.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bssunfold import Detector, RF_GSF

warnings = None
if warnings is None:
    import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "paper_latex" / "figs"
MC_CSV = ROOT / "tests" / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
MC_RESULTS = ROOT / "benchmark" / "results" / "mc_results_all.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

METHOD_LABELS = {
    "cvxpy": "cvxpy[ecos,gcv]",
    "qpsolvers": "qpsolvers[osqp,gcv]",
    "landweber": "Landweber",
    "kaczmarz": "Kaczmarz",
    "tsvd": "tsvd[k=?]",
    "parametric": "parametric",
    "scipy_direct": "scipy_direct",
    "lmfit": "lmfit",
}


def fig_response_functions() -> Path:
    """GSF Bonner sphere response functions for 10 detector sizes."""
    df = pd.DataFrame(RF_GSF)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in df.columns:
        if col == "E_MeV":
            continue
        ax.plot(df["E_MeV"], df[col], lw=1.5, label=f"{col}")
    ax.set_xscale("log")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Response [cm$^2$]")
    ax.set_title("GSF Bonner sphere response functions")
    ax.legend(fontsize=8, ncol=2, loc="best", frameon=True)
    ax.grid(True, which="major", ls="--", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    out = OUT_DIR / "fig_response_functions.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cf252_combined() -> Path:
    """Combined unfolding pipeline on Cf-252 (LANL)."""
    from bssunfold import RF_LANL

    detector = Detector(pd.DataFrame(RF_LANL))
    mc_df = pd.read_csv(MC_CSV)
    cf_col = [c for c in mc_df.columns if "Cf252" in c][0]
    ref_dict = {"E_MeV": mc_df["E_MeV"].values, "Phi": mc_df[cf_col].values.astype(float)}
    interp_df = detector.discretize_spectra(ref_dict)
    ref_on_grid = interp_df["Phi"].values
    readings = detector.get_effective_readings_for_spectra(ref_dict)
    energy_grid = detector.E_MeV

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy_grid, ref_on_grid, "k--", lw=2.0, alpha=0.8, label="Reference (MC)")

    try:
        result = detector.unfold_landweber(readings, max_iterations=2000, save_result=False)
        if result and "spectrum" in result:
            ax.plot(energy_grid, result["spectrum"], "-", lw=1.5, alpha=0.8,
                    color="#ff7f0e", label="Landweber[2000]")
    except Exception:
        pass

    try:
        result = detector.unfold_cvxpy(readings, regularization_method="gcv", save_result=False)
        if result and "spectrum" in result:
            ax.plot(energy_grid, result["spectrum"], "-", lw=1.5, alpha=0.8,
                    color="#1f77b4", label="cvxpy[ecos,gcv]")
    except Exception:
        pass

    try:
        pipeline = [
            {"method": "cvxpy", "params": {"regularization_method": "gcv"}, "use_as_initial": True},
            {"method": "landweber", "params": {"max_iterations": 500}},
        ]
        result = detector.unfold_combined(readings, pipeline)
        if result and "spectrum" in result:
            ax.plot(energy_grid, result["spectrum"], "-", lw=2.0, alpha=0.9, color="#2ca02c",
                    label="cvxpy[ecos,gcv] $\\rightarrow$ Landweber[500]")
    except Exception:
        pass

    ax.set_xscale("log")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(r"$E \cdot \Phi(E)$ [a.u.]")
    ax.set_title("Combined unfolding pipeline — Cf-252 (LANL)")
    ax.legend(fontsize=9, loc="best", frameon=True)
    ax.grid(True, which="major", ls="--", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    out = OUT_DIR / "fig_cf252_combined.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_tikhonov_convergence() -> Path:
    """Tikhonov-Legendre convergence on Cf-252 (LANL)."""
    from bssunfold import RF_LANL
    from bssunfold.core.dose_calculation import calculate_dose_rates

    detector = Detector(pd.DataFrame(RF_LANL))
    mc_df = pd.read_csv(MC_CSV)
    cf_col = [c for c in mc_df.columns if "Cf252" in c][0]
    ref_dict = {"E_MeV": mc_df["E_MeV"].values, "Phi": mc_df[cf_col].values.astype(float)}
    interp_df = detector.discretize_spectra(ref_dict)
    ref_on_grid = interp_df["Phi"].values
    readings = detector.get_effective_readings_for_spectra(ref_dict)

    n_polys = list(range(10, 55, 5))
    dose_errors = []
    residuals = []
    for n_poly in n_polys:
        try:
            result = detector.unfold_tikhonov_legendre(
                readings, delta=0.05, n_polynomials=n_poly, save_result=False
            )
            if result and "spectrum" in result:
                unfolded = result["spectrum"]
                dose_ref = calculate_dose_rates(ref_on_grid)
                dose_unfolded = calculate_dose_rates(unfolded)
                dose_diff = abs(dose_ref["ISO"] - dose_unfolded["ISO"]) / dose_ref["ISO"] * 100
                dose_errors.append(dose_diff)
                residual = np.linalg.norm(unfolded - ref_on_grid) / np.linalg.norm(ref_on_grid)
                residuals.append(residual)
            else:
                dose_errors.append(np.nan)
                residuals.append(np.nan)
        except Exception:
            dose_errors.append(np.nan)
            residuals.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(n_polys, dose_errors, "b-o", lw=2.0, markersize=6, label="Dose error (%)")
    ax2.plot(n_polys, residuals, "r-s", lw=2.0, markersize=6, label="Residual norm")
    ax1.set_xlabel("Number of Legendre polynomials")
    ax1.set_ylabel("Dose error [%]", color="b")
    ax2.set_ylabel("Relative residual norm", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title("Tikhonov-Legendre convergence — Cf-252 (LANL)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best", frameon=True)
    ax1.grid(True, which="major", ls="--", alpha=0.3)
    out = OUT_DIR / "fig_tikhonov_convergence.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cosine_heatmap() -> Path:
    """Cosine similarity heatmap: methods x spectra from benchmark results."""
    df = pd.read_csv(MC_RESULTS)
    pivot = df.pivot_table(index="method", columns="spectrum", values="cosine_similarity")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Cosine similarity")
    ax.set_title("Cosine similarity heatmap across IAEA reference spectra (LANL)")
    out = OUT_DIR / "fig_cosine_heatmap.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_method_ranking() -> Path:
    """Per-detector method ranking from cross-detector cosine table."""
    data = {
        "cvxpy[ecos,gcv]": [0.906, 0.980, 0.939],
        "scipy_direct": [0.830, 0.972, 0.929],
        "qpsolvers[osqp,gcv]": [0.902, 0.916, 0.910],
        "lmfit": [0.895, 0.958, 0.941],
        "landweber": [0.817, 0.858, 0.908],
        "kaczmarz": [0.810, 0.860, 0.910],
        "tsvd[k=?]": [0.818, 0.875, 0.915],
        "parametric": [0.906, 0.907, 0.922],
    }
    methods = list(data.keys())
    lanl = [data[m][0] for m in methods]
    ptb = [data[m][1] for m in methods]
    gsf = [data[m][2] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    w = 0.25
    ax.bar(x - w, lanl, w, label="LANL", color="#1f77b4")
    ax.bar(x, ptb, w, label="PTB", color="#ff7f0e")
    ax.bar(x + w, gsf, w, label="GSF", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0.75, 1.0)
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.set_title("Method ranking by detector configuration")
    out = OUT_DIR / "fig_method_ranking.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_noise_sensitivity() -> Path:
    """Dose-rate angle vs noise level (from tbl:noise)."""
    noise = np.array([0, 1, 2, 5])
    methods = {
        "cvxpy[ecos,gcv]": [45.01, 45.25, 45.48, 46.12],
        "qpsolvers[osqp,gcv]": [44.98, 45.19, 45.41, 46.05],
        "Landweber": [45.21, 45.43, 45.67, 46.48],
        "Kaczmarz": [45.15, 45.38, 45.62, 46.40],
        "tsvd[k=?]": [46.80, 47.12, 47.48, 48.67],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (name, vals), c in zip(methods.items(), colors):
        ax.plot(noise, vals, "-o", lw=1.8, ms=5, color=c, label=name)
    ax.axhline(45.0, ls="--", color="k", alpha=0.7, lw=1.2, label="Ideal (45°)")
    ax.set_xlabel("Noise level [%]")
    ax.set_ylabel("ISO dose rate angle θ [°]")
    ax.set_title("Noise sensitivity of unfolding methods")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    out = OUT_DIR / "fig_noise_sensitivity.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_reference_spectra() -> Path:
    """2x2 composite of representative spectra from existing benchmark figs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sources = [
        (0, 0, "ISO_ref_Cf252.png", "ISO reference — $^{252}$Cf"),
        (0, 1, "ISO_ref_AmBe.png", "ISO reference — $^{241}$Am–Be"),
        (1, 0, "t4-16-s.txt_1.png", "Accelerator — t4-16-s"),
        (1, 1, "t4-17-s.txt_1.png", "Accelerator — t4-17-s"),
    ]
    for r, c, fname, title in sources:
        src = ROOT / "benchmark" / "figs" / fname
        if not src.exists():
            axes[r][c].text(0.5, 0.5, "figure not found", ha="center", va="center")
        else:
            img = plt.imread(str(src))
            axes[r][c].imshow(img)
        axes[r][c].set_title(title, fontsize=10)
        axes[r][c].axis("off")
    fig.suptitle("Representative reference and unfolded spectra (LANL)", fontsize=12)
    out = OUT_DIR / "fig_reference_spectra.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    print(f"Output dir: {OUT_DIR}")
    for fn in (
        fig_response_functions,
        fig_cf252_combined,
        fig_tikhonov_convergence,
        fig_cosine_heatmap,
        fig_method_ranking,
        fig_noise_sensitivity,
        fig_reference_spectra,
    ):
        try:
            out = fn()
            print(f"  OK  {out.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fn.__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())