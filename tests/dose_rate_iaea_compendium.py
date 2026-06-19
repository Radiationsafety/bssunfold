#!/usr/bin/env python3
"""Evaluate unfolding methods by dose rate accuracy on IAEA Compendium dataset.

Reads tests/IAEA_Compendium_dataset.csv (251 spectra × 60 energy bins),
computes effective readings for GSF/PTB/LANL detectors, unfolds with 21
methods, compares dose rates with reference, generates scatter plots + report.

Usage:
    python tests/dose_rate_iaea_compendium.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bssunfold import Detector, RF_PTB, RF_LANL, RF_JINR, RF_FERMILAB
from bssunfold.core.dose_calculation import calculate_dose_rates

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────

IAEA_CSV = Path(__file__).parent / "IAEA_Compendium_dataset.csv"

OUTPUT_MD = Path(__file__).parent / "iaea_compendium_dose_rate_evaluation.md"
OUTPUT_CSV = Path(__file__).parent / "iaea_compendium_dose_rate_results.csv"
OUTPUT_PNG = Path(__file__).parent / "iaea_compendium_dose_rate_scatter.png"
ISO_PLOT_DIR = Path(__file__).parent / "iaea_compendium_iso_plots"

CSV_ENERGIES = np.array(
    [
        1e-9, 2.15e-9, 4.64e-9, 1e-8, 2.15e-8, 4.64e-8,
        1e-7, 2.15e-7, 4.64e-7, 1e-6, 2.15e-6, 4.64e-6,
        1e-5, 2.15e-5, 4.64e-5, 0.0001, 0.000215, 0.000464,
        0.001, 0.00215, 0.00464, 0.01, 0.0125, 0.0158,
        0.0199, 0.0251, 0.0316, 0.0398, 0.0501, 0.063,
        0.0794, 0.1, 0.125, 0.158, 0.199, 0.251, 0.316,
        0.398, 0.501, 0.63, 0.794, 1.0, 1.25, 1.58,
        1.99, 2.51, 3.16, 3.98, 5.01, 6.3, 7.94,
        10.0, 15.8, 25.1, 39.8, 63.0, 100.0, 158.0,
        251.0, 398.0,
    ]
)

DETECTOR_CONFIGS = {
    "GSF": lambda: Detector(),
    "PTB": lambda: Detector(pd.DataFrame(RF_PTB)),
    "LANL": lambda: Detector(pd.DataFrame(RF_LANL)),
    "JINR": lambda: Detector(pd.DataFrame(RF_JINR)),
    "FERMILAB": lambda: Detector(pd.DataFrame(RF_FERMILAB)),
}

DOSE_GEOMETRIES = ["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]

n_iter = 2000

METHODS = {
    "cvxpy": lambda d, r: d.unfold_cvxpy(
        r, regularization_method="gcv", save_result=False
    ),
    "qpsolvers": lambda d, r: d.unfold_qpsolvers(
        r, regularization_method="gcv", save_result=False
    ),
    "landweber": lambda d, r: d.unfold_landweber(
        r, max_iterations=n_iter, save_result=False
    ),
    "mlem": lambda d, r: d.unfold_mlem(
        r, max_iterations=n_iter, save_result=False
    ),
    "doroshenko": lambda d, r: d.unfold_doroshenko(
        r, max_iterations=n_iter, save_result=False
    ),
    "kaczmarz": lambda d, r: d.unfold_kaczmarz(
        r, max_iterations=n_iter, save_result=False
    ),
    "gravel": lambda d, r: d.unfold_gravel(
        r, max_iterations=n_iter, save_result=False
    ),
    "maxed": lambda d, r: d.unfold_maxed(
        r, max_iterations=n_iter, save_result=False
    ),
    "tikhonov_legendre": lambda d, r: d.unfold_tikhonov_legendre(
        r, delta=0.05, n_polynomials=45, save_result=False
    ),
    "bayes": lambda d, r: d.unfold_bayes(
        r, max_iterations=n_iter, save_result=False
    ),
    "bayes_spline": lambda d, r: d.unfold_bayes_spline_regularization(
        r, max_iterations=n_iter, save_result=False
    ),
    "statreg": lambda d, r: d.unfold_statreg(r, save_result=False),
    "scipy_direct": lambda d, r: d.unfold_scipy_direct_method(
        r, method="cg", max_iterations=n_iter, save_result=False
    ),
    "tsvd": lambda d, r: d.unfold_tsvd(
        r, method="discrepancy", save_result=False
    ),
    "lmfit": lambda d, r: d.unfold_lmfit(
        r, method="lbfgsb", model_name="elastic",
        regularization=1e-4, save_result=False,
    ),
    "mlem_odl": lambda d, r: d.unfold_mlem_odl(
        r, max_iterations=n_iter, save_result=False
    ),
    "fruit_like": lambda d, r: d.unfold_fruit_like(r, save_result=False),
    "hybrid_parametric_landweber": lambda d, r: d.unfold_hybrid_parametric(
        r, refinement_method="landweber", save_result=False
    ),
    "hybrid_parametric_mlem": lambda d, r: d.unfold_hybrid_parametric(
        r, refinement_method="mlem", save_result=False
    ),
    "bayesian_parametric": lambda d, r: d.unfold_bayesian_parametric(
        r, n_samples=100, burn_in=20, save_result=False
    ),
    "parametric": lambda d, r: d.unfold_parametric(r, save_result=False),
}


# ── Helpers ───────────────────────────────────────────────────────


def compute_dose(spectrum):
    try:
        return calculate_dose_rates(spectrum)
    except Exception:
        return {g: 0.0 for g in DOSE_GEOMETRIES}


def fit_angle(xs, ys):
    """Fit y=k*x through origin, return (k, angle_deg, classification)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return np.nan, np.nan, "N/A"
    k = float(np.sum(xs * ys) / np.sum(xs ** 2))
    theta = float(np.degrees(np.arctan(k)))
    delta = theta - 45.0
    abs_delta = abs(delta)
    if abs_delta < 1.0:
        cls = "Excellent"
    elif abs_delta < 5.0:
        cls = "Good"
    elif abs_delta < 10.0:
        cls = "Fair"
    else:
        cls = "Poor"
    direction = "over" if delta > 0 else "under"
    return k, theta, f"{cls} ({direction}est.)"


def classify_simple(theta):
    if np.isnan(theta):
        return "N/A"
    delta = theta - 45.0
    abs_delta = abs(delta)
    if abs_delta < 1.0:
        return "Excellent"
    elif abs_delta < 5.0:
        return "Good"
    elif abs_delta < 10.0:
        return "Fair"
    else:
        return "Poor"


def direction(theta):
    if np.isnan(theta):
        return ""
    return "overest." if theta > 45.0 else "underest."


# ── Data loading ──────────────────────────────────────────────────


def load_data():
    df = pd.read_csv(IAEA_CSV)
    energy_cols = [c for c in df.columns if c.startswith("Energy_bin")]
    places = df["Place"].tolist()
    print(f"IAEA Compendium: {len(df)} spectra, {len(energy_cols)} energy bins")
    return df, energy_cols, places


# ── Main evaluation ───────────────────────────────────────────────


def run_evaluation():
    iaea_df, energy_cols, places = load_data()

    detectors = {name: fn() for name, fn in DETECTOR_CONFIGS.items()}
    n_methods = len(METHODS)
    n_detectors = len(DETECTOR_CONFIGS)
    n_spectra = len(places)
    total = n_methods * n_detectors * n_spectra
    print(f"\nMethods: {n_methods}  |  Detectors: {n_detectors}  |  Spectra: {n_spectra}")
    print(f"Total unfold calls: {total}\n")

    rows = []
    done = 0

    for det_name, detector in detectors.items():
        for idx, (_, iaea_row) in enumerate(iaea_df.iterrows()):
            place = iaea_row["Place"]
            csv_spectrum = np.array([iaea_row[c] for c in energy_cols], dtype=float)
            ref_dict = {"E_MeV": CSV_ENERGIES, "Phi": csv_spectrum}

            interp_df = detector.discretize_spectra(ref_dict)
            ref_on_grid = interp_df["Phi"].values

            readings = detector.get_effective_readings_for_spectra(ref_dict)

            dose_ref = compute_dose(ref_on_grid)

            for method_name, method_fn in METHODS.items():
                try:
                    result = method_fn(detector, readings)
                    if result and "spectrum" in result:
                        unfolded = result["spectrum"]
                        dose_unfolded = compute_dose(unfolded)
                        status = "OK"
                    else:
                        dose_unfolded = {g: 0.0 for g in DOSE_GEOMETRIES}
                        status = "ERROR: no spectrum"
                except Exception as e:
                    dose_unfolded = {g: 0.0 for g in DOSE_GEOMETRIES}
                    status = f"ERROR: {e}"

                for geom in DOSE_GEOMETRIES:
                    rows.append(
                        {
                            "detector": det_name,
                            "place": place,
                            "method": method_name,
                            "geometry": geom,
                            "dose_ref": dose_ref.get(geom, 0.0),
                            "dose_unfolded": dose_unfolded.get(geom, 0.0),
                            "status": status,
                        }
                    )

                done += 1
                if done % 200 == 0:
                    print(f"  [{done}/{total}] {det_name} / {place} / {method_name}")

    print(f"\nDone: {len(rows)} row records ({done} unfold calls)")

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "OK"]
    n_ok = len(ok)
    n_err = len(df) - n_ok
    print(f"OK: {n_ok}  |  ERROR: {n_err}")
    return df


# ── Angle computation ─────────────────────────────────────────────


def compute_angles(df):
    ok = df[df["status"] == "OK"].copy()
    methods = sorted(ok["method"].unique())
    geometries = DOSE_GEOMETRIES

    records = []
    for method in methods:
        mdf = ok[ok["method"] == method]
        row = {"method": method}
        all_xs, all_ys = [], []
        for geom in geometries:
            gdf = mdf[mdf["geometry"] == geom]
            xs = gdf["dose_ref"].values
            ys = gdf["dose_unfolded"].values
            k, theta, cls = fit_angle(xs, ys)
            row[f"k_{geom}"] = k
            row[f"theta_{geom}"] = theta
            row[f"class_{geom}"] = cls
            all_xs.extend(xs)
            all_ys.extend(ys)
        k_avg, theta_avg, cls_avg = fit_angle(all_xs, all_ys)
        row["k_avg"] = k_avg
        row["theta_avg"] = theta_avg
        row["class_avg"] = cls_avg
        records.append(row)

    return pd.DataFrame(records)


# ── Scatter plots ─────────────────────────────────────────────────


def plot_scatter(df, angles_df):
    ok = df[df["status"] == "OK"].copy()
    methods = sorted(angles_df["method"].tolist())
    n_methods = len(methods)
    n_cols = 5
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = np.atleast_2d(axes)

    geom_colors = {
        "AP": "#e41a1c",
        "PA": "#377eb8",
        "LLAT": "#4daf4a",
        "RLAT": "#984ea3",
        "ROT": "#ff7f00",
        "ISO": "#a65628",
    }

    for idx, method in enumerate(methods):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        mdf = ok[ok["method"] == method]

        for geom in DOSE_GEOMETRIES:
            gdf = mdf[mdf["geometry"] == geom]
            ax.scatter(
                gdf["dose_ref"],
                gdf["dose_unfolded"],
                c=geom_colors[geom],
                s=8,
                alpha=0.4,
                label=geom,
                edgecolors="none",
            )

        all_xs = mdf["dose_ref"].values
        all_ys = mdf["dose_unfolded"].values
        mask = (all_xs > 0) & (all_ys > 0)
        if mask.sum() > 2:
            xmin = float(np.min(all_xs[mask]))
            xmax = float(np.max(all_xs[mask]))
            line_x = np.linspace(xmin, xmax, 100)
            ax.plot(line_x, line_x, "k--", lw=1, alpha=0.5, label="45\u00b0")
            arow = angles_df[angles_df["method"] == method]
            if not arow.empty:
                k = arow.iloc[0]["k_avg"]
                if np.isfinite(k):
                    ax.plot(line_x, k * line_x, "r-", lw=1.2, alpha=0.7)
                    theta = arow.iloc[0]["theta_avg"]
                    cls = arow.iloc[0]["class_avg"]
                    ax.set_title(f"{method}\n{theta:.1f}\u00b0 | {cls}", fontsize=9)
                else:
                    ax.set_title(method, fontsize=9)
            else:
                ax.set_title(method, fontsize=9)
        else:
            ax.set_title(method, fontsize=9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)

    for idx in range(n_methods, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(DOSE_GEOMETRIES) + 1,
        fontsize=9, frameon=True,
    )
    fig.suptitle(
        "IAEA Compendium \u2014 Dose Rate: Reference vs Unfolded (251 spectra)",
        fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved: {OUTPUT_PNG}")


# ── ISO scatter plots ─────────────────────────────────────────────


def plot_iso_scatter(df, angles_df):
    ok = df[(df["status"] == "OK") & (df["geometry"] == "ISO")].copy()
    methods = sorted(angles_df["method"].tolist())

    ISO_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    from matplotlib.ticker import ScalarFormatter, LogLocator

    det_colors = {
        "GSF": "#377eb8",
        "PTB": "#e41a1c",
        "LANL": "#4daf4a",
        "JINR": "#ff7f00",
        "FERMILAB": "#984ea3",
    }

    for method in methods:
        mdf = ok[ok["method"] == method]

        arow = angles_df[angles_df["method"] == method]
        if arow.empty:
            continue
        k = arow.iloc[0]["k_ISO"]
        theta = arow.iloc[0]["theta_ISO"]
        cls = classify_simple(theta)

        all_xs = mdf["dose_ref"].values
        all_ys = mdf["dose_unfolded"].values
        mask = (all_xs > 0) & (all_ys > 0) & np.isfinite(all_xs) & np.isfinite(all_ys)
        if mask.sum() < 2:
            continue
        vmin = float(np.min(np.concatenate([all_xs[mask], all_ys[mask]])))
        vmax = float(np.max(np.concatenate([all_xs[mask], all_ys[mask]])))
        lo, hi = vmin * 0.5, vmax * 2.0

        fig, ax = plt.subplots(figsize=(6, 6))

        for det_name, color in det_colors.items():
            det_mdf = mdf[mdf["detector"] == det_name]
            if det_mdf.empty:
                continue
            xs = det_mdf["dose_ref"].values
            ys = det_mdf["dose_unfolded"].values
            det_mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(
                xs[det_mask], ys[det_mask],
                c=color, s=28, alpha=0.5, edgecolors="none", zorder=3,
                label=det_name,
            )

        line_x = np.logspace(np.log10(lo), np.log10(hi), 100)
        ax.plot(line_x, line_x, "k--", lw=1.2, alpha=0.5, label="45\u00b0 (perfect)")
        if np.isfinite(k):
            ax.plot(line_x, k * line_x, "r-", lw=1.5, alpha=0.8,
                    label=f"fitted: y = {k:.3f}x")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        fmt = ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 4))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.tick_params(labelsize=10)

        ax.set_xlabel(r"$\dot{H}(10)_{\mathrm{ref}}$ [pSv/s]", fontsize=12)
        ax.set_ylabel(r"$\dot{H}(10)_{\mathrm{unfolded}}$ [pSv/s]", fontsize=12)
        ax.set_title(
            f"ISO \u2014 {method}\n"
            f"\u03b8 = {theta:.2f}\u00b0 | k = {k:.4f} | {cls}",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="lower right", frameon=True)
        ax.grid(True, which="major", ls="--", alpha=0.3)

        out_path = ISO_PLOT_DIR / f"{method}.png"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"ISO plots saved: {ISO_PLOT_DIR}/ ({len(methods)} files)")


# ── Report generation ─────────────────────────────────────────────


def generate_report(df, angles_df):
    ok = df[df["status"] == "OK"]
    n_total = len(df)
    n_ok = len(ok)
    n_err = n_total - n_ok
    n_places = ok["place"].nunique()
    n_detectors = ok["detector"].nunique()

    lines = []
    a = lines.append

    a("# IAEA Compendium \u2014 Dose Rate Evaluation Report\n")
    a("**Script**: `tests/dose_rate_iaea_compendium.py`  ")
    a(f"**Total unfold calls**: {n_total} (OK: {n_ok}, ERROR: {n_err})  ")
    a(f"**Spectra**: {n_places}  |  **Detectors**: {n_detectors}  |  "
      f"**Methods**: {len(angles_df)}  |  **Geometries**: {len(DOSE_GEOMETRIES)}\n")

    a("---\n")

    # ── Overall ranking ──
    a("## Method Ranking (by average angle across all geometries)\n")
    ranked = angles_df.sort_values("theta_avg", key=lambda s: (s - 45.0).abs())
    a("| Rank | Method | \u03b8_avg (\u00b0) | k | Classification | Direction |")
    a("|------|--------|-----------|---|----------------|-----------|")
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        theta = r["theta_avg"]
        k = r["k_avg"]
        cls = classify_simple(theta)
        d = direction(theta)
        a(f"| {rank} | {r['method']} | {theta:.2f} | {k:.4f} | {cls} | {d} |")
    a("")

    # ── Per-geometry tables ──
    a("## Per-Geometry Analysis\n")
    for geom in DOSE_GEOMETRIES:
        a(f"### {geom}\n")
        col_theta = f"theta_{geom}"
        col_k = f"k_{geom}"
        gdf = angles_df.copy()
        gdf["_theta"] = gdf[col_theta]
        gdf["_k"] = gdf[col_k]
        gdf = gdf.sort_values("_theta", key=lambda s: (s - 45.0).abs())

        a("| Rank | Method | \u03b8 (\u00b0) | k | Classification | Direction |")
        a("|------|--------|-------|---|----------------|-----------|")
        for rank, (_, r) in enumerate(gdf.iterrows(), 1):
            theta = r["_theta"]
            k = r["_k"]
            cls = classify_simple(theta)
            d = direction(theta)
            a(f"| {rank} | {r['method']} | {theta:.2f} | {k:.4f} | {cls} | {d} |")
        a("")

    # ── ISO scatter plots ──
    a("## ISO Dose Scatter Plots\n")
    a("Reference ISO dose vs unfolded ISO dose across all 251 spectra and 5 detectors "
      "(GSF, PTB, LANL, JINR, FERMILAB). "
      "Dashed line = perfect 45\u00b0 reconstruction; solid red = fitted line y = k\u00b7x.\n")
    methods_sorted = sorted(angles_df["method"].tolist())
    for method in methods_sorted:
        a(f"### {method}\n")
        a(f"![{method}](iaea_compendium_iso_plots/{method}.png)\n")

    # ── Classification summary ──
    a("## Classification Summary\n")
    a("| Method | AP | PA | LLAT | RLAT | ROT | ISO | Average |")
    a("|--------|----|----|------|------|-----|-----|---------|")
    for _, r in angles_df.iterrows():
        cells = [r[f"class_{geom}"].split(" (")[0] for geom in DOSE_GEOMETRIES]
        a(f"| {r['method']} | " + " | ".join(cells) + f" | {r['class_avg'].split(' (')[0]} |")
    a("")

    # ── Best / worst ──
    a("## Best and Worst Methods\n")
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    a(f"- **Best method**: {best['method']} \u2014 \u03b8 = {best['theta_avg']:.2f}\u00b0 "
      f"(k = {best['k_avg']:.4f}), {best['class_avg']}")
    a(f"- **Worst method**: {worst['method']} \u2014 \u03b8 = {worst['theta_avg']:.2f}\u00b0 "
      f"(k = {worst['k_avg']:.4f}), {worst['class_avg']}")
    a("")

    # ── Per-detector summary ──
    a("## Per-Detector Summary\n")
    for det_name in sorted(ok["detector"].unique()):
        ddf = ok[ok["detector"] == det_name]
        a(f"### {det_name}\n")
        det_rows = []
        for method in sorted(ddf["method"].unique()):
            mdf = ddf[ddf["method"] == method]
            all_xs, all_ys = [], []
            for geom in DOSE_GEOMETRIES:
                gdf = mdf[mdf["geometry"] == geom]
                all_xs.extend(gdf["dose_ref"].values)
                all_ys.extend(gdf["dose_unfolded"].values)
            k, theta, cls = fit_angle(all_xs, all_ys)
            det_rows.append((method, theta, k, cls))
        det_rows.sort(key=lambda x: abs(x[1] - 45.0) if np.isfinite(x[1]) else 999)
        a("| Method | \u03b8 (\u00b0) | k | Classification |")
        a("|--------|-------|---|----------------|")
        for method, theta, k, cls in det_rows:
            a(f"| {method} | {theta:.2f} | {k:.4f} | {cls} |")
        a("")

    # ── Notes ──
    a("---\n")
    a("**Angle interpretation**:\n")
    a("- \u03b8 = 45.0\u00b0 \u2192 perfect dose reconstruction (k = 1.0)\n")
    a("- \u03b8 > 45\u00b0 \u2192 method overestimates dose (k > 1.0)\n")
    a("- \u03b8 < 45\u00b0 \u2192 method underestimates dose (k < 1.0)\n")
    a("- Classification: Excellent (|\u0394| < 1\u00b0), Good (1-5\u00b0), Fair (5-10\u00b0), Poor (>10\u00b0)\n")

    report = "\n".join(lines)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"Report saved: {OUTPUT_MD}")
    return report


# ── Entry point ───────────────────────────────────────────────────


def main():
    if not IAEA_CSV.exists():
        sys.exit(f"CSV not found: {IAEA_CSV}")

    print("=" * 70)
    print("  IAEA Compendium \u2014 Dose Rate Evaluation")
    print("=" * 70)

    df = run_evaluation()
    angles = compute_angles(df)
    plot_scatter(df, angles)
    plot_iso_scatter(df, angles)
    generate_report(df, angles)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Raw results saved: {OUTPUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
