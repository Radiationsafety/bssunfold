"""Test all unfolding methods against 20 IAEA reference spectra.

For each detector type (GSF, PTB, LANL) and each reference spectrum:
  1. Interpolate reference spectrum onto detector energy grid
  2. Compute synthetic detector readings
  3. Run each unfolding method
  4. Compare unfolded spectrum with reference using all comparison metrics
  5. Flag any method/spectrum pair exceeding warning thresholds
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from bssunfold import Detector, RF_PTB, RF_LANL
from bssunfold.utils.comparison import compare_spectra, _ALL_METRICS

logger = logging.getLogger("iaea_validation")

CSV_PATH = Path(__file__).parent / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
ALL_METRIC_KEYS = list(_ALL_METRICS.keys())

WARNING_THRESHOLDS = {
    "cosine_similarity": ("lt", 0.85),
    "r2_score": ("lt", 0.7),
    "mape": ("gt", 100.0),
    "total_flux_ratio": ("out_of_range", (0.5, 2.0)),
}

KEY_METRICS = list(WARNING_THRESHOLDS.keys())

DETECTOR_CONFIGS = {
    "GSF": lambda: Detector(),
    "PTB": lambda: Detector(pd.DataFrame(RF_PTB)),
    "LANL": lambda: Detector(pd.DataFrame(RF_LANL)),
}

METHODS = [
    ("cvxpy", lambda d, r: d.unfold_cvxpy(r, regularization=1e-3, save_result=False)),
    ("qpsolvers", lambda d, r: d.unfold_qpsolvers(r, regularization=1e-3, save_result=False)),
    ("landweber", lambda d, r: d.unfold_landweber(r, max_iterations=500, save_result=False)),
    ("mlem", lambda d, r: d.unfold_mlem(r, max_iterations=500, save_result=False)),
    ("doroshenko", lambda d, r: d.unfold_doroshenko(r, max_iterations=500, save_result=False)),
    ("kaczmarz", lambda d, r: d.unfold_kaczmarz(r, max_iterations=500, save_result=False)),
    ("gravel", lambda d, r: d.unfold_gravel(r, max_iterations=500, save_result=False)),
    ("maxed", lambda d, r: d.unfold_maxed(r, max_iterations=500, save_result=False)),
    ("tikhonov_legendre", lambda d, r: d.unfold_tikhonov_legendre(r, delta=0.05, save_result=False)),
    ("bayes", lambda d, r: d.unfold_bayes(r, max_iterations=500, save_result=False)),
    ("bayes_spline", lambda d, r: d.unfold_bayes_spline_regularization(r, max_iterations=500, save_result=False)),
    ("statreg", lambda d, r: d.unfold_statreg(r, save_result=False)),
    ("scipy_direct", lambda d, r: d.unfold_scipy_direct_method(r, method="cg", max_iterations=500, save_result=False)),
    ("tsvd", lambda d, r: d.unfold_tsvd(r, method="discrepancy", save_result=False)),
    ("lmfit", lambda d, r: d.unfold_lmfit(r, method="lbfgsb", model_name="elastic", regularization=1e-4, save_result=False)),
    ("mlem_odl", lambda d, r: d.unfold_mlem_odl(r, max_iterations=500, save_result=False)),
]


@pytest.fixture(scope="session")
def reference_data():
    return pd.read_csv(CSV_PATH)


def _build_spectrum_dict(energy_col, spectrum_col):
    return {"E_MeV": energy_col, "Phi": spectrum_col}


def _check_warnings(metrics):
    msgs = []
    for m, (op, threshold) in WARNING_THRESHOLDS.items():
        val = metrics.get(m, np.nan)
        if np.isnan(val):
            continue
        if op == "lt" and val < threshold:
            msgs.append(f"{m}={val:.4f} < {threshold}")
        elif op == "gt" and val > threshold:
            msgs.append(f"{m}={val:.1f} > {threshold}")
        elif op == "out_of_range":
            lo, hi = threshold
            if val < lo or val > hi:
                msgs.append(f"{m}={val:.4f} ∉ [{lo}, {hi}]")
    return msgs


@pytest.mark.parametrize("detector_type", ["GSF", "PTB", "LANL"])
def test_iaea_validation(detector_type, reference_data):
    """Run all unfolding methods against all 20 IAEA reference spectra for a given detector."""
    detector = DETECTOR_CONFIGS[detector_type]()
    ref_energy = reference_data["E_MeV"].values
    spectrum_names = [c for c in reference_data.columns if c != "E_MeV"]

    all_warnings = []
    rows = []

    for spec_name in spectrum_names:
        spec_values = reference_data[spec_name].values.astype(float)
        ref_dict = _build_spectrum_dict(ref_energy, spec_values)

        interp_df = detector.discretize_spectra(ref_dict)
        interp_spectrum = interp_df["Phi"].values

        readings = detector.get_effective_readings_for_spectra(ref_dict)

        for method_name, method_fn in METHODS:
            try:
                result = method_fn(detector, readings)
            except ImportError:
                rows.append({
                    "spectrum": spec_name, "method": method_name, "status": "SKIP",
                    **{k: np.nan for k in ALL_METRIC_KEYS},
                })
                continue
            except Exception as e:
                rows.append({
                    "spectrum": spec_name, "method": method_name, "status": f"ERROR: {e}",
                    **{k: np.nan for k in ALL_METRIC_KEYS},
                })
                continue

            if result is None or "spectrum" not in result:
                rows.append({
                    "spectrum": spec_name, "method": method_name, "status": "ERROR: no spectrum",
                    **{k: np.nan for k in ALL_METRIC_KEYS},
                })
                continue

            unfolded = result["spectrum"]
            metrics = compare_spectra(interp_spectrum, unfolded)

            warning_msgs = _check_warnings(metrics)
            status = "OK"
            if warning_msgs:
                status = "WARN: " + "; ".join(warning_msgs)
                all_warnings.append((detector_type, spec_name, method_name, warning_msgs))

            row = {"spectrum": spec_name, "method": method_name, "status": status}
            row.update(metrics)
            rows.append(row)

    if not rows:
        pytest.skip(f"No results for detector type {detector_type}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["spectrum", "method"]).reset_index(drop=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"IAEA Validation Report: {detector_type}")
    logger.info(f"Detector: {detector.n_detectors} spheres, {detector.n_energy_bins} energy bins")
    logger.info(f"Reference spectra: {len(spectrum_names)}")
    logger.info(f"Methods tested: {result_df['method'].nunique()}")
    logger.info(f"{'='*80}")

    n_total = len(result_df)
    n_ok = (result_df["status"] == "OK").sum()
    n_skip = (result_df["status"] == "SKIP").sum()
    n_warn = result_df["status"].str.startswith("WARN").sum()
    n_err = result_df["status"].str.startswith("ERROR").sum()

    logger.info(f"Total: {n_total} | OK: {n_ok} | WARN: {n_warn} | SKIP: {n_skip} | ERR: {n_err}")

    if n_warn > 0:
        logger.warning(f"\nWarnings ({n_warn}):")
        warn_df = result_df[result_df["status"].str.startswith("WARN")]
        for _, r in warn_df.iterrows():
            extra = " | ".join(
                f"{m}={r[m]:.4f}" if not np.isnan(r[m]) else f"{m}=N/A"
                for m in KEY_METRICS if m in warn_df.columns
            )
            logger.warning(f"  {r['spectrum']:30s} | {r['method']:20s} | {r['status']:50s} | {extra}")

    if n_err > 0:
        logger.error(f"\nErrors ({n_err}):")
        err_df = result_df[result_df["status"].str.startswith("ERROR")]
        for _, r in err_df.iterrows():
            logger.error(f"  {r['spectrum']:30s} | {r['method']:20s} | {r['status']}")

    summary_path = Path(__file__).parent / f"iaea_validation_{detector_type}.csv"
    result_df.to_csv(summary_path, index=False)
    logger.info(f"\nDetailed results saved to: {summary_path}")

    assert n_err == 0, (
        f"{detector_type}: {n_err} methods failed across all spectra. "
        f"Check log for details."
    )

    if n_warn > 0:
        warn_summary = "; ".join(
            f"{s}/{m}: {'; '.join(w)}"
            for dt, s, m, w in all_warnings[:5]
        )
        logger.warning(
            f"{detector_type}: {n_warn}/{n_total} combinations exceed thresholds. "
            f"First {min(5, len(all_warnings))}: {warn_summary}"
        )
