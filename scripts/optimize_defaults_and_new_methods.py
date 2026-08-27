"""PART 1: Optimize default parameters for key unfolding methods.
PART 2: Test newly implemented ensemble and iterative refinement methods.

Uses synthetic spectra (thermal, fast, mixed) generated via parametric_model
from unfold_fruit_like.py.  Runs a small grid search over tunable parameters
and reports quality metrics for each configuration.
"""

import itertools
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ─── Reproducibility ───────────────────────────────────────────────
np.random.seed(42)

# ─── Imports ───────────────────────────────────────────────────────
from bssunfold import Detector
from bssunfold.core.unfold_fruit_like import parametric_model
from bssunfold.core._base_unfolder import _build_system

# ─── Quality metrics (inline to avoid dependency issues) ───────────


def relative_residual(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """||Ax - b|| / ||b||"""
    r = A @ x - b
    denom = np.linalg.norm(b)
    if denom < 1e-30:
        return 0.0
    return float(np.linalg.norm(r) / denom)


def smoothness(x: np.ndarray) -> float:
    """Std of second differences (lower = smoother)."""
    if len(x) < 3:
        return 0.0
    d2 = np.diff(x, n=2)
    return float(np.std(d2))


def total_flux_ratio(x: np.ndarray, x_true: np.ndarray) -> float:
    """sum(x) / sum(x_true)."""
    s_true = np.sum(x_true)
    if s_true < 1e-30:
        return 0.0
    return float(np.sum(x) / s_true)


def l2_error(x: np.ndarray, x_true: np.ndarray) -> float:
    """Relative L2 error."""
    denom = np.linalg.norm(x_true)
    if denom < 1e-30:
        return 0.0
    return float(np.linalg.norm(x - x_true) / denom)


# ─── Setup ─────────────────────────────────────────────────────────

def build_detector_and_system():
    """Create Detector and build the raw A, b system."""
    d = Detector()
    # Build the system matrix A (m_detectors x n_energy)
    selected = d.detector_names
    # Use all detectors for synthetic test
    dummy_readings = {name: 1.0 for name in selected}
    A, _, sel = _build_system(dummy_readings, d.detector_names, d.sensitivities)
    return d, A, sel


def make_synthetic_spectra(E_MeV: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate 3 synthetic test spectra using parametric_model."""
    spectra = {}

    # 1. Thermal: strong Maxwellian peak at ~0.025 eV
    spectra["thermal"] = parametric_model(
        E_MeV,
        A_th=1e8,
        T_th=0.025e-6,
        A_epi=1e-7,
        A_f=1e-9,
        T_ev=1.0,
        epi_max=0.1,
    )

    # 2. Fast: dominated by evaporation spectrum
    spectra["fast"] = parametric_model(
        E_MeV,
        A_th=1e-10,
        T_th=0.025e-6,
        A_epi=1e-10,
        A_f=1e-6,
        T_ev=2.0,
        epi_max=0.1,
    )

    # 3. Mixed: all components
    spectra["mixed"] = parametric_model(
        E_MeV,
        A_th=1e6,
        T_th=0.025e-6,
        A_epi=1e-6,
        A_f=1e-7,
        T_ev=1.5,
        epi_max=0.1,
    )

    # Ensure non-negativity
    for k in spectra:
        spectra[k] = np.maximum(spectra[k], 0)

    return spectra


def compute_readings(A: np.ndarray, spectrum: np.ndarray, detector_names: List[str],
                      noise_level: float = 0.01) -> Dict[str, float]:
    """Compute detector readings with optional noise."""
    b = A @ spectrum
    # Add Poisson-like noise
    rng = np.random.default_rng(42)
    noise = noise_level * rng.standard_normal(len(b))
    b_noisy = b * (1.0 + noise)
    b_noisy = np.maximum(b_noisy, 0)
    return {name: float(val) for name, val in zip(detector_names, b_noisy)}


def evaluate(x: np.ndarray, x_true: np.ndarray, A: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute all quality metrics."""
    x = np.maximum(x, 0)
    return {
        "rel_residual": relative_residual(x, A, b),
        "smoothness": smoothness(x),
        "flux_ratio": total_flux_ratio(x, x_true),
        "l2_error": l2_error(x, x_true),
    }


def score_metrics(metrics: Dict[str, float]) -> float:
    """Composite score: lower is better.

    Combines rel_residual, smoothness (normalized), and |flux_ratio - 1|.
    """
    rr = metrics["rel_residual"]
    sm = metrics["smoothness"]
    fr = abs(metrics["flux_ratio"] - 1.0)
    l2 = metrics["l2_error"]
    # Weighted sum (equal weights)
    return rr + 0.1 * sm + fr + l2


# ─── PART 1: Grid search ───────────────────────────────────────────

def run_unfold_method(det: Detector, method_name: str, readings: Dict[str, float],
                      **kwargs) -> Dict[str, Any]:
    """Call a detector unfold method with given kwargs."""
    method = getattr(det, method_name)
    return method(readings, **kwargs)


def grid_search_method(det: Detector, method_name: str, readings: Dict[str, float],
                       x_true: np.ndarray, A: np.ndarray, b: np.ndarray,
                       param_grid: Dict[str, List]) -> Tuple[Dict[str, Any], List[Dict]]:
    """Run grid search for a method. Returns (best_config, all_results)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_results = []
    best_score = np.inf
    best_config = None
    best_metrics = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            result = run_unfold_method(det, method_name, readings, **params)
            x = result["spectrum"]
            metrics = evaluate(x, x_true, A, b)
            sc = score_metrics(metrics)
            metrics["score"] = sc
            metrics["params"] = params
            all_results.append(metrics)
            if sc < best_score:
                best_score = sc
                best_config = params.copy()
                best_metrics = metrics.copy()
        except Exception as e:
            # Skip failed configurations
            pass

    return best_config, best_metrics, all_results


def define_grids() -> Dict[str, Dict[str, List]]:
    """Define parameter grids for each method."""
    return {
        "unfold_mlem": {
            "max_iterations": [100, 200, 500, 1000],
            "tolerance": [1e-3, 1e-4, 1e-6],
        },
        "unfold_bayes": {
            "max_iterations": [500, 1000, 2000, 4000],
            "tolerance": [1e-2, 1e-3, 1e-4],
        },
        "unfold_landweber": {
            "max_iterations": [100, 200, 500, 1000],
            "tolerance": [1e-3, 1e-5, 1e-8],
        },
        "unfold_cgls": {
            "max_iterations": [20, 50, 100, 200],
            "tolerance": [1e-8, 1e-10, 1e-12],
        },
        "unfold_gravel": {
            "max_iterations": [100, 500, 1000],
            "tolerance": [1e-4, 1e-6, 1e-8],
            "regularization": [0.0, 0.01, 0.1],
        },
        "unfold_ferdor": {
            "max_iterations": [50, 100, 200],
            "smoothing": [1e-4, 1e-3, 1e-2],
            "chi_squared_target": [0.8, 1.0, 1.5],
        },
        "unfold_fista": {
            "max_iterations": [100, 300, 500],
            "tolerance": [1e-4, 1e-6, 1e-8],
            "regularization": [0.0, 1e-4, 1e-2],
        },
        "unfold_tikhonov_tv": {
            "max_iterations": [50, 100, 200],
            "tolerance": [1e-3, 1e-4, 1e-6],
        },
        "unfold_statreg": {
            "unfoldermethod": ["EmpiricalBayes", "User"],
            "regularization": [None, 1e-6, 1e-4, 1e-2],
        },
    }


def part1_optimize_defaults():
    """PART 1: Run grid search across methods and spectra."""
    print("=" * 80)
    print("PART 1: DEFAULT PARAMETER OPTIMIZATION")
    print("=" * 80)

    det, A, sel = build_detector_and_system()
    E = det.E_MeV
    spectra = make_synthetic_spectra(E)
    grids = define_grids()

    # Store: method -> spectrum_type -> (best_config, best_metrics)
    summary: Dict[str, Dict[str, Any]] = {}

    for method_name, grid in grids.items():
        print(f"\n{'─' * 60}")
        print(f"Method: {method_name}")
        print(f"{'─' * 60}")

        method_summary = {}
        for spec_name, x_true in spectra.items():
            readings = compute_readings(A, x_true, sel, noise_level=0.02)
            b = np.array([readings[n] for n in sel])

            # For statreg, filter out zero readings
            if method_name == "unfold_statreg":
                readings = {k: v for k, v in readings.items() if v > 0}
                if len(readings) < 4:
                    print(f"  {spec_name}: skipped (too few positive readings)")
                    continue
                b = np.array([readings[n] for n in sel if n in readings])

            best_cfg, best_met, all_res = grid_search_method(
                det, method_name, readings, x_true, A, b, grid
            )

            if best_cfg is not None:
                method_summary[spec_name] = {
                    "best_params": best_cfg,
                    "metrics": best_met,
                }
                print(f"  {spec_name:8s}: score={best_met['score']:.4f}  "
                      f"rel_res={best_met['rel_residual']:.4e}  "
                      f"flux_ratio={best_met['flux_ratio']:.3f}  "
                      f"l2_err={best_met['l2_error']:.4f}  "
                      f"smooth={best_met['smoothness']:.4e}")
                print(f"            params={best_cfg}")
            else:
                print(f"  {spec_name:8s}: ALL CONFIGURATIONS FAILED")

        summary[method_name] = method_summary

    return summary, det, A, sel, spectra


def print_recommendations(summary: Dict[str, Dict[str, Any]]):
    """Print recommended optimized defaults."""
    print("\n" + "=" * 80)
    print("RECOMMENDED OPTIMIZED DEFAULTS")
    print("=" * 80)

    for method_name, spec_results in summary.items():
        if not spec_results:
            continue
        # Average the best params across spectrum types
        # (pick the median or most common value per param)
        all_params = []
        for spec_name, data in spec_results.items():
            all_params.append(data["best_params"])

        # For each parameter key, pick the value that appears most often
        # or the median for numeric values
        recommended = {}
        keys = all_params[0].keys()
        for k in keys:
            vals = [p[k] for p in all_params if k in p]
            # Filter out None values
            vals_clean = [v for v in vals if v is not None]
            if not vals_clean:
                recommended[k] = None
                continue
            # For numeric, pick median; for str, pick most common
            if isinstance(vals_clean[0], (int, float)):
                recommended[k] = float(np.median(vals_clean))
            else:
                # Most common string
                from collections import Counter
                recommended[k] = Counter(vals_clean).most_common(1)[0][0]

        # Format
        params_str = ", ".join(f"{k}={v!r}" for k, v in recommended.items())
        print(f"\n  {method_name}:")
        print(f"    {params_str}")


def print_comparison_table(summary: Dict[str, Dict[str, Any]]):
    """Print quality metrics comparison table."""
    print("\n" + "=" * 80)
    print("QUALITY METRICS COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<22s} {'Spectrum':<10s} {'Rel Resid':<12s} {'Flux Rat':<10s} "
          f"{'L2 Err':<10s} {'Smoothness':<12s} {'Score':<10s}")
    print("-" * 86)

    for method_name, spec_results in summary.items():
        for spec_name, data in spec_results.items():
            m = data["metrics"]
            print(f"{method_name:<22s} {spec_name:<10s} {m['rel_residual']:<12.4e} "
                  f"{m['flux_ratio']:<10.4f} {m['l2_error']:<10.4f} "
                  f"{m['smoothness']:<12.4e} {m['score']:<10.4f}")


# ─── PART 2: Test new methods ──────────────────────────────────────

def part2_test_new_methods(det: Detector, A: np.ndarray, sel: List[str],
                           spectra: Dict[str, np.ndarray]):
    """PART 2: Test the new ensemble and iterative refinement methods."""
    print("\n" + "=" * 80)
    print("PART 2: NEW METHOD EVALUATION")
    print("=" * 80)

    results_new = {}

    for spec_name, x_true in spectra.items():
        readings = compute_readings(A, x_true, sel, noise_level=0.02)
        b = np.array([readings[n] for n in sel])
        print(f"\n--- Spectrum: {spec_name} ---")

        # Test ensemble method (default: weighted_average)
        try:
            t0 = time.time()
            r_ens = det.unfold_ensemble(readings, combination="weighted_average")
            t_ens = time.time() - t0
            m_ens = evaluate(r_ens["spectrum"], x_true, A, b)
            m_ens["time"] = t_ens
            print(f"  Ensemble (weighted_avg):  score={score_metrics(m_ens):.4f}  "
                  f"rel_res={m_ens['rel_residual']:.4e}  flux_ratio={m_ens['flux_ratio']:.3f}  "
                  f"l2_err={m_ens['l2_error']:.4f}  time={t_ens:.2f}s")
        except Exception as e:
            print(f"  Ensemble (weighted_avg): FAILED - {e}")
            m_ens = None

        # Test ensemble (median)
        try:
            t0 = time.time()
            r_ens_med = det.unfold_ensemble(readings, combination="median")
            t_med = time.time() - t0
            m_med = evaluate(r_ens_med["spectrum"], x_true, A, b)
            print(f"  Ensemble (median):        score={score_metrics(m_med):.4f}  "
                  f"rel_res={m_med['rel_residual']:.4e}  flux_ratio={m_med['flux_ratio']:.3f}  "
                  f"l2_err={m_med['l2_error']:.4f}  time={t_med:.2f}s")
        except Exception as e:
            print(f"  Ensemble (median): FAILED - {e}")

        # Test iterative refinement
        try:
            t0 = time.time()
            r_ir = det.unfold_iterative_refinement(readings)
            t_ir = time.time() - t0
            m_ir = evaluate(r_ir["spectrum"], x_true, A, b)
            m_ir["time"] = t_ir
            print(f"  IterativeRefinement:      score={score_metrics(m_ir):.4f}  "
                  f"rel_res={m_ir['rel_residual']:.4e}  flux_ratio={m_ir['flux_ratio']:.3f}  "
                  f"l2_err={m_ir['l2_error']:.4f}  time={t_ir:.2f}s")
        except Exception as e:
            print(f"  IterativeRefinement: FAILED - {e}")
            m_ir = None

        results_new[spec_name] = {
            "ensemble_wavg": m_ens,
            "iterative_refinement": m_ir,
        }

    return results_new


# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("Neutron Spectrum Unfolding: Default Parameter Optimization & New Methods")
    print("=" * 80)

    # PART 1
    summary, det, A, sel, spectra = part1_optimize_defaults()

    print_recommendations(summary)
    print_comparison_table(summary)

    # PART 2
    new_results = part2_test_new_methods(det, A, sel, spectra)

    # Summary of new methods
    print("\n" + "=" * 80)
    print("LIST OF NEW METHODS IMPLEMENTED")
    print("=" * 80)
    print("""
  1. unfold_ensemble (src/bssunfold/core/unfold_ensemble.py)
     - Combines multiple base solvers (MLEM, Bayes, Landweber, CGLS, GRAVEL)
     - Combination strategies: weighted_average, median, trimmed_mean, best_residual
     - Weights derived from inverse residuals or user-supplied
     - Monte-Carlo uncertainty support

  2. unfold_iterative_refinement (src/bssunfold/core/unfold_iterative_refinement.py)
     - Two-pass method: MLEM first pass + Landweber second pass on residual
     - First pass captures gross structure, second pass corrects errors
     - Auto alpha selection via line search on residual norm
     - Monte-Carlo uncertainty support

  Both methods integrated into Detector class and core __init__.py.
""")

    print("\nDone.")


if __name__ == "__main__":
    main()
