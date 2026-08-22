#!/usr/bin/env python3
"""Comprehensive benchmark of all bssunfold methods on IAEA test spectra.

This script runs all available unfolding methods on the IAEA Compendium dataset
and Monte Carlo calculated spectra, then analyzes which method performs best
for different types of spectra (bins).

Usage:
    python tests/benchmark_all_methods.py
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bssunfold import Detector, RF_PTB, RF_LANL
from bssunfold.utils.comparison import compare_spectra, cosine_similarity, r2_score, wasserstein_dist

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
IAEA_COMPENDIUM_PATH = Path(__file__).parent / "IAEA_Compendium_dataset.csv"

DETECTOR_CONFIGS = {
    "GSF": lambda: Detector(),
    "PTB": lambda: Detector(pd.DataFrame(RF_PTB)),
    "LANL": lambda: Detector(pd.DataFrame(RF_LANL)),
}

# All available unfolding methods from bssunfold.core
METHODS = {
    # Basic iterative methods
    "landweber": lambda d, r: d.unfold_landweber(r, max_iterations=500, save_result=False),
    "mlem": lambda d, r: d.unfold_mlem(r, max_iterations=500, save_result=False),
    "mlem_odl": lambda d, r: d.unfold_mlem_odl(r, max_iterations=500, save_result=False),
    "mlem_stop": lambda d, r: d.unfold_mlem_stop(r, save_result=False),
    "kaczmarz": lambda d, r: d.unfold_kaczmarz(r, max_iterations=500, save_result=False),
    "doroshenko": lambda d, r: d.unfold_doroshenko(r, max_iterations=500, save_result=False),
    "sart": lambda d, r: d.unfold_sart(r, max_iterations=500, save_result=False),
    "cgls": lambda d, r: d.unfold_cgls(r, max_iterations=500, save_result=False),
    
    # Regularized methods
    "cvxpy": lambda d, r: d.unfold_cvxpy(r, regularization=1e-3, save_result=False),
    "qpsolvers": lambda d, r: d.unfold_qpsolvers(r, regularization=1e-3, save_result=False),
    "tikhonov_legendre": lambda d, r: d.unfold_tikhonov_legendre(r, delta=0.05, save_result=False),
    "tikhonov_tv": lambda d, r: d.unfold_tikhonov_tv(r, save_result=False),
    "tsvd": lambda d, r: d.unfold_tsvd(r, method="discrepancy", save_result=False),
    "lanczos": lambda d, r: d.unfold_lanczos(r, method="discrepancy", save_result=False),
    "gks": lambda d, r: d.unfold_gks(r, save_result=False),
    "statreg": lambda d, r: d.unfold_statreg(r, save_result=False),
    "scipy_direct": lambda d, r: d.unfold_scipy_direct_method(r, method="cg", max_iterations=500, save_result=False),
    "fista": lambda d, r: d.unfold_fista(r, save_result=False),
    "hybrid_gmres": lambda d, r: d.unfold_hybrid_gmres(r, save_result=False),
    
    # Bayesian and EM methods
    "bayes": lambda d, r: d.unfold_bayes(r, max_iterations=500, save_result=False),
    "bayes_spline": lambda d, r: d.unfold_bayes_spline_regularization(r, max_iterations=500, save_result=False),
    "bayesian_parametric": lambda d, r: d.unfold_bayesian_parametric(r, n_samples=100, burn_in=20, save_result=False),
    "mcmc": lambda d, r: d.unfold_mcmc(r, n_samples=1000, burn_in=200, save_result=False),
    "maxed": lambda d, r: d.unfold_maxed(r, max_iterations=500, save_result=False),
    "gravel": lambda d, r: d.unfold_gravel(r, max_iterations=500, save_result=False),
    "osem": lambda d, r: d.unfold_osem(r, max_iterations=500, save_result=False),
    "mapem": lambda d, r: d.unfold_mapem(r, max_iterations=500, save_result=False),
    "bsrem": lambda d, r: d.unfold_bsrem(r, max_iterations=500, save_result=False),
    
    # Parametric methods
    "parametric": lambda d, r: d.unfold_parametric(r, save_result=False),
    "parametric2": lambda d, r: d.unfold_parametric2(r, save_result=False),
    "lmfit": lambda d, r: d.unfold_lmfit(r, method="lbfgsb", model_name="elastic", regularization=1e-4, save_result=False),
    "fruit_like": lambda d, r: d.unfold_fruit_like(r, save_result=False),
    "hybrid_parametric": lambda d, r: d.unfold_hybrid_parametric(r, refinement_method="landweber", save_result=False),
    
    # Advanced/optimization methods
    "genetic": lambda d, r: d.unfold_genetic(r, population_size=50, generations=100, save_result=False),
    "mystic": lambda d, r: d.unfold_mystic(r, save_result=False),
    "cs": lambda d, r: d.unfold_cs(r, save_result=False),
    "epic": lambda d, r: d.unfold_epic(r, save_result=False),
    "interpret": lambda d, r: d.unfold_interpret(r, save_result=False),
    "maeo": lambda d, r: d.unfold_maeo(r, save_result=False),
    "maeo_ensemble": lambda d, r: d.unfold_maeo_ensemble(r, save_result=False),
    
    # Constraint programming methods
    "scip": lambda d, r: d.unfold_scip(r, save_result=False),
    "docplex": lambda d, r: d.unfold_docplex(r, save_result=False),
    "smt": lambda d, r: d.unfold_smt(r, save_result=False),
    
    # Other methods
    "bunki": lambda d, r: d.unfold_bunki(r, save_result=False),
    "bunkiut": lambda d, r: d.unfold_bunkiut(r, save_result=False),
    "rebunki": lambda d, r: d.unfold_rebunki(r, save_result=False),
    "ferdor": lambda d, r: d.unfold_ferdor(r, save_result=False),
    "nsduaz": lambda d, r: d.unfold_nsduaz(r, save_result=False),
    "sandii": lambda d, r: d.unfold_sandii(r, save_result=False),
    "reconst": lambda d, r: d.unfold_reconst(r, save_result=False),
}

KEY_METRICS = [
    "cosine_similarity",
    "r2_score", 
    "wasserstein_dist",
    "mean_squared_error",
    "total_flux_ratio",
    "dose_difference_percent",
]


def load_monte_carlo_data() -> pd.DataFrame:
    """Load Monte Carlo calculated spectra."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Monte Carlo data not found: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


def load_iaea_compendium() -> pd.DataFrame:
    """Load IAEA Compendium dataset."""
    if not IAEA_COMPENDIUM_PATH.exists():
        raise FileNotFoundError(f"IAEA Compendium not found: {IAEA_COMPENDIUM_PATH}")
    return pd.read_csv(IAEA_COMPENDIUM_PATH)


def unfold_one(
    detector: Detector,
    readings: Dict[str, float],
    method_name: str,
    method_fn,
    timeout: float = 60.0
) -> Tuple[Optional[Dict], Optional[np.ndarray], str]:
    """Unfold with a single method. Returns (result_dict, spectrum_array, status)."""
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def handler(signum, frame):
        raise TimeoutError(f"Method {method_name} timed out after {timeout}s")
    
    try:
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        
        start_time = time.time()
        result = method_fn(detector, readings)
        elapsed = time.time() - start_time
        
        # Cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        if result and "spectrum" in result:
            return result, result["spectrum"], f"OK ({elapsed:.2f}s)"
        else:
            return None, None, "ERROR: no spectrum in result"
            
    except TimeoutError as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None, None, f"TIMEOUT: {e}"
    except ImportError as e:
        return None, None, f"SKIP: ImportError - {e}"
    except Exception as e:
        return None, None, f"ERROR: {type(e).__name__}: {e}"


def compute_spectrum_features(spectrum: np.ndarray, energy: np.ndarray) -> Dict[str, float]:
    """Compute features of a spectrum for binning/classification."""
    # Normalize
    s = spectrum / (np.sum(spectrum) + 1e-10)
    
    # Basic stats
    mean_e = np.sum(s * energy)
    var_e = np.sum(s * (energy - mean_e)**2)
    std_e = np.sqrt(var_e)
    
    # Peak location
    peak_idx = np.argmax(s)
    peak_energy = energy[peak_idx]
    
    # Spectral shape indicators
    thermal_fraction = np.sum(s[energy < 0.5])  # E < 0.5 MeV
    fast_fraction = np.sum(s[energy > 5.0])     # E > 5.0 MeV
    intermediate_fraction = 1.0 - thermal_fraction - fast_fraction
    
    # Hardness ratio
    hardness = fast_fraction / (thermal_fraction + 1e-10)
    
    # Entropy
    s_safe = s + 1e-10
    entropy = -np.sum(s_safe * np.log(s_safe))
    
    return {
        "mean_energy": mean_e,
        "std_energy": std_e,
        "peak_energy": peak_energy,
        "thermal_fraction": thermal_fraction,
        "fast_fraction": fast_fraction,
        "intermediate_fraction": intermediate_fraction,
        "hardness_ratio": hardness,
        "entropy": entropy,
    }


def cluster_spectra_by_features(features_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Cluster spectra based on their features to create bins."""
    from sklearn.cluster import KMeans
    
    feature_cols = [
        "mean_energy", "std_energy", "peak_energy",
        "thermal_fraction", "fast_fraction", "hardness_ratio", "entropy"
    ]
    
    X = features_df[feature_cols].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_df["spectrum_bin"] = clusters
    
    return features_df


def run_benchmark(
    spectra_df: pd.DataFrame,
    detector_name: str,
    detector: Detector,
    methods_to_run: Optional[Dict] = None,
    max_spectra: Optional[int] = None
) -> pd.DataFrame:
    """Run benchmark for all methods on given spectra."""
    
    if methods_to_run is None:
        methods_to_run = METHODS
    
    energy_cols = [c for c in spectra_df.columns if c.startswith("Energy_bin")]
    spectrum_names = [c for c in spectra_df.columns if not c.startswith("Energy_bin") and c != "Place"]
    
    if max_spectra:
        spectrum_names = spectrum_names[:max_spectra]
    
    ref_energy = spectra_df["E_MeV"].values if "E_MeV" in spectra_df.columns else None
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"Benchmarking {len(methods_to_run)} methods on {len(spectrum_names)} spectra ({detector_name})")
    print(f"{'='*80}\n")
    
    for spec_idx, spec_name in enumerate(spectrum_names):
        spec_values = spectra_df[spec_name].values.astype(float)
        
        # Build reference spectrum dict
        if ref_energy is not None:
            ref_dict = {"E_MeV": ref_energy, "Phi": spec_values}
        else:
            # Use energy bins from column names
            ref_dict = {"E_MeV": np.arange(len(spec_values)), "Phi": spec_values}
        
        # Discretize to detector grid
        try:
            interp_df = detector.discretize_spectra(ref_dict)
            ref_on_grid = interp_df["Phi"].values
        except Exception as e:
            print(f"  [WARN] Could not discretize spectrum {spec_name}: {e}")
            continue
        
        # Get readings
        readings = detector.get_effective_readings_for_spectra(ref_dict)
        
        # Compute spectrum features for binning
        features = compute_spectrum_features(ref_on_grid, detector.E_MeV)
        
        print(f"[{spec_idx+1}/{len(spectrum_names)}] Spectrum: {spec_name[:40]}")
        
        for method_name, method_fn in methods_to_run.items():
            result, unfolded, status = unfold_one(detector, readings, method_name, method_fn)
            
            row = {
                "spectrum": spec_name,
                "detector": detector_name,
                "method": method_name,
                "status": status,
                **features,
            }
            
            if unfolded is not None:
                # Compute metrics
                metrics = compare_spectra(ref_on_grid, unfolded, energy=detector.E_MeV)
                row.update(metrics)
            
            results.append(row)
        
        if (spec_idx + 1) % 10 == 0:
            print(f"  ... processed {spec_idx + 1}/{len(spectrum_names)} spectra")
    
    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze benchmark results to find best methods per bin and overall."""
    
    ok_results = results_df[results_df["status"].str.startswith("OK")].copy()
    
    if len(ok_results) == 0:
        print("No successful results to analyze!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Overall ranking
    metric_cols = KEY_METRICS
    available_metrics = [m for m in metric_cols if m in ok_results.columns]
    
    # Normalize metrics (higher is better for all after transformation)
    normalized = ok_results.copy()
    for m in available_metrics:
        if m in ["wasserstein_dist", "mean_squared_error", "dose_difference_percent"]:
            # Lower is better - invert
            v = normalized[m]
            vmin, vmax = v.min(), v.max()
            if vmax > vmin:
                normalized[m] = (vmax - v) / (vmax - vmin)
            else:
                normalized[m] = 1.0
        elif m == "total_flux_ratio":
            # Closer to 1.0 is better
            deviation = (normalized[m] - 1.0).abs()
            max_dev = deviation.max()
            if max_dev > 0:
                normalized[m] = 1.0 - deviation / max_dev
            else:
                normalized[m] = 1.0
        else:
            # Higher is better - normalize
            v = normalized[m]
            vmin, vmax = v.min(), v.max()
            if vmax > vmin:
                normalized[m] = (v - vmin) / (vmax - vmin)
            else:
                normalized[m] = 1.0
    
    # Per-method scores
    method_scores = normalized.groupby("method")[available_metrics].mean()
    method_scores["overall_score"] = method_scores.mean(axis=1)
    method_scores = method_scores.sort_values("overall_score", ascending=False)
    
    # Per-bin analysis (using spectrum features)
    # Create bins based on hardness ratio
    ok_results["hardness_bin"] = pd.cut(
        ok_results["hardness_ratio"],
        bins=[-np.inf, 0.1, 0.3, 0.5, 1.0, np.inf],
        labels=["very_soft", "soft", "intermediate", "hard", "very_hard"]
    )
    
    # Best method per bin
    bin_rankings = {}
    for bin_name, bin_df in ok_results.groupby("hardness_bin"):
        if len(bin_df) > 0:
            bin_scores = bin_df.groupby("method")[available_metrics].mean()
            bin_scores["overall_score"] = bin_scores.mean(axis=1)
            bin_scores = bin_scores.sort_values("overall_score", ascending=False)
            bin_rankings[bin_name] = bin_scores
    
    bin_rankings_df = pd.concat(bin_rankings, names=["bin"])
    
    return method_scores, bin_rankings_df


def create_composite_method(results_df: pd.DataFrame) -> Dict[str, str]:
    """Create a composite method that selects the best method based on spectrum characteristics."""
    
    ok_results = results_df[results_df["status"].str.startswith("OK")].copy()
    
    if len(ok_results) == 0:
        return {}
    
    # Create bins based on hardness ratio
    ok_results["hardness_bin"] = pd.cut(
        ok_results["hardness_ratio"],
        bins=[-np.inf, 0.1, 0.3, 0.5, 1.0, np.inf],
        labels=["very_soft", "soft", "intermediate", "hard", "very_hard"]
    )
    
    # Find best method per bin
    best_methods = {}
    metric_cols = KEY_METRICS
    available_metrics = [m for m in metric_cols if m in ok_results.columns]
    
    for bin_name, bin_df in ok_results.groupby("hardness_bin"):
        if len(bin_df) > 0:
            # Score each method in this bin
            method_perf = bin_df.groupby("method")[available_metrics].agg(["mean", "std"]).reset_index()
            method_perf.columns = ["method"] + [f"{m}_{stat}" for m in available_metrics for stat in ["mean", "std"]]
            
            # Simple scoring: average of normalized metrics
            score_col = "score"
            method_perf[score_col] = 0
            for m in available_metrics:
                mean_col = f"{m}_mean"
                if m in ["wasserstein_dist", "mean_squared_error", "dose_difference_percent"]:
                    # Lower is better
                    method_perf[score_col] -= method_perf[mean_col]
                elif m == "total_flux_ratio":
                    # Closer to 1 is better
                    method_perf[score_col] -= (method_perf[mean_col] - 1.0).abs()
                else:
                    # Higher is better
                    method_perf[score_col] += method_perf[mean_col]
            
            best_method = method_perf.loc[method_perf[score_col].idxmax(), "method"]
            best_methods[bin_name] = best_method
    
    return best_methods


def main():
    """Main benchmark execution."""
    print("="*80)
    print("BSSUNFOLD COMPREHENSIVE BENCHMARK")
    print("="*80)
    
    # Load data
    print("\nLoading Monte Carlo spectra...")
    mc_spectra = load_monte_carlo_data()
    print(f"  Loaded {len(mc_spectra.columns) - 1} spectra")
    
    # Run benchmark for each detector
    all_results = []
    
    for det_name, det_fn in DETECTOR_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Detector: {det_name}")
        print(f"{'='*80}")
        
        detector = det_fn()
        print(f"  Energy bins: {detector.n_energy_bins}")
        print(f"  Detectors: {detector.n_detectors}")
        
        results = run_benchmark(
            mc_spectra,
            det_name,
            detector,
            max_spectra=20  # Limit for initial run
        )
        all_results.append(results)
        
        # Save intermediate results
        out_path = Path(__file__).parent / f"benchmark_results_{det_name}.csv"
        results.to_csv(out_path, index=False)
        print(f"\n  Results saved to: {out_path}")
    
    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_out = Path(__file__).parent / "benchmark_results_all.csv"
    combined_results.to_csv(combined_out, index=False)
    print(f"\nCombined results saved to: {combined_out}")
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    method_scores, bin_rankings = analyze_results(combined_results)
    
    print("\n--- OVERALL METHOD RANKING ---")
    print(method_scores.head(10).to_string())
    
    print("\n--- BEST METHODS PER SPECTRUM BIN ---")
    if len(bin_rankings) > 0:
        print(bin_rankings.to_string())
    
    # Create composite method mapping
    composite_methods = create_composite_method(combined_results)
    print("\n--- COMPOSITE METHOD RECOMMENDATIONS ---")
    for bin_name, method in composite_methods.items():
        print(f"  {bin_name}: {method}")
    
    # Save analysis
    analysis_out = Path(__file__).parent / "benchmark_analysis.csv"
    method_scores.reset_index().to_csv(analysis_out, index=False)
    print(f"\nAnalysis saved to: {analysis_out}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
