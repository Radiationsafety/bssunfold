"""Composite multi-method spectrum unfolding.

This module implements an ensemble/composite approach to neutron spectrum
unfolding that combines multiple methods based on spectrum characteristics.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# ── Default method configurations for different spectrum types ─────────

# Best methods identified from benchmark for each spectrum hardness bin
DEFAULT_BIN_METHODS = {
    # Very soft spectra (thermal dominant) - use regularization-heavy methods
    "very_soft": ["tsvd", "bayes", "cvxpy", "statreg", "lanczos"],
    # Soft spectra - balanced iterative + regularized
    "soft": ["mlem", "landweber", "bayes_spline", "gravel", "qpsolvers"],
    # Intermediate - hybrid approaches work well
    "intermediate": ["cvxpy", "qpsolvers", "hybrid_parametric", "parametric2", "hybrid_gmres"],
    # Hard spectra - optimization-based methods
    "hard": ["genetic", "interpret", "maeo_ensemble", "mystic", "cs"],
    # Very hard (fast neutron dominant) - constraint programming
    "very_hard": ["scip", "docplex", "epic", "cs", "interpret"],
}

# Ensemble weights for combining results - based on overall benchmark performance
DEFAULT_ENSEMBLE_WEIGHTS = {
    "bayes": 0.15,
    "mlem": 0.12,
    "landweber": 0.10,
    "cvxpy": 0.13,
    "qpsolvers": 0.10,
    "tsvd": 0.10,
    "statreg": 0.08,
    "genetic": 0.07,
    "parametric2": 0.07,
    "hybrid_parametric": 0.05,
    "interpret": 0.03,
}


def compute_spectrum_features(
    spectrum: np.ndarray, 
    energy: np.ndarray
) -> Dict[str, float]:
    """Compute features of a spectrum for method selection.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Fluence spectrum values.
    energy : np.ndarray
        Energy grid in MeV.
    
    Returns
    -------
    dict
        Dictionary of spectrum features.
    """
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
    
    # Smoothness indicator
    if len(s) > 1:
        smoothness = 1.0 / (1.0 + np.std(np.diff(np.log(s_safe))))
    else:
        smoothness = 1.0
    
    return {
        "mean_energy": float(mean_e),
        "std_energy": float(std_e),
        "peak_energy": float(peak_energy),
        "thermal_fraction": float(thermal_fraction),
        "fast_fraction": float(fast_fraction),
        "intermediate_fraction": float(intermediate_fraction),
        "hardness_ratio": float(hardness),
        "entropy": float(entropy),
        "smoothness": float(smoothness),
    }


def classify_spectrum_by_hardness(features: Dict[str, float]) -> str:
    """Classify spectrum into hardness bin based on features.
    
    Parameters
    ----------
    features : dict
        Spectrum features from compute_spectrum_features.
    
    Returns
    -------
    str
        Hardness bin name.
    """
    hardness = features.get("hardness_ratio", 0.0)
    
    if hardness < 0.1:
        return "very_soft"
    elif hardness < 0.3:
        return "soft"
    elif hardness < 0.5:
        return "intermediate"
    elif hardness < 1.0:
        return "hard"
    else:
        return "very_hard"


def select_methods_for_spectrum(
    features: Dict[str, float],
    bin_methods: Optional[Dict[str, List[str]]] = None,
    n_methods: int = 5
) -> List[str]:
    """Select best methods for a given spectrum based on its features.
    
    Parameters
    ----------
    features : dict
        Spectrum features.
    bin_methods : dict, optional
        Custom mapping of bins to method lists.
    n_methods : int
        Number of methods to select.
    
    Returns
    -------
    list
        List of method names to use.
    """
    if bin_methods is None:
        bin_methods = DEFAULT_BIN_METHODS
    
    bin_name = classify_spectrum_by_hardness(features)
    methods = bin_methods.get(bin_name, list(DEFAULT_ENSEMBLE_WEIGHTS.keys()))
    
    return methods[:n_methods]


def run_method_ensemble(
    detector: Any,
    readings: Dict[str, float],
    methods: List[str],
    weights: Optional[Dict[str, float]] = None,
    timeout_per_method: float = 30.0
) -> Dict[str, Any]:
    """Run ensemble of unfolding methods and combine results.
    
    Parameters
    ----------
    detector : Detector
        Detector instance.
    readings : dict
        Count rate readings.
    methods : list
        List of method names to run.
    weights : dict, optional
        Weights for combining methods. If None, equal weights are used.
    timeout_per_method : float
        Timeout per method in seconds.
    
    Returns
    -------
    dict
        Combined result with spectrum and metadata.
    """
    import signal
    from bssunfold.utils.comparison import cosine_similarity
    
    class MethodTimeout(Exception):
        pass
    
    def handler(signum, frame):
        raise MethodTimeout("Method timed out")
    
    successful_results = []
    successful_weights = []
    
    if weights is None:
        weights = {m: 1.0 / len(methods) for m in methods}
    
    for method_name in methods:
        try:
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(timeout_per_method))
            
            method_fn = getattr(detector, f"unfold_{method_name}", None)
            if method_fn is None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                continue
            
            result = method_fn(readings, save_result=False)
            
            # Cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            if result and "spectrum" in result:
                spectrum = result["spectrum"]
                if not np.any(np.isnan(spectrum)) and np.sum(spectrum) > 0:
                    successful_results.append(spectrum.copy())
                    w = weights.get(method_name, 1.0 / len(methods))
                    successful_weights.append(w)
                    
        except (MethodTimeout, Exception):
            # Skip failed methods
            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)
            continue
    
    if len(successful_results) == 0:
        return {
            "status": "ERROR",
            "message": "No methods succeeded",
            "spectrum": None,
        }
    
    # Normalize weights
    total_weight = sum(successful_weights)
    normalized_weights = [w / total_weight for w in successful_weights]
    
    # Weighted average of spectra
    combined_spectrum = np.zeros_like(successful_results[0])
    for spectrum, weight in zip(successful_results, normalized_weights):
        combined_spectrum += weight * spectrum
    
    # Compute consistency metric (variance between methods)
    if len(successful_results) > 1:
        stacked = np.stack(successful_results)
        variance = np.var(stacked, axis=0)
        consistency = 1.0 / (1.0 + np.mean(variance))
    else:
        consistency = 1.0
    
    return {
        "status": "OK",
        "spectrum": combined_spectrum,
        "methods_used": methods,
        "successful_methods": [methods[i] for i in range(len(successful_results))],
        "weights": normalized_weights,
        "consistency": consistency,
        "n_methods": len(successful_results),
    }


def unfold_composite(
    detector: Any,
    readings: Dict[str, float],
    initial_guess: Optional[np.ndarray] = None,
    bin_methods: Optional[Dict[str, List[str]]] = None,
    n_methods: int = 5,
    weights: Optional[Dict[str, float]] = None,
    timeout_per_method: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """Composite unfolding using adaptive method selection.
    
    This function:
    1. Uses an initial guess (or default) to estimate spectrum characteristics
    2. Selects the best methods for that spectrum type
    3. Runs an ensemble of those methods
    4. Combines results with weighted averaging
    
    Parameters
    ----------
    detector : Detector
        Detector instance.
    readings : dict
        Count rate readings.
    initial_guess : np.ndarray, optional
        Initial spectrum guess for feature estimation.
    bin_methods : dict, optional
        Custom bin-to-methods mapping.
    n_methods : int
        Number of methods to use in ensemble.
    weights : dict, optional
        Method weights for combination.
    timeout_per_method : float
        Timeout per method in seconds.
    **kwargs
        Additional arguments passed to run_method_ensemble.
    
    Returns
    -------
    dict
        Unfolding result with spectrum and metadata.
    """
    # Get initial guess or use flat spectrum
    if initial_guess is None:
        initial_guess = np.ones(detector.n_energy_bins) / detector.n_energy_bins
    
    # Compute features of initial guess
    features = compute_spectrum_features(initial_guess, detector.E_MeV)
    
    # Select methods based on spectrum type
    selected_methods = select_methods_for_spectrum(
        features, 
        bin_methods=bin_methods,
        n_methods=n_methods
    )
    
    # Run ensemble
    result = run_method_ensemble(
        detector=detector,
        readings=readings,
        methods=selected_methods,
        weights=weights,
        timeout_per_method=timeout_per_method,
        **kwargs
    )
    
    # Add metadata
    result["spectrum_type"] = classify_spectrum_by_hardness(features)
    result["spectrum_features"] = features
    
    return result


def unfold_composite_iterative(
    detector: Any,
    readings: Dict[str, float],
    max_iterations: int = 3,
    bin_methods: Optional[Dict[str, List[str]]] = None,
    n_methods: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """Iterative composite unfolding with method refinement.
    
    This function iteratively refines the method selection:
    1. Start with default methods
    2. Run ensemble
    3. Analyze result features
    4. Adjust method selection if needed
    5. Repeat
    
    Parameters
    ----------
    detector : Detector
        Detector instance.
    readings : dict
        Count rate readings.
    max_iterations : int
        Maximum number of refinement iterations.
    bin_methods : dict, optional
        Custom bin-to-methods mapping.
    n_methods : int
        Number of methods to use.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    dict
        Final unfolding result.
    """
    current_spectrum = None
    
    for iteration in range(max_iterations):
        # Use current spectrum as initial guess
        initial_guess = current_guess if current_spectrum is not None else None
        
        # Run composite unfolding
        result = unfold_composite(
            detector=detector,
            readings=readings,
            initial_guess=initial_guess,
            bin_methods=bin_methods,
            n_methods=n_methods,
            **kwargs
        )
        
        if result.get("spectrum") is None:
            break
        
        current_spectrum = result["spectrum"]
        
        # Check convergence
        if iteration > 0 and result.get("consistency", 0) > 0.95:
            break
    
    return result


# Export functions
__all__ = [
    "compute_spectrum_features",
    "classify_spectrum_by_hardness",
    "select_methods_for_spectrum",
    "run_method_ensemble",
    "unfold_composite",
    "unfold_composite_iterative",
    "DEFAULT_BIN_METHODS",
    "DEFAULT_ENSEMBLE_WEIGHTS",
]
