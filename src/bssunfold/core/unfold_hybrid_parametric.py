"""Hybrid parametric-nonparametric unfolding method.

This module implements a hybrid unfolding method that combines:
1. Parametric model for initial spectrum estimation
2. Iterative refinement using nonparametric methods (Landweber/MLEM)

The parametric model provides a physically motivated initial guess,
which is then refined using iterative methods to better fit the data.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = ["solve_hybrid_parametric", "unfold_hybrid_parametric"]


def _parametric_initial_guess(
    E: np.ndarray,
    readings: Dict[str, float],
    detector_names: List[str],
    sensitivities: Dict[str, np.ndarray],
) -> np.ndarray:
    """Generate initial spectrum guess from parametric model.

    Uses a simple heuristic to estimate spectral parameters from readings.
    """
    E = np.asarray(E, dtype=float)
    n_bins = len(E)
    spectrum = np.zeros(n_bins)

    total_counts = sum(readings.values())
    if total_counts < 1e-15:
        return np.ones(n_bins) / n_bins

    thermal = E < 0.4e-6
    epithermal = (E >= 0.4e-6) & (E < 0.1)
    fast = E >= 0.1

    n_thermal = np.sum(thermal)
    n_epithermal = np.sum(epithermal)
    n_fast = np.sum(fast)

    if n_thermal > 0:
        spectrum[thermal] = total_counts * 0.3 / n_thermal
    if n_epithermal > 0:
        spectrum[epithermal] = total_counts * 0.4 / n_epithermal
    if n_fast > 0:
        spectrum[fast] = total_counts * 0.3 / n_fast

    return spectrum


def _landweber_iteration(
    spectrum: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    step_size: float,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, int]:
    """Single Landweber iteration refinement."""
    x = spectrum.copy()
    for i in range(max_iter):
        residual = b - A @ x
        gradient = A.T @ residual
        x_new = x + step_size * gradient
        x_new = np.maximum(x_new, 0)
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new, i + 1
        x = x_new
    return x, max_iter


def _mlem_iteration(
    spectrum: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, int]:
    """Single MLEM iteration refinement."""
    x = spectrum.copy()
    x = np.maximum(x, 1e-15)
    for i in range(max_iter):
        computed = A @ x
        computed = np.maximum(computed, 1e-15)
        ratio = b / computed
        correction = A.T @ ratio
        x_new = x * correction
        x_new = np.maximum(x_new, 0)
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-15) < tolerance:
            return x_new, i + 1
        x = x_new
    return x, max_iter


def solve_hybrid_parametric(
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    refinement_method: str = "landweber",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    step_size: float = 0.01,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using hybrid parametric-nonparametric method.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (n_detectors x n_energy).
    b : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps.
    refinement_method : str, optional
        Refinement method: "landweber" or "mlem" (default: "landweber").
    max_iterations : int, optional
        Maximum iterations for refinement (default: 100).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    step_size : float, optional
        Step size for Landweber (default: 0.01).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    n_energy = A.shape[1]
    n_det = A.shape[0]

    parametric_guess = np.ones(n_energy) * np.mean(b) / np.mean(A.sum(axis=1))

    if refinement_method == "landweber":
        refined, n_iter = _landweber_iteration(
            parametric_guess, A, b, step_size, max_iterations, tolerance
        )
        success = n_iter < max_iterations
        message = f"Converged in {n_iter} iterations" if success else "Max iterations reached"
    elif refinement_method == "mlem":
        refined, n_iter = _mlem_iteration(
            parametric_guess, A, b, max_iterations, tolerance
        )
        success = n_iter < max_iterations
        message = f"Converged in {n_iter} iterations" if success else "Max iterations reached"
    else:
        raise ValueError(f"Unknown refinement method: {refinement_method}")

    return refined, success, message, n_iter


def unfold_hybrid_parametric(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    refinement_method: str = "landweber",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    step_size: float = 0.01,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using hybrid parametric-nonparametric method.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess.
    refinement_method : str, optional
        Refinement method: "landweber" or "mlem" (default: "landweber").
    max_iterations : int, optional
        Maximum iterations (default: 100).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    step_size : float, optional
        Step size for Landweber (default: 0.01).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: True).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(E_MeV + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if n_energy_bins > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if n_energy_bins > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0

    def solve_wrapper(A_mat, b_vec, **kwargs):
        x_opt, success, message, nfev = solve_hybrid_parametric(
            A_mat, b_vec, E_MeV, log_steps, refinement_method,
            max_iterations, tolerance, step_size
        )
        return x_opt, nfev, success

    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

    result = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name=f"hybrid_parametric ({refinement_method})",
        extra_output={
            "refinement_method": refinement_method,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "step_size": step_size,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
