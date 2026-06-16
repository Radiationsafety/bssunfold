"""Doroshenko coordinate update unfolding method.

This module provides the core solve_doroshenko solver and the unfold_doroshenko
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_doroshenko", "unfold_doroshenko"]


def solve_doroshenko(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization: float = 0.0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Doroshenko coordinate update method.

    Uses incremental residual update for O(n) per-coordinate complexity
    instead of O(n^2) from full matrix-vector products.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    regularization : float, optional
        Regularization strength to prevent division by zero (default: 0.0).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    x = x0.copy()

    denominator_cache = np.sum(A * A, axis=0) + regularization
    residual = b - A @ x

    converged = False
    iterations = 0

    for i in range(max_iterations):
        x_old = x.copy()

        for j in range(x.size):
            if denominator_cache[j] <= 0:
                continue
            Aj = A[:, j]
            old_xj = x[j]
            numerator = np.dot(Aj, residual) + denominator_cache[j] * old_xj
            new_xj = max(0.0, numerator / denominator_cache[j])
            delta = new_xj - old_xj
            if delta != 0:
                residual -= delta * Aj
                x[j] = new_xj

        if np.linalg.norm(x - x_old) < tolerance:
            converged = True
            iterations = i + 1
            break

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


def unfold_doroshenko(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization: float = 0.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the Doroshenko coordinate update method.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess. If None, uniform spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations, default: 1000.
    tolerance : float, optional
        Convergence tolerance for solution change, default: 1e-6.
    regularization : float, optional
        Regularization strength to prevent division by zero, default: 0.0.
    calculate_errors : bool, optional
        Flag to calculate uncertainty via Monte-Carlo, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples for error estimation, default: 100.
    save_result : bool, optional
        If True, save result to internal history, default: True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing unfolding results.
    """
    x0_default = np.ones(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_doroshenko,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        ),
        solve_kwargs={},
        method_name="Doroshenko",
        extra_output={
            "tolerance": tolerance,
            "regularization": regularization,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
