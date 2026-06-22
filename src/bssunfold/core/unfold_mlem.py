"""MLEM (Maximum Likelihood Expectation Maximization) unfolding method.

This module provides the core solve_mlem solver and the unfold_mlem
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_mlem", "unfold_mlem"]


def solve_mlem(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using MLEM iteration.

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

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    x = np.maximum(x0.copy(), 1e-10)
    AT = A.T

    try:
        from ._numba_jit import _mlem_inner, NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            return _mlem_inner(AT, A, x, b, max_iterations, tolerance)
    except ImportError:
        pass

    # Fallback: pure Python implementation
    converged = False
    iterations = 0

    for i in range(max_iterations):
        Ax = A @ x
        Ax = np.maximum(Ax, 1e-10)
        ratio = b / Ax
        correction = AT @ ratio
        x_new = x * correction
        diff = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        x = np.maximum(x_new, 0)
        iterations = i + 1
        if diff < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_mlem(
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
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using MLEM algorithm.

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
        Initial spectrum guess.
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
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
    x0_default = np.ones(n_energy_bins) * 0.5

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
            solve_mlem,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="MLEM",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
