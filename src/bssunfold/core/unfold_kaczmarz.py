"""Kaczmarz (ART) unfolding method for neutron spectrum reconstruction.

This module provides the core solve_kaczmarz solver and the unfold_kaczmarz
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_kaczmarz", "unfold_kaczmarz"]


def solve_kaczmarz(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    omega: float = 1.0,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Kaczmarz algorithm (ART).

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
    omega : float, optional
        Relaxation parameter (0 < omega <= 2), default: 1.0.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    import warnings
    m, n = A.shape
    x = x0.copy()

    if omega <= 0 or omega > 2:
        warnings.warn(f"omega={omega} outside recommended range (0,2]")

    row_norms_sq = np.sum(A * A, axis=1)

    converged = False
    iterations = 0
    x_old = x.copy()

    for k in range(max_iterations):
        i = k % m
        if row_norms_sq[i] > 0:
            update = (b[i] - np.dot(A[i], x)) / row_norms_sq[i]
            x = x + omega * update * A[i]
            x = np.maximum(x, 0)

        if (k + 1) % m == 0:
            if np.linalg.norm(x - x_old) < tolerance:
                converged = True
                iterations = k + 1
                break
            x_old = x.copy()

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


def unfold_kaczmarz(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    omega: float = 1.0,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the Kaczmarz algorithm (ART).

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
        Initial spectrum guess. If None, zero spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations, default: 1000.
    omega : float, optional
        Relaxation parameter (0 < omega <= 2), default: 1.0.
    tolerance : float, optional
        Convergence tolerance for solution change, default: 1e-6.
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
    x0_default = np.zeros(n_energy_bins)

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
            solve_kaczmarz,
            max_iterations=max_iterations,
            omega=omega,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="Kaczmarz",
        extra_output={
            "tolerance": tolerance,
            "omega": omega,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
