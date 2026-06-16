"""Landweber iteration unfolding method for neutron spectrum reconstruction.

This module provides the core solve_landweber solver and the unfold_landweber
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_landweber", "unfold_landweber"]


def solve_landweber(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Landweber iteration.

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
    import warnings
    x = x0.copy()
    sigma_max = np.linalg.norm(A, 2)

    if sigma_max == 0:
        warnings.warn("Response matrix has zero norm. Returning initial guess.")
        return x, 0, False

    step_size = 1.0 / (sigma_max ** 2)
    AT = A.T

    converged = False
    iterations = 0

    for i in range(max_iterations):
        residual = A @ x - b
        residual_norm = np.linalg.norm(residual)

        if residual_norm < tolerance:
            converged = True
            iterations = i
            break

        x = x - step_size * (AT @ residual)
        x = np.maximum(x, 0)

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


def unfold_landweber(
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
    """Unfold using Landweber iteration method.

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
            solve_landweber,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="Landweber",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
