"""GRAVEL unfolding method for neutron spectrum reconstruction.

This module provides the core solve_gravel solver and the unfold_gravel
wrapper for use with the Detector class.
"""

import numpy as np
from numpy import log, exp
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_gravel", "unfold_gravel"]


def solve_gravel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    regularization: float = 0.0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the GRAVEL algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    tolerance : float, optional
        Convergence tolerance (default: 1e-8).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    regularization : float, optional
        Regularization parameter (default: 0.0).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    M, N = A.shape
    x = x0.copy().astype(np.float64)
    b = b.astype(np.float64)

    valid = b > 0
    if np.sum(valid) == 0:
        raise ValueError("All measurements are zero or negative")

    A_valid = A[valid]
    b_valid = b[valid]

    J_prev = 0.0
    dJ_prev = 1.0

    for iteration in range(1, max_iterations + 1):
        computed = A_valid @ x

        W = np.zeros((len(b_valid), N))
        for j in range(N):
            for i in range(len(b_valid)):
                if computed[i] > 0 and x[j] > 0:
                    W[i, j] = b_valid[i] * A_valid[i, j] * x[j] / computed[i]

            numerator = 0.0
            denominator = 0.0
            for i in range(len(b_valid)):
                if (
                    computed[i] > 0
                    and b_valid[i] > 0
                    and W[i, j] > 0
                    and A_valid[i, j] > 0
                ):
                    log_ratio = log(b_valid[i] / computed[i])
                    numerator += W[i, j] * log_ratio
                    denominator += W[i, j]

            if denominator > 0:
                reg_term = regularization * log(x[j] + 1e-10)
                update = exp((numerator - reg_term) / denominator)
                x[j] *= update

        computed_final = A_valid @ x
        chi_sq = np.sum(
            (computed_final - b_valid) ** 2 / np.maximum(b_valid, 1e-10)
        )
        J = chi_sq / np.sum(computed_final)
        dJ = J_prev - J
        ddJ = abs(dJ - dJ_prev)

        if ddJ <= tolerance:
            return x, iteration, True

        J_prev = J
        dJ_prev = dJ

    return x, max_iterations, False


def unfold_gravel(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    regularization: float = 0.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the GRAVEL algorithm.

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
    tolerance : float, optional
        Convergence tolerance (default: 1e-8).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    regularization : float, optional
        Regularization parameter (default: 0.0).
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
    x0_default = np.ones(n_energy_bins)/2

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
            solve_gravel,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
        ),
        solve_kwargs={},
        method_name="GRAVEL",
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
