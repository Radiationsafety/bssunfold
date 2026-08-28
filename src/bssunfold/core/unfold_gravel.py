"""GRAVEL unfolding method for neutron spectrum reconstruction.

This module provides the core solve_gravel solver and the unfold_gravel
wrapper for use with the Detector class.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import exp, log

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

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
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    _, N = A.shape
    x = x0.copy()

    valid = b > 0
    if np.sum(valid) == 0:
        raise ValueError("All measurements are zero or negative")

    A_valid = A[valid]
    b_valid = b[valid]

    try:
        from ._numba_jit import NUMBA_AVAILABLE, _gravel_inner

        if NUMBA_AVAILABLE:
            return _gravel_inner(
                A_valid, x, b_valid, regularization, max_iterations, tolerance
            )
    except ImportError:
        pass

    # Fallback: vectorized pure numpy implementation
    J_prev = 0.0
    dJ_prev = 1.0
    eps = 1e-10

    for iteration in range(1, max_iterations + 1):
        computed = A_valid @ x
        computed_safe = np.maximum(computed, eps)

        # Vectorized W computation: W[i,j] = b[i] * A[i,j] * x[j] / computed[i]
        # Only where computed > 0 and x > 0
        x_safe = np.maximum(x, 0.0)
        W = b_valid[:, None] * A_valid * x_safe[None, :] / computed_safe[:, None]
        # Zero out where conditions aren't met
        valid_mask = (computed[:, None] > 0) & (x_safe[None, :] > 0)
        W *= valid_mask

        # Vectorized log_ratio per row
        log_ratio = log(b_valid / computed_safe)
        # Zero out where conditions aren't met
        weight_mask = (b_valid > 0) & (computed_safe > 0) & (computed > 0)
        log_ratio *= weight_mask

        # Per-column sums
        numerator = (W * log_ratio[:, None]).sum(axis=0)
        denominator = W.sum(axis=0)

        # Update x where denominator > 0
        update_mask = denominator > 0
        if np.any(update_mask):
            reg_term = regularization * log(x[update_mask] + eps)
            update = exp((numerator[update_mask] - reg_term) / denominator[update_mask])
            x[update_mask] *= update

        computed_final = A_valid @ x
        chi_sq = np.sum((computed_final - b_valid) ** 2 / np.maximum(b_valid, eps))
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
    x0_default = np.ones(n_energy_bins) / 2

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
