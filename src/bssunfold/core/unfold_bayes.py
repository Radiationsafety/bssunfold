"""Bayesian iterative unfolding method (D'Agostini) for neutron spectrum reconstruction.

Implements the D'Agostini iterative Bayesian unfolding from scratch.
The response matrix is column-normalised so the algorithm works in
effective-count space, then the result is rescaled to physical units.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_bayes", "unfold_bayes"]


def solve_bayes(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 4000,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Solve unfolding problem using Bayesian iterative unfolding (D'Agostini).

    Pure numpy implementation of the D'Agostini algorithm.  The response
    matrix is column-normalised so each column sums to 1 (conditional
    probability P(D_j | E_i)), then the result is rescaled to physical
    units via division by the column sums.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Prior spectrum. If None, uniform prior is used.
    max_iterations : int, optional
        Maximum iterations (default: 4000).
    tolerance : float, optional
        Relative L2 convergence tolerance (default: 1e-3).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,) in physical units.
    """
    n_detectors, n_energy = A.shape

    # Column-normalise response so each column sums to 1.
    column_sums = np.sum(A, axis=0)
    column_sums_safe = np.where(column_sums > 0, column_sums, 1.0)
    P = A / column_sums_safe  # P(D_j | E_i), columns sum to 1

    # Bins with zero sensitivity cannot be constrained by data.
    zero_sens = column_sums <= 0

    # Prior in effective-count space
    if x0 is not None and np.sum(x0) > 0:
        prior = x0 / np.sum(x0)
    else:
        prior = np.ones(n_energy) / n_energy

    total_counts = np.sum(b)
    y = total_counts * prior  # initial effective counts

    for iteration in range(max_iterations):
        y_old = y.copy()

        # Forward fold
        f_norm = P @ y
        f_norm_safe = np.where(f_norm > 0, f_norm, 1e-300)

        # D'Agostini update in effective-count space:
        #   y_i^(new) = y_i^(old) * sum_j b_j * P_ji / (P @ y)_j
        weight = b[:, np.newaxis] * P / f_norm_safe[:, np.newaxis]
        y = y_old * np.sum(weight, axis=0)

        # Zero-sensitivity bins stay at the prior.
        if np.any(zero_sens):
            y[zero_sens] = prior[zero_sens] * total_counts

        # Convert to physical spectrum for convergence check
        x = y / column_sums_safe

        # Convergence check (in y-space for consistency)
        denom = max(1.0, np.linalg.norm(y_old))
        if np.linalg.norm(y - y_old) / denom < tolerance:
            break

    return x


def unfold_bayes(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 4000,
    tolerance: float = 1e-3,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Bayesian iterative unfolding.

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
        Prior spectrum.
    max_iterations : int, optional
        Maximum iterations (default: 4000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-3).
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
            solve_bayes,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="Bayes_D'Agostini",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
