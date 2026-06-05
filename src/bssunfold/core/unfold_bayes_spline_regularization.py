"""Bayesian iterative unfolding with spline regularization.

Implements the D'Agostini iterative Bayesian unfolding with
UnivariateSpline smoothing applied to the physical spectrum rather
than the effective-counts space. This prevents boundary artifacts
in low-sensitivity bins from being amplified by the column-sum
rescaling step.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_bayes_spline", "unfold_bayes_spline_regularization"]


def solve_bayes_spline(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 4000,
    tolerance: float = 1e-3,
    spline_degree: int = 3,
    spline_smooth: float = 1e-2,
) -> np.ndarray:
    """Solve unfolding problem using Bayes with spline regularization.

    Implements the D'Agostini iterative Bayesian unfolding from scratch.
    The response matrix is column-normalised (each column sums to 1)
    so the algorithm works in effective-count space, but the
    UnivariateSpline smoother is applied to the physical spectrum to
    avoid boundary artifacts that appear when rescaling low-sensitivity
    bins back to physical units.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Prior spectrum.
    max_iterations : int, optional
        Maximum iterations (default: 4000).
    tolerance : float, optional
        Relative L2 convergence tolerance (default: 1e-3).
    spline_degree : int, optional
        Spline degree (default: 3).
    spline_smooth : float, optional
        Spline smoothing parameter (default: 1e-2).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    from scipy.interpolate import UnivariateSpline

    n_detectors, n_energy = A.shape

    # Column-normalise response so each column sums to 1.
    column_sums = np.sum(A, axis=0)
    column_sums_safe = np.where(column_sums > 0, column_sums, 1.0)
    P = A / column_sums_safe  # P(D_j | E_i), columns sum to 1

    # Bins where no detector has any sensitivity remain at the prior.
    zero_sens = column_sums <= 0

    # Prior in effective-count space
    if x0 is not None and np.sum(x0) > 0:
        prior = x0 / np.sum(x0)
    else:
        prior = np.ones(n_energy) / n_energy

    total_counts = np.sum(b)
    y = total_counts * prior  # initial effective counts

    x_indices = np.arange(n_energy, dtype=float)

    for iteration in range(max_iterations):
        y_old = y.copy()

        # Fold the current estimate through the response
        f_norm = P @ y  # expected data vector
        f_norm_safe = np.where(f_norm > 0, f_norm, 1e-300)

        # D'Agostini update in effective-count space:
        #   y_i^(new) = y_i^(old) * sum_j b_j * P_ji / (P @ y)_j
        weight = b[:, np.newaxis] * P / f_norm_safe[:, np.newaxis]
        y = y_old * np.sum(weight, axis=0)

        # Zero-sensitivity bins cannot be constrained by data;
        # keep them at the prior level.
        if np.any(zero_sens):
            y[zero_sens] = prior[zero_sens] * total_counts

        # Convert to physical spectrum
        x = y / column_sums_safe

        # Spline in log10-space to handle the large dynamic range of
        # physical spectra and to prevent edge blow-up: bins with tiny
        # column sums are naturally constrained by their neighbours
        # through spline continuity in log space.
        if n_energy > spline_degree + 1:
            log_x = np.log10(np.maximum(x, 1e-300))
            spline = UnivariateSpline(
                x_indices, log_x, k=spline_degree, s=spline_smooth
            )
            log_x_smooth = spline(x_indices)
            x_smooth = 10.0 ** log_x_smooth
        else:
            x_smooth = x

        x_smooth = np.maximum(x_smooth, 0)

        # Convert back to effective-count space for the next iteration
        y = x_smooth * column_sums_safe

        # Convergence check (in y-space)
        denom = max(1.0, np.linalg.norm(y_old))
        if np.linalg.norm(y - y_old) / denom < tolerance:
            break

    return x_smooth


def unfold_bayes_spline_regularization(
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
    spline_degree: int = 3,
    spline_smooth: float = 1e-2,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Bayes with spline regularization.

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
    spline_degree : int, optional
        Spline degree (default: 3).
    spline_smooth : float, optional
        Spline smoothing parameter (default: 1e-2).
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
            solve_bayes_spline,
            max_iterations=max_iterations,
            tolerance=tolerance,
            spline_degree=spline_degree,
            spline_smooth=spline_smooth,
        ),
        solve_kwargs={},
        method_name="Bayes_Spline",
        extra_output={
            "spline_degree": spline_degree,
            "spline_smooth": spline_smooth,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
