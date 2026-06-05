"""Bayesian iterative unfolding with spline regularization.

This module provides the solve_bayes_spline solver and unfold_bayes_spline_regularization
wrapper using PyUnfold with SplineRegularizer.
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

    Uses PyUnfold's iterative_unfold with SplineRegularizer callback.

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
        Convergence tolerance (default: 1e-3).
    spline_degree : int, optional
        Spline degree (default: 3).
    spline_smooth : float, optional
        Spline smoothing parameter (default: 1e-2).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    try:
        from pyunfold import iterative_unfold
        from pyunfold.callbacks import Logger
        from pyunfold.callbacks import SplineRegularizer
    except ImportError as e:
        raise ImportError(
            "pyunfold is required for unfold_bayes_spline_regularization. "
            "Install with: pip install pyunfold"
        ) from e

    n_detectors, n_energy = A.shape

    # Column-normalize response so each column sums to 1.
    # The D'Agostini algorithm requires P(D_j|E_i) as the response.
    column_sums = np.sum(A, axis=0)
    column_sums = np.where(column_sums > 0, column_sums, 1.0)
    P = A / column_sums

    efficiencies = [1.0] * n_energy
    response_err = np.zeros_like(A)
    efficiencies_err = [0.05] * n_energy
    data_err = [0.05] * n_detectors

    spline_reg = SplineRegularizer(degree=spline_degree, smooth=spline_smooth)

    if x0 is not None:
        x0_sum = np.sum(x0)
        prior = x0 / x0_sum if x0_sum > 0 else None
    else:
        prior = None

    result = iterative_unfold(
        data=b,
        data_err=data_err,
        response=P,
        response_err=response_err,
        efficiencies=efficiencies,
        efficiencies_err=efficiencies_err,
        max_iter=max_iterations,
        callbacks=[Logger(), spline_reg],
        prior=prior,
        ts_stopping=tolerance,
    )

    # Convert from effective counts y back to physical units
    y = np.asarray(result["unfolded"], dtype=float)
    spectrum = y / column_sums

    return spectrum


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
