"""Bayesian iterative unfolding method (D'Agostini) for neutron spectrum reconstruction.

This module provides the solve_bayes solver and unfold_bayes wrapper
using PyUnfold's iterative_unfold implementation.

The response matrix is column-normalized (each column sums to 1)
before passing to pyunfold, because the D'Agostini algorithm treats
the response as a conditional probability matrix P(D_j | E_i).
After unfolding, the result is divided by the column sums to restore
physical units.
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

    Uses PyUnfold's iterative_unfold implementation. The response matrix is
    column-normalized to satisfy the probability normalization P(D_j|E_i)
    required by the D'Agostini algorithm, then the result is rescaled back
    to physical units.

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
        Convergence tolerance (default: 1e-3).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,) in physical units.
    """
    try:
        from pyunfold import iterative_unfold
        from pyunfold.callbacks import Logger
    except ImportError as e:
        raise ImportError(
            "pyunfold is required for unfold_bayes. "
            "Install with: pip install pyunfold"
        ) from e

    n_detectors, n_energy = A.shape

    # Column-normalize response matrix so each column sums to 1.
    # The D'Agostini algorithm requires P(D_j|E_i) as the response —
    # a conditional probability matrix.  The raw matrix A has arbitrary
    # column sums (physical units); we decompose A_ji = C_i * P_ji
    # where C_i = sum_j A_ji.
    column_sums = np.sum(A, axis=0)
    column_sums = np.where(column_sums > 0, column_sums, 1.0)
    P = A / column_sums  # P_ji = P(D_j | E_i), columns sum to 1

    # Prepare pyunfold inputs
    efficiencies = [1.0] * n_energy
    response_err = np.zeros_like(A)
    efficiencies_err = [0.05] * n_energy
    data_err = [0.05] * n_detectors

    # Normalize prior to sum to 1 (pyunfold expects a probability)
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
        callbacks=[Logger()],
        prior=prior,
        ts_stopping=tolerance,
    )

    # pyunfold returns y = effective counts where P @ y = b.
    # Convert to physical spectrum: x_i = y_i / C_i
    y = np.asarray(result["unfolded"], dtype=float)
    spectrum = y / column_sums

    return spectrum


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
