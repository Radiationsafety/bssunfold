"""STAY'SL unfolding method for neutron spectrum reconstruction.

This is an independent open-source reimplementation of the STAY'SL
algorithm following its published Bayesian least-squares formalism (Perey;
the 1981 review of unfolding codes). The original STAY'SL code is a
proprietary package. This
implementation is built solely from the published mathematical formulation
and does not use or reproduce any proprietary source code.

STAY'SL is a single-step (one-shot) linear Bayesian update. Given a prior
spectrum ``x0`` with prior covariance ``Cx`` and measurements ``b`` with
measurement covariance ``Cb``, the posterior mean spectrum is

    x = x0 + Cx A^T (Cb + A Cx A^T)^-1 (b - A x0)

This is the standard linear-Bayes (Kalman-like) estimate: the prior is
refined by the measurements using the full covariance information. By
default ``Cb`` and ``Cx`` are taken as diagonal, derived from relative
uncertainties, but explicit covariance matrices may be supplied.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_staysl", "unfold_staysl"]


def solve_staysl(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    relative_uncertainty: float = 0.1,
    prior_uncertainty: float = 1.0,
    Cb: Optional[np.ndarray] = None,
    Cx: Optional[np.ndarray] = None,
    regularization: float = 1e-12,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the STAY'SL Bayesian algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Prior spectrum guess (n,). Used as the Bayesian prior mean.
    relative_uncertainty : float, optional
        Relative measurement uncertainty used to build the diagonal
        measurement covariance ``Cb = diag((rel * b)^2)`` when ``Cb`` is not
        supplied (default: 0.1).
    prior_uncertainty : float, optional
        Relative prior uncertainty used to build the diagonal prior
        covariance ``Cx = diag((prior * x0)^2)`` when ``Cx`` is not supplied
        (default: 1.0, i.e. a broad prior).
    Cb : np.ndarray, optional
        Explicit measurement covariance matrix (m x m). Overrides
        ``relative_uncertainty`` when given.
    Cx : np.ndarray, optional
        Explicit prior covariance matrix (n x n). Overrides
        ``prior_uncertainty`` when given.
    regularization : float, optional
        Small Tikhonov term added to the bracket ``(Cb + A Cx A^T)`` for
        numerical stability (default: 1e-12).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, 1, True). STAY'SL is a single-step method, so
        ``iterations`` is 1 and ``converged`` is always True.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    if A.size == 0 or b.size == 0:
        raise ValueError("Response matrix and measurements must be non-empty")

    m, n = A.shape
    if Cb is None:
        b_safe = np.maximum(np.abs(b), 1e-12)
        Cb = np.diag((relative_uncertainty * b_safe) ** 2)
    else:
        Cb = np.asarray(Cb, dtype=float)
    if Cx is None:
        x_safe = np.maximum(np.abs(x0), 1e-12)
        Cx = np.diag((prior_uncertainty * x_safe) ** 2)
    else:
        Cx = np.asarray(Cx, dtype=float)

    # Bayesian linear update (posterior mean).
    bracket = Cb + A @ Cx @ A.T
    bracket = bracket + regularization * np.eye(m)
    gain = Cx @ A.T @ np.linalg.inv(bracket)
    spectrum = x0 + gain @ (b - A @ x0)

    return np.maximum(spectrum, 0.0), 1, True


def unfold_staysl(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    relative_uncertainty: float = 0.1,
    prior_uncertainty: float = 1.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the STAY'SL Bayesian algorithm.

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
        Prior spectrum guess. If None, a flat spectrum is used as the prior
        mean.
    relative_uncertainty : float, optional
        Relative measurement uncertainty for ``Cb`` (default: 0.1).
    prior_uncertainty : float, optional
        Relative prior uncertainty for ``Cx`` (default: 1.0).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
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
            solve_staysl,
            relative_uncertainty=relative_uncertainty,
            prior_uncertainty=prior_uncertainty,
        ),
        solve_kwargs={},
        method_name="STAY'SL",
        extra_output={
            "relative_uncertainty": float(relative_uncertainty),
            "prior_uncertainty": float(prior_uncertainty),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
