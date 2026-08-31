"""Ensemble Kalman Inversion (EKI) for neutron spectrum reconstruction.

Implements the Ensemble Kalman Inversion method of Iglesias et al. (2013)
for Bayesian posterior approximation without MCMC.  The ensemble of particles
is propagated through the forward model and updated via the Kalman gain
equation with optional regularization for stability.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_eki", "unfold_eki"]


def solve_eki(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    n_ensemble: int = 50,
    n_iterations: int = 50,
    regularization: float = 1e-4,
    inflation: float = 1.02,
    noise_std: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Ensemble Kalman Inversion.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,). Used as the centre of the initial ensemble.
    n_ensemble : int, optional
        Number of ensemble members (default: 50).
    n_iterations : int, optional
        Number of EKI iterations (default: 50).
    regularization : float, optional
        Tikhonov-style regularization added to the covariance diagonal
        for numerical stability (default: 1e-4).
    inflation : float, optional
        Covariance inflation factor applied after each update step
        to prevent ensemble collapse (default: 1.02).
    noise_std : float, optional
        Standard deviation of measurement noise.  If None, estimated as
        5 % of ``||b|| / sqrt(m)``.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (mean_spectrum, n_iterations, True).
    """
    A, b, x0 = validate_system(A, b, x0=x0)
    m, n = A.shape

    rng = np.random.RandomState(random_state)

    if noise_std is None:
        noise_std = 0.05 * np.linalg.norm(b) / np.sqrt(m) if m > 0 else 1e-6
    noise_var = noise_std ** 2

    sigma_prior = np.abs(x0) + 1e-6
    ensemble = rng.normal(loc=x0[:, None], scale=sigma_prior[:, None],
                          size=(n, n_ensemble))

    for iteration in range(n_iterations):
        predictions = A @ ensemble
        pred_mean = np.mean(predictions, axis=1, keepdims=True)
        state_mean = np.mean(ensemble, axis=1, keepdims=True)

        pred_pert = predictions - pred_mean
        state_pert = ensemble - state_mean

        C_dd = (pred_pert @ pred_pert.T) / max(n_ensemble - 1, 1)
        C_dd += (noise_var + regularization) * np.eye(m)

        C_md = (state_pert @ pred_pert.T) / max(n_ensemble - 1, 1)

        try:
            C_d_inv = np.linalg.solve(
                C_dd, np.eye(m)
            )
        except np.linalg.LinAlgError:
            C_d_inv = np.linalg.pinv(C_dd)

        innovation = b[:, None] + noise_std * rng.randn(m, n_ensemble) - predictions
        ensemble = ensemble + C_md @ C_d_inv @ innovation

        ensemble *= inflation
        np.maximum(ensemble, 0, out=ensemble)

    mean_spectrum = np.mean(ensemble, axis=1)
    np.maximum(mean_spectrum, 0, out=mean_spectrum)

    return mean_spectrum, n_iterations, True


def unfold_eki(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    n_ensemble: int = 50,
    n_iterations: int = 50,
    regularization: float = 1e-4,
    inflation: float = 1.02,
    noise_std: Optional[float] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Ensemble Kalman Inversion.

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
    n_ensemble : int, optional
        Number of ensemble members (default: 50).
    n_iterations : int, optional
        Number of EKI iterations (default: 50).
    regularization : float, optional
        Regularization for covariance stability (default: 1e-4).
    inflation : float, optional
        Covariance inflation factor (default: 1.02).
    noise_std : float, optional
        Measurement noise std (default: None = auto).
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
        Dictionary containing unfolding results.
    """
    x0_default = np.ones(n_energy_bins) / max(n_energy_bins, 1)

    result = run_unfolding(
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
            solve_eki,
            n_ensemble=n_ensemble,
            n_iterations=n_iterations,
            regularization=regularization,
            inflation=inflation,
            noise_std=noise_std,
            random_state=random_state,
        ),
        solve_kwargs={},
        method_name="EKI",
        extra_output={
            "n_ensemble": n_ensemble,
            "n_iterations": n_iterations,
            "regularization": regularization,
            "inflation": inflation,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
    return result
