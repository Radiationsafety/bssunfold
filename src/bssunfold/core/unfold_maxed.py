"""MAXED unfolding method for neutron spectrum reconstruction.

This module provides the core solve_maxed solver and the unfold_maxed
wrapper for use with the Detector class.
"""

import math
import random
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_maxed", "unfold_maxed"]


def _default_phi_0(n_energy: int) -> np.ndarray:
    """Create default reference spectrum with two Gaussian peaks."""
    phi_0 = np.zeros(n_energy)
    pos1 = n_energy // 8
    pos2 = 2 * n_energy // 4
    sigma = max(1, n_energy // 10)
    for i in range(n_energy):
        gauss1 = np.exp(-0.5 * ((i - pos1) / sigma) ** 2)
        gauss2 = np.exp(-0.5 * ((i - pos2) / sigma) ** 2)
        phi_0[i] = gauss1 + gauss2
    if np.max(phi_0) > 0:
        phi_0 = phi_0 / np.max(phi_0)
    return phi_0


def solve_maxed(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.01,
    omega: float = 1.0,
    mu: float = 1.0,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using MAXED (Maximum Entropy Deconvolution).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial (reference) spectrum (n,).
    sigma_factor : float, optional
        Measurement error factor (default: 0.01).
    omega : float, optional
        Chi-square constraint parameter (default: 1.0).
    mu : float, optional
        Lagrange multiplier for constraint (default: 1.0).
    max_iterations : int, optional
        Maximum annealing iterations (default: 5000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    n_energy = A.shape[1]
    phi_0 = x0.copy()
    dlnE = np.full(n_energy, 1.0 / n_energy)

    Q_j_nonzero = np.where(b == 0, 1.0, b)
    sigma_j = sigma_factor * Q_j_nonzero
    K_j = A

    def calc_phi(lam):
        exp_term = -np.dot(K_j.T, lam)
        phi = phi_0 * np.exp(exp_term)
        return np.maximum(phi, 1e-20)

    def calc_predicted(phi):
        return (phi * dlnE) @ K_j.T

    def dual_Z(lam):
        try:
            phi = calc_phi(lam)
            C_pred = calc_predicted(phi)

            ratio = np.where(phi > 1e-20, phi / np.maximum(phi_0, 1e-20), 1.0)
            term1 = phi * np.log(ratio)
            term2 = phi_0 - phi
            entropy = -np.sum((term1 + term2) * dlnE)

            chi_sq = np.sum(((C_pred - b) / sigma_j) ** 2)
            constraint = 0.5 * mu * max(0.0, chi_sq - omega)

            linear = np.dot(lam, C_pred - b)

            return entropy - linear - constraint
        except Exception:
            return -1e10

    rng = np.random.default_rng(seed)
    lam = np.zeros(len(b))
    Z = dual_Z(lam)

    best_lam = lam.copy()
    best_Z = Z
    T = 10.0
    cooling = 0.95
    min_T = 1e-8
    steps_per_T = 50
    converged = False

    for iter_count in range(max_iterations):
        for _ in range(steps_per_T):
            step = 0.5 * T * (1 + iter_count / 500)
            trial_lam = lam + rng.normal(0, step, size=lam.shape)
            trial_lam = np.clip(trial_lam, -50, 50)
            trial_Z = dual_Z(trial_lam)

            dZ = trial_Z - Z
            if dZ > 0 or (T > 1e-12 and random.random() < math.exp(dZ / T)):
                lam = trial_lam
                Z = trial_Z
                if Z > best_Z:
                    best_lam = lam.copy()
                    best_Z = Z

        T *= cooling
        if T < min_T or (iter_count > 200 and abs(best_Z - Z) < tolerance):
            converged = True
            break

    phi_opt = calc_phi(best_lam)
    C_pred = calc_predicted(phi_opt)

    if np.sum(C_pred) > 0 and np.sum(b) > 0:
        A_scale = np.sum(b * C_pred) / np.sum(C_pred ** 2)
        phi_opt *= A_scale

    return phi_opt, iter_count + 1, converged


def unfold_maxed(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma_factor: float = 0.01,
    omega: float = 1.0,
    mu: float = 1.0,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
    seed: int = 42,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the MAXED algorithm.

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
        Reference spectrum. If None, default two-Gaussian spectrum.
    sigma_factor : float, optional
        Measurement error factor (default: 0.01).
    omega : float, optional
        Chi-square constraint (default: 1.0).
    mu : float, optional
        Lagrange multiplier (default: 1.0).
    max_iterations : int, optional
        Maximum iterations (default: 5000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    seed : int, optional
        Random seed (default: 42).
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
    if initial_spectrum is not None:
        x0_ref = np.asarray(initial_spectrum, dtype=float)
    else:
        x0_ref = _default_phi_0(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=x0_ref,
        default_initial=_default_phi_0(n_energy_bins),
        solve_func=make_solve_wrapper(
            solve_maxed,
            sigma_factor=sigma_factor,
            omega=omega,
            mu=mu,
            max_iterations=max_iterations,
            tolerance=tolerance,
            seed=seed,
        ),
        solve_kwargs={},
        method_name="MAXED",
        extra_output={
            "omega": omega,
            "mu": mu,
            "sigma_factor": sigma_factor,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
