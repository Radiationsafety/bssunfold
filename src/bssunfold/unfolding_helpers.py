"""
Helper implementations for unfolding algorithms and uncertainty estimation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def calculate_spectrum_maxed_core(
    E_MeV: np.ndarray,
    K_j: np.ndarray,
    Q_j: np.ndarray,
    phi_0: Optional[np.ndarray] = None,
    sigma_factor: float = 0.01,
    omega: float = 1.0,
    mu: float = 1.0,
    maxiter: int = 5000,
    tol: float = 1e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Core implementation of the MAXED (maximum entropy) algorithm.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid [MeV].
    K_j : np.ndarray
        Response matrix (n_detectors, n_energy_bins).
    Q_j : np.ndarray
        Measured readings vector (n_detectors,).
    phi_0 : Optional[np.ndarray], optional
        Reference spectrum. If None, a default two-peak spectrum is used.
    sigma_factor : float, optional
        Relative measurement uncertainty.
    omega : float, optional
        Chi-square constraint parameter.
    mu : float, optional
        Penalty factor for constraint enforcement.
    maxiter : int, optional
        Maximum iterations for simulated annealing.
    tol : float, optional
        Convergence tolerance.
    seed : int, optional
        RNG seed for annealing.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with spectrum and diagnostics.
    """
    E = np.asarray(E_MeV, dtype=float)
    K_j = np.asarray(K_j, dtype=float)
    Q_j = np.asarray(Q_j, dtype=float)

    if phi_0 is None:
        n_energy = len(E)
        phi_0 = np.zeros_like(E)
        pos1 = n_energy // 8
        pos2 = 2 * n_energy // 4
        sigma = max(1, n_energy // 10)
        for i in range(n_energy):
            gauss1 = np.exp(-0.5 * ((i - pos1) / sigma) ** 2)
            gauss2 = np.exp(-0.5 * ((i - pos2) / sigma) ** 2)
            phi_0[i] = gauss1 + gauss2
        if np.max(phi_0) > 0:
            phi_0 = phi_0 / np.max(phi_0)
    else:
        phi_0 = np.asarray(phi_0, dtype=float)
        if phi_0.shape != E.shape:
            raise ValueError("phi_0 shape mismatch")

    E_lower = E.copy()
    E_upper = np.roll(E_lower, -1)
    E_upper[-1] = E_lower[-1] * 1.1
    dlnE = np.log(E_upper / E_lower)

    Q_j_nonzero = np.where(Q_j == 0, 1.0, Q_j)
    sigma_j = sigma_factor * Q_j_nonzero

    def calc_phi(lam: np.ndarray) -> np.ndarray:
        exp_term = -np.dot(K_j.T, lam)
        phi = phi_0 * np.exp(exp_term)
        return np.maximum(phi, 1e-20)

    def calc_predicted(phi: np.ndarray) -> np.ndarray:
        return (phi * dlnE) @ K_j.T

    def dual_Z(lam: np.ndarray) -> float:
        try:
            phi = calc_phi(lam)
            c_pred = calc_predicted(phi)
            ratio = np.where(phi > 1e-20, phi / np.maximum(phi_0, 1e-20), 1.0)
            term1 = phi * np.log(ratio)
            term2 = phi_0 - phi
            entropy = -np.sum((term1 + term2) * dlnE)
            chi_sq = np.sum(((c_pred - Q_j) / sigma_j) ** 2)
            constraint = 0.5 * mu * max(0.0, chi_sq - omega)
            linear = np.dot(lam, c_pred - Q_j)
            return entropy - linear - constraint
        except Exception:
            return -1e10

    rng = np.random.default_rng(seed)
    lam = np.zeros(len(Q_j))
    Z = dual_Z(lam)
    best_lam = lam.copy()
    best_Z = Z
    T = 10.0
    cooling = 0.95
    min_T = 1e-8
    steps_per_T = 50

    for iter_count in range(maxiter):
        for _ in range(steps_per_T):
            step = 0.5 * T * (1 + iter_count / 500)
            trial_lam = lam + rng.normal(0, step, size=lam.shape)
            trial_lam = np.clip(trial_lam, -50, 50)
            trial_Z = dual_Z(trial_lam)
            dZ = trial_Z - Z
            if dZ > 0 or (T > 1e-12 and rng.random() < np.exp(dZ / T)):
                lam = trial_lam
                Z = trial_Z
                if Z > best_Z:
                    best_lam = lam.copy()
                    best_Z = Z
        T *= cooling
        if T < min_T or (iter_count > 200 and abs(best_Z - Z) < tol):
            break

    phi_opt = calc_phi(best_lam)
    c_pred = calc_predicted(phi_opt)

    if np.sum(c_pred) > 0 and np.sum(Q_j) > 0:
        scale = np.sum(Q_j * c_pred) / np.sum(c_pred**2)
        phi_opt *= scale
        c_pred *= scale

    chi_sq = float(np.sum(((c_pred - Q_j) / sigma_j) ** 2))

    return {
        "energy": E_lower,
        "spectrum": phi_opt,
        "success": True,
        "effective_readings": c_pred,
        "energy_range": (float(E_lower[0]), float(E_upper[-1])),
        "chi_square": chi_sq,
        "omega": omega,
        "mu": mu,
        "lambda_values": best_lam.tolist(),
        "delta_lnE": dlnE.tolist(),
    }


def gravel(
    S: np.ndarray,
    measurements: np.ndarray,
    x0: np.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    regularization: float = 0.0,
) -> Dict[str, Any]:
    """
    GRAVEL algorithm for neutron spectrum unfolding.
    """
    M, N = S.shape
    if len(measurements) != M:
        raise ValueError("Measurement vector length does not match S rows")
    if len(x0) != N:
        raise ValueError("Initial spectrum length does not match S columns")

    x = x0.copy().astype(np.float64)
    measurements = measurements.astype(np.float64)

    valid_indices = measurements > 0
    if np.sum(valid_indices) == 0:
        raise ValueError("All measurements are zero or negative")

    S_valid = S[valid_indices]
    measurements_valid = measurements[valid_indices]
    M_valid = len(measurements_valid)

    J_prev = 0.0
    dJ_prev = 1.0
    error_history = []
    chi_sq_history = []

    for iteration in range(1, max_iterations + 1):
        computed = np.zeros(M_valid)
        for i in range(M_valid):
            computed[i] = S_valid[i, :] @ x

        W = np.zeros((M_valid, N))
        for j in range(N):
            for i in range(M_valid):
                if computed[i] > 0 and x[j] > 0:
                    W[i, j] = (
                        measurements_valid[i] * S_valid[i, j] * x[j] / computed[i]
                    )

            numerator = 0.0
            denominator = 0.0
            for i in range(M_valid):
                if computed[i] > 0 and measurements_valid[i] > 0 and W[i, j] > 0:
                    log_ratio = np.log(measurements_valid[i] / computed[i])
                    numerator += W[i, j] * log_ratio
                    denominator += W[i, j]

            if denominator > 0:
                regularization_term = regularization * np.log(x[j] + 1e-10)
                update = np.exp((numerator - regularization_term) / denominator)
                x[j] *= update

        computed_final = np.zeros(M_valid)
        for i in range(M_valid):
            computed_final[i] = S_valid[i, :] @ x

        chi_sq = np.sum(
            (computed_final - measurements_valid) ** 2
            / np.maximum(measurements_valid, 1e-10)
        )
        chi_sq_history.append(chi_sq)
        J = chi_sq / np.sum(computed_final)
        dJ = J_prev - J
        ddJ = abs(dJ - dJ_prev)
        error_history.append(ddJ)

        if ddJ <= tolerance:
            break
        J_prev = J
        dJ_prev = dJ

    computed_all = np.zeros(M)
    for i in range(M):
        computed_all[i] = S[i, :] @ x

    x_normalized = x / np.sum(x) if np.sum(x) > 0 else x

    return {
        "spectrum": x_normalized,
        "spectrum_absolute": x,
        "error_history": np.array(error_history),
        "chi_sq_history": np.array(chi_sq_history),
        "computed_measurements": computed_all,
        "iterations": iteration,
        "converged": ddJ <= tolerance,
    }


def gravel_with_errors(
    S: np.ndarray,
    measurements: np.ndarray,
    x0: np.ndarray,
    tolerance: float,
    max_iterations: int = 1000,
    regularization: float = 0.0,
) -> Dict[str, Any]:
    """
    GRAVEL algorithm with spectrum error estimation.
    """
    results = gravel(
        S=S,
        measurements=measurements,
        x0=x0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
    )

    spectrum_errors, covariance_matrix, correlation_matrix = calculate_spectrum_errors(
        results["spectrum_absolute"],
        S,
        measurements,
        method="covariance",
    )
    results["spectrum_errors"] = spectrum_errors
    results["covariance_matrix"] = covariance_matrix
    results["correlation_matrix"] = correlation_matrix
    return results


def calculate_spectrum_errors(
    spectrum: np.ndarray,
    S: np.ndarray,
    measurements: np.ndarray,
    method: str = "covariance",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate spectrum errors using selected method.
    """
    if method == "covariance":
        return calculate_covariance_errors(spectrum, S, measurements)
    if method == "bootstrap":
        return calculate_bootstrap_errors(spectrum, S, measurements)
    if method == "jackknife":
        return calculate_jackknife_errors(spectrum, S, measurements)
    return calculate_covariance_errors(spectrum, S, measurements)


def calculate_covariance_errors(
    spectrum: np.ndarray,
    S: np.ndarray,
    measurements: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate spectrum errors via covariance matrix propagation.
    """
    try:
        computed = S @ spectrum
        J = np.zeros_like(S)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if computed[i] > 0 and spectrum[j] > 0:
                    J[i, j] = (
                        measurements[i] * S[i, j] * spectrum[j] / (computed[i] ** 2)
                    ) - (S[i, j] / computed[i])

        cov_meas = np.diag(measurements)
        J_pinv = np.linalg.pinv(J)
        cov_spectrum = J_pinv @ cov_meas @ J_pinv.T
        spectrum_errors = np.sqrt(np.abs(np.diag(cov_spectrum)))
        D = np.sqrt(np.diag(cov_spectrum))
        D_inv = np.linalg.pinv(np.diag(D))
        correlation_matrix = D_inv @ cov_spectrum @ D_inv
        return spectrum_errors, cov_spectrum, correlation_matrix
    except Exception:
        return None, None, None


def calculate_bootstrap_errors(
    spectrum: np.ndarray,
    S: np.ndarray,
    measurements: np.ndarray,
    n_bootstrap: int = 100,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate spectrum errors via bootstrap resampling.
    """
    M, N = S.shape
    bootstrap_spectra = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(M, M, replace=True)
        S_boot = S[indices]
        meas_boot = measurements[indices]
        x_boot = spectrum.copy()
        for _ in range(10):
            computed = S_boot @ x_boot
            for j in range(N):
                numerator = 0.0
                denominator = 0.0
                for k in range(M):
                    if computed[k] > 0 and meas_boot[k] > 0:
                        weight = (
                            meas_boot[k] * S_boot[k, j] * x_boot[j] / computed[k]
                        )
                        numerator += weight * np.log(meas_boot[k] / computed[k])
                        denominator += weight
                if denominator > 0:
                    x_boot[j] *= np.exp(numerator / denominator)
        bootstrap_spectra.append(x_boot)

    if not bootstrap_spectra:
        return None, None, None

    bootstrap_spectra = np.array(bootstrap_spectra)
    spectrum_errors = np.std(bootstrap_spectra, axis=0)
    covariance_matrix = np.cov(bootstrap_spectra.T)
    D = np.sqrt(np.diag(covariance_matrix))
    D_inv = np.linalg.pinv(np.diag(D))
    correlation_matrix = D_inv @ covariance_matrix @ D_inv
    return spectrum_errors, covariance_matrix, correlation_matrix


def calculate_jackknife_errors(
    spectrum: np.ndarray,
    S: np.ndarray,
    measurements: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate spectrum errors via jackknife resampling.
    """
    M, N = S.shape
    jackknife_spectra = []

    for i in range(M):
        indices = [j for j in range(M) if j != i]
        S_jack = S[indices]
        meas_jack = measurements[indices]
        x_jack = spectrum.copy()
        for _ in range(10):
            computed = S_jack @ x_jack
            for j in range(N):
                numerator = 0.0
                denominator = 0.0
                for k in range(len(indices)):
                    if computed[k] > 0 and meas_jack[k] > 0:
                        weight = (
                            meas_jack[k] * S_jack[k, j] * x_jack[j] / computed[k]
                        )
                        numerator += weight * np.log(meas_jack[k] / computed[k])
                        denominator += weight
                if denominator > 0:
                    x_jack[j] *= np.exp(numerator / denominator)
        jackknife_spectra.append(x_jack)

    if not jackknife_spectra:
        return None, None, None

    jackknife_spectra = np.array(jackknife_spectra)
    mean_spectrum = np.mean(jackknife_spectra, axis=0)
    spectrum_errors = np.sqrt(
        (M - 1) / M * np.sum((jackknife_spectra - mean_spectrum) ** 2, axis=0)
    )
    covariance_matrix = np.cov(jackknife_spectra.T)
    D = np.sqrt(np.diag(covariance_matrix))
    D_inv = np.linalg.pinv(np.diag(D))
    correlation_matrix = D_inv @ covariance_matrix @ D_inv
    return spectrum_errors, covariance_matrix, correlation_matrix
