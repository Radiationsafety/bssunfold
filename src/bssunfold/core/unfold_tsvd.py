"""Truncated SVD (TSVD) unfolding method for neutron spectrum reconstruction.

This module provides the core solve_tsvd solver and the unfold_tsvd
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List
from scipy.linalg import svd

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_tsvd", "unfold_tsvd"]


def _automatic_k_selection(
    s: np.ndarray, A: np.ndarray, b: np.ndarray,
    method: str = "discrepancy", noise_level: float = None
) -> int:
    """Automatically select truncation parameter k for TSVD."""
    m, n = A.shape
    max_k = min(m, n)

    if method == "discrepancy":
        if noise_level is None:
            noise_level = s[0] * 1e-3
        U, s_full, Vh = svd(A, full_matrices=False)
        for i in range(1, max_k + 1):
            s_i = s_full[:i]
            U_i = U[:, :i]
            V_i = Vh[:i, :].T
            x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b
            residual = np.linalg.norm(A @ x_i - b)
            if residual <= noise_level * np.sqrt(max(m - i, 1)):
                return i
        return max_k

    elif method == "energy":
        energy_threshold = 0.95
        cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        return int(np.argmax(cumulative_energy >= energy_threshold)) + 1

    elif method == "l_curve":
        U, s_full, Vh = svd(A, full_matrices=False)
        V = Vh.T
        residual_norms = []
        solution_norms = []
        for i in range(1, min(len(s_full), n) + 1):
            s_i = s_full[:i]
            U_i = U[:, :i]
            V_i = V[:, :i]
            x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b
            residual_norms.append(np.linalg.norm(A @ x_i - b))
            solution_norms.append(np.linalg.norm(x_i))

        log_res = np.log(np.maximum(residual_norms, 1e-300))
        log_sol = np.log(np.maximum(solution_norms, 1e-300))
        curvature = []
        for i in range(1, len(log_res) - 1):
            dx1 = log_res[i] - log_res[i - 1]
            dy1 = log_sol[i] - log_sol[i - 1]
            dx2 = log_res[i + 1] - log_res[i]
            dy2 = log_sol[i + 1] - log_sol[i]
            curv = abs(dx1 * dy2 - dx2 * dy1) / ((dx1 ** 2 + dy1 ** 2) ** 1.5 + 1e-10)
            curvature.append(curv)
        if len(curvature) > 0:
            k_idx = np.argmax(curvature) + 1
            return min(k_idx + 1, len(s))
        return len(s) // 2

    elif method == "gcv":
        U, s_full, Vh = svd(A, full_matrices=False)
        beta = U.T @ b
        gcv_values = []
        k_values = list(range(1, min(len(s_full), n) + 1))
        for i in k_values:
            residual = np.sum(beta[i:] ** 2)
            eff_params = m - i
            gcv = residual / (eff_params ** 2) if eff_params > 0 else np.inf
            gcv_values.append(gcv)
        return k_values[np.argmin(gcv_values)]

    elif method == "threshold_ratio":
        threshold_ratio = 1e-2
        s_normalized = s / s[0]
        return int(np.sum(s_normalized > threshold_ratio))

    elif method == "median_threshold":
        median_s = np.median(s)
        return int(np.sum(s >= median_s))

    elif method == "donoho":
        sigma_donoho = 0.05
        n_val = n
        donoho_rcond = 4 / np.sqrt(3) * np.sqrt(n_val) * sigma_donoho
        return int(np.sum(s > donoho_rcond))

    else:
        mean_s = np.mean(s)
        return int(np.sum(s >= mean_s))


def solve_tsvd(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    method: str = "discrepancy",
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    noise_level: Optional[float] = None,
) -> np.ndarray:
    """Solve unfolding problem using Truncated SVD (TSVD).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    method : str, optional
        K-selection method: 'discrepancy', 'l_curve', 'gcv', 'energy',
        'threshold_ratio', 'median_threshold', 'donoho' (default: 'discrepancy').
    k : int, optional
        Fixed number of singular values to keep. Overrides method.
    threshold : float, optional
        Threshold ratio for singular value truncation.
    noise_level : float, optional
        Noise level estimate for discrepancy principle.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    U, s, Vh = svd(A, full_matrices=False)
    V = Vh.T

    if k is not None:
        k = min(k, len(s))
    elif threshold is not None:
        k = np.sum(s / s[0] > threshold)
    else:
        k = _automatic_k_selection(s, A, b, method=method, noise_level=noise_level)

    k = max(1, min(k, min(A.shape)))
    s_k = s[:k]
    U_k = U[:, :k]
    V_k = V[:, :k]

    x = V_k @ np.diag(1.0 / s_k) @ U_k.T @ b
    return np.maximum(x, 0)


def unfold_tsvd(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    method: str = "discrepancy",
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    noise_level: Optional[float] = None,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Truncated SVD (TSVD).

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
    method : str, optional
        K-selection method (default: 'discrepancy').
    k : int, optional
        Fixed truncation parameter.
    threshold : float, optional
        Threshold ratio for truncation.
    noise_level : float, optional
        Noise level estimate.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
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
    x0_default = np.zeros(n_energy_bins)

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
            solve_tsvd,
            method=method,
            k=k,
            threshold=threshold,
            noise_level=noise_level,
        ),
        solve_kwargs={},
        method_name="TSVD",
        extra_output={
            "k": k,
            "k_method": method,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
