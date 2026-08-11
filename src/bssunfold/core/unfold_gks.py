"""Generalized Krylov Subspace (GKS) unfolding method.

This module provides a Python port of the Generalized Krylov Subspace
(GKS) method for neutron spectrum unfolding.  GKS builds a Krylov
subspace by Golub-Kahan bidiagonalization of the response matrix ``A``
and projects both ``A`` and the regularization operator ``L`` onto that
subspace.  At each iteration a small projected Tikhonov problem is
solved, with the regularization parameter selected automatically on the
projected problem by Generalized Cross Validation (GCV), the Discrepancy
Principle (DP) or the L-curve.

The algorithm follows the GKS implementation of the TRIPs-Py library by
Mirjeta Pasha, Silvia Gazzola, Connor Sanderford and Ugochukwu Obinna
Ugwu (Apache-2.0), which itself is a Python port of the GKS method from
the IR Tools package of Gazzola, Hansen and Nagy (3-Clause BSD License).
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_gks", "unfold_gks"]


def _make_regoperator(n: int, smoothness_order: int = 0) -> np.ndarray:
    """Build the regularization operator L for the given derivative order.

    Parameters
    ----------
    n : int
        Number of energy bins.
    smoothness_order : int, optional
        Derivative order of the regularization operator: 0 (identity),
        1 (first derivative) or 2 (second derivative).

    Returns
    -------
    np.ndarray
        Regularization operator as a dense ndarray of shape
        (n - order, n).
    """
    if smoothness_order == 0:
        return np.eye(n)
    if smoothness_order not in (1, 2):
        raise ValueError(
            f"Unsupported smoothness_order: {smoothness_order}. Use 0, 1 or 2."
        )
    return create_derivative_matrix(n, smoothness_order).toarray()


def _projected_gcv(
    RA: np.ndarray,
    RL: np.ndarray,
    bhat: np.ndarray,
    n_lambdas: int = 200,
    lambda_range: Tuple[float, float] = (1e-12, 1e2),
) -> float:
    """Select the regularization parameter on the projected problem by GCV.

    Parameters
    ----------
    RA : np.ndarray
        Projected data matrix (k x k).
    RL : np.ndarray
        Projected regularization matrix (k x k).
    bhat : np.ndarray
        Projected right-hand side (k,).
    n_lambdas : int, optional
        Number of candidate lambdas (default: 200).
    lambda_range : tuple, optional
        Log range of candidate lambdas (default: (1e-12, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    U, s, _ = np.linalg.svd(RA, full_matrices=False)
    c = U.T @ bhat
    s2 = s**2

    lambdas = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambdas
    )
    gcv_values = np.empty_like(lambdas)
    m_proj = RA.shape[0]
    for i, lam in enumerate(lambdas):
        filt = s2 / (s2 + lam)
        residual_coeff = lam / (s2 + lam)
        residual_sq = float(np.sum((residual_coeff * c) ** 2))
        trace_term = float(np.sum(filt))
        gcv_values[i] = residual_sq / (m_proj - trace_term) ** 2

    idx = int(np.argmin(gcv_values))
    return float(lambdas[idx])


def _projected_dp(
    RA: np.ndarray,
    bhat: np.ndarray,
    noise_level: float,
    n_lambdas: int = 200,
    lambda_range: Tuple[float, float] = (1e-12, 1e2),
) -> float:
    """Select the regularization parameter by the Discrepancy Principle."""
    U, s, Vt = np.linalg.svd(RA, full_matrices=False)
    c = U.T @ bhat
    s2 = s**2
    m_proj = RA.shape[0]
    target = noise_level * np.sqrt(m_proj)

    lambdas = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambdas
    )
    residuals = np.empty_like(lambdas)
    for i, lam in enumerate(lambdas):
        filt = s / (s2 + lam)
        x = Vt.T @ (filt * c)
        residuals[i] = float(np.linalg.norm(RA @ x - bhat))

    idx = int(np.argmin(np.abs(residuals - target)))
    return float(lambdas[idx])


def _projected_lcurve(
    RA: np.ndarray,
    bhat: np.ndarray,
    n_lambdas: int = 200,
    lambda_range: Tuple[float, float] = (1e-12, 1e2),
) -> float:
    """Select the regularization parameter by the L-curve corner."""
    U, s, Vt = np.linalg.svd(RA, full_matrices=False)
    c = U.T @ bhat
    s2 = s**2

    lambdas = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambdas
    )
    residuals = []
    norms = []
    for lam in lambdas:
        filt = s / (s2 + lam)
        x = Vt.T @ (filt * c)
        residuals.append(float(np.linalg.norm(RA @ x - bhat)))
        norms.append(float(np.linalg.norm(x)))

    if len(residuals) < 3:
        return float(lambdas[len(lambdas) // 2])

    log_res = np.log(np.maximum(residuals, 1e-300))
    log_norm = np.log(np.maximum(norms, 1e-300))
    p1 = np.array([log_res[0], log_norm[0]])
    p2 = np.array([log_res[-1], log_norm[-1]])
    denom = np.linalg.norm(p2 - p1)
    # Perpendicular distance from each curve point to the chord p1-p2.
    # Compute with explicit 3D cross products to avoid NumPy deprecation
    # of 2D vector inputs.
    d21 = p2 - p1
    distances = np.abs(
        d21[0] * (p1[1] - log_norm) - d21[1] * (p1[0] - log_res)
    ) / max(denom, 1e-300)

    idx_max = int(np.argmax(distances))
    return float(lambdas[idx_max])


def solve_gks(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    smoothness_order: int = 0,
    regularization_method: str = "gcv",
    max_iterations: Optional[int] = None,
    regularization: float = 1e-8,
    noise_level: Optional[float] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with the Generalized Krylov Subspace method.

    Performs Golub-Kahan bidiagonalization of ``A`` and projects both
    ``A`` and the regularization operator ``L`` (identity or a
    derivative matrix) onto the Krylov subspace.  At each iteration the
    projected Tikhonov problem ``min ||R_A y - bhat||^2 +
    lambda * ||R_L y||^2`` is solved, where ``lambda`` is selected
    automatically by GCV, the Discrepancy Principle or the L-curve.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum (unused, kept for API compatibility).
    smoothness_order : int, optional
        Derivative order of the regularization operator L: 0 (identity),
        1 or 2 (default: 0).
    regularization_method : str, optional
        Method for selecting the regularization parameter: ``'gcv'``,
        ``'dp'``, ``'lcurve'`` or ``'manual'`` (default: 'gcv').
    max_iterations : int, optional
        Maximum Krylov dimension. Defaults to ``min(A.shape)``.
    regularization : float, optional
        Manual/fallback regularization parameter (default: 1e-8).
    noise_level : float, optional
        Relative noise level used by the Discrepancy Principle.

    Returns
    -------
    tuple
        ``(spectrum, iterations, converged)`` where ``converged`` reports
        whether the Krylov space was fully spanned or a fixed point of the
        projected solution was reached.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    if max_iterations is None:
        max_iterations = min(m, n)
    max_iterations = max(1, int(max_iterations))

    L = _make_regoperator(n, smoothness_order)

    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        return np.zeros(n), 0, True

    U = np.zeros((m, 1))
    U[:, 0] = b / beta
    V = np.empty((n, 0))
    alphas: List[float] = []
    betas: List[float] = []

    best_x = np.zeros(n)
    iterations = 0
    converged = False

    for k in range(1, max_iterations + 1):
        u = U[:, k - 1]
        if k == 1:
            v = A.T @ u
        else:
            v = A.T @ u - betas[-1] * V[:, -1]
        alpha = float(np.linalg.norm(v))
        if alpha <= 1e-14:
            converged = True
            break
        v = v / alpha
        V = np.hstack([V, v.reshape(-1, 1)])

        u2 = A @ v - alpha * u
        new_beta = float(np.linalg.norm(u2))
        if new_beta <= 1e-14:
            converged = True
        else:
            U = np.hstack([U, (u2 / new_beta).reshape(-1, 1)])

        alphas.append(alpha)
        betas.append(new_beta)

        # Number of columns of U: k+1 when the bidiagonalization was
        # extended, k when it terminated (new_beta ~ 0).
        p = U.shape[1]

        # Projected data matrix B_k (p x k)
        B = np.zeros((p, k))
        B[range(k), range(k)] = alphas
        if k > 1:
            sub = min(k - 1, p - 1)
            B[list(range(1, sub + 1)), list(range(sub))] = betas[:sub]

        bhat = np.zeros(p)
        bhat[0] = beta

        # Project A onto the subspace: R_A = U_{k+1}^T A V_k
        RA = U.T @ (A @ V)
        # Project L onto the subspace: R_L = L V_k
        RL = L @ V

        if regularization_method == "gcv":
            lam = _projected_gcv(RA, RL, bhat)
        elif regularization_method == "dp":
            if noise_level is None:
                noise_level = 0.01
            lam = _projected_dp(RA, bhat, noise_level)
        elif regularization_method == "lcurve":
            lam = _projected_lcurve(RA, bhat)
        elif regularization_method == "manual":
            lam = regularization
        else:
            raise ValueError(
                f"Unsupported regularization method: "
                f"{regularization_method}. "
                "Choose from 'gcv', 'dp', 'lcurve', 'manual'."
            )

        if not np.isfinite(lam) or lam <= 0:
            lam = regularization

        # Solve the projected Tikhonov problem:
        # min ||R_A y - bhat||^2 + lam * ||R_L y||^2
        lhs = np.vstack([RA, np.sqrt(lam) * RL])
        rhs = np.concatenate([bhat, np.zeros(RL.shape[0])])
        y, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        x = V @ y
        best_x = x

        iterations = k

        if converged:
            break

    best_x = np.maximum(best_x, 0)
    return best_x, iterations, converged


def unfold_gks(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    smoothness_order: int = 0,
    regularization_method: str = "gcv",
    max_iterations: Optional[int] = None,
    regularization: float = 1e-8,
    noise_level: Optional[float] = None,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum with the Generalized Krylov Subspace method.

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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess (accepted for API compatibility).
    smoothness_order : int, optional
        Derivative order of the regularization operator L (default: 0).
    regularization_method : str, optional
        Method for selecting the regularization parameter: ``'gcv'``,
        ``'dp'``, ``'lcurve'`` or ``'manual'`` (default: 'gcv').
    max_iterations : int, optional
        Maximum Krylov dimension. Defaults to ``min(n_detectors,
        n_energy_bins)``.
    regularization : float, optional
        Manual/fallback regularization parameter (default: 1e-8).
    noise_level : float, optional
        Relative noise level used by the Discrepancy Principle.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty (default: False).
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
            solve_gks,
            smoothness_order=smoothness_order,
            regularization_method=regularization_method,
            max_iterations=max_iterations,
            regularization=regularization,
            noise_level=noise_level,
        ),
        solve_kwargs={},
        method_name="GKS",
        extra_output={
            "regularization_method": regularization_method,
            "smoothness_order": int(smoothness_order),
            "regularization": float(regularization),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
