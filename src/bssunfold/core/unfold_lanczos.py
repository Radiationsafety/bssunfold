"""Lanczos-type (Golub-Kahan) hybrid unfolding with GCV regularization.

This module provides a Krylov-subspace unfolding method inspired by the
"Hybrid LSQR" family (Golub-Kahan bidiagonalization + projected Tikhonov).
At each iteration a new approximation is built in the Krylov subspace
spanned by the bidiagonalization vectors, and the regularization parameter
is selected automatically on the small projected problem via Generalized
Cross Validation (GCV). No a-priori spectrum is required.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ..utils.validators import validate_system

__all__ = ["solve_lanczos", "unfold_lanczos"]


def _projected_gcv(
    B: np.ndarray,
    bhat: np.ndarray,
    m: int,
    n_lambdas: int = 200,
    lambda_range: Tuple[float, float] = (1e-12, 1e2),
) -> float:
    """Select the regularization parameter on the projected problem by GCV.

    Parameters
    ----------
    B : np.ndarray
        Bidiagonal projected matrix, shape (k + 1, k).
    bhat : np.ndarray
        Projected right-hand side, shape (k + 1,).
    m : int
        Size of the original problem (full residual correction).
    n_lambdas : int, optional
        Number of candidate lambdas (default: 200).
    lambda_range : tuple, optional
        Log range of candidate lambdas (default: (1e-12, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    Ub, s, _ = np.linalg.svd(B, full_matrices=False)
    c = Ub.T @ bhat
    orth_res = float(np.linalg.norm(bhat) ** 2 - np.sum(c**2))
    s2 = s**2

    lambdas = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambdas
    )
    gcv_values = np.empty_like(lambdas)
    for i, lam in enumerate(lambdas):
        num = np.sum((c * lam / (s2 + lam)) ** 2) + orth_res
        den = (m - np.sum(s2 / (s2 + lam))) ** 2
        gcv_values[i] = num / den

    idx = int(np.argmin(gcv_values))
    return float(lambdas[idx])


def solve_lanczos(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: Optional[int] = None,
    regularization: float = 1e-8,
    noise_level: Optional[float] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with a Lanczos-hybrid method.

    Performs Golub-Kahan bidiagonalization of ``A``, generating a sequence
    of Krylov subspaces. On the projected problem ``min ||B_k y - bhat||^2``
    a Tikhonov term ``lambda * ||y||^2`` is added, where ``lambda`` is
    selected automatically by GCV at each iteration. The iterate
    ``x_k = V_k y_k`` is an approximation in the Krylov subspace, so no
    a-priori spectrum is required (``x0`` is accepted for API compatibility
    only).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum (unused, kept for API compatibility).
    max_iterations : int, optional
        Maximum Krylov dimension. Defaults to ``min(A.shape)``.
    regularization : float, optional
        Fallback regularization parameter used if GCV returns a degenerate
        value; default: 1e-8.
    noise_level : float, optional
        Relative noise level. If given, iterations stop early by the
        discrepancy principle ``||A x - b|| <= noise_level * sqrt(m)``.

    Returns
    -------
    tuple
        ``(spectrum, iterations, converged)`` where ``converged`` reports
        whether the discrepancy-principle criterion was met (or the Krylov
        space was fully spanned).
    """
    A, b, _ = validate_system(A, b)
    m, n = A.shape

    if max_iterations is None:
        max_iterations = min(m, n)
    max_iterations = max(1, int(max_iterations))

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

        B = np.zeros((k + 1, k))
        B[range(k), range(k)] = alphas
        if k > 1:
            B[list(range(1, k)), range(k - 1)] = betas[:-1]

        bhat = np.zeros(k + 1)
        bhat[0] = beta

        lam = _projected_gcv(B, bhat, m)
        if lam <= 0.0 or not np.isfinite(lam):
            lam = regularization

        Ub, s, Vb = np.linalg.svd(B, full_matrices=False)
        c = Ub.T @ bhat
        s2 = s**2
        y = Vb @ (s * c / (s2 + lam))
        x = V @ y
        best_x = x

        iterations = k

        if noise_level is not None:
            residual = float(np.linalg.norm(A @ x - b))
            if residual <= noise_level * np.sqrt(m):
                converged = True
                break

        if converged:
            break

    return best_x, iterations, converged


def unfold_lanczos(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    regularization_method: str = "gcv",
    max_iterations: Optional[int] = None,
    regularization: float = 1e-8,
    noise_level: Optional[float] = None,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum with the Lanczos-hybrid (Krylov) method.

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
    regularization_method : str, optional
        Method for selecting the regularization parameter. Only ``'gcv'``
        is supported (default: 'gcv').
    max_iterations : int, optional
        Maximum Krylov dimension. Defaults to ``min(n_detectors,
        n_energy_bins)``.
    regularization : float, optional
        Fallback regularization parameter (default: 1e-8).
    noise_level : float, optional
        Relative noise level used for discrepancy-principle early stopping.
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
    if regularization_method != "gcv":
        raise ValueError(
            f"Unsupported regularization method: {regularization_method}. "
            "The Lanczos hybrid method currently supports 'gcv'."
        )

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
            solve_lanczos,
            max_iterations=max_iterations,
            regularization=regularization,
            noise_level=noise_level,
        ),
        solve_kwargs={},
        method_name="Lanczos",
        extra_output={
            "regularization_method": regularization_method,
            "regularization": float(regularization),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
