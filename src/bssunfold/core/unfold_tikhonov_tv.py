"""Noise-constrained Tikhonov-TV unfolding method with adaptive balancing.

This module provides a 1D adaptation of the ``automatic_Tikhonov_TV``
method of Silvia Gazzola and Ali Gholami for neutron spectrum unfolding.
The method solves

    min f(m)  subject to  ||A m - b||^2 = epsilon,

where the regularizer ``f`` is one of:

    [TT] : f(m) = ||D1 m||_1 + beta/2 * ||D1_bar g2||_2^2
    [TV] : f(m) = ||D1 m||_1
    [T]  : f(m) = beta/2 * ||D1_bar g2||_2^2

Here ``D1`` is a finite-difference discretization of the first derivative
operator (total variation) and ``D1_bar`` a discretization of the second
derivative.  The balancing parameter ``beta`` between the TV and Tikhonov
terms can be estimated adaptively, and the noise constraint is enforced
through an augmented Lagrangian / ADMM scheme.

The original MATLAB implementation by Silvia Gazzola (University of Bath)
and Ali Gholami (University of Tehran) targets 2D image problems; this
module adapts the ADMM scheme to the 1D Bonner-sphere spectrum unfolding
problem.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_tikhonov_tv", "unfold_tikhonov_tv"]


def _first_derivative(n: int) -> np.ndarray:
    """First-derivative (total-variation) operator, shape (n-1, n)."""
    return create_derivative_matrix(n, 1).toarray()


def _second_derivative(n: int) -> np.ndarray:
    """Second-derivative operator, shape (n-2, n)."""
    return create_derivative_matrix(n, 2).toarray()


def _zscore_max(p: np.ndarray, a: float = 2.5) -> float:
    """Return the maximum of the normal components of a 1D array.

    Mirrors the ``zscore`` helper of the original MATLAB code: sorts the
    absolute values, keeps the upper half, computes a robust z-score with
    respect to the median and the mean absolute deviation, and returns the
    maximum value whose z-score stays below the threshold ``a``.

    Parameters
    ----------
    p : np.ndarray
        1D array whose maximum of the normal components should be computed.
    a : float, optional
        Threshold for defining the normal components (default: 2.5).

    Returns
    -------
    float
        Maximum of the normal components of ``p``.
    """
    p = np.sort(np.abs(p))
    if len(p) == 0:
        return 0.0
    p = p[len(p) // 2:]
    mad = 1.4826 * (np.mean(np.abs(p - np.mean(p))) + np.finfo(float).eps)
    med = float(np.median(p))
    z = (p - med) / mad
    idx = np.where(np.abs(z) < a)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(p[idx]))


def _gamma_from_cubic(pp: float, qq: float) -> float:
    """Return the largest real root of ``gamma^3 + pp*gamma + qq = 0``."""
    coeffs = [1.0, 0.0, pp, qq]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        return 0.0
    return float(np.max(real_roots))


def solve_tikhonov_tv(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    mu: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_iterations: int = 100,
    type_: str = "TT",
    beta: float = 1.0,
    zthr: float = 2.5,
    tolerance: float = 1e-4,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the noise-constrained Tikhonov-TV unfolding problem.

    Solves ``min f(m)`` subject to ``||A m - b||^2 = epsilon`` with the
    ADMM scheme of Gazzola & Gholami adapted to 1D spectra.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess (accepted for API compatibility; the
        iteration always starts from a zero vector like the original).
    epsilon : float, optional
        (Estimate of) the squared 2-norm of the noise.  If None, derived
        from the residuals of an unregularized least-squares solve.
    mu : tuple, optional
        Penalty parameters for the Lagrangian terms, ``(mu1, mu2, mu3)``
        (default: (1.0, 1.0, 1.0)).
    max_iterations : int, optional
        Maximum number of ADMM iterations (default: 100).
    type_ : str, optional
        Optimization problem to be solved: ``'TT'`` (TV + Tikhonov),
        ``'TV'`` (pure total variation) or ``'T'`` (pure Tikhonov)
        (default: 'TT').
    beta : float, optional
        Balancing parameter between the TV and Tikhonov terms.  A scalar
        fixes its value; ``'adapt'`` estimates it adaptively (only for
        ``type_='TT'``) (default: 1.0).
    zthr : float, optional
        Threshold used by the adaptive beta estimation (default: 2.5).
    tolerance : float, optional
        Stopping criterion based on the relative change of the solution
        (default: 1e-4).

    Returns
    -------
    tuple
        ``(spectrum, iterations, converged)`` where ``converged`` reports
        whether the stabilization stopping criterion was reached.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n = A.shape

    if type_ not in ("TT", "TV", "T"):
        raise ValueError(
            f"Unsupported type_: {type_}. Choose from 'TT', 'TV', 'T'."
        )

    adapt_beta = isinstance(beta, str) and beta == "adapt"

    mu1 = float(mu[0])
    mu2 = float(mu[1])
    mu3 = float(mu[2])

    D1 = _first_derivative(n)  # (n-1, n)
    D1_bar = _second_derivative(n - 1)  # (n-3, n-1), smooths gradient rows

    if epsilon is None:
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        epsilon = float(np.linalg.norm(b - A @ x_ls) ** 2)
    epsilon = float(epsilon)
    if epsilon <= 0:
        epsilon = 1e-12

    # Fixed matrix for the m-subproblem (eq. (2.11) of the paper).
    B = mu1 * (D1.T @ D1) + mu2 * (A.T @ A)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)

    # Initial guesses for the primal and dual variables.
    m = np.zeros(n)
    g1 = np.zeros(n - 1)
    g2 = np.zeros(n - 1)
    e = np.zeros(m_len)
    lambda_1 = np.zeros(n - 1)
    lambda_2 = np.zeros(m_len)
    lambda_3 = 0.0

    beta_k = float(beta) if not adapt_beta else 1.0
    if type_ == "TV":
        beta_k = 0.0

    converged = False
    m_prev = None
    stopit = max_iterations

    for k in range(1, max_iterations + 1):
        # --------------------------- m-subproblem ---------------------------
        rhs = (
            mu1 * D1.T @ (g1 + g2 + lambda_1)
            + mu2 * A.T @ (b + e + lambda_2)
        )
        m = B_inv @ rhs

        if k == 1:
            m_prev = m
        else:
            norm_prev = float(np.linalg.norm(m_prev))
            if norm_prev > 0:
                diffm = float(np.linalg.norm(m - m_prev) / norm_prev)
            else:
                diffm = float(np.linalg.norm(m))
            m_prev = m
            if diffm < tolerance and not converged:
                stopit = k
                converged = True
                break

        # --------------------------- g1-subproblem ---------------------------
        if type_ in ("TT", "TV"):
            y1 = D1 @ m - g2 - lambda_1
            g1 = np.sign(y1) * np.maximum(np.abs(y1) - 1.0 / mu1, 0.0)

        # --------------------------- g2-subproblem ---------------------------
        if type_ in ("TT", "T"):
            y2 = D1 @ m - g1 - lambda_1
            if beta_k > 0 and mu1 > 0:
                lhs = (
                    np.eye(n - 1)
                    + (beta_k / mu1) * (D1_bar.T @ D1_bar)
                )
                g2 = np.linalg.solve(lhs, y2)
            else:
                g2 = y2

        # ---------------------------- e-subproblem ----------------------------
        y = A @ m - b - lambda_2
        E = float(np.dot(y, y))
        if E > 0:
            pp = (mu2 - 2 * mu3 * (epsilon + lambda_3)) / (2 * mu3 * E)
            qq = -mu2 / (2 * mu3 * E)
            gamma = _gamma_from_cubic(pp, qq)
        else:
            gamma = 0.0
        e = gamma * y

        # --------------------------- dual updates ----------------------------
        lambda_1 = lambda_1 + g1 + g2 - D1 @ m
        lambda_2 = lambda_2 + b + e - A @ m
        lambda_3 = lambda_3 + epsilon - float(np.dot(e, e))

        # -------------------------- beta-update ------------------------------
        if adapt_beta and type_ == "TT":
            g = D1 @ m
            target = _zscore_max(g, zthr)
            value = float(np.max(np.abs(g2))) if len(g2) else 0.0
            denom = value + target
            if denom > 0:
                beta_k = 2.0 * value / denom * beta_k
            else:
                beta_k = beta_k
        elif type_ == "T" and not adapt_beta:
            beta_k = float(beta)

    spectrum = np.maximum(m, 0)
    return spectrum, stopit, converged


def unfold_tikhonov_tv(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    mu: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_iterations: int = 100,
    type_: str = "TT",
    beta: float = 1.0,
    zthr: float = 2.5,
    tolerance: float = 1e-4,
    noise_level: Optional[float] = None,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum with noise-constrained Tikhonov-TV.

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
    epsilon : float, optional
        (Estimate of) the squared 2-norm of the noise.  If None, derived
        from ``noise_level`` (``(noise_level * ||b||)^2``) or from the
        residuals of an unregularized least-squares solve.
    mu : tuple, optional
        Penalty parameters ``(mu1, mu2, mu3)`` (default: (1.0, 1.0, 1.0)).
    max_iterations : int, optional
        Maximum number of ADMM iterations (default: 100).
    type_ : str, optional
        Optimization problem: ``'TT'``, ``'TV'`` or ``'T'`` (default: 'TT').
    beta : float, optional
        Balancing parameter between TV and Tikhonov terms, or ``'adapt'``
        for adaptive estimation (default: 1.0).
    zthr : float, optional
        Threshold for the adaptive beta estimation (default: 2.5).
    tolerance : float, optional
        Stabilization stopping criterion (default: 1e-4).
    noise_level : float, optional
        Relative noise level used to derive a default ``epsilon``.
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

    # Derive a default epsilon from the noise level if not provided.
    if epsilon is None and noise_level is not None:
        selected = [name for name in detector_names if name in readings]
        b_norm = float(
            np.linalg.norm([readings[name] for name in selected])
        )
        epsilon = (noise_level * b_norm) ** 2

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
            solve_tikhonov_tv,
            epsilon=epsilon,
            mu=mu,
            max_iterations=max_iterations,
            type_=type_,
            beta=beta,
            zthr=zthr,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="TikhonovTV",
        extra_output={
            "type_": type_,
            "epsilon": float(epsilon) if epsilon is not None else None,
            "beta": float(beta) if not isinstance(beta, str) else beta,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
