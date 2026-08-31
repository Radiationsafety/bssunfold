"""Regularization parameter selection module for bssunfold package.

This module provides methods for selecting optimal regularization parameters
using various heuristics: L-curve, GCV, Discrepancy Principle.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._matrix_utils import compute_svd_components

__all__ = [
    "select_regularization_parameter",
    "lcurve_selection",
    "gcv_selection",
    "discrepancy_principle_selection",
    "cosine_similarity_selection",
    "quasi_optimality_selection",
    "ncp_selection",
    "snr_criterion_selection",
    "weighted_gcv_poisson_selection",
    "kfold_cv_selection",
    "compare_regularization_methods",
    "randomization_experiment",
]


def _estimate_noise_variance(
    A: np.ndarray,
    b: np.ndarray,
) -> float:
    """Estimate noise variance from least squares residual.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.

    Returns
    -------
    float
        Estimated noise variance.
    """
    x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual = b - A @ x_ls
    return float(np.var(residual))


def select_regularization_parameter(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "lcurve",
    noise_var: Optional[float] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Select regularization parameter using specified method.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    method : str, optional
        Selection method: 'lcurve', 'gcv', 'dp', 'cosine' (default: 'lcurve').
    noise_var : float, optional
        Noise variance for discrepancy principle.
    initial_spectrum : np.ndarray, optional
        Initial spectrum for cosine similarity method.
    **kwargs : dict
        Additional method-specific arguments.

    Returns
    -------
    float
        Selected regularization parameter (lambda).

    Raises
    ------
    ValueError
        If method is unknown or selection fails.
    """
    if method == "lcurve":
        return lcurve_selection(A, b, **kwargs)
    if method == "gcv":
        return gcv_selection(A, b, **kwargs)
    if method == "dp":
        return discrepancy_principle_selection(
            A, b, noise_var=noise_var, **kwargs
        )
    if method == "cosine":
        return cosine_similarity_selection(A, b, initial_spectrum, **kwargs)
    if method == "quasi_optimality":
        return quasi_optimality_selection(A, b, **kwargs)
    if method == "ncp":
        return ncp_selection(A, b, **kwargs)
    if method == "snr":
        return snr_criterion_selection(A, b, **kwargs)
    if method == "weighted_gcv_poisson":
        return weighted_gcv_poisson_selection(A, b, **kwargs)
    if method == "kfold_cv":
        return kfold_cv_selection(A, b, **kwargs)
    raise ValueError(
        f"Unknown regularization selection method: {method}. "
        "Choose from 'lcurve', 'gcv', 'dp', 'cosine', 'quasi_optimality', "
        "'ncp', 'snr', 'weighted_gcv_poisson', 'kfold_cv'."
    )


def lcurve_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using L-curve corner heuristic.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk

        n = A.shape[1]
        L = np.eye(n)

        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.lcorner(fam)
        lam = result.get("opt_lambdah")

        if lam is None:
            raise ValueError("L-curve heuristic did not return lambda.")

        return float(lam)

    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback L-curve implementation.",
            ImportWarning,
        )
        return _lcurve_fallback(A, b, n_alphas, alpha_range)


def _lcurve_fallback(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback L-curve implementation without pytikhonov."""
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )
    residuals = []
    norms = []

    n = A.shape[1]
    L = np.eye(n)
    ATA = A.T @ A
    ATb = A.T @ b
    LTL = L.T @ L

    for alpha in alphas:
        P = ATA + alpha * LTL

        try:
            x = np.linalg.solve(P, ATb)
            x = np.maximum(x, 0)  # Non-negativity
            residual = np.linalg.norm(A @ x - b)
            norm_val = np.linalg.norm(L @ x)
            residuals.append(residual)
            norms.append(norm_val)
        except np.linalg.LinAlgError:
            continue

    if len(residuals) < 3:
        return 1.0  # Default value

    # Find corner using maximum curvature
    log_res = np.log(residuals)
    log_norm = np.log(norms)

    # Simple corner detection: point with maximum distance from line
    # connecting endpoints
    p1 = np.array([log_res[0], log_norm[0]])
    p2 = np.array([log_res[-1], log_norm[-1]])

    distances = []
    for i in range(len(residuals)):
        p = np.array([log_res[i], log_norm[i]])
        # Distance from point to line
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)

    idx_max = np.argmax(distances)
    return float(alphas[idx_max])


def gcv_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using Generalized Cross Validation.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk

        n = A.shape[1]
        L = np.eye(n)

        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.gcvmin(fam)
        lam = result.get("opt_lambdah")

        if lam is None:
            raise ValueError("GCV minimization did not return lambda.")

        return float(lam)

    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback GCV implementation.",
            ImportWarning,
        )
        return _gcv_fallback(A, b, n_alphas, alpha_range)


def _gcv_fallback(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback GCV implementation without pytikhonov.

    Uses precomputed SVD for efficiency: SVD is computed once and reused
    across all alpha values instead of solving a linear system per alpha.
    """
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )
    m, _ = A.shape

    # Precompute SVD once
    U, _, _, s_sq = compute_svd_components(A)
    UTb = U.T @ b  # Precompute projection of b onto left singular vectors

    gcv_values = []

    for alpha in alphas:
        # GCV(alpha) = ||A x_alpha - b||^2 / (m - trace(A A^+_alpha))^2
        # For Tikhonov: x_alpha = V diag(s/(s^2+alpha)) U^T b
        # residual = b - A x_alpha = U diag(alpha/(s^2+alpha)) U^T b
        # trace = sum(s^2 / (s^2 + alpha))

        filt = s_sq / (s_sq + alpha)  # filter factors
        residual_coeff = alpha / (s_sq + alpha)  # residual coefficients

        # ||residual||^2 = sum((residual_coeff * UTb)^2)
        residual_sq = np.sum((residual_coeff * UTb) ** 2)
        trace_term = np.sum(filt)

        gcv = residual_sq / (m - trace_term) ** 2
        gcv_values.append(gcv)

    if not gcv_values or all(v == np.inf for v in gcv_values):
        return 1.0  # Default value

    idx_min = np.argmin(gcv_values)
    return float(alphas[idx_min])


def discrepancy_principle_selection(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using Discrepancy Principle.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    noise_var : float, optional
        Noise variance. If None, estimated from data.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk

        n = A.shape[1]
        L = np.eye(n)

        if noise_var is None:
            noise_var = _estimate_noise_variance(A, b)

        delta = np.sqrt(noise_var)

        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.discrepancy_principle(fam, delta=delta)
        lam = result.get("opt_lambdah")

        if lam is None:
            raise ValueError("Discrepancy principle did not return lambda.")

        return float(lam)

    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback DP implementation.",
            ImportWarning,
        )
        if noise_var is None:
            noise_var = _estimate_noise_variance(A, b)
        return _dp_fallback(A, b, noise_var, n_alphas, alpha_range)


def _dp_fallback(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: float,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback Discrepancy Principle implementation."""
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )
    delta = np.sqrt(noise_var)
    m = len(b)
    target_residual = delta * np.sqrt(m)

    n = A.shape[1]
    ATA = A.T @ A
    ATb = A.T @ b

    residuals = []

    for alpha in alphas:
        P = ATA + alpha * np.eye(n)

        try:
            x = np.linalg.solve(P, ATb)
            x = np.maximum(x, 0)
            residual = np.linalg.norm(A @ x - b)
            residuals.append(residual)
        except np.linalg.LinAlgError:
            residuals.append(np.inf)

    # Find alpha where residual is closest to target
    residuals = np.array(residuals)
    idx = np.argmin(np.abs(residuals - target_residual))

    return float(alphas[idx])


def cosine_similarity_selection(
    A: np.ndarray,
    b: np.ndarray,
    initial_spectrum: np.ndarray,
    n_alphas: int = 100,
    alpha_range: Tuple[float, float] = (-9, 2),
    norm: int = 2,
) -> float:
    """Select regularization parameter by maximizing cosine similarity.

    Uses precomputed SVD for efficient evaluation across alpha values.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    initial_spectrum : np.ndarray
        Initial/reference spectrum for similarity comparison.
    n_alphas : int, optional
        Number of alpha values to test (default: 100).
    alpha_range : Tuple[float, float], optional
        Log range of alpha values (default: (-9, 2)).
    norm : int, optional
        Norm type for regularization (default: 2).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    similarities = []

    # Normalize initial spectrum
    norm_init = np.linalg.norm(initial_spectrum)
    if norm_init == 0:
        raise ValueError("Initial spectrum has zero norm.")
    initial_normalized = initial_spectrum / norm_init

    # Precompute SVD for efficient Tikhonov solutions
    U, s, Vt, s_sq = compute_svd_components(A)
    UTb = U.T @ b

    for alpha in alphas:
        # Tikhonov solution via SVD: x = V diag(s/(s^2+alpha)) U^T b
        filt = s / (s_sq + alpha)
        x = Vt.T @ (filt * UTb)
        x = np.maximum(x, 0)

        # Compute cosine similarity
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            similarities.append(0.0)
        sim = np.dot(x, initial_normalized) / norm_x
        similarities.append(sim)

    idx_max = np.argmax(similarities)
    return float(alphas[idx_max])


def quasi_optimality_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using the quasi-optimality criterion.

    The quasi-optimality criterion (Hochstenbach & Reichel, 2015) minimises
    the noise component of the Tikhonov solution in the SVD basis:

        Q(alpha) = sum_i  (alpha^2 / (s_i^2 + alpha^2))^2 * (U_i^T b / s_i)^2

    The optimal alpha is the one that minimises Q(alpha).

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )
    m, _ = A.shape

    U, s, Vt, s_sq = compute_svd_components(A)
    UTb = U.T @ b

    quotients = np.zeros_like(s)
    nonzero = s > 0
    quotients[nonzero] = UTb[nonzero] / s[nonzero]

    q_values = []
    for alpha in alphas:
        filt = alpha ** 2 / (s_sq + alpha ** 2) ** 2
        q_val = np.sum(filt * quotients ** 2)
        q_values.append(q_val)

    idx_min = np.argmin(q_values)
    return float(alphas[idx_min])


def ncp_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
    significance: float = 0.05,
) -> float:
    """Select regularization parameter using the Normalized Cumulative Periodogram.

    The NCP criterion tests whether the residuals of the regularised solution
    are consistent with white noise.  For each candidate alpha, the residual
    vector is computed, its periodogram is formed, and a Kolmogorov-Smirnov
    test is applied against the uniform distribution on [0, 1].  The alpha
    with the smallest KS statistic (i.e. whitest residuals) is selected.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).
    significance : float, optional
        Reserved for future use (default: 0.05).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )

    U, s, Vt, s_sq = compute_svd_components(A)
    UTb = U.T @ b

    ks_stats = []
    for alpha in alphas:
        filt = s / (s_sq + alpha)
        x = Vt.T @ (filt * UTb)
        residual = b - A @ np.maximum(x, 0)

        r = np.fft.rfft(residual - np.mean(residual))
        periodogram = np.abs(r) ** 2
        cumul = np.cumsum(periodogram)
        if cumul[-1] > 0:
            cumul = cumul / cumul[-1]
        n_p = len(cumul)
        theoretical = np.linspace(1.0 / n_p, 1.0, n_p)
        ks_stat = float(np.max(np.abs(cumul - theoretical)))
        ks_stats.append(ks_stat)

    idx_min = np.argmin(ks_stats)
    return float(alphas[idx_min])


def snr_criterion_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter by maximising signal-to-noise ratio.

    For each candidate alpha the Tikhonov solution is split into a signal
    component (projection onto the leading singular vectors) and a noise
    component (projection onto the trailing singular vectors).  The SNR is
    defined as the ratio of their squared Frobenius norms.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )

    U, s, Vt, s_sq = compute_svd_components(A)
    UTb = U.T @ b

    snr_values = []
    for alpha in alphas:
        filt = s / (s_sq + alpha)
        x = Vt.T @ (filt * UTb)
        x = np.maximum(x, 0)

        signal_part = A @ x
        noise_part = (b - A @ x)

        signal_energy = float(np.sum(signal_part ** 2))
        noise_energy = float(np.sum(noise_part ** 2))

        if noise_energy > 0:
            snr_values.append(signal_energy / noise_energy)
        else:
            snr_values.append(np.inf)

    finite_mask = np.isfinite(snr_values)
    if not np.any(finite_mask):
        return float(alphas[0])

    idx_max = np.argmax(np.where(finite_mask, snr_values, -np.inf))
    return float(alphas[idx_max])


def weighted_gcv_poisson_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using weighted GCV for Poisson noise.

    Under Poisson noise the variance of each measurement equals its
    expectation.  The weighted GCV replaces the ordinary GCV denominator
    ``||r||^2`` with ``sum(r_i^2 / w_i)`` where ``w_i = max(b_i, 1)`` are
    the Poisson variance estimates, and the trace term is replaced by
    ``tr(W (I - H_alpha))`` with ``W = diag(1/w_i)``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )
    m, n = A.shape

    weights = np.maximum(b, 1.0)
    W = np.diag(1.0 / weights)
    WA = W @ A
    Wb = W @ b

    U, s, Vt, s_sq = compute_svd_components(WA)
    UTWb = U.T @ Wb

    wgcv_values = []
    for alpha in alphas:
        filt = s / (s_sq + alpha)
        x = Vt.T @ (filt * UTWb)
        residual = Wb - WA @ x

        wgcv_num = float(np.sum(residual ** 2))
        trace_term = float(np.sum(filt))
        denom = (m - trace_term) ** 2

        if denom > 0:
            wgcv_values.append(wgcv_num / denom)
        else:
            wgcv_values.append(np.inf)

    finite_mask = np.isfinite(wgcv_values)
    if not np.any(finite_mask):
        return 1.0

    idx_min = np.argmin(np.where(finite_mask, wgcv_values, np.inf))
    return float(alphas[idx_min])


def kfold_cv_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_folds: int = 5,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
    random_state: Optional[int] = None,
) -> float:
    """Select regularization parameter using K-fold cross-validation.

    The data is split into K folds.  For each alpha, the Tikhonov problem
    is solved on K-1 folds and the held-out residual is evaluated.  The
    alpha with the smallest mean held-out prediction error is selected.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_folds : int, optional
        Number of cross-validation folds (default: 5).
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).
    random_state : int, optional
        Random seed for fold assignment (default: None).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    rng = np.random.RandomState(random_state)
    m, n = A.shape

    alphas = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas
    )

    indices = rng.permutation(m)
    fold_sizes = np.full(n_folds, m // n_folds, dtype=int)
    fold_sizes[: m % n_folds] += 1

    folds = []
    current = 0
    for fs in fold_sizes:
        folds.append(indices[current : current + fs])
        current += fs

    cv_errors = np.zeros(len(alphas))

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        A_train, b_train = A[train_idx], b[train_idx]
        A_test, b_test = A[test_idx], b[test_idx]

        ATA_train = A_train.T @ A_train
        ATb_train = A_train.T @ b_train

        for i, alpha in enumerate(alphas):
            try:
                x = np.linalg.solve(ATA_train + alpha * np.eye(n), ATb_train)
                x = np.maximum(x, 0)
                residual = b_test - A_test @ x
                cv_errors[i] += float(np.sum(residual ** 2))
            except np.linalg.LinAlgError:
                cv_errors[i] += np.inf

    idx_min = np.argmin(cv_errors)
    return float(alphas[idx_min])


def resolve_regularization_parameter(
    A: np.ndarray,
    b: np.ndarray,
    regularization_method: str,
    regularization: float,
    n_energy_bins: int,
    initial_spectrum: Optional[np.ndarray] = None,
    norm: int = 2,
    noise_var: Optional[float] = None,
    verbose: bool = True,
) -> float:
    """Resolve the regularization parameter alpha from the requested method.

    Shared by the solver wrappers (qpsolvers, docplex, SCIP). Handles
    ``'manual'``, ``'cosine'`` and automatic methods (``'lcurve'``, ``'gcv'``,
    ``'dp'``). Returns the selected lambda, equal to ``regularization`` for
    the manual method.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    regularization_method : str
        One of ``'manual'``, ``'cosine'``, ``'lcurve'``, ``'gcv'``, ``'dp'``.
    regularization : float
        Manual regularization parameter (used when method is ``'manual'``).
    n_energy_bins : int
        Number of energy bins (validates the initial spectrum length).
    initial_spectrum : np.ndarray, optional
        Reference spectrum required by the cosine method.
    norm : int, optional
        Norm type (default: 2).
    noise_var : float, optional
        Noise variance for the discrepancy principle (default: None).
    verbose : bool, optional
        Print the selected value (default: True).

    Returns
    -------
    float
        Selected regularization parameter.
    """
    if regularization_method == "manual":
        selected_lambda = float(regularization)
    elif regularization_method == "cosine":
        if initial_spectrum is None:
            raise ValueError(
                "For 'cosine' regularization method, "
                "initial_spectrum must be provided."
            )
        if norm != 2:
            warnings.warn(
                f"Cosine regularization selection method assumes L2 "
                f"norm, but norm={norm} was requested. Using L2 for "
                f"selection."
            )
        initial_spectrum_norm = np.maximum(initial_spectrum, 0)
        if len(initial_spectrum_norm) != n_energy_bins:
            raise ValueError(
                f"Initial spectrum length ({len(initial_spectrum)}) "
                f"must match number of energy bins ({n_energy_bins})"
            )
        selected_lambda = cosine_similarity_selection(
            A, b, initial_spectrum_norm, norm=norm
        )
        if verbose:
            print(
                f"Selected regularization (method=cosine): "
                f"{selected_lambda:.3e}"
            )
    else:
        if norm != 2:
            warnings.warn(
                f"Automatic regularization selection methods assume L2 "
                f"norm, but norm={norm} was requested. Using L2 for "
                f"selection."
            )
        try:
            selected_lambda = select_regularization_parameter(
                A, b, method=regularization_method, noise_var=noise_var
            )
        except Exception as e:
            raise ValueError(
                f"Regularization selection failed: {e}. "
                "Consider using manual regularization."
            ) from e
        if verbose:
            print(
                f"Selected regularization (method={regularization_method}): "
                f"{selected_lambda:.3e}"
            )
    return float(selected_lambda)


def compare_regularization_methods(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    plot: bool = False,
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare regularization selection methods for given system.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    noise_var : float, optional
        Noise variance for discrepancy principle.
        If None, estimated from residual of unregularized solution.
    plot : bool, optional
        If True, generate comparison plot.
    plot_path : str, optional
        Path to save the plot.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'lcurve': dict from lcorner()
        - 'dp': dict from discrepancy_principle()
        - 'gcv': dict from gcvmin()
        - 'all_data': dict from pytikhonov.all_regparam_methods()
        - 'selected': dict mapping method name to selected lambda.

    Raises
    ------
    ImportError
        If pytikhonov is not available.
    """
    try:
        import pytikhonov as ptk
    except ImportError as e:
        raise ImportError(
            "pytikhonov is required for compare_regularization_methods. "
            "Install with: pip install pytikhonov"
        ) from e

    n = A.shape[1]
    L = np.eye(n)
    fam = ptk.TikhonovFamily(A, L, b)

    # Compute each method
    lc_res = ptk.lcorner(fam)
    if noise_var is None:
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        noise_var = np.var(b - A @ x_ls)
    delta = np.sqrt(noise_var)
    dp_res = ptk.discrepancy_principle(fam, delta=delta)
    gcv_res = ptk.gcvmin(fam)
    all_data = ptk.all_regparam_methods(fam)

    selected = {
        "lcurve": lc_res.get("opt_lambdah"),
        "dp": dp_res.get("opt_lambdah"),
        "gcv": gcv_res.get("opt_lambdah"),
    }

    if plot:
        ptk.plot_all_methods(all_data, plot_path=plot_path)

    return {
        "lcurve": lc_res,
        "dp": dp_res,
        "gcv": gcv_res,
        "all_data": all_data,
        "selected": selected,
    }


def randomization_experiment(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    n_samples: int = 10,
    rseed: int = 0,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run randomization experiments for regularization parameter selection.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    noise_var : float, optional
        Noise variance for generating perturbed measurements.
        If None, estimated from residual of unregularized solution.
    n_samples : int, optional
        Number of random samples for each method, default 10.
    rseed : int, optional
        Random seed for reproducibility, default 0.
    methods : list of str, optional
        List of methods to run: 'lcurve', 'dp', 'gcv', 'lcurve_full'.
        If None, runs all four.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys for each method, each containing:
        - 'lambdas': array of selected lambdas per sample.
        - 'mean': mean of lambdas.
        - 'std': standard deviation.
        - 'median': median.
        - 'min', 'max': range.
        - 'cv': coefficient of variation (std/mean).
        - 'raw_result': raw output from pytikhonov function.

    Raises
    ------
    ImportError
        If pytikhonov is not available.
    """
    try:
        import pytikhonov as ptk
    except ImportError as e:
        raise ImportError(
            "pytikhonov is required for randomization_experiment. "
            "Install with: pip install pytikhonov"
        ) from e

    n = A.shape[1]
    L = np.eye(n)

    # Estimate noise variance if not provided
    if noise_var is None:
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        noise_var = np.var(b - A @ x_ls)

    # Create TikhonovFamily with btrue = b (assumed true signal)
    fam = ptk.TikhonovFamily(A, L, b, btrue=b, noise_var=noise_var)

    if methods is None:
        methods = ["lcurve", "dp", "gcv", "lcurve_full"]

    results = {}
    for method in methods:
        if method == "lcurve":
            raw = ptk.rand_lcorner(fam, n_samples=n_samples, rseed=rseed)
            lambdas = np.array(raw[0])  # first element is list of lambdas
        elif method == "dp":
            raw = ptk.rand_discrepancy_principle(
                fam, n_samples=n_samples, tau=1.01, rseed=rseed
            )
            lambdas = np.array(raw[0])
        elif method == "gcv":
            raw = ptk.rand_gcvmin(fam, n_samples=n_samples, rseed=rseed)
            lambdas = np.array(raw[0])
        elif method == "lcurve_full":
            raw = ptk.rand_lcurve(
                fam, lambdahs=None, n_samples=n_samples, rseed=rseed
            )
            lambdas = np.array(raw[0])
        else:
            warnings.warn(f"Unknown method: {method}. Skipping.")
            continue

        # Compute statistics
        mean = float(np.mean(lambdas))
        std = float(np.std(lambdas))
        median = float(np.median(lambdas))
        min_val = float(np.min(lambdas))
        max_val = float(np.max(lambdas))
        cv = std / mean if mean != 0 else np.inf

        results[method] = {
            "lambdas": lambdas,
            "mean": mean,
            "std": std,
            "median": median,
            "min": min_val,
            "max": max_val,
            "cv": cv,
            "raw_result": raw,
        }

    return results
