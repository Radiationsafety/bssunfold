"""Shared matrix utility functions for bssunfold core modules.

This module provides common matrix operations used across unfolding methods
and regularization modules, avoiding code duplication.
"""


import numpy as np
from scipy.sparse import csc_matrix, diags


def create_derivative_matrix(
    n: int, 
    order: int, 
    E_MeV: np.ndarray | None = None,
    scale_by_lethargy: bool = True,
) -> csc_matrix:
    """Create finite difference derivative matrix in csc format.
    
    This function implements AUDIT FIX #5: it scales derivative operators
    by the lethargy step to ensure consistent regularization on logarithmic
    energy grids. Without this scaling, the regularization penalty would be
    non-uniform across energy groups.

    Parameters
    ----------
    n : int
        Size of spectrum.
    order : int
        Derivative order (1 or 2).
    E_MeV : np.ndarray, optional
        Energy grid in MeV. If provided and scale_by_lethargy=True, the
        derivative operator is scaled by Δ(ln E) to give proper physical
        derivatives with respect to lethargy.
    scale_by_lethargy : bool, optional
        If True (default), scale the derivative by 1/Δ(ln E) to account for
        logarithmic energy spacing. This ensures the regularization penalty
        is uniform across all energy groups.

    Returns
    -------
    csc_matrix
        Derivative matrix in csc format of shape (n-1, n) for order=1
        or (n-2, n) for order=2.

    Raises
    ------
    ValueError
        If order is not 1 or 2.
    """
    if order == 1:
        # First derivative: [-1, 1] on shifted rows
        data = np.concatenate([[-1] * (n - 1), [1] * (n - 1)])
        row = np.concatenate([np.arange(n - 1), np.arange(n - 1)])
        col = np.concatenate([np.arange(n - 1), np.arange(1, n)])
        L = csc_matrix((data, (row, col)), shape=(n - 1, n))
        
        # Scale by lethargy step if requested (AUDIT FIX #5)
        if scale_by_lethargy and E_MeV is not None and len(E_MeV) == n:
            # Compute Δ(ln E) for each interval
            lnE = np.log(E_MeV + 1e-15)
            dlnE = np.diff(lnE)  # Length n-1
            # Avoid division by zero
            dlnE = np.maximum(np.abs(dlnE), 1e-15)
            # Create diagonal scaling matrix D where (D @ phi)_i = (phi_{i+1} - phi_i) / Δ(ln E)_i
            D = diags(1.0 / dlnE, 0, shape=(n - 1, n - 1), format="csc")
            L = D @ L
        
        return L
    
    if order == 2:
        # Second derivative: [1, -2, 1] on diagonals - more efficient with diags
        L = diags(
            [1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float
        )
        
        # Scale by lethargy step squared if requested (AUDIT FIX #5)
        if scale_by_lethargy and E_MeV is not None and len(E_MeV) == n:
            # Compute Δ(ln E) for central differences
            lnE = np.log(E_MeV + 1e-15)
            # Central difference approximation: use average of adjacent intervals
            dlnE_left = np.diff(lnE)[:-1]  # Length n-2
            dlnE_right = np.diff(lnE)[1:]  # Length n-2
            dlnE_avg = (dlnE_left + dlnE_right) / 2.0
            dlnE_avg = np.maximum(np.abs(dlnE_avg), 1e-15)
            # Scale second derivative by 1/Δ(ln E)^2
            D = diags(1.0 / (dlnE_avg ** 2), 0, shape=(n - 2, n - 2), format="csc")
            L = D @ L
        
        return L

    raise ValueError(f"Unsupported derivative order: {order}. Use 1 or 2.")


def build_smoothness_penalty(
    n: int,
    alpha: float,
    smoothness_order: int,
    smoothness_weight: float = 1.0,
    E_MeV: np.ndarray | None = None,
    scale_by_lethargy: bool = True,
) -> csc_matrix | None:
    """Build the additive smoothness (Tikhonov) penalty matrix.
    
    This function implements AUDIT FIX #5: when scale_by_lethargy=True and
    E_MeV is provided, the derivative operator is scaled by the lethargy step
    to ensure consistent regularization on logarithmic energy grids.

    Returns ``alpha * smoothness_weight * L.T @ L`` for the first/second-order
    finite-difference matrix ``L`` (orders 1 and 2). For ``smoothness_order == 0``
    returns ``None``: zeroth-order Tikhonov is the identity term, which callers
    add separately (and only when their method semantics require it).

    Parameters
    ----------
    n : int
        Spectrum length.
    alpha : float
        Regularization strength.
    smoothness_order : int
        Derivative order (1 or 2); use 0 to request no derivative penalty.
    smoothness_weight : float, optional
        Weight applied to the derivative penalty.
    E_MeV : np.ndarray, optional
        Energy grid in MeV. Used for lethargy scaling of the derivative operator.
    scale_by_lethargy : bool, optional
        If True (default), scale the derivative by 1/Δ(ln E) to account for
        logarithmic energy spacing.

    Returns
    -------
    Optional[csc_matrix]
        Sparse penalty matrix, or ``None`` when ``smoothness_order == 0``.
    """
    if smoothness_order not in (1, 2):
        return None
    L = create_derivative_matrix(
        n, smoothness_order, E_MeV=E_MeV, scale_by_lethargy=scale_by_lethargy
    )
    return alpha * smoothness_weight * (L.T @ L)


def make_regularization_operator(
    n: int,
    smoothness_order: int,
    identity_for_zero: bool = True,
    E_MeV: np.ndarray | None = None,
    scale_by_lethargy: bool = True,
) -> np.ndarray | None:
    """Build the regularization operator L (dense) for a derivative order.
    
    This function implements AUDIT FIX #5: when scale_by_lethargy=True and
    E_MeV is provided, the derivative operator is scaled by the lethargy step
    to ensure consistent regularization on logarithmic energy grids.

    A single source of truth for the repeated ``_make_regoperator`` helpers
    across the iterative solvers (CGLS, GKS, ...).

    Parameters
    ----------
    n : int
        Number of energy bins.
    smoothness_order : int
        Derivative order: 0 (identity or ``None``), 1 (first), 2 (second).
    identity_for_zero : bool, optional
        Control order-0 behaviour. If ``True`` (default) returns
        ``np.eye(n)``; if ``False`` returns ``None``. Solvers that apply the
        operator implicitly (e.g. plain CGLS) pass ``False`` so they can skip
        the regularization term entirely.
    E_MeV : np.ndarray, optional
        Energy grid in MeV. Used for lethargy scaling of the derivative operator.
    scale_by_lethargy : bool, optional
        If True (default), scale the derivative by 1/Δ(ln E) to account for
        logarithmic energy spacing.

    Returns
    -------
    Optional[np.ndarray]
        Operator ``L`` as a dense ndarray, or ``None`` for order 0 when
        ``identity_for_zero`` is ``False``.
    """
    if smoothness_order == 0:
        return np.eye(n) if identity_for_zero else None
    if smoothness_order not in (1, 2):
        raise ValueError(
            f"Unsupported smoothness_order: {smoothness_order}. Use 0, 1 or 2."
        )
    L_sparse = create_derivative_matrix(
        n, smoothness_order, E_MeV=E_MeV, scale_by_lethargy=scale_by_lethargy
    )
    return L_sparse.toarray()


def build_tikhonov_system(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    L: np.ndarray,
) -> np.ndarray:
    """Build and solve a Tikhonov-regularized system: (A^T A + alpha * L^T L) x = A^T b.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    L : np.ndarray
        Regularization matrix (e.g., identity or derivative matrix).

    Returns
    -------
    np.ndarray
        Solution vector x (n,), or None if solving fails.
    """
    try:
        P = A.T @ A + alpha * (L.T @ L)
        x = np.linalg.solve(P, A.T @ b)
        return np.maximum(x, 0)
    except np.linalg.LinAlgError:
        return None


def compute_svd_components(
    A: np.ndarray,
) -> tuple:
    """Compute SVD of A and return components needed for GCV and related computations.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).

    Returns
    -------
    tuple
        (U, s, Vt, s_sq) where s_sq = s**2 for reuse.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt, s**2


def compute_log_steps(E_MeV: np.ndarray, n_energy_bins: int) -> np.ndarray:
    """Compute logarithmic bin-width steps for an energy grid.

    Uses log10(energy + 1e-15) with edge differences at the boundaries and
    central differences for interior points, matching the convention used
    across the parametric unfolding modules.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    n_energy_bins : int
        Number of energy bins.

    Returns
    -------
    np.ndarray
        Log10 bin-width steps of length ``n_energy_bins``.
    """
    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(np.asarray(E_MeV, dtype=float) + 1e-15)

    if n_energy_bins > 1:
        log_steps[0] = log_e[1] - log_e[0]
        log_steps[-1] = log_e[-1] - log_e[-2]
    else:
        log_steps[0] = 1.0

    if n_energy_bins > 2:
        log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0

    return log_steps


def estimate_total_fluence(
    A: np.ndarray,
    b: np.ndarray,
) -> float:
    """Estimate the total fluence ``sum(x)`` of the unfolded spectrum.

    The estimate is obtained from an unconstrained non-negative least-squares
    fit ``min ||A x - b||^2, x >= 0`` and taking ``sum(x_nnls)``.  This is
    data-driven and consistent with the units of the response matrix, unlike
    the crude uniform-response heuristic ``mean(b) / mean(A) * n`` which can
    be orders of magnitude off for log-spaced energy grids.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).

    Returns
    -------
    float
        Positive estimate of ``sum(x)`` for the unfolded spectrum.
    """
    from scipy.optimize import nnls

    m, n = A.shape
    try:
        x_nnls, _ = nnls(A, b, maxiter=10 * n)
        total = float(x_nnls.sum())
        if np.isfinite(total) and total > 0.0:
            return total
    except Exception:
        pass
    # fallback: uniform-response heuristic
    mean_response = max(float(np.mean(A)), 1e-30)
    return float(np.mean(b) / mean_response * n)
