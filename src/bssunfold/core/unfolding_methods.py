"""Unfolding methods for neutron spectrum reconstruction.

This module provides standalone functions for various spectrum unfolding
algorithms that can be used independently of the Detector class.
"""

import numpy as np
from typing import Optional, Tuple

import warnings

from scipy.sparse import csc_matrix, diags
from ._matrix_utils import create_derivative_matrix

__all__ = [
    "solve_cvxpy",
    "solve_landweber",
    "solve_mlem",
    "solve_qpsolvers",
    "solve_doroshenko",
    "solve_kaczmarz",
    "solve_lmfit",
]


def solve_cvxpy(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    solver: str = "ECOS",
) -> np.ndarray:
    """Solve unfolding problem using cvxpy.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    norm : int, optional
        Norm type (1 for L1, 2 for L2).
    solver : str, optional
        CVXPY solver name.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for unfold_cvxpy. Install with: pip install cvxpy"
        ) from e

    n = A.shape[1]
    x = cp.Variable(n, nonneg=True)

    objective = cp.Minimize(
        cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm)
    )
    problem = cp.Problem(objective)

    try:
        if solver == "ECOS":
            problem.solve(solver=cp.ECOS)
        else:
            problem.solve()

        status = problem.status
        if status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(
                f"Problem status is not optimal: {status}. "
                "Solution may be inaccurate."
            )

        if x.value is None:
            warnings.warn(
                f"Solution variable is None. Status: {status}. "
                "Returning zero vector."
            )
            return np.zeros(n)

        return np.asarray(x.value)

    except Exception as e:
        warnings.warn(f"CVXPY solving failed: {e}. Returning zero vector.")
        return np.zeros(n)


def solve_landweber(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Landweber iteration.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    x = x0.copy()
    sigma_max = np.linalg.norm(A, 2)

    if sigma_max == 0:
        warnings.warn("Response matrix has zero norm. Returning initial guess.")
        return x, 0, False

    step_size = 1.0 / (sigma_max ** 2)
    AT = A.T

    converged = False
    iterations = 0

    for i in range(max_iterations):
        residual = A @ x - b
        residual_norm = np.linalg.norm(residual)

        if residual_norm < tolerance:
            converged = True
            iterations = i
            break

        x = x - step_size * (AT @ residual)
        x = np.maximum(x, 0)  # Non-negativity

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


def solve_mlem(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using MLEM (Maximum Likelihood Expectation Maximization).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    x = np.maximum(x0.copy(), 1e-10)  # Ensure positive initial values

    # Precompute A^T for efficiency
    AT = A.T

    converged = False
    iterations = 0

    for i in range(max_iterations):
        # Forward projection
        Ax = A @ x

        # Avoid division by zero
        Ax = np.maximum(Ax, 1e-10)

        # MLEM update
        ratio = b / Ax
        correction = AT @ ratio
        x_new = x * correction

        # Check convergence
        diff = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)

        x = np.maximum(x_new, 0)  # Non-negativity
        iterations = i + 1

        if diff < tolerance:
            converged = True
            break

    return x, iterations, converged


def solve_qpsolvers(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    solver: str = "osqp",
    x_init: Optional[np.ndarray] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
) -> Optional[np.ndarray]:
    """Solve unfolding problem using qpsolvers.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    norm : int, optional
        Norm type (1 for L1, 2 for L2).
    solver : str, optional
        QP solver name (default: 'osqp').
    x_init : np.ndarray, optional
        Initial values.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2).
    smoothness_weight : float, optional
        Weight for smoothness term.

    Returns
    -------
    Optional[np.ndarray]
        Unfolded spectrum or None if solving failed.
    """
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as e:
        raise ImportError(
            "qpsolvers is required for unfold_qpsolvers. "
            "Install with: pip install qpsolvers"
        ) from e

    # Check solver availability
    if solver not in available_solvers:
        if "osqp" in available_solvers:
            solver = "osqp"
        elif "ecos" in available_solvers:
            solver = "ecos"
        else:
            warnings.warn(
                f"Solver '{solver}' not available. Available: {available_solvers}"
            )
            return None

    n = A.shape[1]

    # Base QP: min 0.5 * ||Ax - b||^2 = 0.5 * x^T (A^T A) x - (A^T b)^T x
    # Create P as csc_matrix for efficiency with osqp
    P_base = csc_matrix(A.T @ A)
    q_base = -A.T @ b

    if norm == 2:
        # L2 regularization with optional smoothness
        P = P_base.copy()

        if smoothness_order == 1:
            L = create_derivative_matrix(n, 1)
            P += alpha * smoothness_weight * (L.T @ L)
        elif smoothness_order == 2:
            L = create_derivative_matrix(n, 2)
            P += alpha * smoothness_weight * (L.T @ L)
        else:
            # Standard Tikhonov - use sparse identity for efficiency
            P += alpha * csc_matrix(np.eye(n))

        # Constraints: x >= 0 - create as csc_matrix
        G = csc_matrix(-np.eye(n))
        h = np.zeros(n)

        x = solve_qp(
            P=P,
            q=q_base,
            G=G,
            h=h,
            solver=solver,
            initvals=x_init,
            verbose=False,
        )

    elif norm == 1:
        # L1 regularization via variable extension (x = x_plus - x_minus)
        # |x| = sum(x_plus + x_minus) where x_plus >= 0, x_minus >= 0
        n_ext = 2 * n

        # Extended P matrix: [P_base  0; 0 0] - use sparse block construction
        P_ext = csc_matrix((n_ext, n_ext))
        P_ext[:n, :n] = P_base

        # Add smoothness if needed
        if smoothness_order == 1:
            L = create_derivative_matrix(n, 1)
            P_ext[:n, :n] += alpha * smoothness_weight * (L.T @ L)
        elif smoothness_order == 2:
            L = create_derivative_matrix(n, 2)
            P_ext[:n, :n] += alpha * smoothness_weight * (L.T @ L)

        # Extended q vector: [q_base; alpha * ones(n)]
        q_ext = np.zeros(n_ext)
        q_ext[:n] = q_base
        q_ext[n:] = alpha * np.ones(n)

        # Constraints:
        # x_plus >= 0, x_minus >= 0
        # x = x_plus - x_minus, so x_plus - x_minus >= 0 (non-negativity)
        G_ext = csc_matrix((3 * n, n_ext))
        h_ext = np.zeros(3 * n)

        # x_plus >= 0: -x_plus <= 0
        G_ext[:n, :n] = -csc_matrix(np.eye(n))
        # x_minus >= 0: -x_minus <= 0
        G_ext[n:2 * n, n:] = -csc_matrix(np.eye(n))
        # x_plus - x_minus >= 0: -(x_plus - x_minus) <= 0
        G_ext[2 * n:3 * n, :n] = -csc_matrix(np.eye(n))
        G_ext[2 * n:3 * n, n:] = csc_matrix(np.eye(n))

        # Initial values for extended problem
        x_init_ext = None
        if x_init is not None:
            x_init_ext = np.concatenate([x_init, np.abs(x_init)])

        x_ext = solve_qp(
            P=P_ext,
            q=q_ext,
            G=G_ext,
            h=h_ext,
            solver=solver,
            initvals=x_init_ext,
            verbose=False,
        )

        if x_ext is None:
            warnings.warn(f"Solver '{solver}' did not find a solution for L1 problem.")
            return None

        # Extract x = x_plus - x_minus
        x = x_ext[:n] - x_ext[n:]
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

    if x is None:
        warnings.warn(f"Solver '{solver}' did not find a solution.")
        return None

    return np.asarray(x)


def solve_doroshenko(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization: float = 0.0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Doroshenko coordinate update method.

    Uses incremental residual update for O(n) per-coordinate complexity
    instead of O(n^2) from full matrix-vector products.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    regularization : float, optional
        Regularization strength to prevent division by zero (default: 0.0).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    x = x0.copy()

    # Precompute denominators for each coordinate (sum of squares of column A[:,j])
    denominator_cache = np.sum(A * A, axis=0) + regularization

    # Precompute initial residual: r = b - A @ x
    residual = b - A @ x

    converged = False
    iterations = 0

    for i in range(max_iterations):
        x_old = x.copy()

        # Update each coordinate sequentially with incremental residual
        for j in range(x.size):
            if denominator_cache[j] <= 0:
                continue

            # Current contribution of column j to the residual
            Aj = A[:, j]
            old_xj = x[j]

            # Compute optimal update for coordinate j
            numerator = np.dot(Aj, residual) + denominator_cache[j] * old_xj
            new_xj = max(0.0, numerator / denominator_cache[j])

            # Update residual incrementally: r = r - Aj * (new_xj - old_xj)
            delta = new_xj - old_xj
            if delta != 0:
                residual -= delta * Aj
                x[j] = new_xj

        # Check convergence based on change in solution
        if np.linalg.norm(x - x_old) < tolerance:
            converged = True
            iterations = i + 1
            break

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


def solve_kaczmarz(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    omega: float = 1.0,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Kaczmarz algorithm (ART).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    omega : float, optional
        Relaxation parameter (0 < omega <= 2), default: 1.0.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    m, n = A.shape
    x = x0.copy()

    # Validate relaxation parameter
    if omega <= 0 or omega > 2:
        warnings.warn(f"omega={omega} outside recommended range (0,2]")

    # Precompute squared norms of rows for efficiency
    row_norms_sq = np.sum(A * A, axis=1)

    converged = False
    iterations = 0
    x_old = x.copy()

    for k in range(max_iterations):
        i = k % m  # Cyclic access pattern

        # Skip rows with zero norm
        if row_norms_sq[i] > 0:
            # Compute update
            update = (b[i] - np.dot(A[i], x)) / row_norms_sq[i]
            x = x + omega * update * A[i]

            # Apply non-negativity constraint
            x = np.maximum(x, 0)

        # Check convergence after each full cycle
        if (k + 1) % m == 0:
            if np.linalg.norm(x - x_old) < tolerance:
                converged = True
                iterations = k + 1
                break
            x_old = x.copy()

    if not converged:
        iterations = max_iterations

    return x, iterations, converged


# ---------------------------------------------------------------------------
# Residual functions for lmfit (defined at module level for testability)
# ---------------------------------------------------------------------------


def _residual_lasso(params, A, b, regularization, method, m):
    """Lasso (L1) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * np.sqrt(m) * x
        return np.concatenate([residual, reg_residual])
    return np.sum(residual ** 2) + regularization * np.sum(np.abs(x))


def _residual_ridge(params, A, b, regularization, method, m):
    """Ridge (L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * x
        return np.concatenate([residual, reg_residual])
    return np.sum(residual ** 2) + regularization * np.sum(x ** 2)


def _residual_elastic(params, A, b, regularization, regularization2, l1_weight, method, m):
    """Elastic net (L1 + L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        l1_residual = (
            np.sqrt(regularization * l1_weight) * np.sqrt(m) * np.abs(x)
        )
        l2_residual = np.sqrt(regularization2 * (1 - l1_weight)) * x
        reg_residual = np.concatenate([l1_residual, l2_residual])
        return np.concatenate([residual, reg_residual])
    l1_penalty = regularization * l1_weight * np.sum(np.abs(x))
    l2_penalty = regularization2 * (1 - l1_weight) * np.sum(x ** 2)
    return np.sum(residual ** 2) + l1_penalty + l2_penalty


_RESIDUAL_MAP = {
    "lasso": (_residual_lasso, ["regularization"]),
    "ridge": (_residual_ridge, ["regularization"]),
    "elastic": (_residual_elastic, ["regularization", "regularization2", "l1_weight"]),
}


def solve_lmfit(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using lmfit with L1/L2/Elastic regularization.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    method : str, optional
        lmfit solver name (leastsq, lbfgsb, etc.), default: "lbfgsb".
    model_name : str, optional
        Regularization model: elastic, lasso, ridge, default: "elastic".
    regularization : float, optional
        L1 regularization strength, default: 1e-4.
    regularization2 : float, optional
        L2 regularization strength for elastic net, default: 1e-4.
    l1_weight : float, optional
        L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5.

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        Tuple of (solution, success, message, nfev).
    """
    try:
        import lmfit
    except ImportError as e:
        raise ImportError(
            "lmfit is required for unfold_lmfit. Install with: pip install lmfit"
        ) from e

    m = A.shape[1]

    # Initialize parameters with initial spectrum
    params = lmfit.Parameters()
    for i in range(m):
        params.add(f"x{i}", value=max(x0[i], 1e-10), min=0.0)

    # Select residual function and arguments
    if model_name not in _RESIDUAL_MAP:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Choose from: elastic, lasso, ridge"
        )

    residual_func, arg_names = _RESIDUAL_MAP[model_name]
    residual_args = {
        "A": A,
        "b": b,
        "method": method,
        "m": m,
    }
    for name in arg_names:
        residual_args[name] = locals()[name]

    # Create minimizer and run optimization
    result = lmfit.minimize(
        residual_func,
        params,
        args=(A, b, regularization, method, m) if model_name in ("lasso", "ridge")
        else (A, b, regularization, regularization2, l1_weight, method, m),
        method=method,
    )

    spectrum = np.array([result.params[f"x{i}"].value for i in range(m)])
    return spectrum, result.success, result.message, result.nfev
