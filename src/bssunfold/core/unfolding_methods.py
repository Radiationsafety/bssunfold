"""Unfolding methods for neutron spectrum reconstruction.

This module provides standalone functions for various spectrum unfolding
algorithms that can be used independently of the Detector class.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

from ..platform_check import check_proxsuite_availability

__all__ = [
    "solve_cvxpy",
    "solve_landweber",
    "solve_mlem",
    "solve_qpsolvers",
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
    
    step_size = 1.0 / (sigma_max**2)
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


def _create_derivative_matrix(n: int, order: int) -> np.ndarray:
    """Create finite difference derivative matrix.
    
    Parameters
    ----------
    n : int
        Size of spectrum.
    order : int
        Derivative order (1 or 2).
    
    Returns
    -------
    np.ndarray
        Derivative matrix.
    """
    if order == 1:
        # First derivative (n-1 x n)
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i] = -1
            L[i, i + 1] = 1
        return L
    elif order == 2:
        # Second derivative (n-2 x n)
        L = np.zeros((n - 2, n))
        for i in range(n - 2):
            L[i, i] = 1
            L[i, i + 1] = -2
            L[i, i + 2] = 1
        return L
    else:
        raise ValueError(f"Unsupported derivative order: {order}")


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
        # Fallback to available solver
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
    P = A.T @ A
    q = -A.T @ b
    
    # Add regularization and smoothness
    if norm == 2:
        if smoothness_order == 1:
            L = _create_derivative_matrix(n, 1)
            P += alpha * smoothness_weight * (L.T @ L)
        elif smoothness_order == 2:
            L = _create_derivative_matrix(n, 2)
            P += alpha * smoothness_weight * (L.T @ L)
        else:
            # Standard Tikhonov
            P += alpha * np.eye(n)
        
        # Constraints: x >= 0
        G = -np.eye(n)
        h = np.zeros(n)
        
        x = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            solver=solver,
            initvals=x_init,
            verbose=False,
        )
    
    elif norm == 1:
        # L1 regularization via variable extension
        # This is more complex - for simplicity, use L2 fallback
        warnings.warn(
            "L1 norm with qpsolvers uses L2 approximation. "
            "Use cvxpy for proper L1 regularization."
        )
        P += alpha * np.eye(n)
        G = -np.eye(n)
        h = np.zeros(n)
        
        x = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            solver=solver,
            initvals=x_init,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported norm type: {norm}")
    
    if x is None:
        warnings.warn(f"Solver '{solver}' did not find a solution.")
        return None
    
    return np.asarray(x)
