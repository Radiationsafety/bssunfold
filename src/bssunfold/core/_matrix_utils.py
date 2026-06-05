"""Shared matrix utility functions for bssunfold core modules.

This module provides common matrix operations used across unfolding methods
and regularization modules, avoiding code duplication.
"""

import numpy as np
from scipy.sparse import csc_matrix, diags


def create_derivative_matrix(n: int, order: int) -> csc_matrix:
    """Create finite difference derivative matrix in csc format.

    Parameters
    ----------
    n : int
        Size of spectrum.
    order : int
        Derivative order (1 or 2).

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
        # First derivative: [-1, 1] on diagonals
        data = np.concatenate([[-1] * (n - 1), [1] * (n - 1)])
        row = np.arange(2 * (n - 1)) // 2
        col = np.concatenate([np.arange(n - 1), np.arange(1, n)])
        L = csc_matrix((data, (row, col)), shape=(n - 1, n))
        return L
    elif order == 2:
        # Second derivative: [1, -2, 1] on diagonals - more efficient with diags
        L = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format='csc')
        return L
    else:
        raise ValueError(f"Unsupported derivative order: {order}. Use 1 or 2.")


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
    return U, s, Vt, s ** 2