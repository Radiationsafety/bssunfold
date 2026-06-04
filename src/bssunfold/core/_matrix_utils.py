"""Shared matrix utility functions for bssunfold core modules.

This module provides common matrix operations used across unfolding methods
and regularization modules, avoiding code duplication.
"""

import numpy as np


def create_derivative_matrix(n: int, order: int) -> np.ndarray:
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
        Derivative matrix of shape (n-1, n) for order=1 or (n-2, n) for order=2.

    Raises
    ------
    ValueError
        If order is not 1 or 2.
    """
    if order == 1:
        L = np.zeros((n - 1, n))
        np.fill_diagonal(L, -1)
        np.fill_diagonal(L[:, 1:], 1)
        return L
    elif order == 2:
        L = np.zeros((n - 2, n))
        np.fill_diagonal(L, 1)
        np.fill_diagonal(L[:, 1:], -2)
        np.fill_diagonal(L[:, 2:], 1)
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