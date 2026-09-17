"""KKT / Lagrange-duality diagnostics for the NNLS unfolding problem.

The constrained unfolding problem

    min_{x >= 0}  f(x) = 1/2 ||A x - b||^2

is a convex QP.  Its Lagrange dual (Lagrange duality, KKT conditions) yields
a certificate of optimality: for any primal-feasible ``x`` one can construct
a dual-feasible multiplier

    lambda = max(0, A^T (A x - b))

and evaluate the dual objective

    g(lambda) = 1/2 ||b||^2 - 1/2 (c + lambda)^T G^+ (c + lambda),
    G = A^T A,  c = A^T b,

where ``G^+`` is the Moore-Penrose pseudo-inverse (G is singular for typical
Bonner-sphere response matrices with more energy bins than spheres).  The
quantity ``f(x) - g(lambda)`` is then an upper bound on the distance of ``x``
to the primal optimum and vanishes exactly at a KKT point.  This module
turns that observation into per-iteration diagnostics for the
optimization-based unfolding methods (PGD, ADMM, L-BFGS-B, ...).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["nnls_duality_gap", "nnls_kkt_residuals"]


def nnls_duality_gap(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x: NDArray[np.float64],
) -> dict[str, float]:
    """Compute a duality-gap certificate for the NNLS unfolding problem.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x : np.ndarray
        Primal-feasible candidate spectrum (n,), expected ``x >= 0``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        - ``'primal_value'``: 1/2 ||A x - b||^2
        - ``'dual_value'``: g(lambda) computed at the reconstructed multiplier
        - ``'duality_gap'``: primal_value - dual_value (>= 0 up to rounding)
        - ``'relative_gap'``: gap / max(primal_value, eps)

    Notes
    -----
    The dual value is evaluated with the pseudo-inverse of ``G = A^T A`` and
    is therefore an approximate certificate when ``G`` is rank-deficient.
    For well-posed subproblems the gap is a rigorous upper bound on
    ``f(x) - f(x*)``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.maximum(np.asarray(x, dtype=float), 0.0)

    m, n = A.shape
    G = A.T @ A
    c = A.T @ b

    primal_value = 0.5 * float(np.sum((A @ x - b) ** 2))

    # Reconstruct a dual-feasible multiplier from stationarity residual
    lam = np.maximum(A.T @ (A @ x - b), 0.0)

    # Dual objective via pseudo-inverse (rank-deficient G is the norm for BSS)
    G_pinv = np.linalg.pinv(G)
    dual_value = 0.5 * float(b @ b) - 0.5 * float((c + lam) @ (G_pinv @ (c + lam)))

    gap = primal_value - dual_value
    # Relative measure: normalize by the primal value, floored by a small
    # fraction of the trivial energy 1/2||b||^2 so that near-zero primal
    # values (noiseless exact fit) do not blow the ratio up.
    floor = max(primal_value, 1e-8 * float(b @ b), 1e-30)
    return {
        "primal_value": primal_value,
        "dual_value": dual_value,
        "duality_gap": float(max(gap, 0.0)),
        "relative_gap": float(max(gap, 0.0) / floor),
    }


def nnls_kkt_residuals(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Evaluate the KKT residuals (stationarity, complementarity, feasibility).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x : np.ndarray
        Candidate spectrum (n,).
    tolerance : float, optional
        Activity threshold used to split the support into the active set
        (``x <= tolerance``) and the free set (default: 0.0).

    Returns
    -------
    dict[str, Any]
        Keys: ``'stationarity'`` (||max(0, A^T(Ax-b))|| reduced to the active
        set + violation on the free set), ``'complementarity'``
        (|lambda^T x|), ``'feasibility'`` (||min(0, x)||), ``'active_set'``
        boolean mask of active bins.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    grad = A.T @ (A @ x - b)  # Gx - c
    lam = np.maximum(grad, 0.0)
    active = x <= tolerance

    # Stationarity violation: lambda must vanish on the free set and grad must
    # vanish (up to lambda) on the active set
    stationarity = float(np.linalg.norm(grad[~active])) + float(
        np.linalg.norm(x[active] * lam[active])
    )
    complementarity = abs(float(lam @ x))
    feasibility = float(np.linalg.norm(np.minimum(x, 0.0)))

    return {
        "stationarity": stationarity,
        "complementarity": complementarity,
        "feasibility": feasibility,
        "active_set": active,
    }
