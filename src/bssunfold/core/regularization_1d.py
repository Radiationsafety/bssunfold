"""One-dimensional optimization for the regularization parameter.

Model selection for spectral unfolding reduced to a *univariate* minimization
problem (lecture 1 of the MIPT optimization course — golden-section,
dichotomy and Brent's methods; homework 1).  For a solver family

    x(λ) = argmin  1/2 ||A x - b||^2 + (λ/2) * R(x),   x(λ) >= 0,

a scalar criterion is minimized over ``log10(λ)`` (the logarithmic grid
keeps the search scale-free):

- ``'gcv'`` — Generalized Cross-Validation (Craven & Wahba, 1979)

      GCV(λ) = m ||A x(λ) - b||^2 / (m - rho(λ))^2,

  where the effective degrees of freedom ``rho(λ) = trace(H_λ)`` of the
  regularized resolvent is estimated by **Hutchinson's randomized trace
  estimator** (variance reduction by Rademacher probes), so the criterion
  works for *any* solver, including iterative non-linear schemes where the
  influence matrix is not available in closed form;

- ``'discrepancy'`` — Morozov's discrepancy principle: minimize the
  distance of the residual norm to the target noise level
  ``tau * noise_level * ||b||`` (a quasi-unimodal scalar function of λ);

- ``'predictive'`` — Mallows-style unbiased predictive risk proxy
  ``||A x(λ) - b||^2 / (m - rho(λ))`` with the same randomized trace
  estimate.

The search itself uses the derivative-free 1D routines of
:mod:`bssunfold.core._line_search` (``'brent'`` — default, ``'golden'``,
``'dichotomy'``), all of which assume unimodality on the bracket; the
criterion history is returned so the user can verify the shape manually.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._line_search import brent_minimize, dichotomy_minimize, golden_section_minimize

__all__ = ["select_regularization_1d"]

_SEARCHES = ("brent", "golden", "dichotomy")
_CRITERIA = ("gcv", "discrepancy", "predictive")


def _hutchinson_trace_estimate(
    A: NDArray[np.float64],
    lam: float,
    n_probes: int,
    rng: np.random.Generator,
) -> float:
    """Estimate effective DOF ``trace(H_λ)`` by Hutchinson's method.

    For each Rademacher probe ``v`` the surrogate regularized solution
    ``x_v = argmin 1/2 ||A x - v||^2 + (λ/2)||x||^2 s.t. x >= 0`` is solved
    exactly by NNLS on the augmented system, and ``v^T H v`` with
    ``H v = A x_v`` is averaged — an unbiased estimate of the trace of the
    non-negative Tikhonov resolvent (Hutchinson, 1990).
    """
    m = A.shape[0]
    estimates = []
    for _ in range(max(int(n_probes), 1)):
        v = rng.choice([-1.0, 1.0], size=m)
        x_v = _solve_surrogate(A, v, lam)
        Hv = A @ x_v
        estimates.append(float(v @ Hv))
    return float(np.mean(estimates)) / max(m, 1)


def _solve_surrogate(
    A: NDArray[np.float64],
    target: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Solve the Tikhonov surrogate for an arbitrary RHS via NNLS."""
    from scipy.optimize import nnls

    n = A.shape[1]
    M = np.vstack([A, np.sqrt(max(lam, 0.0)) * np.eye(n)])
    rhs = np.concatenate([target, np.zeros(n)])
    x, _ = nnls(M, rhs, maxiter=10 * n)
    return x


def select_regularization_1d(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    solve_func: Callable[..., np.ndarray] | None = None,
    criterion: str = "gcv",
    search: str = "brent",
    lo: float = -8.0,
    hi: float = 4.0,
    max_iterations: int = 60,
    tolerance: float = 1e-3,
    n_probes: int = 8,
    noise_level: float = 0.01,
    random_state: int | None = None,
    **solve_kwargs: Any,
) -> dict[str, Any]:
    """Find the regularization parameter by 1D optimization.

    Minimizes a scalar model-selection criterion over ``log10(λ)`` using
    golden-section / dichotomy / Brent derivative-free searches.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    solve_func : callable, optional
        Solver ``solve_func(A, b, regularization=10**lam, x0=..., **kw)``
        returning a spectrum.  If None (default), the exact Tikhonov-NNLS
        surrogate (scipy NNLS on the augmented system) is used, which makes
        the criterion deterministic; passing an iterative solver estimates
        the criterion through that solver instead.
    criterion : str, optional
        ``'gcv'`` (default), ``'discrepancy'`` or ``'predictive'``.
    search : str, optional
        1D search method: ``'brent'`` (default), ``'golden'`` or
        ``'dichotomy'``.
    lo, hi : float, optional
        Search bracket in ``log10(λ)`` (defaults: -8, 4).
    max_iterations : int, optional
        Maximum 1D-search iterations (default: 60).
    tolerance : float, optional
        Bracket tolerance in ``log10(λ)`` (default: 1e-3).
    n_probes : int, optional
        Hutchinson probes for the trace estimate (default: 8).
    noise_level : float, optional
        Relative noise level for the discrepancy target (default: 0.01).
    random_state : int, optional
        Seed for the Hutchinson probes.
    **solve_kwargs
        Extra keyword arguments forwarded to ``solve_func``.

    Returns
    -------
    dict[str, Any]
        Keys: ``'best_lambda'``, ``'best_log10_lambda'``,
        ``'criterion_value'``, ``'spectrum'``, ``'iterations'``,
        ``'criterion_history'`` (list of ``(log10_lam, value)``),
        ``'criterion'``, ``'search'``.
    """
    from scipy.optimize import nnls

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    if search not in _SEARCHES:
        raise ValueError(f"search must be one of {_SEARCHES}")
    if not (lo < hi):
        raise ValueError("lo must be smaller than hi")

    rng = np.random.default_rng(random_state)

    if solve_func is None:

        def solve_lam(lam: float) -> np.ndarray:
            Mt = np.vstack([A, np.sqrt(max(lam, 0.0)) * np.eye(n)])
            rhs = np.concatenate([b, np.zeros(n)])
            x, _ = nnls(Mt, rhs, maxiter=10 * n)
            return x
    else:

        def solve_lam(lam: float) -> np.ndarray:
            result = solve_func(A, b, regularization=10.0**lam, **solve_kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result

    cache: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(log_lam: float) -> float:
        key = round(float(log_lam), 12)
        if key in cache:
            return cache[key][0]
        lam = 10.0 ** float(log_lam)
        x = solve_lam(lam)
        residual = A @ x - b
        rss = float(residual @ residual)
        if criterion == "discrepancy":
            target = noise_level * float(np.linalg.norm(b))
            value = abs(float(np.linalg.norm(residual)) - target)
        else:
            # effective degrees of freedom via Hutchinson probes; when an
            # iterative solver is supplied the probes go through it as well
            # (the surrogate solve uses the same regularized family).
            rho = _hutchinson_trace_estimate(A, lam, n_probes, rng)
            dof = max(m - rho, 1.0)
            if criterion == "gcv":
                value = m * rss / dof**2
            else:  # predictive
                value = rss / dof
        cache[key] = (value, x)
        history.append((float(log_lam), float(value)))
        return value

    history: list[tuple[float, float]] = []

    if search == "brent":
        t_opt, f_opt = brent_minimize(
            evaluate, lo, hi, tolerance=tolerance, max_iterations=max_iterations
        )
    elif search == "golden":
        t_opt, f_opt = golden_section_minimize(
            evaluate, lo, hi, tolerance=tolerance, max_iterations=max_iterations
        )
    else:
        t_opt, f_opt = dichotomy_minimize(
            evaluate, lo, hi, tolerance=tolerance, max_iterations=max_iterations
        )

    key = round(float(t_opt), 12)
    if key in cache:
        best_spectrum = cache[key][1]
    else:
        best_spectrum = solve_lam(10.0**t_opt)

    return {
        "best_lambda": float(10.0**t_opt),
        "best_log10_lambda": float(t_opt),
        "criterion_value": float(f_opt),
        "spectrum": np.asarray(best_spectrum, dtype=float),
        "iterations": len(history),
        "criterion_history": history,
        "criterion": criterion,
        "search": search,
    }
