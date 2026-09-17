"""One-dimensional line-search utilities for unfolding solvers.

Implements the classical one-dimensional minimization building blocks
(golden-section search, dichotomy, Brent's method, Armijo backtracking)
used as internal steps of the gradient-based unfolding methods (projected
gradient, Frank--Wolfe, ...) and for scalar model-selection problems such
as the search for the regularization parameter.  All derivative-free
bracketing routines follow the standard schemes of lecture 1 of the MIPT
optimization course (homework 1) — see also Demidovich et al., "Lectures on
Mathematical Programming" / Luenberger, "Linear and Nonlinear Programming"
/ Brent, "Algorithms for Minimization without Derivatives" (1973).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "golden_section_minimize",
    "dichotomy_minimize",
    "brent_minimize",
    "backtracking_line_search",
]

_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi ~ 0.618


def golden_section_minimize(
    func,
    lo: float,
    hi: float,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> tuple[float, float]:
    """Minimize a unimodal scalar function over ``[lo, hi]``.

    Golden-section search: at every step the bracket is contracted by the
    factor ``1/phi`` keeping the previously evaluated interior point, so each
    iteration costs a single function evaluation after the first two.

    Parameters
    ----------
    func : callable
        Scalar objective ``func(t: float) -> float``.
    lo : float
        Lower bracket bound.
    hi : float
        Upper bracket bound.
    tolerance : float, optional
        Absolute bracket-width tolerance (default: 1e-8).
    max_iterations : int, optional
        Safety cap on iterations (default: 200).

    Returns
    -------
    tuple[float, float]
        ``(t_opt, f_opt)`` — the minimizer and the minimum value.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("golden_section_minimize requires finite bounds")
    if hi <= lo:
        return lo, float(func(lo))

    a, b = float(lo), float(hi)
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = func(c), func(d)

    for _ in range(max_iterations):
        if b - a <= tolerance:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = func(d)

    t_opt = 0.5 * (a + b)
    return t_opt, float(func(t_opt))


def backtracking_line_search(
    objective,
    x: NDArray[np.float64],
    gradient: NDArray[np.float64],
    direction: ArrayLike,
    initial_step: float = 1.0,
    alpha: float = 1e-4,
    beta: float = 0.5,
    max_trials: int = 30,
) -> float:
    """Armijo backtracking line search along ``direction``.

    Finds a step ``t`` satisfying the sufficient-decrease condition

    ``objective(x + t * direction) <= objective(x) + alpha * t * grad^T dir``.

    Parameters
    ----------
    objective : callable
        Objective ``objective(x: np.ndarray) -> float`` evaluated at trial
        points.
    x : np.ndarray
        Current iterate.
    gradient : np.ndarray
        Current gradient (used only for the directional derivative).
    direction : array-like
        Descent direction (typically the negative gradient).
    initial_step : float, optional
        Starting trial step (default: 1.0).
    alpha : float, optional
        Armijo constant in (0, 1) (default: 1e-4).
    beta : float, optional
        Step contraction factor in (0, 1) (default: 0.5).
    max_trials : int, optional
        Maximum backtracking trials (default: 30).

    Returns
    -------
    float
        The accepted step size. If no Armijo point is found within
        ``max_trials`` the last (smallest) trial step is returned so that the
        caller can still make a (damped) move.
    """
    direction = np.asarray(direction, dtype=float)
    directional_derivative = float(gradient @ direction)
    if directional_derivative >= 0:
        return 0.0

    f0 = float(objective(x))
    step = float(initial_step)
    for _ in range(max_trials):
        trial_val = float(objective(x + step * direction))
        if trial_val <= f0 + alpha * step * directional_derivative:
            return step
        step *= beta
    return step


def dichotomy_minimize(
    func,
    lo: float,
    hi: float,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
    delta_frac: float = 0.1,
) -> tuple[float, float]:
    """Minimize a unimodal scalar function over ``[lo, hi]`` by dichotomy.

    The uniform-partition method: at each step the interval is split into
    two overlapping halves separated by a small ``delta`` (to keep function
    values distinguishable), and the half with the larger objective is
    discarded.  Two evaluations per iteration halve the bracket, giving a
    straightforward logarithmic contraction (lecture 1, homework 1).

    Parameters
    ----------
    func : callable
        Scalar objective ``func(t: float) -> float``.
    lo, hi : float
        Bracket bounds.
    tolerance : float, optional
        Absolute bracket-width tolerance (default: 1e-8).
    max_iterations : int, optional
        Safety cap on iterations (default: 200).
    delta_frac : float, optional
        Half-gap between the two midpoints as a fraction of the bracket
        width (default: 0.1).

    Returns
    -------
    tuple[float, float]
        ``(t_opt, f_opt)`` — the minimizer and the minimum value.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("dichotomy_minimize requires finite bounds")
    if hi <= lo:
        return lo, float(func(lo))

    a, b = float(lo), float(hi)
    for _ in range(max_iterations):
        if b - a <= tolerance:
            break
        mid = 0.5 * (a + b)
        delta = max(delta_frac * (b - a), tolerance * 1e-3)
        c, d = mid - delta, mid + delta
        if float(func(c)) <= float(func(d)):
            b = d
        else:
            a = c

    t_opt = 0.5 * (a + b)
    return t_opt, float(func(t_opt))


def brent_minimize(
    func,
    lo: float,
    hi: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> tuple[float, float]:
    """Minimize a unimodal scalar function by Brent's method.

    Structure follows the classical Netlib ``fmin`` implementation
    (Brent, 1973): at every iteration an inverse-quadratic interpolation
    through the three best points is attempted and accepted only when it
    stays inside the bracket and less than half of the previous step;
    otherwise a golden-section step is taken.  Superlinear convergence for
    smooth objectives with a worst case no worse than golden-section.

    Parameters
    ----------
    func : callable
        Scalar objective ``func(t: float) -> float``.
    lo, hi : float
        Bracket bounds.
    tolerance : float, optional
        Absolute bracket-width tolerance (default: 1e-8).
    max_iterations : int, optional
        Safety cap on iterations (default: 100).

    Returns
    -------
    tuple[float, float]
        ``(t_opt, f_opt)`` — the minimizer and the minimum value.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("brent_minimize requires finite bounds")
    if hi <= lo:
        return lo, float(func(lo))

    GOLD = 0.5 * (3.0 - np.sqrt(5.0))  # ~0.382 golden-section coefficient

    a, b = float(lo), float(hi)
    x = a + GOLD * (b - a)  # best point
    fx = float(func(x))
    v = w = x  # v: previous, w: second previous best
    fv = fw = fx
    d = e = b - a  # current and previous step sizes

    for _ in range(max_iterations):
        m = 0.5 * (a + b)
        tol = tolerance * abs(x) + 0.33 * tolerance
        if abs(x - m) <= 2.0 * tol - 0.5 * (b - a):
            break

        p = q = r = 0.0
        if abs(e) > tol:
            # inverse-quadratic interpolation through (x,fx),(v,fv),(w,fw)
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            r = e
            e = d

        if abs(p) < abs(0.5 * q * r) and q * (a - x) < p < q * (b - x):
            # accept the interpolated step
            d = p / q
            u = x + d
            if (u - a) < 2.0 * tol or (b - u) < 2.0 * tol:
                d = np.sign(m - x) * tol if m != x else tol
        else:
            # golden-section step on the wider side of x
            e = b - x if x < m else a - x
            d = GOLD * e

        u = x + (d if abs(d) > tol else np.sign(d) * tol if d != 0 else tol)
        fu = float(func(u))

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    return x, fx
