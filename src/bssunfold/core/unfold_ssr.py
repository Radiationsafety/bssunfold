"""SSR unfolding: Sign-Simplicity-Regression solver (sisireg port).

Python port of the R package ``sisireg`` 1.2.1 (Lars Metzner, CRAN,
GPL>=2; ``https://cran.r-project.org/package=sisireg``): the
Sign-Simplicity-Regression (SSR) model of Metzner (2020, 2021) is a
nonparametric regression built on the *signs* of the residuals.  It
seeks the most parsimonious (fewest extrema) regression function that
is statistically adequate with respect to two sign criteria:

* the **partial sum criterion**: for every interval length ``k`` the
  maximum absolute sum of consecutive residual signs must not exceed
  the 95% quantile of the partial sums,

  F(n, k) = min( sqrt(1 + 2.33 ln(n) k), k );

* the **maximum run criterion**: the 95% quantile of the maximum run
  length of equal residual signs is

  k_run(n) = int(3.3 + 1.44 ln(n)),

  and no window of ``k_run + 1`` consecutive residuals may sum to more
  than ``k_run`` in absolute value.

The regression function is computed with a quantised Gauss-Seidel
(QSOR) iteration: every interior point is replaced by the simplicitic
linear interpolation of its neighbours, and the update is *reverted*
whenever it would violate the partial sum criterion with threshold
``fn``.  Iterating this projection yields the most parsimonious
statistically adequate function.

Unfolding integration
---------------------
The SSR model regresses directly observed data, while the unfolding
problem only observes the folded readings ``b = A x``.  The method
therefore alternates

1. a **data-fidelity step** -- one multiplicative MLEM update of the
   spectrum (positivity preserving), and
2. an **SSR parsimony step** -- a non-equidistant SSR QSOR sweep of the
   current spectrum estimate over the energy grid (the sweep continues
   from the current estimate instead of re-initialising with rolling
   medians), which removes sign-inadequate wiggles while bounding the
   deviation from the data-fit in every window.

The strength of the parsimony step is the partial sum threshold
``fn``.  Following Metzner's *minimum statistic*, ``fn="auto"`` starts
from ``fn_start = int(0.66 * F(n, k_run))``, decreases ``fn`` by one
and stops at the first candidate whose folded residuals violate the
data-space sign adequacy (partial sum / maximum run test) or whose
parsimony (number of extrema) exceeds that of the start model; the
last adequate candidate is returned (the R implementation recomputes
the same model, and falls back to ``fn_start + 1`` when even the start
model is inadequate).

Module API
----------
Faithful ports of the sisireg building blocks (R names in brackets):

* ``max_run_quantile(n)`` [``maxRunR``],
* ``partial_sum_quantile(n, k)`` [``fnR``],
* ``rolling_median(v, k)`` [``zoo::rollapply`` median, centre aligned],
* ``partial_sum_max(dat, mu, k)`` [``psmaxR``],
* ``number_of_extrema(mu)`` [``numberOfExtremaR``],
* ``partial_sum_valid(dat, mu)`` [``psvalid``],
* ``run_valid(dat, mu, k=None)`` [``runvalid``],
* ``ssr(y, ...)`` [``ssr`` -- equidistant QSOR, L1 and L2 variants],
* ``ssr_ne(x, y, ...)`` [``ssr_neR`` -- non-equidistant QSOR, L1],
* ``ssr_min_statistic(y, ...)`` / ``ssr_min_statistic_ne(x, y, ...)``
  [``ssr_minR`` / ``ssr_ne_minR``],
* ``ssr_predict(x, mu, xx)`` [``ssr_predict``],

and the standard bssunfold solver conventions:

* ``solve_ssr(A, b, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``solve_ssr_full(...)`` -- the same solver returning a rich
  diagnostics dictionary;
* ``unfold_ssr(...)`` -- Detector-facing wrapper (also exposed as
  ``Detector.unfold_ssr``).

Notes
-----
* Pure NumPy implementation: the QSOR sweeps are O(n) sequential
  passes over the energy bins, so Numba adds no benefit at
  Bonner-sphere bin counts.
* The port reproduces the R/C semantics exactly, including the
  start-value layout (head/middle/tail rolling medians, with the
  middle segment winning the one-point overlap for even windows) and
  the truncating integer conversions of the R ``.C`` interface.  The
  L1 non-equidistant solver always applies the partial sum criterion
  with threshold ``h = fn`` (C ``ssr_neC``); the equidistant L2
  variant keeps the separate ``<``/``<=`` loop bounds of C ``ssrC``.
* For fewer than 25 detector readings the data-space partial sum test
  is trivially satisfied (its longest tested interval is ``m // 5``);
  selection is then driven by the maximum run test and the parsimony
  (extrema) criterion.
* License: the R package sisireg is GPL (>= 2), compatible with the
  GPL-3 license of bssunfold.

References
----------
.. [1] L. Metzner, *Trendbasierte Prognostik*, ISBN 979-8-68239-420-3,
       2020.
.. [2] L. Metzner, *Adaequates Maschinelles Lernen*, ISBN
       979-8-59347-027-0, 2021.
.. [3] L. Metzner, "sisireg: Sign-Simplicity-Regression-Solver", R
       package version 1.2.1, CRAN, 2025.
"""

from typing import Any

import numpy as np

from ..logging_config import get_logger
from ..utils.validators import validate_system
from ._base_unfolder import run_unfolding

__all__ = [
    "max_run_quantile",
    "partial_sum_quantile",
    "rolling_median",
    "partial_sum_max",
    "number_of_extrema",
    "partial_sum_valid",
    "run_valid",
    "ssr",
    "ssr_ne",
    "ssr_min_statistic",
    "ssr_min_statistic_ne",
    "ssr_predict",
    "solve_ssr",
    "solve_ssr_full",
    "unfold_ssr",
]

logger = get_logger("unfold_ssr")

# Numerical guards -----------------------------------------------------------
_TINY = 1e-300

# SSR regression needs rolling medians of window k_run ~ 3.3+1.44 ln(n)
# and partial sum windows, so very short series are meaningless.
_MIN_DATA_POINTS = 8


def _sgn(v: float) -> int:
    """Sign of a scalar, exactly the C macro ``(x > 0) - (x < 0)``."""
    v = float(v)
    return (v > 0.0) - (v < 0.0)


def max_run_quantile(n: int) -> int:
    """95% quantile of the maximum run length of residual signs.

    Port of R ``maxRunR``: ``k = as.integer(3.3 + 1.44 log(n))`` with
    the truncating conversion of R ``as.integer``.

    Parameters
    ----------
    n : int
        Number of data points (positive).

    Returns
    -------
    int
        Maximum run length quantile ``k_run``.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return int(3.3 + 1.44 * np.log(n))


def partial_sum_quantile(n: int, k: int | np.ndarray) -> float | np.ndarray:
    """95% quantile of partial sums of residual signs (R ``fnR``).

    ``F(n, k) = min(sqrt(1 + 2.33 ln(n) k), k)``, computed
    element-wise; a scalar is returned for scalar ``k``.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    k_arr = np.asarray(k, dtype=float)
    fn = np.sqrt(1.0 + 2.33 * np.log(n) * k_arr)
    out = np.minimum(fn, k_arr)
    if np.ndim(k) == 0:
        return float(out)
    return out


def rolling_median(v: np.ndarray, k: int) -> np.ndarray:
    """Centre-aligned rolling median with ``n - k + 1`` values.

    Equivalent to ``zoo::rollapply(zoo(v), k, median, align='center')``
    in R: element ``j`` of the result is ``median(v[j : j + k])``.

    Parameters
    ----------
    v : np.ndarray
        Data series.
    k : int
        Window length (``1 <= k <= len(v)``).

    Returns
    -------
    np.ndarray
        Rolling medians, shape ``(len(v) - k + 1,)``.
    """
    v = np.asarray(v, dtype=float).ravel()
    k = int(k)
    n = v.shape[0]
    if k < 1:
        raise ValueError(f"window k must be >= 1, got {k}")
    if k > n:
        raise ValueError(
            f"window k ({k}) must not exceed the data length ({n})"
        )
    windows = np.lib.stride_tricks.sliding_window_view(v, k)
    return np.median(windows, axis=1)


def _ssr_start_values(dat: np.ndarray, k: int) -> np.ndarray:
    """Initial QSOR values: head/middle/tail rolling medians.

    Reproduces the start-value layout of R ``ssrR``/``ssr_neR``: the
    head and tail are filled with the median of the first/last ``k2 =
    k // 2`` points, the middle with the centre-aligned rolling median
    of window ``k``.  For even ``k`` the middle segment overlaps the
    head by one point and wins (R assignment order).
    """
    dat = np.asarray(dat, dtype=float).ravel()
    n = dat.shape[0]
    k = int(k)
    if k < 1 or k > n:
        raise ValueError(f"window k ({k}) must be in [1, {n}]")
    k2 = k // 2
    k1 = k - 2 * k2
    s = np.empty(n, dtype=float)
    if k2 > 0:
        s[:k2] = np.median(dat[:k2])
    start = k2 + k1 - 1
    s[start : n - k2] = rolling_median(dat, k)
    if k2 > 0:
        s[n - k2 :] = np.median(dat[n - k2 :])
    return s


def partial_sum_max(dat: np.ndarray, mu: np.ndarray, k: int) -> int:
    """Maximum absolute partial sum of residual signs (R ``psmaxR``).

    Parameters
    ----------
    dat : np.ndarray
        Observed data ``y``.
    mu : np.ndarray
        Model values ``mu`` (same shape as ``dat``).
    k : int
        Interval length (``1 <= k <= len(dat)``).

    Returns
    -------
    int
        ``max_t |sum(sign(dat - mu)[t : t + k])|``.
    """
    dat = np.asarray(dat, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    if dat.shape[0] != mu.shape[0]:
        raise ValueError(
            f"dat and mu must have the same length, got {dat.shape[0]} "
            f"and {mu.shape[0]}"
        )
    k = int(k)
    if k < 1:
        raise ValueError(f"window k must be >= 1, got {k}")
    if k > dat.shape[0]:
        raise ValueError(
            f"window k ({k}) must not exceed the data length "
            f"({dat.shape[0]})"
        )
    s = np.sign(dat - mu)
    sums = np.lib.stride_tricks.sliding_window_view(s, k).sum(axis=1)
    return int(np.max(np.abs(sums)))


def number_of_extrema(mu: np.ndarray) -> int:
    """Number of local extrema of a discrete function.

    Port of R ``numberOfExtremaR``: counts the slope sign changes of
    the sequence, classifying them by the previous slope direction.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    n = mu.shape[0]
    if n < 2:
        return 0
    n_min = 0
    n_max = 0
    slope = _sgn(mu[1] - mu[0])
    for i in range(2, n):
        s = _sgn(mu[i] - mu[i - 1])
        if s != slope:
            if slope < 0:
                n_min += 1
            else:
                n_max += 1
            slope = s
    return n_min + n_max


def partial_sum_valid(dat: np.ndarray, mu: np.ndarray) -> bool:
    """Partial sum adequacy test for all interval lengths (R ``psvalid``).

    Checks ``partial_sum_max(dat, mu, k) <= partial_sum_quantile(n, k)``
    for every interval length ``k`` in ``[5, n // 5]``.  For ``n < 25``
    the range is empty and the test is trivially satisfied.
    """
    dat = np.asarray(dat, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    n = dat.shape[0]
    maxint = int(n // 5)
    for k in range(5, maxint + 1):
        if partial_sum_max(dat, mu, k) > partial_sum_quantile(n, k):
            return False
    return True


def run_valid(dat: np.ndarray, mu: np.ndarray, k: int | None = None) -> bool:
    """Maximum run adequacy test (R ``runvalid``).

    The maximum absolute sum over windows of ``k + 1`` consecutive
    residual signs must not exceed ``k``.  ``k`` defaults to
    ``max_run_quantile(n)``.
    """
    dat = np.asarray(dat, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    n = dat.shape[0]
    if k is None:
        k = max_run_quantile(n)
    k = int(k)
    runmax = partial_sum_max(dat, mu, k + 1)
    return runmax <= k


def _ssr_equidistant_core(
    y: np.ndarray,
    mu: np.ndarray,
    funk: int,
    k: int,
    h: int,
    ps: bool,
    simanz: int,
) -> np.ndarray:
    """QSOR iteration for equidistant data (port of C ``ssrC``).

    Parameters
    ----------
    y, mu : np.ndarray
        Data and initial model (modified copies are made internally).
    funk : int
        ``1`` -- L1 (simplicitic neighbour interpolation), ``2`` -- L2
        (standardised 4th-order difference system, relaxation 1.9).
    k : int
        Run length quantile (window half width for the partial sum
        test when ``ps`` else the threshold itself).
    h : int
        Partial sum threshold (only used when ``ps``).
    ps : bool
        Partial sum mode (R ``ps`` flag).
    simanz : int
        Maximum number of full Gauss-Seidel sweeps.
    """
    n = y.shape[0]
    if funk == 1:
        for _ in range(max(int(simanz), 0)):
            chng = False
            for i in range(1, n - 1):
                oldval = mu[i]
                oldsig = _sgn(y[i] - mu[i])
                mu[i] = 0.5 * (mu[i - 1] + mu[i + 1])
                newsig = _sgn(y[i] - mu[i])
                if ps:
                    if k < i < n - k and oldsig != newsig:
                        psum = 0
                        for m in range(-k, k + 1):
                            psum += _sgn(y[i + m] - mu[i + m])
                        if abs(psum) > h:
                            mu[i] = oldval
                else:
                    # the C loop tests the (loop-invariant) sign change
                    # inside the j loop; hoisted here
                    if oldsig != newsig:
                        for j in range(max(0, i - k), min(n - k, i + k)):
                            psum = 0
                            for m in range(k + 1):
                                psum += _sgn(y[j + m] - mu[j + m])
                            if abs(psum) > k:
                                mu[i] = oldval
                                break
                if mu[i] != oldval:
                    chng = True
            if not chng:
                break
        return mu

    # funk == 2: standardised QSOR on the 4th-order difference system.
    # For n >= 8 the boundary stencils below are mutually exclusive, so
    # the C chain of separate `if` statements is written as elif.
    a2 = np.sqrt(2.0)
    a10 = np.sqrt(10.0)
    a12 = np.sqrt(12.0)
    q12 = 1.0 / (a2 * a10)
    q13 = 1.0 / (a2 * a12)
    q23 = 1.0 / (a10 * a12)
    a012 = -4.0 * q12
    a023 = -8.0 * q23
    a013 = 2.0 * q13
    a024 = 2.0 * q23
    as2 = 2.0 / 12.0
    as8 = -8.0 / 12.0
    omega = 1.9

    ys = y.copy()
    mus = mu.copy()
    ys[0] *= a2
    mus[0] *= a2
    ys[1] *= a10
    mus[1] *= a10
    ys[2 : n - 2] *= a12
    mus[2 : n - 2] *= a12
    ys[n - 2] *= a10
    mus[n - 2] *= a10
    ys[n - 1] *= a2
    mus[n - 1] *= a2

    for _ in range(max(int(simanz), 0)):
        chng = False
        for i in range(1, n - 1):
            oldval = mus[i]
            oldsig = _sgn(ys[i] - mus[i])
            if i == 1:
                mus[i] -= omega * (
                    mus[i - 1] * a012 + mus[i] + mus[i + 1] * a023
                    + mus[i + 2] * a024
                )
            elif i == 2:
                mus[i] -= omega * (
                    mus[i - 2] * a013 + mus[i - 1] * a023 + mus[i]
                    + mus[i + 1] * as8 + mus[i + 2] * as2
                )
            elif i == 3:
                mus[i] -= omega * (
                    mus[i - 2] * a024 + mus[i - 1] * as8 + mus[i]
                    + mus[i + 1] * as8 + mus[i + 2] * as2
                )
            elif i < n - 4:
                mus[i] -= omega * (
                    mus[i - 2] * as2 + mus[i - 1] * as8 + mus[i]
                    + mus[i + 1] * as8 + mus[i + 2] * as2
                )
            elif i == n - 4:
                mus[i] -= omega * (
                    mus[i - 2] * as2 + mus[i - 1] * as8 + mus[i]
                    + mus[i + 1] * as8 + mus[i + 2] * a024
                )
            elif i == n - 3:
                mus[i] -= omega * (
                    mus[i - 2] * as2 + mus[i - 1] * as8 + mus[i]
                    + mus[i + 1] * a023 + mus[i + 2] * a013
                )
            else:  # i == n - 2
                mus[i] -= omega * (
                    mus[i - 2] * a024 + mus[i - 1] * a023 + mus[i]
                    + mus[i + 1] * a012
                )
            newsig = _sgn(ys[i] - mus[i])
            if ps:
                if k < i < n - k and oldsig != newsig:
                    psum = 0
                    for m in range(-k, k + 1):
                        psum += _sgn(ys[i + m] - mus[i + m])
                    if abs(psum) > h:
                        mus[i] = oldval
            else:
                if oldsig != newsig:
                    for j in range(max(0, i - k), min(n - k, i + k) + 1):
                        psum = 0
                        for m in range(k + 1):
                            psum += _sgn(ys[j + m] - mus[j + m])
                        if abs(psum) > k:
                            mus[i] = oldval
                            break
            if mus[i] != oldval:
                chng = True
        if not chng:
            break

    mus[0] /= a2
    mus[1] /= a10
    mus[2 : n - 2] /= a12
    mus[n - 2] /= a10
    mus[n - 1] /= a2
    return mus


def _ssr_ne_core(
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    k: int,
    h: int,
    simanz: int,
) -> np.ndarray:
    """QSOR iteration, non-equidistant L1 (port of C ``ssr_neC``).

    ``x`` must be sorted ascending.  The partial sum criterion with
    threshold ``h`` is always applied (the non-equidistant solver has
    no ``ps = FALSE`` mode).
    """
    n = y.shape[0]
    for _ in range(max(int(simanz), 0)):
        chng = False
        for i in range(1, n - 1):
            oldval = mu[i]
            oldsig = _sgn(y[i] - mu[i])
            if x[i - 1] != x[i + 1]:
                mu[i] = mu[i - 1] + (x[i] - x[i - 1]) * (
                    mu[i + 1] - mu[i - 1]
                ) / (x[i + 1] - x[i - 1])
            else:
                mu[i] = 0.5 * (mu[i - 1] + mu[i + 1])
            newsig = _sgn(y[i] - mu[i])
            if k < i < n - k and oldsig != newsig:
                psum = 0
                for m in range(-k, k + 1):
                    psum += _sgn(y[i + m] - mu[i + m])
                if abs(psum) > h:
                    mu[i] = oldval
            if mu[i] != oldval:
                chng = True
        if not chng:
            break
    return mu


def _ssr_ne_sweep(
    x: np.ndarray,
    y_ref: np.ndarray,
    mu: np.ndarray,
    k: int,
    h: int,
) -> np.ndarray:
    """Single non-equidistant QSOR pass continuing from ``mu``.

    Unlike :func:`_ssr_ne_core` the model is NOT re-initialised with
    rolling medians: the unfolding solver calls this after every MLEM
    step, so ``mu`` already carries the current spectrum estimate.
    Adequacy is measured against the fixed pre-sweep reference
    ``y_ref``; ``mu`` is updated in place and returned.
    """
    n = y_ref.shape[0]
    for i in range(1, n - 1):
        oldval = mu[i]
        oldsig = _sgn(y_ref[i] - mu[i])
        if x[i - 1] != x[i + 1]:
            mu[i] = mu[i - 1] + (x[i] - x[i - 1]) * (
                mu[i + 1] - mu[i - 1]
            ) / (x[i + 1] - x[i - 1])
        else:
            mu[i] = 0.5 * (mu[i - 1] + mu[i + 1])
        newsig = _sgn(y_ref[i] - mu[i])
        if k < i < n - k and oldsig != newsig:
            psum = 0
            for m in range(-k, k + 1):
                psum += _sgn(y_ref[i + m] - mu[i + m])
            if abs(psum) > h:
                mu[i] = oldval
    return mu


def ssr(
    y: np.ndarray,
    fn: float = 0,
    ps: bool = True,
    funk: int = 1,
    y1: float | None = None,
    yn: float | None = None,
    simanz: int = 10000,
) -> np.ndarray:
    """Equidistant SSR QSOR regression (R ``ssr``/``ssrR`` + C ``ssrC``).

    Parameters
    ----------
    y : np.ndarray
        Data series (equidistant argument scale).
    fn : float, optional
        Partial sum threshold ``h``; non-positive values select the
        automatic default ``max(2, log(n) - 2)`` (in partial sum mode)
        or the run length quantile (otherwise), as in R ``ssrR``.
    ps : bool, optional
        Partial sum mode (default) вЂ” recommended; ``False`` switches
        to the maximum run criterion with window ``k = fn``.
    funk : int, optional
        ``1`` (default) for the L1 solver, ``2`` for the L2 variant.
    y1, yn : float, optional
        Fixed boundary values of the regression function.
    simanz : int, optional
        Maximum number of QSOR sweeps (default 10000, as in R).

    Returns
    -------
    np.ndarray
        Regression function ``mu`` (same length as ``y``).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    if n < _MIN_DATA_POINTS:
        raise ValueError(
            f"SSR requires at least {_MIN_DATA_POINTS} data points, got {n}"
        )
    funk = int(funk)
    if funk not in (1, 2):
        raise ValueError(f"funk must be 1 (L1) or 2 (L2), got {funk}")
    simanz = int(simanz)
    if simanz < 1:
        raise ValueError(f"simanz must be a positive integer, got {simanz}")
    if ps:
        if fn < 2:
            fn = max(2.0, float(np.log(n)) - 2.0)
        k = int(3.3 + 1.44 * np.log(n))
        h = int(fn)
    else:
        if fn < 2:
            fn = int(3.3 + 1.44 * np.log(n))
        k = int(fn)
        h = 0
    mu = _ssr_start_values(y, k)
    if y1 is not None:
        mu[0] = float(y1)
    if yn is not None:
        mu[-1] = float(yn)
    return _ssr_equidistant_core(y, mu, funk, k, h, bool(ps), simanz)


def ssr_ne(
    x: np.ndarray,
    y: np.ndarray,
    fn: float = 0,
    simanz: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-equidistant SSR QSOR regression, L1 (R ``ssr_neR``).

    Parameters
    ----------
    x : np.ndarray
        Argument values (sorted internally, stable sort).
    y : np.ndarray
        Data values (same length as ``x``).
    fn : float, optional
        Partial sum threshold; non-positive values select the default
        ``int(max(2, log(n) - 2))``, as in R ``ssr_neR``.
    simanz : int, optional
        Maximum number of QSOR sweeps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x_sorted, mu)`` -- sorted arguments and regression values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same length, got {x.shape[0]} "
            f"and {y.shape[0]}"
        )
    n = y.shape[0]
    if n < _MIN_DATA_POINTS:
        raise ValueError(
            f"SSR requires at least {_MIN_DATA_POINTS} data points, got {n}"
        )
    simanz = int(simanz)
    if simanz < 1:
        raise ValueError(f"simanz must be a positive integer, got {simanz}")
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order]
    if fn < 2:
        fn = int(max(2.0, float(np.log(n)) - 2.0))
    h = int(fn)
    k = int(3.3 + 1.44 * np.log(n))  # == max_run_quantile(n)
    mu = _ssr_start_values(ys, k)
    mu = _ssr_ne_core(xs, ys, mu, k, h, simanz)
    return xs, mu


def ssr_min_statistic(
    y: np.ndarray,
    funk: int = 1,
    y1: float | None = None,
    yn: float | None = None,
    ps: bool = True,
    simanz: int = 10000,
) -> tuple[np.ndarray, int]:
    """Minimum statistic SSR (R ``ssr_minR``).

    Starts from ``fn = 0.66 * partial_sum_quantile(n, k_run)`` and
    decreases ``fn`` while the model stays statistically adequate
    (partial sum or maximum run test) and does not gain extrema; the
    final model is recomputed at the last adequate ``fn + 1``.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(mu, fn)`` -- regression function and the final threshold.
        (R returns only ``mu``; ``fn`` is reported for diagnostics.)
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    if n < _MIN_DATA_POINTS:
        raise ValueError(
            f"SSR requires at least {_MIN_DATA_POINTS} data points, got {n}"
        )
    if ps:
        k = max_run_quantile(n)
        fn = 0.66 * float(partial_sum_quantile(n, k))
    else:
        fn = float(max_run_quantile(n))
    mu = ssr(y, funk=funk, y1=y1, yn=yn, fn=fn, ps=ps, simanz=simanz)
    valid = partial_sum_valid(y, mu) if ps else run_valid(y, mu)
    extrema_opt = number_of_extrema(mu)
    extrema = extrema_opt
    while valid and extrema <= extrema_opt and fn > 0:
        fn -= 1.0
        mu = ssr(y, funk=funk, y1=y1, yn=yn, fn=fn, ps=ps, simanz=simanz)
        valid = partial_sum_valid(y, mu) if ps else run_valid(y, mu)
        extrema = number_of_extrema(mu)
    fn += 1.0
    mu = ssr(y, funk=funk, y1=y1, yn=yn, fn=fn, ps=ps, simanz=simanz)
    return mu, int(fn)


def ssr_min_statistic_ne(
    x: np.ndarray,
    y: np.ndarray,
    simanz: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Minimum statistic SSR for non-equidistant data (R ``ssr_ne_minR``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(x_sorted, mu, fn)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same length, got {x.shape[0]} "
            f"and {y.shape[0]}"
        )
    n = y.shape[0]
    if n < _MIN_DATA_POINTS:
        raise ValueError(
            f"SSR requires at least {_MIN_DATA_POINTS} data points, got {n}"
        )
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order]
    k = max_run_quantile(n)
    fn = 0.66 * float(partial_sum_quantile(n, k))
    _, mu = ssr_ne(xs, ys, fn=fn, simanz=simanz)
    valid = partial_sum_valid(ys, mu)
    extrema_opt = number_of_extrema(mu)
    extrema = extrema_opt
    while valid and extrema <= extrema_opt and fn > 0:
        fn -= 1.0
        _, mu = ssr_ne(xs, ys, fn=fn, simanz=simanz)
        valid = partial_sum_valid(ys, mu)
        extrema = number_of_extrema(mu)
    fn += 1.0
    xs_final, mu = ssr_ne(xs, ys, fn=fn, simanz=simanz)
    return xs_final, mu, int(fn)


def ssr_predict(x: np.ndarray, mu: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Piecewise-linear prediction of an SSR model (R ``ssr_predict``).

    Linear interpolation between model points with linear
    extrapolation from the end segments outside the support
    (R ``ssr_ne_pred_singleR``).  Duplicate argument values are
    reduced to their first occurrence for a robust interpolation.

    Parameters
    ----------
    x : np.ndarray
        Model arguments (sorted internally, stable sort).
    mu : np.ndarray
        Model values (same length as ``x``).
    xx : np.ndarray
        Query points.

    Returns
    -------
    np.ndarray
        Predicted values at ``xx``.
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    if x.shape[0] != mu.shape[0]:
        raise ValueError(
            f"x and mu must have the same length, got {x.shape[0]} "
            f"and {mu.shape[0]}"
        )
    if x.shape[0] < 2:
        raise ValueError(
            f"ssr_predict requires at least 2 model points, got {x.shape[0]}"
        )
    order = np.argsort(x, kind="stable")
    xs = x[order]
    mus = mu[order]
    if np.any(np.diff(xs) == 0):
        keep = np.concatenate(([True], np.diff(xs) != 0))
        xs = xs[keep]
        mus = mus[keep]
    xx_arr = np.asarray(xx, dtype=float).ravel()
    out = np.empty(xx_arr.shape[0], dtype=float)
    for idx, x_ in enumerate(xx_arr):
        if x_ <= xs[0]:
            out[idx] = (
                mus[0] - (xs[0] - x_) * (mus[1] - mus[0]) / (xs[1] - xs[0])
                if xs[1] != xs[0]
                else mus[0]
            )
        elif x_ >= xs[-1]:
            out[idx] = (
                mus[-1] + (x_ - xs[-1]) * (mus[-1] - mus[-2])
                / (xs[-1] - xs[-2])
                if xs[-1] != xs[-2]
                else mus[-1]
            )
        else:
            out[idx] = np.interp(x_, xs, mus)
    return out


# ---------------------------------------------------------------------------
# SSR unfolding (MLEM data step + SSR parsimony sweep)
# ---------------------------------------------------------------------------


def _folded_adequacy(
    b: np.ndarray, fit: np.ndarray, k_run: int
) -> tuple[bool, bool, int]:
    """Data-space sign adequacy of a candidate solution.

    The SSR criteria are applied to the residuals ``b - A x`` of the
    folded model: ``ps_valid`` tests all interval lengths up to
    ``m // 5``, the maximum run test uses windows of ``k_run + 1``
    residuals.

    Returns
    -------
    tuple[bool, bool, int]
        ``(ps_valid, run_valid, max_run)``.
    """
    max_run = partial_sum_max(b, fit, k_run + 1)
    ps_ok = partial_sum_valid(b, fit)
    return ps_ok, max_run <= k_run, max_run


def _ssr_unfold_fixed_fn(
    A: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray,
    E_sorted: np.ndarray,
    k_run: int,
    fn_try: int,
    max_iterations: int,
    tolerance: float,
    smooth_every: int,
    inner_sweeps: int,
) -> tuple[np.ndarray, int, bool, int]:
    """Alternating MLEM / SSR-sweep scheme for one fixed ``fn``.

    Each outer iteration applies one multiplicative MLEM update
    (data fidelity, positivity preserving) followed -- every
    ``smooth_every`` iterations -- by ``inner_sweeps`` non-equidistant
    SSR QSOR sweeps on the current spectrum over the energy grid.

    Returns
    -------
    tuple[np.ndarray, int, bool, int]
        ``(spectrum, n_iterations, converged, n_sweeps)``.
    """
    floor = max(float(np.sum(b)), 1.0) * 1e-12
    x = np.maximum(np.asarray(x_init, dtype=float).copy(), floor)
    colsum = A.sum(axis=0)
    colsum_safe = np.where(colsum > _TINY, colsum, 1.0)
    AT = A.T
    n_sweeps = 0
    converged = False
    iterations = max_iterations
    for it in range(1, max_iterations + 1):
        x_prev = x
        # (1) data fidelity: one MLEM update
        Ax = A @ x
        np.maximum(Ax, _TINY, out=Ax)
        x = x * (AT @ (b / Ax)) / colsum_safe
        np.maximum(x, 0.0, out=x)
        # (2) SSR parsimony step on the energy grid
        if smooth_every > 0 and it % smooth_every == 0:
            y_ref = x.copy()
            for _ in range(max(int(inner_sweeps), 1)):
                x = _ssr_ne_sweep(E_sorted, y_ref, x, k_run, fn_try)
            n_sweeps += 1
            np.maximum(x, 0.0, out=x)
        # convergence: relative L2 change of the spectrum
        diff = float(np.linalg.norm(x - x_prev))
        base = float(np.linalg.norm(x_prev))
        if diff <= tolerance * max(base, _TINY):
            converged = True
            iterations = it
            break
    return x, iterations, converged, n_sweeps


def solve_ssr_full(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    E_MeV: np.ndarray | None = None,
    fn: str | int = "auto",
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    smooth_every: int = 1,
    inner_sweeps: int = 1,
    fn_ladder_cap: int = 8,
) -> dict[str, Any]:
    """SSR unfolding returning rich diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess; a non-positive or missing guess falls
        back to the flat ``sum(b) / n`` start.
    E_MeV : np.ndarray, optional
        Energy grid (MeV), ``n`` points.  Defaults to an equidistant
        grid ``0 .. n - 1``; unsorted grids are sorted internally.
    fn : str or int, optional
        Partial sum threshold of the SSR parsimony step.  ``"auto"``
        (default) runs Metzner's minimum statistic ladder starting at
        ``int(0.66 * partial_sum_quantile(n, k_run))`` and descending
        while the folded residuals stay sign-adequate and the spectrum
        does not gain extrema; an integer fixes the threshold.
    max_iterations : int, optional
        Maximum number of MLEM + sweep iterations (default 500).
    tolerance : float, optional
        Relative L2 stopping tolerance (default 1e-6).
    smooth_every : int, optional
        Apply the SSR sweep every ``smooth_every``-th iteration
        (default 1).
    inner_sweeps : int, optional
        Number of QSOR passes per SSR step (default 1).
    fn_ladder_cap : int, optional
        Maximum number of ladder candidates for ``fn="auto"``
        (default 8).

    Returns
    -------
    dict[str, Any]
        Diagnostics with keys ``spectrum``, ``n_iterations``,
        ``converged``, ``fn`` (threshold used), ``fn_start``, ``k_run``,
        ``n_extrema``, ``ps_valid_data``, ``run_valid_data``,
        ``max_run_data``, ``ssr_sweeps`` and ``fn_ladder`` (per
        candidate adequacy results).
    """
    A, b, x0v = validate_system(A, b, x0, max_iterations, tolerance)
    m, n = A.shape
    if m < 3:
        raise ValueError(
            "SSR unfolding requires at least 3 detector readings, "
            f"got {m}"
        )
    if n < _MIN_DATA_POINTS:
        raise ValueError(
            f"SSR unfolding requires at least {_MIN_DATA_POINTS} energy "
            f"bins, got {n}"
        )

    # --- energy grid (the SSR sweep needs ascending arguments) ------------
    inverse = None
    order = None
    if E_MeV is None:
        E_sorted = np.arange(n, dtype=float)
    else:
        E = np.asarray(E_MeV, dtype=float).ravel()
        if E.shape[0] != n:
            raise ValueError(
                f"Length of E_MeV ({E.shape[0]}) must match number of "
                f"energy bins ({n})"
            )
        if np.all(np.diff(E) > 0):
            E_sorted = E
        else:
            order = np.argsort(E, kind="stable")
            E_sorted = E[order]
            inverse = np.argsort(order)

    A_use = A if order is None else A[:, order]
    if x0v is not None and np.any(x0v > 0):
        x_init = np.asarray(x0v, dtype=float).copy()
    else:
        total = float(np.sum(b))
        x_init = np.full(n, total / n if total > 0 else 1.0)
    if order is not None:
        x_init = x_init[order]

    # --- threshold ladder ---------------------------------------------------
    k_run = max_run_quantile(n)
    fn_start = max(2, int(0.66 * float(partial_sum_quantile(n, k_run))))
    if isinstance(fn, str):
        if fn != "auto":
            raise ValueError(
                f"fn must be 'auto' or a positive integer, got {fn!r}"
            )
        bottom = max(2, fn_start - int(fn_ladder_cap) + 1)
        ladder = list(range(fn_start, bottom - 1, -1))
    elif isinstance(fn, (int, np.integer)):
        if int(fn) < 1:
            raise ValueError(
                f"fn must be 'auto' or a positive integer, got {fn}"
            )
        fn_start = int(fn)
        ladder = [fn_start]
    else:
        raise ValueError(
            f"fn must be 'auto' or a positive integer, got {fn!r}"
        )

    ladder_results: list[dict[str, Any]] = []
    first_extrema: int | None = None
    best: tuple[np.ndarray, int, bool, int, int] | None = None
    for fn_try in ladder:
        x_run, iters, conv, sweeps = _ssr_unfold_fixed_fn(
            A_use, b, x_init, E_sorted, k_run, int(fn_try),
            max_iterations, tolerance, smooth_every, inner_sweeps,
        )
        ps_ok, run_ok, max_run = _folded_adequacy(b, A_use @ x_run, k_run)
        extrema = number_of_extrema(x_run)
        ladder_results.append(
            {
                "fn": int(fn_try),
                "ps_valid_data": bool(ps_ok),
                "run_valid_data": bool(run_ok),
                "n_extrema": int(extrema),
                "n_iterations": int(iters),
                "converged": bool(conv),
            }
        )
        if first_extrema is None:
            first_extrema = extrema
        if not (ps_ok and run_ok and extrema <= first_extrema):
            break
        best = (x_run, iters, conv, int(fn_try), sweeps)

    if best is not None:
        x_sorted, iters, conv, fn_used, sweeps = best
    else:
        # Even the start model is inadequate: fall back to fn_start + 1,
        # as the R ssr_minR/ssr_ne_minR loop does in that situation.
        fn_used = fn_start + 1
        x_sorted, iters, conv, sweeps = _ssr_unfold_fixed_fn(
            A_use, b, x_init, E_sorted, k_run, fn_used,
            max_iterations, tolerance, smooth_every, inner_sweeps,
        )
        ps_ok, run_ok, max_run = _folded_adequacy(b, A_use @ x_sorted, k_run)
        ladder_results.append(
            {
                "fn": int(fn_used),
                "ps_valid_data": bool(ps_ok),
                "run_valid_data": bool(run_ok),
                "n_extrema": int(number_of_extrema(x_sorted)),
                "n_iterations": int(iters),
                "converged": bool(conv),
            }
        )

    spectrum = x_sorted if inverse is None else x_sorted[inverse]
    ps_ok, run_ok, max_run = _folded_adequacy(b, A @ spectrum, k_run)
    converged = bool(conv and np.all(np.isfinite(spectrum)))
    logger.info(
        "SSR: fn=%d (fn_start=%d), extrema=%d, data adequacy ps=%s run=%s",
        fn_used, fn_start, number_of_extrema(spectrum), ps_ok, run_ok,
    )
    return {
        "spectrum": spectrum,
        "n_iterations": int(iters),
        "converged": converged,
        "fn": int(fn_used),
        "fn_start": int(fn_start),
        "k_run": int(k_run),
        "n_extrema": int(number_of_extrema(spectrum)),
        "ps_valid_data": bool(ps_ok),
        "run_valid_data": bool(run_ok),
        "max_run_data": int(max_run),
        "ssr_sweeps": int(sweeps),
        "fn_ladder": ladder_results,
    }


def solve_ssr(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    E_MeV: np.ndarray | None = None,
    fn: str | int = "auto",
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    smooth_every: int = 1,
    inner_sweeps: int = 1,
    fn_ladder_cap: int = 8,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with SSR sign-parsimony regularisation.

    The spectrum is alternately fitted to the data with one MLEM update
    and smoothed with a non-equidistant SSR QSOR sweep whose partial sum
    threshold follows Metzner's minimum statistic (see the module
    docstring and :func:`solve_ssr_full`).

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess (flat start when missing or all-zero).
    E_MeV : np.ndarray, optional
        Energy grid (MeV); defaults to an equidistant grid.
    fn : str or int, optional
        ``"auto"`` (minimum statistic ladder) or a fixed threshold.
    max_iterations : int, optional
        Maximum number of outer iterations (default 500).
    tolerance : float, optional
        Relative L2 stopping tolerance (default 1e-6).
    smooth_every : int, optional
        SSR sweep frequency (default: every iteration).
    inner_sweeps : int, optional
        QSOR passes per SSR step (default 1).
    fn_ladder_cap : int, optional
        Maximum number of ladder candidates for ``fn="auto"``.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)``.
    """
    diag = solve_ssr_full(
        A,
        b,
        x0=x0,
        E_MeV=E_MeV,
        fn=fn,
        max_iterations=max_iterations,
        tolerance=tolerance,
        smooth_every=smooth_every,
        inner_sweeps=inner_sweeps,
        fn_ladder_cap=fn_ladder_cap,
    )
    spectrum = np.maximum(diag["spectrum"], 0.0)
    return spectrum, diag["n_iterations"], diag["converged"]


def unfold_ssr(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    fn: str | int = "auto",
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    smooth_every: int = 1,
    inner_sweeps: int = 1,
    fn_ladder_cap: int = 8,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum with SSR sign-parsimony regularisation.

    Python port of the R package ``sisireg`` (Sign-Simplicity-Regression
    solver, Metzner 2020/2021): MLEM data-fidelity steps alternate with
    non-equidistant SSR QSOR sweeps of the spectrum, and the partial sum
    threshold is selected by the minimum statistic ladder on the
    data-space sign adequacy of the folded residuals.

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
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess (flat start when missing or all-zero).
    fn : str or int, optional
        ``"auto"`` (default, minimum statistic ladder) or a fixed
        partial sum threshold.
    max_iterations : int, optional
        Maximum number of outer iterations (default 500).
    tolerance : float, optional
        Relative L2 stopping tolerance (default 1e-6).
    smooth_every : int, optional
        SSR sweep frequency (default 1).
    inner_sweeps : int, optional
        QSOR passes per SSR step (default 1).
    fn_ladder_cap : int, optional
        Maximum number of ladder candidates (default 8).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary with additional keys
        ``fn``, ``fn_start``, ``k_run``, ``n_extrema``,
        ``ps_valid_data``, ``run_valid_data`` and ``ssr_converged``.
    """
    b_vec = np.array(
        [readings[name] for name in detector_names if name in readings],
        dtype=float,
    )
    total = float(b_vec.sum()) if b_vec.size else 0.0
    default_initial = np.full(
        n_energy_bins, total / n_energy_bins if total > 0 else 1.0
    )

    def solve_wrapper(A, b, **kwargs):
        return solve_ssr(
            A,
            b,
            x0=kwargs.get("x0"),
            E_MeV=E_MeV,
            fn=fn,
            max_iterations=max_iterations,
            tolerance=tolerance,
            smooth_every=smooth_every,
            inner_sweeps=inner_sweeps,
            fn_ladder_cap=fn_ladder_cap,
        )

    try:
        diag = solve_ssr_full(
            np.array(
                [sensitivities[name] for name in detector_names
                 if name in readings],
                dtype=float,
            ),
            b_vec,
            E_MeV=E_MeV,
            fn=fn,
            max_iterations=max_iterations,
            tolerance=tolerance,
            smooth_every=smooth_every,
            inner_sweeps=inner_sweeps,
            fn_ladder_cap=fn_ladder_cap,
        )
        extra_output = {
            "fn": diag["fn"],
            "fn_start": diag["fn_start"],
            "k_run": diag["k_run"],
            "n_extrema": diag["n_extrema"],
            "ps_valid_data": diag["ps_valid_data"],
            "run_valid_data": diag["run_valid_data"],
            "ssr_converged": diag["converged"],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("SSR diagnostics unavailable: %s", exc)
        extra_output = None

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=default_initial,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="SSR",
        extra_output=extra_output,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
