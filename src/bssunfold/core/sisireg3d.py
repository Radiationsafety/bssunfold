"""Spatial SSR regression: Python port of the sisireg ``ssr3d`` family.

This module is a faithful port of the spatial part of the R package
``sisireg`` 1.2.1 (Lars Metzner, CRAN, GPL>=2;
``https://cran.r-project.org/package=sisireg``), file ``R/ssr3d.R`` and
C kernel ``src/ssr3d.c``: the Sign-Simplicity-Regression (SSR) model of
Metzner (2020, 2021) applied to scattered data ``z(x, y)`` on the plane.
The regression function is the most parsimonious (minimal) surface that
is statistically adequate with respect to the *sign* of the residuals
in the k-quadrant-neighbourhood of every observation:

* the **k-quadrant-neighbourhood** of a reference point collects the
  ``k`` nearest observations in each of the four quadrants north-east
  (``x > x0, y >= y0``), north-west (``x <= x0, y > y0``), south-west
  (``x < x0, y <= y0``) and south-east (``x >= x0, y < y0``) of the
  reference point [R ``nearneighborsQR``];

* the **k-neighbourhood** collects the ``4k + 1`` nearest observations
  (the reference point included) regardless of quadrants and is used
  for the partial sum criterion [R ``nearneighborsR``];

* a candidate value ``mu_i`` is *adequate* when the absolute sum of
  residual signs ``sum(sign(z - mu))`` stays within the threshold
  ``fn`` in every neighbourhood containing point ``i``
  [C ``chi`` / R ``check_ps``-style test];

* the surface is initialised with exponential-distance-weighted means
  of the observations in the partial sum neighbourhoods, inadequate
  start values fall back to the observations themselves, and a
  Gauss-Seidel iteration then replaces every point by the
  exponential-distance-weighted mean of its quadrant neighbours,
  *reverting* the update to the observation whenever the new value
  violates the adequacy test (C ``ssr3dC``).

The port reproduces the R/C semantics exactly, including

* the fractional default ``k = maxRunR(n) / 2`` (true division in R)
  with the truncating ``as.integer`` conversions of ``head``/``seq_len``
  (``int(k)`` neighbours per quadrant, ``int(4k + 1)`` neighbourhood
  size), and the float comparison ``len(nb) == 4k + 1`` in
  ``psmax3dR`` which skips neighbourhoods inflated by distance ties;
* the value-matching semantics of R ``which(d %in% head(sort(d), m))``
  (all points whose distance value is among the ``m`` smallest —
  distance ties inflate the neighbourhood);
* the order of the aggregated quadrant indices (k = 1: NE, NW, SW, SE;
  k > 1: coincident points first, then NE, NW, SW, SE) with duplicate
  removal, which fixes the point order used by the minimal-surface
  prediction;
* the ``0/0`` result of the weighted means when every neighbour sits at
  distance zero (duplicate coordinates), i.e. ``NaN`` propagation with
  sign ``(NaN) = 0`` as in the C macro ``(x > 0) - (x < 0)``.

Module API (R names in brackets):

* ``point_distance(coords, x)`` [``pointDistanceR``],
* ``near_neighbors_quadrant(x, coords, k=1)`` [``nearneighborsQR``],
* ``near_neighbors_grid_quadrant(coords, k)`` [``nearneighboursGridQR``],
* ``near_neighbors(x, coords, k)`` [``nearneighborsR``],
* ``near_neighbors_grid(coords, k)`` [``nearneighboursGridR``],
* ``ps_max_3d(nb, z, mu, k)`` [``psmax3dR``],
* ``ps_statistic_3d(coords, z, mu, max_int=None)`` [computation behind
  ``psplot3d``],
* ``wmean(coords, mu, x)`` [``wmeanR``],
* ``wmean_exp(coords, mu, x)`` [``wmean_expR``],
* ``wmean_ms(coords, mu, x)`` [``wmean_msR``],
* ``ssr3d(coords, dat, k=None, fn=None, iter=1000)`` [``ssr3d`` +
  C ``ssr3dC``],
* ``ssr3d_predict(model, xy, ms=False)`` [``ssr3d_predict``].

The SSR building blocks shared with the 1-D port (``max_run_quantile``,
``partial_sum_quantile``) are re-exported from
:mod:`bssunfold.core.unfold_ssr`.

Notes
-----
* Pure NumPy implementation; the Gauss-Seidel sweeps are O(n) sequential
  passes, so Numba adds no benefit at typical scattered-data sizes.
* ``ssr3d`` requires at least 5 observations (one candidate per
  quadrant); R returns silently degenerate results for smaller samples.
* The R package is GPL (>= 2), compatible with the GPL-3 license of
  bssunfold.

References
----------
.. [1] L. Metzner, *Trendbasierte Prognostik*, ISBN 979-8-68239-420-3,
       2020.
.. [2] L. Metzner, *Adaequates Maschinelles Lernen*, ISBN
       979-8-59347-027-0, 2021.
.. [3] L. Metzner, "sisireg: Sign-Simplicity-Regression-Solver", R
       package version 1.2.1, CRAN, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unfold_ssr import max_run_quantile, partial_sum_quantile

__all__ = [
    "SSR3DModel",
    "max_run_quantile",
    "partial_sum_quantile",
    "point_distance",
    "near_neighbors_quadrant",
    "near_neighbors_grid_quadrant",
    "near_neighbors",
    "near_neighbors_grid",
    "ps_max_3d",
    "ps_statistic_3d",
    "wmean",
    "wmean_exp",
    "wmean_ms",
    "ssr3d",
    "ssr3d_predict",
]

# ssr3d needs one candidate per quadrant; R degrades silently below.
_MIN_DATA_POINTS_3D = 5


def _sgn(v) -> np.ndarray:
    """Sign exactly as the C macro ``(x > 0) - (x < 0)`` (NaN -> 0)."""
    v = np.asarray(v, dtype=float)
    return (v > 0.0).astype(np.int64) - (v < 0.0).astype(np.int64)


def _as_coords(coords) -> np.ndarray:
    """Validate and return the coordinates as an ``(n, 2)`` float array."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (n, 2), got {coords.shape}"
        )
    return coords


def _as_xy(x) -> np.ndarray:
    """Validate and return a single query point as shape ``(2,)``."""
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != 2:
        raise ValueError(f"query point must have 2 components, got {x.shape[0]}")
    return x


@dataclass
class SSR3DModel:
    """Minimal surface model returned by :func:`ssr3d`.

    Attributes
    ----------
    koord : np.ndarray
        Observation coordinates, shape ``(n, 2)`` (R ``df$koord.x/y``).
    z : np.ndarray
        Observed values, shape ``(n,)`` (R ``df$z``).
    mu : np.ndarray
        Regression (minimal surface) values, shape ``(n,)`` (R ``df$mu``).
    """

    koord: np.ndarray
    z: np.ndarray
    mu: np.ndarray


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------


def point_distance(coords, x) -> np.ndarray:
    """Euclidean distances from every coordinate row to ``x``.

    Port of R ``pointDistanceR`` (``lonlat = FALSE`` branch).

    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape ``(n, 2)``.
    x : array_like
        Reference point, shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Distances, shape ``(n,)``.
    """
    coords = _as_coords(coords)
    x = _as_xy(x)
    return np.sqrt(np.sum((coords - x) ** 2, axis=1))


def near_neighbors_quadrant(x, coords, k: float = 1) -> np.ndarray:
    """k nearest neighbours per quadrant of the reference point.

    Port of R ``nearneighborsQR``.  The four quadrants are

    * Q1 north-east: ``x > x0 and y >= y0`` (strict, semi-strict),
    * Q2 north-west: ``x <= x0 and y > y0``,
    * Q3 south-west: ``x < x0 and y <= y0``,
    * Q4 south-east: ``x >= x0 and y < y0``.

    In every quadrant the ``k`` nearest points are kept, where ``k`` is
    truncated to ``int(k)`` exactly as R ``head``/``seq_len`` do for a
    fractional ``k`` (the default ``k = maxRunR(n)/2`` of ``ssr3d`` is
    fractional whenever ``maxRunR(n)`` is odd).  Following the R
    ``which(d %in% head(sort(d1), k))`` construction, all points whose
    distance *value* is among the ``k`` smallest quadrant distances are
    kept, so distance ties inflate a quadrant beyond ``k`` points.

    The reference point itself is never a quadrant neighbour; for
    ``k > 1`` coincident points (distance 0) are prepended first, as in
    the R aggregation ``c(nb0, nb1, nb2, nb3, nb4)``.

    Parameters
    ----------
    x : array_like
        Reference point, shape ``(2,)``.
    coords : np.ndarray
        Coordinates, shape ``(n, 2)``.
    k : float, optional
        Neighbours per quadrant (default 1, truncated to ``int(k)``).

    Returns
    -------
    np.ndarray
        0-based indices into ``coords``, aggregated in the order Q1,
        Q2, Q3, Q4 (coincident points first when ``k > 1``), duplicate
        indices removed preserving the first occurrence.
    """
    coords = _as_coords(coords)
    x = _as_xy(x)
    d = point_distance(coords, x)
    x0, y0 = float(x[0]), float(x[1])
    cx, cy = coords[:, 0], coords[:, 1]
    kq = int(k)  # R head(x, n)/seq_len truncation for fractional n
    masks = (
        (cx > x0) & (cy >= y0),  # Q1 NE
        (cx <= x0) & (cy > y0),  # Q2 NW
        (cx < x0) & (cy <= y0),  # Q3 SW
        (cx >= x0) & (cy < y0),  # Q4 SE
    )
    nb: list[int] = []
    for mask in masks:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        dq = d[idx]
        if kq < dq.size:
            thresh = np.sort(dq)[:kq][-1]
            idx = idx[dq <= thresh]
        nb.extend(int(i) for i in idx)
    if k != 1:
        # R: nb0 (coincident points) only enters for k > 1, and first.
        nb = [int(i) for i in np.flatnonzero(d == 0.0)] + nb
    seen: set[int] = set()
    out: list[int] = []
    for i in nb:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def near_neighbors_grid_quadrant(coords, k: float = 1) -> list[np.ndarray]:
    """k-quadrant-neighbourhood for every coordinate row.

    Port of R ``nearneighboursGridQR``.

    Returns
    -------
    list[np.ndarray]
        One 0-based index array per observation.
    """
    coords = _as_coords(coords)
    return [
        near_neighbors_quadrant(coords[i], coords, k=k) for i in range(coords.shape[0])
    ]


def near_neighbors(x, coords, k: float = 1) -> np.ndarray:
    """``4k + 1`` nearest neighbours of the reference point.

    Port of R ``nearneighborsR``: the reference point itself (distance
    0) is included, and distance ties inflate the neighbourhood beyond
    ``4k + 1`` points (value matching of R ``d %in% head(sort(d), 4k+1)``).
    The neighbourhood size ``4k + 1`` is truncated to ``int(4k + 1)``
    for fractional ``k`` exactly as R ``head`` does.

    Parameters
    ----------
    x : array_like
        Reference point, shape ``(2,)``.
    coords : np.ndarray
        Coordinates, shape ``(n, 2)``.
    k : float, optional
        Neighbours per (virtual) quadrant; the neighbourhood size is
        ``4k + 1`` (default 1).

    Returns
    -------
    np.ndarray
        0-based indices into ``coords``, ordered by ascending distance.
    """
    coords = _as_coords(coords)
    x = _as_xy(x)
    d = point_distance(coords, x)
    m = int(4.0 * float(k) + 1.0)  # R head truncation
    ds = np.sort(d)
    if m >= d.size:
        return np.argsort(d, kind="stable").astype(np.int64)
    thresh = ds[m - 1]
    return np.flatnonzero(d <= thresh).astype(np.int64)


def near_neighbors_grid(coords, k: float = 1) -> list[np.ndarray]:
    """k-neighbourhood for every coordinate row.

    Port of R ``nearneighboursGridR``.

    Returns
    -------
    list[np.ndarray]
        One 0-based index array per observation.
    """
    coords = _as_coords(coords)
    return [near_neighbors(coords[i], coords, k=k) for i in range(coords.shape[0])]


# ---------------------------------------------------------------------------
# Partial sums
# ---------------------------------------------------------------------------


def ps_max_3d(nb, z, mu, k: float) -> int:
    """Maximum absolute partial sum over the given neighbourhoods.

    Port of R ``psmax3dR``: for every neighbourhood with exactly
    ``4k + 1`` members (float comparison, so neighbourhoods inflated by
    distance ties are skipped exactly as in R) the absolute sum of the
    residual signs is accumulated; the maximum over all qualifying
    neighbourhoods is returned (0 when none qualifies).

    Parameters
    ----------
    nb : sequence of np.ndarray
        Neighbourhood index lists (as returned by
        :func:`near_neighbors_grid`).
    z : np.ndarray
        Observed values.
    mu : np.ndarray
        Model values.
    k : float
        Neighbourhood size parameter; the expected membership is
        ``4k + 1``.

    Returns
    -------
    int
        Maximum absolute partial sum (0 if no neighbourhood qualifies).
    """
    z = np.asarray(z, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    if z.shape[0] != mu.shape[0]:
        raise ValueError(
            f"z and mu must have the same length, got {z.shape[0]} and {mu.shape[0]}"
        )
    size = 4.0 * float(k) + 1.0
    best = 0
    for members in nb:
        members = np.asarray(members, dtype=np.int64)
        if members.size != size:  # float comparison as in R
            continue
        s = int(np.sum(_sgn(z[members] - mu[members])))
        if abs(s) > best:
            best = abs(s)
    return best


def ps_statistic_3d(coords, z, mu, max_int: int | None = None):
    """Partial sums of a spatial SSR model for all neighbourhood sizes.

    Computation behind R ``psplot3d`` (without the plot): for every
    neighbourhood size ``k = 1..max_int`` the maximum absolute partial
    sum ``ps[k]`` of the residual signs over the ``4k + 1``-nearest-
    neighbour neighbourhoods is computed, together with the 95%
    quantiles ``fn`` of the partial sums at the matching neighbourhood
    sizes ``4k + 1``.  A model whose ``ps`` stays below ``fn`` is
    statistically adequate.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape ``(n, 2)``.
    z : np.ndarray
        Observed values.
    mu : np.ndarray
        Model values.
    max_int : int, optional
        Largest neighbourhood size parameter; defaults to
        ``int(max(n / 20, 10))`` as in R.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ps, fn)`` — partial sums, shape ``(max_int,)``, and quantile
        curve at the sizes ``seq(5, 4*max_int + 1, by=4)``.
    """
    coords = _as_coords(coords)
    z = np.asarray(z, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    n = z.shape[0]
    if mu.shape[0] != n:
        raise ValueError(
            f"z and mu must have the same length, got {n} and {mu.shape[0]}"
        )
    if max_int is None:
        max_int = int(max(n / 20.0, 10.0))
    max_int = int(max_int)
    ps = np.zeros(max_int, dtype=np.int64)
    for ki in range(1, max_int + 1):
        nb = near_neighbors_grid(coords, ki)
        ps[ki - 1] = ps_max_3d(nb, z, mu, ki)
    fn = partial_sum_quantile(n, np.arange(5.0, 4.0 * max_int + 2.0, 4.0))
    fn = np.asarray(fn, dtype=float)
    return ps, fn


# ---------------------------------------------------------------------------
# Weighted means
# ---------------------------------------------------------------------------


def wmean(coords, mu, x):
    """Reciprocal-distance-weighted mean (R ``wmeanR``).

    Returns ``mu`` of the first coincident point when the query sits on
    an observation (R ``if (0 %in% d) return (mu[which(d == 0)])``),
    otherwise ``sum(mu / d) / sum(1 / d)``.
    """
    coords = _as_coords(coords)
    mu = np.asarray(mu, dtype=float).ravel()
    x = _as_xy(x)
    if coords.shape[0] != mu.shape[0]:
        raise ValueError(
            f"coords and mu must have the same length, got {coords.shape[0]} "
            f"and {mu.shape[0]}"
        )
    d = point_distance(coords, x)
    hit = np.flatnonzero(d == 0.0)
    if hit.size:
        return float(mu[hit[0]])
    return float(np.sum(mu / d) / np.sum(1.0 / d))


def wmean_exp(coords, mu, x):
    """Exponential-distance-weighted mean (R ``wmean_expR``).

    ``sum(exp(-d) * mu) / sum(exp(-d))`` with ``d`` the Euclidean
    distances from the query to the coordinates.  Unlike the C
    iteration kernel there is no distance-0 exclusion here: a
    coincident observation contributes with weight ``exp(0) = 1``.
    """
    coords = _as_coords(coords)
    mu = np.asarray(mu, dtype=float).ravel()
    x = _as_xy(x)
    if coords.shape[0] != mu.shape[0]:
        raise ValueError(
            f"coords and mu must have the same length, got {coords.shape[0]} "
            f"and {mu.shape[0]}"
        )
    d = point_distance(coords, x)
    g = np.exp(-d)
    denom = np.sum(g)
    if denom == 0.0:  # pragma: no cover - exp(-d) > 0 for finite d
        return float("nan")
    return float(np.sum(g * mu) / denom)


def wmean_ms(coords, mu, x):
    """Weighted mean based on the 4-point minimal surface (R ``wmean_msR``).

    The four coordinates are interpreted in the aggregation order of
    :func:`near_neighbors_quadrant` with ``k = 1`` — ``x1`` north-east,
    ``x2`` north-west, ``x3`` south-west, ``x4`` south-east — and the
    query is interpolated bilinearly, with the interpolation fractions
    ``rx``/``ry`` derived from the triangle heights (Heron's formula)
    of the query over the west/east and south/north edge segments.
    Fewer than four coordinates (or degenerate geometry) propagate
    ``NaN`` exactly as the R data-frame NA indexing does.

    Parameters
    ----------
    coords : np.ndarray
        The four neighbour coordinates, shape ``(4, 2)``.
    mu : np.ndarray
        The four model values.
    x : array_like
        Query point.

    Returns
    -------
    float
        Interpolated value (``NaN`` for degenerate configurations).
    """
    coords = _as_coords(coords)
    mu = np.asarray(mu, dtype=float).ravel()
    x = _as_xy(x)
    if coords.shape[0] != 4 or mu.shape[0] != 4:
        # R: data-frame row indexing out of bounds yields NA rows.
        return float("nan")
    x1, x2, x3, x4 = coords[0], coords[1], coords[2], coords[3]

    with np.errstate(invalid="ignore", divide="ignore"):
        # x axis: heights of the query over the segments (x2, x3) and (x1, x4)
        dx2 = float(np.sqrt(np.sum((x2 - x) ** 2)))
        dx3 = float(np.sqrt(np.sum((x3 - x) ** 2)))
        d23 = float(np.sqrt(np.sum((x2 - x3) ** 2)))
        s23 = (dx2 + dx3 + d23) / 2.0
        prod23 = s23 * (s23 - dx3) * (s23 - dx2) * (s23 - d23)
        h23 = 0.0 if prod23 < 0.0 else 2.0 / d23 * np.sqrt(abs(prod23))

        dx1 = float(np.sqrt(np.sum((x1 - x) ** 2)))
        dx4 = float(np.sqrt(np.sum((x4 - x) ** 2)))
        d14 = float(np.sqrt(np.sum((x1 - x4) ** 2)))
        s14 = (dx1 + dx4 + d14) / 2.0
        prod14 = s14 * (s14 - dx4) * (s14 - dx1) * (s14 - d14)
        h14 = 0.0 if prod14 < 0.0 else 2.0 / d14 * np.sqrt(abs(prod14))

        # y axis: heights over the segments (x3, x4) and (x1, x2)
        d34 = float(np.sqrt(np.sum((x3 - x4) ** 2)))
        s34 = (dx3 + dx4 + d34) / 2.0
        prod34 = s34 * (s34 - dx3) * (s34 - dx4) * (s34 - d34)
        h34 = 0.0 if prod34 < 0.0 else 2.0 / d34 * np.sqrt(abs(prod34))

        d12 = float(np.sqrt(np.sum((x1 - x2) ** 2)))
        s12 = (dx1 + dx2 + d12) / 2.0
        prod12 = s12 * (s12 - dx1) * (s12 - dx2) * (s12 - d12)
        # R quirk kept verbatim: this branch tests < 0.001, not < 0.
        h12 = 0.0 if prod12 < 0.001 else 2.0 / d12 * np.sqrt(abs(prod12))

        # x-intercept based on x3
        rx = h23 / (h23 + h14)
        # y-intercept based on x3
        ry = h34 / (h34 + h12)
        if h34 + h12 - max(d14, d23) > 0.1:
            if abs(h34 - h12) < 0.1:
                ry = 0.5
            else:
                ry = h34 / (h34 - h12)

        # line-segment west-east
        a1 = mu[2] + rx * (mu[3] - mu[2])
        b1 = mu[1] + rx * (mu[0] - mu[1])
        m1 = a1 + ry * (b1 - a1)
    return float(m1)


# ---------------------------------------------------------------------------
# The ssr3d solver
# ---------------------------------------------------------------------------


def _wmean_exp_nb(coords: np.ndarray, values: np.ndarray, nb: list, i: int) -> float:
    """C ``wmean_exp`` over the neighbourhood ``nb[i]`` of point ``i``.

    Exponential-distance-weighted mean of ``values`` at the neighbours,
    with the reference point itself (distance 0) excluded; an empty
    weight sum (all neighbours coincident) yields ``NaN`` as in C.
    """
    members = nb[i]
    if members.size == 0:
        return float("nan")
    d = np.sqrt(np.sum((coords[members] - coords[i]) ** 2, axis=1))
    keep = d != 0.0
    if not np.any(keep):
        return float("nan")
    g = np.exp(-d[keep])
    return float(np.sum(g * values[members][keep]) / np.sum(g))


def _chi_3d(nb: list, z: np.ndarray, mu: np.ndarray, ind: int, fn: float) -> bool:
    """Adequacy test of C ``chi`` for point ``ind``.

    Valid when every neighbourhood containing ``ind`` has an absolute
    residual sign sum within the threshold ``fn``.
    """
    for members in nb:
        if ind in members:
            s = int(np.sum(_sgn(z[members] - mu[members])))
            if abs(s) > fn:
                return False
    return True


def _membership_index(nb, n: int) -> list[np.ndarray]:
    """Inverted neighbourhood index: for every point the ids of the
    neighbourhoods containing it (equivalent to the membership scan of
    C ``chi``, but O(1) per lookup)."""
    idx: list[list[int]] = [[] for _ in range(n)]
    for j, members in enumerate(nb):
        for m in members:
            idx[int(m)].append(j)
    return [np.asarray(v, dtype=np.int64) for v in idx]


def ssr3d(
    coords,
    dat,
    k: float | None = None,
    fn: float | None = None,
    iter: int = 1000,
) -> SSR3DModel:
    """Minimal surface SSR regression for scattered planar data.

    Port of R ``ssr3d`` + C ``ssr3dC``.  The regression surface is
    initialised with the exponential-distance-weighted means of the
    observations in the ``4k + 1``-nearest-neighbourhoods; inadequate
    start values (partial sum criterion violated in any containing
    neighbourhood) fall back to the observations.  A Gauss-Seidel
    iteration then replaces every point by the exponential-distance-
    weighted mean of its k-quadrant-neighbourhood, reverting the update
    to the observation whenever the new value violates the adequacy
    test, until the surface changes by less than ``1e-7`` everywhere or
    ``iter`` sweeps are exhausted.

    Parameters
    ----------
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    dat : np.ndarray
        Observed values, shape ``(n,)``.
    k : float, optional
        Neighbours per quadrant for the surface neighbourhoods
        (``int(k)`` after truncation) and quarter of the partial sum
        neighbourhood size (``int(4k + 1)``).  Defaults to
        ``max_run_quantile(n) / 2`` (true division, as in R — fractional
        when ``max_run_quantile(n)`` is odd).
    fn : float, optional
        Partial sum threshold; defaults to
        ``partial_sum_quantile(n, 4k)``.
    iter : int, optional
        Maximum number of Gauss-Seidel sweeps (default 1000).

    Returns
    -------
    SSR3DModel
        Model with the coordinates, observations and regression values;
        pass it to :func:`ssr3d_predict`.
    """
    coords = _as_coords(coords)
    dat = np.asarray(dat, dtype=float).ravel()
    n = dat.shape[0]
    if coords.shape[0] != n:
        raise ValueError(
            f"coords and dat must have the same length, got {coords.shape[0]} "
            f"and {n}"
        )
    if n < _MIN_DATA_POINTS_3D:
        raise ValueError(
            f"ssr3d requires at least {_MIN_DATA_POINTS_3D} data points, got {n}"
        )
    if k is None:
        k = max_run_quantile(n) / 2.0  # R true division, may be fractional
    k = float(k)
    if fn is None:
        fn = float(partial_sum_quantile(n, 4.0 * k))
    fn = float(fn)
    iter = int(iter)

    nb = near_neighbors_grid(coords, k)  # for the partial sums
    nb1 = near_neighbors_grid_quadrant(coords, 1)  # for the surface
    pt_nbs = _membership_index(nb, n)

    # The per-neighbourhood residual sign sums are maintained
    # incrementally (integer-exact): s_all[j] == sum(sign(dat - mu)) over
    # nb[j] holds at any time, so the adequacy test equals C ``chi``.
    mu = np.empty(n, dtype=float)
    for i in range(n):
        mu[i] = _wmean_exp_nb(coords, dat, nb, i)
    s_all = np.asarray(
        [int(np.sum(_sgn(dat[np.asarray(m)] - mu[np.asarray(m)]))) for m in nb],
        dtype=np.int64,
    )

    def _set(i: int, value: float) -> None:
        """Assign ``mu[i]`` and propagate the residual sign change."""
        old_s = _sgn(dat[i] - mu[i])
        mu[i] = value
        new_s = _sgn(dat[i] - mu[i])
        if new_s != old_s:
            for j in pt_nbs[i]:
                s_all[j] += new_s - old_s

    def _adequate(i: int) -> bool:
        return all(abs(int(s_all[j])) <= fn for j in pt_nbs[i])

    for i in range(n):
        if not _adequate(i):
            _set(i, dat[i])

    for _ in range(max(iter, 0)):
        change = False
        for i in range(n):
            oldval = mu[i]
            newval = _wmean_exp_nb(coords, mu, nb1, i)
            _set(i, newval)
            if not _adequate(i):
                if _sgn(dat[i] - newval) == _sgn(dat[i] - oldval):
                    pass  # keep newval
                else:
                    _set(i, dat[i])
            if abs(mu[i] - oldval) > 1e-7:
                change = True
        if not change:
            break
    return SSR3DModel(koord=coords, z=dat, mu=mu)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def ssr3d_predict(model: SSR3DModel, xy, ms: bool = False) -> np.ndarray:
    """Predict the surface at arbitrary query points.

    Port of R ``ssr3d_predict``.  For every query point the
    k-quadrant-neighbourhood (``k = 1``) within the model coordinates is
    determined and aggregated either with exponential distance weights
    (default, R ``predict3dSingleR``) or by the 4-point minimal surface
    interpolation (``ms=True``, R ``predict3dSingle_msR``; requires all
    four quadrants to be populated, otherwise ``NaN``).

    Parameters
    ----------
    model : SSR3DModel
        Model returned by :func:`ssr3d`.
    xy : np.ndarray
        Query points, shape ``(m, 2)``.
    ms : bool, optional
        Use the minimal-surface interpolation instead of the
        exponential weighting (default False).

    Returns
    -------
    np.ndarray
        Predictions, shape ``(m,)``.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (m, 2), got {xy.shape}")
    out = np.empty(xy.shape[0], dtype=float)
    for j in range(xy.shape[0]):
        x = xy[j]
        nb1 = near_neighbors_quadrant(x, model.koord, k=1)
        if ms:
            out[j] = wmean_ms(model.koord[nb1], model.mu[nb1], x)
        else:
            out[j] = wmean_exp(model.koord[nb1], model.mu[nb1], x)
    return out
