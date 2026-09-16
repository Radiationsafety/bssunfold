"""SSR-MLP: two-layer perceptron trained with the partial sum criterion.

Python port of the R package ``sisireg`` 1.2.1 (Lars Metzner, CRAN,
GPL>=2; ``https://cran.r-project.org/package=sisireg``), file
``R/ssrMLP.R``: a 2-hidden-layer perceptron with sigmoid activations
and linear output whose training error is *not* the least squares
residual but Metzner's partial sum criterion of the Sign-Simplicity-
Regression model — the sum of the positive parts of the absolute
neighbourhood residual sign sums above the threshold ``fn`` —
optionally combined with least squares (``opt='ps_lse'``), with an
L1-type curvature penalty of the learned function over the input
neighbourhoods (``opt='ps_l1'``), with plain least squares
(``opt='lse'``) or with user-supplied error/factor functions
(``opt='ext'``).

The training minimises the chosen criterion by per-sample stochastic
gradient descent (one fixed random permutation of the samples, as in R
``sample(nrow(X))``), keeping the best (lowest-error) weight set.

Faithfulness notes
------------------
The port reproduces the R semantics exactly, including two quirks of
the original implementation that a "cleaned up" port would silently
lose:

* the hidden-layer update computes ``dW1 = t(t(temp) %*% O1)`` where R
  coerces the vector ``O1`` to the *row* matrix ``t(O1)`` — the outer
  product ``outer(temp, O1)`` — and then transposes it, so the weight
  increment added to ``W1`` (shape ``(hl2, hl1)``) is the *transposed*
  gradient (shape ``(hl1, hl2)``).  The addition is only conformable
  for square layer configurations ``hl[0] == hl[1]`` (the default);
  asymmetric hidden layers raise ``ValueError`` exactly as the R
  non-conformable-array error;
* the ``fn`` and ``alpha`` arguments of ``ssrmlp_train`` are accepted
  for API fidelity but are *never forwarded* to the error/factor
  functions (in R the calls ``errfct(nb, nb_dst, il, Y, Yp)`` /
  ``facfct(nb, nb_dst, il, Y, Yp, i)`` pass neither), so the effective
  threshold and mixing weights are the function defaults
  ``fn = 4`` and ``alpha = 1e-4``.  Custom values are possible through
  ``opt='ext'`` with suitably wrapped callables;
* unknown ``opt`` values silently fall back to the default ``'ps'``
  branch of the R ``switch``;
* the re-training path (``W`` given) does *not* clean ``NaN`` from the
  re-normalised inputs (only the fresh standardisation path does),
  exactly as in R.

Module API (R names in brackets):

* ``sigmoid(x)`` [``sigmoidR``], ``d1sigmoid(x)`` [``d1sigmoidR``],
* ``check_ps(nb, Y, Yp, i, fn=4)`` [``check_psR``],
* ``err_ps`` / ``fac_ps`` [``err_psR`` / ``fac_psR``],
* ``err_lse`` / ``fac_lse`` [``err_lseR`` / ``fac_lseR``],
* ``err_ps_lse`` / ``fac_ps_lse`` [``err_ps_lseR`` / ``fac_ps_lseR``],
* ``err_ps_l1`` / ``fac_ps_l1`` [``err_ps_l1R`` / ``fac_ps_l1R``],
* ``calc_out(X, W0, W1, W2)`` [``calcOutR``],
* ``ssrmlp_train(X, Y, ...)`` [``ssrmlp_train``],
* ``ssrmlp_predict(X, W)`` [``ssrmlp_predict``],
* ``fii_model(W)`` [``fii_model``],
* ``fii_prediction(W, X)`` [``fii_prediction``].

Notes
-----
* Pure NumPy; the neighbourhood sign sums of the training loop are
  maintained incrementally (integer-exact) so the per-sample factor
  evaluation matches the direct recomputation of :func:`fac_ps` and
  friends while staying O(neighborhood) instead of O(n).
* The R package is GPL (>= 2), compatible with the GPL-3 license of
  bssunfold.

References
----------
.. [1] L. Metzner, *Adaequates Maschinelles Lernen*, ISBN
       979-8-59347-027-0, 2021.
.. [2] L. Metzner, "sisireg: Sign-Simplicity-Regression-Solver", R
       package version 1.2.1, CRAN, 2025.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..logging_config import get_logger

__all__ = [
    "SSRMLPModel",
    "sigmoid",
    "d1sigmoid",
    "check_ps",
    "err_ps",
    "fac_ps",
    "err_lse",
    "fac_lse",
    "err_ps_lse",
    "fac_ps_lse",
    "err_ps_l1",
    "fac_ps_l1",
    "calc_out",
    "ssrmlp_train",
    "ssrmlp_predict",
    "fii_model",
    "fii_prediction",
]

logger = get_logger("sisireg_mlp")

# Defaults of the R error/factor functions; these are the *effective*
# criterion parameters of ssrmlp_train (see module docstring).
_DEFAULT_FN = 4.0
_DEFAULT_ALPHA = 1e-4


def _sgn(v: float) -> int:
    """Sign of a scalar exactly as the C macro ``(x > 0) - (x < 0)``."""
    v = float(v)
    return (v > 0.0) - (v < 0.0)


def _sgn_arr(v) -> np.ndarray:
    """Elementwise sign as the C macro ``(x > 0) - (x < 0)`` (NaN -> 0)."""
    v = np.asarray(v, dtype=float)
    return (v > 0.0).astype(np.int64) - (v < 0.0).astype(np.int64)


@dataclass
class SSRMLPModel:
    """Trained 2-layer perceptron (the R ``list(W0, W1, W2, ...)``).

    Attributes
    ----------
    W0 : np.ndarray
        First-layer weights, shape ``(hl1, dim + 1)`` (bias column last).
    W1 : np.ndarray
        Second-layer weights, shape ``(hl2, hl1)``.
    W2 : np.ndarray
        Output weights, shape ``(1, hl2)``.
    minX, maxX : np.ndarray
        Per-input-column standardisation bounds, shape ``(dim,)``.
    minY, maxY : float
        Output standardisation bounds.
    """

    W0: np.ndarray
    W1: np.ndarray
    W2: np.ndarray
    minX: np.ndarray
    maxX: np.ndarray
    minY: float
    maxY: float


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def sigmoid(x):
    """Logistic function ``1 / (1 + exp(-x))`` (R ``sigmoidR``)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def d1sigmoid(x):
    """First derivative of :func:`sigmoid` (R ``d1sigmoidR``)."""
    s = sigmoid(x)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Error functions and their factor (derivative) functions
# ---------------------------------------------------------------------------


def check_ps(nb, Y, Yp, i: int, fn: float = _DEFAULT_FN) -> bool:
    """Partial sum adequacy of the prediction at sample ``i``.

    Port of R ``check_psR``: valid (``True``) when *every* input-space
    neighbourhood containing ``i`` has an absolute residual sign sum
    ``|sum(sign(Y - Yp))|`` within the threshold ``fn``.

    Parameters
    ----------
    nb : sequence of np.ndarray
        Input-space neighbourhoods (index lists sorted by distance,
        the centre first), as built by :func:`ssrmlp_train`.
    Y, Yp : np.ndarray
        Target and predicted values.
    i : int
        Sample index to check.
    fn : float, optional
        Partial sum threshold (default 4, the R function default).
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    for members in nb:
        members = np.asarray(members, dtype=np.int64)
        if i in members:
            s = int(np.sum(np.sign(Y[members] - Yp[members])))
            if abs(s) > fn:
                return False
    return True


def _ps_part(nb, Y, Yp, fn: float) -> float:
    """Sum of the positive parts of the absolute neighbourhood sign sums."""
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    s = 0.0
    for members in nb:
        members = np.asarray(members, dtype=np.int64)
        tmp = int(np.sum(np.sign(Y[members] - Yp[members])))
        s += max(0.0, abs(tmp) - fn)
    return s


def err_ps(nb, nb_dst, dim, Y, Yp, fn: float = _DEFAULT_FN) -> float:
    """Partial sum error (R ``err_psR``).

    ``sum_j max(0, |sum sign(Y - Yp)| - fn)`` over all neighbourhoods.
    ``nb_dst`` and ``dim`` are unused and kept for the uniform R call
    convention ``errfct(nb, nb_dst, dim, Y, Yp)``.
    """
    return _ps_part(nb, Y, Yp, fn)


def fac_ps(nb, nb_dst, dim, Y, Yp, i: int, fn: float = _DEFAULT_FN) -> float:
    """Factor of the partial sum error w.r.t. ``Yp[i]`` (R ``fac_psR``).

    ``0`` when the prediction at ``i`` is adequate
    (:func:`check_ps`), otherwise
    ``sign(Y[i] - Yp[i]) * |Y[i] - Yp[i]|**0.01``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    if check_ps(nb, Y, Yp, i, fn=fn):
        return 0.0
    return _sgn(Y[i] - Yp[i]) * abs(Y[i] - Yp[i]) ** 0.01


def err_lse(nb, nb_dst, dim, Y, Yp) -> float:
    """Mean squared residual error (R ``err_lseR``).

    ``sum((Y - Yp)**2) / n`` — the mean, unlike the *sum* used inside
    :func:`err_ps_lse` (quirk of the original kept verbatim).
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    return float(np.sum((Y - Yp) ** 2) / Y.shape[0])


def fac_lse(nb, nb_dst, dim, Y, Yp, i: int) -> float:
    """Factor of the LSE error w.r.t. ``Yp[i]`` (R ``fac_lseR``)."""
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    return float(Y[i] - Yp[i])


def err_ps_lse(
    nb, nb_dst, dim, Y, Yp, fn: float = _DEFAULT_FN, alpha: float = _DEFAULT_ALPHA
) -> float:
    """Combined partial sum + least squares error (R ``err_ps_lseR``).

    ``err_ps(...) + alpha * sum((Yp - Y)**2)`` — note the *sum* (not
    the mean) of the squared residuals, as in R.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    return _ps_part(nb, Y, Yp, fn) + alpha * float(np.sum((Yp - Y) ** 2))


def fac_ps_lse(
    nb,
    nb_dst,
    dim,
    Y,
    Yp,
    i: int,
    fn: float = _DEFAULT_FN,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """Factor of the combined PS + LSE error (R ``fac_ps_lseR``)."""
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    yfactor = alpha * float(Y[i] - Yp[i])
    if not check_ps(nb, Y, Yp, i, fn=fn):
        yfactor += _sgn(Y[i] - Yp[i]) * abs(Y[i] - Yp[i]) ** 0.01
    return yfactor


def _l1_curvature(nb, nb_dst, Yp, start: int) -> float:
    """L1-type curvature term; ``start = 0`` (all members, squared
    slopes, R ``err_ps_l1R``) or ``start = 1`` (skip the centre, linear
    slopes, R ``fac_ps_l1R``)."""
    Yp = np.asarray(Yp, dtype=float).ravel()
    l1 = 0.0
    for members, dst in zip(nb, nb_dst):
        members = np.asarray(members, dtype=np.int64)
        dst = np.asarray(dst, dtype=float)
        i0 = members[0]
        for m in range(start, members.size):
            if dst[m] != 0.0:
                slope = (Yp[i0] - Yp[members[m]]) / dst[m]
                l1 += slope * slope if start == 0 else slope
    return l1


def err_ps_l1(
    nb, nb_dst, dim, Y, Yp, fn: float = _DEFAULT_FN, alpha: float = _DEFAULT_ALPHA
) -> float:
    """Combined partial sum + curvature error (R ``err_ps_l1R``).

    ``err_ps(...) + alpha * sum over neighbourhoods of the squared
    centred slopes (Yp[i0] - Yp[j]) / dst[j]`` over all non-coincident
    members ``j`` (``i0`` is the neighbourhood centre).
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    return _ps_part(nb, Y, Yp, fn) + alpha * _l1_curvature(nb, nb_dst, Yp, start=0)


def fac_ps_l1(
    nb,
    nb_dst,
    dim,
    Y,
    Yp,
    i: int,
    fn: float = _DEFAULT_FN,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """Factor of the combined PS + curvature error (R ``fac_ps_l1R``).

    The curvature part accumulates, over every neighbourhood in which
    ``i`` is among the first ``dim`` members, the *linear* centred
    slopes over all non-coincident members — the R derivative
    convention, reproduced verbatim.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    yfactor = 0.0
    if not check_ps(nb, Y, Yp, i, fn=fn):
        yfactor = _sgn(Y[i] - Yp[i]) * abs(Y[i] - Yp[i]) ** 0.01
    l1 = 0.0
    for members, dst in zip(nb, nb_dst):
        members = np.asarray(members, dtype=np.int64)
        if i in members[:dim]:
            l1 += _l1_curvature([members], [dst], Yp, start=1)
    return yfactor + alpha * l1


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def calc_out(X, W0, W1, W2) -> np.ndarray:
    """Batch forward pass (R ``calcOutR``).

    Parameters
    ----------
    X : np.ndarray
        Standardised inputs *with* bias column, shape ``(n, il)``.
    W0, W1, W2 : np.ndarray
        Layer weights ``(hl1, il)``, ``(hl2, hl1)``, ``(1, hl2)``.

    Returns
    -------
    np.ndarray
        Network outputs, shape ``(n,)``.
    """
    X = np.asarray(X, dtype=float)
    O1 = sigmoid(W0 @ X.T)  # (hl1, n)
    O2 = sigmoid(W1 @ O1)  # (hl2, n)
    y = (W2 @ O2).T  # (n, 1)
    return np.asarray(y, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_neighborhoods(Xnb: np.ndarray, k: int, size: int):
    """Input-space neighbourhoods sorted by ascending distance.

    Port of the R construction ``nb <- lapply(..., head(sort(d[,i],
    index.return=TRUE)$ix, (il-1)*k+1))`` together with the matching
    sorted distances ``nb_dst``.  Returns ``(nb, nb_dst)``.
    """
    n = Xnb.shape[0]
    d = np.sqrt(
        np.maximum(
            np.sum(Xnb**2, axis=1)[:, None]
            + np.sum(Xnb**2, axis=1)[None, :]
            - 2.0 * (Xnb @ Xnb.T),
            0.0,
        )
    )
    nb = []
    nb_dst = []
    for i in range(n):
        order = np.argsort(d[:, i], kind="stable")[:size]
        nb.append(order)
        nb_dst.append(d[order, i])
    return nb, nb_dst


def _membership_index(nb, n: int, head: int | None = None):
    """Inverted neighbourhood index.

    Returns a list of arrays: for every sample the ids of the
    neighbourhoods containing it (or containing it among the first
    ``head`` members when ``head`` is given).
    """
    idx: list[list[int]] = [[] for _ in range(n)]
    for j, members in enumerate(nb):
        sel = members if head is None else members[:head]
        for m in np.unique(sel):
            idx[int(m)].append(j)
    return [np.asarray(v, dtype=np.int64) for v in idx]


def _ps_sums(nb, Y, Yp) -> np.ndarray:
    """Per-neighbourhood residual sign sums (integer-exact)."""
    Y = np.asarray(Y, dtype=float).ravel()
    Yp = np.asarray(Yp, dtype=float).ravel()
    return np.asarray(
        [int(np.sum(np.sign(Y[np.asarray(m)] - Yp[np.asarray(m)]))) for m in nb],
        dtype=np.int64,
    )


def ssrmlp_train(
    X,
    Y,
    std: bool = True,
    opt: str = "ps",
    hl=None,
    W: SSRMLPModel | None = None,
    k: int = 10,
    fn: float = 4,
    eta: float = 0.75,
    max_iter: int = 1000,
    facfct_ex: Callable | None = None,
    errfct_ex: Callable | None = None,
    alpha: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> SSRMLPModel:
    """Train the 2-layer SSR perceptron (R ``ssrmlp_train``).

    Parameters
    ----------
    X : np.ndarray
        Inputs, shape ``(n, dim)`` (a matrix, as in R).
    Y : np.ndarray
        Targets, shape ``(n,)``.
    std : bool, optional
        Standardise inputs and targets to the unit interval
        (default True); with ``std=False`` the identity bounds
        ``minX = 0, maxX = 1, minY = 0, maxY = 1`` are stored instead,
        as in R.
    opt : str, optional
        Training criterion: ``'ps'`` (default) partial sums only,
        ``'ps_lse'`` partial sums + squared residuals,
        ``'ps_l1'`` partial sums + input-space curvature,
        ``'lse'`` plain least squares, ``'ext'`` user-supplied
        ``errfct_ex``/``facfct_ex``.  Unknown values fall back to
        ``'ps'`` (R ``switch`` default).
    hl : sequence of int, optional
        Hidden layer sizes ``(hl1, hl2)``; defaults to
        ``hln = int(-(il+ol+1)/2 + sqrt((il+ol+1)**2 + n)) * 2`` for
        both layers.  ``hl[0] != hl[1]`` raises ``ValueError`` — the R
        implementation is only conformable for square configurations.
    W : SSRMLPModel, optional
        Existing model to re-train (keeps its standardisation).
    k : int, optional
        Neighbours per input axis of the criterion neighbourhoods; the
        neighbourhood size is ``dim * k + 1`` (default 10).
    fn : float, optional
        Accepted for API fidelity; **not forwarded** to the criterion
        functions (R quirk) — the effective threshold is the function
        default ``fn = 4``.  Custom thresholds require ``opt='ext'``.
    eta : float, optional
        Learning rate (default 0.75).
    max_iter : int, optional
        Number of epochs (default 1000); one fixed random permutation
        of the samples per epoch, as in R.
    facfct_ex, errfct_ex : callable, optional
        External factor/error functions for ``opt='ext'``, called as
        ``errfct_ex(nb, nb_dst, dim, Y, Yp)`` and
        ``facfct_ex(nb, nb_dst, dim, Y, Yp, i)`` — the R convention.
    alpha : float, optional
        Accepted for API fidelity; **not forwarded** (R quirk) — the
        effective mixing weight is the function default
        ``alpha = 1e-4``.
    rng : int | np.random.Generator, optional
        Seed or generator for the weight initialisation and the sample
        permutation (R uses the global RNG).  Any object exposing a
        Generator-like ``permutation(n)`` method is accepted as well,
        which allows replaying a recorded sample order.

    Returns
    -------
    SSRMLPModel
        The best (lowest training error) weight set.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (n, dim) matrix, got shape {X.shape}")
    Y = np.asarray(Y, dtype=float).ravel()
    n, dim = X.shape
    if Y.shape[0] != n:
        raise ValueError(f"X and Y must have n rows, got {n} and {Y.shape[0]}")
    if n < 2:
        raise ValueError(f"ssrmlp_train requires at least 2 samples, got {n}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    if isinstance(rng, np.random.Generator) or (
        not isinstance(rng, (int, np.integer, type(None)))
        and hasattr(rng, "permutation")
    ):
        pass  # duck-typed generator (allows replaying a sample order)
    else:
        rng = np.random.default_rng(rng)

    il = dim + 1  # inputs + bias
    ol = 1

    if hl is None:
        hln = int(-(il + ol + 1) / 2.0 + np.sqrt((il + ol + 1) ** 2 + n)) * 2
        hl = (hln, hln)
    hl = (int(hl[0]), int(hl[1]))
    if hl[0] != hl[1]:
        # R: W1 (hl2 x hl1) += dW1 (hl1 x hl2) is non-conformable.
        raise ValueError(
            "ssrmlp_train requires hl[0] == hl[1] (the R implementation "
            f"adds the transposed W1 gradient); got {list(hl)}"
        )
    if hl[0] < 1:
        raise ValueError(f"hidden layer sizes must be >= 1, got {list(hl)}")
    logger.debug("ssrMLP: number of neurons per layer: %d", hl[0])

    # Standardisation / initialisation --------------------------------------
    if W is None:
        if std:
            with np.errstate(invalid="ignore", divide="ignore"):
                minX = np.min(X, axis=0)
                maxX = np.max(X, axis=0)
                Xn = (X - minX) / (maxX - minX)
                Xn[~np.isfinite(Xn)] = 0.0  # constant columns -> 0, as in R
                minY = float(np.min(Y))
                maxY = float(np.max(Y))
                Yn = (Y - minY) / (maxY - minY)
        else:
            minX = np.zeros(dim)
            maxX = np.ones(dim)
            minY = 0.0
            maxY = 1.0
            Xn = X.copy()
            Yn = Y.copy()
        W0 = rng.uniform(-0.5, 0.5, size=(hl[0], il))
        W1 = rng.uniform(-0.5, 0.5, size=(hl[1], hl[0]))
        W2 = rng.uniform(-0.5, 0.5, size=(ol, hl[1]))
    else:
        logger.debug("ssrMLP: re-training...")
        W0 = np.array(W.W0, dtype=float, copy=True)
        W1 = np.array(W.W1, dtype=float, copy=True)
        W2 = np.array(W.W2, dtype=float, copy=True)
        minX = np.asarray(W.minX, dtype=float)
        maxX = np.asarray(W.maxX, dtype=float)
        minY = float(W.minY)
        maxY = float(W.maxY)
        with np.errstate(invalid="ignore", divide="ignore"):
            Xn = (X - minX) / (maxX - minX)
            Yn = (Y - minY) / (maxY - minY)

    if W0.shape != (hl[0], il) or W1.shape != (hl[1], hl[0]) or W2.shape != (ol, hl[1]):
        raise ValueError(
            f"weight shapes {(W0.shape, W1.shape, W2.shape)} do not match "
            f"hl={list(hl)}, il={il}"
        )

    Xnb = Xn  # coordinates without bias-coordinate
    Xb = np.hstack([Xn, np.ones((n, 1))])  # bind additional bias neuron

    # Criterion dispatch -----------------------------------------------------
    # The R switch falls through to the trailing default ('ps') branch for
    # any unknown option; 'fn'/'alpha' are accepted but never forwarded to
    # the criterion functions (R quirk), so the function defaults
    # fn=4 / alpha=1e-4 are the effective parameters.
    del fn, alpha
    opt_eff = opt if opt in ("lse", "ps_lse", "ps_l1", "ext") else "ps"
    if opt_eff != opt:  # R switch default branch
        logger.debug("ssrMLP: unknown opt %r, falling back to 'ps'", opt)

    nb = None
    nb_dst = None
    if opt_eff != "lse":
        # neighbourhood for partial sums: k times the number of axis plus center
        nb, nb_dst = _build_neighborhoods(Xnb, k, dim * k + 1)

    if opt_eff == "ext":
        if errfct_ex is None or facfct_ex is None:
            raise ValueError("opt='ext' requires errfct_ex and facfct_ex")

        def errfct(Yp):
            return float(errfct_ex(nb, nb_dst, il, Yn, Yp))

        def facfct(i, Yp):
            return float(facfct_ex(nb, nb_dst, il, Yn, Yp, i))

    elif opt_eff == "lse":
        logger.debug("ssrMLP: optimizing with least-squares-residuals")

        def errfct(Yp):
            return err_lse(nb, nb_dst, il, Yn, Yp)

        def facfct(i, Yp):
            return fac_lse(nb, nb_dst, il, Yn, Yp, i)

    else:
        # Fast, integer-exact equivalents of err_ps / fac_ps(_lse|_l1) with
        # the R function defaults fn=4, alpha=1e-4 (train args not forwarded).
        if opt_eff == "ps_lse":
            logger.debug(
                "ssrMLP: optimizing with combined partial sum and "
                "least-squares-residuals criterion"
            )
        elif opt_eff == "ps_l1":
            logger.debug(
                "ssrMLP: optimizing with combined partial sum and "
                "l1 curvature criterion"
            )
        else:
            logger.debug("ssrMLP: optimizing with partial sums")
        pt_nbs = _membership_index(nb, n)
        pt_head_nbs = (
            _membership_index(nb, n, head=il) if opt_eff == "ps_l1" else None
        )

        def errfct(Yp):
            s_all = _ps_sums(nb, Yn, Yp)
            s = 0.0
            for v in s_all:
                s += max(0.0, abs(int(v)) - _DEFAULT_FN)
            if opt_eff == "ps_lse":
                s += _DEFAULT_ALPHA * float(np.sum((Yp - Yn) ** 2))
            elif opt_eff == "ps_l1":
                s += _DEFAULT_ALPHA * _l1_curvature(nb, nb_dst, Yp, start=0)
            return s

        def facfct(i, Yp):
            # Integer-exact incremental evaluation of check_ps at sample i:
            # the loop keeps the per-neighbourhood sign sums up to date, so
            # the sums below equal sum(sign(Yn - Yp)) over the members.
            adequate = True
            for j in pt_nbs[i]:
                members = nb[j]
                s = int(np.sum(_sgn_arr(Yn[members] - Yp[members])))
                if abs(s) > _DEFAULT_FN:
                    adequate = False
                    break
            yfactor = 0.0
            if not adequate:
                yfactor = _sgn(Yn[i] - Yp[i]) * abs(Yn[i] - Yp[i]) ** 0.01
            if opt_eff == "ps_lse":
                yfactor += _DEFAULT_ALPHA * float(Yn[i] - Yp[i])
            elif opt_eff == "ps_l1":
                l1 = 0.0
                for j in pt_head_nbs[i]:
                    members = nb[j]
                    dst = nb_dst[j]
                    i0 = members[0]
                    for m in range(1, members.size):
                        if dst[m] != 0.0:
                            l1 += (Yp[i0] - Yp[members[m]]) / dst[m]
                yfactor += _DEFAULT_ALPHA * l1
            return yfactor

    # Training loop -----------------------------------------------------------
    Yp = calc_out(Xb, W0, W1, W2)
    mean_e = errfct(Yp)
    logger.debug("ssrMLP: start error: %f", mean_e)
    min_error = mean_e
    minW0, minW1, minW2 = W0.copy(), W1.copy(), W2.copy()

    chng = False
    mix_set = rng.permutation(n)  # R sample(nrow(X)); one fixed permutation
    for c in range(1, max_iter + 1):
        for i in mix_set:
            i = int(i)
            x = Xb[i]
            O1 = sigmoid(W0 @ x)
            O2 = sigmoid(W1 @ O1)
            temp = W2[0] * O2 * (1.0 - O2)  # R: W2*O2*(1-O2), row (1, hl2)
            dW2 = O2
            # R: dW1 <- t(t(temp) %*% O1) — O1 coerced to a row matrix, so the
            # product is outer(temp, O1) and the outer t() transposes it.
            dW1 = np.outer(temp, O1).T  # (hl1, hl2)
            dW0 = np.outer(O1 * (1.0 - O1) * (temp @ W1), x)  # (hl1, il)
            Yp[i] = calc_out(x[None, :], W0, W1, W2)[0]
            yfactor = facfct(i, Yp)
            W0 = W0 + eta * yfactor * dW0
            W1 = W1 + eta * yfactor * dW1
            W2 = W2 + eta * yfactor * dW2
        Yp = calc_out(Xb, W0, W1, W2)
        mean_e = errfct(Yp)
        if mean_e < min_error:
            chng = True
            min_error = mean_e
            minW0, minW1, minW2 = W0.copy(), W1.copy(), W2.copy()
        if c % 50 == 0:
            logger.debug("ssrMLP: epoch %d (%s)", c, "X" if chng else "x")
            chng = False
    logger.debug("ssrMLP: final minError: %f", min_error)
    return SSRMLPModel(
        W0=minW0,
        W1=minW1,
        W2=minW2,
        minX=np.asarray(minX, dtype=float),
        maxX=np.asarray(maxX, dtype=float),
        minY=float(minY),
        maxY=float(maxY),
    )


# ---------------------------------------------------------------------------
# Prediction / factor importance
# ---------------------------------------------------------------------------


def ssrmlp_predict(X, W: SSRMLPModel) -> np.ndarray:
    """Predict with a trained SSR-MLP model (R ``ssrmlp_predict``).

    The inputs are standardised with the stored bounds (non-finite
    values, e.g. from constant columns, are replaced by 0), the bias
    column is appended and the network output is mapped back through
    the target standardisation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (n, dim) matrix, got shape {X.shape}")
    with np.errstate(invalid="ignore", divide="ignore"):
        Xn = (X - W.minX) / (W.maxX - W.minX)
    Xn[~np.isfinite(Xn)] = 0.0  # to avoid NaN from div / 0
    Xb = np.hstack([Xn, np.ones((X.shape[0], 1))])
    y = calc_out(Xb, W.W0, W.W1, W.W2)
    return y * (W.maxY - W.minY) + W.minY


def fii_model(W: SSRMLPModel) -> np.ndarray:
    """Factor-wise contribution of the model weights (R ``fii_model``).

    Feeds the identity through the network and normalises the per-input
    (bias included) outputs to sum to one.
    """
    il = W.W0.shape[1]
    X = np.eye(il)
    O1 = sigmoid(W.W0 @ X.T)  # (hl1, il)
    O2 = sigmoid(W.W1 @ O1)  # (hl2, il)
    y = (W.W2 @ O2).T.ravel()  # (il,)
    return y / np.sum(y)


def fii_prediction(W: SSRMLPModel, X) -> np.ndarray:
    """Per-input factor importance for given samples (R ``fii_prediction``).

    Every sample is fed through the network once per input factor (the
    input matrix replaced by the diagonal matrix of the standardised
    factor values plus the bias), the outputs are de-standardised,
    normalised to sum to one per sample, and averaged over the samples
    (non-finite contributions omitted, as in R ``mean(na.omit(x))``).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (n, dim) matrix, got shape {X.shape}")
    il = W.W0.shape[1]
    cols = np.empty((il, X.shape[0]), dtype=float)
    for j in range(X.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            fii = np.append((X[j] - W.minX) / (W.maxX - W.minX), 1.0)
        fii[~np.isfinite(fii)] = 0.0  # to avoid NaN from div / 0
        D = np.diag(fii)
        O1 = sigmoid(W.W0 @ D.T)
        O2 = sigmoid(W.W1 @ O1)
        y = (W.W2 @ O2).T.ravel()
        y = y * (W.maxY - W.minY) + W.minY
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[:, j] = y / np.sum(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(cols, axis=1)
    return out
