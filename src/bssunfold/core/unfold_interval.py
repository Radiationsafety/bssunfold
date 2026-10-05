"""Interval analysis unfolding methods.

Computes guaranteed bounds on the neutron spectrum using interval linear
programming. For each energy bin, solves two LPs (min and max) subject to
interval constraints on the readings and a TV smoothness bound.

The methods follow the interval data analysis framework of Dagesenov,
Zyabin, Kumkov and Shary, "Обработка и анализ интервальных данных"
(Izhevsk, 2024), and Shary, "Analysis of interval data" (2018):

- Interval LP enclosures of the information set (united/tolerable solution
  set of ``A x = b`` with interval ``b``).
- Shary's recognizing functionals ``Tol`` (strong compatibility, tolerable
  set) and ``Uss`` (weak compatibility, united set), maximized exactly via
  an LP reduction (cf. the ``tolprog`` approach).
- Center-of-uncertainty method (Askerkin & Sukhanov) with a minimal
  data-widening LP that quantifies incompatibility of the readings.
- Simple interval approximation (PIA, Rutkowski): Chebyshev-type robust
  fit to interval corridors when the information set is empty.
- Kaucher/Rohn tolerable-set enclosures for an *interval response matrix*
  (uncertain detector sensitivities), plus Khlebnikov's inner box.
- X-variativity (SEV/TEV): a scalar measure of the size of the information
  set attached to a max-compatibility estimate.

Optional integration with the ``intvalpy`` package is provided.
"""

from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import linprog

from ..logging_config import get_logger
from ._base_unfolder import _build_system, _standardize_output

logger = get_logger("unfold_interval")

__all__ = [
    "interval_compatibility_report",
    "interval_sev_variativity",
    "solve_interval",
    "solve_interval_center",
    "solve_interval_matrix",
    "solve_interval_pia",
    "solve_interval_posterior",
    "solve_interval_tol",
    "solve_interval_intvalpy",
    "unfold_interval",
    "unfold_interval_center",
    "unfold_interval_matrix",
    "unfold_interval_pia",
    "unfold_interval_posterior",
    "unfold_interval_tol",
    "unfold_interval_intvalpy",
]

# Guard rails for the exponential vertex enumerations (Khlebnikov, SEV).
_INNER_BOX_MAX_BINS = 10
_FULL_X_MAX_BINS = 10


def _validate_interval_system(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate the point matrix and interval right-hand side."""
    A = np.asarray(A, dtype=float)
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be a 2D matrix, got shape {A.shape}")
    m, _ = A.shape
    if b_lo.shape != (m,) or b_hi.shape != (m,):
        raise ValueError(
            f"b_lo and b_hi must have shape ({m},), "
            f"got {b_lo.shape} and {b_hi.shape}"
        )
    if np.any(b_lo > b_hi):
        raise ValueError("b_lo must be <= b_hi for all elements")
    if np.any(b_lo < 0):
        raise ValueError("b_lo must be non-negative")
    return A, b_lo, b_hi


def _validate_interval_matrix(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate interval matrix bounds."""
    A_lo = np.asarray(A_lo, dtype=float)
    A_hi = np.asarray(A_hi, dtype=float)
    if A_lo.shape != A_hi.shape or A_lo.ndim != 2:
        raise ValueError(
            f"A_lo and A_hi must be 2D matrices of equal shape, got "
            f"{A_lo.shape} and {A_hi.shape}"
        )
    if np.any(A_lo > A_hi):
        raise ValueError("A_lo must be <= A_hi for all elements")
    A_lo, b_lo, b_hi = _validate_interval_system(A_lo, b_lo, b_hi)
    A_hi = np.asarray(A_hi, dtype=float)
    return A_lo, A_hi, b_lo, b_hi


def _tv_rows(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """TV regularization rows for variables ``[x(n), t(n-1)]``.

    Returns ``(A_tv, b_tv, n_tv)`` with ``A_tv`` of shape
    ``(2(n-1) + 1, n + n_tv)``: ``t_i >= |x_{i+1} - x_i|`` and
    ``sum(t) <= 1`` (the caller scales the last row by ``tv_bound``).
    """
    n_tv = n - 1
    rows = np.zeros((2 * n_tv + 1, n + n_tv))
    rhs = np.zeros(2 * n_tv + 1)
    for i in range(n_tv):
        rows[2 * i, i + 1] = 1.0
        rows[2 * i, i] = -1.0
        rows[2 * i, n + i] = -1.0
        rows[2 * i + 1, i + 1] = -1.0
        rows[2 * i + 1, i] = 1.0
        rows[2 * i + 1, n + i] = -1.0
    rows[-1, n:] = 1.0
    return rows, rhs, n_tv


def _build_lp_system(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    """Build the LP constraint system for interval analysis.

    Variables: [x_0, ..., x_{n-1}, t_0, ..., t_{n-2}] where t_i are
    auxiliary TV variables (only present when tv_bound is not None).

    Constraints:
        b_lo <= A @ x <= b_hi          (2m inequalities)
        t_i >= x_{i+1} - x_i           (n-1 inequalities, TV)
        t_i >= -(x_{i+1} - x_i)        (n-1 inequalities, TV)
        sum(t_i) <= T                  (1 inequality, TV)
    """
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([b_hi, -b_lo])
    n = A.shape[1]

    if tv_bound is not None:
        rows_tv, rhs_tv, _ = _tv_rows(n)
        A_ub = np.vstack(
            [
                np.hstack(
                    [
                        A_ub,
                        np.zeros((A_ub.shape[0], rows_tv.shape[1] - n)),
                    ]
                ),
                rows_tv,
            ]
        )
        rhs_tv = rhs_tv.copy()
        rhs_tv[-1] = tv_bound
        b_ub = np.concatenate([b_ub, rhs_tv])

    bounds = [(0, None)] * A_ub.shape[1]
    return A_ub, b_ub, bounds


def _component_bounds(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize/maximize each of the first ``n`` LP variables.

    Returns ``(x_min, x_max)`` with failed LPs contributing zero-width
    entries at 0 (same convention as :func:`solve_interval`).
    """
    n_vars = len(bounds)
    x_min = np.zeros(n)
    x_max = np.zeros(n)
    for i in range(n):
        c = np.zeros(n_vars)
        c[i] = 1.0
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
        if res_min.success and res_min.x is not None:
            x_min[i] = res_min.x[i]
        res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
        if res_max.success and res_max.x is not None:
            x_max[i] = res_max.x[i]
    x_min = np.maximum(x_min, 0.0)
    x_max = np.maximum(x_max, x_min)
    return x_min, x_max


def solve_interval(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve interval LP for each energy bin.

    For each bin i, computes the minimum and maximum possible value of x_i
    subject to b_lo <= A @ x <= b_hi, x >= 0, and optional TV regularization.
    The resulting box is the componentwise hull of the information set
    (united = tolerable solution set for a point matrix A).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound. If None, no TV regularization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_min, x_max) arrays of shape (n,).
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    n = A.shape[1]
    A_ub, b_ub, bounds = _build_lp_system(A, b_lo, b_hi, tv_bound)
    return _component_bounds(A_ub, b_ub, bounds, n)


def _tol_functional(
    x: np.ndarray,
    A_mid: np.ndarray,
    b_mid: np.ndarray,
    b_rad: np.ndarray,
    A_rad: np.ndarray | None = None,
    functional: str = "tol",
) -> float:
    """Shary's recognizing functional at a point.

    ``Tol(x) = min_i [rad(b_i) - rad(A_i)|x| - |mid(b_i) - a_mid_i x|]``
    (tolerable set) or ``Uss(x)`` with ``+ rad(A_i)|x|`` (united set).
    For a point matrix both reduce to
    ``min_i [rad(b_i) - |mid(b_i) - a_i x|]``.
    """
    residuals = b_mid - A_mid @ x
    reserve = b_rad - np.abs(residuals)
    if A_rad is not None and np.any(A_rad):
        signed = np.sum(A_rad * np.abs(x), axis=1)
        reserve = reserve - signed if functional == "tol" else reserve + signed
    return float(np.min(reserve))


def _tol_max_lp(
    A_mid: np.ndarray,
    A_rad: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    functional: str = "tol",
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Maximize a recognizing functional exactly via a single LP.

    Variables ``[x(n), t(n_tv), tau(1)]``. For ``Tol`` the constraints
    ``w_i [rad(b_i) - A_rad x - |mid(b_i) - (A_mid x)_i|] >= tau`` read
    (with x >= 0, ``u_i = 1 / w_i`` for unit weights):

        (A_mid + A_rad) x + u_i tau <= b_hi
        -(A_mid - A_rad) x + u_i tau <= -b_lo

    For ``Uss`` the ``A_rad`` terms change sign. ``tau`` is free; the
    optimum ``tau* >= 0`` certifies that the corresponding solution set
    is non-empty (the LP also handles the empty case gracefully).

    Returns
    -------
    dict
        tol_max, x_pseudo, generators, generators_profile, violations,
        converged.
    """
    if functional not in ("tol", "uss"):
        raise ValueError(
            f"functional must be 'tol' or 'uss', got {functional!r}"
        )
    m, n = A_mid.shape
    b_mid = (b_lo + b_hi) / 2.0
    b_rad = (b_hi - b_lo) / 2.0
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,) or np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and positive, shape (m,)")
    inv_w = 1.0 / w

    if functional == "tol":
        row_hi = A_mid + A_rad
        row_lo = -(A_mid - A_rad)
    else:
        row_hi = A_mid - A_rad
        row_lo = -(A_mid + A_rad)

    n_tv = 0
    rhs_tv = None
    if tv_bound is not None:
        rows_tv, rhs_tv, n_tv = _tv_rows(n)
        rows_tv = np.hstack([rows_tv, np.zeros((rows_tv.shape[0], 1))])
        rhs_tv = rhs_tv.copy()
        rhs_tv[-1] = tv_bound

    pad = (m, n_tv)
    blocks = [
        np.hstack([row_hi, np.zeros(pad), inv_w[:, None]]),
        np.hstack([row_lo, np.zeros(pad), inv_w[:, None]]),
    ]
    rhs = [b_hi, -b_lo]
    if tv_bound is not None:
        blocks.append(rows_tv)
        rhs.append(rhs_tv)

    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    n_vars = n + n_tv + 1
    bounds = [(0, None)] * (n + n_tv) + [(None, None)]
    c = np.zeros(n_vars)
    c[-1] = -1.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success and res.x is not None:
        x_pseudo = np.maximum(res.x[:n], 0.0)
        tol_max = float(-res.fun)
        converged = True
    else:
        logger.warning(
            "Tol LP did not converge (status=%s); using least-squares "
            "fallback pseudo-solution", getattr(res, "status", "unknown")
        )
        x_pseudo = np.maximum(
            np.linalg.lstsq(A_mid, b_mid, rcond=None)[0], 0.0
        )
        tol_max = _tol_functional(
            x_pseudo, A_mid, b_mid, b_rad, A_rad, functional
        )
        converged = False

    residuals = np.abs(b_mid - A_mid @ x_pseudo)
    reserve = b_rad - residuals
    if np.any(A_rad):
        signed = np.sum(A_rad * np.abs(x_pseudo), axis=1)
        reserve = reserve - signed if functional == "tol" else reserve + signed
    if not converged:
        tol_max = float(np.min(w * reserve))
    gen_values = w * reserve
    scale = max(1.0, abs(tol_max))
    generators = np.flatnonzero(gen_values <= tol_max + 1e-9 * scale)
    order = np.argsort(gen_values)
    generators_profile = np.column_stack([order.astype(float), gen_values[order]])
    fitted = A_mid @ x_pseudo
    violations = np.maximum(np.maximum(b_lo - fitted, fitted - b_hi), 0.0)

    return {
        "tol_max": tol_max,
        "x_pseudo": x_pseudo,
        "generators": generators,
        "generators_profile": generators_profile,
        "violations": violations,
        "converged": converged,
    }


def solve_interval_tol(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    drop_infeasible: int = 0,
    functional: str = "tol",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve interval system via Shary's recognizing functional.

    Maximizes the recognizing functional exactly with one LP (Tol for the
    tolerable set, Uss for the united set), then computes the guaranteed
    componentwise bounds of the information set with 2n LPs including the
    TV constraint when ``tv_bound`` is given. A negative ``tol_max``
    certifies an empty information set (incompatible readings); the
    ``generators`` indices mark the most restrictive readings and
    ``drop_infeasible`` iteratively removes the worst offenders.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization (also applied to the
        componentwise bounds).
    max_iter : int, optional
        Legacy parameter, unused by the LP core (kept for API
        compatibility).
    tol : float, optional
        Legacy parameter, unused by the LP core.
    drop_infeasible : int, optional
        Number of worst-violating readings to drop while the system stays
        incompatible (default 0, no dropping).
    functional : str, optional
        "tol" (default) or "uss"; for a point matrix they coincide.
    weights : np.ndarray, optional
        Positive weights (m,) for the individual generators of the
        functional — the value of each reading in the compatibility
        measure (cf. tolsolvty ``weight``); default all ones.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) where info contains tol_max, x_pseudo,
        generators, generators_profile, violations, dropped, converged,
        n_iter.
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    m = A.shape[0]
    if functional not in ("tol", "uss"):
        raise ValueError(
            f"functional must be 'tol' or 'uss', got {functional!r}"
        )
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (m,) or np.any(weights <= 0):
            raise ValueError("weights must be positive with shape (m,)")

    zero_rad = np.zeros_like(A)
    keep = list(range(m))
    dropped: list[int] = []
    lp = _tol_max_lp(
        A, zero_rad, b_lo, b_hi, tv_bound, functional, weights
    )
    n_lp = 1
    for _ in range(int(drop_infeasible)):
        if lp["tol_max"] >= 0 or not lp["converged"] or len(keep) <= 1:
            break
        worst = int(np.argmax(lp["violations"]))
        if lp["violations"][worst] <= 0.0:
            break
        dropped.append(keep.pop(worst))
        idx = np.asarray(keep, dtype=int)
        lp = _tol_max_lp(
            A[idx],
            zero_rad[idx],
            b_lo[idx],
            b_hi[idx],
            tv_bound,
            functional,
            None if weights is None else weights[idx],
        )
        n_lp += 1

    idx = np.asarray(keep, dtype=int)
    x_min, x_max = solve_interval(
        A[idx], b_lo[idx], b_hi[idx], tv_bound=tv_bound
    )

    info = {
        "tol_max": lp["tol_max"],
        "x_pseudo": lp["x_pseudo"],
        "converged": lp["converged"],
        "n_iter": n_lp,
        "generators": [int(k) for k in idx[lp["generators"]]],
        "generators_profile": np.column_stack(
            [idx[lp["generators_profile"][:, 0].astype(int)],
             lp["generators_profile"][:, 1]]
        ),
        "violations": lp["violations"],
        "dropped": dropped,
        "functional": functional,
    }
    return x_min, x_max, info


def solve_interval_posterior(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    n_samples: int = 100,
    normalize: bool = False,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve interval system with posterior Monte-Carlo analysis.

    Computes the guaranteed componentwise bounds with
    :func:`solve_interval` (outer enclosure of the information set) and
    supplements them with a Monte-Carlo sample cloud inside the set: for
    each of ``n_samples`` random non-negative objectives ``c`` the LP
    ``max c.x`` over ``{x >= 0, b_lo <= Ax <= b_hi}`` is solved. The
    empirical envelope of the cloud is an *inner* (statistical, not
    guaranteed) approximation that typically tightens the outer box, and
    the cloud gives per-bin spread statistics for uncertainty analysis.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization.
    n_samples : int, optional
        Number of Monte-Carlo LP samples.
    normalize : bool, optional
        If True, scale the bounds and the MC envelope so the spectrum
        total matches the total fluence ``sum(A @ x_mid)``.
    random_state : int, optional
        Seed for the random objectives (reproducibility).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info). x_min/x_max are the guaranteed outer bounds;
        info carries the MC envelope (``spectrum_mc_lower/upper``) and
        statistics.
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    m, n = A.shape

    x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=tv_bound)

    b_mid = (b_lo + b_hi) / 2.0
    b_rad = (b_hi - b_lo) / 2.0
    x_mid = (x_min + x_max) / 2.0
    residuals = b_mid - A @ x_mid

    # Sensitivity of each bin to the reading uncertainty (column norm of A
    # weighted by the reading radii).
    sensitivity = np.sum(np.abs(A) * b_rad[:, None], axis=0)

    A_ub, b_ub, bounds = _build_lp_system(A, b_lo, b_hi, tv_bound)
    rng = np.random.default_rng(random_state)
    cloud: list[np.ndarray] = []
    for _ in range(int(n_samples)):
        c = np.zeros(len(bounds))
        c[:n] = rng.random(n)
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success and res.x is not None:
            cloud.append(np.maximum(res.x[:n], 0.0))

    if len(cloud) >= 2:
        X = np.vstack(cloud)
        mc_lo = X.min(axis=0)
        mc_hi = X.max(axis=0)
        mc_mean = X.mean(axis=0)
        mc_std = X.std(axis=0)
    else:
        logger.warning(
            "posterior Monte-Carlo produced %d feasible samples; "
            "MC envelope degenerates to the guaranteed bounds", len(cloud)
        )
        mc_lo = x_min.copy()
        mc_hi = x_max.copy()
        mc_mean = x_mid.copy()
        mc_std = np.zeros(n)

    if normalize:
        total = np.sum(x_mid)
        if total > 0:
            scale = np.sum(A @ x_mid) / total
            x_min = x_min * scale
            x_max = x_max * scale
            mc_lo = mc_lo * scale
            mc_hi = mc_hi * scale
            mc_mean = mc_mean * scale
            mc_std = mc_std * scale

    info = {
        "n_samples": int(n_samples),
        "n_feasible": len(cloud),
        "sensitivity": sensitivity,
        "residuals": residuals,
        "normalize": normalize,
        "spectrum_mc_lower": mc_lo,
        "spectrum_mc_upper": mc_hi,
        "spectrum_mc_mean": mc_mean,
        "spectrum_mc_std": mc_std,
        "random_state": random_state,
    }
    return x_min, x_max, info


def solve_interval_center(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Center-of-uncertainty method with minimal data widening.

    Implements the Askerkin-Sukhanov approach (Dagesenov et al. 2024,
    sec. 4.14.4). The maximizing point of the recognizing functional is
    the "center of uncertainty" and the maximum ``tol_max`` its reserve
    of compatibility. If the information set is empty (incompatible
    readings, ``tol_max < 0``), the minimal data widening LP

        min sum_i gamma_i (u_i + w_i)
        s.t.  A x - u <= b_hi,  -A x - w <= -b_lo,  x, u, w >= 0

    is solved; its optimum gives per-reading widening amounts
    ``epsilon_i = u_i + w_i`` — a quantitative incompatibility measure
    (cf. the Demidenko/Khlebnikov paradoxes discussion) — and the
    componentwise bounds are computed on the repaired intervals
    ``[b_lo - u, b_hi + w]``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization.
    weights : np.ndarray, optional
        Positive widening weights gamma (m,); default all ones.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) with info keys tol_max, x_pseudo, epsilon,
        incompatibility, compatible, b_lo_effective, b_hi_effective,
        converged.
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    m, n = A.shape
    if weights is None:
        gamma = np.ones(m)
    else:
        gamma = np.asarray(weights, dtype=float)
        if gamma.shape != (m,) or np.any(gamma <= 0):
            raise ValueError("weights must be positive with shape (m,)")

    # Variables: [x(n), u(m), w(m), t(n_tv)]
    n_tv = 0
    rows_tv = None
    if tv_bound is not None:
        rows_tv, rhs_tv, n_tv = _tv_rows(n)
        # move TV rows from [x, t] layout into [x, 0(u), 0(w), t]
        rows_tv = np.hstack(
            [rows_tv[:, :n], np.zeros((rows_tv.shape[0], 2 * m)),
             rows_tv[:, n:]]
        )
        rhs_tv = rhs_tv.copy()
        rhs_tv[-1] = tv_bound

    block_hi = np.hstack([A, -np.eye(m), np.zeros((m, m)),
                          np.zeros((m, n_tv))])
    block_lo = np.hstack([-A, np.zeros((m, m)), -np.eye(m),
                          np.zeros((m, n_tv))])
    blocks = [block_hi, block_lo]
    rhs = [b_hi, -b_lo]
    if tv_bound is not None:
        blocks.append(rows_tv)
        rhs.append(rhs_tv)
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)

    total = n + 2 * m + n_tv
    bounds = [(0, None)] * total
    c = np.zeros(total)
    c[n : n + m] = gamma
    c[n + m : n + 2 * m] = gamma

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success and res.x is not None:
        u = np.maximum(res.x[n : n + m], 0.0)
        w = np.maximum(res.x[n + m : n + 2 * m], 0.0)
        converged = True
    else:
        logger.warning(
            "center-of-uncertainty widening LP failed (status=%s); "
            "reporting zero widening", getattr(res, "status", "unknown")
        )
        u = np.zeros(m)
        w = np.zeros(m)
        converged = False

    lp_tol = _tol_max_lp(A, np.zeros_like(A), b_lo, b_hi, tv_bound, "tol")
    tol_max = lp_tol["tol_max"]
    compatible = bool(tol_max >= 0.0)

    if compatible:
        b_lo_eff = b_lo.copy()
        b_hi_eff = b_hi.copy()
        epsilon = np.zeros(m)
        incompatibility = 0.0
    else:
        # u relaxes the upper bounds (Ax <= b_hi + u), w the lower ones
        # (Ax >= b_lo - w)
        b_lo_eff = np.maximum(b_lo - w, 0.0)
        b_hi_eff = b_hi + u
        epsilon = u + w
        incompatibility = float(np.sum(gamma * epsilon))

    x_min, x_max = solve_interval(A, b_lo_eff, b_hi_eff, tv_bound=tv_bound)

    info = {
        "tol_max": tol_max,
        "x_pseudo": lp_tol["x_pseudo"],
        "epsilon": epsilon,
        "incompatibility": incompatibility,
        "compatible": compatible,
        "b_lo_effective": b_lo_eff,
        "b_hi_effective": b_hi_eff,
        "converged": converged,
        "generators": [int(k) for k in lp_tol["generators"]],
    }
    return x_min, x_max, info


def solve_interval_pia(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    norm: str = "inf",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Simple interval approximation (PIA, Rutkowski).

    Dagesenov et al. 2024, sec. 4.16 (after Rutkowski): estimates the
    spectrum by minimizing ``||e(x)||`` with the *signed* residual of
    formula (2.5)/(4.72)

        e_i(x) = max{(Ax)_i - b_hi_i, b_lo_i - (Ax)_i},

    which is negative when the model value lies strictly inside the
    reading corridor (its absolute value is the distance from the
    corridor boundary). Chebyshev and L1 norms make the objective
    polyhedral, hence the problem is an LP. Unlike max-compatibility,
    PIA ignores the containment semantics: the estimate may leave the
    information set even when it is non-empty (book example 4.16.1).
    At zero interval widths it reduces to Chebyshev (``inf``) or
    weighted L1 approximation of the point data (principle of
    agreement).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization.
    norm : str, optional
        "inf" (Chebyshev, default) or "one".
    weights : np.ndarray, optional
        Positive weights for the "one" norm objective (m,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_pia, e, info) with the estimate and per-reading distances.
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    m, n = A.shape
    if norm not in ("inf", "one"):
        raise ValueError(f"norm must be 'inf' or 'one', got {norm!r}")
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,) or np.any(w <= 0):
            raise ValueError("weights must be positive with shape (m,)")

    # Variables: [x(n), d(m), tau(1 if norm=='inf' else 0), t(n_tv)]
    n_extra = 1 if norm == "inf" else 0
    n_tv = 0
    rows_tv = rhs_tv = None
    if tv_bound is not None:
        rows_tv, rhs_tv, n_tv = _tv_rows(n)
        # move TV rows from [x, t] layout into [x, 0(d), 0(tau), t]
        rows_tv = np.hstack(
            [
                rows_tv[:, :n],
                np.zeros((rows_tv.shape[0], m + n_extra)),
                rows_tv[:, n:],
            ]
        )
        rhs_tv = rhs_tv.copy()
        rhs_tv[-1] = tv_bound
    blocks = [
        np.hstack(
            [A, -np.eye(m), np.zeros((m, n_extra + n_tv))]
        ),
        np.hstack(
            [-A, -np.eye(m), np.zeros((m, n_extra + n_tv))]
        ),
    ]
    rhs = [b_hi, -b_lo]
    if norm == "inf":
        # d_i <= tau  (Chebyshev norm of the distance vector)
        row_t = np.zeros((m, n + m + n_extra + n_tv))
        row_t[:, n : n + m] = np.eye(m)
        row_t[:, n + m] = -1.0
        blocks.append(row_t)
        rhs.append(np.zeros(m))
    if rows_tv is not None:
        blocks.append(rows_tv)
        rhs.append(rhs_tv)

    total = n + m + n_extra + n_tv
    bounds = [(0, None)] * total
    c = np.zeros(total)
    if norm == "inf":
        c[n + m] = 1.0
    else:
        c[n : n + m] = w

    res = linprog(c, A_ub=np.vstack(blocks), b_ub=np.concatenate(rhs),
                  bounds=bounds, method="highs")
    if res.success and res.x is not None:
        x_pia = np.maximum(res.x[:n], 0.0)
        e = np.maximum(res.x[n : n + m], 0.0)
        converged = True
    else:
        logger.warning("PIA LP failed (status=%s)", getattr(res, "status", "unknown"))
        x_pia = np.maximum(
            np.linalg.lstsq(A, (b_lo + b_hi) / 2.0, rcond=None)[0], 0.0
        )
        fitted = A @ x_pia
        e = np.maximum(np.maximum(b_lo - fitted, fitted - b_hi), 0.0)
        converged = False

    info = {
        "norm": norm,
        "max_distance": float(np.max(e)) if m else 0.0,
        "total_distance": float(np.sum(w * e)),
        "mean_distance": float(np.mean(e)) if m else 0.0,
        "distances": e,
        "converged": converged,
        "tv_bound": tv_bound,
    }
    return x_pia, e, info


def interval_sev_variativity(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    x_hat: np.ndarray,
    tol_max: float,
    mode: str = "reduced",
) -> float:
    """X-variativity of a max-compatibility estimate (SEV/TEV).

    Dagesenov et al. 2024, eqs. (4.65)-(4.71)::

        SEV = sqrt(n) * K_s * max_Tol * ||x_hat||_2 / ||b'_||_2

    with ``b'_i = (|mid b_i + rad b_i| + |mid b_i - rad b_i|) / 2`` and
    coverage coefficient ``K_s = cond_2(X)``, the spectral condition
    number of the end-combination matrix X. In ``"reduced"`` mode X
    stacks the two interval-matrix bounds ``[A_lo; A_hi]`` (exact for a
    point matrix: cond_2(A)); ``"full"`` enumerates all vertex rows per
    reading (exponential, capped at ``_FULL_X_MAX_BINS`` variables).

    SEV approximates the half-diameter of the information set: larger
    values mean a larger spread of admissible spectra. It is scale-free
    and uses only quantities already produced by the max-compatibility
    solve.

    Parameters
    ----------
    A_lo : np.ndarray
        Lower bound of the (interval) response matrix (m x n).
    A_hi : np.ndarray
        Upper bound of the response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    x_hat : np.ndarray
        Max-compatibility estimate (argmax Tol).
    tol_max : float
        Maximum of the recognizing functional.
    mode : str, optional
        "reduced" (default) or "full".

    Returns
    -------
    float
        Variativity measure (may be inf for rank-deficient systems).
    """
    if mode not in ("reduced", "full"):
        raise ValueError(f"mode must be 'reduced' or 'full', got {mode!r}")
    A_lo = np.asarray(A_lo, dtype=float)
    A_hi = np.asarray(A_hi, dtype=float)
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    m, n = A_lo.shape

    b_mid = (b_lo + b_hi) / 2.0
    b_rad = (b_hi - b_lo) / 2.0
    b_prime = 0.5 * (np.abs(b_mid + b_rad) + np.abs(b_mid - b_rad))
    norm_b = float(np.linalg.norm(b_prime))
    if norm_b <= 0.0:
        return float("inf")
    if tol_max <= 0.0:
        # empty (or boundary-degenerate) solution set: SEV is proportional
        # to max Tol, hence zero; also avoids the inf * 0 = nan product
        return 0.0

    if mode == "reduced":
        X = np.vstack([A_lo, A_hi])
    else:
        if n > _FULL_X_MAX_BINS:
            raise ValueError(
                f"full end-combination matrix needs n <= "
                f"{_FULL_X_MAX_BINS} bins, got n={n}"
            )
        rows = []
        for i in range(m):
            rows.extend(product(*list(zip(A_lo[i], A_hi[i]))))
        X = np.asarray(rows, dtype=float)

    s = np.linalg.svd(X, compute_uv=False)
    # numerical rank cutoff (as in np.linalg.matrix_rank): a rank-
    # deficient end-combination has no bounded inverse on its range
    tol_rank = max(X.shape) * np.finfo(float).eps * s[0] if s.size else 0.0
    if s.size == 0 or s[-1] <= tol_rank:
        return float("inf")
    cond2 = float(s[0] / s[-1])

    sev = (
        np.sqrt(n)
        * cond2
        * max(float(tol_max), 0.0)
        * float(np.linalg.norm(x_hat))
        / norm_b
    )
    return float(sev)


def interval_compatibility_report(
    b_lo: np.ndarray,
    b_hi: np.ndarray,
) -> dict[str, Any]:
    """Compatibility diagnostics for a sample of interval data.

    Implements the interval-sample characteristics of Dagesenov et al.
    2024, sec. 3.2-3.3: the information interval is the intersection
    ``[max b_lo, min b_hi]``; the sample is compatible iff the
    intersection is non-empty. Reports the sample Jaccard index
    ``Ji = wid(intersection) / width(hull)`` in [-1, 1] (negative =
    incompatible), the incompatibility amount, the maximal number of
    mutually compatible intervals (clique number of the interval
    compatibility graph, computed by a sweep), and the indices whose
    removal restores compatibility (outlier candidates).

    Parameters
    ----------
    b_lo : np.ndarray
        Lower bounds (k,).
    b_hi : np.ndarray
        Upper bounds (k,).

    Returns
    -------
    dict
        compatible, ji_sample, intersection, incompatibility,
        clique_number, outlier_candidates, pairwise_jaccard.
    """
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)
    if b_lo.shape != b_hi.shape or b_lo.ndim != 1:
        raise ValueError("b_lo and b_hi must be 1D arrays of equal shape")
    if np.any(b_lo > b_hi):
        raise ValueError("b_lo must be <= b_hi for all elements")
    k = b_lo.size

    inter_lo = float(np.max(b_lo))
    inter_hi = float(np.min(b_hi))
    compatible = inter_lo <= inter_hi
    wid_inter = max(0.0, inter_hi - inter_lo)
    hull_lo = float(np.min(b_lo))
    hull_hi = float(np.max(b_hi))
    wid_hull = hull_hi - hull_lo
    if wid_hull <= 0.0:
        ji_sample = 1.0
    else:
        ji_sample = wid_inter / wid_hull if compatible else -(
            inter_lo - inter_hi
        ) / wid_hull
    incompatibility = 0.0 if compatible else inter_lo - inter_hi

    # Pairwise Jaccard indices for interval data:
    # Ji(a, b) = wid(a & b) / wid(a | b) with signed intersection width.
    J = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            i_lo = max(b_lo[i], b_lo[j])
            i_hi = min(b_hi[i], b_hi[j])
            o_lo = min(b_lo[i], b_lo[j])
            o_hi = max(b_hi[i], b_hi[j])
            w_union = o_hi - o_lo
            if w_union <= 0.0:
                ji = 1.0
            else:
                ji = (i_hi - i_lo) / w_union  # signed: negative if disjoint
            J[i, j] = ji
            J[j, i] = ji

    # Clique number of the interval graph = max number of intervals
    # covering a common point (sweep over endpoints; intervals closed,
    # so starts are processed before ends at equal coordinates).
    events = sorted(
        [(float(v), 0) for v in b_lo] + [(float(v), 1) for v in b_hi]
    )
    active = 0
    clique = 0
    for _, kind in events:
        active += 1 if kind == 0 else -1
        clique = max(clique, active)

    outlier_candidates: list[int] = []
    if not compatible:
        for j in range(k):
            others_i = np.delete(b_lo, j)
            others_h = np.delete(b_hi, j)
            if others_i.size == 0 or float(np.max(others_i)) <= float(
                np.min(others_h)
            ):
                outlier_candidates.append(j)

    return {
        "compatible": bool(compatible),
        "ji_sample": float(ji_sample),
        "intersection": (inter_lo, inter_hi),
        "incompatibility": float(incompatibility),
        "clique_number": int(clique),
        "outlier_candidates": outlier_candidates,
        "pairwise_jaccard": J,
    }


def _augment_tv(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    x_map: np.ndarray,
    tv_bound: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Append TV variables ``t`` and constraints to an LP row system.

    ``x_map`` (n x n_base) expresses the spectrum ``x`` in the LP's
    existing variables (identity for plain ``x``; ``[I, -I]`` for the
    Rohn ``x' - x''`` split). The rows ``t_i >= |x_{i+1} - x_i|`` and
    ``sum(t) <= tv_bound`` are added with the matching coefficient
    pattern; base rows are padded with zero columns.
    """
    n = x_map.shape[0]
    n_tv = n - 1
    rows, rhs, _ = _tv_rows(n)
    C = rows[:, :n] @ x_map  # (2*n_tv + 1, n_base)
    t_part = rows[:, n:]
    tv_rows = np.hstack([C, t_part])
    rhs = rhs.copy()
    rhs[-1] = tv_bound
    A_ub = np.hstack([A_ub, np.zeros((A_ub.shape[0], n_tv))])
    return np.vstack([A_ub, tv_rows]), np.concatenate([b_ub, rhs]), n_tv


def _rohn_tol_bounds(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Componentwise outer enclosure of the tolerable set (Rohn).

    Uses the x' - x'' parametrization of Dagesenov et al. 2024, eq.
    (4.36): x in Xi_tol iff x = x' - x'', x', x'' >= 0 with
    ``A_hi x' - A_lo x'' <= b_hi`` and ``A_lo x' - A_hi x'' >= b_lo``.
    Spectra additionally require x >= 0 (x' >= x''). With ``tv_bound``
    the total variation of x = x' - x'' is limited as well.
    """
    m, n = A_lo.shape
    A_ub = np.vstack(
        [
            np.hstack([A_hi, -A_lo]),
            np.hstack([-A_lo, A_hi]),
            np.hstack([-np.eye(n), np.eye(n)]),
        ]
    )
    b_ub = np.concatenate([b_hi, -b_lo, np.zeros(n)])
    n_tv = 0
    if tv_bound is not None:
        x_map = np.hstack([np.eye(n), -np.eye(n)])
        A_ub, b_ub, n_tv = _augment_tv(A_ub, b_ub, x_map, tv_bound)
    bounds = [(0, None)] * (2 * n + n_tv)

    x_min = np.zeros(n)
    x_max = np.zeros(n)
    for i in range(n):
        c = np.zeros(2 * n + n_tv)
        c[i] = 1.0
        c[n + i] = -1.0
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
        if res_min.success and res_min.x is not None:
            x_min[i] = max(res_min.fun, 0.0)  # min of x_i' - x_i''
        res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
        if res_max.success and res_max.x is not None:
            x_max[i] = max(-res_max.fun, 0.0)
        if not res_min.success:
            x_min[i] = 0.0
            x_max[i] = 0.0
    x_min = np.maximum(x_min, 0.0)
    x_max = np.maximum(x_max, x_min)
    return x_min, x_max


def _uss_bounds(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Componentwise enclosure of the united set intersected with x >= 0.

    By Belek's criterion (Dagesenov et al. 2024, sec. 2.8/4.8) for the
    nonnegative orthant the united solution set of ``[A] x = [b]`` is the
    polyhedron

        Xi_uni ∩ {x >= 0} = {x >= 0 : A_lo x <= b_hi,  A_hi x >= b_lo},

    so the 2n-LP componentwise enclosure needs no vertex enumeration.
    """
    n = A_lo.shape[1]
    A_ub = np.vstack([A_lo, -A_hi])
    b_ub = np.concatenate([b_hi, -b_lo])
    n_tv = 0
    if tv_bound is not None:
        A_ub, b_ub, n_tv = _augment_tv(
            A_ub, b_ub, np.eye(n), tv_bound
        )
    bounds = [(0, None)] * (n + n_tv)
    return _component_bounds(A_ub, b_ub, bounds, n)


def _khlebnikov_inner_box(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Inner box of the tolerable set (Khlebnikov's LP).

    Maximizes the width of a box [c - z, c + z] (x >= 0 branch) all of
    whose 2^n vertices lie in the tolerable set; sufficient because the
    set is convex. Exponential in n; capped at ``_INNER_BOX_MAX_BINS``.
    """
    m, n = A_lo.shape
    if n > _INNER_BOX_MAX_BINS:
        return None
    A_mid = (A_lo + A_hi) / 2.0
    A_rad = (A_hi - A_lo) / 2.0
    signs = np.asarray(
        list(product(*([[1.0, -1.0]] * n))), dtype=float
    )  # (2^n, n)
    hi = A_mid + A_rad
    lo = A_mid - A_rad

    blocks = []
    rhs = []
    for S in signs:
        SH = hi * S  # column scaling
        SL = lo * S
        # hi (c + S z) <= b_hi
        blocks.append(np.hstack([hi, SH]))
        rhs.append(b_hi)
        # lo (c + S z) >= b_lo
        blocks.append(np.hstack([-lo, -SL]))
        rhs.append(-b_lo)
        # c + S z >= 0 (box inside the nonnegative orthant)
        D = np.zeros((n, 2 * n))
        D[:, :n] = -np.eye(n)
        D[:, n:] = -np.diag(S)
        blocks.append(D)
        rhs.append(np.zeros(n))
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    bounds = [(0, None)] * (2 * n)
    c = np.zeros(2 * n)
    c[n:] = -1.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None or -res.fun <= 0.0:
        return None
    cen = res.x[:n]
    wid = res.x[n:]
    lower = np.maximum(cen - wid, 0.0)
    upper = cen + wid
    return lower, upper


def solve_interval_matrix(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    functional: str = "tol",
    inner_box: bool = False,
    variativity: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interval system with an uncertain response matrix (Kaucher/Rohn).

    Handles simultaneous interval uncertainty of the readings ``b`` and
    of the response matrix ``A`` (detector sensitivity errors). The
    point estimate maximizes the recognizing functional exactly via an
    LP: ``Tol`` (default, strong compatibility: admissible for *all*
    matrices in the interval family, tolerable set Xi_tol) or ``Uss``
    (weak compatibility: united set Xi_uni). Componentwise enclosures:

    - ``tol``: Rohn's theorem (4.36) via 2n LPs in x' - x'' variables;
    - ``uss``: Belek's polyhedron ``{x >= 0 : A_lo x <= b_hi, A_hi x >=
      b_lo}`` via 2n LPs.

    Optionally tightens the enclosure with Khlebnikov's inner box for
    small n and reports the SEV variativity of the max-compatibility
    estimate.

    Parameters
    ----------
    A_lo : np.ndarray
        Lower bound of the response matrix (m x n).
    A_hi : np.ndarray
        Upper bound of the response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    functional : str, optional
        "tol" (default) or "uss".
    tv_bound : float, optional
        Total variation bound applied to the enclosure LPs (the Khlebnikov
        inner box is not evaluated under TV).
    inner_box : bool, optional
        Compute the Khlebnikov inner box (n <= 10).
    variativity : bool, optional
        Compute the SEV/TEV variativity measure.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) with tol_max, x_pseudo, generators,
        violations, sev, inner_lower/upper, converged.
    """
    A_lo, A_hi, b_lo, b_hi = _validate_interval_matrix(
        A_lo, A_hi, b_lo, b_hi
    )
    if functional not in ("tol", "uss"):
        raise ValueError(
            f"functional must be 'tol' or 'uss', got {functional!r}"
        )
    m, n = A_lo.shape
    A_mid = (A_lo + A_hi) / 2.0
    A_rad = (A_hi - A_lo) / 2.0

    lp = _tol_max_lp(A_mid, A_rad, b_lo, b_hi, tv_bound, functional)

    if functional == "tol":
        x_min, x_max = _rohn_tol_bounds(A_lo, A_hi, b_lo, b_hi, tv_bound)
    else:
        x_min, x_max = _uss_bounds(A_lo, A_hi, b_lo, b_hi, tv_bound)

    info: dict[str, Any] = {
        "tol_max": lp["tol_max"],
        "x_pseudo": lp["x_pseudo"],
        "generators": [int(k) for k in lp["generators"]],
        "violations": lp["violations"],
        "functional": functional,
        "converged": lp["converged"],
        "inner_lower": None,
        "inner_upper": None,
        "sev": None,
    }

    if inner_box and functional == "tol" and tv_bound is None:
        box = _khlebnikov_inner_box(A_lo, A_hi, b_lo, b_hi)
        if box is not None:
            info["inner_lower"], info["inner_upper"] = box

    if variativity:
        info["sev"] = interval_sev_variativity(
            A_lo, A_hi, b_lo, b_hi, lp["x_pseudo"], lp["tol_max"]
        )

    return x_min, x_max, info


def _inflation_radius(
    A: np.ndarray,
    tau: float,
    inflation: str,
) -> np.ndarray:
    """Radius matrix of the interval inflation ``[A] = A +- A_rad``.

    ``"shift"`` puts the symmetric shift ``tau`` on the main diagonal
    (``A + tau I``, the Lavrentiev shift applied in all directions at
    once); ``"relative"`` widens every element proportionally,
    ``tau * |a_ij|``.
    """
    if inflation not in ("shift", "relative"):
        raise ValueError(
            f"inflation must be 'shift' or 'relative', got {inflation!r}"
        )
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if inflation == "shift":
        return tau * np.eye(*A.shape)
    return tau * np.abs(A)


def _smooth_on_max_face(
    A: np.ndarray,
    A_rad: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tau_floor: float,
    tv_bound: float | None = None,
    functional: str = "tol",
    mode: str = "curvature",
    hull_upper: np.ndarray | None = None,
) -> np.ndarray | None:
    """Pick a smooth representative of the max-Tol face ``Tol(x) >= floor``.

    The LP argmax of the recognizing functional is a vertex of the face,
    typically spiky. Restricting to the face (for ``x >= 0`` the inequality
    ``Tol(x) >= tau_floor`` is linear: ``|A x - mid(b)| + A_rad x <=
    rad(b) - tau_floor``) and minimizing the first- or second-difference
    absolute sum (``"variation"`` / ``"curvature"``, LP-exact) returns a
    smooth point attaining the same tolerance.

    ``hull_upper`` (the componentwise enclosure, valid whenever
    ``tol_max >= 0``) pins the face in directions the response matrix
    cannot see (dead columns), where the tolerance set is genuinely
    unbounded and the smoothing LP could otherwise drift.

    Returns ``None`` when the face LP fails.
    """
    m, n = A.shape
    if functional == "tol":
        row_hi = A + A_rad
        row_lo = -(A - A_rad)
    else:
        row_hi = A - A_rad
        row_lo = -(A + A_rad)

    order = 1 if mode == "variation" else 2
    k = n - order
    n_tv = n - 1 if tv_bound is not None else 0
    total = n + k + n_tv

    def _tv_block() -> tuple[np.ndarray, np.ndarray]:
        """TV rows laid out as [x(n), u(k), t(n_tv)]."""
        rows_tv, rhs_tv, _ = _tv_rows(n)
        if k:
            rows_tv = np.hstack([
                rows_tv[:, :n],
                np.zeros((rows_tv.shape[0], k)),
                rows_tv[:, n:],
            ])
        rhs_tv = rhs_tv.copy()
        rhs_tv[-1] = tv_bound
        return rows_tv, rhs_tv

    if k < 1:
        return None
    diff = np.zeros((k, n))
    for i in range(k):
        if order == 1:
            diff[i, i] = -1.0
            diff[i, i + 1] = 1.0
        else:
            diff[i, i] = 1.0
            diff[i, i + 1] = -2.0
            diff[i, i + 2] = 1.0
    eye_k = np.eye(k)
    pad_u = np.hstack([diff, -eye_k, np.zeros((k, n_tv))])
    blocks = [
        np.hstack([row_hi, np.zeros((m, total - n))]),
        np.hstack([row_lo, np.zeros((m, total - n))]),
        pad_u,
        np.hstack([-diff, -eye_k, np.zeros((k, n_tv))]),
    ]
    rhs = [b_hi - tau_floor, -b_lo - tau_floor,
           np.zeros(k), np.zeros(k)]
    if hull_upper is not None:
        blocks.append(
            np.hstack([np.eye(n), np.zeros((n, total - n))])
        )
        rhs.append(np.asarray(hull_upper, dtype=float))
    if n_tv:
        rows_tv, rhs_tv = _tv_block()
        blocks.append(rows_tv)
        rhs.append(rhs_tv)
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    bounds = [(0, None)] * total
    c = np.zeros(total)
    c[n:n + k] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        logger.warning(
            "Smoothing LP (%s) on the max-Tol face did not converge "
            "(status=%s); keeping the vertex pseudo-solution",
            mode, getattr(res, "status", "unknown"),
        )
        return None
    return np.maximum(res.x[:n], 0.0)


def solve_interval_regularization(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tau: float = 0.05,
    inflation: str = "shift",
    tv_bound: float | None = None,
    functional: str = "tol",
    inner_box: bool = False,
    tau_grid: list[float] | None = None,
    smoothing: str = "none",
    face_slack: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interval regularization of an ill-conditioned point system.

    Implements Shary's interval regularization (2017): the point
    response matrix ``A`` is embedded into an interval family by
    *inflating* it — ``inflation="shift"`` gives ``[A] = A + [-tau, tau] I``
    (a Lavrentiev shift in all directions at once), ``inflation="relative"``
    widens each element by ``tau * |a_ij|``. The regularized estimate is
    the pseudo-solution, i.e. the maximizer of the recognizing functional
    Tol (or Uss) over the inflated family, computed exactly by LP; the
    componentwise bounds enclose the tolerable (united) solution set via
    :func:`solve_interval_matrix`. ``tau`` is the regularization
    parameter: as ``tau -> 0`` the estimate tends to the Chebyshev
    solution of the point system, and growing ``tau`` increasingly
    stabilizes the tolerable set (which shrinks, while the united set
    expands). ``tau_grid`` adds a cheap one-LP-per-level sweep of
    ``(tau, tol_max, residual, x_pseudo)`` for selecting the inflation
    level (the choice of ``tau`` is an open question, analogous to the
    shift selection in classical Lavrentiev regularization).

    Parameters
    ----------
    A : np.ndarray
        Point response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tau : float, optional
        Inflation level (regularization parameter), default 0.05.
    inflation : str, optional
        "shift" (diagonal, default) or "relative" (elementwise).
    tv_bound : float, optional
        Total variation bound for the enclosure LPs.
    functional : str, optional
        "tol" (default) or "uss".
    inner_box : bool, optional
        Compute Khlebnikov's inner box (small n only).
    tau_grid : list[float], optional
        Additional inflation levels to sweep for parameter selection.
    smoothing : str, optional
        Replace the spiky LP vertex by a smooth representative of the
        max-Tol face: "none" (default, vertex), "curvature" (minimize
        the second-difference absolute sum) or "variation" (first
        differences) — both LP-exact. Adds ``x_smoothed`` to info; the
        tolerance level of the smoothed point is
        ``tol_max - face_slack * |tol_max|``.
    face_slack : float, optional
        Relative slack (0..1) of the face constraint when smoothing
        (default 0, stay on the exact maximum face).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) with the same keys as
        :func:`solve_interval_matrix` plus ``tau`` and ``inflation``,
        ``smoothing``, ``face_slack`` and ``x_smoothed`` (when
        smoothing succeeded), and ``tau_sweep`` (list of dicts) when
        ``tau_grid`` is given.
    """
    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    if functional not in ("tol", "uss"):
        raise ValueError(
            f"functional must be 'tol' or 'uss', got {functional!r}"
        )
    if smoothing not in ("none", "curvature", "variation"):
        raise ValueError(
            "smoothing must be 'none', 'curvature' or 'variation', "
            f"got {smoothing!r}"
        )
    if not 0.0 <= face_slack < 1.0:
        raise ValueError("face_slack must be in [0, 1)")
    A_rad = _inflation_radius(A, float(tau), inflation)
    x_min, x_max, info = solve_interval_matrix(
        A - A_rad,
        A + A_rad,
        b_lo,
        b_hi,
        tv_bound=tv_bound,
        functional=functional,
        inner_box=inner_box,
        variativity=False,
    )
    info["tau"] = float(tau)
    info["inflation"] = inflation

    if smoothing != "none":
        tol_max = float(info["tol_max"])
        tau_floor = tol_max - face_slack * abs(tol_max)
        hull = x_max if tol_max >= 0.0 else None
        x_smoothed = _smooth_on_max_face(
            A, A_rad, b_lo, b_hi, tau_floor, tv_bound, functional,
            smoothing, hull_upper=hull,
        )
        if x_smoothed is not None:
            info["x_smoothed"] = x_smoothed
    info["smoothing"] = smoothing
    info["face_slack"] = float(face_slack)

    if tau_grid is not None:
        b_mid = (b_lo + b_hi) / 2.0
        sweep: list[dict[str, Any]] = []
        for t in tau_grid:
            rad_t = _inflation_radius(A, float(t), inflation)
            lp_t = _tol_max_lp(A, rad_t, b_lo, b_hi, tv_bound, functional)
            residual = float(
                np.max(np.abs(b_mid - A @ lp_t["x_pseudo"]))
            )
            sweep.append(
                {
                    "tau": float(t),
                    "tol_max": lp_t["tol_max"],
                    "residual_inf": residual,
                    "x_pseudo": lp_t["x_pseudo"],
                    "converged": lp_t["converged"],
                }
            )
        info["tau_sweep"] = sweep

    return x_min, x_max, info


def _intervals_from_readings(
    b: np.ndarray,
    selected: list[str],
    detector_names: list[str],
    reading_uncertainties: dict[str, float] | np.ndarray | None,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build reading intervals [b_lo, b_hi] from uncertainties or noise."""
    if reading_uncertainties is not None:
        if isinstance(reading_uncertainties, dict):
            delta = np.array(
                [reading_uncertainties.get(name, 0.0) for name in selected],
                dtype=float,
            )
        else:
            delta = np.asarray(reading_uncertainties, dtype=float)
            if delta.shape == (len(detector_names),):
                delta = np.array(
                    [delta[detector_names.index(name)] for name in selected]
                )
    else:
        delta = noise_level * b
    b_lo = np.maximum(b - delta, 0.0)
    b_hi = b + delta
    return b_lo, b_hi


def _dose_bounds(
    x_min: np.ndarray,
    x_max: np.ndarray,
    cc_icrp116: dict[str, np.ndarray],
    ln_steps: np.ndarray | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Dose-rate bounds from per-bin spectrum bounds (linear monotone)."""
    from .dose_calculation import calculate_dose_rates

    if ln_steps is not None:
        lower = calculate_dose_rates(x_min, cc_icrp116, dlnE_array=ln_steps)
        upper = calculate_dose_rates(x_max, cc_icrp116, dlnE_array=ln_steps)
    else:
        lower = calculate_dose_rates(x_min, cc_icrp116)
        upper = calculate_dose_rates(x_max, cc_icrp116)
    return lower, upper


def _width_metrics(
    x_min: np.ndarray,
    x_max: np.ndarray,
) -> dict[str, np.ndarray]:
    width = x_max - x_min
    mid = (x_min + x_max) / 2.0
    rel = width / np.maximum(mid, 1e-30)
    return {"width": width, "relative_width": rel}


def _interval_spectrum_output(
    x_mid: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    selected: list[str],
    cc_icrp116: dict[str, np.ndarray],
    method: str,
    extra: dict[str, Any],
    ln_steps: np.ndarray | None,
    noise_level: float,
    tv_bound: float | None,
) -> dict[str, Any]:
    """Assemble the standard output with shared interval diagnostics."""
    dose_lower, dose_upper = _dose_bounds(x_min, x_max, cc_icrp116, ln_steps)
    base_extra = {
        "spectrum_lower": x_min,
        "spectrum_upper": x_max,
        "tv_bound": tv_bound,
        "noise_level": noise_level,
        "doserates_lower": dose_lower,
        "doserates_upper": dose_upper,
    }
    base_extra.update(_width_metrics(x_min, x_max))
    base_extra.update(extra)
    return _standardize_output(
        spectrum=x_mid,
        A=A,
        b=b,
        E_MeV=E_MeV,
        selected=selected,
        cc_icrp116=cc_icrp116,
        method=method,
        extra=base_extra,
        ln_steps=ln_steps,
    )


def unfold_interval(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    compatibility_report: bool = False,
) -> dict[str, Any]:
    """Unfold using interval linear programming.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    compatibility_report : bool, optional
        If True, add a 'compatibility' key with pairwise corridor
        compatibility diagnostics (Jaccard indices, maximal
        compatible subsets, outlier candidates).

    Returns
    -------
    dict[str, Any]
        Unfolding results with 'spectrum_lower'/'spectrum_upper',
        'doserates_lower'/'doserates_upper' and per-bin 'width' metrics.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )

    x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=tv_bound)
    x_mid = (x_min + x_max) / 2.0

    extra: dict[str, Any] = {}
    if compatibility_report:
        extra["compatibility"] = interval_compatibility_report(b_lo, b_hi)

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalLP", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_tol(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-6,
    drop_infeasible: int = 0,
    functional: str = "tol",
    variativity: bool = False,
    weights: np.ndarray | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Unfold using Shary's recognizing functional method.

    Maximizes the recognizing functional (Tol/Uss) exactly via an LP,
    then computes guaranteed bounds with the interval LP machinery.
    Reports the compatibility reserve ``tol_max`` (negative = empty
    information set), the active generator readings and per-reading
    violations; ``drop_infeasible`` removes the worst-violating readings
    as an outlier workflow. ``weights`` sets the value of each reading
    in the functional (as in tolsolvty), and
    ``generators_profile`` lists all generator values sorted ascending.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    max_iter : int, optional
        Legacy parameter, unused by the LP core.
    tol : float, optional
        Legacy parameter, unused by the LP core.
    drop_infeasible : int, optional
        Number of worst-violating readings to drop if incompatible.
    functional : str, optional
        "tol" (default) or "uss".
    variativity : bool, optional
        Also compute the SEV variativity of the pseudo-solution.
    weights : np.ndarray or dict, optional
        Positive per-reading weights (m,) or a dict keyed by detector
        name (unlisted readings default to 1).

    Returns
    -------
    dict[str, Any]
        Unfolding results with tol_max, x_pseudo, generators,
        generators_profile, violations, dropped and (optionally) sev
        keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )
    if isinstance(weights, dict):
        weights = np.array([float(weights.get(name, 1.0)) for name in selected])

    x_min, x_max, info = solve_interval_tol(
        A,
        b_lo,
        b_hi,
        tv_bound=tv_bound,
        max_iter=max_iter,
        tol=tol,
        drop_infeasible=drop_infeasible,
        functional=functional,
        weights=weights,
    )

    x_mid = (x_min + x_max) / 2.0

    extra: dict[str, Any] = {
        "tol_max": info["tol_max"],
        "x_pseudo": info["x_pseudo"],
        "converged": info["converged"],
        "n_iter": info["n_iter"],
        "generators": info["generators"],
        "generators_profile": info["generators_profile"],
        "violations": info["violations"],
        "dropped_readings": [
            selected[i] for i in info["dropped"] if i < len(selected)
        ],
        "functional": info["functional"],
    }
    if variativity:
        extra["sev"] = interval_sev_variativity(
            A, A, b_lo, b_hi, info["x_pseudo"], info["tol_max"]
        )

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalTol", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_posterior(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    n_samples: int = 100,
    normalize: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold using posterior interval analysis.

    Computes the guaranteed interval LP enclosure and adds a Monte-Carlo
    sample cloud inside the information set (random-direction LP
    sampling): the empirical envelope ``spectrum_mc_lower/upper`` is a
    statistically motivated inner approximation, and
    ``spectrum_mc_std`` provides per-bin spread statistics. MC sampling
    is a heuristic, not a guarantee: the guaranteed bounds remain
    ``spectrum_lower``/``spectrum_upper``.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    n_samples : int, optional
        Number of Monte-Carlo samples for the inner envelope.
    normalize : bool, optional
        If True, normalize the spectrum to match the total fluence.
    random_state : int, optional
        Seed for the Monte-Carlo sampling.

    Returns
    -------
    dict[str, Any]
        Unfolding results with guaranteed bounds and MC envelope keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )

    x_min, x_max, info = solve_interval_posterior(
        A,
        b_lo,
        b_hi,
        tv_bound=tv_bound,
        n_samples=n_samples,
        normalize=normalize,
        random_state=random_state,
    )

    x_mid = (x_min + x_max) / 2.0

    extra = {
        "n_samples": info["n_samples"],
        "n_feasible": info["n_feasible"],
        "sensitivity": info["sensitivity"],
        "residuals": info["residuals"],
        "normalize": normalize,
        "spectrum_mc_lower": info["spectrum_mc_lower"],
        "spectrum_mc_upper": info["spectrum_mc_upper"],
        "spectrum_mc_mean": info["spectrum_mc_mean"],
        "spectrum_mc_std": info["spectrum_mc_std"],
        "posterior_mc_inner": True,
    }

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalPosterior", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_center(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    weights: dict[str, float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Unfold with the center-of-uncertainty method + minimal widening.

    See :func:`solve_interval_center`. In addition to the interval
    bounds, reports the compatibility flag, per-reading widening
    ``epsilon`` (0 for compatible readings) and the weighted
    ``incompatibility`` measure.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    weights : dict or np.ndarray, optional
        Positive widening weights gamma per reading.

    Returns
    -------
    dict[str, Any]
        Unfolding results with epsilon, incompatibility, compatible,
        b_lo_effective, b_hi_effective keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )
    if weights is not None:
        if isinstance(weights, dict):
            w_arr = np.array(
                [weights.get(name, 1.0) for name in selected], dtype=float
            )
        else:
            w_arr = np.asarray(weights, dtype=float)
    else:
        w_arr = None

    x_min, x_max, info = solve_interval_center(
        A, b_lo, b_hi, tv_bound=tv_bound, weights=w_arr
    )

    x_mid = (x_min + x_max) / 2.0

    extra = {
        "tol_max": info["tol_max"],
        "x_pseudo": info["x_pseudo"],
        "epsilon": info["epsilon"],
        "incompatibility": info["incompatibility"],
        "compatible": info["compatible"],
        "b_lo_effective": info["b_lo_effective"],
        "b_hi_effective": info["b_hi_effective"],
        "generators": info["generators"],
        "converged": info["converged"],
    }

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalCenter", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_pia(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    norm: str = "inf",
    weights: dict[str, float] | np.ndarray | None = None,
    compute_bounds: bool = False,
) -> dict[str, Any]:
    """Unfold with the simple interval approximation (PIA).

    See :func:`solve_interval_pia`. The standardized ``spectrum`` is the
    PIA point estimate (LP for the Chebyshev/L1 distance to the reading
    corridors); ``distances`` reports the per-reading distances.
    ``spectrum_lower``/``spectrum_upper`` are only filled with the
    guaranteed interval bounds when ``compute_bounds=True`` (cost 2n
    extra LPs), otherwise they mirror the estimate.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    norm : str, optional
        "inf" (Chebyshev, default) or "one".
    weights : dict or np.ndarray, optional
        Positive weights per reading for the "one" norm.
    compute_bounds : bool, optional
        Also compute guaranteed interval LP bounds.

    Returns
    -------
    dict[str, Any]
        Unfolding result with distances/max_distance keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )
    w_arr = None
    if weights is not None:
        if isinstance(weights, dict):
            w_arr = np.array(
                [weights.get(name, 1.0) for name in selected], dtype=float
            )
        else:
            w_arr = np.asarray(weights, dtype=float)

    x_pia, e, info = solve_interval_pia(
        A, b_lo, b_hi, tv_bound=tv_bound, norm=norm, weights=w_arr
    )

    if compute_bounds:
        x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=tv_bound)
    else:
        x_min = x_pia.copy()
        x_max = x_pia.copy()

    extra = {
        "norm": info["norm"],
        "distances": e,
        "max_distance": info["max_distance"],
        "total_distance": info["total_distance"],
        "mean_distance": info["mean_distance"],
        "converged": info["converged"],
        "bounds_computed": compute_bounds,
    }

    output = _interval_spectrum_output(
        x_pia, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalPIA", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_matrix(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    save_result: bool = False,
    sensitivity_uncertainties: (
        dict[str, float] | np.ndarray | float | None
    ) = None,
    A_lo: np.ndarray | None = None,
    A_hi: np.ndarray | None = None,
    tv_bound: float | None = None,
    functional: str = "tol",
    inner_box: bool = False,
    variativity: bool = True,
) -> dict[str, Any]:
    """Unfold with interval response matrix (Kaucher/Rohn tolerability).

    Propagates uncertainty of the detector sensitivities (relative 1-sigma
    values per detector, building ``A_lo = A (1 - u)``, ``A_hi = A (1 +
    u)``, or given directly via ``A_lo``/``A_hi``) together with reading
    intervals. The estimate maximizes Tol (all admissible response
    matrices, default) or Uss (united-set, weak compatibility) via an LP;
    the bounds are the componentwise enclosure of the corresponding
    solution set. Reports the SEV variativity by default.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    save_result : bool, optional
        If True, save result to history.
    sensitivity_uncertainties : dict, np.ndarray or float, optional
        Relative uncertainty of sensitivities per detector (dict by name,
        array over ``detector_names``, or scalar). Ignored when
        ``A_lo``/``A_hi`` are provided.
    A_lo : np.ndarray, optional
        Explicit lower response-matrix bound (m x n) for the selected
        detectors.
    A_hi : np.ndarray, optional
        Explicit upper response-matrix bound (m x n).
    functional : str, optional
        "tol" (default) or "uss".
    tv_bound : float, optional
        Total variation bound for the enclosure LPs.
    inner_box : bool, optional
        Compute Khlebnikov inner box (n <= 10).
    variativity : bool, optional
        Compute the SEV variativity measure.

    Returns
    -------
    dict[str, Any]
        Unfolding results with tol_max, x_pseudo, generators, sev,
        inner_lower/upper keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )

    if A_lo is None or A_hi is None:
        if sensitivity_uncertainties is None:
            raise ValueError(
                "unfold_interval_matrix requires sensitivity_uncertainties "
                "or explicit A_lo/A_hi bounds"
            )
        if isinstance(sensitivity_uncertainties, dict):
            u = np.array(
                [sensitivity_uncertainties.get(name, 0.0) for name in selected],
                dtype=float,
            )
        elif np.ndim(sensitivity_uncertainties) == 0:
            u = np.full(len(selected), float(sensitivity_uncertainties))
        else:
            arr = np.asarray(sensitivity_uncertainties, dtype=float)
            if arr.shape == (len(detector_names),):
                arr = np.array(
                    [arr[detector_names.index(name)] for name in selected]
                )
            u = arr
        if np.any(u < 0) or np.any(u >= 1):
            raise ValueError(
                "sensitivity_uncertainties must be relative values in [0, 1)"
            )
        A_lo = A * (1.0 - u[:, None])
        A_hi = A * (1.0 + u[:, None])

    x_min, x_max, info = solve_interval_matrix(
        np.asarray(A_lo, dtype=float),
        np.asarray(A_hi, dtype=float),
        b_lo,
        b_hi,
        tv_bound=tv_bound,
        functional=functional,
        inner_box=inner_box,
        variativity=variativity,
    )

    x_mid = (x_min + x_max) / 2.0

    extra = {
        "tol_max": info["tol_max"],
        "x_pseudo": info["x_pseudo"],
        "converged": info["converged"],
        "generators": info["generators"],
        "violations": info["violations"],
        "functional": info["functional"],
        "sev": info["sev"],
        "inner_lower": info["inner_lower"],
        "inner_upper": info["inner_upper"],
        "sensitivity_uncertainties": sensitivity_uncertainties,
    }

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalMatrix", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_regularization(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    save_result: bool = False,
    tau: float = 0.05,
    inflation: str = "shift",
    tv_bound: float | None = None,
    functional: str = "tol",
    inner_box: bool = False,
    tau_grid: list[float] | None = None,
    smoothing: str = "none",
    face_slack: float = 0.0,
) -> dict[str, Any]:
    """Unfold via Shary's interval regularization of the point matrix.

    Embeds the ill-conditioned point response matrix into an interval
    family by inflating it (``inflation="shift"``: diagonal ``tau I``
    shift, the Lavrentiev shift in all directions at once;
    ``inflation="relative"``: each element widened by ``tau * |a_ij|``)
    and takes the argmax of the recognizing functional as the
    regularized spectrum; ``spectrum_lower/upper`` enclose the tolerable
    solution set of the inflated system. ``tau`` is the regularization
    parameter (``tau -> 0`` recovers the Chebyshev solution);
    ``tau_grid`` reports a sweep for its selection.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    save_result : bool, optional
        If True, save result to history.
    tau : float, optional
        Inflation level / regularization parameter (default: 0.05).
    inflation : str, optional
        "shift" (default) or "relative".
    tv_bound : float, optional
        Total variation bound for the enclosure LPs.
    functional : str, optional
        "tol" (default) or "uss".
    inner_box : bool, optional
        Compute Khlebnikov inner box (n <= 10).
    tau_grid : list[float], optional
        Inflation levels to sweep for parameter selection.
    smoothing : str, optional
        "none" (default, spiky LP vertex), "curvature" or "variation"
        — smooth representative of the max-Tol face used as
        ``spectrum`` (see :func:`solve_interval_regularization`).
    face_slack : float, optional
        Relative slack of the face constraint when smoothing.

    Returns
    -------
    dict[str, Any]
        Unfolding result whose ``spectrum`` is the regularized
        pseudo-solution (smoothed when requested), with tol_max, tau,
        inflation and (optionally) tau_sweep keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )

    x_min, x_max, info = solve_interval_regularization(
        A,
        b_lo,
        b_hi,
        tau=tau,
        inflation=inflation,
        tv_bound=tv_bound,
        functional=functional,
        inner_box=inner_box,
        tau_grid=tau_grid,
        smoothing=smoothing,
        face_slack=face_slack,
    )

    extra = {
        "tol_max": info["tol_max"],
        "x_pseudo": info["x_pseudo"],
        "converged": info["converged"],
        "generators": info["generators"],
        "violations": info["violations"],
        "functional": info["functional"],
        "tau": info["tau"],
        "inflation": info["inflation"],
        "inner_lower": info["inner_lower"],
        "inner_upper": info["inner_upper"],
        "smoothing": info["smoothing"],
        "face_slack": info["face_slack"],
    }
    if "tau_sweep" in info:
        extra["tau_sweep"] = info["tau_sweep"]

    x_est = info.get("x_smoothed", info["x_pseudo"])
    if "x_smoothed" in info:
        extra["x_smoothed"] = info["x_smoothed"]

    output = _interval_spectrum_output(
        x_est, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalRegularization", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def solve_interval_intvalpy(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    method: str = "rohn",
    normalize: bool = False,
    regularization: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve interval system using the intvalpy package.

    Uses Tol.maximize to find a pseudo-solution, then computes bounds
    using LP for each variable. If the tolerance set is empty, uses
    the pseudo-solution as the best approximation.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    method : str, optional
        Method for finding bounds: "rohn" (default) or "shary".
    normalize : bool, optional
        If True, normalize the spectrum to match the total fluence.
    regularization : float, optional
        Regularization parameter for Tikhonov regularization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) where info contains metadata.
    """
    try:
        import intvalpy as ip
    except ImportError:
        raise ImportError(
            "intvalpy is required for solve_interval_intvalpy. "
            "Install it with: pip install intvalpy"
        ) from None

    A, b_lo, b_hi = _validate_interval_system(A, b_lo, b_hi)
    n = A.shape[1]
    b_mid = (b_lo + b_hi) / 2.0

    A_int = ip.Interval(A, A)
    b_int = ip.Interval(b_lo, b_hi)

    x_pseudo, tol_max, n_iter, n_calls, exit_code = ip.Tol.maximize(A_int, b_int)

    x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=None)

    if tol_max < 0:
        x_min = np.maximum(x_pseudo - abs(tol_max), 0.0)
        x_max = x_pseudo + abs(tol_max)

    if regularization is not None and regularization > 0:
        x_mid_ = (x_min + x_max) / 2.0
        D = np.diff(np.eye(n), axis=0)
        A_reg = np.vstack([A, regularization * D])
        b_reg = np.concatenate([b_mid, np.zeros(n - 1)])
        res_reg = linprog(
            np.zeros(n),
            A_ub=np.vstack([A_reg, -A_reg]),
            b_ub=np.concatenate([b_reg, -b_reg]),
            bounds=[(0, None)] * n,
            method="highs",
        )
        if res_reg.success:
            x_reg = res_reg.x
            x_min = np.maximum(x_reg - (x_max - x_min) / 2.0, 0.0)
            x_max = x_reg + (x_max - x_min) / 2.0

    if normalize:
        x_mid_ = (x_min + x_max) / 2.0
        total = np.sum(x_mid_)
        if total > 0:
            scale = np.sum(A @ x_mid_) / total
            x_min = x_min * scale
            x_max = x_max * scale

    info = {
        "tol_max": float(tol_max),
        "x_pseudo": x_pseudo,
        "n_iter": n_iter,
        "n_calls": n_calls,
        "exit_code": exit_code,
        "method": method,
        "normalize": normalize,
        "regularization": regularization,
    }

    return x_min, x_max, info


def unfold_interval_intvalpy(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    method: str = "rohn",
    normalize: bool = False,
    regularization: float | None = None,
) -> dict[str, Any]:
    """Unfold using interval analysis with the intvalpy package.

    This method uses the intvalpy package for interval analysis.
    It finds a pseudo-solution using Tol.maximize, then computes
    bounds using LP for each variable. If the tolerance set is empty,
    uses the pseudo-solution as the best approximation.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    method : str, optional
        Method for finding bounds: "rohn" (default) or "shary".
    normalize : bool, optional
        If True, normalize the spectrum to match the total fluence.
    regularization : float, optional
        Regularization parameter for Tikhonov regularization.

    Returns
    -------
    dict[str, Any]
        Unfolding results with 'spectrum_lower' and 'spectrum_upper' keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    b_lo, b_hi = _intervals_from_readings(
        b, selected, detector_names, reading_uncertainties, noise_level
    )

    x_min, x_max, info = solve_interval_intvalpy(
        A,
        b_lo,
        b_hi,
        method=method,
        normalize=normalize,
        regularization=regularization,
    )

    x_mid = (x_min + x_max) / 2.0

    extra = {
        "tol_max": info["tol_max"],
        "x_pseudo": info["x_pseudo"],
        "n_iter": info["n_iter"],
        "n_calls": info["n_calls"],
        "exit_code": info["exit_code"],
        "intvalpy_method": method,
        "normalize": normalize,
        "regularization": regularization,
    }

    output = _interval_spectrum_output(
        x_mid, x_min, x_max, A, b, E_MeV, selected, cc_icrp116,
        method="IntervalIntvalpy", extra=extra, ln_steps=ln_steps,
        noise_level=noise_level, tv_bound=tv_bound,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output
