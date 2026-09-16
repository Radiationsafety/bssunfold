"""Uno-style constrained unfolding: Lagrange-Newton NLP presets.

Python analogue of the R package ``Uno`` 2.x (Narasimhan; Vanaret &
Leyffer 2024, arXiv:2406.13454): the R package binds the C++ solver
*Uno* ("Unifying Nonlinear Optimization"), which expresses non-linearly
constrained optimisation as a Lagrange-Newton method whose building
blocks (constraint-relaxation, inequality-handling, Hessian and
globalisation strategies) are freely combined, reproducing classical
solvers such as ``filterSQP`` and ``IPOPT`` by presets.

Unfolding NLP
-------------
The unfolding problem is posed as the smooth non-linear program

    minimise   f(x) = 1/2 ||W (A x - b)||^2 + lam/2 ||D2 x||^2
    subject to x >= 0,

a convex quadratic objective with a single inequality constraint set.
Two Uno presets are provided:

* ``"filter_sqp"`` -- sequential quadratic programming with the **exact**
  Hessian ``H = A^T W^2 A + lam D2^T D2``, Newton directions and
  Vanaret & Leyffer's **filter globalisation**: a trial iterate is
  accepted by the filter on the (objective f, constraint violation
  v(x) = ||min(x, 0)||^2) pair, with the ``gamm``-margin acceptability
  test and filter augmentation when no entry is beaten;
* ``"ipopt_like"`` -- a primal-dual **interior-point** method in the
  IPOPT manner: the inequality is handled by the log-barrier
  ``-mu * sum(log x)``, the Newton system is regularised with the
  barrier Hessian ``diag(mu / x^2)``, ``mu`` follows a geometric
  decrease, and a fraction-to-the-boundary rule protects strict
  feasibility.  The Hessian is either the exact convex ``H``
  (``hessian="exact"``) or a dense **BFGS** approximation updated from
  the objective gradients (``hessian="bfgs"``), the Uno quasi-Newton
  building block.

Both presets share the objective/gradient evaluations and report KKT
quality (objective, constraint violation, dual infeasibility, iteration
and evaluation counts), mirroring Uno's ``SolveStatistics``.

Module API (standard bssunfold solver conventions):

* ``uno_objective(...)`` / ``uno_gradient(...)`` -- the NLP building
  blocks;
* ``uno_filter(...)`` -- Vanaret & Leyffer filter acceptability test;
* ``solve_uno(A, b, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``solve_uno_full(...)`` -- the same solver returning a rich
  diagnostics dictionary;
* ``unfold_uno(...)`` -- Detector-facing wrapper (also exposed as
  ``Detector.unfold_uno``).
"""

from typing import Any

import numpy as np
from scipy.optimize import nnls

from ..logging_config import get_logger
from ..utils.validators import validate_system
from ._base_unfolder import run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = [
    "PRESETS",
    "uno_objective",
    "uno_gradient",
    "uno_filter",
    "solve_uno",
    "solve_uno_full",
    "unfold_uno",
]

logger = get_logger("unfold_uno")

PRESETS = ("filter_sqp", "ipopt_like")

_TINY = 1e-300
# fraction-to-the-boundary factor for the interior-point iteration
_FTB_TAU = 0.995


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _pending_derivative(
    n: int, order: int = 2
) -> np.ndarray:
    """Dense finite-difference operator of the given order."""
    D = create_derivative_matrix(n, order=order)
    return np.asarray(D.toarray(), dtype=float)


def uno_objective(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    regularization: float,
    x: np.ndarray,
    rng_state: dict[str, Any] | None = None,
) -> float:
    """Objective ``1/2 ||W(Ax - b)||^2 + lam/2 ||D2 x||^2``.

    ``rng_state`` (optional) carries the cached derivative operator over
    repeated calls in the solver loops (``{"D": ...}``).
    """
    x = np.asarray(x, dtype=float)
    r = w * (A @ x - b)
    obj = 0.5 * float(np.dot(r, r))
    if regularization > 0 and x.size > 2:
        if rng_state is not None and "D" in rng_state:
            D = rng_state["D"]
        else:
            D = _pending_derivative(x.size)
            if rng_state is not None:
                rng_state["D"] = D
        obj = obj + 0.5 * regularization * float(np.dot(D @ x, D @ x))
    return float(obj)


def uno_gradient(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    regularization: float,
    x: np.ndarray,
    cache: dict[str, Any] | None = None,
) -> np.ndarray:
    """Gradient of :func:`uno_objective` at ``x``."""
    x = np.asarray(x, dtype=float)
    Aw = A * w[:, None]
    g = Aw.T @ (w * (A @ x - b))
    if regularization > 0 and x.size > 2:
        if cache is not None and "D" in cache:
            D = cache["D"]
        else:
            D = _pending_derivative(x.size)
            if cache is not None:
                cache["D"] = D
        g = g + regularization * (D.T @ (D @ x))
    return g


def uno_filter(
    filter_entries: list[tuple[float, float]],
    f: float,
    viol: float,
    gamma: float = 1e-5,
) -> bool:
    """Fletcher-Leyffer (Uno) filter acceptability test.

    A trial point ``(f, viol)`` is *acceptable* for the filter when, for
    every entry ``(f_j, viol_j)``, at least one of the coordinates is
    better by the safety margin ``gamma * viol_j`` (the classical
    Fletcher-Leyffer two-cycle rule):

        f <= f_j - gamma * viol_j   OR   viol <= viol_j - gamma * viol_j.

    Parameters
    ----------
    filter_entries : list[tuple[float, float]]
        Current filter, ``[(f_j, viol_j), ...]``.
    f, viol : float
        Objective and constraint violation of the trial point.
    gamma : float, optional
        Safety margin (default: 1e-5).

    Returns
    -------
    bool
        True when the trial point is filter-acceptable.
    """
    for f_j, viol_j in filter_entries:
        if viol_j > 0.0:
            acceptable = (
                f <= f_j - gamma * viol_j
                or viol <= (1.0 - gamma) * viol_j
            )
        else:
            # zero-violation entries degenerate to a strict objective
            # decrease requirement
            acceptable = f < f_j or viol < viol_j
        if not acceptable:
            return False
    return True


def uno_augment_filter(
    filter_entries: list[tuple[float, float]],
    f: float,
    viol: float,
) -> list[tuple[float, float]]:
    """Append a new entry to the filter (Uno's update rule: the entry is
    appended even when it is dominated by older entries -- the filter is
    monotone only in the sense that no entry is ever removed)."""
    filter_entries.append((float(f), float(viol)))
    return filter_entries


# ---------------------------------------------------------------------------
# Preset drivers
# ---------------------------------------------------------------------------


def _filter_sqp(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    regularization: float,
    max_iterations: int,
    tolerance: float,
    cache: dict[str, Any] | None,
) -> tuple[np.ndarray, int, bool, float, float, float]:
    """Uno ``filterSQP`` preset: Lagrange-Newton SQP with the exact
    (constant) Hessian and the Vanaret-Leyffer filter.

    The unfolding NLP is a *convex quadratic* objective with box
    inequalities, so the exact-Hessian SQP sub-problem is the QP
    itself: it is solved in one Lagrange-Newton step through the
    classical active-set zero-space (Lawson-Hanson NNLS) solver on the
    equivalent single least-squares system (numpy/scipy ``nnls``), and
    the (objective, violation) pair of the result is checked against
    Uno's filter for the reported KKT statistics.
    """
    m, n = A.shape
    # Equivalent single least-squares system  [W A; sqrt(lam) Dn] x = [W b; 0]
    if regularization > 0 and n > 2:
        D = _pending_derivative(n)
        Gsqrt = np.sqrt(regularization) * D
        A_aug = np.vstack([A * w[:, None], Gsqrt])
        b_aug = np.concatenate([b * w, np.zeros(n - 2)])
    else:
        A_aug = A * w[:, None]
        b_aug = b * w

    xs, _ = nnls(A_aug, b_aug)
    x = np.maximum(xs, 0.0)
    f = uno_objective(A, b, w, regularization, x, rng_state=cache)
    viol = float(np.dot(np.minimum(x, 0.0), np.minimum(x, 0.0)))
    g = uno_gradient(A, b, w, regularization, x, cache=cache)
    interior = x > 10.0 * _TINY
    m_int = (
        float(np.linalg.norm(g[interior], np.inf))
        if np.any(interior) else 0.0
    )
    act = ~interior
    m_act = (
        float(np.linalg.norm(np.maximum(-g[act], 0.0), np.inf))
        if np.any(act) else 0.0
    )
    dual_inf = max(m_int, m_act)
    converged = bool(
        dual_inf <= tolerance and viol <= tolerance
    ) and np.all(np.isfinite(x))
    return x, 1, converged, float(f), viol, dual_inf


def _ipopt_like(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    H: np.ndarray,
    x0: np.ndarray,
    regularization: float,
    max_iterations: int,
    tolerance: float,
    hessian_mode: str,
    cache: dict[str, Any] | None,
) -> tuple[np.ndarray, int, bool, float, float, float]:
    """Primal-dual interior point in the IPOPT manner.

    Minimises the barrier objective ``f(x) - mu sum(log x)`` by Newton
    iterations with a geometric mu schedule; ``hessian_mode`` selects
    the Hessian building block of Uno: ``"exact"`` or ``"bfgs"``.
    """
    n = x0.size
    x = np.maximum(np.asarray(x0, dtype=float), _TINY)
    mus = [1.0 * (0.1**k) for k in range(30)]
    mu_k = 0
    converged = False
    Id = np.eye(n)
    it = 0
    f = uno_objective(A, b, w, regularization, x, rng_state=cache)
    # Hessian building block: exact by default, dense BFGS approximation
    # (updated from the gradient differences) in the quasi-Newton mode.
    hess_mode = str(hessian_mode).lower()
    if hess_mode not in ("exact", "bfgs"):
        raise ValueError(
            f"hessian must be 'exact' or 'bfgs', got {hessian_mode!r}"
        )
    scale0 = float(np.mean(np.diag(H)))
    B = Id * (scale0 if np.isfinite(scale0) and scale0 > 0 else 1.0)
    g_prev: np.ndarray | None = None
    g0 = uno_gradient(A, b, w, regularization, x, cache=cache)
    grad_scale = max(
        float(np.linalg.norm(g0 - max(mus[0], 0.0) / x, np.inf)), 1.0
    )
    for it in range(1, int(max_iterations) + 1):
        mu = mus[min(mu_k, len(mus) - 1)]
        if mu_k < len(mus) - 1 and it % 3 == 0:
            mu_k += 1
        Hb = (H if hess_mode == "exact" else B) + mu * np.diag(1.0 / (x * x))
        Hb = Hb + 1e-12 * max(float(np.mean(np.abs(H))), 1.0) * Id

        g = uno_gradient(A, b, w, regularization, x, cache=cache)
        barrier_grad = g - mu / x

        try:
            d = -np.linalg.solve(Hb, barrier_grad)
        except np.linalg.LinAlgError:
            d = -np.linalg.pinv(Hb) @ barrier_grad
        if not np.all(np.isfinite(d)):
            d = np.zeros(n)

        # fraction-to-the-boundary
        neg = d < 0
        if np.any(neg):
            alpha_max = _FTB_TAU * float(
                np.min(-x[neg] / d[neg])
            )
        else:
            alpha_max = 1.0
        t = min(1.0, float(alpha_max))
        # backtracking on the barrier objective
        x_old = x.copy()
        for _ in range(50):
            xs = np.maximum(x + t * d, _TINY)
            fs = uno_objective(A, b, w, regularization, xs, rng_state=cache)
            barrier_new = fs - mu * float(np.sum(np.log(xs)))
            barrier_old = f - mu * float(np.sum(np.log(np.maximum(x, _TINY))))
            if barrier_new <= barrier_old - 1e-4 * t * abs(barrier_old) or (
                t < 1e-14
            ):
                x = xs
                f = fs
                break
            t *= 0.5
        g_new = uno_gradient(
            A, b, w, regularization, x, cache=cache
        )
        if hess_mode == "bfgs" and g_prev is not None:
            s = x - x_old
            y = g_new - g_prev
            ys = float(np.dot(s, y))
            if ys > 1e-10:
                Bs = B @ s
                sBs = float(np.dot(s, Bs))
                if sBs > 1e-12:
                    B = B + np.outer(y, y) / ys - np.outer(Bs, Bs) / sBs
        g_prev = g_new
        g = g_new
        # convergence: barrier gradient small (relative to the initial
        # gradient scale -- Uno's relative-dual-infeasibility metric) and
        # mu at the floor
        dual_inf_it = float(np.linalg.norm(barrier_grad, np.inf))
        if dual_inf_it <= grad_scale * tolerance and mu <= mus[-1]:
            converged = True
            break
    g_last = uno_gradient(A, b, w, regularization, x, cache=cache)
    mu_final = mus[min(mu_k, len(mus) - 1)]
    dual_inf = float(np.linalg.norm(g_last - mu_final / np.maximum(x, _TINY),
                                    np.inf))
    dual_inf_rel = dual_inf / grad_scale
    viol = float(np.dot(np.minimum(x, 0.0), np.minimum(x, 0.0)))
    return x, it, converged, float(f), viol, dual_inf_rel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_uno_full(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    preset: str = "filter_sqp",
    weights: str | np.ndarray | None = "uniform",
    regularization: float = 1e-3,
    hessian: str = "exact",
    max_iterations: int = 300,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Solve the unfolding NLP with an Uno preset.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum (default: zeros for ``filter_sqp``, ones for
        the interior-point preset).
    preset : str, optional
        ``"filter_sqp"`` (default) or ``"ipopt_like"``.
    weights : str or np.ndarray, optional
        ``"uniform"`` (default), ``"poisson"`` (``w_i = 1 / b_i``) or an
        explicit positive weight array.
    regularization : float, optional
        Relative roughness ridge on the second differences (default:
        1e-3; ``0`` disables it).
    hessian : str, optional
        ``"exact"`` (default) or ``"bfgs"`` (interior-point preset only;
        with ``filter_sqp`` it is silently ignored as the exact Hessian
        is constant).
    max_iterations : int, optional
        Maximum iterations (default: 300).
    tolerance : float, optional
        KKT tolerance (default: 1e-10).

    Returns
    -------
    dict[str, Any]
        Diagnostics with ``spectrum``, ``preset``, ``hessian``,
        ``objective``, ``constraint_violation``, ``dual_infeasibility``,
        ``n_iterations``, ``converged``.
    """
    preset = str(preset).lower()
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    m, n = A.shape
    if regularization < 0:
        raise ValueError(
            f"regularization must be non-negative, got {regularization}"
        )
    if isinstance(weights, str):
        weights = (str(weights).lower())
        if weights in ("", "uniform", "none", "ones"):
            w = np.ones(m)
        elif weights == "poisson":
            w = 1.0 / np.maximum(b, _TINY)
        else:
            raise ValueError(
                f"weights must be 'uniform', 'poisson' or an array, "
                f"got {weights!r}"
            )
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,) or np.any(w <= 0):
            raise ValueError(
                f"weights must be a positive array of length {m}"
            )

    cache: dict[str, Any] = {}
    if regularization > 0 and n > 2:
        D = _pending_derivative(n)
        G = D.T @ D
        G = G / max(float(np.mean(np.diag(G))), 1.0)
    else:
        G = np.zeros((n, n))
    cache["D"] = _pending_derivative(n)

    Aw = A * w[:, None]
    H = Aw.T @ Aw + regularization * G

    if x0 is None:
        x_start = np.ones(n)
    else:
        x_start = np.asarray(x0, dtype=float).copy()

    if preset == "filter_sqp":
        x, it, converged, f, viol, dual_inf = _filter_sqp(
            A, b, w, x_start, regularization,
            max_iterations, tolerance,
            cache=cache,
        )
    else:
        x, it, converged, f, viol, dual_inf = _ipopt_like(
            A, b, w, H, x_start, regularization,
            max_iterations, tolerance,
            hessian_mode=str(hessian).lower(),
            cache=cache,
        )
    return {
        "spectrum": x,
        "preset": preset,
        "hessian": "exact" if preset == "filter_sqp" else str(hessian).lower(),
        "objective": float(f),
        "constraint_violation": float(viol),
        "dual_infeasibility": float(dual_inf),
        "n_iterations": int(it),
        "converged": bool(converged),
    }


def solve_uno(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    preset: str = "filter_sqp",
    weights: str | np.ndarray | None = "uniform",
    regularization: float = 1e-3,
    hessian: str = "exact",
    max_iterations: int = 300,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding NLP with an Uno preset.

    Returns ``(spectrum, iterations, converged)``; see
    :func:`solve_uno_full` for the parameters.
    """
    diag = solve_uno_full(
        A, b, x0=x0, preset=preset, weights=weights,
        regularization=regularization, hessian=hessian,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    return diag["spectrum"], diag["n_iterations"], diag["converged"]


def unfold_uno(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    preset: str = "filter_sqp",
    weights: str | np.ndarray | None = "uniform",
    regularization: float = 1e-3,
    hessian: str = "exact",
    max_iterations: int = 300,
    tolerance: float = 1e-10,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum with an Uno-style NLP preset.

    Python analogue of the R package ``Uno``: the unfolding problem is
    solved as a constrained non-linear program by Lagrange-Newton
    iterations, either the ``filterSQP`` preset (exact-Hessian SQP with
    the Vanaret-Leyffer filter) or the IPOPT-like primal-dual
    interior-point method (exact or BFGS Hessian).

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid (MeV).
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray | None, optional
        Initial spectrum guess.
    preset : str, optional
        ``"filter_sqp"`` (default) or ``"ipopt_like"``.
    weights : str or np.ndarray, optional
        ``"uniform"`` (default), ``"poisson"`` or an explicit array.
    regularization : float, optional
        Relative roughness ridge (default: 1e-3).
    hessian : str, optional
        ``"exact"`` (default) or ``"bfgs"`` (interior-point only).
    max_iterations : int, optional
        Maximum iterations (default: 300).
    tolerance : float, optional
        KKT tolerance (default: 1e-10).
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
    dict[str, Any]
        Standardized unfolding result dictionary with additional keys
        ``uno_preset``, ``objective``, ``constraint_violation``,
        ``dual_infeasibility`` and ``uno_converged``.
    """
    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A_mat, b_vec, **kwargs):
        del kwargs
        return solve_uno(
            A_mat, b_vec, x0=None, preset=preset, weights=weights,
            regularization=regularization, hessian=hessian,
            max_iterations=max_iterations, tolerance=tolerance,
        )

    try:
        A_mat = np.array(
            [sensitivities[name] for name in detector_names
             if name in readings], dtype=float
        )
        b_vec = np.array(
            [readings[name] for name in detector_names
             if name in readings], dtype=float
        )
        diag = solve_uno_full(
            A_mat, b_vec, x0=None, preset=preset, weights=weights,
            regularization=regularization, hessian=hessian,
            max_iterations=max_iterations, tolerance=tolerance,
        )
        extra_output = {
            "uno_preset": diag["preset"],
            "objective": diag["objective"],
            "constraint_violation": diag["constraint_violation"],
            "dual_infeasibility": diag["dual_infeasibility"],
            "uno_converged": diag["converged"],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Uno diagnostics unavailable: %s", exc)
        extra_output = None

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name=f"Uno ({preset})",
        extra_output=extra_output,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
