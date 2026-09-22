"""Shared cvxpy backend for the commercial-license QP solvers.

The public methods :func:`bssunfold.core.unfold_commercial.solve_gurobi` and
friends (Gurobi, MOSEK, CPLEX, COPT, XPRESS — all **license required**)
delegate the actual optimization here.  The problem is written as a plain
convex quadratic program in cvxpy,

    min  0.5 * x' G x + c' x        s.t.  x >= lb,  x <= ub

with ``G = A'A`` plus the (scaled) Tikhonov/smoothness penalty, so every
commercial QP-capable backend solves the same formulation.  The requested
solver is used *exclusively*: unlike :mod:`unfold_cvxpy` there is no
fallback to the open-source conic solvers (ECOS/SCS/CLARABEL), because the
point of these methods is to exercise the licensed engine.  If the solver
or its license is unavailable the solve returns ``None`` and the wrapper
warns and yields a zero spectrum.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from ._matrix_utils import build_smoothness_penalty

__all__ = [
    "COMMERCIAL_SOLVER_ALIASES",
    "COMMERCIAL_SOLVER_PY_MODULES",
    "commercial_solver_info",
    "is_commercial_solver_available",
    "solve_commercial_qp",
]

logger = logging.getLogger(__name__)

#: Alias -> cvxpy solver name for the commercial QP backends.  Values are
#: resolved lazily via ``getattr(cvxpy, name)`` because the constants may
#: be absent on older cvxpy versions.
COMMERCIAL_SOLVER_ALIASES: dict[str, str] = {
    "gurobi": "GUROBI",
    "mosek": "MOSEK",
    "cplex": "CPLEX",
    "copt": "COPT",
    "xpress": "XPRESS",
}

#: Alias -> pip import name of the third-party package providing the engine
#: (plus its license).  The cvxpy interface itself ships with cvxpy.
COMMERCIAL_SOLVER_PY_MODULES: dict[str, str] = {
    "gurobi": "gurobipy",
    "mosek": "mosek",
    "cplex": "cplex",
    "copt": "coptpy",
    "xpress": "xpress",
}


def _cvxpy_solver_constant(alias: str):
    """Return the cvxpy solver-name constant for *alias* (None if unknown)."""
    import cvxpy as cp

    name = COMMERCIAL_SOLVER_ALIASES.get(alias)
    if name is None:
        return None
    return getattr(cp, name, None)


def commercial_solver_info(alias: str) -> dict[str, Any]:
    """Return license/backend metadata for a commercial solver alias.

    Parameters
    ----------
    alias : str
        One of ``"gurobi"``, ``"mosek"``, ``"cplex"``, ``"copt"``,
        ``"xpress"``.

    Returns
    -------
    Dict[str, Any]
        Metadata dict with keys ``alias``, ``cvxpy_solver`` (None if the
        installed cvxpy has no interface for it), ``pip_package``,
        ``license_required`` (always True) and ``available`` (whether the
        engine is importable and cvxpy reports it as installed).

    Raises
    ------
    ValueError
        If *alias* is not a supported commercial solver.
    """
    if alias not in COMMERCIAL_SOLVER_ALIASES:
        supported = ", ".join(sorted(COMMERCIAL_SOLVER_ALIASES))
        raise ValueError(
            f"Unknown commercial solver '{alias}'. Supported: {supported}"
        )
    return {
        "alias": alias,
        "cvxpy_solver": _cvxpy_solver_constant(alias),
        "pip_package": COMMERCIAL_SOLVER_PY_MODULES[alias],
        "license_required": True,
        "available": is_commercial_solver_available(alias),
    }


def is_commercial_solver_available(alias: str) -> bool:
    """Check whether a commercial solver engine is usable through cvxpy.

    True only if the third-party package imports *and* cvxpy lists the
    solver among its installed solvers.  No license is ever checked or
    requested by bssunfold itself — the engine's own license machinery
    decides whether a solve may run.
    """
    try:
        import cvxpy as cp

        solver = _cvxpy_solver_constant(alias)
        if solver is None:
            return False
        return solver in cp.installed_solvers()
    except ImportError:
        return False


def _require_imports():
    """Import cvxpy lazily, raising a license-aware ImportError."""
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for the commercial solver methods. "
            "Install with: pip install cvxpy"
        ) from e
    return cp


def _commercial_kwargs(cp, solver, timeout, random_state, x0) -> dict:
    """Assemble solver kwargs, degrading gracefully when unsupported.

    The timeout is mapped to each engine's native cvxpy option name
    (``TimeLimit`` for Gurobi, ``cplex_params``/``timelimit`` for CPLEX,
    ``mosek_params``/``MSK_DPAR_OPTIMIZER_MAX_TIME`` for MOSEK); engines
    without a portable timeout option (COPT, XPRESS) simply omit it. A
    bare ``timeout=...`` kwarg raises "Unknown parameter" on several of
    the cvxpy interfaces, which previously surfaced masked as a license
    error.
    """
    kwargs: dict[str, Any] = {"solver": solver, "verbose": False}
    if timeout is not None:
        limit = max(float(timeout), 1e-3)
        if solver == "GUROBI":
            kwargs["TimeLimit"] = limit
        elif solver == "CPLEX":
            kwargs["cplex_params"] = {"timelimit": limit}
        elif solver == "MOSEK":
            kwargs["mosek_params"] = {"MSK_DPAR_OPTIMIZER_MAX_TIME": limit}
        # COPT / XPRESS: no portable timeout option — omit it.
    if random_state is not None:
        kwargs["seed"] = int(random_state)
    if x0 is not None:
        # Supported by Gurobi/CPLEX/MOSEK/XPRESS interfaces (warm start);
        # ignored via the TypeError retry below when not.
        kwargs["warm_start"] = True
    return kwargs


def solve_commercial_qp(
    A: np.ndarray,
    b: np.ndarray,
    alias: str,
    x0: np.ndarray | None = None,
    alpha: float = 1e-4,
    norm: int = 2,
    timeout: float = 10.0,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    nonneg: bool = True,
    random_state: int | None = None,
    ub: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve the unfolding QP with one commercial (license-required) solver.

    Minimizes ``0.5 * ||A x - b||^2 + penalty(x)`` subject to ``x >= 0``
    (when ``nonneg``) and optional per-bin upper bounds, using exclusively
    the cvxpy interface of the requested commercial engine.

    Parameters
    ----------
    A : np.ndarray
        Response matrix of size (m, n).
    b : np.ndarray
        Measurement vector of size (m,).
    alias : str
        Commercial solver alias: 'gurobi', 'mosek', 'cplex', 'copt', 'xpress'.
    x0 : np.ndarray, optional
        Initial values, forwarded as a warm start when the backend supports it.
    alpha : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    timeout : float, optional
        Time limit in seconds, default: 10.0.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2), default: 0.
    smoothness_weight : float, optional
        Weight for the smoothness term, default: 1.0.
    nonneg : bool, optional
        Constrain the solution to ``x >= 0``, default: True.
    random_state : int, optional
        Random seed for the solver, for reproducibility.
    ub : np.ndarray, optional
        Per-bin upper bounds; non-finite entries mean "unbounded".

    Returns
    -------
    Optional[np.ndarray]
        Unfolded spectrum (n,), or None if the solver/license is unavailable
        or the solve failed.
    """
    cp = _require_imports()

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError(f"{alias} solver: received ill-formed input.")
    if norm not in (1, 2):
        raise ValueError(f"Unsupported norm type: {norm}")
    if norm == 1 and not nonneg:
        raise ValueError(
            f"{alias} solver: L1 penalty equals alpha * sum(x) only under "
            "the non-negativity constraint (pass nonneg=True or use norm=2)."
        )

    info = commercial_solver_info(alias)
    if not info["available"]:
        warnings.warn(
            f"Commercial solver '{alias}' is not available: the "
            f"'{info['pip_package']}' package (license required) must be "
            "installed and licensed for cvxpy to solve with it. "
            "No open-source fallback is attempted."
        )
        return None
    solver = info["cvxpy_solver"]

    n = A.shape[1]
    # Canonical QP convention of the library (solve_qpsolvers, norm=2):
    #   min 0.5 * x' P x + q' x,  P = A'A + penalty,  q = -A'b (+ alpha for L1)
    # The Tikhonov penalty is the identity term alpha*I for
    # smoothness_order == 0, else the derivative term alpha*w*L'L (the two
    # are alternatives, matching solve_qpsolvers and solve_docplex).
    P = A.T @ A
    q = -A.T @ b
    if norm == 2:
        if smoothness_order in (1, 2):
            P = P + build_smoothness_penalty(
                n, alpha, smoothness_order, smoothness_weight
            ).toarray()
        else:
            P = P + alpha * np.eye(n)
    else:
        q = q + alpha * np.ones(n)
        if smoothness_order in (1, 2):
            P = P + build_smoothness_penalty(
                n, alpha, smoothness_order, smoothness_weight
            ).toarray()

    x = cp.Variable(n, nonneg=nonneg)
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
    constraints = []
    if ub is not None:
        finite = np.isfinite(ub)
        if finite.any():
            constraints.append(x[finite] <= ub[finite])
    problem = cp.Problem(objective, constraints)
    if x0 is not None:
        x.value = np.asarray(x0, dtype=float)

    kwargs = _commercial_kwargs(cp, solver, timeout, random_state, x0)
    for attempt in (kwargs, {"solver": solver, "verbose": False}):
        try:
            problem.solve(**attempt)
            break
        except (cp.error.SolverError, TypeError, ValueError) as exc:
            if attempt is kwargs:
                logger.debug(
                    "%s: retrying without extended kwargs (%s)", alias, exc
                )
                continue
            warnings.warn(
                f"Commercial solver '{alias}' (license required) failed to "
                f"solve: {exc}. Returning None."
            )
            return None
        except Exception as exc:  # license errors are engine-specific
            warnings.warn(
                f"Commercial solver '{alias}' (license required) raised "
                f"{type(exc).__name__}: {exc}. Check that a valid license is "
                "configured. Returning None."
            )
            return None

    if problem.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        warnings.warn(
            f"Commercial solver '{alias}' did not find a solution "
            f"(status={problem.status}). Returning None."
        )
        return None

    result = np.asarray(x.value, dtype=float)
    if ub is not None:
        result[ub == 0.0] = 0.0
    return result
