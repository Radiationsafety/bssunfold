"""Shared solver-backend resolution helpers for the parametric unfolding modules.

These were previously duplicated across ``unfold_parametric.py`` and
``unfold_parametric2.py``.
"""


def _parse_solver_backend(solver_backend):
    """Parse a solver_backend string into (library, backend).

    Examples:
        "auto"            -> ("auto", "default")
        "cvxpy"           -> ("cvxpy", "default")
        "cvxpy:ECOS"      -> ("cvxpy", "ECOS")
        "qpsolvers"       -> ("qpsolvers", "auto")
        "qpsolvers:osqp"  -> ("qpsolvers", "osqp")
    """
    if solver_backend == "auto":
        return "auto", "default"

    parts = solver_backend.split(":", 1)
    library = parts[0]
    backend = parts[1] if len(parts) > 1 else "default"
    return library, backend


def _resolve_cvxpy_solvers(backend):
    """Return list of cvxpy solvers to try."""
    try:
        import cvxpy as cp

        installed = cp.installed_solvers()
    except ImportError:
        installed = []

    if backend == "default":
        candidates = [s for s in ["ECOS", "SCS", "CLARABEL"] if s in installed]
        return candidates or ["ECOS"]
    fallbacks = [s for s in ["ECOS", "SCS", "CLARABEL"] if s != backend]
    return [backend] + fallbacks


def _resolve_qpsolver_name(backend):
    """Return the qpsolvers backend name to use."""
    if backend != "default":
        return backend
    try:
        from qpsolvers import available_solvers

        if "osqp" in available_solvers:
            return "osqp"
        if "ecos" in available_solvers:
            return "ecos"
    except ImportError:
        pass
    return "osqp"
