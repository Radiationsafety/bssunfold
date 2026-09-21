"""Platform compatibility checks and conditional imports.

This module handles platform-specific dependencies, particularly for jaxlib
and proxsuite which are not available on Windows.
"""

import sys
from typing import Any

__all__ = [
    "is_windows",
    "is_unix",
    "check_jax_availability",
    "check_proxsuite_availability",
    "check_qpsolvers_extra_availability",
    "check_scip_availability",
    "check_docplex_availability",
    "check_commercial_solvers_availability",
    "check_cuqi_availability",
    "get_available_solvers",
    "get_recommended_solver",
    "JAX_AVAILABLE",
    "PROXSUITE_AVAILABLE",
    "QPSOLVERS_EXTRA_AVAILABLE",
    "SCIP_AVAILABLE",
    "DOCPLEX_AVAILABLE",
]

# Platform detection
is_windows = sys.platform == "win32"
is_unix = sys.platform in ("linux", "darwin")

# Dependency availability flags
JAX_AVAILABLE: bool = False
PROXSUITE_AVAILABLE: bool = False
QPSOLVERS_EXTRA_AVAILABLE: bool = False
SCIP_AVAILABLE: bool = False
DOCPLEX_AVAILABLE: bool = False
CUQI_AVAILABLE: bool = False


def check_jax_availability() -> bool:
    """Check if jax and jaxlib are available.

    Returns
    -------
    bool
        True if jax and jaxlib can be imported, False otherwise.
    """
    global JAX_AVAILABLE
    try:
        import jax  # noqa: F401  # pylint: disable=unused-import
        import jaxlib  # noqa: F401  # pylint: disable=unused-import

        JAX_AVAILABLE = True
        return True
    except ImportError:
        JAX_AVAILABLE = False
        return False


def check_proxsuite_availability() -> bool:
    """Check if proxsuite is available.

    Returns
    -------
    bool
        True if proxsuite can be imported, False otherwise.
    """
    global PROXSUITE_AVAILABLE
    try:
        import proxsuite  # noqa: F401  # pylint: disable=unused-import

        PROXSUITE_AVAILABLE = True
        return True
    except ImportError:
        PROXSUITE_AVAILABLE = False
        return False


def check_qpsolvers_extra_availability() -> bool:
    """Check if extra qpsolvers (osqp, piqp, qpalm, etc.) are available.

    These solvers are installed via the ``solvers-core`` optional dependency
    group and are available on all platforms.

    Returns
    -------
    bool
        True if extra qpsolvers can be imported, False otherwise.
    """
    global QPSOLVERS_EXTRA_AVAILABLE
    try:
        from qpsolvers import available_solvers

        # Check that at least one extra solver beyond the base qpsolvers
        # is actually installed. Base qpsolvers always includes at least
        # one solver, but we want to confirm extras are present.
        extra_solvers = {"osqp", "piqp", "qpalm", "ecos", "scs", "clarabel"}
        if extra_solvers & set(available_solvers):
            QPSOLVERS_EXTRA_AVAILABLE = True
            return True
        QPSOLVERS_EXTRA_AVAILABLE = False
        return False
    except ImportError:
        QPSOLVERS_EXTRA_AVAILABLE = False
        return False


def check_scip_availability() -> bool:
    """Check if pyscipopt (SCIP Optimization Suite interface) is available.

    Returns
    -------
    bool
        True if pyscipopt can be imported, False otherwise.
    """
    global SCIP_AVAILABLE
    try:
        import pyscipopt  # noqa: F401  # pylint: disable=unused-import

        SCIP_AVAILABLE = True
        return True
    except ImportError:
        SCIP_AVAILABLE = False
        return False


def check_docplex_availability() -> bool:
    """Check if docplex and the CPLEX engine are available.

    Returns
    -------
    bool
        True if docplex and cplex can be imported, False otherwise.
    """
    global DOCPLEX_AVAILABLE
    try:
        import cplex  # noqa: F401  # pylint: disable=unused-import
        import docplex  # noqa: F401  # pylint: disable=unused-import

        DOCPLEX_AVAILABLE = True
        return True
    except ImportError:
        DOCPLEX_AVAILABLE = False
        return False


# Commercial engines (Gurobi/MOSEK/CPLEX/COPT/XPRESS), alias -> availability.
# All of them are **license required**: the engines are proprietary and are
# never distributed or licensed by bssunfold.  Populated by
# check_commercial_solvers_availability().
COMMERCIAL_SOLVERS_AVAILABLE: dict[str, bool] = {}


def check_commercial_solvers_availability() -> dict[str, bool]:
    """Check which license-required commercial solvers are usable.

    Uses the lazy cvxpy availability probe from
    :mod:`bssunfold.core._commercial_qp`; no import of this package's core
    modules happens unless cvxpy itself is installed.

    Returns
    -------
    Dict[str, bool]
        Mapping of solver alias ('gurobi', 'mosek', 'cplex', 'copt',
        'xpress') to availability.  Cached in COMMERCIAL_SOLVERS_AVAILABLE.
    """
    global COMMERCIAL_SOLVERS_AVAILABLE
    try:
        from bssunfold.core._commercial_qp import (
            COMMERCIAL_SOLVER_ALIASES,
            is_commercial_solver_available,
        )
    except ImportError:
        COMMERCIAL_SOLVERS_AVAILABLE = {}
        return COMMERCIAL_SOLVERS_AVAILABLE
    COMMERCIAL_SOLVERS_AVAILABLE = {
        alias: is_commercial_solver_available(alias)
        for alias in COMMERCIAL_SOLVER_ALIASES
    }
    return COMMERCIAL_SOLVERS_AVAILABLE


def get_available_solvers() -> dict[str, Any]:
    """Get dictionary of available solvers with their status.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping solver names to their availability status.
    """
    check_jax_availability()
    check_proxsuite_availability()
    check_qpsolvers_extra_availability()
    check_scip_availability()
    check_docplex_availability()

    # Base solvers always available via cvxpy
    solvers = {
        "ecos": True,
        "scs": True,
        "clarabel": True,
    }

    # Algebraic modeling / optimization backends
    solvers["scip"] = SCIP_AVAILABLE
    solvers["docplex"] = DOCPLEX_AVAILABLE
    # Commercial engines (license required): probed lazily here.
    for alias, available in check_commercial_solvers_availability().items():
        solvers[alias] = available

    # Extra qpsolvers (may require solvers-core optional dependency)
    solvers["osqp"] = QPSOLVERS_EXTRA_AVAILABLE
    solvers["piqp"] = QPSOLVERS_EXTRA_AVAILABLE
    solvers["qpalm"] = QPSOLVERS_EXTRA_AVAILABLE

    # Platform-specific solvers
    solvers["proxqp"] = PROXSUITE_AVAILABLE
    solvers["jaxqp"] = JAX_AVAILABLE

    return solvers


def get_recommended_solver() -> str:
    """Get recommended solver based on platform and availability.

    Returns
    -------
    str
        Name of recommended solver.
    """
    check_proxsuite_availability()

    if PROXSUITE_AVAILABLE and not is_windows:
        return "proxqp"
    return "osqp"



def check_cuqi_availability() -> bool:
    """Check if CUQIpy (DTU uncertainty quantification library) is available.

    CUQIpy powers the Bayesian MCMC unfolding methods in
    :mod:`bssunfold.core.unfold_cuqi` (PCN, CWMH, ULA, MALA, NUTS and
    hierarchical Gibbs samplers).

    Returns
    -------
    bool
        True if cuqi can be imported, False otherwise.
    """
    global CUQI_AVAILABLE
    try:
        import cuqi  # noqa: F401  # pylint: disable=unused-import

        CUQI_AVAILABLE = True
        return True
    except ImportError:
        CUQI_AVAILABLE = False
        return False
# Initialize on module load
check_jax_availability()
check_proxsuite_availability()
check_qpsolvers_extra_availability()
check_scip_availability()
check_docplex_availability()
check_cuqi_availability()
