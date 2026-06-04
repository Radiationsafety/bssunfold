"""Platform compatibility checks and conditional imports.

This module handles platform-specific dependencies, particularly for jaxlib
and proxsuite which are not available on Windows.
"""

import sys
import warnings
from typing import Dict, Any

__all__ = [
    "is_windows",
    "is_unix",
    "check_jax_availability",
    "check_proxsuite_availability",
    "check_qpsolvers_extra_availability",
    "get_available_solvers",
    "get_recommended_solver",
    "JAX_AVAILABLE",
    "PROXSUITE_AVAILABLE",
    "QPSOLVERS_EXTRA_AVAILABLE",
]

# Platform detection
is_windows = sys.platform == "win32"
is_unix = sys.platform in ("linux", "darwin")

# Dependency availability flags
JAX_AVAILABLE: bool = False
PROXSUITE_AVAILABLE: bool = False
QPSOLVERS_EXTRA_AVAILABLE: bool = False


def check_jax_availability() -> bool:
    """Check if jax and jaxlib are available.

    Returns
    -------
    bool
        True if jax and jaxlib can be imported, False otherwise.
    """
    global JAX_AVAILABLE
    try:
        import jax  # noqa: F401
        import jaxlib  # noqa: F401
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
        import proxsuite  # noqa: F401
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


def get_available_solvers() -> Dict[str, Any]:
    """Get dictionary of available solvers with their status.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping solver names to their availability status.
    """
    check_jax_availability()
    check_proxsuite_availability()
    check_qpsolvers_extra_availability()

    # Base solvers always available via cvxpy
    solvers = {
        "ecos": True,
        "scs": True,
        "clarabel": True,
    }

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
    else:
        return "osqp"


# Initialize on module load
check_jax_availability()
check_proxsuite_availability()
check_qpsolvers_extra_availability()
