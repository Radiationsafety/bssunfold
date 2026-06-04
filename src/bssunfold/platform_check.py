"""Platform compatibility checks and conditional imports.

This module handles platform-specific dependencies, particularly for jaxlib
and proxsuite which are not available on Windows.
"""

import sys
import warnings
from typing import Optional, Dict, Any

__all__ = [
    "is_windows",
    "is_unix",
    "check_jax_availability",
    "check_proxsuite_availability",
    "get_available_solvers",
    "JAX_AVAILABLE",
    "PROXSUITE_AVAILABLE",
    "PLATFORM_WARNING_SHOWN",
]

# Platform detection
is_windows = sys.platform == "win32"
is_unix = sys.platform in ("linux", "darwin")

# Dependency availability flags
JAX_AVAILABLE: bool = False
PROXSUITE_AVAILABLE: bool = False
PLATFORM_WARNING_SHOWN: bool = False


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


def _show_platform_warning() -> None:
    """Show platform compatibility warning if needed."""
    global PLATFORM_WARNING_SHOWN
    if not PLATFORM_WARNING_SHOWN and is_windows:
        warnings.warn(
            "Running on Windows. Some solvers (proxqp via proxsuite) "
            "are not available. Using fallback solvers (ECOS, OSQP, etc.).",
            UserWarning,
            stacklevel=2
        )
        PLATFORM_WARNING_SHOWN = True


def get_available_solvers() -> Dict[str, Any]:
    """Get dictionary of available solvers with their status.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping solver names to their availability status.
    """
    _show_platform_warning()
    check_jax_availability()
    check_proxsuite_availability()
    
    solvers = {
        "ecos": True,  # Always available via cvxpy
        "osqp": True,  # Available on all platforms
        "scs": True,   # Available via cvxpy
        "clarabel": True,  # Available via cvxpy
        "proxqp": PROXSUITE_AVAILABLE,
        "piqp": True,  # Pure Python implementation
        "qpalm": True,  # Available on all platforms
        "jaxqp": JAX_AVAILABLE,
    }
    
    return solvers


def get_recommended_solver() -> str:
    """Get recommended solver based on platform and availability.
    
    Returns
    -------
    str
        Name of recommended solver.
    """
    _show_platform_warning()
    check_proxsuite_availability()
    
    if PROXSUITE_AVAILABLE and not is_windows:
        return "proxqp"
    else:
        return "osqp"


# Initialize on module load
check_jax_availability()
check_proxsuite_availability()
