# bssunfold/__init__.py
"""bssunfold - Bonner Sphere Spectrum Unfolding Package.

This package provides tools for neutron spectrum unfolding using various
algorithms including convex optimization, Landweber iteration, and MLEM.

Backward Compatibility Note:
    This package has been refactored to improve modularity and maintainability.
    All original public API functions and classes are still available at the
    same import paths to ensure 100% backward compatibility.
"""

import importlib.metadata
import warnings

# Import main classes and constants for backward compatibility
from .core.detector import Detector
from .constants import (
    ICRP116_COEFF_EFFECTIVE_DOSE,
    RF_GSF,
    RF_PTB,
    RF_LANL,
)

# Import utility modules for convenience
from . import utils
from . import core
from .platform_check import (
    is_windows,
    JAX_AVAILABLE,
    PROXSUITE_AVAILABLE,
    get_available_solvers,
    get_recommended_solver,
)
from .logging_config import setup_logging, get_logger

__all__ = [
    # Main class
    "Detector",
    # Constants
    "ICRP116_COEFF_EFFECTIVE_DOSE",
    "RF_GSF",
    "RF_PTB",
    "RF_LANL",
    # Platform info
    "is_windows",
    "JAX_AVAILABLE",
    "PROXSUITE_AVAILABLE",
    "get_available_solvers",
    "get_recommended_solver",
    # Logging
    "setup_logging",
    "get_logger",
    # Submodules
    "utils",
    "core",
]

try:
    __version__ = importlib.metadata.version("bssunfold")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

# Show platform warning on import if running on Windows
if is_windows:
    warnings.warn(
        "Running on Windows. Some solvers (proxqp via proxsuite) "
        "are not available. Using fallback solvers (ECOS, OSQP, etc.).",
        UserWarning,
        stacklevel=2
    )
