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

# Import main classes and constants for backward compatibility
from .core.detector import Detector
from .constants import (
    ICRP116_COEFF_EFFECTIVE_DOSE,
    ICRP74_COEFF_EFFECTIVE_DOSE,
    NRB99_2009_COEFF_EFFECTIVE_DOSE,
    ICRP74_COEFF_OPERATIONAL_QUANTITIES,
    RF_GSF,
    RF_PTB,
    RF_LANL,
    RF_JINR,
    RF_FERMILAB,
    RF_EURADOS,
    RF_IHEP,
)
from .core.dose_calculation import (
    get_coefficients,
    interpolate_coefficients,
    DOSE_COEFFICIENTS_REGISTRY,
)

# Import utility modules for convenience
from . import utils
from . import core
from .utils.comparison import compare_spectra
from .platform_check import (
    is_windows,
    is_unix,
    JAX_AVAILABLE,
    PROXSUITE_AVAILABLE,
    QPSOLVERS_EXTRA_AVAILABLE,
    get_available_solvers,
    get_recommended_solver,
)
from .logging_config import setup_logging, get_logger

__all__ = [
    # Main class
    "Detector",
    # Constants
    "ICRP116_COEFF_EFFECTIVE_DOSE",
    "ICRP74_COEFF_EFFECTIVE_DOSE",
    "NRB99_2009_COEFF_EFFECTIVE_DOSE",
    "ICRP74_COEFF_OPERATIONAL_QUANTITIES",
    "RF_GSF",
    "RF_PTB",
    "RF_LANL",
    "RF_JINR",
    "RF_FERMILAB",
    "RF_EURADOS",
    "RF_IHEP",
    # Dose coefficient utilities
    "get_coefficients",
    "interpolate_coefficients",
    "DOSE_COEFFICIENTS_REGISTRY",
    # Platform info
    "is_windows",
    "is_unix",
    "JAX_AVAILABLE",
    "PROXSUITE_AVAILABLE",
    "QPSOLVERS_EXTRA_AVAILABLE",
    "get_available_solvers",
    "get_recommended_solver",
    # Logging
    "setup_logging",
    "get_logger",
    # Comparison
    "compare_spectra",
    # Submodules
    "utils",
    "core",
]

try:
    __version__ = importlib.metadata.version("bssunfold")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"
