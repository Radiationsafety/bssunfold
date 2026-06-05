"""Utility modules for bssunfold package.

This subpackage provides utility functions for data validation,
conversion, interpolation, and plotting.
"""

from .validators import (
    validate_readings,
    validate_energy_grid,
    validate_spectrum,
    validate_response_matrix,
)
from .converters import (
    convert_to_dataframe,
    convert_to_dict,
    convert_sensitivities_to_matrix,
    extract_detector_names,
)
from .interpolation import (
    interpolate_spectrum,
    discretize_spectra,
    resample_to_log_grid,
)
from .plotting import (
    plot_spectrum,
    plot_response_functions,
    plot_with_uncertainty,
    plot_residuals,
)

__all__ = [
    # validators
    "validate_readings",
    "validate_energy_grid",
    "validate_spectrum",
    "validate_response_matrix",
    # converters
    "convert_to_dataframe",
    "convert_to_dict",
    "convert_sensitivities_to_matrix",
    "extract_detector_names",
    # interpolation
    "interpolate_spectrum",
    "discretize_spectra",
    "resample_to_log_grid",
    # plotting
    "plot_spectrum",
    "plot_response_functions",
    "plot_with_uncertainty",
    "plot_residuals",
]
