"""Utility modules for bssunfold package.

This subpackage provides utility functions for data validation,
conversion, interpolation, and plotting.
"""

from .validators import validate_readings, validate_energy_grid
from .converters import convert_to_dataframe, convert_to_dict
from .interpolation import interpolate_spectrum, discretize_spectra
from .plotting import plot_spectrum, plot_response_functions

__all__ = [
    # validators
    "validate_readings",
    "validate_energy_grid",
    # converters
    "convert_to_dataframe",
    "convert_to_dict",
    # interpolation
    "interpolate_spectrum",
    "discretize_spectra",
    # plotting
    "plot_spectrum",
    "plot_response_functions",
]
