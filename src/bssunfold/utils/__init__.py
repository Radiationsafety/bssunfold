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
    round_to_sigfig,
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
    plot_comparison,
)
from .comparison import (
    compare_spectra,
    compare_multiple,
    kl_divergence,
    cross_entropy,
    entropy,
    entropy_difference_percent,
    wasserstein_dist,
    energy_dist,
    kolmogorov_smirnov_stat,
    pearson_r,
    spearman_r,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mape,
    r2_score,
    max_error,
    median_absolute_error,
    cosine_similarity,
    mmd_rbf,
    chi_squared,
    g_test,
    freeman_tukey,
    cressie_read,
    anderson_darling,
    standardized_mean_difference,
    wilcoxon_test,
    mannwhitneyu_test,
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
    "round_to_sigfig",
    # interpolation
    "interpolate_spectrum",
    "discretize_spectra",
    "resample_to_log_grid",
    # plotting
    "plot_spectrum",
    "plot_response_functions",
    "plot_with_uncertainty",
    "plot_residuals",
    "plot_comparison",
    # comparison
    "compare_spectra",
    "compare_multiple",
    "kl_divergence",
    "cross_entropy",
    "entropy",
    "entropy_difference_percent",
    "wasserstein_dist",
    "energy_dist",
    "kolmogorov_smirnov_stat",
    "pearson_r",
    "spearman_r",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "mape",
    "r2_score",
    "max_error",
    "median_absolute_error",
    "cosine_similarity",
    "mmd_rbf",
    "chi_squared",
    "g_test",
    "freeman_tukey",
    "cressie_read",
    "anderson_darling",
    "standardized_mean_difference",
    "wilcoxon_test",
    "mannwhitneyu_test",
]
