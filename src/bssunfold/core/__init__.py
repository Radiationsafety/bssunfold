"""Core modules for bssunfold package.

This subpackage contains the main functionality for neutron spectrum
unfolding, including the Detector class and unfolding methods.
"""

from .detector import Detector
from .unfolding_methods import (
    solve_cvxpy,
    solve_landweber,
    solve_mlem,
    solve_qpsolvers,
    solve_doroshenko,
    solve_kaczmarz,
    solve_lmfit,
)
from .regularization import (
    select_regularization_parameter,
    lcurve_selection,
    gcv_selection,
    discrepancy_principle_selection,
    cosine_similarity_selection,
    compare_regularization_methods,
    randomization_experiment,
)
from .dose_calculation import calculate_dose_rates

__all__ = [
    # detector
    "Detector",
    # unfolding methods
    "solve_cvxpy",
    "solve_landweber",
    "solve_mlem",
    "solve_qpsolvers",
    "solve_doroshenko",
    "solve_kaczmarz",
    "solve_lmfit",
    # regularization
    "select_regularization_parameter",
    "lcurve_selection",
    "gcv_selection",
    "discrepancy_principle_selection",
    "cosine_similarity_selection",
    "compare_regularization_methods",
    "randomization_experiment",
    # dose calculation
    "calculate_dose_rates",
]
