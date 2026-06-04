"""Core modules for bssunfold package.

This subpackage contains the main functionality for neutron spectrum
unfolding, including the Detector class and unfolding methods.
"""

from .detector import Detector
from .unfolding_methods import (
    unfold_cvxpy,
    unfold_landweber,
    unfold_mlem,
    unfold_qpsolvers,
)
from .regularization import (
    select_regularization_parameter,
    lcurve_selection,
    gcv_selection,
    discrepancy_principle_selection,
)
from .dose_calculation import calculate_dose_rates

__all__ = [
    # detector
    "Detector",
    # unfolding methods
    "unfold_cvxpy",
    "unfold_landweber",
    "unfold_mlem",
    "unfold_qpsolvers",
    # regularization
    "select_regularization_parameter",
    "lcurve_selection",
    "gcv_selection",
    "discrepancy_principle_selection",
    # dose calculation
    "calculate_dose_rates",
]
