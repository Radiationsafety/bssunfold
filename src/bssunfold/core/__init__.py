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
from .unfold_cvxpy import unfold_cvxpy
from .unfold_landweber import unfold_landweber
from .unfold_mlem import unfold_mlem
from .unfold_qpsolvers import unfold_qpsolvers
from .unfold_doroshenko import unfold_doroshenko
from .unfold_kaczmarz import unfold_kaczmarz
from .unfold_lmfit import unfold_lmfit
from .unfold_mlem_odl import unfold_mlem_odl
from .unfold_combined import unfold_combined

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
    # unfold modules
    "unfold_cvxpy",
    "unfold_landweber",
    "unfold_mlem",
    "unfold_qpsolvers",
    "unfold_doroshenko",
    "unfold_kaczmarz",
    "unfold_lmfit",
    "unfold_mlem_odl",
    "unfold_combined",
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
