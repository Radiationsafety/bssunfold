"""Core modules for bssunfold package.

This subpackage contains the main functionality for neutron spectrum
unfolding, including the Detector class and unfolding methods.
"""

from .detector import Detector
from .unfold_landweber import solve_landweber
from .unfold_mlem import solve_mlem
from .unfold_kaczmarz import solve_kaczmarz
from .unfold_doroshenko import solve_doroshenko
from .unfold_cvxpy import solve_cvxpy
from .unfold_qpsolvers import solve_qpsolvers
from .unfold_lmfit import solve_lmfit
from .unfold_gravel import solve_gravel
from .unfold_maxed import solve_maxed
from .unfold_tikhonov_legendre import solve_tikhonov_legendre
from .unfold_bayes import solve_bayes
from .unfold_bayes_spline_regularization import solve_bayes_spline
from .unfold_statreg import solve_statreg
from .unfold_scipy_direct_method import solve_scipy_direct
from .unfold_tsvd import solve_tsvd
from .unfold_parametric2 import solve_parametric2
from .unfold_fruit_like import solve_fruit_like
from .unfold_hybrid_parametric import solve_hybrid_parametric
from .unfold_bayesian_parametric import solve_bayesian_parametric
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
from .unfold_gravel import unfold_gravel
from .unfold_maxed import unfold_maxed
from .unfold_tikhonov_legendre import unfold_tikhonov_legendre
from .unfold_bayes import unfold_bayes
from .unfold_bayes_spline_regularization import unfold_bayes_spline_regularization
from .unfold_statreg import unfold_statreg
from .unfold_scipy_direct_method import unfold_scipy_direct_method
from .unfold_tsvd import unfold_tsvd
from .unfold_parametric2 import unfold_parametric2

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
    "solve_gravel",
    "solve_maxed",
    "solve_tikhonov_legendre",
    "solve_bayes",
    "solve_bayes_spline",
    "solve_statreg",
    "solve_scipy_direct",
    "solve_tsvd",
    "solve_parametric2",
    "solve_fruit_like",
    "solve_hybrid_parametric",
    "solve_bayesian_parametric",
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
    "unfold_gravel",
    "unfold_maxed",
    "unfold_tikhonov_legendre",
    "unfold_bayes",
    "unfold_bayes_spline_regularization",
    "unfold_statreg",
    "unfold_scipy_direct_method",
    "unfold_tsvd",
    "unfold_parametric2",
    "unfold_fruit_like",
    "unfold_hybrid_parametric",
    "unfold_bayesian_parametric",
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
