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
from .unfold_mystic import solve_mystic
from .unfold_genetic import solve_genetic
from .unfold_lmfit import solve_lmfit
from .unfold_gravel import solve_gravel
from .unfold_maxed import solve_maxed
from .unfold_tikhonov_legendre import solve_tikhonov_legendre
from .unfold_bayes import solve_bayes
from .unfold_bayes_spline_regularization import solve_bayes_spline
from .unfold_statreg import solve_statreg
from .unfold_reconst import solve_reconst
from .unfold_scipy_direct_method import solve_scipy_direct
from .unfold_tsvd import solve_tsvd
from .unfold_lanczos import solve_lanczos
from .unfold_cgls import solve_cgls
from .unfold_gks import solve_gks
from .unfold_tikhonov_tv import solve_tikhonov_tv
from .unfold_sandii import solve_sandii
from .unfold_bunki import solve_bunki
from .unfold_bunkiut import solve_bunkiut
from .unfold_ferdor import solve_ferdor
from .unfold_rebunki import solve_rebunki
from .unfold_nsduaz import (
    solve_nsduaz,
    select_catalogue_initial,
    builtin_catalogue,
)
from .unfold_osem import solve_osem
from .unfold_mapem import solve_mapem
from .unfold_bsrem import solve_bsrem
from .unfold_sart import solve_sart
from .unfold_parametric2 import solve_parametric2
from .unfold_fruit_like import solve_fruit_like
from .unfold_hybrid_parametric import solve_hybrid_parametric
from .unfold_bayesian_parametric import solve_bayesian_parametric
from .unfold_smt import (
    solve_integer_linear_eqs,
    solve_integer_linear_eqs_all,
    solve_rational_linear_eqs,
    solve_rational_linear_eqs_all,
    solve_smt,
)
from .unfold_scip import solve_scip
from .unfold_docplex import solve_docplex
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
from .unfold_mystic import unfold_mystic
from .unfold_genetic import unfold_genetic
from .unfold_doroshenko import unfold_doroshenko
from .unfold_kaczmarz import unfold_kaczmarz
from .unfold_lmfit import unfold_lmfit
from .unfold_mlem_odl import unfold_mlem_odl
from .unfold_mlem_stop import solve_mlem_stop
from .unfold_mlem_stop import unfold_mlem_stop
from .unfold_combined import unfold_combined
from .unfold_gravel import unfold_gravel
from .unfold_maxed import unfold_maxed
from .unfold_tikhonov_legendre import unfold_tikhonov_legendre
from .unfold_bayes import unfold_bayes
from .unfold_bayes_spline_regularization import (
    unfold_bayes_spline_regularization,
)
from .unfold_statreg import unfold_statreg
from .unfold_reconst import unfold_reconst
from .unfold_scipy_direct_method import unfold_scipy_direct_method
from .unfold_tsvd import unfold_tsvd
from .unfold_lanczos import unfold_lanczos
from .unfold_cgls import unfold_cgls
from .unfold_gks import unfold_gks
from .unfold_tikhonov_tv import unfold_tikhonov_tv
from .unfold_sandii import unfold_sandii
from .unfold_bunki import unfold_bunki
from .unfold_bunkiut import unfold_bunkiut
from .unfold_ferdor import unfold_ferdor
from .unfold_rebunki import unfold_rebunki
from .unfold_nsduaz import unfold_nsduaz
from .unfold_osem import unfold_osem
from .unfold_mapem import unfold_mapem
from .unfold_bsrem import unfold_bsrem
from .unfold_sart import unfold_sart
from .unfold_mcmc import solve_bayesian_mcmc, unfold_mcmc
from .unfold_parametric2 import unfold_parametric2
from .unfold_smt import unfold_smt
from .unfold_scip import unfold_scip
from .unfold_docplex import unfold_docplex
from .unfold_cs import solve_cs, solve_omp, solve_ksvd, solve_sl0, unfold_cs
from .unfold_epic import solve_epic, unfold_epic
from .unfold_interpret import (
    InterpretationResult,
    build_interpretation_qp,
    solve_interpret,
    interpret_qp,
    unfold_interpret,
)
from .unfold_imaxed import solve_imaxed, unfold_imaxed
from .unfold_amaxed import solve_amaxed, unfold_amaxed
from .unfold_amaxed_regularization import (
    solve_amaxed_regularization,
    unfold_amaxed_regularization,
)
from .unfold_odl_advanced import (
    solve_odl_pdhg,
    solve_odl_douglas_rachford,
    unfold_odl_pdhg,
    unfold_odl_douglas_rachford,
)
from .unfold_qubo import solve_qubo_unfold, unfold_qubo
from .unfold_zfit import solve_zfit_unfold, unfold_zfit
from .unfold_cascade import unfold_cascade
from .unfold_composite import unfold_composite

__all__ = [
    # detector
    "Detector",
    # unfolding methods
    "solve_cvxpy",
    "solve_landweber",
    "solve_mlem",
    "solve_qpsolvers",
    "solve_mystic",
    "solve_genetic",
    "solve_doroshenko",
    "solve_kaczmarz",
    "solve_lmfit",
    "solve_gravel",
    "solve_maxed",
    "solve_tikhonov_legendre",
    "solve_bayes",
    "solve_bayes_spline",
    "solve_statreg",
    "solve_reconst",
    "solve_scipy_direct",
    "solve_tsvd",
    "solve_lanczos",
    "solve_cgls",
    "solve_gks",
    "solve_tikhonov_tv",
    "solve_sandii",
    "solve_bunki",
    "solve_bunkiut",
    "solve_ferdor",
    "solve_rebunki",
    "solve_nsduaz",
    "select_catalogue_initial",
    "builtin_catalogue",
    "solve_osem",
    "solve_mapem",
    "solve_bsrem",
    "solve_sart",
    "solve_bayesian_mcmc",
    "solve_parametric2",
    "solve_fruit_like",
    "solve_hybrid_parametric",
    "solve_bayesian_parametric",
    "solve_mlem_stop",
    "solve_integer_linear_eqs",
    "solve_integer_linear_eqs_all",
    "solve_rational_linear_eqs",
    "solve_rational_linear_eqs_all",
    "solve_smt",
    "solve_scip",
    "solve_docplex",
    "solve_cs",
    "solve_omp",
    "solve_ksvd",
    "solve_sl0",
    # unfold modules
    "unfold_cvxpy",
    "unfold_landweber",
    "unfold_mlem",
    "unfold_qpsolvers",
    "unfold_mystic",
    "unfold_genetic",
    "unfold_doroshenko",
    "unfold_kaczmarz",
    "unfold_lmfit",
    "unfold_mlem_odl",
    "unfold_mlem_stop",
    "unfold_combined",
    "unfold_gravel",
    "unfold_maxed",
    "unfold_tikhonov_legendre",
    "unfold_bayes",
    "unfold_bayes_spline_regularization",
    "unfold_statreg",
    "unfold_reconst",
    "unfold_scipy_direct_method",
    "unfold_tsvd",
    "unfold_lanczos",
    "unfold_cgls",
    "unfold_gks",
    "unfold_tikhonov_tv",
    "unfold_sandii",
    "unfold_bunki",
    "unfold_bunkiut",
    "unfold_ferdor",
    "unfold_rebunki",
    "unfold_nsduaz",
    "unfold_osem",
    "unfold_mapem",
    "unfold_bsrem",
    "unfold_sart",
    "unfold_mcmc",
    "unfold_parametric2",
    "unfold_fruit_like",
    "unfold_hybrid_parametric",
    "unfold_bayesian_parametric",
    "unfold_smt",
    "unfold_scip",
    "unfold_docplex",
    "unfold_cs",
    "solve_epic",
    "unfold_epic",
    # ODL advanced methods
    "solve_odl_pdhg",
    "solve_odl_douglas_rachford",
    "unfold_odl_pdhg",
    "unfold_odl_douglas_rachford",
    # QUBO quantum-inspired method
    "solve_qubo_unfold",
    "unfold_qubo",
    # zfit Bayesian method
    "solve_zfit_unfold",
    "unfold_zfit",
    # cascade / composite (ensemble) methods
    "unfold_cascade",
    "unfold_composite",
    # interpretation
    "InterpretationResult",
    "build_interpretation_qp",
    "solve_interpret",
    "interpret_qp",
    "unfold_interpret",
    # Wong 2024 PhD thesis methods
    "solve_imaxed",
    "unfold_imaxed",
    "solve_amaxed",
    "unfold_amaxed",
    "solve_amaxed_regularization",
    "unfold_amaxed_regularization",
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
