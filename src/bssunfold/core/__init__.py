"""Core modules for bssunfold package.

This subpackage contains the main functionality for neutron spectrum
unfolding, including the Detector class and unfolding methods.
"""

from ._dual_diagnostics import nnls_duality_gap, nnls_kkt_residuals
from ._line_search import (
    backtracking_line_search,
    brent_minimize,
    dichotomy_minimize,
    golden_section_minimize,
)
from .detector import Detector
from .dose_calculation import calculate_dose_rates
from .regularization import (
    compare_regularization_methods,
    cosine_similarity_selection,
    discrepancy_principle_selection,
    gcv_selection,
    lcurve_selection,
    randomization_experiment,
    select_regularization_parameter,
)
from .regularization_1d import select_regularization_1d
from .sisireg3d import (
    SSR3DModel,
    near_neighbors,
    near_neighbors_grid,
    near_neighbors_grid_quadrant,
    near_neighbors_quadrant,
    point_distance,
    ps_max_3d,
    ps_statistic_3d,
    ssr3d,
    ssr3d_predict,
    wmean,
    wmean_exp,
    wmean_ms,
)
from .sisireg_mlp import (
    SSRMLPModel,
    calc_out,
    check_ps,
    d1sigmoid,
    err_lse,
    err_ps,
    err_ps_l1,
    err_ps_lse,
    fac_lse,
    fac_ps,
    fac_ps_l1,
    fac_ps_lse,
    fii_model,
    fii_prediction,
    sigmoid,
    ssrmlp_predict,
    ssrmlp_train,
)
from .unfold_admm import solve_admm, unfold_admm
from .unfold_amaxed import solve_amaxed, unfold_amaxed
from .unfold_amaxed_regularization import (
    solve_amaxed_regularization,
    unfold_amaxed_regularization,
)
from .unfold_amg import solve_amg, unfold_amg
from .unfold_bayes import solve_bayes, unfold_bayes
from .unfold_bayes_spline_regularization import (
    solve_bayes_spline,
    unfold_bayes_spline_regularization,
)
from .unfold_bayesian_parametric import solve_bayesian_parametric
from .unfold_binned import (
    build_bin_lookup,
    load_bin_lookup,
    save_bin_lookup,
    solve_binned,
    unfold_binned,
)
from .unfold_bsrem import solve_bsrem, unfold_bsrem
from .unfold_bunki import solve_bunki, unfold_bunki
from .unfold_bunkiut import solve_bunkiut, unfold_bunkiut
from .unfold_cascade import unfold_cascade
from .unfold_cgls import solve_cgls, unfold_cgls
from .unfold_combined import unfold_combined
from .unfold_composite import unfold_composite
from .unfold_coordinate_descent import (
    solve_coordinate_descent,
    unfold_coordinate_descent,
)
from .unfold_crystal_ball import solve_crystal_ball, unfold_crystal_ball
from .unfold_cs import solve_cs, solve_ksvd, solve_omp, solve_sl0, unfold_cs
from .unfold_cvxpy import solve_cvxpy, unfold_cvxpy
from .unfold_directed_divergence import (
    solve_directed_divergence,
    unfold_directed_divergence,
)
from .unfold_docplex import solve_docplex, unfold_docplex
from .unfold_doroshenko import solve_doroshenko, unfold_doroshenko
from .unfold_ensemble import solve_ensemble, unfold_ensemble
from .unfold_epic import solve_epic, unfold_epic
from .unfold_express import solve_express, unfold_express
from .unfold_extragradient import solve_extragradient, unfold_extragradient
from .unfold_ferdor import solve_ferdor, unfold_ferdor
from .unfold_fission_ga import (
    ED_EPITHERMAL,
    FISSION_PARAM_BOUNDS,
    FISSION_PARAM_NAMES,
    T0_THERMAL,
    fission_model,
    solve_fission_ga,
    unfold_fission_ga,
)
from .unfold_frank_wolfe import solve_frank_wolfe, unfold_frank_wolfe
from .unfold_fruit_like import solve_fruit_like
from .unfold_gee import (
    estimate_alpha,
    gee_fit,
    solve_gee,
    solve_gee_full,
    unfold_gee,
    working_correlation,
)
from .unfold_genetic import solve_genetic, unfold_genetic
from .unfold_gks import solve_gks, unfold_gks
from .unfold_gnowee import solve_gnowee, unfold_gnowee
from .unfold_gravel import solve_gravel, unfold_gravel
from .unfold_hybrid_parametric import solve_hybrid_parametric
from .unfold_imaxed import solve_imaxed, unfold_imaxed
from .unfold_interpret import (
    InterpretationResult,
    build_interpretation_qp,
    interpret_qp,
    solve_interpret,
    unfold_interpret,
)
from .unfold_iterative_refinement import (
    solve_iterative_refinement,
    unfold_iterative_refinement,
)
from .unfold_kaczmarz import solve_kaczmarz, unfold_kaczmarz
from .unfold_lanczos import solve_lanczos, unfold_lanczos
from .unfold_landweber import solve_landweber, unfold_landweber
from .unfold_lbfgsb import solve_lbfgsb, unfold_lbfgsb
from .unfold_lmfit import solve_lmfit, unfold_lmfit
from .unfold_mapem import solve_mapem, unfold_mapem
from .unfold_maxed import solve_maxed, unfold_maxed
from .unfold_mcmc import solve_bayesian_mcmc, unfold_mcmc
from .unfold_mirror_descent import solve_mirror_descent, unfold_mirror_descent
from .unfold_mlem import solve_mlem, unfold_mlem
from .unfold_mlem_bs import (
    AUTO_BETA_RELATIVE_GRID,
    build_bspline_basis,
    ks_statistic,
    second_difference_matrix,
    solve_mlem_bs,
    solve_mlem_bs_full,
    unfold_mlem_bs,
)
from .unfold_mlem_odl import unfold_mlem_odl
from .unfold_mlem_stop import solve_mlem_stop, unfold_mlem_stop
from .unfold_mystic import (
    solve_mystic,
    solve_mystic_hybrid,
    unfold_mystic,
    unfold_mystic_hybrid,
)
from .unfold_nnksvd import (
    solve_nn_omp,
    solve_nnksvd,
    solve_nnksvd_unfold,
    solve_nnls_topk,
    solve_tikhonov_nnls,
    unfold_nnksvd,
)
from .unfold_nnqp import solve_nnqp, unfold_nnqp
from .unfold_nsduaz import (
    builtin_catalogue,
    select_catalogue_initial,
    solve_nsduaz,
    unfold_nsduaz,
)
from .unfold_nspline import (
    NSPLINE_KNOT_PRESETS,
    auto_knots,
    build_continuity_matrix,
    directed_divergence,
    fit_nspline,
    nspline_eval,
    solve_nspline,
    solve_nspline_full,
    unfold_nspline,
)
from .unfold_odl_advanced import (
    solve_odl_douglas_rachford,
    solve_odl_pdhg,
    unfold_odl_douglas_rachford,
    unfold_odl_pdhg,
)
from .unfold_osem import solve_osem, unfold_osem
from .unfold_parametric2 import solve_parametric2, unfold_parametric2
from .unfold_pgd import project_onto_set, solve_pgd, unfold_pgd
from .unfold_pspline_reml import (
    select_lambda_reml,
    solve_pspline_reml,
    solve_pspline_reml_full,
    unfold_pspline_reml,
)
from .unfold_qpmad import solve_qpmad, unfold_qpmad
from .unfold_qpsolvers import solve_qpsolvers, unfold_qpsolvers
from .unfold_qubo import solve_qubo_unfold, unfold_qubo
from .unfold_rebunki import solve_rebunki, unfold_rebunki
from .unfold_reconst import solve_reconst, unfold_reconst
from .unfold_rfsp_jul import solve_rfsp_jul, unfold_rfsp_jul
from .unfold_sandii import solve_sandii, unfold_sandii
from .unfold_sart import solve_sart, unfold_sart
from .unfold_scip import solve_scip, unfold_scip
from .unfold_scipy_direct_method import solve_scipy_direct, unfold_scipy_direct_method
from .unfold_smt import (
    solve_integer_linear_eqs,
    solve_integer_linear_eqs_all,
    solve_rational_linear_eqs,
    solve_rational_linear_eqs_all,
    solve_smt,
    unfold_smt,
)
from .unfold_ssr import (
    max_run_quantile,
    number_of_extrema,
    partial_sum_max,
    partial_sum_quantile,
    partial_sum_valid,
    rolling_median,
    run_valid,
    solve_ssr,
    solve_ssr_full,
    ssr,
    ssr_min_statistic,
    ssr_min_statistic_ne,
    ssr_ne,
    ssr_predict,
    unfold_ssr,
)
from .unfold_statreg import solve_statreg, unfold_statreg
from .unfold_staysl import solve_staysl, unfold_staysl
from .unfold_subgradient import solve_subgradient, unfold_subgradient
from .unfold_tikhonov_legendre import solve_tikhonov_legendre, unfold_tikhonov_legendre
from .unfold_tikhonov_sobolev_dp import (
    alpha_finder_generalized_discrepancy,
    generalized_discrepancy,
    solve_tikhonov_sobolev_dp,
    unfold_tikhonov_sobolev_dp,
)
from .unfold_tikhonov_tv import solve_tikhonov_tv, unfold_tikhonov_tv
from .unfold_tsvd import solve_tsvd, unfold_tsvd
from .unfold_uno import solve_uno, solve_uno_full, unfold_uno, uno_filter
from .unfold_zfit import solve_zfit_unfold, unfold_zfit

__all__ = [
    # detector
    "Detector",
    # unfolding methods
    "solve_cvxpy",
    "solve_landweber",
    "solve_mlem",
    "solve_nnqp",
    "solve_qpmad",
    "solve_qpsolvers",
    "solve_mystic",
    "solve_mystic_hybrid",
    "solve_genetic",
    "solve_gnowee",
    "solve_doroshenko",
    "solve_directed_divergence",
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
    "solve_amg",
    "solve_pspline_reml",
    "solve_pspline_reml_full",
    "select_lambda_reml",
    "solve_lanczos",
    "solve_cgls",
    "solve_gks",
    "solve_tikhonov_tv",
    "solve_tikhonov_sobolev_dp",
    "alpha_finder_generalized_discrepancy",
    "generalized_discrepancy",
    "solve_fission_ga",
    "unfold_fission_ga",
    "fission_model",
    "FISSION_PARAM_NAMES",
    "FISSION_PARAM_BOUNDS",
    "T0_THERMAL",
    "ED_EPITHERMAL",
    "solve_sandii",
    "solve_ssr",
    "solve_ssr_full",
    "solve_gee",
    "solve_gee_full",
    "gee_fit",
    "estimate_alpha",
    "working_correlation",
    "solve_uno",
    "solve_uno_full",
    "uno_filter",
    # Sign-Simplicity-Regression building blocks (R sisireg 1.2.1 port)
    "ssr",
    "ssr_ne",
    "ssr_min_statistic",
    "ssr_min_statistic_ne",
    "ssr_predict",
    "max_run_quantile",
    "partial_sum_quantile",
    "partial_sum_max",
    "partial_sum_valid",
    "run_valid",
    "rolling_median",
    "number_of_extrema",
    # Spatial SSR regression (ssr3d, R sisireg 1.2.1 port)
    "SSR3DModel",
    "point_distance",
    "near_neighbors_quadrant",
    "near_neighbors_grid_quadrant",
    "near_neighbors",
    "near_neighbors_grid",
    "ps_max_3d",
    "ps_statistic_3d",
    "wmean",
    "wmean_exp",
    "wmean_ms",
    "ssr3d",
    "ssr3d_predict",
    # SSR-MLP: 2-layer perceptron with the partial sum criterion
    "SSRMLPModel",
    "sigmoid",
    "d1sigmoid",
    "check_ps",
    "err_ps",
    "fac_ps",
    "err_lse",
    "fac_lse",
    "err_ps_lse",
    "fac_ps_lse",
    "err_ps_l1",
    "fac_ps_l1",
    "calc_out",
    "ssrmlp_train",
    "ssrmlp_predict",
    "fii_model",
    "fii_prediction",
    "solve_bunki",
    "solve_bunkiut",
    "solve_ferdor",
    "solve_rebunki",
    "solve_nsduaz",
    "select_catalogue_initial",
    "builtin_catalogue",
    # N-spline method (Islamgulov & Lartsev, Atomic Energy 104(5) 2008)
    "solve_nspline",
    "solve_nspline_full",
    "NSPLINE_KNOT_PRESETS",
    "auto_knots",
    "build_continuity_matrix",
    "fit_nspline",
    "nspline_eval",
    "directed_divergence",
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
    # B-spline MLEM-BS (Mazankova et al., CNDGS 2026)
    "solve_mlem_bs",
    "solve_mlem_bs_full",
    "build_bspline_basis",
    "second_difference_matrix",
    "ks_statistic",
    "AUTO_BETA_RELATIVE_GRID",
    "solve_integer_linear_eqs",
    "solve_integer_linear_eqs_all",
    "solve_rational_linear_eqs",
    "solve_rational_linear_eqs_all",
    "solve_smt",
    "solve_scip",
    "solve_docplex",
    "solve_cs",
    "solve_omp",
    "solve_crystal_ball",
    "solve_rfsp_jul",
    "solve_staysl",
    "solve_express",
    "solve_ksvd",
    "solve_sl0",
    # Non-negative K-SVD (Xu et al. NIMA 2026)
    "solve_nnksvd",
    "solve_nnksvd_unfold",
    "solve_nnls_topk",
    "solve_nn_omp",
    "solve_tikhonov_nnls",
    # unfold modules
    "unfold_cvxpy",
    "unfold_landweber",
    "unfold_mlem",
    "unfold_nnqp",
    "unfold_qpmad",
    "unfold_qpsolvers",
    "unfold_mystic",
    "unfold_mystic_hybrid",
    "unfold_genetic",
    "unfold_gnowee",
    "unfold_doroshenko",
    "unfold_directed_divergence",
    "unfold_kaczmarz",
    "unfold_lmfit",
    "unfold_mlem_odl",
    "unfold_mlem_stop",
    "unfold_mlem_bs",
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
    "unfold_amg",
    "unfold_pspline_reml",
    "unfold_gee",
    "unfold_uno",
    "unfold_lanczos",
    "unfold_cgls",
    "unfold_gks",
    "unfold_tikhonov_tv",
    "unfold_tikhonov_sobolev_dp",
    "unfold_sandii",
    "unfold_ssr",
    "unfold_express",
    "unfold_bunki",
    "unfold_bunkiut",
    "unfold_ferdor",
    "unfold_rebunki",
    "unfold_nsduaz",
    "unfold_nspline",
    "unfold_osem",
    "unfold_mapem",
    "unfold_bsrem",
    "unfold_sart",
    "unfold_mcmc",
    "unfold_parametric2",
    "unfold_fruit_like",
    "unfold_fission_ga",
    "unfold_hybrid_parametric",
    "unfold_bayesian_parametric",
    "unfold_smt",
    "unfold_scip",
    "unfold_docplex",
    "unfold_cs",
    "unfold_crystal_ball",
    "unfold_rfsp_jul",
    "unfold_staysl",
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
    # bin-wise adaptive method
    "build_bin_lookup",
    "load_bin_lookup",
    "save_bin_lookup",
    "solve_binned",
    "unfold_binned",
    # ensemble method
    "solve_ensemble",
    "unfold_ensemble",
    # iterative refinement
    "solve_iterative_refinement",
    "unfold_iterative_refinement",
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
    # Non-negative K-SVD unfold (Xu et al. NIMA 2026)
    "unfold_nnksvd",
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
    # Optimization-course methods (MIPT OPTIMIZATION-METHODS-COURSE)
    "solve_pgd",
    "unfold_pgd",
    "project_onto_set",
    "solve_frank_wolfe",
    "unfold_frank_wolfe",
    "solve_mirror_descent",
    "unfold_mirror_descent",
    "solve_admm",
    "unfold_admm",
    "soft_threshold",
    "solve_lbfgsb",
    "unfold_lbfgsb",
    "second_difference_matrix",
    "solve_coordinate_descent",
    "unfold_coordinate_descent",
    "solve_subgradient",
    "unfold_subgradient",
    "solve_extragradient",
    "unfold_extragradient",
    # 1D optimization building blocks (golden section / dichotomy / Brent)
    "golden_section_minimize",
    "dichotomy_minimize",
    "brent_minimize",
    "backtracking_line_search",
    "select_regularization_1d",
    # Duality / KKT diagnostics
    "nnls_duality_gap",
    "nnls_kkt_residuals",
]
