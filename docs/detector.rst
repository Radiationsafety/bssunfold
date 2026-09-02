Detector Class
==============

.. autoclass:: bssunfold.Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __str__, __repr__
   
Unfold Methods
==============

The following unfolding methods are available through the Detector class:

.. autofunction:: bssunfold.core.unfold_cvxpy.unfold_cvxpy

.. autofunction:: bssunfold.core.unfold_landweber.unfold_landweber

.. autofunction:: bssunfold.core.unfold_mlem.unfold_mlem

.. autofunction:: bssunfold.core.unfold_qpsolvers.unfold_qpsolvers

.. autofunction:: bssunfold.core.unfold_mystic.unfold_mystic

.. autofunction:: bssunfold.core.unfold_mystic.unfold_mystic_hybrid

.. autofunction:: bssunfold.core.unfold_genetic.unfold_genetic

.. autofunction:: bssunfold.core.unfold_smt.unfold_smt

.. autofunction:: bssunfold.core.unfold_scip.unfold_scip

.. autofunction:: bssunfold.core.unfold_docplex.unfold_docplex

.. autofunction:: bssunfold.core.unfold_epic.unfold_epic

.. autofunction:: bssunfold.core.unfold_cs.unfold_cs

.. autofunction:: bssunfold.core.unfold_doroshenko.unfold_doroshenko

.. autofunction:: bssunfold.core.unfold_kaczmarz.unfold_kaczmarz

.. autofunction:: bssunfold.core.unfold_lmfit.unfold_lmfit

.. autofunction:: bssunfold.core.unfold_mlem_odl.unfold_mlem_odl

.. autofunction:: bssunfold.core.unfold_mlem_stop.unfold_mlem_stop

.. autofunction:: bssunfold.core.unfold_combined.unfold_combined

.. autofunction:: bssunfold.core.unfold_interpret.unfold_interpret

.. autofunction:: bssunfold.core.unfold_gravel.unfold_gravel

.. autofunction:: bssunfold.core.unfold_maxed.unfold_maxed

.. autofunction:: bssunfold.core.unfold_tikhonov_legendre.unfold_tikhonov_legendre

.. autofunction:: bssunfold.core.unfold_bayes.unfold_bayes

.. autofunction:: bssunfold.core.unfold_bayes_spline_regularization.unfold_bayes_spline_regularization

.. autofunction:: bssunfold.core.unfold_statreg.unfold_statreg

.. autofunction:: bssunfold.core.unfold_reconst.unfold_reconst

.. autofunction:: bssunfold.core.unfold_scipy_direct_method.unfold_scipy_direct_method

.. autofunction:: bssunfold.core.unfold_tsvd.unfold_tsvd

.. autofunction:: bssunfold.core.unfold_lanczos.unfold_lanczos

.. autofunction:: bssunfold.core.unfold_cgls.unfold_cgls

.. autofunction:: bssunfold.core.unfold_gks.unfold_gks

.. autofunction:: bssunfold.core.unfold_tikhonov_tv.unfold_tikhonov_tv

.. autofunction:: bssunfold.core.unfold_sandii.unfold_sandii

.. autofunction:: bssunfold.core.unfold_crystal_ball.unfold_crystal_ball

.. autofunction:: bssunfold.core.unfold_rfsp_jul.unfold_rfsp_jul

.. autofunction:: bssunfold.core.unfold_staysl.unfold_staysl

.. autofunction:: bssunfold.core.unfold_bunki.unfold_bunki

.. autofunction:: bssunfold.core.unfold_bunkiut.unfold_bunkiut

.. autofunction:: bssunfold.core.unfold_osem.unfold_osem

.. autofunction:: bssunfold.core.unfold_mapem.unfold_mapem

.. autofunction:: bssunfold.core.unfold_bsrem.unfold_bsrem

.. autofunction:: bssunfold.core.unfold_sart.unfold_sart

.. autofunction:: bssunfold.core.unfold_ferdor.unfold_ferdor

.. autofunction:: bssunfold.core.unfold_rebunki.unfold_rebunki

.. autofunction:: bssunfold.core.unfold_nsduaz.unfold_nsduaz

.. autofunction:: bssunfold.core.unfold_parametric.unfold_parametric

.. autofunction:: bssunfold.core.unfold_parametric.solve_parametric_cvxpy

.. autofunction:: bssunfold.core.unfold_parametric.solve_parametric_qpsolvers

.. autofunction:: bssunfold.core.unfold_parametric.solve_parametric_combined

.. autofunction:: bssunfold.core.unfold_parametric2.unfold_parametric2

.. autofunction:: bssunfold.core.unfold_parametric2.solve_parametric2

.. autofunction:: bssunfold.core.unfold_parametric2.solve_bon95_parametric

.. autofunction:: bssunfold.core.unfold_parametric2.directed_divergence_iteration

.. autofunction:: bssunfold.core.unfold_parametric2.solve_bon95_cvxpy

.. autofunction:: bssunfold.core.unfold_parametric2.solve_bon95_qpsolvers

.. autofunction:: bssunfold.core.unfold_parametric2.solve_bon95_combined

.. autofunction:: bssunfold.core.unfold_fruit_like.unfold_fruit_like

.. autofunction:: bssunfold.core.unfold_hybrid_parametric.unfold_hybrid_parametric

.. autofunction:: bssunfold.core.unfold_bayesian_parametric.unfold_bayesian_parametric

.. autofunction:: bssunfold.core.unfold_imaxed.unfold_imaxed

.. autofunction:: bssunfold.core.unfold_amaxed.unfold_amaxed

.. autofunction:: bssunfold.core.unfold_amaxed_regularization.unfold_amaxed_regularization

.. autofunction:: bssunfold.core.unfold_fista.unfold_fista

.. autofunction:: bssunfold.core.unfold_hybrid_gmres.unfold_hybrid_gmres

.. autofunction:: bssunfold.core.unfold_mcmc.unfold_mcmc

.. autofunction:: bssunfold.core.unfold_zfit.unfold_zfit

.. autofunction:: bssunfold.core.unfold_qubo.unfold_qubo

.. autofunction:: bssunfold.core.unfold_maeo.unfold_maeo

.. autofunction:: bssunfold.core.unfold_odl_advanced.unfold_odl_pdhg

.. autofunction:: bssunfold.core.unfold_odl_advanced.unfold_odl_douglas_rachford

.. autofunction:: bssunfold.core.unfold_cascade.unfold_cascade

.. autofunction:: bssunfold.core.unfold_composite.unfold_composite

.. autofunction:: bssunfold.core.unfold_binned.unfold_binned

.. autofunction:: bssunfold.core.unfold_ensemble.unfold_ensemble

.. autofunction:: bssunfold.core.unfold_iterative_refinement.unfold_iterative_refinement

.. autofunction:: bssunfold.core.unfold_randomized_kaczmarz.unfold_randomized_kaczmarz

.. autofunction:: bssunfold.core.unfold_eki.unfold_eki

Core Functions
==============

Underlying solver functions:

.. autofunction:: bssunfold.core.unfold_cvxpy.solve_cvxpy

.. autofunction:: bssunfold.core.unfold_landweber.solve_landweber

.. autofunction:: bssunfold.core.unfold_mlem.solve_mlem

.. autofunction:: bssunfold.core.unfold_mlem_stop.solve_mlem_stop

.. autofunction:: bssunfold.core.unfold_qpsolvers.solve_qpsolvers

.. autofunction:: bssunfold.core.unfold_mystic.solve_mystic

.. autofunction:: bssunfold.core.unfold_mystic.solve_mystic_hybrid

.. autofunction:: bssunfold.core.unfold_genetic.solve_genetic

.. autofunction:: bssunfold.core.unfold_smt.solve_smt

.. autofunction:: bssunfold.core.unfold_scip.solve_scip

.. autofunction:: bssunfold.core.unfold_docplex.solve_docplex

.. autofunction:: bssunfold.core.unfold_epic.solve_epic

.. autofunction:: bssunfold.core.unfold_interpret.solve_interpret

.. autofunction:: bssunfold.core.unfold_interpret.interpret_qp

.. autofunction:: bssunfold.core.unfold_interpret.build_interpretation_qp

.. autofunction:: bssunfold.core.unfold_cs.solve_cs

.. autofunction:: bssunfold.core.unfold_cs.solve_omp

.. autofunction:: bssunfold.core.unfold_cs.solve_ksvd

.. autofunction:: bssunfold.core.unfold_cs.solve_sl0

.. autofunction:: bssunfold.core.unfold_doroshenko.solve_doroshenko

.. autofunction:: bssunfold.core.unfold_kaczmarz.solve_kaczmarz

.. autofunction:: bssunfold.core.unfold_lmfit.solve_lmfit

.. autofunction:: bssunfold.core.unfold_lmfit.select_regularization_aic_bic

.. autofunction:: bssunfold.core.unfold_gravel.solve_gravel

.. autofunction:: bssunfold.core.unfold_maxed.solve_maxed

.. autofunction:: bssunfold.core.unfold_tikhonov_legendre.solve_tikhonov_legendre

.. autofunction:: bssunfold.core.unfold_bayes.solve_bayes

.. autofunction:: bssunfold.core.unfold_bayes_spline_regularization.solve_bayes_spline

.. autofunction:: bssunfold.core.unfold_statreg.solve_statreg

.. autofunction:: bssunfold.core.unfold_reconst.solve_reconst

.. autofunction:: bssunfold.core.unfold_scipy_direct_method.solve_scipy_direct

.. autofunction:: bssunfold.core.unfold_tsvd.solve_tsvd

.. autofunction:: bssunfold.core.unfold_lanczos.solve_lanczos

.. autofunction:: bssunfold.core.unfold_cgls.solve_cgls

.. autofunction:: bssunfold.core.unfold_gks.solve_gks

.. autofunction:: bssunfold.core.unfold_tikhonov_tv.solve_tikhonov_tv

.. autofunction:: bssunfold.core.unfold_sandii.solve_sandii

.. autofunction:: bssunfold.core.unfold_crystal_ball.solve_crystal_ball

.. autofunction:: bssunfold.core.unfold_rfsp_jul.solve_rfsp_jul

.. autofunction:: bssunfold.core.unfold_staysl.solve_staysl

.. autofunction:: bssunfold.core.unfold_bunki.solve_bunki

.. autofunction:: bssunfold.core.unfold_bunkiut.solve_bunkiut

.. autofunction:: bssunfold.core.unfold_osem.solve_osem

.. autofunction:: bssunfold.core.unfold_mapem.solve_mapem

.. autofunction:: bssunfold.core.unfold_bsrem.solve_bsrem

.. autofunction:: bssunfold.core.unfold_sart.solve_sart

.. autofunction:: bssunfold.core.unfold_ferdor.solve_ferdor

.. autofunction:: bssunfold.core.unfold_rebunki.solve_rebunki

.. autofunction:: bssunfold.core.unfold_nsduaz.solve_nsduaz

.. autofunction:: bssunfold.core.unfold_randomized_kaczmarz.solve_randomized_kaczmarz

.. autofunction:: bssunfold.core.unfold_eki.solve_eki

.. autofunction:: bssunfold.core.unfold_binned.solve_binned

.. autofunction:: bssunfold.core.unfold_binned.build_bin_lookup

Comparison Methods
==================

.. autofunction:: bssunfold.utils.comparison.compare_spectra

.. autofunction:: bssunfold.utils.comparison.compare_multiple

Comparison Metrics
==================

Entropy-based
-------------

.. autofunction:: bssunfold.utils.comparison.kl_divergence

.. autofunction:: bssunfold.utils.comparison.cross_entropy

.. autofunction:: bssunfold.utils.comparison.entropy

.. autofunction:: bssunfold.utils.comparison.entropy_difference_percent

Distribution distances
----------------------

.. autofunction:: bssunfold.utils.comparison.wasserstein_dist

.. autofunction:: bssunfold.utils.comparison.energy_dist

.. autofunction:: bssunfold.utils.comparison.kolmogorov_smirnov_stat

Correlation
-----------

.. autofunction:: bssunfold.utils.comparison.pearson_r

.. autofunction:: bssunfold.utils.comparison.spearman_r

Error metrics
-------------

.. autofunction:: bssunfold.utils.comparison.mean_squared_error

.. autofunction:: bssunfold.utils.comparison.root_mean_squared_error

.. autofunction:: bssunfold.utils.comparison.mean_absolute_error

.. autofunction:: bssunfold.utils.comparison.mape

.. autofunction:: bssunfold.utils.comparison.r2_score

.. autofunction:: bssunfold.utils.comparison.max_error

.. autofunction:: bssunfold.utils.comparison.median_absolute_error

Kernel / similarity
-------------------

.. autofunction:: bssunfold.utils.comparison.cosine_similarity

.. autofunction:: bssunfold.utils.comparison.mmd_rbf

.. autofunction:: bssunfold.utils.comparison.total_flux_ratio

.. autofunction:: bssunfold.utils.comparison.spectral_shape_similarity

Chi-squared family
------------------

.. autofunction:: bssunfold.utils.comparison.chi_squared

.. autofunction:: bssunfold.utils.comparison.g_test

.. autofunction:: bssunfold.utils.comparison.freeman_tukey

.. autofunction:: bssunfold.utils.comparison.cressie_read

Statistical tests
-----------------

.. autofunction:: bssunfold.utils.comparison.anderson_darling

.. autofunction:: bssunfold.utils.comparison.wilcoxon_test

.. autofunction:: bssunfold.utils.comparison.mannwhitneyu_test

.. autofunction:: bssunfold.utils.comparison.standardized_mean_difference

Integral quantity metrics
-----------------------------------

.. autofunction:: bssunfold.utils.comparison.fluence_averaged_energy

.. autofunction:: bssunfold.utils.comparison.energy_group_fluence

.. autofunction:: bssunfold.utils.comparison.dose_averaged_energy

.. autofunction:: bssunfold.utils.comparison.ambient_dose_equivalent_rate

Spectral diagnostics (energy grid required)
-------------------------------------------

These metrics compare two spectra in physically meaningful terms
(fluence, dose, energy groups, peaks) and are computed automatically by
:func:`~bssunfold.utils.comparison.compare_spectra` when an ``energy``
grid is supplied.

.. autofunction:: bssunfold.utils.comparison.fluence_difference_percent

.. autofunction:: bssunfold.utils.comparison.energy_group_fluence_diff

.. autofunction:: bssunfold.utils.comparison.dose_difference_percent

.. autofunction:: bssunfold.utils.comparison.fluence_averaged_energy_diff

.. autofunction:: bssunfold.utils.comparison.dose_averaged_energy_diff

.. autofunction:: bssunfold.utils.comparison.log_lethargy_correlation

.. autofunction:: bssunfold.utils.comparison.peak_location_error

.. autofunction:: bssunfold.utils.comparison.peak_width_error

.. autofunction:: bssunfold.utils.comparison.dose_weighted_error

.. autofunction:: bssunfold.utils.comparison.response_matrix_consistency

Regularization Selection
========================

.. autofunction:: bssunfold.core.regularization.select_regularization_parameter

.. autofunction:: bssunfold.core.regularization.lcurve_selection

.. autofunction:: bssunfold.core.regularization.gcv_selection

.. autofunction:: bssunfold.core.regularization.discrepancy_principle_selection

.. autofunction:: bssunfold.core.regularization.cosine_similarity_selection

.. autofunction:: bssunfold.core.regularization.quasi_optimality_selection

.. autofunction:: bssunfold.core.regularization.ncp_selection

.. autofunction:: bssunfold.core.regularization.snr_criterion_selection

.. autofunction:: bssunfold.core.regularization.weighted_gcv_poisson_selection

.. autofunction:: bssunfold.core.regularization.kfold_cv_selection
   
