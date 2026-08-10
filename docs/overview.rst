Package Overview
================

BSSUnfold is a Python package for neutron spectrum unfolding from Bonner Sphere
Spectrometers (BSS). It provides 51 unfolding algorithms, 25 spectrum
comparison metrics, ICRP-116 dose calculations, and Monte Carlo uncertainty
quantification. Iterative solvers are accelerated with Numba JIT compilation.

.. contents::
   :local:
   :depth: 2

Unfolding Methods
-----------------

All 51 methods are accessible as instance methods on the
:class:`bssunfold.Detector` class. They are organised into the following
categories:

.. mermaid::

   graph TD
       A["Unfolding Methods"] --> B["Tikhonov-type"]
       A --> C["Iterative"]
       A --> D["Bayesian"]
       A --> E["Maximum Entropy"]
       A --> F["Statistical Regularization"]
       A --> G["Optimization-based"]
       A --> H["Pipeline"]
       A --> I["Parametric"]
       A --> J["Krylov/hybrid"]
       A --> K["EM family"]
       A --> L["Multi-sphere ratio"]

       B --> B1["unfold_cvxpy"]
       B --> B2["unfold_qpsolvers"]
       B --> B3["unfold_tsvd"]
       B --> B4["unfold_tikhonov_legendre"]

       J --> J1["unfold_lanczos"]
       J --> J2["unfold_gks"]
       J --> J3["unfold_cgls"]
       J --> J4["unfold_hybrid_gmres"]
       J --> J5["unfold_fista"]

       C --> C1["unfold_landweber"]
        C --> C2["unfold_mlem"]
        C --> C3["unfold_mlem_stop"]
        C --> C4["unfold_mlem_odl"]
        C --> C5["unfold_gravel"]
        C --> C6["unfold_doroshenko"]
        C --> C7["unfold_kaczmarz"]
        C --> C8["unfold_sart"]

       K --> K1["unfold_osem"]
       K --> K2["unfold_mapem"]
       K --> K3["unfold_bsrem"]

       L --> L1["unfold_sandii"]
       L --> L2["unfold_bunki"]
       L --> L3["unfold_bunkiut"]
       L --> L4["unfold_rebunki"]
       L --> L5["unfold_nsduaz"]
       L --> L6["unfold_ferdor"]

       D --> D1["unfold_bayes"]
       D --> D2["unfold_bayes_spline_regularization"]

       E --> E1["unfold_maxed"]
        F --> F1["unfold_statreg"]
        F --> F2["unfold_reconst"]

       G --> G1["unfold_lmfit"]
       G --> G2["unfold_scipy_direct_method"]
       G --> G3["unfold_mystic"]
       G --> G4["unfold_smt"]
       G --> G5["unfold_genetic"]
       G --> G6["unfold_cs"]
       G --> G7["unfold_scip"]
       G --> G8["unfold_docplex"]
       G --> G9["unfold_epic"]

       H --> H1["unfold_combined"]

       I --> I1["unfold_parametric"]
       I --> I2["unfold_parametric_cvxpy"]
       I --> I3["unfold_parametric_qpsolvers"]
       I --> I4["unfold_parametric_combined"]
       I --> I5["unfold_parametric2"]
       I --> I6["unfold_fruit_like"]
       I --> I7["unfold_hybrid_parametric"]
       I --> I8["unfold_bayesian_parametric"]

       style A fill:#4a90d9,color:#fff
       style B fill:#e8f0fe
       style C fill:#e8f0fe
       style D fill:#e8f0fe
       style E fill:#e8f0fe
       style F fill:#e8f0fe
       style G fill:#e8f0fe
       style H fill:#e8f0fe
       style I fill:#e8f0fe
       style J fill:#e8f0fe

Method Reference
~~~~~~~~~~~~~~~~

.. list-table:: Complete method reference
   :header-rows: 1
   :widths: 5 4 4 15 8 12

   * - #
     - Method
     - Category
     - Unique Parameters
     - Dependencies
     - Description
   * - 1
     - ``unfold_cvxpy``
     - Tikhonov
     - `regularization`, `norm` (1/2), `solver`, `regularization_method`
     - cvxpy
     - Convex optimization with Tikhonov regularization
   * - 2
     - ``unfold_qpsolvers``
     - Tikhonov
     - `regularization`, `norm` (1/2), `solver`, `smoothness_order`, `smoothness_weight`, `regularization_method`
     - qpsolvers
     - QP-based unfolding with L1/L2/smoothness norms
   * - 3
     - ``unfold_tsvd``
     - Tikhonov
     - `method` (l_curve/gcv/discrepancy/energy/median/donoho), `k`, `threshold`, `noise_level`
     - —
     - Truncated SVD with automatic k-selection
   * - 4
     - ``unfold_lanczos``
     - Krylov/hybrid
     - `regularization_method` (gcv), `max_iterations`, `regularization`, `noise_level`
     - —
     - Lanczos-hybrid (Golub-Kahan bidiagonalization) with automatic per-iteration GCV regularization; no a-priori spectrum required
   * - 5
     - ``unfold_tikhonov_legendre``
     - Tikhonov
     - `delta`, `n_polynomials`
     - —
     - Tikhonov regularization in Legendre polynomial basis
   * - 6
     - ``unfold_landweber``
     - Iterative
     - `max_iterations`, `tolerance`
     - —
     - Landweber fixed-point iteration
   * - 7
     - ``unfold_mlem``
     - Iterative
     - `max_iterations`, `tolerance`
     - —
     - Pure-NumPy MLEM (expectation maximization)
   * - 8
     - ``unfold_mlem_stop``
     - Iterative
     - `max_iterations`, `cps_crossover`, `j_threshold`
     - —
     - MLEM with J-factor early stopping criterion (Montgomery et al. 2020)
   * - 9
     - ``unfold_mlem_odl``
     - Iterative
     - `max_iterations`, `tolerance`
     - odl
     - MLEM via ODL operator framework
   * - 10
     - ``unfold_gravel``
     - Iterative
     - `max_iterations`, `tolerance`, `regularization`
     - —
     - GRAVEL algorithm with relative entropy weighting
   * - 11
     - ``unfold_doroshenko``
     - Iterative
     - `max_iterations`, `tolerance`, `regularization`
     - —
     - Coordinate-update iterative method
   * - 12
     - ``unfold_kaczmarz``
     - Iterative
     - `max_iterations`, `omega`, `tolerance`
     - —
     - ART (Algebraic Reconstruction Technique)
   * - 13
     - ``unfold_bayes``
     - Bayesian
     - `max_iterations`, `tolerance`
     - —
     - D'Agostini Bayesian iterative unfolding
   * - 14
     - ``unfold_bayes_spline_regularization``
     - Bayesian
     - `max_iterations`, `tolerance`, `spline_degree`, `spline_smooth`
     - —
     - Bayes iteration with spline smoothing
   * - 15
     - ``unfold_maxed``
     - MaxEnt
     - `sigma_factor`, `max_iterations`, `tolerance`
     - —
     - Maximum entropy deconvolution (Reginatto & Goldhagen)
   * - 16
     - ``unfold_statreg``
     - Statistical Reg.
     - `unfoldermethod` (EmpiricalBayes/...), `regularization`, `basis_name`, `boundary`, `derivative_degree`
     - —
     - Turchin's statistical regularization
   * - 17
     - ``unfold_reconst``
     - Statistical Reg.
     - `alpha`, `beta`, `max_iter_alpha`, `max_iter_beta`, `tol_alpha`, `tol_beta`
     - —
     - Fortran STREG1 port: auto α/β with discrepancy principle & ω-criterion
   * - 18
     - ``unfold_lmfit``
     - Optimization
     - `method` (lbfgsb/leastsq/...), `model_name` (elastic/lasso/ridge), `regularization`, `regularization2`, `l1_weight`
     - lmfit
     - L1/L2/Elastic Net via lmfit
   * - 19
     - ``unfold_scipy_direct_method``
     - Optimization
     - `method` (cg/gmres/lsqr/lsmr/minres), `tolerance`, `max_iterations`
     - —
     - Direct SciPy linear solvers
   * - 20
     - ``unfold_combined``
     - Pipeline
     - `pipeline` (list of `{method, params}` dicts)
     - —
     - Sequential multi-method pipeline
   * - 21
     - ``unfold_parametric``
     - Parametric
     - `parametric_method`, `optimizer`, `solver_backend`, `initial_params`
     - lmfit, cvxpy, qpsolvers
     - FRUIT-style thermal/epithermal/fast model
   * - 22
     - ``unfold_parametric_cvxpy``
     - Parametric
     - `parametric_method`, `initial_params`, `solver_backend`
     - cvxpy
     - SQP solver using cvxpy for parametric fitting
   * - 23
     - ``unfold_parametric_qpsolvers``
     - Parametric
     - `parametric_method`, `initial_params`, `solver_backend`
     - qpsolvers
     - SQP solver using qpsolvers backends
   * - 24
     - ``unfold_parametric_combined``
     - Parametric
     - `parametric_method`, `initial_params`, `solver_backend`
     - lmfit, cvxpy, qpsolvers
     - lmfit first-pass + QP refinement
   * - 25
     - ``unfold_parametric2``
     - Parametric
     - `b_range`, `Tf_range`, `c_range`, `noise_level`, `max_iter`, `tol_chi2`, `optimizer`, `solver_backend`
     - grid, cvxpy, qpsolvers, combined
     - BON95 4-component model + directed-divergence iterations
   * - 26
     - ``unfold_fruit_like``
     - Parametric
     - `initial_params`, `max_iterations`, `tolerance`
     - —
     - FRUIT-like model: Maxwellian thermal + 1/E epithermal + evaporation fast
   * - 27
     - ``unfold_hybrid_parametric``
     - Parametric
     - `refinement_method` (landweber/mlem), `max_iterations`, `tolerance`
     - —
     - Parametric initial guess refined by Landweber or MLEM
   * - 28
     - ``unfold_bayesian_parametric``
     - Parametric
     - `n_samples`, `burn_in`, `proposal_scale`, `prior_mean`, `prior_std`
     - —
     - Metropolis-Hastings MCMC for spectral parameter estimation
   * - 29
     - ``unfold_mystic``
     - Optimization
     - `regularization`, `norm` (1/2), `solver` (fmin/fmin_powell/diffev/diffev2), `maxiter`, `maxfun`, `smoothness_order`, `smoothness_weight`, `regularization_method`
     - mystic
     - Direct-search minimization of the penalized least-squares objective
   * - 30
     - ``unfold_smt``
     - Optimization
     - `nonneg`, `timeout_ms`
     - z3-solver
     - Exact SMT solving of `A·x = b` (integer/rational) with fluence minimization
   * - 31
     - ``unfold_genetic``
     - Optimization
     - `solver` (pso/ga/de/es/ep/abc/gwo/cmaes/nsga2), `epoch`, `pop_size`, `regularization`, `norm` (1/2), `smoothness_order`, `smoothness_weight`, `entropy_weight`, `n_runs`, `early_stop`, `half_range`, `two_step`, `n_coarse`, `smoother`, `sigma_smooth`, `crossover` (single/arithmetic), `mutation` (random/iterative), `pareto_select` (knee/min_residual/max_entropy)
     - mealpy
     - Population-based meta-heuristic unfolding (PSO/GA/DE/ES/EP/ABC/GWO/CMA-ES/NSGA-II), with an optional TGASU-style two-step coarse-to-fine scheme, NSGA-II Pareto selection, arithmetic crossover/iterative mutation and post-processing smoothers
   * - 32
     - ``unfold_cs``
     - Optimization
     - `n_atoms`, `sparsity`, `dictionary`, `n_dictionary_iterations`, `sigma_min`, `sigma_decrease_factor`, `mu_0`, `L`, `max_iterations`, `tolerance`
     - —
     - Compressive sensing: K-SVD dictionary + OMP sparse coding + SL0 reconstruction
   * - 33
     - ``unfold_scip``
     - Optimization
     - `regularization`, `norm` (1/2), `timeout`, `smoothness_order`, `smoothness_weight`, `nonneg`, `regularization_method`
     - pyscipopt
     - Tikhonov QP solved by the SCIP Optimization Suite (global NLP/QP optimizer)
   * - 34
     - ``unfold_docplex``
     - Optimization
     - `regularization`, `norm` (1/2), `timeout`, `smoothness_order`, `smoothness_weight`, `nonneg`, `regularization_method`
     - docplex, cplex
     - Tikhonov QP solved by IBM CPLEX via docplex.mp (CPLEX Community Edition)
   * - 35
     - ``unfold_epic``
     - Regularization
     - `target_sigmas`, `sigma_frac`, `regularization_order` (0/1/2), `non_neg`, `noise_var`, `homogeneous_step`, `regularize`, `beta_shift_k`, `beta_distance`, `EPIC_bool`, `V`, `LSQpar`
     - —
     - EPIC Tikhonov regularization (Ortega-Culaciati et al. 2021): prior variances chosen so a posteriori variances match target sigmas
   * - 36
     - ``unfold_interpret``
     - Interpretation
     - `regularization`, `norm` (1/2), `smoothness_order`, `smoothness_weight`, `enforce_norm`, `norm_value`, `regularization_method`, `interpret_options`
     - pyoptexplain (optional)
     - Unfolding QP solved via pyoptexplain plus an interpretation report (robustness, shadow prices, detector sensitivity, regularization sweep, scenarios). Also `Detector.interpret_result` for interpretation-only runs
   * - 37
     - ``unfold_cgls``
     - Krylov/iterative
     - `max_iterations`, `tolerance`, `regularization`, `smoothness_order`, `noise_level`
     - —
     - CGLS (conjugate gradient for least squares) with optional ``||L x||^2`` Tikhonov term and discrepancy-principle stopping; nonnegative spectrum via clamping
   * - 38
     - ``unfold_gks``
     - Krylov/hybrid
     - `regularization_method` (gcv/dp/lcurve/manual), `max_iterations`, `smoothness_order`, `regularization`, `noise_level`
     - —
     - Generalized Krylov Subspace (Golub-Kahan bidiagonalization + projected regularization selection); no a-priori spectrum required
   * - 39
     - ``unfold_tikhonov_tv``
     - Regularization
     - `epsilon`, `mu`, `max_iterations`, `type_` (TT/TV/T), `beta` (float or ``'adapt'``), `zthr`, `tolerance`, `noise_level`
     - —
     - Noise-constrained Tikhonov+TV via ADMM (Gazzola & Gholami); adaptive balancing of the TV and Tikhonov terms
   * - 40
     - ``unfold_sandii``
     - Multi-sphere ratio
     - `max_iterations`, `tolerance`, `chi_fac` (0/1), `relative_uncertainty`, `noise_level`
     - —
     - SAND-II geometric-mean ratio method (McElroy et al. 1967): chi-square or max-relative-deviation stopping
   * - 41
     - ``unfold_bunki``
     - Multi-sphere ratio
     - `smoothing`, `max_iterations`, `tolerance`, `noise_level`
     - —
     - BUNKI (SPUNIT) iterative unfolding with three-point smoothing (RSICC PSR-266)
   * - 42
     - ``unfold_bunkiut``
     - Multi-sphere ratio
     - `smoothing`, `max_iterations`, `tolerance`, `noise_level`
     - —
     - BUNKI-UT (BON31G) modernised unfolding (University of Texas)
   * - 43
     - ``unfold_osem``
     - EM family
     - `max_iterations`, `n_subsets`, `tolerance`, `noise_level`
     - —
     - Ordered-subset expectation maximisation (Hudson & Larkin 1994); ``n_subsets=1`` reduces to standard MLEM
   * - 44
     - ``unfold_mapem``
     - EM family
     - `prior` (none/quadratic/logcosh/relative_difference), `beta`, `prior_delta`, `gamma`, `max_iterations`, `tolerance`, `noise_level`
     - —
     - MAP-EM (OSMAPOSL one-step-late penalised EM) with nearest-neighbour priors over the energy axis
   * - 45
     - ``unfold_bsrem``
     - EM family
     - `prior` (none/quadratic/logcosh/relative_difference), `beta`, `prior_delta`, `gamma`, `max_iterations`, `n_subsets`, `tolerance`, `relaxation`, `addition_after_iteration`, `noise_level`
     - —
     - Block-sequential regularised EM with relaxation sequence and floor clamping (guaranteed convergence for non-convex priors)
   * - 46
     - ``unfold_sart``
     - Iterative
     - `max_iterations`, `tolerance`, `relaxation`, `noise_level`
     - —
     - Simultaneous algebraic reconstruction technique: relaxed, residual-normalised additive correction
   * - 47
     - ``unfold_ferdor``
     - Multi-sphere deconvolution
     - `max_iterations`, `tolerance`, `smoothing`, `chi_squared_target`, `relative_uncertainty`
     - —
     - FERDOR few-channel unfolding: constrained least squares with an automatically adjusted smoothing weight chosen by the discrepancy principle
   * - 48
     - ``unfold_rebunki``
     - Multi-sphere ratio
     - `smoothing`, `max_iterations`, `tolerance`
     - —
     - ReBUNKI (SPUNIT) few-iteration spectral stripping with three-point smoothing and ~1% convergence tolerance
   * - 49
     - ``unfold_nsduaz``
     - Multi-sphere ratio
     - `initial_spectrum`, `catalogue`, `use_catalogue`, `reference_name`, `smoothing`, `max_iterations`, `tolerance`
     - —
     - NSDUAZ unfolding: catalogue-selected initial spectrum (nuclear-data reference fluxes) refined by the SPUNIT iteration, with a flat-spectrum mode
   * - 50
     - ``unfold_fista``
     - Krylov/hybrid
     - `max_iterations`, `tolerance`, `regularization`, `l1_penalty`, `tv_penalty`, `nonnegativity`, `x_min`, `x_max`, `noise_level`, `eta`
     - —
     - FISTA (Fast Iterative Shrinkage-Thresholding Algorithm): accelerated proximal gradient method for L1/L2/TV regularized problems with box constraints; O(1/k²) convergence
   * - 51
     - ``unfold_hybrid_gmres``
     - Krylov/hybrid
     - `max_iterations`, `regularization_method`, `regularization`, `noise_level`, `eta`, `reorthogonalization`
     - —
     - Hybrid GMRES: combines GMRES iteration with Tikhonov regularization on projected problem; automatic regularization selection via GCV/discrepancy principle

.. note::

   **Common parameters** shared by most methods:
   ``readings``, ``initial_spectrum``, ``calculate_errors``, ``noise_level``,
   ``n_montecarlo``, ``save_result``, ``random_state``.

   See the :ref:`genindex` or :doc:`detector` for complete API signatures.

Built-in Response Functions
---------------------------

7 response function datasets are included as Python dicts, importable from the
package root:

.. list-table:: Built-in RF datasets
   :header-rows: 1
   :widths: 12 20 12 15 30

   * - Dataset
     - Source
     - Detectors
     - Energy Range
     - Notes
   * - ``RF_GSF``
     - GSF (Germany)
     - 10 (0in–18in)
     - 1e-9 – 631 MeV
     - Standard range
   * - ``RF_PTB``
     - PTB (Germany)
     - 15 (0in–18in)
     - 1e-9 – 631 MeV
     - Standard range
   * - ``RF_LANL``
     - LANL (USA)
     - 11 (3in–18in)
     - 1e-9 – 631 MeV
     - Includes Pb-shielded (9inPb, 12inPb, 18inPb)
   * - ``RF_JINR``
     - JINR (Dubna)
     - 9 (0in–12in)
     - 1e-9 – 631 MeV
     - Includes Cd-covered (Cd0in) and Pb-shielded (10inPb)
   * - ``RF_FERMILAB``
     - Fermilab (USA)
     - 8 (0in–18in)
     - 1e-9 – 631 MeV
     - Standard range
   * - ``RF_EURADOS``
     - EURADOS round-robin
     - 13 (0in–12in)
     - 1e-9 – 20 MeV
     - Narrower range; includes Cd2in, 3.5in, 4.5in
   * - ``RF_IHEP``
     - IHEP (Protvino)
     - 12 (0in–18in)
     - 1e-9 – 2000 MeV
     - Wider range; includes 15in

.. warning::

   ``RF_EURADOS`` max energy is 20 MeV and ``RF_IHEP`` max energy is 2000 MeV,
   compared to 631 MeV for the other datasets. Use caution when comparing
   results across datasets with different energy ranges.

.. code-block:: python

   from bssunfold import Detector, RF_JINR

   detector = Detector(RF_JINR)
   result = detector.unfold_cvxpy(readings, regularization=1e-4)


Dose Conversion Coefficients
----------------------------

4 dose conversion coefficient datasets are included for flexible dose rate
calculations. The default is ICRP-116 effective dose.

.. list-table:: Dose conversion coefficient datasets
   :header-rows: 1
   :widths: 20 15 25 20 20

   * - Dataset
     - Standard
     - Quantities
     - Energy Range
     - Notes
   * - ``ICRP116`` (default)
     - ICRP-116
     - AP, PA, LLAT, RLAT, ISO, ROT
     - 1e-9 – 631 MeV
     - Standard range
   * - ``ICRP74_effective``
     - ICRP-74
     - AP, PA, RLAT, ROT, ISO
     - 1e-9 – 398 MeV
     - Effective dose
   * - ``NRB99_2009_effective``
     - NRB99-2009
     - AP, ISO
     - 25 eV – 20 MeV
     - Limited range
   * - ``ICRP74_operational``
     - ICRP-74
     - ADE, PDE0, PDE45, PDE60, PDE75
     - 1e-9 – 398 MeV
     - Operational quantities

.. warning::

   ``NRB99_2009_effective`` covers a limited energy range (25 eV – 20 MeV).
   Values outside this range are set to zero during interpolation.

.. code-block:: python

   from bssunfold import Detector, get_coefficients, interpolate_coefficients

   # Set on Detector (affects all subsequent unfolds)
   detector = Detector(cc_type="ICRP74_effective")

   # Change after creation
   detector.set_dose_coefficients("ICRP74_operational")

   # Get coefficients directly for custom use
   cc = get_coefficients("NRB99_2009_effective")
   cc_interp = interpolate_coefficients(cc, detector.E_MeV)


Spectrum Comparison Metrics
---------------------------

25 metrics organised into 7 categories. All implemented with pure
NumPy/SciPy.

.. mermaid::

   graph TD
       A["Comparison Metrics"] --> B["Entropy"]
       A --> C["Distribution"]
       A --> D["Correlation"]
       A --> E["Error"]
       A --> F["Similarity"]
       A --> G["Chi-squared"]
       A --> H["Statistical"]

       B --> B1["kl_divergence"]
       B --> B2["cross_entropy"]
       B --> B3["entropy_difference_percent"]

       C --> C1["wasserstein_dist"]
       C --> C2["energy_dist"]
       C --> C3["kolmogorov_smirnov_stat"]

       D --> D1["pearson_r"]
       D --> D2["spearman_r"]

       E --> E1["mean_squared_error"]
       E --> E2["root_mean_squared_error"]
       E --> E3["mean_absolute_error"]
       E --> E4["mape"]
       E --> E5["r2_score"]
       E --> E6["max_error"]
       E --> E7["median_absolute_error"]

       F --> F1["cosine_similarity"]
       F --> F2["mmd_rbf"]

       G --> G1["chi_squared"]
       G --> G2["g_test"]
       G --> G3["freeman_tukey"]
       G --> G4["cressie_read"]

       H --> H1["anderson_darling"]
       H --> H2["wilcoxon_test"]
       H --> H3["mannwhitneyu_test"]
       H --> H4["standardized_mean_difference"]

       style A fill:#4a90d9,color:#fff

Metrics Reference
~~~~~~~~~~~~~~~~~

.. list-table:: Complete metrics reference
   :header-rows: 1
   :widths: 5 6 12 5

   * - Category
     - Metric Key
     - Description
     - Range
   * - Entropy
     - ``kl_divergence``
     - Kullback-Leibler divergence D_KL(p‖q)
     - [0, ∞)
   * -
     - ``cross_entropy``
     - Cross-entropy H(p,q) = -∑p·log(q)
     - [0, ∞)
   * -
     - ``entropy_difference_percent``
     - Relative cross-entropy excess (%)
     - [0, ∞)
   * - Distribution
     - ``wasserstein_dist``
     - Earth mover's / Wasserstein distance
     - [0, ∞)
   * -
     - ``energy_dist``
     - Energy distance between distributions
     - [0, ∞)
   * -
     - ``kolmogorov_smirnov_stat``
     - Kolmogorov-Smirnov D-statistic
     - [0, 1]
   * - Correlation
     - ``pearson_r``
     - Pearson correlation coefficient
     - [-1, 1]
   * -
     - ``spearman_r``
     - Spearman rank correlation
     - [-1, 1]
   * - Error
     - ``mean_squared_error``
     - Mean squared error
     - [0, ∞)
   * -
     - ``root_mean_squared_error``
     - Root mean squared error
     - [0, ∞)
   * -
     - ``mean_absolute_error``
     - Mean absolute error
     - [0, ∞)
   * -
     - ``mape``
     - Mean absolute percentage error (%)
     - [0, 100]
   * -
     - ``r2_score``
     - R² (coefficient of determination)
     - (-∞, 1]
   * -
     - ``max_error``
     - Maximum residual error
     - [0, ∞)
   * -
     - ``median_absolute_error``
     - Median absolute error
     - [0, ∞)
   * - Similarity
     - ``cosine_similarity``
     - Cosine similarity cos(θ)
     - [0, 1]
   * -
     - ``mmd_rbf``
     - Maximum Mean Discrepancy (RBF kernel)
     - [0, ∞)
   * - Chi-squared
     - ``chi_squared``
     - Pearson's chi-squared statistic
     - [0, ∞)
   * -
     - ``g_test``
     - G-test (log-likelihood ratio)
     - [0, ∞)
   * -
     - ``freeman_tukey``
     - Freeman-Tukey statistic
     - [0, ∞)
   * -
     - ``cressie_read``
     - Cressie-Read power divergence
     - [0, ∞)
   * - Statistical
     - ``anderson_darling``
     - Anderson-Darling k-sample statistic
     - [0, ∞)
   * -
     - ``wilcoxon_test``
     - Wilcoxon signed-rank test statistic
     - [0, ∞)
   * -
     - ``mannwhitneyu_test``
     - Mann-Whitney U test statistic
     - [0, ∞)
   * -
     - ``standardized_mean_difference``
     - Cohen's d (standardized mean difference)
     - (-∞, ∞)


Performance
-----------

All iterative solvers use Numba JIT-compiled inner loops when numba is installed,
with automatic fallback to pure Python.

.. list-table:: Benchmark results (60-bin grid, 500 iterations, macOS arm64)
   :header-rows: 1
   :widths: 20 15 15 15

   * - Solver
     - Before
     - After
     - Speedup
   * - Doroshenko
     - 40.6 ms
     - 0.8 ms
     - **50x**
   * - Kaczmarz
     - 1.4 ms
     - 0.1 ms
     - **14x**
   * - MLEM
     - 2.7 ms
     - 0.4 ms
     - **7x**
   * - GRAVEL
     - ~2 ms
     - 0.6 ms
     - **3x**
   * - cvxpy
     - 84 ms
     - 78 ms
     - ~1x (external solver)
   * - qpsolvers
     - 1.7 ms
     - 1.6 ms
     - ~1x (external solver)

Install numba for the best performance:

.. code-block:: bash

   pip install bssunfold[numba]

The JIT functions are defined in :mod:`bssunfold.core._numba_jit` and use
``@njit(cache=True)`` for automatic disk caching of compiled code.
