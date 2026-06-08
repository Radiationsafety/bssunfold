Package Overview
================

BSSUnfold is a Python package for neutron spectrum unfolding from Bonner Sphere
Spectrometers (BSS). It provides 17 unfolding algorithms, 25 spectrum
comparison metrics, ICRP-116 dose calculations, and Monte Carlo uncertainty
quantification.

.. contents::
   :local:
   :depth: 2

Unfolding Methods
-----------------

All 17 methods are accessible as instance methods on the
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

       B --> B1["unfold_cvxpy"]
       B --> B2["unfold_qpsolvers"]
       B --> B3["unfold_tsvd"]
       B --> B4["unfold_tikhonov_legendre"]

       C --> C1["unfold_landweber"]
       C --> C2["unfold_mlem"]
       C --> C3["unfold_mlem_odl"]
       C --> C4["unfold_gravel"]
       C --> C5["unfold_doroshenko"]
       C --> C6["unfold_kaczmarz"]

       D --> D1["unfold_bayes"]
       D --> D2["unfold_bayes_spline_regularization"]

       E --> E1["unfold_maxed"]
       F --> F1["unfold_statreg"]

       G --> G1["unfold_lmfit"]
       G --> G2["unfold_scipy_direct_method"]

       H --> H1["unfold_combined"]

       style A fill:#4a90d9,color:#fff
       style B fill:#e8f0fe
       style C fill:#e8f0fe
       style D fill:#e8f0fe
       style E fill:#e8f0fe
       style F fill:#e8f0fe
       style G fill:#e8f0fe
       style H fill:#e8f0fe

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
     - ``regularization``, ``norm`` (1/2), ``solver``, ``regularization_method``
     - cvxpy
     - Convex optimization with Tikhonov regularization
   * - 2
     - ``unfold_qpsolvers``
     - Tikhonov
     - ``regularization``, ``norm`` (1/2), ``solver``, ``smoothness_order``, ``smoothness_weight``, ``regularization_method``
     - qpsolvers
     - QP-based unfolding with L1/L2/smoothness norms
   * - 3
     - ``unfold_tsvd``
     - Tikhonov
     - ``method`` (l_curve/gcv/discrepancy/energy/median/donoho), ``k``, ``threshold``, ``noise_level``
     - —
     - Truncated SVD with automatic k-selection
   * - 4
     - ``unfold_tikhonov_legendre``
     - Tikhonov
     - ``delta``, ``n_polynomials``
     - —
     - Tikhonov regularization in Legendre polynomial basis
   * - 5
     - ``unfold_landweber``
     - Iterative
     - ``max_iterations``, ``tolerance``
     - —
     - Landweber fixed-point iteration
   * - 6
     - ``unfold_mlem``
     - Iterative
     - ``max_iterations``, ``tolerance``
     - —
     - Pure-NumPy MLEM (expectation maximization)
   * - 7
     - ``unfold_mlem_odl``
     - Iterative
     - ``max_iterations``, ``tolerance``
     - odl
     - MLEM via ODL operator framework
   * - 8
     - ``unfold_gravel``
     - Iterative
     - ``max_iterations``, ``tolerance``, ``regularization``
     - —
     - GRAVEL with relative entropy weighting
   * - 9
     - ``unfold_doroshenko``
     - Iterative
     - ``max_iterations``, ``tolerance``, ``regularization``
     - —
     - Coordinate-update iterative method
   * - 10
     - ``unfold_kaczmarz``
     - Iterative
     - ``max_iterations``, ``omega``, ``tolerance``
     - —
     - ART (Algebraic Reconstruction Technique)
   * - 11
     - ``unfold_bayes``
     - Bayesian
     - ``max_iterations``, ``tolerance``
     - —
     - D'Agostini Bayesian iterative unfolding
   * - 12
     - ``unfold_bayes_spline_regularization``
     - Bayesian
     - ``max_iterations``, ``tolerance``, ``spline_degree``, ``spline_smooth``
     - —
     - Bayes with spline smoothing on log10-spectrum
   * - 13
     - ``unfold_maxed``
     - MaxEnt
     - ``sigma_factor``, ``max_iterations``, ``tolerance``
     - —
     - Maximum entropy deconvolution
   * - 14
     - ``unfold_statreg``
     - Statistical Reg.
     - ``unfoldermethod`` (EmpiricalBayes/...), ``regularization``, ``basis_name``, ``boundary``, ``derivative_degree``
     - —
     - Turchin's statistical regularization
   * - 15
     - ``unfold_lmfit``
     - Optimization
     - ``method`` (lbfgsb/leastsq/...), ``model_name`` (elastic/lasso/ridge), ``regularization``, ``regularization2``, ``l1_weight``
     - lmfit
     - L1/L2/Elastic Net via lmfit
   * - 16
     - ``unfold_scipy_direct_method``
     - Optimization
     - ``method`` (cg/gmres/lsqr/lsmr/minres), ``tolerance``, ``max_iterations``
     - —
     - Direct SciPy linear solvers
   * - 17
     - ``unfold_combined``
     - Pipeline
     - ``pipeline`` (list of ``{"method", "params"}`` dicts)
     - —
     - Sequential multi-method pipeline

.. note::

   **Common parameters** shared by most methods:
   ``readings``, ``initial_spectrum``, ``calculate_errors``, ``noise_level``,
   ``n_montecarlo``, ``save_result``, ``random_state``.

   See the :ref:`genindex` or :doc:`detector` for complete API signatures.

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
