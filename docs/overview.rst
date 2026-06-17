Package Overview
================

BSSUnfold is a Python package for neutron spectrum unfolding from Bonner Sphere
Spectrometers (BSS). It provides 21 unfolding algorithms, 25 spectrum
comparison metrics, ICRP-116 dose calculations, and Monte Carlo uncertainty
quantification.

.. contents::
   :local:
   :depth: 2

Unfolding Methods
-----------------

All 21 methods are accessible as instance methods on the
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

       I --> I1["unfold_parametric"]
       I --> I2["unfold_parametric_cvxpy"]
       I --> I3["unfold_parametric_qpsolvers"]
       I --> I4["unfold_parametric_combined"]

       style A fill:#4a90d9,color:#fff
       style B fill:#e8f0fe
       style C fill:#e8f0fe
       style D fill:#e8f0fe
       style E fill:#e8f0fe
       style F fill:#e8f0fe
       style G fill:#e8f0fe
       style H fill:#e8f0fe
       style I fill:#e8f0fe

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
   * - 18
     - ``unfold_parametric``
     - Parametric
     - ``parametric_method`` (thermal/epithermal/fast/custom), ``optimizer`` (lmfit/cvxpy/qpsolvers/combined), ``solver_backend``, ``initial_params``, ``max_iter``, ``tolerance``
     - lmfit, cvxpy, qpsolvers
     - FRUIT-style parametric spectrum model (thermal + epithermal + fast)
   * - 19
     - ``unfold_parametric_cvxpy``
     - Parametric
     - ``parametric_method``, ``initial_params``, ``max_iter``, ``tolerance``, ``solver_backend``
     - cvxpy
     - SQP solver using cvxpy for parametric model fitting
   * - 20
     - ``unfold_parametric_qpsolvers``
     - Parametric
     - ``parametric_method``, ``initial_params``, ``max_iter``, ``tolerance``, ``solver_backend``
     - qpsolvers
     - SQP solver using qpsolvers backends for parametric model fitting
   * - 21
     - ``unfold_parametric_combined``
     - Parametric
     - ``parametric_method``, ``initial_params``, ``max_iter``, ``tolerance``, ``solver_backend``
     - lmfit, cvxpy, qpsolvers
     - lmfit first-pass + QP refinement for parametric model

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
