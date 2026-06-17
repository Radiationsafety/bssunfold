# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],

and this project adheres to [Semantic Versioning].


## [0.8.1] - 2026-06-17

### Added
- **Built-in dose conversion coefficient datasets** (4 datasets):
  - `ICRP116` — ICRP-116 effective dose (AP, PA, LLAT, RLAT, ISO, ROT; 60 points, 1e-9 – 631 MeV)
  - `ICRP74_effective` — ICRP-74 effective dose (AP, PA, RLAT, ROT, ISO; 60 points, 1e-9 – 398 MeV)
  - `NRB99_2009_effective` — NRB99-2009 effective dose (AP, ISO; 24 points, 25 eV–20 MeV, limited range)
  - `ICRP74_operational` — ICRP-74 operational quantities (ADE, PDE0, PDE45, PDE60, PDE75; 60 points, 1e-9–398 MeV)
  - `get_coefficients(name)` — lookup coefficient datasets by string key
  - `interpolate_coefficients(cc, E_target)` — interpolate coefficients to detector energy grid
  - `Detector(cc_type=...)` — select dose coefficients at construction time
  - `Detector.set_dose_coefficients(name)` — change dose coefficients after construction
  - Exported from `bssunfold` package root
- **Built-in response function datasets** (7 datasets from CSV sources):
  - `RF_JINR` — JINR (Dubna): 9 detectors, 60 energy bins (1e-9–631 MeV)
  - `RF_FERMILAB` — Fermilab: 8 detectors, 60 energy bins (1e-9–631 MeV)
  - `RF_EURADOS` — EURADOS round-robin: 13 detectors, 105 energy bins (1e-9–20 MeV, narrower range)
  - `RF_IHEP` — IHEP (Protvino): 12 detectors, 107 energy bins (1e-9–2000 MeV, wider range)
  - Special detectors documented: Cd-covered (Cd0in, Cd2in), lead-shielded (10inPb)
  - Exported from `bssunfold` package root alongside `RF_GSF`, `RF_PTB`, `RF_LANL`
- **SQP-based parametric unfolding** (`unfold_parametric.py`):
  - `unfold_parametric_cvxpy` — Sequential Quadratic Programming solver using cvxpy (ECOS/MOSEK/SCS)
  - `unfold_parametric_qpsolvers` — SQP solver using qpsolvers backends (OSQP, GUROBI, etc.)
  - `unfold_parametric_combined` — lmfit first-pass + QP refinement (cvxpy or qpsolvers)
  - Numerical Jacobian with bound-aware clamping for SQP linearization
  - Brute-force grid scan (`_find_initial_params`) for robust initial parameter estimation
  - Fit quality warning when residual exceeds 10x the readings norm
  - Unified `solver_backend` parameter format: `"auto"`, `"cvxpy"`, `"cvxpy:ECOS"`, `"qpsolvers"`, `"qpsolvers:osqp"`
- Simplified parameter interface: replaced 6 params (`cvxpy_solver`, `qpsolver_name`, `qp_solver`, `norm`, `smoothness_order`, `smoothness_weight`) with single `solver_backend` string

### Fixed
- Combined method no longer re-runs lmfit redundantly after QP refinement
- Jacobian perturbations now clamped within parameter bounds (backward difference at boundaries)
- SQP penalty corrected from `α||Jδ + s_k||²` to `α||δ||²` (regularizes parameter updates, not spectrum values)
- Brute-force scan finds better starting point for fast-dominated spectra (e.g., Cf-252)

### Changed
- Test suite: 632 tests (was 46 parametric-specific tests)
- Updated docs: Sphinx API, README method table, Mermaid diagrams, examples

### Security
- Updated `pillow` 12.1.0 → 12.2.0 (CVE-2026-25990: out-of-bounds write via crafted PSD image)
- Updated `pygments` 2.19.2 → 2.20.0 (CVE-2026-4539: DoS via inefficient regex in AdlLexer)
- Updated `pytest` 9.0.2 → 9.1.0 (CVE-2025-71176: insecure temporary directory handling)

## [0.8.0] - 2026-06-15

### Added
- **EURADOS-style spectrum comparison metrics** (`comparison.py`):
  - `fluence_difference_percent` — relative difference in total fluence (%)
  - `energy_group_fluence_diff` — fluence difference by energy groups (thermal / epithermal / fast)
  - `dose_difference_percent` — relative difference in H*(10) (%)
  - `fluence_averaged_energy_diff` — difference in fluence-averaged energy
  - `dose_averaged_energy_diff` — difference in H*(10)-averaged energy
  - `spectral_shape_similarity` — cosine similarity of unit-normalized spectra
  - `log_lethargy_correlation` — Pearson correlation in E·Φ(E) lethargy coordinates
  - `peak_location_error` — relative error in peak energy position (%)
  - `peak_width_error` — relative error in peak FWHM (%)
  - `dose_weighted_error` — dose-weighted root mean squared error
  - `response_matrix_consistency` — χ² consistency between spectrum and readings
- **FRUIT-based parametric unfolding** (`unfold_parametric.py`):
  - Parametric spectrum as weighted sum of thermal, epithermal, and fast components (FRUIT model)
  - `Detector.unfold_parametric()` method
- **FRUIT-like parametric unfolding** (`unfold_fruit_like.py`):
  - Parametric model: Maxwellian thermal + 1/E epithermal + evaporation fast spectrum
  - `Detector.unfold_fruit_like()` method
- **Hybrid parametric-nonparametric unfolding** (`unfold_hybrid_parametric.py`):
  - Parametric initial guess refined by Landweber or MLEM iteration
  - `Detector.unfold_hybrid_parametric()` method
- **Bayesian parametric unfolding** (`unfold_bayesian_parametric.py`):
  - Metropolis-Hastings MCMC sampling for spectral parameter estimation
  - `Detector.unfold_bayesian_parametric()` method
- 24 new tests in `tests/test_new_metrics.py`


## [0.7.0] - 2026-06-08

### Added
- Comparison metrics: 'kl_divergence', 'cross_entropy', 'entropy_difference_percent', 'wasserstein_dist', 'energy_dist', 'kolmogorov_smirnov_stat', 'pearson_r', 'spearman_r','mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'mape','r2_score', 'max_error', 'median_absolute_error', 'cosine_similarity', 'mmd_rbf', 'chi_squared', 'g_test', 'freeman_tukey', 'cressie_read', 'anderson_darling', 'standardized_mean_difference', 'wilcoxon_test', 'mannwhitneyu_test'
- ipynb example 15


## [0.6.0] - 2026-06-04

### Added
- TSVD
- **Bayesian**: D'Agostini iterative (Bayes), Bayes with spline regularization
- **Maximum Entropy**: MAXED (primal log-space dual minimisation)
- **Statistical Regularization**: Turchin's method (StatReg)
- ipynb examples 12,13,14

### Changed
 - file structure of the project

## [0.5.0] - 2026-06-04

### Added
- github actions
- github releases

### Changed
- `solvers-jax` group — now includes `solvers-core` + `solvers-jax`.`proxsuite` and `open-source-solvers` have been removed from core dependencies. `qpsolvers[open-source-solvers]` was pulling `proxsuite` as a required dependency, which is not available on Windows, causing the package to fail installation.

## [0.4.1] - 2026-03-17
### Added
 - qpsolvers: smoothness with 1st and 2nd derivatives
 - 11-QP_solvers_smooth.ipynb example for qpsolvers smooth
 - lmfit initial_spectrum

## [0.4.0] - 2026-03-16
### Added
 - Doroshenko iterative method
 - Karcmarz algorithm
 - lmfit package 
 - examples 9-10 for new methods
 - error bar with std for plot_with_uncertainty function

  ### Changed
 - python 3.14 not supported because of proxsuite==0.7.2

## [0.3.0] - 2026-03-11
### Added
 - qpsolvers for QP open source solvers
 - combined algorithm
 - examples 6-8 for combined algorithm, plot with uncertainty, qpsolvers
 - plot_with_uncertainty function
 - save figure with response functions to png, pdf, eps, jpg
 - automatic selection of regularization parameter via pytikhonov package

 ### Changed
 - docs updated

## [0.2.0] - 2026-02-02
### Added
 - mlem algorithm via ODL, with example

## [0.1.3] - 2026-01-15

### Added
 - RF_PTB  in constants (response function for PTB BSS)
 - RF_LANL in constants (response function for LANL BSS)

### Changed
 - numpy 2.0.2 for micropip in marimo


## [0.1.2] - 2026-01-14

### Added
 - conda recipe

### Changed
 - pandas 2.3.3 for micropip in marimo
 - readme.md


## [0.1.1] - 2026-01-12

### Added
 - shields 
 - Citation.cff
 - Codeowners
 - Code of conduct
 - Response functions as a dict to the constants. 
 - github workflows

### Changed
 - 01 basic example


## [0.1.0] - 2025-12-25

- initial release

### Added
- Landweber iterative method
- Tikhonov regularization with CVXPY
- docs
- example
- simple tests

<!-- Links -->
[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->
<!-- [unreleased]: https://github.com/Author/Repository/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Author/Repository/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Author/Repository/releases/tag/v0.0.1 -->

<!-- ### Changed

### Deprecated

### Removed

### Fixed

### Security
 -->