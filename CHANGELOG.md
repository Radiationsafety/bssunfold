# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],

and this project adheres to [Semantic Versioning].


## [0.16.0] - 2026-08-07

### Added
- **Quadratic-program interpretation with pyoptexplain** — new
  `unfold_interpret()` / `solve_interpret()` and the standalone
  `interpret_qp()` entry point that solve the same unfolding QP used by
  `unfold_qpsolvers`/`unfold_cvxpy` and then *interpret* the solution:
  - Solve report: solver status, objective value, per-group spectrum, residual
    per detector, active (zeroed) energy groups.
  - Shadow prices (duals) for the non-negativity bounds and, with
    `enforce_norm=True`, the norm-equality dual.
  - Robustness analysis (empirical `+/-1%..5%` perturbation sweep), detector
    informativeness (one-detector-at-a-time perturbation), regularization
    sweep, non-negativity trust, pyoptexplain what-if scenarios and a norm
    relaxation curve.
  - Output is an `InterpretationResult` with a Markdown `report`,
    JSON-friendly `metrics` and raw `tables` (pandas DataFrames).
  - Exposed on `Detector` as `unfold_interpret()` (standard result dict plus
    `report`/`interpretation_metrics`) and `interpret_result()` (interpretation
    only).
  - Optional dependency: `pip install bssunfold[interpret]` (pyoptexplain>=0.1.1).
  - Tests: `tests/test_interpret.py`.
- **Lanczos-hybrid (Krylov + GCV) unfolding** — new `unfold_lanczos()` method
  and `solve_lanczos()` solver. Performs Golub-Kahan (Lanczos-type)
  bidiagonalization, building a Krylov subspace in which a new approximation
  is computed at each iteration; the regularization parameter is selected
  automatically on the small projected problem by Generalized Cross Validation
  (GCV). No a-priori spectrum is required (pure NumPy/SciPy, no new deps).
  Supports discrepancy-principle early stopping via `noise_level` and is
  registered in `unfold_combined`. Tests: `tests/test_lanczos.py`.

## [0.15.0] - 2026-08-06

### Added
- **EPIC Tikhonov regularization unfolding** — new `unfold_epic()` method and
  `solve_epic()` solver (port of EPIC_LS, Ortega-Culaciati et al. 2021,
  https://github.com/frortega/EPIC_LS). Prior variances of the regularization
  operator are chosen so the a posteriori variances of the model parameters
  match target sigmas (Equal Posterior Information Condition); the weighted
  least-squares problem is then solved under optional non-negativity.
  - Defaults: first-derivative operator (`regularization_order=1`), target
    sigmas = `sigma_frac * max(|x_ls|)` with `sigma_frac=0.1`.
  - `EPIC_bool`, `V` (change of variables), `noise_var` (data covariance),
    `regularize` (minimum-norm damping) and `LSQpar` (solver tuning, incl.
    `tr_solver`) exposed for advanced use.
  - Registered in `unfold_combined()` pipelines (`"epic"`).
  - Fix: `create_derivative_matrix(order=1)` produced a rank-deficient
    operator (two `-1` per row with a stagger); rows now place `-1`/`+1` at
    the correct indices.
  - Tests: `tests/test_epic.py`.

## [0.14.1] - 2026-08-06

### Added
- **EURADOS integral-quantity comparison metrics** (following Gómez-Ros et al.,
  Radiat. Meas. 153 (2022) 106755) in `utils/comparison.py`:
  - `fluence_averaged_energy` — fluence-averaged energy Ē
  - `energy_group_fluence` — fluence rate in the thermal (E<0.4 eV),
    epithermal (0.4 eV–0.1 MeV) and fast (E>0.1 MeV) energy regions
  - `dose_averaged_energy` — ambient dose equivalent-averaged energy Ẽ
    (ISO 2001, ICRP-74 ADE coefficients)
  - `ambient_dose_equivalent_rate` — ambient dose equivalent rate H*(10)
  - New `_get_ade_cc()` helper resolves ICRP-74 operational coefficients by
    default (`get_coefficients("ICRP74_operational")`).
  - `compare_spectra()` now accepts the new single-spectrum metrics by name
    (`metrics="fluence_averaged_energy"`, etc.) and returns them under
    `_ref`/`_test` keys (energy-group fluence flattened to
    `energy_group_fluence_{thermal,epithermal,fast}_{ref,test}`); NaNs are
    reported when `energy` is missing or a metric raises.
  - Exported via `bssunfold.utils.__init__` and documented in
    `docs/detector.rst`.
- Tests: `tests/test_new_metrics.py`, edge cases in
  `tests/test_coverage_boost.py`, extended metric-key set in
  `tests/test_iaea_validation.py`.

### Fixed
- ICRP-74 operational dose-coefficient energy grid in `constants.py`
  (`ICRP74_COEFF_OPERATIONAL_QUANTITIES`): added the missing bin boundary
  node at 398.0 MeV (last bin now 398–630.957 MeV, 61 points total), with the
  coefficient value duplicated onto the 630.957 node. Previously the last
  coefficient applied to a narrower bin than the source table.
  - `tests/test_dose_coefficients.py` updated to the 61-point grid plus a
    duplicate-last-bin regression test.

## [0.14.0] - 2026-08-05

### Added
- **Mystic-based unfolding** — new `unfold_mystic()` method using the
  `mystic` constrained-optimization framework. Minimizes
  `||A·x − b||² + α·||x||_norm` with `x ≥ 0` via a quadratic penalty.
  Supports `norm` (L1/L2), smoothness constraints (order 1/2), multiple
  mystic solvers (`fmin`, `fmin_powell`, `diffev`, `diffev2`) and all
  regularization selection methods (manual/cosine/lcurve/gcv/dp).
  - New file: `core/unfold_mystic.py` (`solve_mystic` + `unfold_mystic`)
  - Optional dependency group: `bssunfold[mystic]` (`mystic>=0.4.5`)
  - Registered in `unfold_combined()` pipelines as `'mystic'`
  - 24 new tests in `tests/test_mystic.py`
- Static (bandit, pip-audit) and dynamic (DynaPyt) security analysis.
- **SMT-based unfolding** — new `unfold_smt()` method, a port of the
  Haskell/SBV `linearEqSolver` backed by the optional Z3 solver. Minimizes
  `||A·x − b||₁` and then the total fluence `Σx` over the non-negative
  orthant using Z3's optimizer, with exact solvers for integer and rational
  systems.
  - New file: `core/unfold_smt.py` (`solve_integer_linear_eqs`,
    `solve_integer_linear_eqs_all`, `solve_rational_linear_eqs`,
    `solve_rational_linear_eqs_all`, `solve_smt`, `unfold_smt`)
  - New `Detector.unfold_smt()` method
  - Registered in `unfold_combined()` pipelines as `'smt'`
  - Optional dependency group: `bssunfold[smt]` (`z3-solver>=4.13.0`)
  - New tests in `tests/test_smt.py`
- **Genetic / meta-heuristic unfolding** — new `unfold_genetic()` method
  using population-based meta-heuristic algorithms from MEALPY. Minimizes
  `||A·x − b||²/||b||² + α·||x||_norm` with `x ≥ 0` and optional
  second-difference smoothing and Shannon-entropy terms, following the
  PSO (Shahabinejad & Sohrabpour 2017), GA (Suman & Sarkar 2012) and
  entropy-based (Woo et al. 2019) unfolding works. No initial spectrum is
  required (random population initialization).
  - New file: `core/unfold_genetic.py` (`solve_genetic` + `unfold_genetic`)
  - 8 MEALPY solvers: `pso` (chaotic PSO, default), `ga`, `de`, `es`, `ep`,
    `abc`, `gwo`, `cmaes`
  - New `Detector.unfold_genetic()` method
  - Registered in `unfold_combined()` pipelines as `'genetic'`
  - Optional dependency group: `bssunfold[mealpy]` (`mealpy>=3.0.2`)
  - 30 new tests in `tests/test_genetic.py`
  - **scip and cplex**
  - **Compressive Sensing (CS) unfolding** — new `unfold_cs()` method based on
  compressive sensing. The spectrum is represented sparsely in a learned
  dictionary (`x = D @ alpha`), the dictionary is learned with **K-SVD**,
  sparse coding is performed with **OMP**, and reconstruction is done with the
  **SL0** algorithm. Well suited for the highly underdetermined problem where
  the number of energy groups greatly exceeds the number of detector readings.
  - New file: `core/unfold_cs.py` (`solve_omp`, `solve_ksvd`, `solve_sl0`,
    `solve_cs`, `unfold_cs`)
  - New `Detector.unfold_cs()` method
  - No extra dependencies (pure NumPy)
  - 21 new tests in `tests/test_cs.py`
  - New example notebook: `examples/23-CS.ipynb`

### Changed
- `numba` promoted to a core dependency (the `bssunfold[numba]` extra is kept
  for compatibility).
- Test-suite coverage gate raised to 98.3% (was 91.9%): numba JIT bodies are
  excluded from coverage via `# pragma: no cover` (compiled to LLVM, never run
  as CPython bytecode) and targeted branch tests added in
  `tests/test_coverage.py` and `tests/test_coverage_boost.py`.

## [0.13.0] - 2026-06-30

### Added
- **RECONST statistical regularization** — `unfold_reconst()` method, a direct numpy port
  of the FORTRAN STREG1 algorithm (Turchin/Vapnik, 1967). Solves
  `(B·β + Ω·α)·f = A_vec·β` with automatic α/β selection via discrepancy
  principle and ω-criterion. Supports manual α, β, and pp tuning parameters.
  - New file: `core/unfold_reconst.py`
  - New example notebook: `examples/20-RECONST.ipynb`
  - 52 new tests in `tests/test_reconst.py`

## [0.12.0] - 2026-06-29

### Added
- **MLEM with J-factor early stopping criterion** — new `unfold_mlem_stop()` method
  based on Montgomery et al. (2020), "A novel MLEM stopping criterion for unfolding
  neutron fluence spectra in radiation therapy", Nucl. Instrum. Meth. A 957, 163400.
  Uses J-factor + CPS crossover rule for automatic iteration termination
  (see `examples/19-MLEM_stopping_criteria.ipynb`).
- **Flexible column names** in `Detector.compare()`, `plot_with_uncertainty()`,
  `plot_comparison()` — arbitrary spectrum column names accepted (was hardcoded `'Phi'`)

### Changed
- Methods table in README updated to 26 methods (+ `unfold_mlem_stop`)

## [0.11.1] - 2026-06-26

### Changed 
- `numba` fix as optional dependency for conda-forge

## [0.11.0] - 2026-06-22

### Added
- **Numba JIT-compiled inner loops** (`_numba_jit.py`) for iterative solvers:
  - `@njit(cache=True)` compiled functions with automatic disk caching
  - Graceful fallback to pure Python when numba is not installed
  - JIT functions: `_doroshenko_inner`, `_kaczmarz_inner`, `_mlem_inner`, `_gravel_inner`, `_compute_log_steps_jit`, `_dose_weighted_mse_jit`
- `numba` added as optional dependency (>=0.65.1)

### Changed — Performance
- **Doroshenko solver**: **50x speedup** (40.6 ms → 0.8 ms) — element-wise inner loop eliminates per-coordinate numpy overhead
- **Kaczmarz solver**: **14x speedup** (1.4 ms → 0.1 ms) — JIT-compiled row update loop
- **MLEM solver**: **7x speedup** (2.7 ms → 0.4 ms) — JIT-compiled multiplicative update
- **GRAVEL solver**: **3x speedup** (~2 ms → 0.6 ms) — JIT-compiled weighted geometric mean update
- **Monte Carlo uncertainty**: pre-generates all noise vectors at once instead of per-sample dict creation
- **Comparison metrics**: `_compute_log_steps` and `dose_weighted_error` use JIT-compiled helpers when numba available

### Fixed
- `total_flux_ratio` returned `sum(reference)/sum(test)` instead of `sum(test)/sum(reference)` per docstring

### Improved
- Extracted `_compute_log_steps` DRY helper in `comparison.py` (was duplicated in 3 functions)
- Extracted `_handle_extrapolation` DRY helper in `interpolation.py` (was duplicated in 2 functions)
- 110 new tests in `tests/test_improvements.py` (validators, converters, matrix utils, Monte Carlo, dose calculation, interpolation, comparison metrics, EURADOS metrics, Detector integration)
- Test suite: 910 tests (was ~800)

## [0.10.0] - 2026-06-22

### Added
- **SQP-based parametric unfolding v2** (`unfold_parametric2.py`):
  - Alternative parametric unfolding implementation with SQP optimization
  - `Detector.unfold_parametric2()` method

## [0.9.1] - 2026-06-19

### Added
- **5-detector comparison** in dose rate evaluation scripts (`dose_rate_evaluation.py`, `dose_rate_iaea_compendium.py`):
  - Added JINR and FERMILAB to detector configurations (now 5: GSF, PTB, LANL, JINR, FERMILAB)
  - ISO scatter plots with per-detector color differentiation and legend
  - Updated evaluation reports with 5-detector results

## [0.9.0] - 2026-06-17

### Added
- **Built-in dose conversion coefficient datasets** (4 datasets):
  - `ICRP116` — ICRP-116 effective dose (AP, PA, LLAT, RLAT, ISO, ROT; 60 points, 1e-9 – 631 MeV)
  - `ICRP74_effective` — ICRP-74 effective dose (AP, PA, RLAT, ROT, ISO; 60 points, 1e-9 – 631 MeV)
  - `NRB99_2009_effective` — NRB99-2009 effective dose (AP, ISO; 24 points, 25 eV–20 MeV, limited range)
  - `ICRP74_operational` — ICRP-74 operational quantities (ADE, PDE0, PDE45, PDE60, PDE75; 60 points, 1e-9 – 631 MeV)
  - `get_coefficients(name)` — lookup coefficient datasets by string key
  - `interpolate_coefficients(cc, E_target)` — interpolate coefficients to detector energy grid
  - `Detector(cc_type=...)` — select dose coefficients at construction time
  - `Detector.set_dose_coefficients(name)` — change dose coefficients after construction
  - Exported from `bssunfold` package root
- **Built-in response function  datasets** (7 datasets from CSV sources):
  - `RF_JINR` — JINR (Dubna): 9 detectors, 60 energy bins (1e-9–631 MeV)
  - `RF_FERMILAB` — Fermilab: 8 detectors, 60 energy bins (1e-9–631 MeV)
  - `RF_EURADOS` — EURADOS round-robin: 13 detectors, 105 energy bins (1e-9 – 20 MeV, narrower range)
  - Exported from `bssunfold` package root alongside `RF_GSF`, `RF_PTB`, `RF_LANL`
- **SQP-based parametric unfolding** (`unfold_parametric.py`):

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