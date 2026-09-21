# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],

and this project adheres to [Semantic Versioning].

## [0.28.0] - 2026-09-21

### Added

- **OSEM-ANLM unfolding method** (`unfold_osem_anlm` / `solve_osem_anlm`):
  port of the OSEM-ANLM algorithm of Jamaati et al. (2026), "Enhanced
  sparse view CT reconstruction using ordered subset expectation
  maximization and asymptotic non-local means algorithms", Scientific
  Reports (https://doi.org/10.1038/s41598-026-70607-1), adapted to neutron
  spectrum unfolding. Ordered-subset EM updates interleaved with the
  two-stage asymptotic non-local means (ANLM) filter applied after every
  subset update (article pseudo-code steps 4-5), or once to the OSEM
  result (`anlm_mode='post'`). The ANLM filter implements the article's
  two-stage scheme: stage 1 with the uniform parameter
  `h1 = 0.5 * sigma`, stage 2 with the point-wise parameter
  `h2(i) = sqrt(sum_j w(i,j)^2 * sigma^2)` (article eq. 6); the NLM
  weights use the Gaussian-weighted patch distances over the search
  window `N` (default 11) and similarity window `nu` (default 3, article
  optima). The noise level `sigma` is either user-supplied (`h`) or
  estimated automatically with a robust MAD estimator on second
  differences (`estimate_noise_1d`); by default the filter operates in
  log space (`log_space=True`), making it scale-free for spectra spanning
  orders of magnitude (`log_space=False` reproduces the raw-unit CT
  formulation). New file `core/unfold_osem_anlm.py` with the standalone
  1D filter `anlm_filter_1d`.
- New `Detector.unfold_osem_anlm()` method; `solve_osem_anlm`,
  `unfold_osem_anlm`, `anlm_filter_1d` and `estimate_noise_1d` registered
  in `core/__init__.py`.
- New test file `tests/test_osem_anlm.py` (38 tests): filter identity and
  smoothing properties, noise estimator, solver equivalence with plain
  OSEM for the identity filter, `post`-mode composition
  `solve_osem_anlm(post) == anlm_filter_1d(solve_osem(...))`, subset
  variants, validation errors and `Detector` wrapper coverage.
- **LOUHI78 unfolding** (`unfold_louhi` / `solve_louhi`, Routti & Sandberg
  1980, Computer Physics Communications 21,
  doi:10.1016/0010-4655(80)90021-4): constrained weighted least squares
  with generalized smoothing — the classic Bonner-sphere unfolding
  program, ported as
  `min ||(b - A phi)/sigma||^2 + lambda^2 ||L (phi - phi0)||^2` s.t.
  `phi >= 0`:
  - The quadratic program is solved by Hildreth's iterative coordinate
    algorithm (the LSI step of LOUHI78) with a relative-objective-change
    stopping rule; the default spectrum `phi0` anchors both the
    smoothing term and the starting point.
  - Generalized smoothing operators (new file `core/unfold_louhi.py`,
    helper `louhi_smoothing_matrix`): order 0 shrinks the solution
    toward the a-priori spectrum, orders 1/2 penalize first/second
    differences of the deviation from the a-priori.
  - Nonlinear regression mode (`auto_smooth=True`): the smoothing weight
    is adjusted automatically by a golden-section search on
    `log10(lambda)` (reusing the `core/_line_search.py` building blocks)
    until the data chi-square reaches `chi2_target` (default: the number
    of detectors, i.e. the expected chi-square value).
  - Statistical error propagation (`louhi_covariance`): inverse Hessian
    on the free (strictly positive) bins of the active set, as in the
    LOUHI78 error report.
  - New `Detector.unfold_louhi()` method; registered in
    `core/__init__.py`.
- New tests in `tests/test_louhi.py` (42 tests covering the smoothing
  operators, the Hildreth QP core (interior-point equivalence and
  projection behavior), the automatic smoothing regression, the error
  propagation, parameter validation and the Detector integration).
- New example notebook `examples/79-louhi-iaea.ipynb` (LOUHI unfolding of
  an IAEA Compendium benchmark spectrum with the linear and nonlinear
  smoothing modes) and `examples/80-osem-anlm-iaea.ipynb` (OSEM-ANLM in
  the `subset`/`post` modes, the exact `post == OSEM + anlm_filter_1d`
  composition property and the `h` smoothing-strength knob); every
  public `unfold_*` method now has a dedicated IAEA benchmark notebook.
- **IAEA-Compendium benchmark notebook suite** — 23 new example
  notebooks (`examples/56-gnowee-iaea.ipynb` through
  `examples/78-ensemble-iaea.ipynb`) for every public `unfold_*` method
  that previously lacked a dedicated example (gnowee, maeo, nnqp, qpmad,
  zfit, iterative_refinement, odl_pdhg, amaxed_regularization,
  crystal_ball, directed_divergence, express, ferdor, imaxed,
  mystic_hybrid, nsduaz, odl_douglas_rachford, qubo, rebunki, rfsp_jul,
  scipy_direct_method, staysl, tikhonov_legendre, ensemble). Each
  notebook follows the same template: build a `Detector` from the
  built-in `RF_GSF` response functions, fold an IAEA Compendium
  reference spectrum into readings, unfold, compute spectral-distance
  metrics (`utils.comparison.compare_spectra`), plot and sweep across
  several IAEA spectra. Documented in `docs/examples.rst`.
- New API test module `tests/test_all_unfold_methods_api.py`
  (~250 tests): every public `unfold_*` must be importable from
  `bssunfold.core`, accept the canonical
  `(readings, initial_spectrum, ...)` signature plus the common
  Monte-Carlo/housekeeping kwargs, return the standardized result keys
  on a smoke reading set, accept every documented kwarg non-default and
  reject unknown kwargs, run end-to-end on the IAEA benchmark, plus
  headless nbconvert execution of the new notebooks (auto-skipped when
  an optional backend or `jupyter nbconvert` is unavailable).
- Test-infrastructure fixes: `tests/test_smt.py` defers `import z3`
  behind `pytest.importorskip("z3")` (collection no longer breaks when
  the optional `z3-solver` package is not installed) and the `qubo` /
  `zfit` pytest markers are registered in `pyproject.toml`.

## [0.26.0] - 2026-09-18

### Added

- **Optimization-course method suite** — a family of eight new unfolding
  methods ported from the MIPT "Optimization Methods in Machine Learning"
  course curriculum (lectures 1-15, homeworks 1/7/8/9/10/11/12/13/14/15/16/18/20),
  filling the main algorithmic gaps of the package:
  - **Projected gradient descent** (`unfold_pgd` / `solve_pgd`, lecture 9 /
    homework 14): gradient step + Euclidean projection onto the nonnegative
    orthant, box or fluence simplex (exact total-fluence preservation),
    optional Armijo backtracking, `project_onto_set` exported; the result
    carries an NNLS Lagrange-duality-gap optimality certificate
    (`duality_gap` key). New file `core/unfold_pgd.py`.
  - **Frank-Wolfe conditional gradient** (`unfold_frank_wolfe` /
    `solve_frank_wolfe`, lecture 9): linear minimization oracle over the
    fluence simplex, Wolfe away-steps, exact quadratic line search or
    Armijo backtracking, Frank-Wolfe (duality) gap stopping. New file
    `core/unfold_frank_wolfe.py`.
  - **Mirror descent** (`unfold_mirror_descent` / `solve_mirror_descent`,
    lecture 10 / homework 16): Bregman-geometry descent with entropy
    (multiplicative updates generalizing MLEM/GRAVEL/SAND-II, fluence
    preserved), log-barrier, L2 and p-norm mirror maps; per-iteration
    golden-section line search along the mirror trajectory with
    orthant-interior step guards. New file `core/unfold_mirror_descent.py`.
  - **Consensus ADMM** (`unfold_admm` / `solve_admm`, lecture 11 /
    homework 18): `min 1/2||Ax-b||^2 + l1||x||_1 + tv||Dx||_1 s.t. x >= 0`
    via an exact NNLS x-update on the augmented system, soft-thresholding
    z-updates and scaled dual ascent; Boyd primal/dual-residual stopping;
    adaptive `rho` (Boyd sec. 3.4.1). New file `core/unfold_admm.py`
    (`soft_threshold` exported).
  - **L-BFGS-B quasi-Newton** (`unfold_lbfgsb` / `solve_lbfgsb`, lecture 7 /
    homework 10): scipy L-BFGS-B with analytic gradients, box bounds,
    Tikhonov L2 plus a second-difference (curvature) smoothing term
    (`second_difference_matrix` exported). New file `core/unfold_lbfgsb.py`.
  - **Coordinate descent** (`unfold_coordinate_descent` /
    `solve_coordinate_descent`, lecture 15): exact closed-form coordinate
    minimization of the NNLS + L1/L2 objective with O(m) per-coordinate
    residual updates, cyclic or seeded random order. New file
    `core/unfold_coordinate_descent.py`.
  - **Subgradient methods** (`unfold_subgradient` / `solve_subgradient`,
    lecture 8 / homework 12): projected subgradient descent for nonsmooth
    L1/TV objectives with Polyak, diminishing (square-summable) and fixed
    step-size policies; best-iterate return. New file
    `core/unfold_subgradient.py`.
  - **Extragradient** (`unfold_extragradient` / `solve_extragradient`,
    lecture 13 / homework 20): Korpelevich's two-step method on the robust
    saddle formulation `min_{x>=0} max_{||y||<=1} 1/2||Ax-b||^2 +
    delta y^T(Ax-b)` with `delta = noise_level||b||_2` (least squares made
    robust against bounded measurement noise). New file
    `core/unfold_extragradient.py`.
  - New `Detector.unfold_pgd()`, `unfold_mirror_descent()`,
    `unfold_frank_wolfe()`, `unfold_admm()`, `unfold_lbfgsb()`,
    `unfold_coordinate_descent()`, `unfold_subgradient()` and
    `unfold_extragradient()` methods; all registered in `core/__init__.py`.
- **1D optimization building blocks** (lecture 1 / homework 1): new file
  `core/_line_search.py` with `golden_section_minimize`,
  `dichotomy_minimize`, `brent_minimize` (Netlib-fmin-style Brent with
  inverse-quadratic interpolation and golden-section fallback) and
  `backtracking_line_search` (Armijo), all exported from `core`.
- **1D regularization-parameter search** (`select_regularization_1d`):
  new file `core/regularization_1d.py` — minimizes GCV (with Hutchinson
  randomized effective-DOF estimation), Morozov discrepancy or predictive
  risk over `log10(lambda)` using golden-section / dichotomy / Brent
  searches; works with the built-in Tikhonov-NNLS family or any
  user-supplied solver.
- **Monte-Carlo variance reduction** (lecture 14): `monte_carlo_uncertainty`
  now supports `variance_reduction` = `'antithetic'` (paired ±noise draws,
  halving solver calls and cancelling the odd response part), `'control'`
  (delta-method linearized response `pinv(A) db` as control variate with
  known mean and per-bin regression coefficients; reports
  `variance_reduction_factor`) and `'both'`; threaded through
  `run_unfolding` and available on all new Detector methods via the new
  `variance_reduction` parameter.
- **Duality / KKT diagnostics** (seminars 9-10): new file
  `core/_dual_diagnostics.py` with `nnls_duality_gap` (dual-feasible
  multiplier reconstruction, pseudo-inverse dual objective, absolute and
  relative gap certificates) and `nnls_kkt_residuals` (stationarity,
  complementarity, feasibility and active-set mask); exported from `core`
  and attached to PGD results.
- New tests in `tests/test_optimization_methods.py` (75 tests covering the
  1D minimizers, all eight solvers, projections, the regularization search,
  the duality diagnostics, the variance-reduced Monte-Carlo and the
  Detector integration).
- New example notebooks: `examples/55-optimization-course-methods.ipynb`
  (solver comparison, duality certificates, 1-D regularization selection),
  `examples/53-montecarlo-20-spectra.ipynb` (20 Monte-Carlo spectra per
  solver, variance reduction) and `examples/54-noise-robustness.ipynb`
  (noise-robustness study with a bias-variance decomposition, 11 solvers).

### Fixed
- **Default total-fluence estimate in Frank-Wolfe and entropy mirror
  descent.** The previous fallback
  `mean(b) / mean(A) * n_bins` over-estimated the simplex level by orders
  of magnitude on typical log-spaced GSF grids (215 vs the correct 2.25
  for the ISO Cf-252 example), pinning both solvers to a wrong scale with
  a large residual and strongly biased dose rates. The default is now a
  data-driven NNLS estimate `estimate_total_fluence(A, b)` (new helper in
  `_matrix_utils`), with the old heuristic kept only as a fallback when
  the NNLS solve fails.

## [0.25.0] - 2026-09-17

### Added
- **NNQP unfolding** (`unfold_nnqp` / `solve_nnqp`, Giovannucci & Pehlevan,
  https://github.com/simonsfoundation/NNQP). Solves the regularised
  non-negative least-squares problem
  ``min 0.5 ||A x - b||^2 + α/2 ||L x||^2 + α0/2 ||x||^2  s.t.  x >= 0``
  by recasting it as the NNQP ``min 0.5 x^T Q x + f^T x s.t. x >= 0``
  with ``Q = A^T A + α L^T L + α0 I`` and ``f = -A^T b``, then applying
  the coordinate-descent NNQP solver. Pure-NumPy port of the original
  `nnqp.py` (which used ``numba``); drops the ``numba`` hard-dependency
  in favour of a vectorised inner loop.  No external dependency required.
  - New file: `core/unfold_nnqp.py` (`solve_nnqp` + `unfold_nnqp`)
  - New `Detector.unfold_nnqp()` method
  - Registered in `core/__init__.py`, exposed in `docs/detector.rst`
    and `docs/overview.rst` (method #76)
- **qpmad unfolding** (`unfold_qpmad` / `solve_qpmad`, Sherikov,
  https://github.com/asherikov/qpmad). Solves the strictly-convex QP
  ``min 0.5 ||A x - b||^2 + α/2 ||L x||^2 + α0/2 ||x||^2  s.t.
  lb <= x <= ub`` (default: ``x >= 0``) by recasting it as
  ``min 0.5 x^T H x + g^T x`` with ``H = A^T A + α L^T L + α0 I``
  (symmetric PD). The default ``backend='python'`` uses a self-contained
  NumPy port of an active-set QP solver (Nocedal & Wright, *Numerical
  Optimization*, ch. 16.4). When the upstream qpmad C++ library with
  Python bindings is installed, ``backend='qpmad'`` calls it directly
  and otherwise gracefully falls back to the python backend.
  - New file: `core/unfold_qpmad.py` (`solve_qpmad` + `unfold_qpmad`)
  - New `Detector.unfold_qpmad()` method
  - Registered in `core/__init__.py`, exposed in `docs/detector.rst`
    and `docs/overview.rst` (method #77)
  - Added `nnqp` and `qpmad` to the
    `TestMaxNeutronEnergy::test_spectrum_zero_above_emax` parametrization
    in `tests/test_coverage.py`.
  - New tests in `tests/test_nnqp_qpmad.py` (32 tests covering the inner
    NNQP and active-set QP solvers, the BSS-unfolding solvers, and the
    `Detector.unfold_nnqp` / `Detector.unfold_qpmad` methods).
- **Gnowee unfolding** (`unfold_gnowee` / `solve_gnowee`, plus the
  `bssunfold.core._gnowee` pure-Python 3 port of the Gnowee hybrid
  metaheuristic, Bevins & Parsons, UC Berkeley / Slaybaugh Lab,
  https://github.com/SlaybaughLab/Gnowee). Gnowee combines Lévy flights
  (Cuckoo Search via the Mantegna algorithm), golden-ratio crossover
  (Modified Cuckoo Search / Differential Evolution), scatter search
  (Egea 2009) and DE-style mutation in an elitist population with
  Metropolis-Hastings acceptance and stall-driven restarts. Searches in
  log space, seeded with a Landweber warm-start solution (or the
  user-provided `initial_spectrum`), bounded to
  `log(seed) ± half_range` decades, with a scale-consistent objective
  combining the relative L2 residual, Tikhonov regularisation,
  second-difference smoothness and (optionally) negative Shannon
  entropy. No external optimisation library required.
  - New file: `core/_gnowee.py` (Lévy / TLF / `GnoweeHeuristics` /
    `run_gnowee`)
  - New file: `core/unfold_gnowee.py` (`solve_gnowee` + `unfold_gnowee`)
  - New `Detector.unfold_gnowee()` method
  - Registered in `core/__init__.py`, exposed in `docs/detector.rst`
    and `docs/overview.rst` (method #75)
  - New tests in `tests/test_gnowee.py` (28 tests; covers the samplers,
    the `run_gnowee` optimizer on the 5-D sphere benchmark,
    `solve_gnowee` truth-recovery on a synthetic under-determined BSS
    problem, and `Detector.unfold_gnowee` end-to-end including
    reproducibility, `initial_spectrum` injection, `max_neutron_energy`
    truncation and diagnostics).
  - Added `gnowee` to the `TestMaxNeutronEnergy::test_spectrum_zero_above_emax`
    parametrization in `tests/test_coverage.py`.
- **CUQIpy Bayesian uncertainty-quantified unfolding** (`unfold_cuqi` /
  `solve_cuqi_bayesian`, integration of the DTU package
  [CUQIpy](https://github.com/CUQI-DTU/CUQIpy) — *Computational
  Uncertainty Quantification for Inverse Problems*; optional dependency,
  new extra `bssunfold[cuqi]`, `cuqipy>=1.5.0`): the spectrum is modelled
  on the log scale `f = exp(theta)` with a smoothness prior anchored on
  a data-driven NNLS center — `prior="gmrf"` (CUQIpy GMRF with a
  first/second-order difference precision operator, `gmrf_order` 1/2) or
  `prior="ou"` (dense Ornstein-Uhlenbeck correlation Gaussian,
  `lengthscale`) — and the posterior is explored with the CUQIpy
  samplers `sampler=` `pcn` (preconditioned Crank-Nicolson, centred
  parameterisation), `cwmh` (component-wise Metropolis-Hastings),
  `mala` / `ula` ((M)ALA on a MAP-whitened parameterisation with the
  analytic gradient; ula experimental), `nuts` (native No-U-Turn
  Sampler) and the hierarchical `gibbs` / `gibbs_nuts` (CUQIpy
  `HybridGibbs`: the GMRF smoothness precision `delta` is inferred from
  the data through a conjugate `Gamma(alpha, beta)` hyperprior with an
  exact conjugate update, the spectral block updated by pCN or NUTS).
  The result carries the posterior mean spectrum, per-bin posterior
  std, configurable HPD credible intervals (`credible_level`) and
  convergence diagnostics under `cuqi_stats` (effective sample size,
  Gelman-Rubin R-hat for multi-chain runs, acceptance rates, raw
  posterior draws and the hyperparameter draws
  `delta_samples`).  Lazy cuqi import: the package imports normally
  without cuqipy (`CUQI_AVAILABLE` flag / `check_cuqi_availability()`)
  and `unfold_cuqi` raises an informative `ImportError` otherwise.
  NumPy 2.4 compatibility: upstream `cuqipy` 1.5.1 caps `numpy<=2.2.0`,
  which conflicts with this package's `numpy>=2.4.1` floor (and with
  `odl`'s `numpy>=2.3` in the aggregated `all` extra).  A maintained fork
  (https://github.com/Radiationsafety/CUQIpy, branch `numpy2-support`,
  dist version 1.5.2) relaxes the cap to `numpy<2.5` and fixes NUTS for
  `numpy>=2.4` (`int()` conversion of 1-element arrays in the slice and
  U-turn checks of `cuqi/sampler/_hmc.py`).  The uv lockfile/CI resolves
  `cuqipy` from the fork through `[tool.uv.sources]`; pip installs of the
  `cuqi` extra on NumPy 2.4 should pre-install the fork with
  `pip install "cuqipy @ git+https://github.com/Radiationsafety/CUQIpy@numpy2-support"`
  (pip then accepts it as satisfying `cuqipy>=1.5.0`); on NumPy <= 2.2 the
  plain PyPI `cuqipy` works unchanged.
  Documentation: `docs/cuqi_bayes.rst`.
- **GEE unfolding** (`unfold_gee` / `solve_gee` / `solve_gee_full` /
  `gee_fit`, Python analogue of the R package `gee` 4.13-30, Liang &
  Zeger 1986): the detector spheres are treated as a correlated
  cluster with a working correlation matrix `R(alpha)`
  (`corstr="exchangeable"|"ar1"|"independence"`, moment-estimated from
  the Pearson residuals each iteration) and a quasi-likelihood family
  select (`family="gaussian"|"poisson"|"gamma"`); the penalised score
  `A^T R(alpha)^-1 (b-Ax) - lam G x = 0` (second-difference roughness
  ridge) is solved by the classical IRLS loop with projected
  (non-negativity) fixed-point fallback.  Robust Liang-Zeger
  sandwich uncertainties for the unfolded spectrum are reported as
  `robust_se` / `naive_se` (with `spectrum_uncert_robust`) and the
  full `cov_robust`/`cov_naive` matrices in the `_full` diagnostics,
  plus `alpha`, `phi`, `pearson_chi2`, `df`, `iterations`,
  `converged`.  Pure NumPy.
- **Uno-style Lagrange-Newton constrained unfolding** (`unfold_uno` /
  `solve_uno` / `solve_uno_full`, analogue of the R package `Uno`,
  Vanaret & Leyffer 2024 arXiv:2406.13454): solves the constrained
  non-linear program `min 1/2||W(Ax-b)||^2 + lam/2||D2 x||^2 s.t.
  x >= 0` with two Uno presets: `filter_sqp` (Lagrange-Newton SQP with
  the exact constant Hessian and the Fletcher-Leyffer filter
  globalisation; the convex QP sub-problem is solved exactly in one
  Lawson-Hanson active-set step on the stacked least-squares system)
  and `ipopt_like` (primal-dual interior point with the log-barrier
  regularisation `diag(mu/x^2)`, geometric mu schedule and the
  fraction-to-the-boundary rule; `hessian="exact"|"bfgs"` Hessian
  building blocks).  SolveStatistics-style quality diagnostics
  (`objective`, `constraint_violation`, `dual_infeasibility`,
  `n_iterations`, `converged`).  Pure NumPy/SciPy (the QP sub-solve
  uses `scipy.optimize.nnls`).
- **P-spline mixed-model unfolding with REML smoothing selection**
  (`unfold_pspline_reml` / `solve_pspline_reml` /
  `solve_pspline_reml_full`, Python analogue of the R package
  `LMMsolver`): the spectrum is represented as a P-spline; the spline
  coefficients are split spectrally into an unpenalised fixed part
  (polynomial trend, null space of the `d`-th order difference penalty)
  and a penalised random part (range space), the smoothing parameter is
  the variance ratio estimated by maximising the REML profile
  likelihood on scale-free relative bounds, and the final spectrum is
  obtained from the Henderson mixed-model equations.  Reports `lam`,
  `lam_relative`, `reml_loglik`, `sigma2`, effective dimension (`ed`,
  `ed_norm`).  Uniform / Poisson / explicit weights; uniform / log
  knots; pure NumPy/SciPy (no new dependencies).
- **AMG/stationary-preconditioned Krylov unfolding** (`unfold_amg` /
  `solve_amg` / `build_preconditioner`, analogue of the R package
  `Rlinsolve` + pyamg): damped normal equations
  `(A^T A + reg I) x = A^T b` solved with `cg`/`bicgstab`/`gmres`
  accelerated by an algebraic multigrid preconditioner (smoothed
  aggregation, optional dependency `pyamg` — new extra
  `bssunfold[amg]`) or by one sweep of a classical stationary
  iteration (Jacobi / Gauss-Seidel / SOR / SSOR); nonsymmetric
  `gs`/`sor` preconditioners are transparently swapped for `ssor` with
  `cg` (warning); non-negativity enforced with projected outer
  restarts; auto Tikhonov damping (`regularization=None` selects
  `1e-4 * mean(diag(A^T A))`) stabilises rank-deficient normal
  matrices and the AMG hierarchy; deterministic AMG setup.
- **Selectable SVD backends for TSVD** (`svd_solver` parameter of
  `unfold_tsvd` / `solve_tsvd`): `full` (dense LAPACK, default),
  `arpack` (implicitly restarted Lanczos — the R `rARPACK`/`RSpectra`
  analogue) and `propack` (Lanczos bidiagonalization — the R
  `svd::propack.svd` analogue).  Iterative backends compute only the
  leading `k` triplets when a fixed truncation `k` is given and
  degrade gracefully to the dense solver (with a warning) on failure;
  automatic k-selection keeps using the dense decomposition.
- **SSR Sign-Simplicity-Regression unfolding** (`unfold_ssr` /
  `solve_ssr` / `solve_ssr_full`, Python port of the R package
  `sisireg` 1.2.1, Metzner 2020/2021, GPL>=2): MLEM data-fidelity
  updates alternate with non-equidistant SSR QSOR sweeps of the
  spectrum over the energy grid; each sweep replaces interior bins by
  the simplicitic neighbour interpolation and reverts updates that
  would violate the partial sum criterion (threshold `fn`),
  suppressing sign-inadequate wiggles while bounding the deviation
  from the data-fit in every window.  `fn="auto"` runs Metzner's
  minimum-statistic ladder: the threshold starts at
  `int(0.66 * partial_sum_quantile(n, k_run))` and descends while the
  folded residuals stay sign-adequate (data-space partial sum /
  maximum run tests) and the spectrum does not gain extrema; the last
  adequate candidate is returned.  The pure regression building
  blocks are exported as well (`ssr`, `ssr_ne`, `ssr_min_statistic`,
  `ssr_min_statistic_ne`, `ssr_predict`, `max_run_quantile`,
  `partial_sum_quantile`, `partial_sum_max`, `partial_sum_valid`,
  `run_valid`, `rolling_median`, `number_of_extrema`) and reproduce
  the R/C semantics exactly (rolling-median start values, truncating
  integer conversions, L1 and standardised-L2 equidistant solvers).
  Reports `fn`, `fn_start`, `k_run`, `n_extrema`, `ps_valid_data`,
  `run_valid_data`, `max_run_data`, `ssr_sweeps` and the `fn_ladder`
  trail.  Pure NumPy (no new dependencies); non-negativity preserved
  by construction; sisireg's GPL (>= 2) is compatible with the
  GPL-3 license of bssunfold.
- **Spatial SSR regression (`ssr3d`) and SSR-MLP (`ssrMLP`)** (Python
  ports of the remaining solver parts of the R package `sisireg`
  1.2.1, files `R/ssr3d.R` + `src/ssr3d.c` and `R/ssrMLP.R`): the
  spatial module regresses scattered planar data with the minimal
  surface that is statistically adequate in the residual signs of the
  k-quadrant-neighbourhoods (`ssr3d`, `ssr3d_predict` with exponential
  and 4-point minimal-surface prediction modes, `ssr3d` Gauss-Seidel
  solver with the C `chi` adequacy reverts, neighbourhood builders
  `near_neighbors_quadrant`/`near_neighbors` with the R value-matching
  tie semantics, `ps_max_3d`/`ps_statistic_3d` partial sum statistics,
  weighted means `wmean`/`wmean_exp`/`wmean_ms`, model dataclass
  `SSR3DModel`); the MLP module trains a two-hidden-layer perceptron
  with Metzner's partial sum criterion instead of least squares
  (`ssrmlp_train` with `opt='ps'/'ps_lse'/'ps_l1'/'lse'/'ext'`,
  `ssrmlp_predict`, criterion building blocks
  `check_ps`/`err_ps`/`fac_ps`/`err_ps_lse`/`fac_ps_lse`/`err_ps_l1`/`fac_ps_l1`,
  `calc_out`, factor importance `fii_model`/`fii_prediction`, model
  dataclass `SSRMLPModel`).  Both ports reproduce the R semantics
  exactly — including the fractional `k = maxRunR(n)/2` truncations,
  the distance-tie neighbourhood inflation, the transposed hidden
  layer update of `ssrMLP` (square hidden layers only, as in R) and
  its not-forwarded `fn`/`alpha` arguments — and were verified
  against the original R output to machine precision (recorded as
  fixture tests).  Pure NumPy, no new dependencies.
  - **Fission-model GA+LM unfolding** (`unfold_fission_ga` /
  `solve_fission_ga`, port of the multisphere algorithm of
  Ogorodnikov 2024, sections 4-5, `BonnerFinder()`): parameterized
  model curves in the FRUIT paradigm — the spectrum is a
  superposition of thermal Maxwellian, epithermal tail with cutoff
  and Watt-type fast fission fractions (article eq. 4.29) with the
  seven free parameters `a1, a2, a3, b, beta, alpha, TF` and the
  article's bounds.  Stage 1 performs a stochastic global search of
  the parameter hypercube with a differential-evolution (genetic)
  algorithm minimizing the L1 discrepancy of the folded readings
  (article eq. 4.32); stage 2 refines the best point with a bounded
  nonlinear least-squares routine (SciLab `leastsq` analogue; `trf`
  default, `lm` available).  An optional free overall scale factor
  `phi_scale` (log10-parameterized, `fit_scale=True`) matches
  absolutely calibrated readings; `fit_scale=False` reproduces the
  exact 7-parameter normalized formulation.  The article's
  validation criteria (per-sphere relative uncertainties with sign
  alternation, FOM, model-spectrum norm in [0.6, 1.2] for normalized
  problems) are reported in the `validation` result entry, the fitted
  parameters (with normalized `weight_fractions`) in `model_params`.
  Pure NumPy + SciPy (`scipy.optimize.differential_evolution` /
  `least_squares`); deterministic under `random_state`.
- **Tikhonov + generalized discrepancy unfolding** (`unfold_tikhonov_sobolev_dp` /
  `solve_tikhonov_sobolev_dp`, port of Ogorodnikov 2024, sections 3
  and 5, `alfaFinder()`): Tikhonov regularization with the discrete
  Sobolev `W_2^1` penalty (first-difference operator — the discrete
  analogue of the Euler equation `A*A z + alpha (z - z'') = A* u`,
  article eq. 3.10; `curvature` and `identity` penalties selectable),
  with the regularization parameter `alpha*` selected by the
  generalized discrepancy principle `||A z - b||^2 = delta^2`
  (article eq. 3.8).  The root of the monotone discrepancy `rho(alpha)`
  is bracketed on a log10 grid and refined with Brent's method (the
  robust counterpart of the Newton/chord root finding used in the
  article); diagnostic status codes mirror the article's `FFinder`
  `IERR` conventions (0 = root found, 1 = delta too small for the
  data, 2 = delta larger than any achievable misfit).  The standalone
  selection routine `alpha_finder_generalized_discrepancy` (with the
  `generalized_discrepancy` function) is exported for use with other
  solvers.  Pure NumPy + SciPy.
- **Tests** (`tests/test_ogorodnikov2024.py`, 36 tests): quasi-real
  experiment fixtures built from the packaged GSF response functions
  (Fission-model truth + uniform noise, article eqs. 4.21-4.22),
  discrepancy-principle property checks on synthetic linear systems,
  well-posed recovery checks, Detector-workflow tests and IAEA
  Compendium package-data cases (Cf-252 via the Fission model, AmBe
  via Tikhonov-DP).
- **Worked example** (`examples/51-ogorodnikov2024.ipynb`): the
  quasi-real experiment (shape recovery to pearson 0.9999 and dose
  rates within ~1 % via the Fission-model GA, with the validation
  report), the generalized-discrepancy alpha selection with the
  monotone `rho(alpha)` curve and the penalty-family comparison
  (including the null-space fluence/dose drift discussion), and the
  IAEA Compendium package-data cases.

### Fixed
- `solve_tsvd` ignored the explicit `k` and `threshold` parameters:
  the automatic k-selection unconditionally overwrote them.  Fixed to
  match the documented behaviour (`k`/`threshold` override `method`).

## [0.24.1] - 2026-09-15
### Fixed
-  warnings.simplefilter("always") in test_randomization_experiment_unknown_method

## [0.24.0] - 2026-09-15

### Added
- **B-spline MLEM unfolding method (MLEM-BS)** — `unfold_mlem_bs` /
  `solve_mlem_bs` / `solve_mlem_bs_full`, implementing the neutron
  spectrum unfolding algorithm of Mazankova et al., "Experimental
  Measurement of Neutron Flux and Its Mathematical Data Processing",
  Proceedings of CNDGS'2026 (Brno), https://doi.org/10.47459/cndcgs.2026.61.
  The method:
  - **B-spline parameterisation** — the spectrum is represented as
    `x(E) = sum_s b_s B_s(E)` (clamped cubic B-splines, order `p = 4`
    in the paper) so the effective system matrix is `RB = R B`;
    `build_bspline_basis` assembles the design matrix with uniform or
    logarithmic knot grids (`knot_spacing="auto"` selects log knots
    when the energy grid spans more than two decades).
  - **Regularized MLEM iteration** (Eq. 4 of the paper) on the
    B-spline coefficients with the second-derivative penalty
    `P(b) = ||D^(2) b||_2^2` (Eq. 5, `second_difference_matrix`) and
    the sieve restriction to non-negative coefficients (Szkutnik,
    J. Multivar. Anal. 93, 2005 — ref. [7] of the paper); the
    multiplicative update preserves positivity automatically.
    The absolute penalty `beta` of the paper is problem-scale
    dependent (the paper uses `1.0e-17`), so a scale-free
    `beta_relative` (relative to the mean MLEM sensitivity) is also
    accepted.
  - **K_S-based parameter selection** (Eq. 6) — the goodness-of-fit
    statistic `K_S = |sum_i (n_i - model_i)^2 / sum_i model_i - 1|`
    is tracked per iteration; `auto_params=True` selects the number
    of iterations, the B-spline dimension `N_s` and the penalty
    strength by minimizing `K_S` over a candidate grid
    (`AUTO_BETA_RELATIVE_GRID`); in manual mode the iteration keeps
    the best `K_S` iterate with `ks_patience` early stopping.
  - **Poisson-bootstrap confidence intervals** (Eqs. 7-9) —
    backward-reconstructed counts `n^(0) = R x^(0)` are resampled as
    Poisson replicates, each replicate is unfolded and percentile
    (alpha/2, 1-alpha/2) intervals are reported as `ci_low` /
    `ci_high` (plus `bootstrap_mean` / `bootstrap_std`);
    `bootstrap_ci=True, ci_alpha=0.05` gives the paper's 95% CI.
  - `Detector.unfold_mlem_bs` wrapper (full `calculate_errors`
    Monte-Carlo support, `max_neutron_energy` truncation), exports in
    `bssunfold.core`, Sphinx page `docs/mlem_bs.rst`, worked example
    `examples/44-mlem-bs.ipynb` and test suite `tests/test_mlem_bs.py`
    (21 tests: basis/penalty/statistic units, noise suppression vs
    plain MLEM, auto-selection, bootstrap coverage on an
    overdetermined system, truncation and validation).

## [0.23.1] - 2026-09-08

### Changed
- python 3.11 syntax
- max_neutron_energy for unfold_parametric

## [0.23.0] - 2026-09-08

### Added
- **N-spline unfolding method** — `unfold_nspline` / `solve_nspline` /
  `solve_nspline_full`, implementing the neutron spectrum unfolding
  approach of Islamgulov & Lartsev, "Reconstruction of neutron spectra
  from activation measurements in the form of N-splines",
  Atomic Energy 104(5), 295-302 (2008) (RFNC-VNIITF).  The method:
  - **N-spline parameterisation** — the spectrum is represented by
    `N(E) = exp(a_k + q_k ln E + r_k E)` on each segment
    `[E_k, E_{k+1}]` (Eq. 2 of the paper) with C0/C1 continuity at the
    interior knots imposed through the block constraint matrix
    `D X = 0` (Eqs. 3-5); `build_continuity_matrix` assembles `D`,
    `fit_nspline` solves the weighted log-domain least-squares
    approximation (Eqs. 6-7) via the KKT system, and `nspline_eval`
    evaluates the spline.
  - **Directed-divergence (MIRD) unfolding loop** — the functional
    `H = sum_i [pN_i ln(pN_i/p_i) - pN_i + p_i]` (Eqs. 8-9) is
    minimised by the flux-conserving gradient iteration with the
    paper's conservative step `dmu = 0.1/sup|R - Rbar|` and
    backtracking; the spectrum is re-fitted by the N-spline after
    every iteration (the paper's regularisation, `smoothing=True`;
    `smoothing=False` reduces to the plain MIRD loop).
  - **Paper's stopping criteria and quality control** — iterations stop
    at the measurement-error level `H <= 0.5 sum_i p_i (dQ_i/Q_i)^2`
    or on stalled relative decrease; the reconstruction is qualified
    by `nev = sqrt(1/(N-1) sum_i ((Qr_i-Q_i)/dQ_i)^2)` with the
    acceptance bound `nev <= 1 + 2/sqrt(N)`; results expose `H`,
    `H_history`, `H_target`, `nev`, `nev_limit`, `acceptable`,
    `fluence`, `mean_energy` and the spline parameters.
  - **Knot presets from the paper** — `NSPLINE_KNOT_PRESETS` with the
    BARS-5 channel, IGRIK channel/surface and YAGUAR channel knot sets
    (MeV); automatic log-uniform knots (`auto_knots`) and explicit
    user knots are also supported; `continuity` option selects
    `"C0C1"` (default), `"C0"` or `"none"`.
  - `Detector.unfold_nspline` wrapper (full `calculate_errors`
    Monte-Carlo support, `max_neutron_energy` truncation), exports in
    `bssunfold.core`, Sphinx page `docs/nspline.rst`, worked examples
    `examples/41-nspline.ipynb` (comparison with GRAVEL on a synthetic
    spectrum) and `examples/42-nspline-iaea.ipynb` (IAEA Compendium
    Monte-Carlo BSA spectrum `t4-14-s.txt_1` unfolded from GSF
    readings; N-spline recovers the shape at ~0.25 dex while GRAVEL /
    MLEM diverge from a flat prior) and test suite
    `tests/test_nspline.py` (32 tests).
- **Directed-divergence unfolding for Bonner spheres** — new standalone
  `unfold_directed_divergence` / `solve_directed_divergence` methods that
  expose the multiplicative I-divergence iteration previously used only as a
  BON95 refinement step. The solver operates directly on the detector
  response matrix, supports optional first- and second-order smoothness
  regularization, and is available both as a core function and a
  `Detector` method.
- **Express unfolding for Bonner spheres** — new standalone
  `unfold_express` / `solve_express` methods that adapt the historical
  Express idea to sphere response functions by fitting a piecewise-
  exponential spectrum model directly to the measured readings. The method
  accepts explicit coarse-group boundaries, works with the existing response
  matrix, and is exposed through the core API and `Detector` wrapper.
- **Non-negative K-SVD unfolding method** — `unfold_nnksvd` /
  `solve_nnksvd_unfold`, implementing the BNCT epithermal neutron
  spectrum unfolding method of Xu et al. (NIMA 2026,
  https://doi.org/10.1016/j.nima.2026.172070).  The pipeline combines:
  - **Non-negative K-SVD dictionary learning** (`solve_nnksvd`) — K-SVD
    with non-negative truncation of dictionary atoms during the rank-1
    SVD update, plus an automatically-derived training-sample-driven
    prior (`alpha_prior` = mean sparse code of the training signals).
  - **Three sparse-coding strategies**, switchable via the
    `sparse_coder` keyword argument:
    - `"nnls_topk"` (the article's proposed method) — global NNLS
      coarse solution → top-K atom screening → local NNLS fine
      optimization.  Hierarchical strategy that avoids the cumulative
      selection error of greedy algorithms.
    - `"omp"` — classic Orthogonal Matching Pursuit.
    - `"nn_omp"` — OMP with non-negativity constraint on the support
      least-squares step (solved as NNLS).
  - **Tikhonov-regularized NNLS via augmented-matrix form**
    (`solve_tikhonov_nnls`) — Eq. 2.5 / 2.6 of the article, solvable
    by any off-the-shelf NNLS routine, with an optional
    training-sample prior constraint.
  - **Equivalent (column-normalized) detection dictionary**
    `M_norm = normalize(R @ D)` (Eq. 2.4) — removes the interference
    caused by atom-amplitude differences.
  Default hyperparameters follow the article: `lambda_tik=0.01`,
    `prior_wt=0.5`, `n_dictionary_iterations=80`, `n_atoms=15`,
    `sparsity=2`, `random_state=42` (the article's optimal
    configuration: 15 atoms, sparsity K=2).
  Core solver in `core/unfold_nnksvd.py`, Detector wrapper,
  helper functions `solve_nn_omp`, `solve_nnls_topk`,
  `solve_tikhonov_nnls`.  52 dedicated tests in
  `tests/test_nnksvd.py`.
- **Two new spectrum-comparison metrics** from Xu et al. (NIMA 2026,
  Section 2.2.3) in `utils/comparison.py`:
  - `relative_flux_error` — Eq. 2.7: `||phi_true - phi_hat|| / ||phi_true||`.
  - `comprehensive_score` — Eq. 2.9: `flux_err - 0.5 * flux_corr`
    (lower is better; the article's best score is `-0.3612`; Eq. 2.8
    Pearson correlation is `pearson_r` — not duplicated).
  Both are exposed by `compare_spectra`, registered in
  `DEFAULT_UNFOLD_BENCHMARK_METRICS`, and the three NN-KSVD sparse
  coders are registered in `DEFAULT_UNFOLD_BENCHMARK_METHODS` as
  `nnksvd_nnls_topk`, `nnksvd_omp`, `nnksvd_nn_omp`.

### Changed
- **`unfold_nnksvd` training signals: truncated cosines → log-spaced Gaussians**
  — the default training signals for the K-SVD dictionary learning were
  truncated cosines (`max(0, cos(i·t))`) that all peak at the lowest energy
  bin, producing a dictionary blind to the fast and epithermal regions.
  Replaced with 30 log-spaced Gaussian bumps spanning the full energy grid
  (plus a flat prior), so dictionary atoms cover the full spectral range
  from thermal to fast energies.  New `E_MeV` parameter on
  `solve_nnksvd_unfold`, `unfold_nnksvd`, and `Detector.unfold_nnksvd`
  provides the energy grid for training-signal generation.  Example notebook
  `examples/40-nnksvd.ipynb` updated with IAEA Compendium reference spectrum
  and parameter grid sweep showing optimal configuration per detector.

### Fixed
- **`unfold_interpret` / `interpret_qp` convergence for `norm=1`** — when the
  L1 penalty norm is used, the QP matrix ``P = A'A`` can be rank-deficient
  (e.g. 11 detectors × 60 energy bins). OSQP's default tolerance was too
  strict for this ill-conditioned problem, causing ``iteration_limit`` and NaN
  metrics. A small diagonal ridge (``ridge_coeff``, default ``"auto"`` =
  ``1e-8 * trace(P) / n``) is now added to ``P`` when ``norm == 1`` and
  ``smoothness_order == 0``, making it positive definite without distorting the
  L1 solution. The parameter is exposed at all three API levels
  (``build_interpretation_qp``, ``interpret_qp``, ``unfold_interpret``) and can
  be set to ``0.0`` to restore the legacy behaviour.


## [0.22.0] - 2026-09-02

### Added
- **Bin-wise adaptive unfolding** — `unfold_binned` / `solve_binned`: for each
  of the 60 energy bins, selects the best-performing method from a pre-computed
  benchmark lookup table (built from 60+ methods x 271 reference spectra across
  41 quality metrics) and assembles the final spectrum by direct bin-picking.
  The lookup ships as `data/bin_lookup.json` (51 unique candidate methods,
  top-5 per bin).  Core solver in `core/unfold_binned.py`, Detector wrapper,
  I/O helpers (`load_bin_lookup`, `save_bin_lookup`, `build_bin_lookup`).
  Pre-computation script: `tools/build_bin_lookup.py`.
  13 dedicated tests in `tests/test_binned.py`.
  Example notebook: `examples/39-unfold-binned.ipynb`.
- **Pre-computed bin lookup table** (`src/bssunfold/data/bin_lookup.json`)
  built from the MC + Compendium gridsearch benchmark.  Contains per-bin
  method rankings (mean absolute error across spectra) for 67 unfolding
  methods, with top-5 kept per bin.

## [0.21.0] - 2026-08-31

### Added
- **Maximum neutron energy cutoff** — `max_neutron_energy` parameter on every
  `unfold_*` method (including `unfold_maeo`, `unfold_interpret`): pass
  e.g. `max_neutron_energy=10.0` to force zero fluence above 10 MeV.
  Two internal strategies:
  - **UB array** for QP solvers (`cvxpy`, `qpsolvers`, `docplex`, `scip`,
    `mystic`): full response matrix + per-bin upper bound `ub = 0` above the
    cutoff is passed to the solver.
  - **Trimming** for iterative / matrix solvers (all others): response matrix
    is sliced to active energy bins, solved, and expanded back with zeros above
    the cutoff.
  New helper module `core/_max_energy.py` (`upper_bounds`, `max_energy_mask`).
  10 dedicated tests in `TestMaxNeutronEnergy`.
- **Randomized Kaczmarz** — `unfold_randomized_kaczmarz` /
  `solve_randomized_kaczmarz`: stochastic row-projection method with
  probability proportional to squared row norms (Strohmer & Vershynin 2009),
  achieving faster convergence than the cyclic variant for ill-conditioned
  response matrices.  Core solver in `core/unfold_randomized_kaczmarz.py`,
  Detector wrapper, Numba-free pure-NumPy fallback.
- **Ensemble Kalman Inversion (EKI)** — `unfold_eki` / `solve_eki`:
  Bayesian posterior approximation without MCMC, propagating an ensemble of
  particles through the forward model and updating via the Kalman gain
  equation (Iglesias et al. 2013).  Regularized variant adds `αI` to the
  covariance matrices; covariance inflation prevents ensemble collapse.
  Core solver in `core/unfold_eki.py`, Detector wrapper.
- **Five new regularization-parameter selection criteria** in
  `core/regularization.py`, all accessible via `select_regularization_parameter`:
  - `quasi_optimality_selection` — minimises the noise component in the SVD
    basis (Hochstenbach & Reichel 2015).
  - `ncp_selection` — Normalized Cumulative Periodogram; KS-test on residual
    whiteness selects α giving the whitest residuals.
  - `snr_criterion_selection` — maximises the signal-to-noise ratio in the
    Tikhonov solution.
  - `weighted_gcv_poisson_selection` — GCV with Poisson variance weights for
    heteroscedastic (counting) noise.
  - `kfold_cv_selection` — K-fold cross-validation; noise-independent
    alternative to GCV.
- **IAEA validation test** (`tests/test_iaea_validation.py`) expanded from
  21 to 54 methods, now covering all general-purpose `Detector.unfold_*`
  methods including CGLS, GKS, Lanczos, FISTA, Tikhonov-TV, SAND-II,
  OSEM, MAP-EM, BSREM, SART, BON95, BUNKI, BUNKI-UT, STAY'SL, IMAXED,
  AMAXED, CRYSTAL BALL, RECONST, EPIC, iterative refinement, hybrid GMRES,
  FERDOR, ReBUNKI, NSDUAZ, RFSP-JUL, ensemble, cascade, composite, MAEO,
  randomized Kaczmarz, and EKI.
- **New test files**:
  - `tests/test_randomized_kaczmarz_eki.py` — 16 tests for both solvers
    (basic, deterministic, zero-input, relaxation, ensemble size, Detector
    wrapper, save_result, Monte-Carlo errors, exports).
  - `tests/test_regularization_new_criteria.py` — 22 tests for the five
    new selection criteria (basic, ill-conditioned, white-noise, custom range,
    reproducibility, dispatcher integration).
- **New example notebooks**:
  - `examples/36-randomized-kaczmarz-eki.ipynb` — side-by-side comparison
    of randomized Kaczmarz and EKI with established methods.
  - `examples/37-regularization-criteria.ipynb` — visual comparison of all
    nine regularization selection criteria.

## [0.20.0] - 2026-08-28

### Added
- **Ensemble unfolding method** — `unfold_ensemble` / `solve_ensemble` combines
  several base solvers (default MLEM, Bayes, Landweber, CGLS, GRAVEL) into a single
  robust solution via weighted-average (inverse-residual weights), median,
  trimmed-mean, or best-residual combination strategies.
- **Iterative refinement method** — `unfold_iterative_refinement` /
  `solve_iterative_refinement` performs a two-pass unfold and blends the two
  spectra with an automatically selected blending factor α (line search over
  `max_alpha_search` candidates) to reduce method-specific bias.
- **Input validation**: `run_unfolding` now validates `readings`,
  `detector_names`, `n_energy_bins`, `noise_level` (range 0–1) and
  `n_montecarlo` before building the system; new validators `validate_system`
  and `validate_solver_params` (shape, NaN/Inf and parameter-range checks) are
  used across iterative solvers and exported from `bssunfold.utils.validators`.
- **Extended Numba JIT acceleration** to the Landweber and D'Agostini Bayes
  inner loops (in addition to Doroshenko, Kaczmarz, MLEM, GRAVEL), with
  automatic disk caching and pure-Python fallback.
- **Batch dose-rate computation** in `calculate_dose_rates` (a single matrix
  multiply over all conversion-coefficient geometries instead of a per-geometry
  Python loop).
- Large test-coverage boost: ~5,200 new test lines across
  `tests/test_boost_part1..4.py` and `tests/test_new_ensemble_refinement.py`,
  exercising validation, solvers, parametric families and the new methods.

### Fixed
- **Landweber JIT convergence regression**: the Numba inner loop converged on
  `‖Ax‖` instead of `‖Ax−b‖`. With the wrapper's default zero initial guess this
  made the residual zero at the first iteration, so `solve_landweber` returned an
  all-zero spectrum (regression vs 0.19.x). The JIT path now receives `b` and
  uses the true residual norm, matching the pure-Python fallback.

## [0.19.1] - 2026-08-27
### Fixed
 - version in pyproject.toml

## [0.19.0] - 2026-08-27

### Added
- **Classic unfolding codes reimplemented** — three
  historically codes from Zijp, Willem L., and Henk J. Nolthenius. Experience with 
  neutron spectrum unfolding codes. No. ECN--105. Stichting Energieonderzoek Centrum 
  Nederland, Petten, 1981 are now available
  as independent, from-scratch Python reimplementations built solely from their
  published mathematical descriptions. They are exposed as `Detector` methods
  and exported from `bssunfold.core`:
  - **CRYSTAL BALL** (`unfold_crystal_ball` / `solve_crystal_ball`) — a direct
    (non-iterative) method that represents the spectrum as a linear combination
    of the detector response functions and solves the regularized normal
    equations `α = (R Rᵀ + λI)⁻¹ b`, then `φ = Rᵀ α`. Based on the
    delta-operator approximation (Kam & Stallmann; 1981 review).
  - **RFSP-JUL** (`unfold_rfsp_jul` / `solve_rfsp_jul`) — an iterative damped
    least-squares method minimizing a weighted residual functional with a
    Marquardt-style damping term tying each iterate to the previous one; the
    minimizer is found from the symmetric positive-definite normal equations at
    each step. Based on the description by Fischer.
  - **STAY'SL** (`unfold_staysl` / `solve_staysl`) — a single-step linear
    Bayesian least-squares update `x = x0 + Cx Aᵀ (Cb + A Cx Aᵀ)⁻¹ (b − A x0)`
    that refines a prior spectrum using full measurement and prior covariance
    information. Based on the Bayesian formalism of Perey.
  - All three are documented in `docs/detector.rst` (autodoc), `docs/overview.rst`
    (method catalogue + mermaid diagram), and the README method reference table,
    each explicitly noted as an independent reimplementation of a proprietary
    original. Core-solver and `Detector`-wrapper tests live in
    `tests/test_classic_unfolders.py`.

- **Two-stage hybrid `mystic` solver** — `solve_mystic_hybrid` /
  `unfold_mystic_hybrid` is a two-stage hybrid that
  first uses `diffev2` for global exploration of the penalized least-squares
  objective and then refines the result with `fmin_powell` for precise local
  convergence. It is registered in `unfold_combined` / `unfold_composite`
  pipelines as `'mystic_hybrid'` and requires the optional `mystic` dependency
  (`bssunfold[mystic]`). It is now listed in `docs/detector.rst` (autodoc),
  `docs/overview.rst` (method catalogue + mermaid diagram) and the README
  method reference table (row 66); covered by `tests/test_mystic.py`.


## [0.18.0] - 2026-08-24

### Added
- **Cascade and Composite unfolding wired into the `Detector` API** —
  `unfold_cascade` / `unfold_composite` are now public `Detector` methods
  (previously standalone module functions only) and are exported from
  `bssunfold.core`. Both are covered by wrapper smoke tests in
  `tests/test_detector.py`.
  - `unfold_cascade`: sequential multi-method cascade; each stage may use the
    previous result as an initial guess or a prior, with optional early
    stopping on a quality threshold.
  - `unfold_composite`: adaptive ensemble (stacked generalization) that
    classifies the spectrum by hardness, runs a pool of individual methods,
    and combines them with confidence-weighted averaging.

- **Multi-resolution (coarse-to-fine) cascades** — true multi-resolution
  support in the cascade pipeline:
  - New `multi_resolution` / `coarse_bins` parameters on `unfold_cascade`
    and `unfold_adaptive_cascade` (and the `Detector.unfold_cascade`
    wrapper): the first stage runs on a coarse energy grid and its
    prolongated solution seeds the fine-grid stages.
  - New `coarse` / `coarse_bins` fields on `CascadeStage` for explicit
    per-stage coarse-grid pre-solves.
  - New shared helpers in `core/_multires.py`: `build_coarse_detector`,
    `prolongate_spectrum`, `_coarsen_columns`, `_split_coarse` (extracted
    from `unfold_genetic.py`, which re-exports them for backward
    compatibility). Coarsening sums adjacent response columns so a coarse
    bin-total spectrum reproduces fine readings; prolongation preserves
    total fluence.
  - Assessment note with usage examples, literature context and limitations:
    `docs/multires_cascade.rst`.
  - Tests: coarse-response consistency, fluence-preserving prolongation and
    multi-resolution cascade runs (`tests/test_cascade.py`).

- **ODL Advanced regularization methods** — new `unfold_odl_pdhg()` and
  `unfold_odl_douglas_rachford()` methods for advanced proximal optimization:
  - **PDHG (Primal-Dual Hybrid Gradient / Chambolle-Pock)** — efficient
    first-order method for non-smooth convex optimization with TV (Total
    Variation) regularization
  - **Douglas-Rachford Splitting** — operator splitting method for problems
    with composite objectives
  - Better preservation of sharp spectral boundaries compared to standard
    Tikhonov smoothness
  - **Implemented in pure NumPy** (no ODL dependency). ODL 1.0's own
    `odl.solvers.pdhg` / `douglas_rachford_pd` break on translated data terms,
    so the algorithms follow the ODL formulation but are self-contained.
  - No optional dependency required for these two methods.
  - Tests: `tests/test_new_unfold_methods.py` (class `TestODLSolvers`)

- **QUBO Quantum-Inspired Annealing** — new `unfold_qubo()` method implementing
  quantum-inspired optimization via Quadratic Unconstrained Binary Optimization:
  - Binary discretization of spectrum amplitudes with multi-bit precision
  - Simulated annealing solver adapted from D-Wave QUBO formulation
  - Effective for non-convex landscapes and discrete spectrum reconstruction
  - No quantum hardware required — classical simulated annealing backend
  - Optional dependency: `bssunfold[qubo]` (`pyqubo>=1.4.0`, `dwave-neal>=0.6.0`)
  - Tests: `tests/test_new_unfold_methods.py` (class `TestQUBOBackend`)

- **zfit Bayesian Inference** — new `unfold_zfit()` method using the zfit library
  for likelihood-based Bayesian spectrum reconstruction:
  - Poissonian likelihood model for detector readings
  - MCMC sampling via zfit's minimizers (Minuit, scipy)
  - Automatic uncertainty quantification from posterior samples
  - Compatible with zfit ecosystem for extended statistical analysis
  - Optional dependency: `bssunfold[zfit]` (`zfit>=0.10.0`, `tensorflow>=2.15.0`)
  - Tests: `tests/test_new_unfold_methods.py` (class `TestZfitBackend`)

- **MAEO (Multi-Algorithm Evolutionary Optimization)** — new `unfold_maeo()`
  method implementing ensemble evolutionary optimization:
  - Combines 4 multi-objective algorithms: NSGA-III, C-TAEA, AGE-MOEA-II, SPEA2
  - Multi-cycle evolution with convergence assistance mechanism
  - Hypervolume-based quality tracking across generations
  - Prior spectrum integration for informed initialization
  - Non-negativity constraints enforced throughout evolution
  - Reproducible results via deterministic random seeding
  - Optional dependency: `bssunfold[pymoo]` (`pymoo>=0.6.0`, `numba>=0.65.1`)
  - Tests: `tests/test_maeo.py` (8 comprehensive test cases)

- **Integration with external libraries**:
  - ODL (Operator Discretization Library) for advanced proximal algorithms
  - zfit for likelihood-based Bayesian inference
  - QUBO formulation inspired by quantum annealing approaches
  - pymoo for multi-objective evolutionary optimization

### Changed
- Updated method count from 51 to 55 unfolding algorithms
- Enhanced documentation with new method categories in README.md and Sphinx docs

### Documentation
- README and Sphinx docs synced to the actual method inventory (60+
  methods): features counts corrected (36 → 60+, 51 → 60+), method
  reference tables extended to #62 (IMAXED/AMAXED family, MAEO, MCMC,
  zfit, QUBO, ODL PDHG/Douglas-Rachford, cascade, composite), mermaid
  diagrams updated, README project structure now lists
  `unfold_cascade.py` / `unfold_composite.py` and all extracted helper
  modules (`_bon95.py`, `_fruit.py`, `_parametric_shared.py`,
  `_solver_backends.py`, `_interpret_pyopt.py`, `_interpret_report.py`,
  `_multires.py`).
- `docs/detector.rst`: added 13 missing `autofunction` entries
  (imaxed/amaxed/amaxed_regularization, fista, hybrid_gmres, mcmc, zfit,
  qubo, maeo, odl_pdhg, odl_douglas_rachford, cascade, composite).

### Fixed
- Sphinx build error: broken list-table indentation in the method
  reference table of `docs/overview.rst` prevented the table from
  rendering.


## [0.17.3] - 2026-08-22

### Added
- **IMAXED, AMAXED, and AMAXED-Regularization unfolding methods** — new algorithms
  from Wong's 2024 PhD thesis "Modernising neutron spectrum unfolding for fusion
  applications" (Sheffield Hallam University). These methods use cross-entropy
  regularization with Newton-type optimization and line search for improved
  convergence and stability.
  - `unfold_imaxed` / `solve_imaxed`: Improved MAXED using gradient-based
    optimization in log-space with cross-entropy regularization relative to
    a prior spectrum. Provides faster convergence than standard MAXED.
  - `unfold_amaxed` / `solve_amaxed`: Alternative MAXED with reversed
    cross-entropy definition, using Lagrangian multipliers to enforce
    chi-squared constraints.
  - `unfold_amaxed_regularization` / `solve_amaxed_regularization`: AMAXED
    with Tikhonov-style simultaneous minimization of chi-squared and
    cross-entropy, eliminating the need for manual chi-squared tuning.
    This method showed best performance in the thesis for fusion neutron
    spectrum unfolding.
  - All methods support Monte Carlo uncertainty propagation and are
    compatible with the existing Detector API.
  - Tests: `tests/test_wong2024_methods.py` (basic functionality, noise
    robustness, and comparison tests).
  - Reference: Wong, O. (2024). Modernising neutron spectrum unfolding for
    fusion applications. PhD Thesis. https://shura.shu.ac.uk/36014/


## [0.17.2] - 2026-08-19

### Added
- **AIC/AICc/BIC regularization selection for `unfold_lmfit`** — new
  `regularization_method` parameter (`'manual'` | `'aic'` | `'aicc'` | `'bic'`)
  on `unfold_lmfit()` / `Detector.unfold_lmfit()`. When a non-manual method is
  selected, the L1 (and for elastic net the L2) regularization strength is
  swept over a log-spaced grid of `n_lambda` candidates in `lambda_range`,
  each solved by lmfit and scored with the Akaike information criterion using
  effective degrees of freedom (ridge: `sum(s_i^2/(s_i^2+lambda))`; lasso and
  elastic net: active-set/SVD heuristic). The candidate minimizing the chosen
  criterion is used for the final unfolding. The selected values and the full
  sweep path are reported in the result dict (`selected_regularization`,
  `selected_regularization2`, `best_df`, `best_criterion_value`,
  `aic_bic_path`).
  - New public helper `select_regularization_aic_bic()` with a manual-parameter
    fallback when every candidate solve fails.
  - Tests: `tests/test_new_methods_fixed.py` (`TestUnfoldLmfit`,
    `TestDirectSolveFunctions`).


## [0.17.1] - 2026-08-17

### Changed
- **MCMC rework: log-space Ornstein-Uhlenbeck smoothness prior** — the Bayesian
  NUTS model (`unfold_mcmc`/`solve_bayesian_mcmc`) now models the spectrum on the
  log scale (`f = exp(theta)` with `theta ~ MvNormal(mu_prior, s * C_ou)`) instead
  of independent per-bin `HalfNormal` priors. The prior is anchored on a
  data-driven center: the user-supplied `initial_spectrum` when given, otherwise
  the non-negative least-squares solution of `A @ x = b`. This keeps the severely
  underdetermined unfolding problem positive, smooth and bounded in the null space
  of the response matrix, and the posterior mean matches deterministic solvers
  (e.g. `unfold_cvxpy`) on the IAEA reference-spectrum database.
  - New parameters: `lengthscale` (OU correlation length in energy bins, default
    3.0) and a now-effective `initial_spectrum` (previously unused). Defaults
    updated: `sigma_prior=0.05`, `lambda_prior=0.5`, `target_accept=0.95`.
  - Statistics extended with `lengthscale` and `prior_center`; the 95% HPD
    interval is computed with pure NumPy (avoids ArviZ version drift).
  - Tests: `tests/test_mcmc.py` (fake-PyMC wrapper/statistics tests plus real
    NUTS smoke tests).
  - **New example notebook `examples/30-smt.ipynb`** — demonstrates the SMT-based
    unfolding method (`unfold_smt()`), backed by the Z3 solver. Uses a reduced
    12-bin LANL grid (the full 60-bin system is too large for Z3's exact rational
    arithmetic) and covers both residual objectives: the default `objective='l2'`
    (exact KKT characterization of the least-squares optimum) and `objective='l1'`
    (the historical lexicographic fallback), each solving the Cf-252 benchmark
    exactly (`residual_norm = 0`), plus a Monte-Carlo uncertainty section and an
    L2-vs-L1 comparison panel.
- **Rewritten example `examples/29-MCMC_example.ipynb`** — demonstrates that
  `unfold_mcmc` works well: with a fast deterministic `unfold_cvxpy` result passed
  as `initial_spectrum`, the Cf-252 benchmark is recovered with R-hat ~1.00,
  ESS > 400, R² ≈ 0.99, total fluence within ~0.05 % and ICRP-116 dose rates
  within ~0.1 %, including a hierarchical-noise-model comparison.


## [0.17.0] - 2026-08-12

### Added
- **MCMC** draft version

### Fixed
- fix test of CS 


## [0.16.0] - 2026-08-10

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
- **CGLS, GKS and Tikhonov-TV unfolding** — three new Krylov/regularized
  methods:
  - `unfold_cgls()` / `solve_cgls()` — Conjugate Gradient for Least Squares
    with an optional `||L x||^2` Tikhonov term (`regularization`,
    `smoothness_order`), discrepancy-principle stopping via `noise_level`
    and non-negative spectrum via clamping.
  - `unfold_gks()` / `solve_gks()` — Generalized Krylov Subspace
    (Golub-Kahan bidiagonalization) with the regularization parameter
    selected automatically on the projected problem by GCV, the Discrepancy
    Principle or the L-curve (`regularization_method='gcv'|'dp'|'lcurve'|
    'manual'`). No a-priori spectrum is required.
  - `unfold_tikhonov_tv()` / `solve_tikhonov_tv()` — noise-constrained
    Tikhonov + total variation solved by an ADMM scheme adapted to 1D
    spectra (Gazzola & Gholami, 2022). The balancing parameter `beta` can
    be fixed or estimated adaptively (`beta='adapt'`), with `type_` selecting
    `'TT'` (TV + Tikhonov), `'TV'` (pure TV) or `'T'` (pure Tikhonov).
  - All three are registered in `unfold_combined()` pipelines (`"cgls"`,
    `"gks"`, `"tikhonov_tv"`) and documented in `docs/detector.rst`.
    Tests: `tests/test_krylov_tv.py`.
- **SAND-II, BUNKI, BUNKI-UT, OSEM, MAP-EM, BSREM and SART unfolding** —
  seven new multi-sphere / iterative-EM methods:
  - `unfold_sandii()` / `solve_sandii()` — the SAND-II geometric-mean ratio
    method (McElroy et al., 1967) with chi-square (`chi_fac=1`) or
    max-relative-deviation (`chi_fac=0`) stopping and optional per-detector
    `sigma`.
  - `unfold_bunki()` / `solve_bunki()` and `unfold_bunkiut()` /
    `solve_bunkiut()` — the BUNKI (SPUNIT) and BUNKI-UT (BON31G) multi-sphere
    unfolding algorithms with three-point spectral smoothing.
  - `unfold_osem()` / `solve_osem()` — ordered-subset EM (Hudson & Larkin,
    1994); `n_subsets=1` reduces to standard MLEM.
  - `unfold_mapem()` / `solve_mapem()` — one-step-late penalised EM
    (OSMAPOSL) with nearest-neighbour `quadratic`, `logcosh` or
    `relative_difference` priors over the energy axis.
  - `unfold_bsrem()` / `solve_bsrem()` — block-sequential regularised EM with
    a relaxation sequence (constant or callable) and a bin floor to prevent
    locking at zero; guaranteed convergence for non-convex priors.
  - `unfold_sart()` / `solve_sart()` — simultaneous algebraic reconstruction
    with relaxed, residual-normalised additive updates.
  - All seven are registered in `unfold_combined()` pipelines and documented
    in `docs/detector.rst` and `docs/overview.rst`. Tests:
    `tests/test_em_methods.py`.
- **FISTA, Hybrid-GMRES** new methods added: The Fast Iterative 
    Shrinkage-Thresholding Algorithm (FISTA) and
    The hybrid GMRES method combines the GMRES iterative solver with
    Tikhonov regularization applied to the projected problem at each
    iteration. The regularization parameter is selected automatically
    using GCV or discrepancy principle.
- **code refactoring** after vulture dead-code removal and pylint.

### Fixed
- **First-bin artifact in EM-family unfolding methods** — `unfold_bsrem`,
  `unfold_mapem`, `unfold_osem`, `unfold_sart`, `unfold_bunki`,
  `unfold_bunkiut` and `unfold_sandii` started from a flat all-ones initial
  guess. At the lowest energy bin the detector response is (near-)zero, so
  the iterative update never corrected that bin and it stayed pinned at the
  initial value (1.0 on the default GSF detector), while the reference
  spectra have `Phi[0] = 0`. The default initial spectrum now zeroes the
  first energy bin, giving exactly 0 there for all seven methods.
- **SART first-bin tail** — `solve_sart` additionally holds the first
  (lowest-energy) bin fixed at its initial-guess value during iteration,
  because SART's additive update is unconstrained at that edge and otherwise
  accumulates a spurious tail (up to ~0.3 on LANL/PTB response functions).
  - Regression tests in `tests/test_em_methods.py` (`TestFirstBinZero`)
    covering all seven methods across the GSF/PTB/LANL response functions.

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
