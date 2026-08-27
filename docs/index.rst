bssunfold Documentation
=======================

bssunfold is a Python package for neutron spectrum unfolding using various algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   detector
   interpretation
   examples
   reconst_comparison
   multires_cascade

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities. Iterative solvers are accelerated with Numba JIT compilation for 3–50x speedups.

Features
--------

- **Multiple Unfolding Algorithms** (60+ methods):
  - **Tikhonov-type**: CVXPY, qpsolvers (L1/L2/smoothness), Legendre basis, TSVD, EPIC (Equal Posterior Information Condition)
  - **Krylov/hybrid**: Lanczos, GKS (Golub-Kahan bidiagonalization + projected GCV/DP/L-curve), CGLS, FISTA (accelerated proximal gradient), Hybrid GMRES
  - **Iterative**: Landweber, MLEM (pure NumPy + ODL), MLEM-STOP (J-factor stopping), GRAVEL, Doroshenko, Kaczmarz
  - **Bayesian**: D'Agostini (Bayes), Bayes with spline regularisation, zfit (Poisson likelihood), full Bayesian MCMC (NUTS via pymc)
  - **Maximum Entropy**: MAXED (primal log-space dual minimisation), IMAXED, AMAXED, AMAXED-Regularization (Wong 2024 PhD thesis)
  - **Statistical Regularisation**: Turchin's method (StatReg), Fortran STREG1 port (Reconst)
  - **Optimisation-based**: lmfit (L1/L2/Elastic Net), Scipy direct (CG, GMRES, LSQR), Mystic (direct-search: fmin, Powell, diffev), SMT (exact constraint solving via Z3), Genetic (meta-heuristic: PSO, GA, DE, ES, EP, ABC, GWO, CMA-ES, NSGA-II via MEALPY), CS (compressive sensing), SCIP (pyscipopt), CPLEX (docplex), QUBO (quantum-inspired simulated annealing)
  - **Advanced proximal**: ODL-style Primal-Dual Hybrid Gradient (PDHG) and Douglas-Rachford splitting with TV (pure-NumPy)
  - **Evolutionary**: MAEO (multi-island NSGA-III / C-TAEA / AGE-MOEA-II / SPEA2 ensemble)
  - **Pipeline**: Combined (chaining), Cascade (sequential coarse-to-fine multi-resolution), Composite (adaptive ensemble / stacked generalization)
  - **Parametric**: FRUIT-style thermal/epithermal/fast model (lmfit, cvxpy SQP, qpsolvers SQP, combined); BON95 4-component model with directed-divergence iterations; hybrid parametric + iterative refinement

- **Numba JIT-Accelerated Iterative Solvers**:
  - ``@njit(cache=True)`` compiled inner loops for Doroshenko, Kaczmarz, MLEM, GRAVEL
  - 3–50x speedup on iterative solvers
  - Automatic disk caching; graceful fallback when numba is not installed

- **Radiation Dose Calculations**:
  - ICRP-116 conversion coefficients for effective dose

- **Comprehensive Data Management**:
  - Automatic response function processing
  - Uncertainty quantification via Monte Carlo methods

- **Advanced Visualization**:
  - Spectrum plotting with uncertainty bands
  - Detector reading comparisons