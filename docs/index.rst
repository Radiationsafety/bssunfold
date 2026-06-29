bssunfold Documentation
=======================

bssunfold is a Python package for neutron spectrum unfolding using various algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   detector
   examples

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

- **Multiple Unfolding Algorithms** (26 methods):
  - **Tikhonov-type**: CVXPY, qpsolvers (L1/L2/smoothness), Legendre basis, TSVD
  - **Iterative**: Landweber, MLEM (pure NumPy + ODL), MLEM-STOP, GRAVEL, Doroshenko, Kaczmarz
  - **Bayesian**: D'Agostini (Bayes), Bayes with spline regularisation
  - **Maximum Entropy**: MAXED (primal log-space dual minimisation)
  - **Statistical Regularisation**: Turchin's method (StatReg)
  - **Optimisation-based**: lmfit (L1/L2/Elastic Net), Scipy direct (CG, GMRES, LSQR)
  - **Pipeline**: Combined approach for chaining multiple methods
   - **Parametric**: FRUIT-style thermal/epithermal/fast model (lmfit, cvxpy SQP, qpsolvers SQP, combined); BON95 4-component model with directed-divergence iterations

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