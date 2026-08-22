bssunfold Documentation
=======================

bssunfold is a Python package for neutron spectrum unfolding using various algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   detector
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities.

Features
--------

- **Multiple Unfolding Algorithms** (16 methods):
  - **Tikhonov-type**: CVXPY, qpsolvers (L1/L2/smoothness), TSVD
  - **Iterative**: Landweber, MLEM (pure NumPy + ODL), GRAVEL, Doroshenko, Kaczmarz
  - **Bayesian**: D'Agostini (Bayes), Bayes with spline regularisation
  - **Maximum Entropy**: MAXED (primal log-space dual minimisation)
  - **Statistical Regularisation**: Turchin's method (StatReg)
  - **Optimisation-based**: lmfit (L1/L2/Elastic Net), Scipy direct (CG, GMRES, LSQR)
  - **Pipeline**: Combined approach for chaining multiple methods

- **Pluggable Spectral Basis**:
  - **BinBasis**: piecewise-constant (default)
  - **LegendreBasis**: Legendre polynomials on log-energy grid
  - **FourierBasis**: sin/cos series on log-energy grid
  - Any method can use any basis via the ``basis=`` parameter

- **Radiation Dose Calculations**:
  - ICRP-116 conversion coefficients for effective dose

- **Comprehensive Data Management**:
  - Automatic response function processing
  - Uncertainty quantification via Monte Carlo methods

- **Advanced Visualization**:
  - Spectrum plotting with uncertainty bands
  - Detector reading comparisons