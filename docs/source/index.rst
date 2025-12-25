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

## Overview

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities.

## Features

- **Multiple Unfolding Algorithms**:
  - Tikhonov regularization with convex optimization (CVXPY)
  - Landweber iterative method
  - Combined approach for improved accuracy

- **Radiation Dose Calculations**:
  - ICRP-116 conversion coefficients for effective dose

- **Comprehensive Data Management**:
  - Automatic response function processing
  - Uncertainty quantification via Monte Carlo methods

- **Advanced Visualization**:
  - Spectrum plotting with uncertainty bands
  - Detector reading comparisons