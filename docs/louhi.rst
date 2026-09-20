LOUHI78 unfolding: constrained weighted least squares with generalized smoothing
================================================================================

The ``unfold_louhi`` method is the Python port of the LOUHI78 general
purpose unfolding program of J. T. Routti and V. Sandberg (*General
purpose unfolding program LOUHI78 with linear and nonlinear
regressions*, Computer Physics Communications **21** (1980) 119-135,
doi:10.1016/0010-4655(80)90021-4).  LOUHI is one of the classical
Bonner-sphere unfolding codes: it formulates the unfolding problem as a
constrained weighted least-squares fit with a generalized smoothing
term and determines the solution by quadratic programming.

Mathematical formulation
------------------------

Given the response matrix :math:`A`, the measurement vector
:math:`b`, the per-detector uncertainties :math:`\sigma_i` and the
default (a-priori) spectrum :math:`\phi_0`, LOUHI minimizes the total
chi-square functional

.. math::

   \chi^2(\phi) \;=\;
   \sum_i \left( \frac{b_i - (A\phi)_i}{\sigma_i} \right)^2
   \;+\; \lambda^2 \, \big\| L (\phi - \phi_0) \big\|_2^2

subject to the physical non-negativity constraints
:math:`\phi_j \ge 0`.  The smoothing operator :math:`L` supports three
orders:

* ``smooth_order=0`` — identity: the solution shrinks toward the
  a-priori spectrum;
* ``smooth_order=1`` (default) — first differences of the deviation
  from the a-priori spectrum;
* ``smooth_order=2`` — second differences (curvature) of the deviation.

Assembling the normal equations turns the problem into the quadratic
program

.. math::

   \min_{\phi \ge 0} \; \tfrac12 \phi^T H \phi - g^T \phi, \qquad
   H = 2\left( A^T W A + \lambda^2 L^T L \right), \quad
   g = 2\left( A^T W b + \lambda^2 L^T L \phi_0 \right),

with :math:`W = \operatorname{diag}(1/\sigma_i^2)`, which is solved by
Hildreth's iterative coordinate algorithm (the LSI step of LOUHI78):
every sweep minimizes one coordinate exactly and projects it onto its
non-negativity constraint, :math:`\phi_j \leftarrow \max(0,\, \phi_j -
\nabla_j / H_{jj})`.  The sweep stops when the relative change of the
quadratic objective drops below ``tolerance``.

Linear and nonlinear modes
--------------------------

In the *linear* mode the smoothing weight ``smoothness``
(:math:`\lambda`) is fixed by the user.  In the *nonlinear* mode
(``auto_smooth=True``) LOUHI adjusts :math:`\lambda` automatically by a
nonlinear regression: a golden-section search on
:math:`\log_{10}\lambda` drives the data chi-square to its expected
value :math:`\chi^2_{\text{target}}` (default: the number of detectors,
the expectation of the chi-square distribution).  This reproduces the
automatic smoothing-parameter search of the original program.

Error propagation
-----------------

The statistical error report of LOUHI78 propagates the measurement
uncertainties through the inverse of the Hessian restricted to the
*free* (strictly positive) bins of the active set; constrained bins at
zero carry no variance in the first-order propagation.  The helper
:func:`bssunfold.core.unfold_louhi.louhi_covariance` returns the
resulting per-bin standard deviations.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()
   result = detector.unfold_louhi(
       readings={"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034},
       smoothness=1.0,        # fixed smoothing weight (linear mode)
       smooth_order=1,        # first-difference smoothing operator
   )

   # nonlinear mode: automatic smoothing-weight regression
   result = detector.unfold_louhi(readings, auto_smooth=True)

   # explicit a-priori spectrum (anchored smoothing)
   import numpy as np
   phi0 = ...  # physically informed default spectrum
   result = detector.unfold_louhi(readings, initial_spectrum=phi0)

API reference
-------------

.. autofunction:: bssunfold.core.unfold_louhi.unfold_louhi

.. autofunction:: bssunfold.core.unfold_louhi.solve_louhi

.. autofunction:: bssunfold.core.unfold_louhi.louhi_smoothing_matrix

.. autofunction:: bssunfold.core.unfold_louhi.louhi_covariance
