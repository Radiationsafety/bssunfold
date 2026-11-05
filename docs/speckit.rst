SpecKit-style gradient unfolding with log-smoothness
===================================================

``bssunfold`` provides a gradient-descent spectrum unfolding method inspired
by the SpecKit toolkit: a least-squares data term plus a **log-smoothness**
regulariser on the spectrum,

.. math::

   F(x) = \frac{1}{m}\sum_{i=1}^{m}(A x - b)_i^2
        + \lambda\,\overline{\bigl(\Delta \log x\bigr)^2},

where :math:`\Delta \log x` is the first difference of :math:`\log x`.
The regulariser penalises rapid relative changes of the spectrum between
neighbouring energy bins and keeps the solution smooth in the logarithmic
domain, which is the natural scale for neutron spectra spanning many orders
of magnitude.

The implementation in :mod:`bssunfold.core.unfold_speckit` is a robust
adaptation of the original SpecKit heuristic:

* the gradient contains **both** the data term and the log-smoothness term;
* the step size is chosen by **backtracking line search**, so the objective
  is monotonically non-increasing;
* the initial spectrum is rescaled to match the measurements in the
  least-squares sense;
* stopping uses a **chi-square criterion** on the sum of squared residuals
  (95 % level, :math:`m` degrees of freedom) together with a
  relative-change tolerance.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()

   result = detector.unfold_speckit(readings, max_iterations=1000)
   print(result["spectrum"].shape)

   # tune the regularisation strength and the step size
   result = detector.unfold_speckit(
       readings,
       smoothness_lambda=0.05,
       learning_rate=1e-2,
       max_iterations=2000,
   )

   # Monte-Carlo uncertainty propagation
   result = detector.unfold_speckit(
       readings,
       calculate_errors=True,
       n_montecarlo=100,
       random_state=0,
   )

Parameters
----------

* ``smoothness_lambda`` -- weight of the log-smoothness regulariser
  (default ``0.01``).  Larger values give smoother spectra.
* ``learning_rate`` -- initial gradient step.  ``None`` (default) selects a
  conservative estimate from the Lipschitz constant of the data term.
* ``max_iterations`` -- maximum number of gradient steps (default ``1000``).
* ``tolerance`` -- relative-change convergence tolerance (default ``1e-6``).
* ``min_value`` (solver-level) -- lower bound enforced on every bin
  (default ``1e-12``).

Notes
-----

* The method needs no extra dependencies beyond NumPy/SciPy.
* As with the other iterative solvers, ``max_neutron_energy`` is supported
  through the standard ``Detector`` machinery.
* The original SpecKit code used a fixed learning rate of ``1.0`` and a
  chi-square threshold expressed in the mean-squared-error scale; this
  implementation keeps the same spirit but uses a sounder (backtracked,
  fully-graduated) update, so exact numerical reproduction of SpecKit
  results is not guaranteed -- the result is generally *better*.

API reference
-------------

.. autofunction:: bssunfold.core.unfold_speckit.solve_speckit

.. autofunction:: bssunfold.core.unfold_speckit.unfold_speckit