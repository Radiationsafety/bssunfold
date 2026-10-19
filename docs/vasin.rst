Hölder-space regularization for discontinuous spectra (Vasin & Serazhitnikova 2010)
===================================================================================

The ``unfold_vasin`` method implements the regularized approximation scheme of
V. V. Vasin and T. I. Serazhitnikova (*Regular algorithm for approximation of
irregular solutions for Fredholm integral equations of the first kind*,
Вычислительные технологии **15**(2) (2010) 15-23).  It is a Tikhonov scheme
whose stabilizer is deliberately **non-differentiable**: instead of penalizing
curvature it penalizes the Hölder (Lipschitz) seminorm of the solution, and it
measures the data misfit in the uniform norm.  A smooth penalty charges a jump
by how fast it occurs, the Hölder seminorm charges it by how tall it is, which
is what lets the method keep sharp spectral structures.

Mathematical formulation
------------------------

With grid distances :math:`d_{ij}` between the energy nodes and an exponent
:math:`0 < \\mu \\le 1`, the seminorm of eq. (3)-(4) of the paper is

.. math::

   [u]_{\\mu,\\infty} \\;=\\; \\max_{i \\ne j}
   \\frac{|u_i - u_j|}{d_{ij}^{\\mu}} .

The discrete problem (eq. (8), stabilizer at :math:`p = 2`) minimized here is

.. math::

   \\min_{u \\ge 0} \\; \\Phi(u), \\qquad
   \\Phi(u) = \\max_i \\left| w_i \\big( (Au)_i - b_i \\big) \\right|
   + \\alpha \\left[ \\tfrac12 \\sum_j h_j u_j^2
   + \\max_j |u_j| + [u]_{\\mu,\\infty} \\right],

where :math:`w_i` are the per-reading discrepancy weights and :math:`h_j` the
grid weights of the quadratic term.  The three penalty terms play different
roles: :math:`\\tfrac12 \\sum_j h_j u_j^2` makes the problem coercive (and
selects a unique minimizer on the null space of :math:`A`),
:math:`\\max_j |u_j|` bounds the amplitude and
:math:`[u]_{\\mu,\\infty}` bounds the oscillation.

The article minimizes :math:`\\Phi` by the prox-method (eq. (9)) with an inner
normalized-subgradient (eq. (10)) or Fejér level-set (eq. (12)) process.  This
port minimizes the **same functional exactly**, through its epigraph form:
:math:`\\Phi` is a sum of maxima of affine functions plus one quadratic term, so

.. math::

   \\min \\; t + \\alpha \\left( \\tfrac12 u^T H u + s + \\ell \\right),
   \\quad t \\ge \\pm w_i (Au - b)_i, \\;
   s \\ge |u|_\\infty, \\;
   \\ell \\ge [u]_{\\mu,\\infty}, \\; u \\ge 0

is a conic program that CVXPY solves in a few tens of milliseconds.  The
substitution is deliberate: the subgradient iteration of the paper needs tens of
thousands of steps to reach the level the exact solve attains immediately, and
the article's own convergence statement (Theorem 3) is about the minimizer of
:math:`\\Phi`, which is what is returned here.  Each maximum enters as a single
second-order-cone norm constraint rather than one linear constraint per bin and
pair, which keeps the program small enough for the conic solver to certify it.

Adaptation to Bonner-sphere unfolding
--------------------------------------

* **Distances.**  :math:`d_{ij}` are the logarithmic energy-grid distances
  rescaled to unit diameter, because the unfolding grid is logarithmic.
  ``solve_vasin`` accepts an arbitrary ``dist`` matrix, which is how the
  numerical experiment of the paper (a uniform grid on :math:`[0, 2]`) is
  reproduced in the test suite.
* **Quadrature weights.**  ``ln_steps``, the log-energy bin widths, enter the
  quadratic term as :math:`h_j`, which makes the stabilizer independent of the
  grid resolution.
* **Relative weights.**  With ``weights='relative'`` (default) row :math:`i` is
  scaled by :math:`1/|b_i|`, so the Chebyshev fit acts on *relative* residuals;
  with unit weights (``weights=None``, the paper's convention) the loudest
  sphere dictates the solution.
* **Underdetermined systems.**  With 10 detectors and 60 energy bins a
  non-negative exact fit usually exists, the achievable discrepancy is zero for
  every :math:`\\alpha`, and :math:`\\alpha` cancels out of the trade-off —
  ``mu`` carries all of the regularization.  :math:`\\alpha` regains influence
  for inconsistent readings (noise, dead-time corrections) and for the
  Monte-Carlo replicates.
* **The exponent.**  :math:`\\mu \\to 0` degenerates the seminorm into the
  oscillation :math:`\\max u - \\min u`: a jump and a ramp of equal height cost
  the same, which is exactly why the method preserves discontinuities.
  :math:`\\mu = 1` is the Lipschitz case, where a jump across one cell costs
  :math:`1/h` and is strongly suppressed.  The paper uses
  :math:`\\mu = 0.001`; the default here is ``mu=0.1``, which behaves the same
  on the unfolding grid while staying numerically tame.
* **Cost.**  The seminorm couples every pair of bins, so the epigraph program
  grows quadratically with the number of energy bins (:math:`n = 60` takes about
  30 ms, ``n`` of a few thousand becomes expensive).
* **Known bias.**  The :math:`\\max_j |u_j|` term shrinks peak heights — visible
  when comparing against MLEM or GRAVEL in
  ``examples/91-vasin-iaea.ipynb`` — the price of a prior that is cheap for
  discontinuities.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()
   readings = {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

   # default: Chebyshev discrepancy with relative weights, mu = 0.1
   result = detector.unfold_vasin(readings, alpha=1e-2, mu=0.1)

   # Lipschitz limit: strong suppression of steep flanks
   result = detector.unfold_vasin(readings, mu=1.0)

   # Euclidean instead of uniform discrepancy norm
   result = detector.unfold_vasin(readings, data_norm="l2")

   # paper's convention: unit weights
   result = detector.unfold_vasin(readings, weights=None)

API reference
-------------

See also :doc:`detector` for the full ``Detector`` API reference.

.. autofunction:: bssunfold.core.unfold_vasin.unfold_vasin
   :no-index:

.. autofunction:: bssunfold.core.unfold_vasin.solve_vasin
   :no-index:

.. autofunction:: bssunfold.core.unfold_vasin.holder_seminorm
   :no-index:
