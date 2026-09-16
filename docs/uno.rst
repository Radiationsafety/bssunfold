Uno unfolding: Lagrange-Newton NLP presets (Uno port)
=====================================================

The ``unfold_uno`` method is the Python analogue of the R package
``Uno`` 2.x (B. Narasimhan, CRAN, MIT): the R package binds the C++
solver *Uno* — "Unifying Nonlinear Optimization" — described in
Vanaret & Leyffer (2024, `arXiv:2406.13454
<https://arxiv.org/abs/2406.13454>`_), which expresses non-linearly
constrained optimisation as a Lagrange-Newton method whose building
blocks (constraint relaxation, inequality handling, Hessian and
globalisation strategies) are freely combined, reproducing classical
solvers such as ``filterSQP`` and ``IPOPT`` by presets.

Unfolding NLP
-------------

The unfolding problem is posed as the smooth non-linear program

.. math::

   \min_x\; \tfrac12 \| W (A x - b) \|^2
   + \tfrac{\lambda}{2}\, \|D_2 x\|^2
   \quad\text{s.t.}\quad x \ge 0,

with the diagonal reading weights ``W`` (``weights="uniform" |
"poisson" | array``) and the relative second-difference ridge
``regularization = lam`` (like in the other bssunfold methods).

Presets
-------

* ``preset="filter_sqp"`` (default) — Lagrange-Newton SQP with the
  **exact** (constant) Hessian :math:`H = A^{\mathsf T} W^2 A +
  \lambda D_2^{\mathsf T} D_2` and the **Fletcher-Leyffer filter**
  globalisation on the (objective f, constraint violation
  :math:`v(x) = \|\min(x,0)\|^2`) pair.  Because the unfolding NLP is
  a *convex quadratic* objective with box inequalities, the
  exact-Hessian SQP sub-problem is the QP itself: it is solved in one
  Lagrange-Newton step through the classical active-set (Lawson-Hanson
  NNLS) solver on the equivalent stacked least-squares system
  :math:`[\,W A;\ \sqrt\lambda D_2\,]x = [\,W b;\ 0\,]`;
* ``preset="ipopt_like"`` — a primal-dual **interior-point** method in
  the IPOPT manner: the inequalities are handled by the log-barrier
  :math:`-\mu \sum \log x`, the Newton system is regularised with the
  barrier Hessian :math:`\operatorname{diag}(\mu / x^2)`, :math:`\mu`
  follows a geometric schedule with a fraction-to-the-boundary rule.
  The Hessian building block is selectable: ``hessian="exact"`` (the
  convex default) or ``hessian="bfgs"`` (a dense quasi-Newton
  approximation updated from the gradient differences — Uno's
  quasi-Newton block).

Diagnostics
-----------

The solver reports Uno's ``SolveStatistics``-style quality measures in
the result: ``preset``, ``hessian``, ``objective``
:math:`f(x^\*)`, ``constraint_violation``, ``dual_infeasibility``
(the infinity norm of the projected KKT stationarity residual,
relative to the initial gradient scale), ``n_iterations`` and
``converged``.

Example
-------

.. code-block:: python

   from bssunfold import Detector

   det = Detector(RF_GSF)
   result = det.unfold_uno(readings)                        # filterSQP
   result = det.unfold_uno(readings, preset="ipopt_like")   # IPOPT-style IPM
   result = det.unfold_uno(
       readings, preset="ipopt_like", hessian="bfgs",
       regularization=1e-2, weights="poisson",
   )
   print(result["objective"], result["constraint_violation"])

Notes
-----

* Pure NumPy/SciPy; the QP sub-solve uses the classical
  ``scipy.optimize.nnls`` active-set routine.
* Both presets enforce :math:`x \ge 0` by construction (projection /
  strictly feasible barrier iterates).
* ``hessian="bfgs"`` is only meaningful with ``preset="ipopt_like"``
  (the SQP preset uses the exact constant Hessian).
