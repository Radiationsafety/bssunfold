Minimum Fisher Information unfolding
====================================

The ``unfold_fisher_information`` method implements the **method of
minimum Fisher information (MFI)** — one of the four neutron-spectrum
reconstruction techniques applied by K. Miksuta-Michalik, A. Cubi,
M. Fabbri and M. Di Giacomo, *A feasibility study of neutron spectrum
reconstruction using synthetic activation data for JT-60SA*,
Plasma Phys. Control. Fusion **68** (2026) 105001 (eqs. 3-6).

Mathematical formulation
------------------------

Given the discrete activation system :math:`G f = A` (the paper's
eq. 1, written here as :math:`A x = b`), MFI minimizes the
Tikhonov-form functional of eq. (3) whose regularization operator is
the **Fisher information matrix** of eq. (4):

.. math::

   \min_x \; \|A x - b\|_2^2 + \alpha\, x^T B^T W B x,

where :math:`B` is the first- or second-derivative difference operator
and :math:`W` is the solution-dependent diagonal weight matrix of
eqs. (5)-(6):

.. math::

   W_{ii} = \frac{1}{f_i} \;\; (f_i > 0), \qquad
   W_{ii} = W_{\max} \;\; (f_i \le 0).

The penalty :math:`\sum_i (\Delta f_i)^2 / f_i` — the discretized
Fisher information :math:`\int (f')^2/f\, dE` — diverges whenever flux
is moved through a nearly-empty bin, which keeps spectral edges sharp
while still smoothing the high-flux regions.

Because :math:`W` depends on the solution, the method is **iterative**:
starting from a positive initial guess (flat by default), each step
solves the linear regularized system

.. math::

   (A^T A + \alpha\, B^T W B)\, x = A^T b,

clips the result to be non-negative, rebuilds the weights and repeats
until the relative change of the solution drops below ``tolerance`` or
``max_iterations`` is reached. The solver reports the iteration count
and convergence flag in the result dict (``iterations``,
``converged``).

Parameters
----------

* ``alpha`` — regularization strength :math:`\lambda^2`; ``alpha = 0``
  returns the clipped least-squares solution. The meaningful range
  depends on the units of the readings because :math:`W \sim 1/f`
  carries flux units; sweep it (see the example notebooks).
* ``derivative_order`` — order of the difference operator :math:`B`
  (1 or 2). With the energy grid supplied the rows approximate
  derivatives w.r.t. :math:`\ln E` (grid-aware penalty).
* ``w_max`` — the cap :math:`W_{\max}` of eq. (6); by default it is set
  adaptively per iteration to the inverse of 1e-4 of the current peak
  flux.
* ``damping`` — relaxation factor :math:`\omega \in (0, 1]` of the
  solution update, useful when the fixed point oscillates.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()
   result = detector.unfold_fisher_information(
       readings={"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034},
       alpha=1e-4,             # Fisher penalty strength
       derivative_order=1,
       max_iterations=50,
   )
   print(result["iterations"], result["converged"])

   # with an informed positive initial guess
   result = detector.unfold_fisher_information(
       readings, alpha=1e-4, initial_spectrum=prior_spectrum,
   )

Examples: ``examples/92-fisher-information-iaea.ipynb`` (method demo
with an alpha sweep and the paper's TR/MFI/ME/ML comparison panel) and
``examples/93-mfi-vs-tr-benchmark.ipynb`` (head-to-head MFI vs
Tikhonov benchmark with the paper-style per-region error table).

API reference
-------------

See also :doc:`detector` for the full ``Detector`` API reference.

.. autofunction:: bssunfold.core.unfold_fisher_information.unfold_fisher_information
   :no-index:

.. autofunction:: bssunfold.core.unfold_fisher_information.solve_fisher_information
   :no-index:
