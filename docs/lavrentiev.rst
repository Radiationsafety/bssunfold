Lavrentiev regularization unfolding
===================================

The ``unfold_lavrentiev`` method implements **Lavrentiev
regularization** — the classical regularization scheme for Fredholm
integral equations of the first kind proposed by M.M. Lavrentiev
(1962), developed in Tikhonov & Arsenin, *Solutions of Ill-Posed
Problems* (Wiley, 1977).

Mathematical formulation
------------------------

Given the ill-posed operator equation :math:`A z = b` with
:math:`A \in \mathbb{R}^{m \times n}` (:math:`m` detectors ≪ :math:`n`
energy bins), the **direct Lavrentiev scheme** replaces the operator
:math:`A` by the shifted operator :math:`A + \alpha I` and solves

.. math::

   (A + \alpha I)\, z_\alpha = b.

For rectangular :math:`A` the identity cannot be added directly; the
standard generalization applies the shifted-operator scheme to the
square positive semi-definite Gram operator :math:`B = A A^T`:

.. math::

   (B + \alpha I_m)\, y = b, \qquad z_\alpha = A^T y.

This *Gram form* is equivalent to zeroth-order Tikhonov
:math:`(A^T A + \alpha I) z = A^T b` via the push-through identity,
but solves an :math:`m \times m` system instead of :math:`n \times n`
— the practical choice for Bonner-sphere unfolding.

Forms
-----

The ``form`` keyword selects the variant:

* ``form="gram"`` (default) — Lavrentiev scheme on the Gram operator
  :math:`B = A A^T`; works for any :math:`A`.
* ``form="direct"`` — classical scheme :math:`(A + \alpha I) z = b`;
  requires a square response matrix.
* ``form="padded"`` — zero-pads rectangular :math:`A` to square and
  applies the direct scheme; for :math:`m < n` the recovered
  :math:`z_i = 0` for :math:`i > m` (prefer ``"gram"`` for BSS).
* ``form="iterated"`` — iterated Lavrentiev with Bakushinsky's
  geometric :math:`\alpha`-decay: :math:`\alpha_k = \alpha\, q^k`,
  :math:`K` defect-correction steps on the Gram operator.

Newton-Kantorovich discrepancy principle
----------------------------------------

When ``unfold_tikhonov_sobolev_dp`` is used with
``method="newton_kantorovich"``, the regularization parameter
:math:`\alpha^*` is found as the root of the generalized discrepancy
:math:`\rho(\alpha) = \|A z_\alpha - b\|^2 - \delta^2` by a
Newton-Kantorovich iteration on :math:`\log_{10}(\alpha)` using the
analytic derivative :math:`\rho'(\alpha)` (chain rule on the Tikhonov
solution). The iteration falls back to Brent's method when the Newton
step leaves the bracket, giving quadratic convergence in the typical
case with 30–50 % fewer Tikhonov solves.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()
   result = detector.unfold_lavrentiev(
       readings={"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034},
       alpha=0.05,           # regularization parameter
       form="gram",          # gram / direct / padded / iterated
   )

   # iterated Lavrentiev (Bakushinsky geometric alpha decay)
   result = detector.unfold_lavrentiev(
       readings, alpha=0.1, form="iterated", q=0.5, n_iterations=5,
   )

   # Tikhonov + discrepancy principle with Newton-Kantorovich root-finder
   result = detector.unfold_tikhonov_sobolev_dp(
       readings, noise_level=0.02, method="newton_kantorovich",
   )

API reference
-------------

See also :doc:`detector` for the full ``Detector`` API reference.

.. autofunction:: bssunfold.core.unfold_lavrentiev.unfold_lavrentiev
   :no-index:

.. autofunction:: bssunfold.core.unfold_lavrentiev.solve_lavrentiev
   :no-index:

.. autofunction:: bssunfold.core.unfold_tikhonov_sobolev_dp.generalized_discrepancy
   :no-index:

.. autofunction:: bssunfold.core.unfold_tikhonov_sobolev_dp.generalized_discrepancy_derivative
   :no-index:

.. autofunction:: bssunfold.core.unfold_tikhonov_sobolev_dp.alpha_finder_generalized_discrepancy
   :no-index:

.. autofunction:: bssunfold.core.unfold_tikhonov_sobolev_dp.alpha_finder_newton_kantorovich
   :no-index:
