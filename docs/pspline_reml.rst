P-spline mixed-model unfolding with REML smoothing selection
============================================================

The ``unfold_pspline_reml`` method is the Python analogue of the R
package ``LMMsolver`` (M. P. Boer, *LMMsolver: Linear Mixed Models with
Sparse Matrix Methods*, CRAN, 2023;
M. P. Boer, *Tensor product P-splines using a sparse mixed model
formulation*, Stat. Model. 23(5-6), 465-481, 2023): the unfolded
spectrum is a P-spline (Eilers & Marx, *Flexible smoothing with
B-splines and penalties*, Stat. Sci. 11(2), 89-121, 1996) whose
smoothness is selected automatically by restricted maximum likelihood
(REML) in a linear mixed-model (LMM) formulation (Wand & Ormerod,
*On semiparametric regression with O'Sullivan penalized splines*,
Aust. N. Z. J. Stat. 50(2), 179-198, 2008).

Method outline
--------------

The spectrum is parameterised with a B-spline basis,

.. math::

   x(E) = \sum_{s=1}^{N_s} c_s\, B_s(E) \;=\; B c,

so the Fredholm system :math:`b = A x` becomes the linear mixed model

.. math::

   b = A B c + \varepsilon, \qquad
   c \sim \mathcal{N}\!\left(0,\; \sigma_e^2\,\lambda\, G^{-1}\right),
   \qquad G = D^{(d)\,T} D^{(d)},

where :math:`D^{(d)}` is the :math:`d`-th order difference matrix
(:func:`~bssunfold.core.difference_matrix`, the classic P-spline
penalty) and :math:`\lambda` is the smoothing parameter.  The penalty
eigen-decomposition :math:`G = U \operatorname{diag}(g) U^T` splits the
coefficient space into

* a **fixed** part spanning the null space of :math:`G` (dimension
  :math:`d`) — an unpenalised polynomial trend,
  :math:`X = A B U_{\mathrm{fixed}}`;
* a **random** part spanning the range space with prior precision
  :math:`L = \operatorname{diag}(g_{\mathrm{random}}) > 0`,
  :math:`Z = A B U_{\mathrm{random}}`.

For a trial smoothing parameter :math:`\lambda` the coefficients follow
from the Henderson mixed-model equations (the system solved by
``LMMsolver::LMMsolve``):

.. math::

   \begin{bmatrix}
   X^T W X & X^T W Z \\
   Z^T W X & Z^T W Z + \lambda L
   \end{bmatrix}
   \begin{bmatrix} \hat\beta \\ \hat b \end{bmatrix}
   =
   \begin{bmatrix} X^T W y \\ Z^T W y \end{bmatrix},

with per-detector weights :math:`W` (uniform by default, or
:math:`w_i = 1/b_i` for counting statistics — ``weights="poisson"``).
The spectrum estimate is :math:`\hat x = X\hat\beta + Z\hat b`.

The smoothing parameter is estimated by maximising the REML profile
log-likelihood of the marginal model
:math:`y \sim \mathcal{N}(X\beta,\; \sigma_e^2 V)`,
:math:`V = I + \lambda Z L^{-1} Z^T`:

.. math::

   \ell_R(\lambda) = -\tfrac{1}{2}\Big[ (m - p_f)
   \log \hat\sigma_e^2(\lambda)
   + \log|V| + \log|X^T V^{-1} X| \Big],

optimised with Brent's method.  Because :math:`G` and the data term
live on very different scales, the search is performed on a
*scale-free* relative grid,
:math:`\lambda = \lambda_{\mathrm{ref}}\,\lambda_{\mathrm{rel}}` with
:math:`\lambda_{\mathrm{ref}}` equalising the average trace of both
terms and :math:`\lambda_{\mathrm{rel}} \in [10^{-6}, 10^{6}]`; both
:math:`\lambda` and :math:`\lambda_{\mathrm{rel}}` are reported.  A
fixed relative value can be forced with ``lam_relative``.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   det = Detector()
   result = det.unfold_pspline_reml(readings, weights="poisson")
   print(result["lam_relative"], result["ed"], result["reml_loglik"])

Diagnostics returned with the standard result dictionary:
``lam`` (absolute smoothing), ``lam_relative``, ``lam_ref``,
``sigma2`` (residual variance estimate), ``reml_loglik``, ``ed`` and
``ed_norm`` (effective dimension of the fit and its fraction of the
basis size), ``reml_converged``.  The effective dimension

.. math::

   \mathrm{ed} = p_f + \operatorname{tr}\!\left[
   (Z^T W Z + \lambda L)^{-1} Z^T W Z \right]

quantifies the flexibility actually used by the fit.

Notes
-----

* Requires at least ``diff_order + 2`` detector readings.
* The solution is linear in the data for a fixed :math:`\lambda`
  (like TSVD); non-negativity is enforced by clamping, so the method
  is best suited for smooth spectra without sharp low-energy edges.
* Only NumPy/SciPy are used — no additional dependencies.
