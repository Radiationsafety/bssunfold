GEE unfolding: generalized estimating equations (gee port)
==========================================================

The ``unfold_gee`` method is the Python analogue of the R package
``gee`` 4.13-30 (V. Carey, T. Lumley, B. Ripley, CRAN, GPL-2; the
Liang-Zeger quasi-score solver).  GEE treats the detector spheres of a
single Bonner-sphere irradiation as a *correlated cluster* of
observations of the same measurement ``b = A x`` and estimates the
spectrum from the generalized estimating equations

.. math::

   U(x) \;=\; A^{\mathsf T} R(\alpha)^{-1} (b - A x)
   \;-\; \lambda\, G\, x \;=\; 0,

where :math:`R(\alpha)` is the *working correlation* matrix and
:math:`G = D_2^{\mathsf T} D_2` is the second-difference roughness
penalty (the ridge that makes the underdetermined system well posed;
``regularization = lam`` is relative to the mean diagonal of
:math:`A^{\mathsf T} R^{-1} A`, like in the other bssunfold methods).

Working correlation structures
------------------------------

* ``corstr="independence"`` — :math:`R = I` (the Liang-Zeger classical
  first model);
* ``corstr="exchangeable"`` (default) — :math:`R_{ii} = 1`,
  :math:`R_{ij} = \alpha`; :math:`\alpha` is estimated by the classic
  method-of-moments (mean of the off-diagonal products of the
  standardised residuals), clipped into the SPD region;
* ``corstr="ar1"`` — :math:`R_{ij} = \alpha^{|i-j|}` with the lag-1
  moment estimator.

Quasi-likelihood families
-------------------------

``family`` selects the variance function :math:`v(\mu)` (identity link):

* ``"gaussian"`` — :math:`v = 1` (default);
* ``"poisson"`` — :math:`v = \mu` (counts-like readings);
* ``"gamma"`` — :math:`v = \mu^2`.

Robust (sandwich) uncertainties
-------------------------------

The selling point of ``gee`` over plain weighted least squares is the
inference: the module reports the two canonical Liang-Zeger covariance
estimators for the unfolded spectrum,

.. math::

   V_{\mathrm{robust}} =
   N \big(A^{\mathsf T} \operatorname{diag}(\hat r^2) A\big) N,
   \qquad
   V_{\mathrm{naive}} = N H N,
   \qquad
   N = (A^{\mathsf T} R^{-1} A + \lambda G)^{-1},

where :math:`\hat r_i` are the Pearson residuals of the (working)
model.  The *robust* sandwich stays consistent when the working
correlation is misspecified; the *naive* estimator is the
model-based inverse of the penalised information matrix.  Both are
returned as ``robust_se`` / ``naive_se`` (and the full matrices in the
``_full`` diag), plus ``alpha``,
``phi``, ``pearson_chi2``, ``iterations`` and ``converged``.

Example
-------

.. code-block:: python

   from bssunfold import Detector

   det = Detector(RF_GSF)
   result = det.unfold_gee(readings)                    # gaussian/exchangeable
   result = det.unfold_gee(readings, family="poisson", corstr="ar1")
   print(result["robust_se"])

Notes
-----

* The iteration is the standard GEE/IRLS loop: solve the (penalised)
  GLS for the current :math:`R(\alpha)`, re-estimate ``alpha`` and the
  dispersion ``phi``, repeat until the relative spectrum update is
  below ``tolerance`` (or the projected iteration stalls on an active
  set).
* Non-negativity is enforced by the standard bssunfold clipping
  convention (GEE itself is unconstrained, like the R package).
* PURE NumPy: with 10-30 spheres the ``m x m`` working-correlation
  algebra is negligible.
