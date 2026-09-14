B-spline MLEM unfolding (MLEM-BS)
=================================

The ``unfold_mlem_bs`` method implements the MLEM-BS algorithm of

  V. Mazankova, L. Torokova, D. Trunec, Z. Kopecky, Z. Matej,
  *Experimental Measurement of Neutron Flux and Its Mathematical Data
  Processing*, Proceedings of the 5th International Conference
  CNDGS'2026, Brno, Czech Republic, 7–10 September 2026,
  ISSN 2538-8959, https://doi.org/10.47459/cndcgs.2026.61.

Method outline
--------------

The measurement is described by the discrete Fredholm integral equation
of the first kind with Poisson-distributed counts,

.. math::

   n_i = \sum_j R_{ij}\, x_j, \qquad i = 1,\dots,I. \qquad (2)

Instead of solving for the :math:`n` bin values :math:`x_j` directly,
the paper represents the sought spectrum in a B-spline basis,

.. math::

   x(E) = \sum_{s=1}^{N_s} b_s\, B_s(E),

so that the effective system matrix becomes
:math:`\widetilde{R} = R\,B`
(:func:`~bssunfold.core.build_bspline_basis` assembles the design
matrix :math:`B`; clamped uniform or logarithmic knot grids are
supported, selected automatically from the span of the energy grid).
The B-spline coefficients :math:`b_s` are found with the regularized
MLEM iteration (Eq. 4 of the paper):

.. math::

   b_s^{(k+1)} = \frac{b_s^{(k)}}{\sum_i (RB)_{is}
   + \beta\, \partial P/\partial b_s}
   \sum_i (RB)_{is}\,
   \frac{n_i}{\sum_{s'} (RB)_{is'}\, b_{s'}^{(k)}},

where the penalty (Eq. 5) suppresses the noise influence,

.. math::

   P(b) = \|D^{(2)} b\|_2^2,
   \qquad \frac{\partial P}{\partial b}
   = 2\, (D^{(2)})^T D^{(2)} b,

with :math:`D^{(2)}` the second-derivative (second finite difference)
matrix (:func:`~bssunfold.core.second_difference_matrix`).

The solution is restricted to the *B-spline sieve* — the cone of
non-negative coefficients — following Z. Szkutnik, *B-splines and
discretization in an inverse problem for Poisson processes*, Journal of
Multivariate Analysis 93, 198–221 (2005) (reference [7] of the paper).
Positivity of the spectrum is preserved automatically by the
multiplicative MLEM update started from a non-negative sieve projection
of the initial spectrum.  In the overdetermined regime of the paper
(thousands of measured bins vs :math:`N_s` splines) the sieve
restriction also provides a strong noise-suppressing regularisation by
itself, even with :math:`\beta = 0`.

Penalty strength
----------------

The absolute penalty parameter :math:`\beta` of Eq. 4 is problem-scale
dependent (the paper uses :math:`\beta = 1.0 \times 10^{-17}` for its
counting setup).  For convenience the method additionally accepts
``beta_relative``, which defines the effective penalty as
:math:`\beta = \beta_{rel}\, \bar{s}` with :math:`\bar{s}` the mean
column sum of the effective matrix :math:`RB`, i.e. the penalty
gradient is measured relative to the MLEM sensitivity.  Exactly one of
``beta`` / ``beta_relative`` may be provided; with neither, the
iteration reduces to a pure sieve MLEM on the B-spline
parameterisation.

Automatic parameter selection
-----------------------------

Following the paper, the number of iterations and the parameters
:math:`N_s` (B-spline space dimension) and :math:`\beta` (penalty
strength) are selected by minimization of the goodness-of-fit
statistic (Eq. 6):

.. math::

   K_S = \left| \frac{\sum_i \left(n_i - \sum_s (RB)_{is}\, b_s^{(k)}\right)^2}
   {\sum_i \sum_s (RB)_{is}\, b_s^{(k)}} - 1 \right|,

which approaches zero when the residuals of the fit are consistent
with Poisson noise.  With ``auto_params=True`` the solver scans a grid
of candidate :math:`N_s` and relative penalty strengths
(:data:`~bssunfold.core.AUTO_BETA_RELATIVE_GRID`), runs the iteration
for each candidate, and keeps the combination with the smallest
:math:`K_S`; the selected values are reported in the
``auto_selection`` key of the result.  Even in manual mode the
iteration tracks the :math:`K_S` history and keeps the best iterate
(early stop with ``ks_patience`` iterations without improvement).

Confidence intervals
--------------------

The paper estimates confidence intervals of the unfolded spectrum with
a Poisson bootstrap (Eqs. 7–9): the backward-reconstructed counts
:math:`n^{(0)} = R\, x^{(0)}` are resampled as
:math:`n^{*(b)}_i \sim \mathrm{Poisson}(n^{(0)}_i)`, each replicate is
unfolded with the same settings, and the :math:`100(1-\alpha)\%`
interval is formed from the :math:`\alpha/2` and :math:`1-\alpha/2`
quantiles of the bootstrap distribution (a 95% CI corresponds to
:math:`\alpha = 0.05`).  Enable it with ``bootstrap_ci=True``; the
result then contains ``ci_low``, ``ci_high``, ``bootstrap_mean`` and
``bootstrap_std`` arrays.

.. note::

   The count-resampling bootstrap captures the *statistical* (Poisson)
   uncertainty.  In strongly underdetermined BSS systems (few spheres,
   many bins) the systematic model bias dominates and the intervals
   become optimistic; smaller :math:`N_s`, ``auto_params=True`` or an
   informative ``initial_spectrum`` make the intervals more
   conservative (see the module docstring of
   ``bssunfold.core.unfold_mlem_bs``).

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()
   readings = {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

   result = detector.unfold_mlem_bs(
       readings,
       n_basis=None,          # None -> automatic (log knots for wide grids)
       beta_relative=None,    # None -> pure sieve MLEM (beta = 0)
       max_iterations=500,
       bootstrap_ci=True,     # Poisson bootstrap CI (Eqs. 7-9)
       n_bootstrap=100,
       random_state=42,
   )

   spectrum = result["spectrum"]      # unfolded fluence spectrum
   ks = result["ks_final"]            # K_S statistic of Eq. 6
   ci = (result["ci_low"], result["ci_high"])

   # fully automatic selection of N_s, beta and the iteration count
   result = detector.unfold_mlem_bs(readings, auto_params=True)
   print(result["auto_selection"]["chosen"])

Diagnostics returned by the method: ``ks_history`` (per-iteration
:math:`K_S`), ``ks_final``, ``chi2_pearson``, ``n_basis``,
``interior_knots``, ``beta_effective``, ``coefficients`` (non-negative
B-spline coefficients), and, with ``auto_params=True``,
``auto_selection`` with the full candidate table.
