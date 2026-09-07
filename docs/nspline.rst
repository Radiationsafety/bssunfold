N-spline unfolding (Islamgulov & Lartsev, 2008)
===============================================

The ``unfold_nspline`` method implements the neutron spectrum unfolding
approach of

  R. F. Islamgulov, V. D. Lartsev, *Reconstruction of neutron spectra
  from activation measurements in the form of N-splines*,
  **Atomic Energy** 104(5), 295–302 (May 2008)
  (RFNC — VNIITF named after E. I. Zababakhin).

Method outline
--------------

The activation measurements are described by the integral system

.. math::

   Q_i = \int_0^\infty \sigma_i(E)\,\varphi(E)\,dE,
   \qquad i = 1,\dots,N, \qquad (1)

which is a classical ill-posed inverse problem.  Instead of solving for
:math:`n` bin values of :math:`\varphi(E)` directly, the paper
parameterises the spectrum by a specialised *neutron spline* (N-spline)
with basis functions

.. math::

   N_k(E) = \exp(a_k + q_k \ln E + r_k E),
   \qquad E_k \le E \le E_{k+1},\; k = 1,\dots,M, \qquad (2)

i.e. piecewise functions whose logarithm is linear both in :math:`\ln E`
and in :math:`E`.  This family contains the classical model spectra —
:math:`1/E`, Maxwellian evaporation :math:`\exp(-E/T)`, fission-like
:math:`\sqrt{E}\exp(-bE)`, two-component Maxwell + slowed-down
representations — as particular members, so the basis is close to
complete for reactor and accelerator spectra while using only
:math:`3M` parameters.

Continuity of the spline value and of its derivative at the interior
knots (Eqs. 3–4 of the paper),

.. math::

   a_k - a_{k+1} + u_k (q_k - q_{k+1}) + E_k (r_k - r_{k+1}) = 0,
   \qquad u_k = \ln E_k,

   (q_k - q_{k+1}) + E_k (r_k - r_{k+1}) = 0,

is assembled into the block matrix :math:`D` (Eq. 5),
:math:`X = (a, q, r)^T`, giving the constraint system :math:`DX = 0`
(:func:`~bssunfold.core.build_continuity_matrix`).  The pointwise
approximation of a tabulated spectrum (Eqs. 6–7) is a weighted
least-squares problem in the log domain with weights
:math:`w_{ki} = 1/\varepsilon_{ki}` subject to :math:`DX = 0`:

.. math::

   GX = Y, \qquad DX = 0,

solved via the KKT (Lagrange multiplier) system
(:func:`~bssunfold.core.fit_nspline`).

The activation equations are then solved by the *directed divergence
minimisation* loop (generalised MIRD algorithm of Lartsev, Preprint
RFNC-VNIITF No. 216, 2005; Tarasko, Preprint FEI No. 1446, 1983).  With
normalised measured activations :math:`p_i = Q_i/\sum_j Q_j` the
functional

.. math::

   H = \sum_i \left[ pN_i \ln \frac{pN_i}{p_i} - pN_i + p_i \right] \ge 0

(measured :math:`p_i` vs calculated :math:`pN_i`) is decreased by the
flux-conserving gradient iteration

.. math::

   \varphi_{n+1}(E) = \varphi_n(E)\,\bigl[1 - \Delta\mu_n
   (R_n(E) - \bar R_n)\bigr],

   R_n(E) = \sum_i \frac{p_i}{Q_i}\,\sigma_i(E)
   \ln\frac{pN_i}{p_i},

where :math:`\bar R_n` is the flux-weighted mean of :math:`R_n` and the
step :math:`\Delta\mu_n` starts from the paper's conservative value
:math:`0.1/\sup|R_n - \bar R_n|` and is halved while :math:`H` increases
(backtracking).  After *every* iteration the current spectrum is
re-fitted by the N-spline — the paper's regularisation, which makes the
loop act on :math:`3M` spline parameters instead of :math:`n` bin values
and avoids the nonlinearity / local-minimum difficulties of a direct
spline fit to the activation integrals.

Stopping criteria and quality control
-------------------------------------

Iterations stop when :math:`H` reaches the level corresponding to the
measurement errors,

.. math::

   H \le H_{\text{target}} = \tfrac12 \sum_i p_i \left(\frac{\Delta
   Q_i}{Q_i}\right)^2,

or when the relative decrease of :math:`H` per iteration falls below
``tol``.  The acceptability of the reconstruction is measured by the
paper's mean-squared residual

.. math::

   nev = \sqrt{\frac{1}{N-1} \sum_i \left(\frac{Qr_i - Q_i}
   {\Delta Q_i}\right)^2},
   \qquad \text{acceptable when } nev \le 1 + \frac{2}{\sqrt{N}},

with :math:`Qr_i` the activation integrals recalculated from the
unfolded spectrum.

Knots
-----

Knot sets used in the paper for the BARS-5, IGRIK (channel and surface)
and YAGUAR reactors are available as
:data:`~bssunfold.core.NSPLINE_KNOT_PRESETS`
(``"BARS5_channel"``, ``"IGRIK_channel"``, ``"IGRIK_surface"``,
``"YAGUAR_channel"``).  ``knots=None`` (default) builds a log-uniform
grid, and explicit knot sequences (MeV) are accepted as well.  User and
preset knots are clipped to the energy grid range and the outer knots
are extended so that the spline domain always spans the whole grid, as
in the paper.

API
---

.. autofunction:: bssunfold.core.unfold_nspline.unfold_nspline

.. autofunction:: bssunfold.core.unfold_nspline.solve_nspline

.. autofunction:: bssunfold.core.unfold_nspline.solve_nspline_full

.. autofunction:: bssunfold.core.unfold_nspline.fit_nspline

.. autofunction:: bssunfold.core.unfold_nspline.build_continuity_matrix

.. autofunction:: bssunfold.core.unfold_nspline.nspline_eval

.. autofunction:: bssunfold.core.unfold_nspline.auto_knots

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   det = Detector()
   result = det.unfold_nspline(
       readings,
       knots="BARS5_channel",     # or None (auto), or explicit knots
       continuity="C0C1",         # spline continuity option
       relative_uncertainty=0.05,
       max_iterations=300,
   )
   print(result["nev"], result["acceptable"])   # paper's statistic
   print(result["H_history"])                   # convergence trace

See ``examples/41-nspline.ipynb`` for a worked comparison with GRAVEL on
a synthetic spectrum, and ``examples/42-nspline-iaea.ipynb`` for
unfolding an IAEA Compendium Monte-Carlo BSA spectrum
(``t4-14-s.txt_1``) from GSF Bonner-sphere readings.

Notes and limitations
---------------------

* ``nev`` uses unweighted relative residuals: with readings spanning
  many orders of magnitude the weakest detector dominates the statistic
  (its contribution to :math:`H` is negligible because :math:`H` works
  with normalised activations).  In the paper's activation-foil
  applications all :math:`Q_i` are comparable and
  :math:`nev \approx 1`–2.
* Energy regions where all detector responses vanish are shaped by the
  C0/C1 spline continuation (prior-driven); the pointwise smoothing
  weights follow the paper's :math:`w = 1/\varepsilon` philosophy with
  sensitivity-graded errors.
* The method is verified by ``tests/test_nspline.py`` (32 tests): exact
  interpolation properties, continuity constraints, synthetic-spectrum
  recovery, the ``nev`` statistic and the ``Detector`` integration.
