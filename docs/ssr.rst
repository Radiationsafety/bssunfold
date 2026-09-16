SSR unfolding: Sign-Simplicity-Regression solver (sisireg port)
===============================================================

The ``unfold_ssr`` method is the Python port of the R package
``sisireg`` 1.2.1 (L. Metzner, *sisireg: Sign-Simplicity-Regression-
Solver*, CRAN, GPL>=2).  The Sign-Simplicity-Regression (SSR) model of
Metzner (2020, *Trendbasierte Prognostik*; 2021, *Adaequates
Maschinelles Lernen*) is a nonparametric regression built on the
*signs* of the residuals: it seeks the most parsimonious (fewest
extrema) regression function that is statistically adequate with
respect to two sign criteria.

Sign criteria
-------------

For residuals ``r_i = y_i - mu_i`` and their signs ``s_i = sign(r_i)``
the model must satisfy

* the **partial sum criterion**: for every interval length ``k`` the
  maximum absolute sum of consecutive residual signs stays below the
  95% quantile of the partial sums,

  .. math::

     \max_t \Big| \sum_{j=t}^{t+k-1} s_j \Big| \;\le\;
     F(n, k) = \min\!\big(\sqrt{1 + 2.33 \ln(n)\, k},\; k\big);

* the **maximum run criterion**: with the 95% quantile of the maximum
  run length of equal signs,

  .. math::

     k_{\mathrm{run}}(n) = \lfloor 3.3 + 1.44 \ln(n) \rfloor,

  no window of :math:`k_{\mathrm{run}} + 1` consecutive residuals may
  sum to more than :math:`k_{\mathrm{run}}` in absolute value.

The regression function itself is computed with the quantised
Gauss-Seidel (QSOR) iteration of the package: every interior point is
replaced by the simplicitic linear interpolation of its neighbours and
the update is *reverted* whenever it would violate the partial sum
criterion with threshold :math:`h` (``fn``).  Iterating this
projection yields the most parsimonious statistically adequate
function.  The port reproduces the R/C semantics exactly, including
the rolling-median start values, the truncating integer conversions of
the ``.C`` interface and both the L1 (``ssr``/``ssr_ne``) and the L2
standardised-stencil (``ssr(..., funk=2)``) solvers.

Unfolding scheme
----------------

The SSR model regresses directly observed data, while unfolding only
observes the folded readings :math:`b = A x`.  ``solve_ssr`` therefore
alternates

1. a **data-fidelity step** — one multiplicative MLEM update of the
   spectrum (positivity preserving), and
2. an **SSR parsimony step** — a non-equidistant SSR QSOR sweep of the
   current spectrum over the energy grid, which removes
   sign-inadequate wiggles while bounding the deviation from the
   data-fit in every window.

The strength of the parsimony step is the partial sum threshold
:math:`h`.  Following Metzner's *minimum statistic*, ``fn="auto"``
starts from

.. math::

   h_{\mathrm{start}} = \lfloor 0.66\, F(n, k_{\mathrm{run}}) \rfloor,

decreases the threshold by one and stops at the first candidate whose
*folded* residuals violate the data-space sign adequacy (partial sum /
maximum run test applied to :math:`b - A x`) or whose parsimony
(number of extrema) exceeds that of the start model; the last adequate
candidate is returned — the analogue of the R ``ssr_ne_minR`` loop
transplanted to the unfolding system.  A fixed threshold can be forced
with ``fn=<int>``.

Usage
-----

.. code-block:: python

   from bssunfold import Detector

   det = Detector()
   result = det.unfold_ssr(readings)             # fn="auto"
   print(result["fn"], result["n_extrema"],
         result["ps_valid_data"], result["run_valid_data"])

   # fixed parsimony threshold
   result = det.unfold_ssr(readings, fn=3)

The pure regression building blocks are exported as well and can be
used standalone:

.. code-block:: python

   from bssunfold.core import ssr, ssr_ne, ssr_min_statistic_ne
   from bssunfold.core import max_run_quantile, partial_sum_valid
   from bssunfold.core import ssr_predict

Diagnostics returned with the standard result dictionary: ``fn``
(threshold used), ``fn_start`` (ladder start), ``k_run`` (run length
quantile), ``n_extrema`` (parsimony of the unfolded spectrum),
``ps_valid_data`` / ``run_valid_data`` (data-space adequacy of the
folded residuals), ``ssr_converged`` and, from ``solve_ssr_full``, the
full ``fn_ladder`` trail with per-candidate adequacy results.

Notes
-----

* Requires at least 8 energy bins (rolling-median windows) and 3
  detector readings; no upper limit — the QSOR sweeps are O(n)
  sequential passes.
* Non-negativity is preserved by construction (interpolation of a
  non-negative spectrum), so the method suits spectra with strong
  low-energy components.
* For fewer than 25 detector readings the partial sum test is
  trivially satisfied (its longest tested interval is ``m // 5``);
  the selection is then driven by the maximum run test and the
  parsimony criterion.
* Pure NumPy — no additional dependencies.  The R package sisireg is
  GPL (>= 2), compatible with the GPL-3 license of bssunfold.

Spatial regression: ``ssr3d``
-----------------------------

The 1-D solvers generalise to scattered data :math:`z(x, y)` on the
plane (R ``ssr3d``, ``R/ssr3d.R`` + ``src/ssr3d.c``).  The regression
surface is the most parsimonious *minimal surface* that is
statistically adequate with respect to the residual signs in the
k-quadrant-neighbourhood of every observation: the ``k`` nearest
observations in each of the four quadrants north-east, north-west,
south-west and south-east of a reference point.  Adequacy is tested
over the ``4k + 1`` nearest-neighbourhoods (the partial sum criterion
with threshold ``fn``); the Gauss-Seidel iteration replaces every
point by the exponential-distance-weighted mean of its quadrant
neighbours and *reverts* the update to the observation whenever the
new value violates the criterion.

.. code-block:: python

   from bssunfold.core import ssr3d, ssr3d_predict, ps_statistic_3d

   model = ssr3d(coords, dat)          # coords (n, 2), dat (n,)
   z = ssr3d_predict(model, queries)   # exponential weights
   z_ms = ssr3d_predict(model, queries, ms=True)  # 4-point minimal surface

   # partial sums vs. 95% quantiles (computation behind R psplot3d)
   ps, fn = ps_statistic_3d(coords, dat, model.mu)

Neighbourhood construction (``near_neighbors_quadrant``,
``near_neighbors``), the partial sum statistic (``ps_max_3d``,
``ps_statistic_3d``), the weighted means (``wmean``, ``wmean_exp``,
``wmean_ms``) and the model dataclass ``SSR3DModel`` are exported as
well.  The default ``k = max_run_quantile(n) / 2`` reproduces the R
true division including its truncating conversions.  The port was
verified to reproduce the original R output to machine precision
(same values of ``mu`` for scattered and jittered-grid data); on
exactly regular grids, distance ties make the neighbourhood
composition depend on the last bits of the floating-point distances,
so tiny deviations of the *tie handling* are possible there — as
between any two R builds.

Neural regression: ``ssrMLP``
-----------------------------

``ssrMLP`` (R ``ssrMLP.R``) is a two-hidden-layer perceptron with
sigmoid activations and linear output whose SGD training criterion is
not the least squares residual but Metzner's partial sum criterion:
the sum of the positive parts of the absolute input-neighbourhood
residual sign sums above the threshold, optionally combined with
squared residuals (``opt='ps_lse'``), with a curvature penalty of the
learned function over the input neighbourhoods (``opt='ps_l1'``), with
plain least squares (``opt='lse'``) or with user-supplied error and
factor functions (``opt='ext'``).  The port reproduces the R
semantics exactly, including two quirks of the original: the hidden
layer receives the *transposed* weight update (``t(t(temp) %*% O1)``),
which confines the configuration to square hidden layers
``hl[0] == hl[1]``, and the ``fn`` / ``alpha`` arguments of
``ssrmlp_train`` are accepted but never forwarded, so the criterion
defaults ``fn = 4`` / ``alpha = 1e-4`` apply (custom values are
possible through ``opt='ext'``).

.. code-block:: python

   from bssunfold.core import ssrmlp_train, ssrmlp_predict, fii_model

   model = ssrmlp_train(X, Y, opt="ps", max_iter=1000, rng=0)
   yp = ssrmlp_predict(X, model)
   importance = fii_model(model)   # per-input (bias included), sums to 1

The criterion building blocks (``check_ps``, ``err_ps``/``fac_ps``,
``err_lse``/``fac_lse``, ``err_ps_lse``/``fac_ps_lse``,
``err_ps_l1``/``fac_ps_l1``), the forward pass ``calc_out`` and the
per-sample factor importance ``fii_prediction`` are exported as well.
Two training epochs reproduce the original R result bit-exactly
(fixture-tested); longer runs can diverge in the last bits because
the sign criterion is discontinuous — a residual within BLAS noise of
zero can flip its sign — which is inherent to any cross-platform
reimplementation of the R algorithm.
