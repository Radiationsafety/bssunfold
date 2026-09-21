CUQIpy Bayesian unfolding: uncertainty-quantified MCMC
======================================================

The ``unfold_cuqi`` method integrates `CUQIpy
<https://github.com/CUQI-DTU/CUQIpy>`_ — *Computational Uncertainty
Quantification for Inverse Problems* (Technical University of Denmark,
DTU) — as a full Bayesian back end for neutron spectrum unfolding.
Instead of a point estimate, the method draws samples from the joint
posterior distribution of the (log-)spectrum and reports the posterior
mean, per-bin standard deviations, highest-posterior-density (HPD)
credible intervals and MCMC convergence diagnostics (effective sample
size, Gelman-Rubin R-hat, acceptance rate).

CUQIpy is an **optional** dependency: install it with the package extra
``pip install bssunfold[cuqi]`` (or ``pip install cuqipy``).  When
cuqipy is missing, the package still imports normally (the flag
``bssunfold.CUQI_AVAILABLE`` is ``False``) and ``unfold_cuqi`` raises
an informative ``ImportError``.

**NumPy 2.4 note.**  Upstream ``cuqipy`` 1.5.1 declares
``numpy<=2.2.0``, which conflicts with this package's ``numpy>=2.4.1``
floor.  A maintained fork with the cap relaxed to ``numpy<2.5`` and the
NUTS ``int()``-on-1-element-array issue fixed is available at
https://github.com/Radiationsafety/CUQIpy (branch ``numpy2-support``,
dist version 1.5.2); it is validated to reproduce upstream results
bit-for-bit on identical seeded chains.  On NumPy >= 2.4 pre-install it
with::

   pip install "cuqipy @ git+https://github.com/Radiationsafety/CUQIpy@numpy2-support"

before ``pip install bssunfold[cuqi]`` (pip accepts the fork as
satisfying ``cuqipy>=1.5.0``).  The uv-managed environment (lockfile and
CI) resolves ``cuqipy`` from the fork automatically via
``[tool.uv.sources]``.  On NumPy <= 2.2 the plain PyPI ``cuqipy`` works
unchanged.

Bayesian model
--------------

**Terms.**  *Posterior distribution*
:math:`p(\boldsymbol{\theta} \mid b)` — the probability density of the
model parameters (here the log-spectrum) after the measurements are
taken; it combines the *likelihood* :math:`p(b \mid \theta)` with the
*prior* :math:`p(\theta)` via Bayes' theorem.  *Credible interval* —
an interval containing a stated fraction (e.g. 95 %) of the posterior
mass; unlike a frequentist confidence interval it makes a direct
probabilistic statement about the spectrum itself
(Gelman et al., 2013).  *HPD (highest posterior density) interval* —
the shortest credible interval at a given level; every point inside an
HPD interval has higher posterior density than any point outside it.

The unknown spectrum is modelled on the log scale,

.. math::

   f_j = \exp(\theta_j), \qquad j = 1, \dots, n,

which enforces non-negativity by construction and linearises the
multiplicative structure of the folded readings.  The data model is a
Gaussian likelihood on the (whitened, optionally relative-error scaled)
readings,

.. math::

   b \sim \mathcal{N}\!\big(A\, f(\theta),\; \sigma^2 I\big),
   \qquad \sigma = \texttt{noise\_level} \cdot \|b\|,

and the log-spectrum carries a smoothness prior anchored on a
data-driven center :math:`\mu` (the non-negative least-squares solution
of :math:`A x = b`, or the user-supplied ``initial_spectrum`` mapped to
the log scale):

* ``prior="gmrf"`` — a Gaussian Markov random field with a
  first- or second-order difference precision operator
  (:math:`\texttt{gmrf\_order}` = 1 or 2; higher is smoother);

* ``prior="ou"`` — a dense Ornstein-Uhlenbeck (exponentially
  decaying) correlation Gaussian prior with the correlation
  length ``lengthscale`` energy bins.

Samplers
--------

The posterior is explored with one of the CUQIpy samplers selected by
``sampler``:

* ``sampler="pcn"`` — preconditioned Crank-Nicolson (robust random-walk
  on the log scale; centred parameterisation
  :math:`\theta = \mu + t`);
* ``sampler="cwmh"`` — component-wise Metropolis-Hastings (per-bin
  updates, small memory footprint);
* ``sampler="mala"`` / ``sampler="ula"`` — (Metropolis-adjusted) Langevin
  algorithms using the analytic gradient of the log posterior; the
  sampler operates on a MAP-whitened parameterisation (Gauss-Newton MAP
  preconditioning) because the posterior curvature of the unfolding
  problem spans many orders of magnitude across directions.  ``ula`` is
  experimental and may diverge on stiff problems;
* ``sampler="nuts"`` — the No-U-Turn Sampler (native CUQIpy NUTS with
  dual-averaging step-size adaptation);
* ``sampler="gibbs"`` / ``sampler="gibbs_nuts"`` — hierarchical
  ``HybridGibbs`` sampling where the GMRF smoothness precision
  :math:`\delta` is *inferred from the data* through a conjugate Gamma
  hyperprior (:math:`\delta \sim \mathrm{Gamma}(\alpha, \beta)`, exact
  conjugate update each scan) while the spectral block is updated with
  pCN or NUTS respectively.  The posterior draws of the hyperparameter
  are returned in ``cuqi_stats["delta_samples"]``.

Multi-chain runs (``chains`` >= 2) are seeded independently and the
between-chain agreement is quantified with the Gelman-Rubin statistic.

Diagnostics
-----------

The result dictionary contains the usual unfolding fields
(``spectrum`` — posterior mean, ``spectrum_uncertainty`` — per-bin
posterior standard deviation, ``spectrum_lower`` / ``spectrum_upper`` —
HPD bounds at ``credible_level``, ``residual`` / ``residual_norm``)
plus a ``cuqi_stats`` sub-dictionary with the sampling diagnostics:

* ``mean``, ``median``, ``std`` — posterior summaries per energy bin;
* ``hpd_lower`` / ``hpd_upper`` — HPD credible interval bounds;
* ``ess`` — effective sample size of the (thinned) chains;
* ``rhat`` — Gelman-Rubin potential scale reduction factor;
* ``acc_rate`` — sampler acceptance rate(s);
* ``theta_samples`` / ``samples`` — raw posterior draws;
* ``delta_samples`` — posterior draws of the smoothness precision
  (hierarchical samplers only);
* ``prior_center`` — the NNLS-anchored prior center.

Interpreting the posterior (CUQIpy interpretability)
-----------------------------------------------------

A point estimate from a classical solver answers *"what spectrum fits
the data?"*; the CUQIpy posterior answers the interpretability
questions a point estimate cannot:

* *How uncertain is each energy bin?* — the per-bin posterior standard
  deviation and the HPD interval width.  Bins whose HPD interval spans
  decades are not resolved by the sphere set; narrow intervals flag
  data-determined regions.  This is the Bayesian counterpart of the
  local detector-sensitivity analysis of :doc:`interpretation`, but it
  is *global*: it marginalises over all parameter directions at once
  instead of perturbing one reading at a time (Tarantola, 2005).
* *Are the reported uncertainties trustworthy?* — only if the MCMC
  chains have converged and are long enough.  Check ``rhat`` and
  ``ess`` (below) before quoting any credible interval.
* *Which smoothness prior does the data support?* — with the
  hierarchical samplers the posterior of the smoothness precision
  ``delta_samples`` shows which smoothness levels the measurements
  themselves favour, a data-driven regularisation choice rather than a
  hand-tuned :math:`\alpha` (Gelman et al., 2013).
* *Is the non-negativity constraint distorting the result?* — the
  log-scale parameterisation :math:`f = \exp(\theta)` enforces
  positivity exactly, so the posterior mass can never cross zero; a
  posterior concentrated far from zero means the constraint is
  inactive, while a posterior piling up at the lower edge flags the
  same "artificial corner" that the shadow-price analysis of
  :doc:`interpretation` detects for QP solvers.

Diagnostics and how to read them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Gelman-Rubin :math:`\hat R`** (``cuqi_stats["rhat"]``) compares the
  between-chain variance :math:`B` with the within-chain variance
  :math:`W` of :math:`m` chains of length :math:`n`:

  .. math::

     \hat V = \frac{n-1}{n}\,W + \frac{B}{n},
     \qquad
     \hat R = \sqrt{\hat V / W}.

  :math:`\hat R \approx 1` means the chains agree — they sample the
  same posterior.  :math:`\hat R > 1.1` is a hard warning: the credible
  intervals are unreliable and the run needs more samples or a better
  tuned sampler (Gelman and Rubin, 1992).  The rank-normalised
  split-:math:`\hat R` computed by ArviZ, used here, is the modern
  standard with the stricter practical threshold
  :math:`\hat R < 1.01` (Vehtari et al., 2021).

* **Effective sample size (ESS)** (``cuqi_stats["ess"]``) converts
  :math:`N` correlated MCMC draws into the number of *independent*
  draws that would carry the same information,
  :math:`\mathrm{ESS} = N / (1 + 2\sum_t \rho_t)` with :math:`\rho_t`
  the autocorrelation at lag :math:`t` (Vehtari et al., 2021).
  An ESS below a few hundred means the reported posterior mean is
  itself noisy — increase ``n_samples`` or ``thin``.  ESS also
  quantifies *why* one sampler beats another: NUTS typically achieves
  a much higher ESS per draw than random-walk pCN on the strongly
  correlated unfolding posterior, at a higher cost per step
  (Hoffman and Gelman, 2014).

* **Acceptance rate** (``cuqi_stats["acc_rate"]``) — the fraction of
  proposals accepted.  For random-walk samplers the optimal regime is
  roughly 0.2-0.5 (Roberts et al., 1997); near 1 means tiny steps
  (slow exploration, low ESS), near 0 means the chain is stuck.
  NUTS adapts its step size to a target acceptance automatically
  (Hoffman and Gelman, 2014).

* **HPD width vs. bin energy** — plot
  ``result["spectrum_upper"] - result["spectrum_lower"]`` on a log
  scale: a narrow plateau means the spectrum shape is identified where
  the sphere responses overlap; flaring wings mark the thermal and
  high-energy edges where the Bonner sphere set loses resolution
  (Thomas and Alevra, 2002).

Relation to the pyoptexplain interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`interpretation` (pyoptexplain) explains the *optimisation*
problem — which constraints bind, what each detector is worth — by
analysing the solved QP.  CUQIpy explains the *inference* problem —
how much of the spectrum is actually determined by the data.  They are
complementary views of the same underdetermined system:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Question
     - pyoptexplain (:doc:`interpretation`)
     - CUQIpy (``unfold_cuqi``)
   * - Is the solution stable?
     - ±1-5 % reading perturbation tests
     - Posterior std / HPD width per bin
   * - Which detectors matter?
     - One-at-a-time detector importance
     - Global marginalisation over all directions
   * - Is :math:`x \ge 0` distorting?
     - Shadow prices / non-negativity relaxation
     - Posterior mass position near zero
   * - Is the regularisation right?
     - :math:`\alpha` sweep of point solutions
     - Hierarchical posterior of the smoothness precision
   * - Convergence certificate
     - Solver status, KKT residuals
     - :math:`\hat R < 1.01`, ESS, acceptance rate

Example: quantify which bins are data-determined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bssunfold import Detector

   det = Detector()
   result = det.unfold_cuqi(
       readings,
       sampler="gibbs_nuts",
       prior="gmrf",
       hierarchical=True,
       n_samples=2000,
       n_burnin=1000,
       chains=2,
       credible_level=95.0,
       random_state=42,
   )
   stats = result["cuqi_stats"]

   # 1) Convergence gate: only trust intervals after this passes
   assert stats["rhat"] < 1.01 and stats["ess"] > 200

   # 2) Per-bin relative uncertainty = interpretability map
   rel_unc = result["spectrum_uncertainty"] / result["spectrum"]
   # bins with rel_unc << 1 are data-determined;
   # bins with rel_unc ~ 1 are prior-dominated.

   # 3) HPD width per bin (log-scale view)
   hpd_width = result["spectrum_upper"] - result["spectrum_lower"]

Example
-------

.. code-block:: python

   from bssunfold import Detector

   det = Detector()                # default GSF response functions
   result = det.unfold_cuqi(
       readings,                   # measured sphere readings
       sampler="gibbs_nuts",       # hierarchical Gibbs with a NUTS block
       prior="gmrf",               # GMRF log-spectrum prior, order 1
       gmrf_order=1,
       hierarchical=True,          # infer the smoothness precision
       n_samples=2000,
       n_burnin=1000,
       thin=1,
       chains=2,                   # >= 2 chains enable R-hat
       credible_level=95.0,
       random_state=42,
   )

   spec = result["spectrum"]       # posterior mean spectrum
   lo, hi = result["spectrum_lower"], result["spectrum_upper"]
   stats = result["cuqi_stats"]
   print(stats["ess"], stats["rhat"], stats["acc_rate"])

A lower-level entry point :func:`~bssunfold.core.unfold_cuqi.solve_cuqi_bayesian`
exposes the same model on the raw response-matrix/response-vector level
for use outside the :class:`~bssunfold.Detector` workflow.

References
----------

* CUQIpy: *CUQIpy — Computational Uncertainty Quantification for
  Inverse Problems*, DTU, https://github.com/CUQI-DTU/CUQIpy
* Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.,
  Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
  `doi:10.1201/b16018 <https://doi.org/10.1201/b16018>`__
* Gelman, A. & Rubin, D. B. (1992). *Inference from iterative
  simulation using multiple sequences*, Statistical Science 7(4),
  457-472. `doi:10.1214/ss/1177011136
  <https://doi.org/10.1214/ss/1177011136>`__
* Hoffman, M. D., Gelman, A. (2014). The No-U-Turn sampler:
  adaptively setting path lengths in Hamiltonian Monte Carlo.
  *J. Machine Learning Research* **15**, 1593-1623.
* Riis, N. A. B. et al. (2019). *pCN sampling for Bayesian inverse
  problems* (preconditioned Crank-Nicolson in CUQI/CUQIpy).
* Roberts, G. O., Gelman, A., Gilks, W. R. (1997). Weak convergence
  and optimal scaling of random walk Metropolis algorithms.
  *Ann. Appl. Probab.* **7**, 110-120.
  `doi:10.1214/aoap/1034625254
  <https://doi.org/10.1214/aoap/1034625254>`__
* Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model
  Parameter Estimation*. SIAM.
  `doi:10.1137/1.9780898717791
  <https://doi.org/10.1137/1.9780898717791>`__
* Thomas, D. J., Alevra, A. V. (2002). Bonner sphere spectrometers — a
  critical review. *Nucl. Instrum. Meth. A* **476**, 12-20.
  `doi:10.1016/S0168-9002(01)01379-1
  <https://doi.org/10.1016/S0168-9002(01)01379-1>`__
* Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C.
  (2021). Rank-normalization, folding, and localization: an improved
  :math:`\hat R` for assessing convergence of MCMC. *Bayesian
  Analysis* **16**, 667-718. `doi:10.1214/20-BA1221
  <https://doi.org/10.1214/20-BA1221>`__
