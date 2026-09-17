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
* Riis, N. A. B. et al. (2019). *pCN sampling for Bayesian inverse
  problems* (preconditioned Crank-Nicolson in CUQI/CUQIpy).
* Gelman, A. & Rubin, D. B. (1992). *Inference from iterative
  simulation using multiple sequences*, Statistical Science 7(4).
