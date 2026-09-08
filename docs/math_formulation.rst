Mathematical Formulation
========================

This section collects the common variational, iterative, entropy-based,
Bayesian, and parametric formulations used throughout :mod:`bssunfold`.
It complements the method table in ``overview.rst`` with the underlying
optimization objectives, positivity constraints, regularizers, and stopping
criteria.

.. contents::
   :local:
   :depth: 2

General Inverse Problem
-----------------------

.. _math-general:

The neutron spectrum unfolding problem is modeled as a discretized Fredholm
integral equation of the first kind:

.. math::

   N_d = \int_0^\infty R_d(E)\Phi(E)\,dE + \varepsilon_d,
   \qquad \mathbf{N} = \mathbf{R}\mathbf{\Phi} + \boldsymbol{\varepsilon}.

Here ``N`` denotes the detector readings, ``R`` the response matrix, and
``Phi`` the non-negative spectrum to be reconstructed. Since the system is
highly underdetermined and ill-conditioned, all practical solvers introduce
regularization, positivity constraints, or early stopping rules.

Tikhonov Family
---------------

.. _math-tikhonov:

Methods: ``unfold_cvxpy``, ``unfold_qpsolvers``, ``unfold_tikhonov_tv``,
``unfold_tikhonov_legendre``, ``unfold_statreg``.

The generic penalized objective is

.. math::

   \min_{\mathbf{\Phi} \ge 0}
   \Phi_\alpha(\mathbf{\Phi})
   = \frac{1}{2}\|\mathbf{W}(\mathbf{R}\mathbf{\Phi} - \mathbf{N})\|_2^2
   + \frac{\alpha}{2}\|\mathbf{L}(\mathbf{\Phi} - \mathbf{\Phi}_0)\|_p^p.

For total-variation regularization, the penalty becomes

.. math::

   \min_{\mathbf{\Phi} \ge 0}
   \frac{1}{2}\|\mathbf{W}(\mathbf{R}\mathbf{\Phi} - \mathbf{N})\|_2^2
   + \alpha_{\mathrm{TV}} \sum_{j=1}^{K-1} |\Phi_{j+1} - \Phi_j|.

The solution is unique for ``alpha > 0`` under the standard convexity
assumptions. In QP-based implementations, convergence is monitored through the
Karush-Kuhn-Tucker residuals and the Morozov discrepancy principle.

Krylov and Hybrid Solvers
-------------------------

.. _math-krylov:

Methods: ``unfold_tsvd``, ``unfold_lanczos``, ``unfold_cgls``,
``unfold_gks``, ``unfold_hybrid_gmres``.

The TSVD solution is defined by the truncated singular value expansion

.. math::

   \mathbf{\Phi}_k = \sum_{i=1}^k \frac{\mathbf{u}_i^T\mathbf{N}}{\sigma_i}\mathbf{v}_i.

Hybrid Krylov methods project the problem onto a low-dimensional Krylov
subspace and solve a reduced regularized system. These methods exhibit
semi-convergence: early iterations recover the smooth signal component, while
later iterations amplify noise associated with the smallest singular values.
Practical stopping criteria are typically based on GCV or discrepancy rules.

Iterative Projection Methods
----------------------------

.. _math-iterative:

Methods: ``unfold_landweber``, ``unfold_kaczmarz``, ``unfold_randomized_kaczmarz``,
``unfold_sart``, ``unfold_fista``, ``unfold_doroshenko``.

The Landweber iteration with positivity projection is

.. math::

   \mathbf{\Phi}^{(k+1)}
   = \mathcal{P}_{\ge 0}\left(\mathbf{\Phi}^{(k)}
   + \omega \mathbf{R}^T(\mathbf{N} - \mathbf{R}\mathbf{\Phi}^{(k)})\right).

For FISTA, the accelerated proximal step reads

.. math::

   \mathbf{y}^{(k)} = \mathbf{\Phi}^{(k)}
   + \frac{t_{k-1}-1}{t_k}(\mathbf{\Phi}^{(k)} - \mathbf{\Phi}^{(k-1)}),
   \qquad
   \mathbf{\Phi}^{(k+1)} = \operatorname{prox}_{\lambda \gamma \|\cdot\|_1}
   \left(\mathbf{y}^{(k)} - \gamma \nabla f(\mathbf{y}^{(k)})\right).

Convergence of Landweber-type methods requires a step size bounded by the
inverse operator norm. In practice, the iteration is stopped by a relative
change threshold or a maximum number of iterations.

MaxEnt and Information Divergence
---------------------------------

.. _math-maxent:

Methods: ``unfold_maxed``, ``unfold_amaxed``, ``unfold_imaxed``,
``unfold_gravel``, ``unfold_directed_divergence``, ``unfold_sandii``,
``unfold_ferdor``.

The entropy-regularized reconstruction minimizes the relative entropy with
respect to a prior spectrum ``Phi_0``:

.. math::

   \min_{\mathbf{\Phi} > 0}
   S(\mathbf{\Phi}, \mathbf{\Phi}_0)
   = \sum_{j=1}^K \left[
   \Phi_j \ln\left(\frac{\Phi_j}{\Phi_{0,j}}\right)
   + \Phi_{0,j} - \Phi_j\right].

GRAVEL and related algorithms use multiplicative updates and are typically
stopped when the reduced chi-squared approaches unity or when the relative
change between iterations becomes sufficiently small.

EM and Poisson Likelihood Family
---------------------------------

.. _math-em:

Methods: ``unfold_mlem``, ``unfold_mlem_stop``, ``unfold_mlem_odl``,
``unfold_osem``, ``unfold_mapem``, ``unfold_bsrem``.

The Poisson log-likelihood maximization problem is

.. math::

   \max_{\mathbf{\Phi} \ge 0} L(\mathbf{\Phi})
   = \sum_{i=1}^M \left[
   N_i \ln\left(\sum_{j=1}^K R_{ij}\Phi_j\right)
   - \sum_{j=1}^K R_{ij}\Phi_j\right]
   - \beta V(\mathbf{\Phi}).

The classical MLEM update is multiplicative and preserves positivity. Because
of ill-posedness, regularized or early-stopped variants are preferred in
practice.

Bayesian and Stochastic Methods
-------------------------------

.. _math-bayesian:

Methods: ``unfold_bayes``, ``unfold_bayes_spline_regularization``,
``unfold_mcmc``, ``unfold_bayesian_parametric``, ``unfold_eki``.

The posterior density combines the likelihood and prior terms:

.. math::

   P(\mathbf{\Phi} \mid \mathbf{N})
   \propto P(\mathbf{N} \mid \mathbf{\Phi}) P(\mathbf{\Phi}).

For MCMC methods, convergence is assessed by standard diagnostics such as the
Gelman-Rubin statistic and effective sample size. For ensemble methods, the
ensemble is advanced until the spread stabilizes and the data misfit stops
decreasing materially.

Parametric Models
-----------------

.. _math-parametric:

Methods: ``unfold_parametric``, ``unfold_parametric2``,
``unfold_hybrid_parametric``, ``unfold_crystal_ball``, ``unfold_nspline``,
``unfold_zfit``, ``unfold_lmfit``.

The spectrum is represented by a low-dimensional parameter vector ``theta`` and
the corresponding data misfit is minimized:

.. math::

   \min_{\boldsymbol{\theta} \in \Theta}
   \chi^2(\boldsymbol{\theta})
   = \sum_{i=1}^M \left(
   \frac{\int_0^\infty R_i(E)\Phi(E;\boldsymbol{\theta})\,dE - N_i}{\sigma_i}
   \right)^2.

These methods are solved with nonlinear least squares, quasi-Newton updates,
or hybrid local/global optimizers, depending on the backend.

Integration Notes
-----------------

The mathematical section is referenced from the overview page and is rendered
with MathJax via ``sphinx.ext.mathjax``. This keeps the main method table compact
while providing a dedicated theoretical reference for the optimization
functional, non-negativity constraints, regularizers, and stopping criteria.
