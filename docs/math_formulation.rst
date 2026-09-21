Mathematical Formulation
========================

This section collects the common variational, iterative, entropy-based,
Bayesian, parametric, and first-order-optimisation formulations used
throughout :mod:`bssunfold`.
The package currently ships **93 public unfolding methods** in
``bssunfold.core``, which this page groups into **eleven method families**.
For each family we state the optimisation objective, the constraint
structure, the convergence facts that are actually guaranteed by the
literature, and the stopping criteria used in practice.  Terms are
defined where they first appear; every quantitative claim is backed by a
reference with a DOI in :ref:`math-refs`.

.. contents::
   :local:
   :depth: 2

.. _math-general:

General Inverse Problem
-----------------------

The neutron spectrum unfolding problem is modelled as a discretised
Fredholm integral equation of the first kind:

.. math::

   N_d = \int_0^\infty R_d(E)\,\Phi(E)\,dE + \varepsilon_d,
   \qquad \mathbf{N} = \mathbf{R}\,\boldsymbol{\Phi} + \boldsymbol{\varepsilon}.

Here :math:`\mathbf{N}\in\mathbb{R}^M` denotes the detector readings,
:math:`\mathbf{R}\in\mathbb{R}^{M\times K}` the response matrix, and
:math:`\boldsymbol{\Phi}\in\mathbb{R}^K_{\ge 0}` the non-negative
spectrum to be reconstructed.  The reference ``RF_GSF`` detector of this
package uses :math:`K = 60` logarithmically spaced energy bins spanning
1e-9 to 631 MeV and :math:`M = 10` Bonner spheres; multisphere systems in
general use between roughly :math:`M = 8` and :math:`M = 18` spheres.
Because :math:`M \ll K`, the discretised problem is severely
*underdetermined* (there exist infinitely many non-negative spectra that
reproduce the readings exactly), and because the singular values of
:math:`\mathbf{R}` decay exponentially fast (Hansen, 1994), it is
*ill-posed*: small perturbations of :math:`\mathbf{N}` can produce
arbitrarily large perturbations of :math:`\boldsymbol{\Phi}`.  Every
practical solver therefore combines one of the formulations below with
regularisation, positivity constraints, or early stopping.

**Terms used throughout this page.**

* **Response matrix** :math:`\mathbf{R}` — the :math:`M\times K` table of
  detector sensitivities: entry :math:`\mathbf{R}_{ij}` is the count rate
  that sphere :math:`i` registers per unit fluence in energy bin
  :math:`j` (IAEA, 2001).
* **Regularisation parameter** :math:`\alpha` — the weight that trades
  data fit against smoothness (or another prior belief).  Too small:
  the solver fits noise (*over-fitting*, oscillating spectrum).  Too
  large: the solver ignores the data (*over-smoothing*, biased fluence
  and dose).  Selection rules are discussed in :ref:`math-selection`.
* **Condition number** :math:`\kappa(\mathbf{R}) =
  \sigma_{\max}/\sigma_{\min}` — the ratio of extreme singular values;
  it lower-bounds the relative error amplification
  :math:`\|\delta\boldsymbol{\Phi}\|/\|\boldsymbol{\Phi}\| \lesssim
  \kappa(\mathbf{R})\,\|\delta\mathbf{N}\|/\|\mathbf{N}\|`
  (Tikhonov et al., 1995).  Bonner-sphere matrices are
  ill-conditioned precisely because many columns (energy responses of
  adjacent sphere sizes) are nearly collinear.
* **Prior spectrum** :math:`\boldsymbol{\Phi}_0` — a physically
  plausible initial guess (flat, Watt fission, or a catalogue shape)
  used by ratio, entropy, and Bayesian methods.  A good prior speeds up
  convergence and stabilises ill-conditioned directions, but a wrong
  prior biases the result towards itself — the classic
  bias-variance trade-off of regularisation
  (Reginatto, 2010; Engl, Hanke and Neubauer, 1996).

**Noise model.**  Counting statistics make the readings Poisson
distributed,

.. math::

   N_i \sim \operatorname{Poisson}\!\left((\mathbf{R}\boldsymbol{\Phi})_i\right),
   \qquad i = 1,\dots,M.

Most solvers in :mod:`bssunfold` work with the standard Gaussian
approximation of this model, :math:`\boldsymbol{\varepsilon}\sim
\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})` with diagonal
:math:`\boldsymbol{\Sigma}`, where :math:`\sigma_i^2 \approx N_i` for
pure counting statistics; an additional relative-uncertainty floor (a
systematic component, e.g. ``sigma_factor`` in ``unfold_maxed``) is
added in practice to account for calibration and response-matrix
uncertainties.  The Gaussian likelihood underlies all
:math:`\chi^2`-based objectives in this page, while the Poisson
likelihood is used explicitly by the EM family (``unfold_mlem`` and
relatives).

**Semi-convergence.**  Iterative methods applied to this problem share
one important property: after an initial phase in which the smooth,
data-dominated components of the solution are recovered, further
iterations start amplifying noise associated with the smallest singular
values of :math:`\mathbf{R}`.  This *semi-convergence* behaviour is not
restricted to Krylov methods — it is characteristic of Landweber and
CGLS iterations, ART/SART, MLEM and its Bayesian (D'Agostini) relative
alike (Engl, Hanke and Neubauer, 1996; Hansen, 1994).  Consequently the
iteration number itself acts as a regularisation parameter, and every
iterative family in this package offers or recommends an early-stopping
rule, most commonly the discrepancy principle or GCV (see
:ref:`Parameter selection <math-selection>`).

.. _math-selection:

Parameter Selection and Stopping Criteria
-----------------------------------------

Several families share the same parameter-selection machinery, so the
definitions are collected once here.

**Discrepancy principle (Morozov).**  Let :math:`\delta` be a bound on
the data noise, :math:`\|\boldsymbol{\varepsilon}\|_2 \le \delta`
(e.g. :math:`\delta^2 = \sum_i \sigma_i^2`).  The regularisation
parameter :math:`\alpha` (or the stopping index :math:`k`) is chosen as
the smallest value for which the residual satisfies

.. math::

   \|\mathbf{R}\boldsymbol{\Phi}_\alpha - \mathbf{N}\|_2 \le \tau\,\delta,

with a safety factor :math:`\tau > 1`, typically :math:`\tau \in
[1.01,\,1.1]` (Morozov, 1984; Engl, Hanke and Neubauer, 1996).  When the
:math:`M` residuals are scaled by their known standard deviations
:math:`\sigma_i`, the reduced criterion is
:math:`\chi^2 \approx M`: for :math:`M` independent bins with correctly
known variances, :math:`\chi^2` is asymptotically
:math:`\chi^2_M`-distributed, with expectation :math:`M` and variance
:math:`2M`, so values outside roughly :math:`M \pm \sqrt{2M}` indicate
over-fitting (chi-square too small, noise fitted) or over-regularisation
(chi-square too large).  This :math:`M \pm \sqrt{2M}` band is used by
several solvers (e.g. the automatic smoothing-weight bisection in
``unfold_ferdor`` and the entropy-weight calibration of ``unfold_maxed``
derivatives) as a rule of thumb, not as a rigorous test.

**Generalised cross-validation (GCV).**  For a filtering or iterative
method whose effective influence matrix is :math:`\mathbf{A}_k`, choose
:math:`k` minimising

.. math::

   G(k) = \frac{\|\mathbf{R}\boldsymbol{\Phi}_k - \mathbf{N}\|_2^2}
   {\operatorname{tr}(\mathbf{I} - \mathbf{A}_k)^2},

which requires no noise-level estimate (Golub, Heath and Wahba, 1979).

**L-curve.**  For a family of regularised solutions
:math:`\boldsymbol{\Phi}_\alpha`, plot
:math:`\log\|\mathbf{L}\boldsymbol{\Phi}_\alpha\|_2` against
:math:`\log\|\mathbf{R}\boldsymbol{\Phi}_\alpha - \mathbf{N}\|_2` and
select the point of maximum curvature (the corner) as the best
compromise between residual size and solution norm (Hansen, 1992).

**KKT residuals.**  For the QP-based solvers (``unfold_cvxpy``,
``unfold_qpsolvers``, ``unfold_interpret``), the Karush-Kuhn-Tucker
residuals reported at termination measure *first-order optimality of
the numerical solver* — they certify that the convex program has been
solved to the requested tolerance.  They are solver diagnostics, not a
statement about statistical convergence of the unfolding itself; the
statistical quality of the unfolded spectrum is still governed by the
regularisation choice and the noise model.

.. _math-tikhonov:

Variational (Tikhonov-Type) Methods
-----------------------------------

Methods: ``unfold_cvxpy``, ``unfold_qpsolvers``, ``unfold_tikhonov_tv``,
``unfold_tikhonov_legendre``, ``unfold_statreg``, ``unfold_reconst``,
``unfold_epic``, ``unfold_scipy_direct_method``, ``unfold_cs``,
``unfold_nnksvd``, ``unfold_docplex``, ``unfold_scip``, ``unfold_smt``,
``unfold_qubo``, ``unfold_interpret``.

The generic penalised objective is

.. math::

   \min_{\boldsymbol{\Phi} \ge 0}\;
   \frac{1}{2}\,\bigl\|\mathbf{W}\,(\mathbf{R}\boldsymbol{\Phi} -
   \mathbf{N})\bigr\|_2^2
   + \frac{\alpha}{2}\,\bigl\|\mathbf{L}\,
   (\boldsymbol{\Phi} - \boldsymbol{\Phi}_0)\bigr\|_p^p,

where :math:`\mathbf{W}` encodes the measurement covariances,
:math:`\mathbf{L}_0 = \mathbf{I}`, :math:`\mathbf{L}_1` (first
difference) or :math:`\mathbf{L}_2` (second difference / Legendre-basis
derivatives) selects the roughness penalty, and
:math:`p \in \{1, 2\}`.  For :math:`p = 2` the unconstrained objective
is quadratic; if :math:`\ker(\mathbf{R}) \cap \ker(\mathbf{L}) =
\{\mathbf{0}\}`, the Hessian
:math:`\mathbf{R}^T\mathbf{W}^T\mathbf{W}\mathbf{R} +
\alpha\,\mathbf{L}^T\mathbf{L}` is positive definite and the minimiser
is unique (Tikhonov et al., 1995; Engl, Hanke and Neubauer, 1996).
This is the typical situation for :math:`\mathbf{L} = \mathbf{L}_1` or
:math:`\mathbf{L}_2` on the log-spaced grid of a Bonner sphere system.

For total-variation regularisation the penalty becomes

.. math::

   \min_{\boldsymbol{\Phi} \ge 0}\;
   \frac{1}{2}\,\bigl\|\mathbf{W}\,(\mathbf{R}\boldsymbol{\Phi} -
   \mathbf{N})\bigr\|_2^2
   + \alpha_{\mathrm{TV}} \sum_{j=1}^{K-1}
   |\Phi_{j+1} - \Phi_j|.

TV regularisation is convex but **not strictly convex**: for
:math:`p = 1` uniqueness of the minimiser is *not* guaranteed, and
solutions may be non-unique along flat segments.  The statement
"unique for :math:`\alpha > 0`" therefore applies only to the
:math:`p = 2` (quadratic) case above.  ``unfold_tikhonov_tv`` uses the
standard smoothed-TV approximation to retain differentiability.

**Closed-form statistical regularisation.**  ``unfold_statreg``
(Turchin's method, cf. the RECONST algorithm ported by
``unfold_reconst``) solves the linear system directly:

.. math::

   \boldsymbol{\Phi}_\alpha =
   \bigl(\mathbf{R}^T\boldsymbol{\Sigma}^{-1}\mathbf{R}
   + \alpha\,\boldsymbol{\Omega}\bigr)^{-1}
   \mathbf{R}^T\boldsymbol{\Sigma}^{-1}\mathbf{N},

implemented as ``solve(RᵀΣ⁻¹R + αΩ, RᵀΣ⁻¹N)`` with a fallback to
``lstsq``.  This closed form is exact **only for the unconstrained
problem**: the moment a non-negativity constraint
:math:`\boldsymbol{\Phi}\ge 0` is enforced, the problem becomes a
quadratic program and the closed-form solution no longer applies (it
is then merely the solution of the equality-constrained relaxation).

**Sparse formulations.**  ``unfold_cs`` (compressed sensing with OMP /
K-SVD style dictionaries), ``unfold_nnksvd`` (non-negative K-SVD
dictionary learning) and the QUBO/Ising formulation ``unfold_qubo``
express the solution as sparse over-complete expansions; they are
solved with the corresponding combinatorial or convex optimisation
backends (``unfold_docplex``, ``unfold_scip``, ``unfold_smt``).
Convergence guarantees in this family are those of the underlying
numerical backend, plus the general regularisation theory above.

.. _math-krylov:

Spectral Filtering and Krylov Methods
-------------------------------------

Methods: ``unfold_tsvd``, ``unfold_lanczos``, ``unfold_cgls``,
``unfold_gks``, ``unfold_hybrid_gmres``.

The TSVD solution is defined by the truncated singular value expansion

.. math::

   \boldsymbol{\Phi}_k = \sum_{i=1}^{k}
   \frac{\mathbf{u}_i^T\mathbf{N}}{\sigma_i}\,\mathbf{v}_i,

where :math:`\mathbf{u}_i`, :math:`\mathbf{v}_i` and :math:`\sigma_i`
are the singular vectors and values of :math:`\mathbf{R}`.  The
truncation index :math:`k` is chosen by GCV, the discrepancy principle
or the L-curve (see :ref:`math-selection`).  In the implementation of
``unfold_tsvd`` the small singular-value contributions are discarded
and any remaining negative bins are clipped,
:math:`\boldsymbol{\Phi}\leftarrow\max(0, \boldsymbol{\Phi})`.  This
post-hoc clipping is a heuristic projection applied *after* the
filtering step; it is not equivalent to solving the non-negativity-
constrained filtering problem exactly.

CGLS (``unfold_cgls``, Hestenes and Stiefel, 1952) and LSQR-type
recurrences (Paige and Saunders, 1982) applied to the regularised
least-squares problem converge towards the (Tikhonov-regularised)
solution while implicitly filtering the singular spectrum; the
Krylov-subspace methods ``unfold_gks`` and ``unfold_hybrid_gmres``
build the Golub-Kahan bidiagonalisation

.. math::

   \mathbf{R}\,\mathbf{V}_m = \mathbf{U}_{m+1}\,\mathbf{B}_m

(Golub and Kahan, 1965) and solve the projected problem with an
explicit Tikhonov regularisation term whose parameter is re-selected at
each iteration — the *hybrid* strategy (Gazzola, Hansen and Nagy,
2018).  Hybrid methods delay but do not eliminate semi-convergence; the
inner regularisation parameter is chosen by GCV or discrepancy rules.
All methods in this section exhibit the semi-convergence behaviour
described in :ref:`math-general` and must be stopped early.

.. _math-iterative:

Algebraic Iterative Methods
---------------------------

Methods: ``unfold_landweber``, ``unfold_kaczmarz``,
``unfold_randomized_kaczmarz``, ``unfold_sart``, ``unfold_fista``,
``unfold_doroshenko``.

The Landweber iteration with positivity projection is

.. math::

   \boldsymbol{\Phi}^{(k+1)}
   = \mathcal{P}_{\ge 0}\!\left(\boldsymbol{\Phi}^{(k)}
   + \omega\,\mathbf{R}^T\bigl(\mathbf{N}
   - \mathbf{R}\boldsymbol{\Phi}^{(k)}\bigr)\right),

which converges for :math:`0 < \omega < 2/\sigma_{\max}^2` where
:math:`\sigma_{\max}` is the largest singular value of
:math:`\mathbf{R}` (``unfold_landweber`` uses the conservative step
:math:`\omega = 1/\sigma_{\max}^2`).  Kaczmarz/ART-type updates
(``unfold_kaczmarz``) project onto the hyperplane defined by one
detector row at a time; the *randomized* Kaczmarz variant
(``unfold_randomized_kaczmarz``) selects rows with probability
proportional to :math:`\|\mathbf{R}_i\|_2^2` and converges linearly in
expectation at the rate
:math:`\mathbb{E}\,\|\boldsymbol{\Phi}^{(k)} -
\boldsymbol{\Phi}_\star\|_2^2 \le (1 - \sigma_{\min}^2 /
\|\mathbf{R}\|_F^2)^k\,\|\boldsymbol{\Phi}^{(0)} -
\boldsymbol{\Phi}_\star\|_2^2` (Strohmer and Vershynin, 2009).

For SART the implementation in ``unfold_sart`` follows the classical
row-and-column-normalised scheme (Andersen and Kak, 1984):

.. math::

   \boldsymbol{\Phi}^{(k+1)} = \boldsymbol{\Phi}^{(k)}
   + \frac{\lambda}{\bigl(\mathbf{R}^T\mathbf{1}\bigr)_j + \epsilon}\,
   \sum_i \frac{\mathbf{R}_{ij}\,
   \bigl(N_i - (\mathbf{R}\boldsymbol{\Phi}^{(k)})_i\bigr)}
   {\bigl(\mathbf{R}\,\mathbf{1}\bigr)_i + \epsilon},

i.e. the residual is normalised by the forward-projected unit spectrum
and the update by the back-projected unit sensitivity, with relaxation
:math:`\lambda(n)` (constant or sequence; convergence of the exact SART
recursion is guaranteed for :math:`0 < \lambda \le 1`, and more
generally the algebraic family requires :math:`0 < \lambda < 2`;
Jiang and Wang, 2003).  Note that this is the *simultaneous* scheme —
all rows contribute in each sweep — which is distinct from the
purely row-by-row ART/Kaczmarz recursion.

For FISTA the accelerated proximal step reads

.. math::

   \mathbf{y}^{(k)} = \boldsymbol{\Phi}^{(k)}
   + \frac{t_{k-1}-1}{t_k}\bigl(\boldsymbol{\Phi}^{(k)} -
   \boldsymbol{\Phi}^{(k-1)}\bigr),
   \qquad
   \boldsymbol{\Phi}^{(k+1)} =
   \operatorname{prox}_{\gamma h}\!\bigl(\mathbf{y}^{(k)} -
   \gamma\nabla f(\mathbf{y}^{(k)})\bigr),

with :math:`f` the smooth least-squares term, :math:`h` the
non-smooth penalty (non-negativity or :math:`\ell_1`) and the step
:math:`\gamma \le 1/\|\mathbf{R}^T\mathbf{R}\|_2 = 1/\sigma_{\max}^2`.
FISTA improves the worst-case objective-value rate from
:math:`\mathcal{O}(1/k)` to :math:`\mathcal{O}(1/k^2)` (Beck and
Teboulle, 2009); the *iterate* sequence itself is non-monotone, so the
objective may temporarily increase — monotone behaviour requires
modified variants (e.g. MFISTA).  ``unfold_fista`` follows the IRtools
implementation (Gazzola, Hansen and Nagy, 2018).
``unfold_doroshenko`` applies a related coordinate-update relaxation of
the same least-squares objective.

All methods in this section are stopped by semi-convergence-aware
rules: a relative-change threshold, a discrepancy-principle test, or a
maximum iteration count.

.. _math-optcourse:

First-Order Optimisation Methods (MIPT Course Port)
----------------------------------------------------

Methods: ``unfold_pgd``, ``unfold_frank_wolfe``, ``unfold_mirror_descent``,
``unfold_admm``, ``unfold_lbfgsb``, ``unfold_coordinate_descent``,
``unfold_subgradient``, ``unfold_extragradient``,
``select_regularization_1d``.

This family ports the algorithmic core of the MIPT course
*"Optimization Methods in Machine Learning"* (lectures 7-15; the
`course repository <https://github.com/Radiationsafety/OPTIMIZATION-METHODS-COURSE>`_
contains lecture notes and homework notebooks).  Each method is a
classical first-order scheme applied to the unfolding objective; the
pedagogical value is that *every design choice has a provable effect*,
stated below.

**Projected gradient descent (PGD)** — ``unfold_pgd`` (lecture 9 /
homework 14).  The plain gradient step
:math:`\boldsymbol{\Phi}^{(k+1)} = \boldsymbol{\Phi}^{(k)} -
\gamma\nabla f(\boldsymbol{\Phi}^{(k)})` is followed by the Euclidean
projection :math:`\mathcal{P}_{\mathcal{C}}` onto the constraint set
:math:`\mathcal{C}` — the non-negative orthant, a box, or the
*fluence simplex* :math:`\{\boldsymbol{\Phi}\ge 0,\ \sum_j \Phi_j = F\}`.
For a fixed total fluence the simplex projection preserves the physical
normalisation *exactly* at every iterate — unlike iterative solvers,
whose fluence drifts during semi-convergence.  Convergence for smooth
convex :math:`f` is
:math:`f(\boldsymbol{\Phi}^{(k)}) - f^\star = \mathcal{O}(1/k)` for
:math:`\gamma \le 1/L` with :math:`L = \|\mathbf{R}^T\mathbf{R}\|_2`
(Beck and Teboulle, 2009); Armijo *backtracking* (halving
:math:`\gamma` until the Armijo descent condition holds) makes the step
adaptive to the local curvature and removes the need to estimate
:math:`L` manually (Nocedal and Wright, 2006).  The reported
*NNLS duality gap* — the difference between the primal objective and a
dual-feasible lower bound reconstructed from multipliers — is an
optimality *certificate*: when it reaches zero the returned spectrum is
provably the constrained optimum, not merely a stationary point
(Nocedal and Wright, 2006; Boyd and Vandenberghe, 2004).

**Frank-Wolfe (conditional gradient)** — ``unfold_frank_wolfe``
(lecture 9).  Instead of a gradient step plus projection, the method
minimises the *linearisation* of :math:`f` over the fluence simplex —
a trivial *linear minimisation oracle* that picks the single most
descent-deserving energy vertex — and moves a fraction
:math:`\gamma \in (0,1]` towards it.  The *Frank-Wolfe gap*
:math:`g(\boldsymbol{\Phi}^{(k)}) =
\nabla f(\boldsymbol{\Phi}^{(k)})^T(\boldsymbol{\Phi}^{(k)} -
\mathbf{s}^{(k)})` with :math:`\mathbf{s}^{(k)}` the oracle vertex is
always an upper bound on the sub-optimality, so it doubles as a
natural stopping certificate (Frank and Wolfe, 1956;
Jaggi, 2013).  Wolfe's *away steps* let iterates move away from
previously chosen vertices, curing the classical zig-zagging of
boundary solutions (Wolfe, 1970).  Because every iterate is a convex
combination of simplex vertices, the total fluence is preserved to
machine precision — the method of choice when the fluence
normalisation is itself a measured quantity.

**Mirror descent** — ``unfold_mirror_descent`` (lecture 10 / homework
16).  Generalises projected gradient descent by replacing the
Euclidean geometry with a *Bregman divergence* generated by a
*mirror map* :math:`\psi` (Nemirovski and Yudin, 1983;
Beck and Teboulle, 2003).  The entropy map
:math:`\psi(\boldsymbol{\Phi}) = \sum_j \Phi_j \ln \Phi_j` yields the
multiplicative update
:math:`\Phi_j^{(k+1)} \propto \Phi_j^{(k)}
\exp(-\gamma \nabla_j f)` — the same recursion family as MLEM, GRAVEL
and SAND-II, derived here from first principles.  Multiplicative
updates preserve positivity and act *relatively* (a bin with large
fluence is corrected proportionally more), which is why
entropy-geometry solvers dominate Poisson-count unfolding.  The
:math:`\ell_2` and p-norm maps give additive updates with different
bias profiles: the :math:`\ell_2` geometry distributes corrections
uniformly and is more robust to near-zero bins that dominate the
log-space relative error.  A per-iteration golden-section line search
along the mirror trajectory keeps the step inside the orthant interior
(Nemirovski and Yudin, 1983).

**Consensus ADMM** — ``unfold_admm`` (lecture 11 / homework 18).  The
splitting
:math:`\min f(\boldsymbol{\Phi}) + g(\mathbf{z})` s.t.
:math:`\boldsymbol{\Phi} = \mathbf{z}` turns the L1/TV-regularised
problem into alternating (i) an exact *non-negative least squares*
x-update on the augmented system, (ii) a soft-thresholding z-update
that realises the proximal operator of the :math:`\ell_1` penalty, and
(iii) a scaled dual ascent step (Boyd et al., 2011; Gabay and Mercier,
1976).  The primal/dual residual stopping test
:math:`\|\mathbf{R}\boldsymbol{\Phi}^{(k)} - \mathbf{z}^{(k)}\| \to 0`,
:math:`\rho(\mathbf{z}^{(k+1)}-\mathbf{z}^{(k)}) \to 0` certifies
consensus between the data-fidelity and penalty variables.  The
*adaptive penalty* :math:`\rho` (Boyd et al., 2011, sec. 3.4.1)
balances the two residual norms each outer iteration; it makes the
solver insensitive to the absolute count scale of the problem — the
single most common failure mode of fixed-:math:`\rho` ADMM on
Poisson-counted data.

**L-BFGS-B quasi-Newton** — ``unfold_lbfgsb`` (lecture 7 / homework
10).  A limited-memory quasi-Newton method with box bounds: curvature
pairs :math:`(\mathbf{s}_i, \mathbf{y}_i)` from the last
:math:`m` iterates approximate the Hessian inverse, giving
superlinear-type local convergence on the smooth Tikhonov objective at
:math:`\mathcal{O}(n\,m)` memory (Byrd et al., 1995; Nocedal and
Wright, 2006).  The optional second-difference (curvature) penalty
:math:`\|\mathbf{D}_2\boldsymbol{\Phi}\|^2` penalises oscillations;
increasing it visibly suppresses high-frequency noise but flattens
sharp spectral peaks — the parameter should be raised only until the
residual chi-square leaves the :math:`M \pm \sqrt{2M}` band of
:ref:`math-selection`.

**Coordinate descent** — ``unfold_coordinate_descent`` (lecture 15).
Minimises the NNLS objective one coordinate at a time in closed form,
:math:`\Phi_j \leftarrow \max\!\bigl(0,\,
(\mathbf{a}_j^T\mathbf{r} + \|\mathbf{a}_j\|^2\Phi_j -
\lambda_1)/(\|\mathbf{a}_j\|^2 + \lambda_2)\bigr)`, with the residual
:math:`\mathbf{r}` updated in :math:`\mathcal{O}(M)` per coordinate
(Wright, 2015; Luo and Tseng, 1992).  Each coordinate update
*monotonically decreases* the objective, which makes the method
extremely robust on ill-conditioned systems — at the cost of slower
progress on strongly coupled bins.  Cyclic order is deterministic;
seeded random order escapes the worst-case cycling of badly scaled
columns (Wright, 2015).

**Projected subgradient** — ``unfold_subgradient`` (lecture 8 /
homework 12).  For the nonsmooth L1/TV objective the gradient is
replaced by a *subgradient* — any vector in the subdifferential
(Nemirovski and Yudin, 1983; Shor, 1985).  The step-size policies have
sharply different guarantees: the *Polyak step*
:math:`\gamma_k = (f(\boldsymbol{\Phi}^{(k)}) -
f^\star)/\|\mathbf{g}^{(k)}\|^2` converges when the optimal value
:math:`f^\star` is estimated well (Polyak, 1967), square-summable
*diminishing* steps (:math:`\sum_k \gamma_k = \infty`,
:math:`\sum_k \gamma_k^2 < \infty`) converge for any fixed
:math:`f^\star` but slowly, and fixed steps converge only to a
neighbourhood of the optimum whose radius is proportional to
:math:`\gamma`.  The best-iterate-by-objective return is the standard
remedy for subgradient non-monotonicity.

**Extragradient (Korpelevich)** — ``unfold_extragradient``
(lecture 13 / homework 20).  The robust saddle formulation
:math:`\min_{\boldsymbol{\Phi}\ge 0}\ \max_{\|\mathbf{y}\|\le 1}
\frac{1}{2}\|\mathbf{R}\boldsymbol{\Phi}-\mathbf{N}\|^2 +
\delta\,\mathbf{y}^T(\mathbf{R}\boldsymbol{\Phi}-\mathbf{N})` is
equivalent to least squares against an *adversary* that may add any
bounded perturbation of L2 norm :math:`\delta` to the readings — the
uncertainty-aware reformulation of noise-robust unfolding.  The plain
projected gradient method on saddle problems oscillates; Korpelevich's
extra *prediction-correction* gradient step restores convergence at the
rate :math:`\mathcal{O}(1/k)` for monotone operators
(Korpelevich, 1976; Facchinei and Pang, 2003).  Larger
``noise_level`` widens the guaranteed robustness radius at the price
of a more conservative (smoother) spectrum.

**1D parameter search** — ``select_regularization_1d`` (lecture 1 /
homework 1) sweeps :math:`\log_{10}\alpha` with golden-section /
dichotomy / Brent minimisation of a GCV, Morozov or predictive-risk
score, replacing hand-tuned regularisation grids (Brent, 1973;
Golub, Heath and Wahba, 1979).

*When to choose which.*  Use PGD/FISTA-type methods for the fastest
smooth-objective descent; Frank-Wolfe when the fluence must be
conserved exactly; mirror descent (entropy map) for Poisson-counted
data; ADMM when an exact L1/TV penalty is required with certified
consensus; L-BFGS-B for the largest smooth problems; coordinate descent
for maximal per-iteration robustness; subgradient methods only when a
nonsmooth objective cannot be split; extragradient when robustness to
bounded reading errors must be explicit.

.. _math-ratio:

Iterative Ratio Methods (SAND-II Family)
----------------------------------------

Methods: ``unfold_sandii``, ``unfold_gravel``, ``unfold_bunki``,
``unfold_bunkiut``, ``unfold_rebunki``, ``unfold_nsduaz``.

The SAND-II family corrects a trial spectrum multiplicatively from the
measured-to-calculated count-rate ratios.  In ``unfold_sandii``
(McElroy et al., 1967; Griffin, Kelly and VanDenburg, 1994) the update
is the weighted geometric mean

.. math::

   \hat N_i = \sum_j \mathbf{R}_{ij}\,\Phi_j^{(k)},
   \qquad
   W_{ij} = \frac{\mathbf{R}_{ij}\,\Phi_j^{(k)}}{\hat N_i},
   \qquad
   \Phi_j^{(k+1)} = \Phi_j^{(k)} \exp\!\left(
   \frac{\sum_i W_{ij}\,\ln(N_i/\hat N_i)}
   {\sum_i W_{ij}}\right),

and the iteration stops when the chi-square of the fit drops to the
number of detectors or the maximum relative change of the spectrum
falls below the tolerance.  GRAVEL (``unfold_gravel``; Matzke, 2003)
uses the same ratio logic with count-based weights
:math:`W_{ij} \propto N_i\,\mathbf{R}_{ij}\,\Phi_j/\hat N_i`, an
optional regularisation term on the log-spectrum, and a stop test on
the relative change of the spectrum; its objective is the weighted sum
of squared *logarithmic* ratio deviations, which dampens the influence
of high-count channels.  The BUNKI/BUNKI-UT variants
(``unfold_bunki``, ``unfold_bunkiut``) implement the SPUNIT and BON31G
ratio recursions, ``unfold_rebunki`` adds iteration-dependent
relaxation, and ``unfold_nsduaz`` replaces the initial guess by a
catalogue of analytic shapes (fission Watt, evaporation, thermal — see
:ref:`math-parametric`) before applying ratio corrections.

These methods require strictly positive measurements (zero or negative
channels are dropped), preserve positivity exactly, and — as all
iterative schemes here — display semi-convergence; they are typically
run for a modest number of iterations with chi-square monitoring
(Machado, García-Baonza and Vega-Carrillo, 2024).

.. _math-maxent:

Maximum Entropy and Information Divergence
------------------------------------------

Methods: ``unfold_maxed``, ``unfold_amaxed``, ``unfold_amaxed_regularization``,
``unfold_imaxed``, ``unfold_directed_divergence``.

The entropy-regularised reconstruction minimises the relative entropy
(Kullback and Leibler, 1951) with respect to a prior spectrum
:math:`\boldsymbol{\Phi}_0 > 0`:

.. math::

   S(\boldsymbol{\Phi}, \boldsymbol{\Phi}_0)
   = \sum_{j=1}^K \left[
   \Phi_j \ln\!\left(\frac{\Phi_j}{\Phi_{0,j}}\right)
   + \Phi_{0,j} - \Phi_j\right] \;\ge\; 0.

In the canonical MAXED formulation (Reginatto and Goldhagen, 1999;
Reginatto, Goldhagen and Neumann, 2002) one solves the constrained
problem

.. math::

   \min_{\boldsymbol{\Phi} > 0}\; S(\boldsymbol{\Phi}, \boldsymbol{\Phi}_0)
   \quad\text{subject to}\quad
   \chi^2(\boldsymbol{\Phi}) = \sum_i
   \frac{\bigl((\mathbf{R}\boldsymbol{\Phi})_i - N_i\bigr)^2}{\sigma_i^2}
   \le \Omega,

whose Lagrangian dual yields the solution
:math:`\boldsymbol{\Phi}(\theta)` parametrised by the multiplier
:math:`\theta > 0` attached to the chi-square constraint; :math:`\theta`
is then adjusted (bisection on :math:`\Omega`) until
:math:`\chi^2 \approx \Omega`, with the usual choice
:math:`\Omega = M` (or the :math:`M \pm \sqrt{2M}` band) for known
counting variances.  The dual functional
:math:`\Psi(\boldsymbol{\lambda})` is strictly concave in the dual
variables, and its Hessian,
:math:`\mathbf{R}^T\operatorname{diag}(\boldsymbol{\Phi})\mathbf{R}
+ \operatorname{diag}(\sigma_i^2/2\theta)`, is positive definite for
every :math:`\theta > 0` thanks to its diagonal term — no additional
rank condition on :math:`\mathbf{R}` is required.

The implementation of ``unfold_maxed`` in this package minimises the
equivalent *primal* objective directly,

.. math::

   f(\boldsymbol{\Phi}) = -S(\boldsymbol{\Phi}, \boldsymbol{\Phi}_0)
   + \frac{1}{2}\sum_i
   \frac{\bigl((\mathbf{R}\boldsymbol{\Phi})_i - N_i\bigr)^2}{\sigma_i^2},

in log-space (:math:`y_j = \ln\Phi_j`, which enforces positivity
exactly) with L-BFGS-B; the relative weight of the chi-square term is
controlled by ``sigma_factor`` rather than by an explicit bisection on
:math:`\Omega`.  ``unfold_amaxed``, ``unfold_amaxed_regularization``
and ``unfold_imaxed`` are adaptive/iterative refinements of the same
primal scheme (Wong, 2024).  ``unfold_directed_divergence`` minimises
the directed (Kullback) divergence
:math:`\sum_j \Phi_j \ln(\Phi_j/\Phi_{0,j})` with multiplicative
updates — the same functional used inside the N-spline solver
``unfold_nspline`` (see :ref:`math-parametric`).

Because the entropy functional is strictly convex on
:math:`\mathbb{R}^K_{>0}` (for the Kullback-Leibler form above, the
Hessian is :math:`\operatorname{diag}(1/\Phi_j) \succ 0`), the primal
problem is well posed once the feasible set is non-empty; positivity of
the solution is automatic since :math:`S` is defined only for
:math:`\boldsymbol{\Phi} > 0`.  Semi-convergence does not arise for the
strictly convex entropy-plus-quadratic objective solved to optimality,
but the noise-fitting problem reappears when :math:`\Omega` (or
``sigma_factor``) is chosen too small.

.. _math-em:

Poisson-Likelihood (EM) Family
------------------------------

Methods: ``unfold_mlem``, ``unfold_mlem_stop``, ``unfold_mlem_odl``,
``unfold_odl_advanced``, ``unfold_osem``, ``unfold_osem_anlm``,
``unfold_mapem``, ``unfold_bsrem``.

The Poisson log-likelihood maximisation problem is

.. math::

   \max_{\boldsymbol{\Phi} \ge 0}\; L(\boldsymbol{\Phi})
   = \sum_{i=1}^M \left[
   N_i \ln\!\left(\sum_{j=1}^K \mathbf{R}_{ij}\Phi_j\right)
   - \sum_{j=1}^K \mathbf{R}_{ij}\Phi_j\right]
   - \beta\,V(\boldsymbol{\Phi}),

where :math:`V` is an optional roughness penalty (MAP estimation).
The classical MLEM update is multiplicative,

.. math::

   \Phi_j^{(k+1)} = \Phi_j^{(k)}\,
   \frac{1}{\sum_i \mathbf{R}_{ij}}
   \sum_i \mathbf{R}_{ij}\,
   \frac{N_i}{(\mathbf{R}\boldsymbol{\Phi}^{(k)})_i},

preserves positivity exactly, and increases the likelihood
monotonically (Shepp and Vardi, 1982; Richardson, 1972; Lucy, 1974).
Because the likelihood is ill-posed, unconstrained MLEM converges to
the maximum-likelihood fit of the noise — the semi-convergence problem
again — so regularised or early-stopped variants are preferred:
ordered-subset acceleration (``unfold_osem``), MAP penalties
(``unfold_mapem``), block-sequential regularised EM (``unfold_bsrem``;
De Pierro, 1995) whose step sizes satisfy the stochastic-approximation
conditions :math:`\sum_k \alpha_k = \infty` and
:math:`\sum_k \alpha_k^2 < \infty` (Robbins and Monro, 1951), and the
proximal ODL variants (``unfold_mlem_odl``, ``unfold_odl_advanced``).

**OSEM with asymptotic non-local means.**  ``unfold_osem_anlm``
(Jamaati et al. 2026) interleaves the ordered-subset update with the
two-stage asymptotic non-local means (ANLM) filter applied to the
intermediate spectrum after every subset update (optionally only once
after the last update, ``anlm_mode='post'``).  Stage 1 applies NLM with
the uniform parameter :math:`h_1 = 0.5\,\sigma`; stage 2 applies the
point-wise parameter of the article's eq. 6,

.. math::

   h_2(i) = \sigma_2(i) = \Big(\sum_{j\in N_i} w(i,j)^2\,\sigma^2\Big)^{1/2},

i.e. the noise standard deviation smoothed by the initial NLM weights
:math:`w(i,j)`.  For one-dimensional spectra the 2D windows become index
windows on the energy grid (search window :math:`N`, Gaussian-weighted
similarity window :math:`\nu`), and the filter by default operates in
log space, where the relative EM noise is approximately additive and a
single noise-level estimate matches every bin; the automatic estimate
uses the MAD of the second differences.

**MLEM with J-factor stopping.**  ``unfold_mlem_stop`` implements the
early-stopping criterion of Montgomery et al. (2020): at iteration
:math:`k` the indicator

.. math::

   J(k) = \frac{\sum_i \bigl(N_i - \hat N_i^{(k)}\bigr)^2}
   {\sum_i \hat N_i^{(k)}},
   \qquad \hat N_i^{(k)} = (\mathbf{R}\boldsymbol{\Phi}^{(k)})_i,

is computed and the iteration is stopped once :math:`J(k)` falls below
a threshold.  In the reference implementation the threshold is selected
automatically from the count level (the ``cps_crossover`` parameter,
default 30000 counts, switches between two threshold regimes); the
statistically motivated check :math:`J(k) \le M + \sqrt{2M}` corresponds
to the discrepancy band discussed in :ref:`math-selection`.  The point
estimate returned is the spectrum at the stopping index, which acts as
the regularisation parameter.

.. _math-quadratic:

Quadratic Methods with Full Covariance Treatment
------------------------------------------------

Methods: ``unfold_staysl``, ``unfold_ferdor``.

This family solves weighted least-squares problems with full
propagation of measurement covariances — it is *not* entropy-based,
although historically these codes are discussed alongside the MaxEnt
codes in the Bonner sphere literature (Reginatto, 2010).

**STAY'SL** (``unfold_staysl``; Perey, 1977) performs least-squares
dosimetry unfolding with input covariance matrices for the measured
responses, the response functions and the prior spectrum, and returns
the updated spectrum together with its full covariance matrix.  The
solution minimises the quadratic form

.. math::

   \chi^2 = \bigl(\mathbf{R}\boldsymbol{\Phi} - \mathbf{N}\bigr)^T
   \boldsymbol{\Sigma}_N^{-1}
   \bigl(\mathbf{R}\boldsymbol{\Phi} - \mathbf{N}\bigr)

subject to linear constraints, solved through the corresponding
normal equations with the appropriate covariance weighting.

**FERDOR** (``unfold_ferdor``; Burrus, ORNL-4154, 1965) seeks the
spectrum that reproduces the measurements within their uncertainties
while being as smooth as possible, realised here as

.. math::

   \min_{\boldsymbol{\Phi} \ge 0}\;
   \frac{1}{2}\,
   \bigl\|\boldsymbol{\Sigma}^{-1/2}
   (\mathbf{R}\boldsymbol{\Phi} - \mathbf{N})\bigr\|_2^2
   + \frac{\alpha}{2}\,
   \bigl\|\mathbf{D}_2\boldsymbol{\Phi}\bigr\|_2^2,

where :math:`\boldsymbol{\Sigma}` is the diagonal
measurement-covariance matrix and :math:`\mathbf{D}_2` the second
difference operator.  The smoothing weight :math:`\alpha` is not fixed
a priori: it is tuned iteratively (bisection) so that the reduced
chi-square of the fit meets the discrepancy band of
:ref:`math-selection`.

.. _math-bayesian:

Bayesian and Stochastic Methods
-------------------------------

Methods: ``unfold_bayes``, ``unfold_bayes_spline_regularization``,
``unfold_bayesian_parametric``, ``unfold_mcmc``, ``unfold_eki``.

The posterior density combines the likelihood and prior terms,

.. math::

   P(\boldsymbol{\Phi} \mid \mathbf{N})
   \propto P(\mathbf{N} \mid \boldsymbol{\Phi})\,
   P(\boldsymbol{\Phi}),

with a Poisson or Gaussian likelihood as described in
:ref:`math-general`.

**Bayesian iterative unfolding (D'Agostini).**  ``unfold_bayes``
implements the Bayesian iterative scheme of D'Agostini (1995).  With
the response normalised column-wise,
:math:`P(E_j \mid C_i) = \mathbf{R}_{ij}\Phi_j^{(k)} /
\sum_m \mathbf{R}_{im}\Phi_m^{(k)}`, the update is

.. math::

   \Phi_j^{(k+1)} = \sum_{i=1}^M P(E_j \mid C_i)\; N_i,

or, equivalently, in multiplicative form

.. math::

   \Phi_j^{(k+1)} = \Phi_j^{(k)} \sum_{i=1}^M
   \frac{\mathbf{R}_{ij}\,N_i}{\hat N_i^{(k)}},
   \qquad
   \hat N_i^{(k)} = \sum_m \mathbf{R}_{im}\Phi_m^{(k)},

i.e. exactly the MLEM recursion without the sensitivity normalisation
:math:`1/\sum_i \mathbf{R}_{ij}` (the missing factor is absorbed by the
column normalisation of :math:`\mathbf{R}`).  The two expressions above
are algebraically identical; in particular the update contains the
*same* factor :math:`\Phi_j^{(k)}` only once, and no additional power
of the current spectrum appears.  The implementation works in
effective-count space and rescales the result to physical units.

**MCMC.**  ``unfold_mcmc`` samples the posterior with the NUTS
Hamiltonian sampler (PyMC), placing a smoothness (Ornstein-Uhlenbeck)
prior on the log-spectrum anchored at a data-driven centre.  Chain
convergence is assessed with the Gelman-Rubin statistic and effective
sample sizes as computed by ArviZ.  For :math:`m` chains of length
:math:`n`, with within-chain variance :math:`W` and between-chain
variance :math:`B`,

.. math::

   \hat V = \frac{n-1}{n}\,W + \frac{B}{n},
   \qquad
   \hat R = \sqrt{\frac{\hat V}{W}},

(Gelman and Rubin, 1992; Brooks and Gelman, 1998); the threshold
:math:`\hat R < 1.05` used by the package is a common practical choice,
while the rank-normalised split-:math:`\hat R` recommended today
(Vehtari et al., 2021) — and implemented by ArviZ — tightens this to
:math:`\hat R < 1.01`.

**Ensemble Kalman inversion.**  ``unfold_eki`` (Iglesias, Law and
Stuart, 2013) propagates an ensemble of spectra through the Kalman
update of the forward model.  In the limit :math:`N_{\mathrm{ens}} \to
\infty` and :math:`t \to \infty` the ensemble collapses onto the
least-squares solution of the data-misfit problem; useful regularised
solutions are obtained either by *early stopping* of the ensemble
evolution (the iteration index playing the role of
:math:`\alpha^{-1}`, with the stopping level selected by a discrepancy
or GCV rule) or by the Tikhonov-type regularisation term and prior
covariance supported by the implementation (``regularization``,
``inflation``, ensemble size ``n_ensemble``).  Convergence is monitored
through the ensemble spread and the data misfit; the spread stabilises
when the ensemble has collapsed onto the attractor of the update.

.. _math-parametric:

Parametric Models
-----------------

Methods: ``unfold_parametric``, ``unfold_parametric2``,
``unfold_hybrid_parametric``, ``unfold_crystal_ball``, ``unfold_nspline``,
``unfold_zfit``, ``unfold_lmfit``, ``unfold_express``, ``unfold_fruit_like``,
``unfold_rfsp_jul``, ``unfold_mystic``, ``unfold_genetic``, ``unfold_maeo``.

The spectrum is represented by a low-dimensional parameter vector
:math:`\boldsymbol{\theta}` and the corresponding data misfit is
minimised:

.. math::

   \min_{\boldsymbol{\theta} \in \Theta}\;
   \chi^2(\boldsymbol{\theta})
   = \sum_{i=1}^M \left(
   \frac{\int_0^\infty R_i(E)\,\Phi(E;\boldsymbol{\theta})\,dE - N_i}
   {\sigma_i}\right)^2.

Analytic building blocks used by the shape catalogues (e.g.
``unfold_nsduaz``, ``unfold_express``) must be written for the *fluence*
density.  The fission (Watt) spectrum is

.. math::

   \Phi_{\mathrm{Watt}}(E) \propto \exp(-E/a)\,
   \sinh\!\bigl(\sqrt{b\,E}\bigr),

with the common :math:`{}^{252}\mathrm{Cf}` values
:math:`a = 1.025` MeV, :math:`b = 2.926\ \mathrm{MeV}^{-1}` (Watt,
1952); the evaporation and thermal Maxwellian fluence components enter
with the same functional form :math:`\Phi(E) \propto E\,e^{-E/kT}` at
different temperatures, with a :math:`1/v`-controlled tail in the
thermal limit.  Note that the form :math:`\sqrt{E}\,e^{-E/T}` sometimes
seen in the literature is the Maxwellian *flux density* convention, not
the fluence density; the shapes above are the fluence forms used by the
code.

The N-spline model of ``unfold_nspline`` represents
:math:`\ln\Phi(E)` piecewise by splines of the form
:math:`\exp(a_k + q_k\ln E + r_k E)` on segments between free knots
(Islamgulov and Lartsev, 2008); the node-continuity constraints are
enforced through KKT elimination and the directed divergence is
minimised at each iteration — see :doc:`nspline` for the full
formulation and preset knot layouts.

These models are solved with nonlinear least squares (L-BFGS-B, least-squares
and probabilistic backends) or hybrid local/global optimisers:
``unfold_lmfit`` and ``unfold_zfit`` wrap the corresponding fitting
libraries, while ``unfold_mystic``, ``unfold_genetic`` and
``unfold_maeo`` provide global and multiobjective search.  For
bound-constrained smooth problems the package relies on L-BFGS-B (Byrd
et al., 1995), which converges globally to first-order stationary
points under standard line-search conditions; its observed rate is
typically linear, and no superlinear convergence is claimed for the
limited-memory variant.

.. _math-meta:

Meta-Strategies and Ensembles
-----------------------------

Methods: ``unfold_ensemble``, ``unfold_combined``, ``unfold_composite``,
``unfold_cascade``, ``unfold_iterative_refinement``, ``unfold_binned``.

Meta-methods do not introduce new variational principles; they combine
base solvers.  ``unfold_ensemble`` averages (weighted mean, median,
trimmed mean, or best-residual selection) several base solutions;
``unfold_combined`` chains methods in a pipeline; ``unfold_composite``
and ``unfold_cascade`` run coarse-to-fine sequences;
``unfold_iterative_refinement`` performs a two-pass unfold with an
auto-selected blending factor; and ``unfold_binned`` selects the
per-bin best method from a pre-computed benchmark lookup table.  The
statistical behaviour of a meta-method is inherited from its
constituents: it can only average out solver-specific artefacts, not
replace an informed regularisation choice.

.. _math-refs:

References
----------

.. list-table::
   :header-rows: 0
   :widths: 100

   * - Andersen, A. C., Kak, A. C. (1984). Simultaneous algebraic reconstruction
       technique (SART): a superior implementation of the ART algorithm. *Ultrasonic
       Imaging* **6**, 81-94. `doi:10.1177/016173468400600107
       <https://doi.org/10.1177/016173468400600107>`__
   * - Beck, A., Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm
       for linear inverse problems. *SIAM J. Imaging Sci.* **2**, 183-202.
       `doi:10.1137/080716542 <https://doi.org/10.1137/080716542>`__
   * - Beck, A., Teboulle, M. (2003). Mirror descent and nonlinear projected
       subgradient methods for convex optimization. *Operations Research Letters*
       **31**, 167-175. `doi:10.1016/S0167-6377(02)00231-6
       <https://doi.org/10.1016/S0167-6377(02)00231-6>`__
   * - Boyd, S., Parikh, N., Chu, E., Peleato, B., Eckstein, J. (2011). Distributed
       optimization and statistical learning via the alternating direction method of
       multipliers. *Foundations and Trends in Machine Learning* **3**, 1-122.
       `doi:10.1561/2200000016 <https://doi.org/10.1561/2200000016>`__
   * - Boyd, S., Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University
       Press. `doi:10.1017/CBO9780511804441
       <https://doi.org/10.1017/CBO9780511804441>`__
   * - Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*.
       Prentice-Hall, Englewood Cliffs.
   * - Brooks, S. P., Gelman, A. (1998). General methods for monitoring convergence of
       iterative simulations. *J. Comput. Graph. Stat.* **7**, 434-455.
       `doi:10.1080/10618600.1998.10474787
       <https://doi.org/10.1080/10618600.1998.10474787>`__
   * - Burrus, W. R. (1965). Utilization of a priori information in the statistical
       interpretation of measured distributions. Report ORNL-4154, Oak Ridge National
       Laboratory.
   * - Byrd, R. H., Lu, P., Nocedal, J., Zhu, C. (1995). A limited memory algorithm for
       bound constrained optimization. *SIAM J. Sci. Comput.* **16**, 1190-1208.
       `doi:10.1137/0916069 <https://doi.org/10.1137/0916069>`__
   * - D'Agostini, G. (1995). A multidimensional unfolding method based on Bayes'
       theorem. *Nucl. Instrum. Meth. A* **362**, 487-498.
       `doi:10.1016/0168-9002(95)00274-X
       <https://doi.org/10.1016/0168-9002(95)00274-X>`__
   * - De Pierro, A. R. (1995). A modified expectation maximization algorithm for
       penalized likelihood estimation. *IEEE Trans. Med. Imaging* **14**, 132-137.
       `doi:10.1109/42.370409 <https://doi.org/10.1109/42.370409>`__
   * - Engl, H. W., Hanke, M., Neubauer, A. (1996). *Regularization of Inverse
       Problems*. Kluwer, Dordrecht. `doi:10.1007/978-94-009-1740-8
       <https://doi.org/10.1007/978-94-009-1740-8>`__
   * - Facchinei, F., Pang, J.-S. (2003). *Finite-Dimensional Variational Inequalities
       and Complementarity Problems*, vol. II. Springer, New York. `doi:10.1007/b97411
       <https://doi.org/10.1007/b97411>`__
   * - Frank, M., Wolfe, P. (1956). An algorithm for quadratic programming. *Naval
       Research Logistics Quarterly* **3**, 95-110. `doi:10.1002/nav.3800030109
       <https://doi.org/10.1002/nav.3800030109>`__
   * - Gabay, D., Mercier, B. (1976). A dual algorithm for the solution of nonlinear
       variational problems via finite element approximation. *Computers & Mathematics
       with Applications* **2**, 17-40. `doi:10.1016/0898-1221(76)90003-1
       <https://doi.org/10.1016/0898-1221(76)90003-1>`__
   * - Gazzola, S., Hansen, P. C., Nagy, J. G. (2018). IR Tools: a MATLAB package of
       iterative regularization methods and large-scale test problems. *Numer.
       Algorithms* **81**, 773-811. `doi:10.1007/s11075-018-0570-7
       <https://doi.org/10.1007/s11075-018-0570-7>`__
   * - Gelman, A., Rubin, D. B. (1992). Inference from iterative simulation using
       multiple sequences. *Statist. Sci.* **7**, 457-472. `doi:10.1214/ss/1177011136
       <https://doi.org/10.1214/ss/1177011136>`__
   * - Golub, G. H., Heath, M., Wahba, G. (1979). Generalized cross-validation as a
       method for choosing a good ridge parameter. *Technometrics* **21**, 215-223.
       `doi:10.1080/00401706.1979.10489751
       <https://doi.org/10.1080/00401706.1979.10489751>`__
   * - Golub, G. H., Kahan, W. (1965). Calculating the singular values and
       pseudo-inverse of a matrix. *J. SIAM Numer. Anal. B* **2**, 205-224.
       `doi:10.1137/0702016 <https://doi.org/10.1137/0702016>`__
   * - Griffin, P. J., Kelly, D. G., VanDenburg, S. W. (1994). User's manual for
       SNL-SAND-II code. Report SAND92-2357 / OSTI 10149711, Sandia National
       Laboratories. `doi:10.2172/10149711 <https://doi.org/10.2172/10149711>`__
   * - Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the
       L-curve. *SIAM Review* **34**, 561-580. `doi:10.1137/1034115
       <https://doi.org/10.1137/1034115>`__
   * - Hansen, P. C. (1994). Regularization tools: a Matlab package for analysis and
       solution of discrete ill-posed problems. *Numer. Algorithms* **6**, 1-35.
       `doi:10.1007/BF02149761 <https://doi.org/10.1007/BF02149761>`__
   * - Hestenes, M. R., Stiefel, E. (1952). Methods of conjugate gradients for solving
       linear systems. *J. Res. NBS* **49**, 409-436. `doi:10.6028/jres.049.044
       <https://doi.org/10.6028/jres.049.044>`__
   * - Iglesias, M. A., Law, K. J. H., Stuart, A. M. (2013). Ensemble Kalman methods
       for inverse problems. *Inverse Problems* **29**, 045001.
       `doi:10.1088/0266-5611/29/4/045001
       <https://doi.org/10.1088/0266-5611/29/4/045001>`__
   * - IAEA (2001). *Compendium of Neutron Spectra and Detector Responses for Radiation
       Protection Purposes*. Technical Reports Series No. 403, Vienna.
   * - Islamgulov, D. G., Lartsev, A. V. (2008). N-spline reconstruction of neutron
       spectra on the basis of activation measurements. *Atomic Energy* **104**,
       387-397. `doi:10.1007/s10512-008-9045-6
       <https://doi.org/10.1007/s10512-008-9045-6>`__
   * - Jaggi, M. (2013). Revisiting Frank-Wolfe: projection-free sparse convex
       optimization. *Proc. ICML 2013*, 427-435.
   * - Jiang, M., Wang, G. (2003). Convergence of the simultaneous algebraic
       reconstruction technique (SART). *IEEE Trans. Image Process.* **12**, 957-961.
       `doi:10.1109/TIP.2003.815295 <https://doi.org/10.1109/TIP.2003.815295>`__
   * - Korpelevich, G. M. (1976). The extragradient method for finding saddle points
       and other problems. *Ekonomika i Matematicheskie Metody* **12**, 747-756.
   * - Kullback, S., Leibler, R. A. (1951). On information and sufficiency. *Ann. Math.
       Statist.* **22**, 79-86. `doi:10.1214/aoms/1177729694
       <https://doi.org/10.1214/aoms/1177729694>`__
   * - Lucy, L. B. (1974). An iterative technique for the rectification of observed
       distributions. *Astron. J.* **79**, 745-754. `doi:10.1086/111605
       <https://doi.org/10.1086/111605>`__
   * - Luo, Z.-Q., Tseng, P. (1992). On the linear convergence of the coordinate
       descent method for convex differentiable minimization. *J. Optim. Theory Appl.*
       **72**, 7-35. `doi:10.1007/BF00939952 <https://doi.org/10.1007/BF00939952>`__
   * - Machado, G. V., García-Baonza, A. M., Vega-Carrillo, H. R. (2024). Stopping
       criteria for neutron spectrum unfolding algorithms. *Appl. Radiat. Isot.*
       **212**, 111456. `doi:10.1016/j.apradiso.2024.111456
       <https://doi.org/10.1016/j.apradiso.2024.111456>`__
   * - Matzke, M. (2003). Unfolding procedures. *Radiat. Prot. Dosimetry* **107**,
       155-174. `doi:10.1093/oxfordjournals.rpd.a006384
       <https://doi.org/10.1093/oxfordjournals.rpd.a006384>`__
   * - McElroy, W. N., Berg, S., Crockett, T., Hawkins, R. G. (1967). A
       computer-automated iterative method for neutron flux spectra determination by
       foil activation. Report AFWL-TR-67-41 (vols. I-IV), Air Force Weapons
       Laboratory.
   * - Montgomery, L. et al. (2020). A novel MLEM stopping criterion for unfolding
       neutron fluence spectra in radiation therapy. *Nucl. Instrum. Meth. A* **957**,
       163400. `doi:10.1016/j.nima.2020.163400
       <https://doi.org/10.1016/j.nima.2020.163400>`__
   * - Morozov, V. A. (1984). *Methods for Solving Incorrectly Posed Problems*.
       Springer, New York. `doi:10.1007/978-1-4612-5280-1
       <https://doi.org/10.1007/978-1-4612-5280-1>`__
   * - Nemirovski, A., Yudin, D. (1983). *Problem Complexity and Method Efficiency in
       Optimization*. Wiley, New York.
   * - Nocedal, J., Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer,
       New York. `doi:10.1007/978-0-387-40065-5
       <https://doi.org/10.1007/978-0-387-40065-5>`__
   * - Paige, C. C., Saunders, M. A. (1982). LSQR: an algorithm for sparse linear
       equations and sparse least squares. *ACM Trans. Math. Software* **8**, 43-71.
       `doi:10.1145/355984.355989 <https://doi.org/10.1145/355984.355989>`__
   * - Perey, F. G. (1977). Least squares dosimetry unfolding: the program STAY'SL.
       Report ORNL/TM-6062, Oak Ridge National Laboratory.
   * - Polyak, B. T. (1967). A general method for solving extremal problems. *Doklady
       Akademii Nauk SSSR* **174**, 33-36.
   * - Reginatto, M. (2010). Overview of spectral unfolding techniques and uncertainty
       estimation in neutron spectrometry. *Radiat. Meas.* **45**, 1323-1329.
       `doi:10.1016/j.radmeas.2010.06.016
       <https://doi.org/10.1016/j.radmeas.2010.06.016>`__
   * - Reginatto, M., Goldhagen, P. (1999). MAXED, a computer code for maximum entropy
       deconvolution of multisphere neutron spectrometry measurements. *Health Phys.*
       **77**, 579-583. `doi:10.1097/00004032-199911000-00012
       <https://doi.org/10.1097/00004032-199911000-00012>`__
   * - Reginatto, M., Goldhagen, P., Neumann, S. (2002). Spectrum unfolding,
       sensitivity analysis and propagation of uncertainties with the MAXED unfolding
       algorithm. *Nucl. Instrum. Meth. A* **476**, 242-246.
       `doi:10.1016/S0168-9002(01)01439-5
       <https://doi.org/10.1016/S0168-9002(01)01439-5>`__
   * - Richardson, W. H. (1972). Bayesian-based iterative method of image restoration.
       *J. Opt. Soc. Am.* **62**, 55-59. `doi:10.1364/JOSA.62.000055
       <https://doi.org/10.1364/JOSA.62.000055>`__
   * - Robbins, H., Monro, S. (1951). A stochastic approximation method. *Ann. Math.
       Statist.* **22**, 400-407. `doi:10.1214/aoms/1177729586
       <https://doi.org/10.1214/aoms/1177729586>`__
   * - Shepp, L. A., Vardi, Y. (1982). Maximum likelihood reconstruction for emission
       tomography. *IEEE Trans. Med. Imaging* **1**, 113-122.
       `doi:10.1109/TMI.1982.4307558 <https://doi.org/10.1109/TMI.1982.4307558>`__
   * - Shor, N. Z. (1985). *Minimization Methods for Non-Differentiable Functions*.
       Springer, Berlin. `doi:10.1007/978-3-642-82118-9
       <https://doi.org/10.1007/978-3-642-82118-9>`__
   * - Strohmer, T., Vershynin, R. (2009). A randomized Kaczmarz algorithm with
       exponential convergence. *J. Fourier Anal. Appl.* **15**, 262-278.
       `doi:10.1007/s00041-008-9030-4 <https://doi.org/10.1007/s00041-008-9030-4>`__
   * - Thomas, D. J., Alevra, A. V. (2002). Bonner sphere spectrometers — a critical
       review. *Nucl. Instrum. Meth. A* **476**, 12-20.
       `doi:10.1016/S0168-9002(01)01379-1
       <https://doi.org/10.1016/S0168-9002(01)01379-1>`__
   * - Tikhonov, A. N., Goncharsky, A. V., Stepanov, V. V., Yagola, A. G. (1995).
       *Numerical Methods for the Solution of Ill-Posed Problems*. Kluwer, Dordrecht.
       `doi:10.1007/978-94-015-8480-7 <https://doi.org/10.1007/978-94-015-8480-7>`__
   * - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021).
       Rank-normalization, folding, and localization: an improved :math:`\hat R` for
       assessing convergence of MCMC. *Bayesian Analysis* **16**, 667-718.
       `doi:10.1214/20-BA1221 <https://doi.org/10.1214/20-BA1221>`__
   * - Watt, B. E. (1952). Energy spectrum of neutrons from thermal fission of U-235.
       *Phys. Rev.* **87**, 1037-1041. `doi:10.1103/PhysRev.87.1037
       <https://doi.org/10.1103/PhysRev.87.1037>`__
   * - Wolfe, P. (1970). Convergence theory in nonlinear programming. In: *Integer and
       Nonlinear Programming* (J. Abadie, ed.), North-Holland, Amsterdam.
   * - Wong, O. (2024). *Modernising neutron spectrum unfolding for fusion
       applications*. PhD Thesis, Sheffield Hallam University. `shura.shu.ac.uk/36014
       <https://shura.shu.ac.uk/36014/>`__
   * - Wright, S. J. (2015). Coordinate descent algorithms. *Math. Program.* **151**,
       3-34. `doi:10.1007/s10107-015-0892-3
       <https://doi.org/10.1007/s10107-015-0892-3>`__

Integration Notes
-----------------


The mathematical section is referenced from the overview page through
labelled cross-references (``:ref:`` targets ``math-tikhonov``,
``math-krylov``, ``math-iterative``, ``math-ratio``, ``math-maxent``,
``math-em``, ``math-quadratic``, ``math-bayesian``, ``math-parametric``,
``math-meta``) and is rendered with MathJax via ``sphinx.ext.mathjax``
using the ``mathjax3_config`` entry in ``conf.py`` (Sphinx >= 4
semantics).  Inline mathematics in reStructuredText sources must use
the ``:math:`` role; the MathJax ``inlineMath`` configuration only
affects the browser-side rendering of pre-existing dollar-sign input,
not the Sphinx parsing.  The optional diagram extension
``sphinxcontrib-mermaid`` is imported defensively in ``conf.py`` and
requires the package (and, for image output, the mermaid CLI) to be
present in the documentation build environment.

Integration Notes
-----------------

The mathematical section is referenced from the overview page through
labelled cross-references (``:ref:`` targets ``math-tikhonov``,
``math-krylov``, ``math-iterative``, ``math-optcourse``, ``math-ratio``,
``math-maxent``, ``math-em``, ``math-quadratic``, ``math-bayesian``,
``math-parametric``, ``math-meta``) and is rendered with MathJax via
``sphinx.ext.mathjax``
using the ``mathjax3_config`` entry in ``conf.py`` (Sphinx >= 4
semantics).  Inline mathematics in reStructuredText sources must use
the ``:math:`` role; the MathJax ``inlineMath`` configuration only
affects the browser-side rendering of pre-existing dollar-sign input,
not the Sphinx parsing.  The optional diagram extension
``sphinxcontrib-mermaid`` is imported defensively in ``conf.py`` and
requires the package (and, for image output, the mermaid CLI) to be
present in the documentation build environment.
