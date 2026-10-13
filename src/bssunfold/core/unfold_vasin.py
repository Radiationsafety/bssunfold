"""Regularized approximation of irregular (discontinuous) solutions.

This module ports the method of V. V. Vasin and T. I. Serazhitnikova,
"Regular algorithm for approximation of irregular solutions for Fredholm
integral equations of the first kind" (Vyčislitel'nye Tekhnologii 15(2),
2010, pp. 15-23) to neutron-spectrum unfolding.

The classical Tikhonov method with a Hilbert-space stabilizer (``L2`` or
``W_2^k``) smears out jumps.  Vasin and Serazhitnikova replace the smooth
penalty by a *non-differentiable* one built on the Hölder (Lipschitz)
seminorm, which is almost blind to the frequency of oscillation and
therefore cheap for discontinuous functions, and they measure the
discrepancy in the uniform norm.  With

.. math::

    [u]_{\\mu,\\infty} = \\max_{i \\ne j}
        \\frac{|u_i - u_j|}{|x_i - x_j|^\\mu},  \\quad 0 < \\mu \\le 1,

their discrete regularized problem (eq. (8) of the paper, stabilizer (4)
at ``p = 2``) reads

.. math::

    \\min_u \\Phi(u), \\quad
    \\Phi(u) = \\max_i \\left| \\sum_j h_j K_{ij} u_j - f_i \\right|
      + \\alpha \\left[ \\tfrac{1}{2} \\sum_j h_j u_j^2
        + \\max_j |u_j| + [u]_{\\mu,\\infty} \\right].

The paper minimizes ``\\Phi`` by the *prox-method* (eq. (9)) with an inner
normalized-subgradient (eq. (10)) or Fejer level-set (eq. (12)) process.
Here the same functional is minimized **exactly**, through its epigraph
form as a convex program: ``\\Phi`` is a sum of maxima of affine functions
plus one quadratic term, so

.. math::

    \\min \\; t + \\alpha (\\tfrac{1}{2} u^T H u + s + \\ell),
    \\quad -t \\le W(Au - b) \\le t,\\; s \\ge \\pm u_j,\\;
    \\ell \\ge \\pm (u_i - u_j)/d_{ij}^\\mu,\\; u \\ge 0

is a second-order-cone representable convex problem.  This substitution is
deliberate: the article's own convergence rates are those of a subgradient
scheme, which at unfolding sizes (``n`` of a few tens of bins) needs tens of
thousands of iterations to reach a level the epigraph solve attains in a few
tens of milliseconds, and the article's numerical target is precisely the
minimizer of ``\\Phi`` (their Theorem 3).  The stabilizer, the discrepancy
norm and the positivity constraint are the article's; only the inner
iteration is replaced by an exact solve.

Adaptation to Bonner-sphere unfolding
------------------------------------
* The response matrix ``A`` is rectangular (``m`` detectors, ``n`` energy
  bins) and typically underdetermined, so the discrepancy can often be
  driven to zero by a non-negative spectrum.  In that regime ``\\alpha``
  cancels out of the problem and the regularization is carried entirely by
  the *shape* of the stabilizer, i.e. by ``mu``.  ``alpha`` regains
  influence as soon as the readings are inconsistent (noise, dead-time
  corrections), which is when the trade-off between fit and shape matters.
* With ``weights="relative"`` row ``i`` of the discrepancy is scaled by
  ``1/|b_i|``, so the Chebyshev fit is applied to *relative* residuals;
  without this the single loudest sphere would dictate the solution.
  ``data_norm="l2"`` selects the weighted Euclidean norm instead.
* The Hölder distances ``d_ij`` are the logarithmic energy-grid distances
  (the grid is logarithmic), rescaled to unit diameter.  Pass ``dist`` to
  :func:`solve_vasin` to reproduce an experiment on another grid.
* ``quad_weights`` are the grid weights ``h_j`` of the quadratic term, i.e.
  the log-energy bin widths in :func:`unfold_vasin`; they make the
  stabilizer independent of the grid resolution.
* ``mu`` interpolates between two regimes: ``mu -> 0`` degenerates the
  seminorm into the oscillation ``max(u) - min(u)`` (a jump then costs the
  same as a ramp of equal height, which is exactly why the method keeps
  discontinuities), while ``mu = 1`` is the Lipschitz case, where a jump
  across one cell costs ``1/h`` and is strongly suppressed.
* Non-negativity is imposed as a constraint, corresponding to the Fejer
  projection ``P_Q`` of the paper's remark following eq. (12).

The epigraph program has ``n (n - 1)`` inequality constraints for the
seminorm, so cost grows quadratically in the number of energy bins; for
``n`` of a few hundred this is still well under a second.

Reference
---------
V. V. Vasin, T. I. Serazhitnikova, "Regular algorithm for approximation of
irregular solutions for Fredholm integral equations of the first kind",
Vyčislitel'nye Tekhnologii 15(2), 2010, 15-23 (math. eqs. (3)-(14),
Theorems 1-4).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["holder_seminorm", "solve_vasin", "unfold_vasin"]

logger = logging.getLogger(__name__)

_DATA_NORMS = ("linf", "l2")
_SOLVER_ORDER = ("CLARABEL", "SCS", "ECOS")


def _pair_inverse_distance(
    dist: NDArray[np.float64] | None, n: int, mu: float
) -> NDArray[np.float64]:
    """Build ``1 / |x_i - x_j|**mu`` for the Hölder seminorm.

    Parameters
    ----------
    dist : np.ndarray, optional
        Pairwise grid distances ``(n, n)``.  ``None`` selects a uniform
        grid.
    n : int
        Number of grid nodes.
    mu : float
        Hölder exponent.

    Returns
    -------
    np.ndarray
        ``(n, n)`` matrix with zero diagonal and ``1 / d_ij**mu`` off the
        diagonal, where ``d`` is ``dist`` rescaled to unit diameter.
    """
    if dist is None:
        idx = np.arange(n, dtype=float)
        D = np.abs(idx[:, None] - idx[None, :])
    else:
        D = np.abs(np.asarray(dist, dtype=float))
        if D.shape != (n, n):
            raise ValueError(f"dist must have shape ({n}, {n}), got {D.shape}")
    off_diag = D > 0
    diameter = float(D.max())
    if diameter > 0:
        D = D / diameter
    out = np.zeros((n, n), dtype=float)
    out[off_diag] = D[off_diag] ** (-mu)
    np.fill_diagonal(out, 0.0)
    return out


def _resolve_weights(
    weights: Any, b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Turn the ``weights`` argument into a per-reading scale vector."""
    m = len(b)
    if weights is None:
        return np.ones(m)
    if isinstance(weights, str):
        if weights != "relative":
            raise ValueError("weights must be None, 'relative' or an array")
        return 1.0 / np.maximum(np.abs(b), np.finfo(float).tiny)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != (m,):
        raise ValueError(f"weights must have length {m}, got {w.shape}")
    return w


def holder_seminorm(
    x: NDArray[np.float64], dist: NDArray[np.float64], mu: float = 0.1
) -> float:
    """Evaluate the Hölder seminorm ``max_ij |x_i - x_j| / d_ij**mu``.

    Parameters
    ----------
    x : np.ndarray
        Function values on the grid (n,).
    dist : np.ndarray
        Pairwise grid distances ``(n, n)``; a 1-D array is interpreted as
        the grid coordinates themselves.
    mu : float, optional
        Hölder exponent in ``(0, 1]`` (default: 0.1).

    Returns
    -------
    float
        The seminorm, computed with the same distance normalization
        (:func:`_pair_inverse_distance`) the solver uses.
    """
    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(dist, dtype=float)
    if d.ndim == 1:
        d = np.abs(d[:, None] - d[None, :])
    inv_d_mu = _pair_inverse_distance(d, len(x), mu)
    return float(np.max((x[:, None] - x[None, :]) * inv_d_mu))


def solve_vasin(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64] | None = None,
    alpha: float = 1e-2,
    mu: float = 0.1,
    data_norm: str = "linf",
    weights: Any = "relative",
    dist: NDArray[np.float64] | None = None,
    quad_weights: NDArray[np.float64] | None = None,
    solver: str = "default",
) -> tuple[NDArray[np.float64], int, bool]:
    """Minimize the Vasin-Serazhitnikova functional (eq. (8) of the paper).

    Solves the problem::

        min over x >= 0 of   ||W (A x - b)||_inf
            + alpha * (1/2 sum_j q_j x_j^2 + ||x||_inf + [x]_{mu,inf})

    exactly through its epigraph form, where ``[x]_{mu,inf}`` is the Hölder
    seminorm ``max over i != j of |x_i - x_j| / d_ij^mu``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (accepted for API compatibility): the functional has no
        a-priori term, and the paper's prox iteration starts from zero.
    alpha : float, optional
        Regularization parameter (default: 1e-2).  See the module notes:
        for an underdetermined system with an exact non-negative fit the
        solution is ``alpha``-independent.
    mu : float, optional
        Hölder exponent in ``(0, 1]`` (default: 0.1).  Small ``mu`` makes
        the stabilizer insensitive to the frequency of oscillation and
        therefore jump-preserving; ``mu = 1`` is the Lipschitz case.
    data_norm : {"linf", "l2"}, optional
        Discrepancy norm: Chebyshev maximum (the paper, default) or the
        weighted Euclidean norm ``||W (A x - b)||_2``.
    weights : None, "relative" or array-like, optional
        Per-reading scale of the discrepancy term (default: ``'relative'``,
        i.e. ``1/|b_i|``).  ``None`` uses unit weights, as in the paper.
    dist : np.ndarray, optional
        Pairwise grid distances ``(n, n)`` for the Hölder seminorm.
        ``None`` uses a uniform grid; :func:`unfold_vasin` passes the
        log-energy distances of the actual grid.
    quad_weights : np.ndarray, optional
        Grid weights ``q_j`` of the quadratic stabilizer term (default:
        ones, i.e. the plain ``1/2 ||x||^2``).
    solver : str, optional
        CVXPY solver name; ``'default'`` tries CLARABEL, SCS, ECOS.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        ``(spectrum, attempts, converged)``, where ``attempts`` is the
        number of solvers tried and ``converged`` reports whether one of
        them returned an optimal (or optimal-inaccurate) status.
    """
    A, b, _ = validate_system(A, b, x0=x0)
    if data_norm not in _DATA_NORMS:
        raise ValueError(f"data_norm must be one of {_DATA_NORMS}")
    if not 0.0 < mu <= 1.0:
        raise ValueError(f"mu must be in (0, 1], got {mu}")
    alpha = max(float(alpha), 0.0)

    import cvxpy as cp

    m, n = A.shape
    W = _resolve_weights(weights, b)
    inv_d_mu = _pair_inverse_distance(dist, n, mu)
    if quad_weights is None:
        q = np.ones(n)
    else:
        q = np.asarray(quad_weights, dtype=float).ravel()
        if q.shape != (n,):
            raise ValueError(f"quad_weights must have length {n}, got {q.shape}")

    iu, ju = np.triu_indices(n, 1)
    scale = inv_d_mu[iu, ju]
    # Signed pair-difference operator, with the distance weight baked in:
    # the seminorm is then the infinity norm of one matrix product.
    n_pairs = len(iu)
    pairs = sparse.csr_matrix(
        (
            np.column_stack((scale, -scale)).ravel(),
            (np.repeat(np.arange(n_pairs), 2), np.column_stack((iu, ju)).ravel()),
        ),
        shape=(n_pairs, n),
    )

    x = cp.Variable(n, nonneg=True)
    s = cp.Variable()
    ell = cp.Variable()
    residual = cp.multiply(W, A @ x - b)
    # Each stabilizer term is a maximum, so it enters the epigraph as a
    # single second-order-cone norm rather than as one linear constraint per
    # bin or pair.  Same optimum, far fewer rows: the linear form routinely
    # stops at 'optimal_inaccurate' because the objective is orders of
    # magnitude smaller than the raw readings.
    constraints = [
        s >= cp.norm(x, "inf"),
        # ell >= 0 also covers a single-bin grid, where the pairs are empty
        ell >= cp.norm(pairs @ x, "inf"),
        ell >= 0,
    ]
    if data_norm == "linf":
        t = cp.Variable()
        constraints.append(t >= cp.norm(residual, "inf"))
        discrepancy = t
    else:
        discrepancy = cp.norm(residual, 2)
    objective = cp.Minimize(
        discrepancy + alpha * (0.5 * cp.sum(cp.multiply(q, cp.square(x))) + s + ell)
    )
    problem = cp.Problem(objective, constraints)

    if solver == "default" or solver is None:
        order = list(_SOLVER_ORDER)
    else:
        order = [solver] + [s_name for s_name in _SOLVER_ORDER if s_name != solver]

    attempts = 0
    for name in order:
        attempts += 1
        try:
            getattr(cp, name)
            problem.solve(solver=name)
        except Exception as exc:  # solver not installed, or numerically failed
            logger.debug("Solver %s failed: %s", name, exc)
        else:
            solved = problem.status in ("optimal", "optimal_inaccurate")
            if solved and x.value is not None:
                return np.maximum(np.asarray(x.value, dtype=float), 0.0), attempts, True

    return np.zeros(n), attempts, False


def unfold_vasin(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    alpha: float = 1e-2,
    mu: float = 0.1,
    data_norm: str = "linf",
    weights: Any = "relative",
    solver: str = "default",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum with the Vasin-Serazhitnikova method.

    Tikhonov regularization with the non-differentiable Hölder-space
    stabilizer and a Chebyshev discrepancy, minimized exactly through the
    epigraph form of the functional.  The stabilizer is almost blind to the
    frequency of oscillation, so the method keeps sharp spectral structures
    that smooth-penalty methods smear out.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin logarithmic grid widths; used as the quadrature weights of
        the quadratic stabilizer term and, with ``E_MeV``, for the Hölder
        distances.
    initial_spectrum : np.ndarray, optional
        Accepted for API compatibility; the functional has no a-priori term.
    alpha : float, optional
        Regularization parameter (default: 1e-2).
    mu : float, optional
        Hölder exponent in ``(0, 1]`` (default: 0.1).
    data_norm : {"linf", "l2"}, optional
        Discrepancy norm (default: ``'linf'``, the paper's choice).
    weights : None, "relative" or array-like, optional
        Per-reading scale of the discrepancy term (default: ``'relative'``).
    solver : str, optional
        CVXPY solver name (default: ``'default'``).
    calculate_errors : bool, optional
        Calculate Monte-Carlo uncertainty (default: False).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        Monte-Carlo variance reduction (default: 'none').
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma reading uncertainties for Monte-Carlo propagation.
    reading_covariance : np.ndarray, optional
        Absolute reading covariance matrix for Monte-Carlo propagation.
    noise_model : str, optional
        Monte-Carlo noise model (default: 'gaussian').
    measurement_time : float, optional
        Measurement time for the 'poisson' noise model.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary, with extra keys ``alpha``, ``mu``,
        ``data_norm`` and ``weights``, plus ``iterations`` and
        ``converged`` from the solver.
    """
    E = np.asarray(E_MeV, dtype=float).ravel()
    if len(E) != n_energy_bins:
        raise ValueError(f"E_MeV has length {len(E)}, expected {n_energy_bins}")
    log_E = np.log(np.maximum(E, np.finfo(float).tiny))
    dist = np.abs(log_E[:, None] - log_E[None, :])
    if ln_steps is None:
        quad_weights = None
    else:
        quad_weights = np.asarray(ln_steps, dtype=float).ravel()
        if len(quad_weights) != n_energy_bins:
            raise ValueError(
                f"ln_steps has length {len(quad_weights)}, expected {n_energy_bins}"
            )

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.zeros(n_energy_bins),
        solve_func=make_solve_wrapper(
            solve_vasin,
            alpha=alpha,
            mu=mu,
            data_norm=data_norm,
            weights=weights,
            dist=dist,
            quad_weights=quad_weights,
            solver=solver,
        ),
        solve_kwargs={},
        method_name="Vasin",
        extra_output={
            "alpha": float(alpha),
            "mu": float(mu),
            "data_norm": data_norm,
            "weights": (
                weights if isinstance(weights, (str, type(None))) else "custom"
            ),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        variance_reduction=variance_reduction,
        random_state=random_state,
        save_result=save_result,
        reading_uncertainties=reading_uncertainties,
        reading_covariance=reading_covariance,
        noise_model=noise_model,
        measurement_time=measurement_time,
    )
