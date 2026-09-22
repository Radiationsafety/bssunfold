"""P-spline mixed-model unfolding with REML smoothing selection.

Python analogue of the R package ``LMMsolver`` (Boer 2023): the unfolded
spectrum is represented as a P-spline (penalised B-spline) and the
smoothness is selected automatically by restricted maximum likelihood
(REML) in a linear mixed-model (LMM) formulation.

Method summary
--------------
The spectrum is parameterised with a B-spline basis (Wand & Ormerod 2008;
Eilers & Marx 1996 P-splines),

    x(E) = sum_s c_s B_s(E)  =:  B c,

so the Fredholm system ``b = A x`` becomes a linear mixed model

    b = A B c + eps,
    c ~ N(0, sigma2_e * lam * G^-1),   G = D^(d)^T D^(d),

where ``D^(d)`` is the ``d``-th order difference matrix (the classic
P-spline penalty) and ``lam`` is the smoothing parameter.  Following the
mixed-model reparameterisation, the coefficient vector is split into a
fixed (unpenalised) part spanning the null space of ``G`` -- a polynomial
trend of degree ``d - 1`` -- and a random (penalised) part spanning its
range space:

    c = U_fixed beta + U_random b_random,   b_random ~ N(0, sigma2_e * lam * L^-1),

with ``G = U diag(g) U^T`` and ``L = diag(g_random)`` the positive
eigenvalues.  For a trial smoothing parameter ``lam`` the coefficients
are obtained from the Henderson mixed-model equations (exactly the sparse
system solved by ``LMMsolver::LMMsolve``):

    [ X^T W X    X^T W Z      ] [ beta    ]   [ X^T W y ]
    [ Z^T W X    Z^T W Z + lam L ] [ b_ran ] = [ Z^T W y ],

and ``lam`` itself is estimated by maximising the REML profile
log-likelihood

    l_R(lam) = -1/2 [ (m - p_f) log sigma2_hat
                     + log|V| + log|X^T V^-1 X| ],
    V = I + lam Z L^-1 Z^T,   sigma2_hat = SS / (m - p_f),

where ``SS`` is the weighted residual sum of squares of the REML
projection.  The 1-D profile is optimised with Brent's method on the
log-relative scale, which makes the method insensitive to the absolute
scale of the response matrix and of the readings.

The result is a smooth, data-adaptive unfolding with a *statistically*
selected smoothing parameter -- no manual trial and error -- together
with diagnostics (REML value, effective dimension of the fit, residual
variance estimate).

Module API (standard bssunfold solver conventions):

* ``difference_matrix(n, order)`` -- the P-spline difference penalty;
* ``pspline_penalty(n, order)`` -- symmetric penalty matrix ``G``;
* ``mixed_model_split(n, order, ...)`` -- eigen-based fixed/random split;
* ``reml_profile(...)`` -- REML profile log-likelihood for one ``lam``;
* ``solve_pspline_reml(A, b, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``solve_pspline_reml_full(...)`` -- the same solver returning a rich
  diagnostics dictionary;
* ``unfold_pspline_reml(...)`` -- Detector-facing wrapper (also exposed
  as ``Detector.unfold_pspline_reml``).

Scaling note
------------
The penalty ``G`` and the data term ``B^T A^T W A B`` live on very
different scales.  Internally the smoothing parameter is optimised on a
*relative* scale, ``lam = lam_relative * lam_ref`` with ``lam_ref``
equalising the average trace of both terms, and the scan covers
``lam_relative in [1e-6, 1e6]`` by default.  Both ``lam`` and
``lam_relative`` are reported in the result.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from ..logging_config import get_logger
from ..utils.validators import validate_system
from ._base_unfolder import run_unfolding

__all__ = [
    "difference_matrix",
    "pspline_penalty",
    "mixed_model_split",
    "reml_profile",
    "select_lambda_reml",
    "solve_pspline_reml",
    "solve_pspline_reml_full",
    "unfold_pspline_reml",
]

logger = get_logger("unfold_pspline_reml")

# Numerical guards -----------------------------------------------------------
_TINY = 1e-300

# Default search interval for the relative smoothing parameter (log10 units)
LAMBDA_REL_BOUNDS = (1e-6, 1e6)

_VALID_KNOT_SPACING = ("auto", "uniform", "log")
_VALID_WEIGHTS = ("uniform", "poisson")

# Eigenvalues of G below this relative threshold are treated as the
# (exactly zero) null space of the difference penalty.
_NULLSPACE_RTOL = 1e9 * np.finfo(float).eps  # ~2.2e-7 relative


def difference_matrix(n: int, order: int = 2) -> np.ndarray:
    """Return the ``order``-th order difference matrix ``D^(order)``.

    The P-spline penalty is ``||D^(order) c||^2``; rows of ``D`` are
    finite differences of successive coefficients, so the penalty
    shrinks polynomial trends of degree ``order - 1`` but leaves them
    (and only them) unpenalised.

    Parameters
    ----------
    n : int
        Number of coefficients (columns).  Must exceed ``order``.
    order : int, optional
        Difference order ``d`` in [1, 4] (default: 2, the classic
        second-difference P-spline penalty of Eilers & Marx 1996).

    Returns
    -------
    np.ndarray
        Dense ``(n - order, n)`` difference matrix.
    """
    n = int(n)
    order = int(order)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    if order < 1 or order > 4:
        raise ValueError(f"order must be in [1, 4], got {order}")
    if n <= order:
        raise ValueError(
            f"n ({n}) must be greater than the difference order ({order})"
        )
    return np.diff(np.eye(n), order, axis=0)


def pspline_penalty(n: int, order: int = 2) -> np.ndarray:
    """Return the symmetric P-spline penalty matrix ``G = D^T D``.

    Parameters
    ----------
    n : int
        Number of spline coefficients.
    order : int, optional
        Difference order (default: 2).

    Returns
    -------
    np.ndarray
        Dense symmetric positive semi-definite ``(n, n)`` penalty matrix.
    """
    D = difference_matrix(n, order)
    return D.T @ D


def mixed_model_split(
    n: int, order: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a P-spline space into fixed and random subspaces.

    Diagonalises the penalty ``G = D^T D`` and splits its eigenbasis:

    * the null space of ``G`` (eigenvalues ~ 0, dimension ``order``)
      becomes the **fixed** part -- an unpenalised polynomial trend;
    * the range space (positive eigenvalues) becomes the **random**
      part with prior precision ``L = diag(g_random)``.

    This is the standard spectral reparameterisation of P-splines into
    a linear mixed model (Wand & Ormerod 2008) and mirrors the sparse
    fixed/random formulation used by ``LMMsolver``.

    Parameters
    ----------
    n : int
        Dimension of the spline coefficient space.
    order : int, optional
        Difference order of the penalty (default: 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(U_fixed, U_random, g_random)`` where ``U_fixed`` is
        ``(n, order)``, ``U_random`` is ``(n, n - order)`` and
        ``g_random`` are the positive penalty eigenvalues (ascending).
    """
    G = pspline_penalty(n, order)
    g, U = np.linalg.eigh(G)  # ascending eigenvalues
    g_max = max(float(g[-1]), _TINY)
    is_fixed = g <= _NULLSPACE_RTOL * g_max
    n_fixed = int(np.sum(is_fixed))
    if n_fixed == 0:
        # Degenerate configuration: treat the smallest eigen-direction
        # as fixed to keep a proper mixed model (should not happen for
        # difference penalties of order >= 1).
        is_fixed[np.argmin(g)] = True
        n_fixed = 1
    U_fixed = U[:, is_fixed]
    U_random = U[:, ~is_fixed]
    g_random = np.maximum(g[~is_fixed], _TINY)
    return U_fixed, U_random, g_random


def reml_profile(
    y_w: np.ndarray,
    X_w: np.ndarray,
    Z_w: np.ndarray,
    g_random: np.ndarray,
    lam: float,
) -> tuple[float, float]:
    """Evaluate the REML profile log-likelihood for one smoothing value.

    For the weighted mixed model ``y_w = X_w beta + Z_w b + eta`` with
    ``b ~ N(0, lam^-1 L^-1)`` (variance ratio ``lam``) the marginal
    covariance is ``V = I + lam Z_w L^-1 Z_w^T``.  The restricted
    log-likelihood (up to an additive constant) is

        l_R = -1/2 [ (m - p_f) log sigma2_hat
                     + log|V| + log|X_w^T V^-1 X_w| ],

    with ``sigma2_hat = SS / (m - p_f)`` and ``SS`` the residual sum of
    squares of the GLS projection of ``y_w``.

    Parameters
    ----------
    y_w : np.ndarray
        Weighted response ``(m,)``.
    X_w : np.ndarray
        Weighted fixed-effects design ``(m, p_f)``.
    Z_w : np.ndarray
        Weighted random-effects design ``(m, p_r)``.
    g_random : np.ndarray
        Prior precision entries ``L = diag(g_random)`` (``p_r,``).
    lam : float
        Smoothing parameter (variance ratio); must be positive.

    Returns
    -------
    tuple[float, float]
        ``(loglik, sigma2_hat)``.  Returns ``(-inf, nan)`` when the
        profile is not finite at this ``lam`` (optimizer contracts).
    """
    m = y_w.shape[0]
    p_f = X_w.shape[1]
    df = m - p_f
    if df < 1:
        return -np.inf, np.nan

    L_inv = 1.0 / np.maximum(g_random, _TINY)
    V = np.eye(m) + lam * (Z_w * L_inv) @ Z_w.T  # Z_w diag(L^-1) Z_w^T
    try:
        cV = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan

    def _solve_V(rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(cV.T, np.linalg.solve(cV, rhs))

    Vinv_X = _solve_V(X_w)
    XtVinvX = X_w.T @ Vinv_X
    sign, logdet_XtVX = np.linalg.slogdet(XtVinvX)
    if sign <= 0:
        return -np.inf, np.nan

    XtVinv_y = X_w.T @ _solve_V(y_w)
    try:
        beta = np.linalg.solve(XtVinvX, XtVinv_y)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan

    resid = y_w - Vinv_X @ beta
    ss = float(resid @ _solve_V(resid))
    if not np.isfinite(ss) or ss <= 0:
        # Exact fit (ss == 0) is legitimate for clean synthetic data but
        # makes log(sigma2) undefined; clamp to a tiny positive value.
        ss = max(ss, _TINY)

    sigma2 = ss / df
    _, logdet_V = np.linalg.slogdet(V)
    loglik = -0.5 * (df * np.log(sigma2) + logdet_V + logdet_XtVX)
    return float(loglik), float(sigma2)


def select_lambda_reml(
    y_w: np.ndarray,
    X_w: np.ndarray,
    Z_w: np.ndarray,
    g_random: np.ndarray,
    lam_ref: float,
    lam_bounds: tuple[float, float] = LAMBDA_REL_BOUNDS,
) -> dict[str, Any]:
    """Maximise the REML profile over the relative smoothing parameter.

    The profile is scanned with Brent's method in ``t = log10(lam_rel)``
    over ``log10(lam_bounds)``; the absolute smoothing parameter is
    ``lam = lam_ref * lam_rel``.  ``lam_ref`` equalises the average
    scale of the data term and the penalty term so that the default
    bounds are meaningful for any problem scaling.

    Parameters
    ----------
    y_w, X_w, Z_w : np.ndarray
        Weighted mixed-model design (see :func:`reml_profile`).
    g_random : np.ndarray
        Prior precision entries of the random part.
    lam_ref : float
        Reference (scale-equalising) smoothing parameter.
    lam_bounds : tuple[float, float], optional
        Search interval for the *relative* smoothing parameter.

    Returns
    -------
    dict[str, Any]
        Diagnostics with keys ``lam``, ``lam_relative``, ``sigma2``,
        ``reml_loglik``, ``converged`` and ``n_iterations`` (number of
        profile evaluations).
    """
    lo, hi = np.log10(max(lam_bounds[0], _TINY)), np.log10(lam_bounds[1])
    n_fev = 0
    cache: dict[float, float] = {}

    def neg_loglik(t: float) -> float:
        nonlocal n_fev
        n_fev += 1
        if t in cache:
            return cache[t]
        lam = lam_ref * (10.0**t)
        ll, _ = reml_profile(y_w, X_w, Z_w, g_random, lam)
        val = -ll
        cache[t] = val
        return val

    res = minimize_scalar(neg_loglik, bounds=(lo, hi), method="bounded")
    converged = bool(np.isfinite(res.fun))
    t_opt = float(res.x)
    lam_rel = 10.0**t_opt
    lam = lam_ref * lam_rel
    ll, sigma2 = reml_profile(y_w, X_w, Z_w, g_random, lam)
    return {
        "lam": float(lam),
        "lam_relative": float(lam_rel),
        "sigma2": float(sigma2),
        "reml_loglik": float(ll),
        "converged": converged,
        "n_iterations": int(n_fev),
    }


def _build_weights(
    weights: str | np.ndarray | None,
    b: np.ndarray,
) -> np.ndarray:
    """Resolve the per-detector weights for the weighted least squares."""
    if weights is None or (isinstance(weights, str) and weights == "uniform"):
        return np.ones(b.shape[0])
    if isinstance(weights, str):
        if weights == "poisson":
            floor = 1e-3 * max(float(np.max(b)), _TINY)
            return 1.0 / np.maximum(b, floor)
        raise ValueError(
            f"weights must be 'uniform', 'poisson' or an array, got {weights!r}"
        )
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != b.shape[0]:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match number of "
            f"readings ({b.shape[0]})"
        )
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive")
    return w


def solve_pspline_reml_full(
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    x0: np.ndarray | None = None,
    n_basis: int | None = None,
    spline_order: int = 4,
    diff_order: int = 2,
    knot_spacing: str = "auto",
    weights: str | np.ndarray | None = "uniform",
    lam_relative: float | None = None,
    lam_bounds: tuple[float, float] = LAMBDA_REL_BOUNDS,
) -> dict[str, Any]:
    """P-spline REML unfolding returning rich diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    E_MeV : np.ndarray
        Energy grid (MeV), ``n`` points.
    x0 : np.ndarray, optional
        Unused (kept for API compatibility with other solvers).
    n_basis : int, optional
        Dimension of the B-spline space; default
        ``min(n // 2, 30)`` clamped to ``[diff_order + 2, n]``.
    spline_order : int, optional
        B-spline order ``p`` (degree ``p - 1``), default 4 (cubic).
    diff_order : int, optional
        Difference order of the P-spline penalty, default 2.
    knot_spacing : str, optional
        ``"auto"`` (default), ``"uniform"`` or ``"log"`` interior knots.
    weights : str or np.ndarray, optional
        ``"uniform"`` (default), ``"poisson"`` (``w_i = 1 / b_i``) or an
        explicit positive weight array.
    lam_relative : float, optional
        Fixed *relative* smoothing parameter; skips REML optimisation
        when provided.
    lam_bounds : tuple[float, float], optional
        Search interval for the relative smoothing parameter.

    Returns
    -------
    dict[str, Any]
        Diagnostics with keys ``spectrum`` (before non-negativity
        clamping), ``coefficients``, ``n_basis``, ``spline_order``,
        ``diff_order``, ``knot_spacing``, ``weights``, ``lam``,
        ``lam_relative``, ``lam_ref``, ``sigma2``, ``reml_loglik``,
        ``ed`` (effective dimension), ``ed_norm``, ``reml_converged``
        and ``n_iterations``.
    """
    A, b, _ = validate_system(A, b)
    m, n = A.shape
    if m < diff_order + 2:
        raise ValueError(
            "P-spline REML requires at least diff_order + 2 = "
            f"{diff_order + 2} detector readings, got {m}"
        )

    # --- spline space -----------------------------------------------------
    if n_basis is None:
        n_basis = max(diff_order + 2, min(n // 2, 30))
    n_basis = int(n_basis)
    if n_basis < diff_order + 2:
        raise ValueError(
            f"n_basis ({n_basis}) must be >= diff_order + 2 ({diff_order + 2})"
        )
    if n_basis > n:
        raise ValueError(
            f"n_basis ({n_basis}) cannot exceed the number of energy "
            f"bins ({n})"
        )

    E = np.asarray(E_MeV, dtype=float).ravel()
    if E.shape[0] != n:
        raise ValueError(
            f"Length of E_MeV ({E.shape[0]}) must match number of energy "
            f"bins ({n})"
        )
    if knot_spacing not in _VALID_KNOT_SPACING:
        raise ValueError(
            f"knot_spacing must be one of {_VALID_KNOT_SPACING}, "
            f"got {knot_spacing!r}"
        )

    from .unfold_mlem_bs import build_bspline_basis

    B = build_bspline_basis(E, n_basis, spline_order=spline_order,
                            knot_spacing=knot_spacing)

    # --- mixed-model split of the penalty ---------------------------------
    U_fixed, U_random, g_random = mixed_model_split(n_basis, diff_order)
    X = A @ (B @ U_fixed)   # (m, p_f) unpenalised polynomial trend
    Z = A @ (B @ U_random)  # (m, p_r) penalised wiggly part

    w = _build_weights(weights, b)
    sw = np.sqrt(w)
    y_w = sw * b
    X_w = sw[:, None] * X
    Z_w = sw[:, None] * Z

    # --- scale-equalising reference smoothing parameter --------------------
    data_scale = float(np.sum(Z_w**2)) / max(Z_w.shape[1], 1)
    pen_scale = float(np.mean(g_random))
    lam_ref = max(data_scale / max(pen_scale, _TINY), _TINY)

    # --- smoothing parameter selection -------------------------------------
    if lam_relative is not None:
        lam_rel = float(lam_relative)
        if lam_rel <= 0 or not np.isfinite(lam_rel):
            raise ValueError(
                f"lam_relative must be positive and finite, got {lam_relative}"
            )
        lam = lam_ref * lam_rel
        ll, sigma2 = reml_profile(y_w, X_w, Z_w, g_random, lam)
        selection = {
            "lam": float(lam),
            "lam_relative": lam_rel,
            "sigma2": float(sigma2),
            "reml_loglik": float(ll),
            "converged": bool(np.isfinite(ll)),
            "n_iterations": 0,
        }
    else:
        selection = select_lambda_reml(
            y_w, X_w, Z_w, g_random, lam_ref, lam_bounds=lam_bounds
        )
    lam = selection["lam"]
    logger.info(
        "P-spline REML: lam_rel=%.3e (lam=%.3e), loglik=%.3f",
        selection["lam_relative"], lam, selection["reml_loglik"],
    )

    # --- Henderson mixed model equations ------------------------------------
    # [X^T W X   X^T W Z    ] [beta ]   [X^T W y]
    # [Z^T W X   Z^T W Z+lam L ] [b_r] = [Z^T W y]
    XtWX = X.T @ (w[:, None] * X)
    XtWZ = X.T @ (w[:, None] * Z)
    ZtWZ = Z.T @ (w[:, None] * Z)
    rhs = np.concatenate(
        [
            X.T @ (w * b),
            Z.T @ (w * b),
        ]
    )
    saddle = np.block(
        [
            [XtWX, XtWZ],
            [XtWZ.T, ZtWZ + lam * np.diag(g_random)],
        ]
    )
    try:
        coef = np.linalg.solve(saddle, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(saddle, rhs, rcond=None)[0]

    beta_hat = coef[: U_fixed.shape[1]]
    b_random_hat = coef[U_fixed.shape[1]:]
    spectrum = B @ (U_fixed @ beta_hat + U_random @ b_random_hat)

    # --- effective dimension ------------------------------------------------
    try:
        ZtZ_lam = ZtWZ + lam * np.diag(g_random)
        ed = U_fixed.shape[1] + float(
            np.trace(np.linalg.solve(ZtZ_lam, ZtWZ))
        )
    except np.linalg.LinAlgError:
        ed = float("nan")

    return {
        "spectrum": spectrum,
        "coefficients": coef.copy(),
        "n_basis": int(n_basis),
        "spline_order": int(spline_order),
        "diff_order": int(diff_order),
        "knot_spacing": knot_spacing,
        "weights": weights if isinstance(weights, str) else "array",
        "lam": float(lam),
        "lam_relative": float(selection["lam_relative"]),
        "lam_ref": float(lam_ref),
        "sigma2": float(selection["sigma2"]),
        "reml_loglik": float(selection["reml_loglik"]),
        "ed": float(ed),
        "ed_norm": float(ed / n_basis),
        "reml_converged": bool(selection["converged"]),
        "n_iterations": int(selection["n_iterations"]),
    }


def solve_pspline_reml(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    E_MeV: np.ndarray | None = None,
    n_basis: int | None = None,
    spline_order: int = 4,
    diff_order: int = 2,
    knot_spacing: str = "auto",
    weights: str | np.ndarray | None = "uniform",
    lam_relative: float | None = None,
    lam_bounds: tuple[float, float] = LAMBDA_REL_BOUNDS,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding problem with P-spline REML smoothing.

    The spectrum is represented by a P-spline; the smoothing parameter is
    selected by maximising the REML profile likelihood of the equivalent
    linear mixed model (see module docstring), then the Henderson mixed
    model equations are solved for the spline coefficients.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Unused (provided for API compatibility).
    E_MeV : np.ndarray, optional
        Energy grid (MeV), ``n`` points.  Required.
    n_basis : int, optional
        Dimension of the B-spline space (default: ``min(n // 2, 30)``).
    spline_order : int, optional
        B-spline order (default: 4, cubic).
    diff_order : int, optional
        Difference order of the P-spline penalty (default: 2).
    knot_spacing : str, optional
        Interior knot placement (default: ``"auto"``).
    weights : str or np.ndarray, optional
        ``"uniform"``, ``"poisson"`` or an explicit weight array.
    lam_relative : float, optional
        Fixed relative smoothing parameter (skips REML selection).
    lam_bounds : tuple[float, float], optional
        Search interval for the relative smoothing parameter.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)`` where ``iterations`` is
        the number of REML profile evaluations and ``converged``
        reflects the success of the REML optimisation and the final
        mixed-model solve.
    """
    if E_MeV is None:
        raise ValueError("E_MeV is required by solve_pspline_reml")
    diag = solve_pspline_reml_full(
        A,
        b,
        E_MeV,
        x0=x0,
        n_basis=n_basis,
        spline_order=spline_order,
        diff_order=diff_order,
        knot_spacing=knot_spacing,
        weights=weights,
        lam_relative=lam_relative,
        lam_bounds=lam_bounds,
    )
    spectrum = np.maximum(diag["spectrum"], 0.0)
    converged = diag["reml_converged"] and np.all(np.isfinite(spectrum))
    return spectrum, diag["n_iterations"], converged


def unfold_pspline_reml(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    n_basis: int | None = None,
    spline_order: int = 4,
    diff_order: int = 2,
    knot_spacing: str = "auto",
    weights: str | np.ndarray | None = "uniform",
    lam_relative: float | None = None,
    lam_bounds: tuple[float, float] = LAMBDA_REL_BOUNDS,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum with P-spline REML smoothing selection.

    Python analogue of the ``LMMsolver`` approach: the spectrum is a
    P-spline; the fixed part of the mixed model carries the unpenalised
    polynomial trend, the random part carries the wiggly components, and
    the smoothing parameter is the variance ratio estimated by REML.

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
    initial_spectrum : Optional[np.ndarray], optional
        Unused (kept for API compatibility).
    n_basis : Optional[int], optional
        Dimension of the B-spline space (default: ``min(n // 2, 30)``).
    spline_order : int, optional
        B-spline order (default: 4, cubic).
    diff_order : int, optional
        Difference order of the penalty (default: 2).
    knot_spacing : str, optional
        Interior knot placement: ``"auto"``, ``"uniform"`` or ``"log"``.
    weights : str or np.ndarray, optional
        ``"uniform"`` (default), ``"poisson"`` or an explicit array.
    lam_relative : Optional[float], optional
        Fixed relative smoothing parameter; skips REML selection.
    lam_bounds : Tuple[float, float], optional
        Search interval for the relative smoothing parameter.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary with additional keys
        ``lam``, ``lam_relative``, ``reml_loglik``, ``sigma2``, ``ed``
        and ``ed_norm``.
    """
    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
        return solve_pspline_reml(
            A,
            b,
            x0=kwargs.get("x0"),
            E_MeV=E_MeV,
            n_basis=n_basis,
            spline_order=spline_order,
            diff_order=diff_order,
            knot_spacing=knot_spacing,
            weights=weights,
            lam_relative=lam_relative,
            lam_bounds=lam_bounds,
        )

    try:
        diag = solve_pspline_reml_full(
            np.array([sensitivities[name] for name in detector_names
                      if name in readings], dtype=float),
            np.array([readings[name] for name in detector_names
                      if name in readings], dtype=float),
            E_MeV,
            n_basis=n_basis,
            spline_order=spline_order,
            diff_order=diff_order,
            knot_spacing=knot_spacing,
            weights=weights,
            lam_relative=lam_relative,
            lam_bounds=lam_bounds,
        )
        extra_output = {
            "n_basis": diag["n_basis"],
            "lam": diag["lam"],
            "lam_relative": diag["lam_relative"],
            "lam_ref": diag["lam_ref"],
            "sigma2": diag["sigma2"],
            "reml_loglik": diag["reml_loglik"],
            "ed": diag["ed"],
            "ed_norm": diag["ed_norm"],
            "reml_converged": diag["reml_converged"],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("P-spline REML diagnostics unavailable: %s", exc)
        extra_output = None

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
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="P-spline REML",
        extra_output=extra_output,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
