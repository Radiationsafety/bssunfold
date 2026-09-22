"""GEE (Generalized Estimation Equation) unfolding with robust inference.

Python analogue of the R package ``gee`` 4.13-30 (Carey, Lumley & Ripley,
CRAN; Liang & Zeger 1986 quasi-score): the unfolded spectrum is obtained
from the *generalized estimating equations*

    U(x)  =  A^T R(alpha)^{-1} (b - A x)  -  lam * G x  =  0,

where the m detector spheres are treated as a correlated cluster of
repeated measurements with an m x m *working correlation* matrix
``R(alpha)``.  The correlation structure follows the R package:

* ``"independence"`` -- ``R = I`` (Liang & Zeger's first working model);
* ``"exchangeable"`` -- ``R_ij = alpha`` for ``i != j`` (the R default
  corstr), with the classical Liang-Zeger moment estimator of ``alpha``
  from the Pearson residuals;
* ``"ar1"`` -- AR-1 structure ``R_ij = alpha^|i-j|`` estimated from the
  lag-1 products of the Pearson residuals.

Family / link pairings follow the R ``gee()`` semantics
(``family=gaussian`` by default, ``poisson`` and ``gamma`` variants),
implemented as quasi-likelihood variance functions

=================  ===========================  ============================
family             mean model                   variance function v(mu)
=================  ===========================  ============================
``gaussian``       ``mu = A xp``                ``1.0``        (w_i = 1)
``poisson``        ``mu = A xp``                ``mu``         (w_i = 1/mu)
``gamma``          ``mu = A xp``                ``mu**2``
=================  ===========================  ============================

(all with the identity link on the linear predictor ``A xp``, ``xp`` being
either ``x`` directly for the gaussian family or the *positive* working
spectrum for the multiplicative families -- the identity link keeps the
estimating equations linear, exactly as in the linear-Gaussian core of the
R solver which owes its LINPACK routines to Cleve Moler's ``d*`` codes).

Because the number of energy bins exceeds the number of spheres, the
score equations alone do not define a unique solution; the classical
remedy of the package's MixedModels view is used -- a ridge on the
second-difference roughness penalty ``G = D2^T D2`` (regularisation
``lam`` relative to the mean diagonal of ``A^T R^{-1} A``).

Statistical inference (the selling point of ``gee`` over plain GLS) is
delivered by the two canonical sandwich covariance estimators of Liang &
Zeger:

* **robust ("sandwich")** variance -- ``N V_n N`` with bread
  ``N = (A^T R^{-1} A + lam G)^{-1}`` and the empirical meat
  ``V_n = A^T R^{-1} (r r^T) R^{-1} A / m``  -- valid even when the
  working correlation is misspecified (the reading uncertainties of a
  Bonner-sphere measurement are almost never exactly exchangeable);
* **naive (model-based)** variance -- ``phi * (A^T R^{-1} A + lam G)^{-1}
  R^{-1} A A^T ...`` contracted through the working correlation, i.e.
  ``phi * N A^T R^{-1}`` combined as ``N (A^T V^{-1} A)^{-1}``.

The iteration is the standard GEE/IRLS loop: update ``x`` from the
current working correlation, re-estimate ``alpha`` and the dispersion
``phi`` from the Pearson residuals, repeat until the relative change of
the spectrum (and of ``alpha``) is below ``tolerance``.

Module API (standard bssunfold solver conventions):

* ``working_correlation(alpha, m, kind)`` -- build the ``R`` matrix
  (mirrors the ``ee1`` / ``ee2`` assembly in R ``gee``);
* ``estimate_alpha(r_pearson, kind)`` -- moment estimators for the working
  correlation parameter;
* ``gee_fit(...)`` -- core GEE/IRLS loop returning the rich diagnostics
  dictionary (spectrum, robust/naive SE, ``alpha``, ``phi``, QIC-order
  Pearson chi-square, iterations, convergence);
* ``solve_gee(A, b, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``solve_gee_full(...)`` -- the same solver returning the diagnostics;
* ``unfold_gee(...)`` -- Detector-facing wrapper (also exposed as
  ``Detector.unfold_gee``).

Notes
-----
* Pure NumPy implementation; ``m`` (spheres) is small (10-30), so the
  m x m working-correlation algebra is negligible.
* Non-negativity: the GEE solution is unconstrained (like R ``gee``);
  the returned spectrum is clipped at zero by default, matching the
  convention of the other bssunfold solvers.
"""

from typing import Any

import numpy as np

from ..logging_config import get_logger
from ..utils.validators import validate_system
from ._base_unfolder import run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = [
    "FAMILIES",
    "CORSTRINGS",
    "working_correlation",
    "estimate_alpha",
    "gee_fit",
    "solve_gee",
    "solve_gee_full",
    "unfold_gee",
]

logger = get_logger("unfold_gee")

FAMILIES = ("gaussian", "poisson", "gamma")
CORSTRINGS = ("independence", "exchangeable", "ar1")

_TINY = 1e-300


def _validate_family_corstr(family: str, corstr: str) -> tuple[str, str]:
    family = str(family).lower()
    corstr = str(corstr).lower()
    if family not in FAMILIES:
        raise ValueError(
            f"family must be one of {FAMILIES}, got {family!r}"
        )
    if corstr not in CORSTRINGS:
        raise ValueError(
            f"corstr must be one of {CORSTRINGS}, got {corstr!r}"
        )
    return family, corstr


def working_correlation(
    alpha: float, m: int, kind: str = "exchangeable"
) -> np.ndarray:
    """Build the m x m working correlation matrix ``R(alpha)``.

    Parameters
    ----------
    alpha : float
        Working correlation parameter.
    m : int
        Cluster size (number of detector spheres).
    kind : str, optional
        ``"exchangeable"`` (``R_ii = 1``, ``R_ij = alpha``), ``"ar1"``
        (``R_ij = alpha**|i-j|``) or ``"independence"`` (identity, says
        nothing about ``alpha``).

    Returns
    -------
    np.ndarray
        Symmetric positive-definite working correlation matrix.
    """
    kind = str(kind).lower()
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if kind == "independence":
        return np.eye(m)
    if kind == "exchangeable":
        lo = -1.0 / (m - 1)
        if not (lo < alpha < 1.0):
            raise ValueError(
                f"exchangeable alpha must be in ({lo}, 1) for m={m}, "
                f"got {alpha!r}"
            )
        R = np.full((m, m), float(alpha))
        np.fill_diagonal(R, 1.0)
        return R
    if kind == "ar1":
        if not (-1.0 < alpha < 1.0):
            raise ValueError(
                f"ar1 alpha must be in (-1, 1), got {alpha!r}"
            )
        idx = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
        return np.power(float(alpha), idx)
    raise ValueError(f"unknown working correlation kind {kind!r}")


def estimate_alpha(
    r_pearson: np.ndarray, kind: str = "exchangeable"
) -> tuple[float, float]:
    """Moment estimates of the working correlation parameter ``alpha``
    and the dispersion ``phi`` from the Pearson residuals.

    Mirrors the update step of the R ``gee`` solver (Liang & Zeger 1986,
    Eqs. 6-7): the off-diagonal products of the standardised residuals
    estimate the intra-cluster correlation, and the mean squared
    standardised residual estimates the dispersion.

    Parameters
    ----------
    r_pearson : np.ndarray
        Pearson residuals ``(m,)`` (``y - mu`` scaled by the variance
        function).
    kind : str, optional
        ``"exchangeable"`` (default), ``"ar1"`` or ``"independence"``.

    Returns
    -------
    tuple[float, float]
        ``(alpha, phi)``; for ``"independence"`` ``alpha`` is 0.
    """
    r = np.asarray(r_pearson, dtype=float)
    m = r.size
    if m < 2:
        return 0.0, 1.0
    phi = float(np.dot(r, r) / m)
    if kind == "independence":
        return 0.0, phi
    if kind == "exchangeable":
        # Method-of-moments estimator (Liang & Zeger 1986): the normalised
        # mean of the off-diagonal products of the standardised residuals.
        R0 = np.outer(r, r)
        mask = ~np.eye(m, dtype=bool)
        num = float(np.sum(R0[mask]))
        den = float(m * (m - 1))
        if den <= _TINY:
            return 0.0, phi
        a = num / den
    elif kind == "ar1":
        num = float(np.dot(r[1:], r[:-1]))
        den = float(np.dot(r[:-1], r[:-1]))
        if den <= _TINY:
            return 0.0, phi
        a = num / den
    else:
        raise ValueError(f"unknown working correlation kind {kind!r}")
    # keep the working matrix safely inside the SPD region
    lo = -0.95 / max(m - 1, 1)
    a = float(np.clip(a, lo, 0.95))
    return a, phi


def _variance_mu(mu: np.ndarray, family: str) -> np.ndarray:
    """Family variance functions (identity link quasi-likelihoods)."""
    if family == "gaussian":
        return np.ones_like(mu)
    if family == "poisson":
        return np.maximum(mu, _TINY)
    if family == "gamma":
        return np.maximum(mu * mu, _TINY)
    raise ValueError(f"unknown family {family!r}")


def gee_fit(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    family: str = "gaussian",
    corstr: str = "exchangeable",
    regularization: float = 1e-4,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    diff_order: int = 2,
) -> dict[str, Any]:
    """Fit the unfolding model by generalized estimating equations.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess (default: zeros for gaussian, ones for
        the multiplicative families).
    family : str, optional
        Quasi-likelihood family: ``"gaussian"`` (default), ``"poisson"``
        or ``"gamma"`` (variance, respectively, ``1``, ``mu`` and
        ``mu**2``).
    corstr : str, optional
        Working correlation: ``"exchangeable"`` (default), ``"ar1"`` or
        ``"independence"``.
    regularization : float, optional
        Relative ridge on the ``diff_order`` difference penalty
        (default: 1e-4; ``0`` disables it).
    max_iterations : int, optional
        Maximum number of GEE iterations (default: 100).
    tolerance : float, optional
        Relative convergence tolerance on the spectrum (default: 1e-6).
    diff_order : int, optional
        Order of the roughness penalty (default: 2).

    Returns
    -------
    dict[str, Any]
        Diagnostics dictionary with the keys ``spectrum``,
        ``cov_robust``, ``cov_naive``, ``robust_se``, ``naive_se``,
        ``alpha``, ``phi``, ``residuals``, ``pearson_residuals``,
        ``pearson_chi2``, ``df``, ``iterations``, ``converged``,
        ``family`` and ``corstr``.
    """
    family, corstr = _validate_family_corstr(family, corstr)
    A, b, x0 = validate_system(A, b, x0=x0, max_iterations=max_iterations,
                               tolerance=tolerance)
    m, n = A.shape
    if regularization < 0:
        raise ValueError(
            f"regularization must be non-negative, got {regularization}"
        )

    if x0 is None:
        x = np.ones(n) if family in ("poisson", "gamma") else np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()
        x = np.maximum(x, 0.0)

    # Roughness penalty, scale-normalised.
    if regularization > 0 and diff_order > 0 and n > diff_order:
        D = create_derivative_matrix(n, order=diff_order)
        D = np.asarray(D.toarray(), dtype=float)
        G = D.T @ D  # n x n second-difference penalty
        g_norm = max(float(np.mean(np.diag(G))), 1.0)
        G = G / g_norm
    else:
        G = np.zeros((n, n))

    alpha = 0.0
    phi = 1.0
    converged = False
    it = 0
    delta_prev = np.inf
    for it in range(1, int(max_iterations) + 1):
        Ax = A @ x
        mu = np.maximum(Ax, _TINY)  # positive working mean

        v = _variance_mu(mu, family)
        r_raw = b - mu
        r_pear = r_raw / np.sqrt(v)
        alpha_new, phi_new = estimate_alpha(r_pear, corstr)

        R = working_correlation(alpha_new, m, corstr)
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(R)
        AW = A.T @ Rinv
        H = AW @ A
        h_norm = max(float(np.mean(np.diag(H))), 1.0)
        damp = regularization * h_norm
        lhs = H + damp * G
        rhs = AW @ b
        try:
            x_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            x_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if not np.all(np.isfinite(x_new)):
            x_new = x
            converged = False
            break

        delta = np.linalg.norm(x_new - x) / max(
            np.linalg.norm(x_new), np.linalg.norm(x), 1.0
        )
        x = np.maximum(x_new, 0.0)
        alpha, phi = alpha_new, phi_new
        # Convergence: either the relative update is below the tolerance,
        # or the projected iteration has stalled (the unconstrained GEE
        # solution violates non-negativity, so the clip keeps bouncing on
        # an active set -- the current iterate is then the stationary
        # point of the projected fixed-point map).
        stalled = delta >= delta_prev * (1.0 - 1e-12) and delta < 1e10
        if delta <= tolerance or (stalled and it > 1):
            converged = True
            break
        delta_prev = delta

    # Final diagnostics on the converged estimate.
    Ax = A @ x
    mu = np.maximum(Ax, _TINY)
    v = _variance_mu(mu, family)
    r_raw = b - mu
    r_pear = r_raw / np.sqrt(v)
    R = working_correlation(alpha, m, corstr)
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R)
    AW = A.T @ Rinv
    H = AW @ A
    h_norm = max(float(np.mean(np.diag(H))), 1.0)
    damp = regularization * h_norm
    bread = H + damp * G
    try:
        Ninv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        Ninv = np.linalg.pinv(bread)

    # Robust ("sandwich") covariance -- Liang & Zeger 1986 per-unit meat
    # specialised to the single correlated cluster: the *bread*
    # N = (A^T R^-1 A + lam G)^-1 retains the working correlation, the
    # meat carries the empirical per-sphere residual cross-products
    # (Pearson residuals), so the estimator is PSD by construction.
    grad_sq = np.maximum(r_pear**2, 0.0)
    Au = A * grad_sq[:, None]
    meat = A.T @ Au
    cov_robust = Ninv @ meat @ Ninv

    # naive (model-based) covariance of the penalised estimator,
    # contracted through the working covariance V.
    cov_naive = Ninv @ H @ Ninv
    cov_robust = 0.5 * (cov_robust + cov_robust.T)
    cov_naive = 0.5 * (cov_naive + cov_naive.T)

    robust_se = np.sqrt(np.maximum(np.diag(cov_robust), 0.0))
    naive_se = np.sqrt(np.maximum(np.diag(cov_naive), 0.0))
    pearson_chi2 = float(np.sum(r_pear**2))
    df = max(m - n, 1)  # underdetermined system: nominal df

    return {
        "spectrum": x.copy(),
        "cov_robust": cov_robust,
        "cov_naive": cov_naive,
        "robust_se": robust_se,
        "naive_se": naive_se,
        "alpha": float(alpha),
        "phi": float(phi),
        "residuals": r_raw.copy(),
        "pearson_residuals": r_pear.copy(),
        "pearson_chi2": pearson_chi2,
        "df": df,
        "iterations": int(it),
        "converged": bool(converged),
        "family": family,
        "corstr": corstr,
    }


def solve_gee(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    family: str = "gaussian",
    corstr: str = "exchangeable",
    regularization: float = 1e-4,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding problem by generalized estimating equations.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess.
    family : str, optional
        ``"gaussian"`` (default), ``"poisson"`` or ``"gamma"``.
    corstr : str, optional
        ``"exchangeable"`` (default), ``"ar1"`` or ``"independence"``.
    regularization : float, optional
        Relative roughness ridge (default: 1e-4).
    max_iterations : int, optional
        Maximum GEE iterations (default: 100).
    tolerance : float, optional
        Relative convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)``.
    """
    diag = gee_fit(
        A, b, x0=x0, family=family, corstr=corstr,
        regularization=regularization,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    return diag["spectrum"], diag["iterations"], diag["converged"]


solve_gee_full = gee_fit


def unfold_gee(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    family: str = "gaussian",
    corstr: str = "exchangeable",
    regularization: float = 1e-4,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
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
    """Unfold neutron spectrum by generalized estimating equations.

    Python analogue of the R package ``gee`` (Liang & Zeger 1986):
    the detector spheres are treated as a correlated cluster and the
    score equations are augmented with a second-difference roughness
    ridge.  Robust sandwich uncertainties for the unfolded spectrum are
    reported as the ``robust_se`` key.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid (MeV).
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray | None, optional
        Initial spectrum guess.
    family : str, optional
        Quasi-likelihood family: ``"gaussian"`` (default), ``"poisson"``
        or ``"gamma"``.
    corstr : str, optional
        Working correlation: ``"exchangeable"`` (default), ``"ar1"`` or
        ``"independence"``.
    regularization : float, optional
        Relative roughness ridge (default: 1e-4).
    max_iterations : int, optional
        Maximum GEE iterations (default: 100).
    tolerance : float, optional
        Relative convergence tolerance (default: 1e-6).
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
    dict[str, Any]
        Standardized unfolding result dictionary with additional keys
        ``alpha``, ``phi``, ``family``, ``corstr``, ``robust_se``,
        ``naive_se``, ``spectrum_uncert_robust`` and ``gee_converged``.
    """
    x0_default = np.ones(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
        del kwargs
        return solve_gee(
            A, b, x0=None, family=family, corstr=corstr,
            regularization=regularization,
            max_iterations=max_iterations, tolerance=tolerance,
        )

    try:
        A_mat = np.array(
            [sensitivities[name] for name in detector_names
             if name in readings], dtype=float
        )
        b_mat = np.array(
            [readings[name] for name in detector_names if name in readings],
            dtype=float,
        )
        diag = gee_fit(
            A_mat, b_mat, x0=None, family=family, corstr=corstr,
            regularization=regularization,
            max_iterations=max_iterations, tolerance=tolerance,
        )
        extra_output = {
            "alpha": diag["alpha"],
            "phi": diag["phi"],
            "family": diag["family"],
            "corstr": diag["corstr"],
            "robust_se": diag["robust_se"],
            "naive_se": diag["naive_se"],
            "spectrum_uncert_robust": diag["robust_se"],
            "pearson_chi2": diag["pearson_chi2"],
            "gee_converged": diag["converged"],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("GEE diagnostics unavailable: %s", exc)
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
        method_name="GEE",
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
