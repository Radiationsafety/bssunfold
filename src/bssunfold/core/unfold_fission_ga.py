"""Stochastic parametric unfolding with the Fission model (BonnerFinder).

Python port of the two-stage algorithm for the multisphere spectrometer
problem described in:

    I. N. Ogorodnikov, "Inverse problems of spectroscopy and spectrometry
    in applied research" (obratnye zadachi spektroskopii i
    spektrometrii v prikladnykh issledovaniyakh), Traektoriya
    Issledovaniy -- Chelovek, Priroda, Tekhnologii, no. 2 (10), pp. 42-83
    (2024).  Sections 4-5, function ``BonnerFinder()``.

The method follows the FRUIT paradigm of parameterized model curves
(Bedogni et al., NIM A 580, 1301 (2007)), which the article adopts as
its reference implementation.  The model spectrum is a superposition of
three neutron fractions (article eq. 4.29)::

    Phi(E) = a1 * (E/T0^2) * exp(-E/T0)
           + a2 * [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta)
           + a3 * E^alpha * exp(-E/TF),

with fixed constants ``T0 = 2.53e-8`` MeV (thermal peak) and
``Ed = 7.07e-8`` MeV (epithermal cutoff), and seven free parameters
``a1, a2, a3, b, beta, alpha, TF`` with the article's bounds

    a1, a2, a3 in [0, 1],  b in [-1/2, 1/2],  beta in [0, 1] MeV,
    alpha in [0, 1],  TF in [1, 2] MeV.

Stage 1 (article section "Elementy stokhasticheskogo algoritma") is a
stochastic global search of the parameter hypercube -- the article uses
the SciLab ``optim_ga`` genetic algorithm; here
``scipy.optimize.differential_evolution`` plays the same role while
minimizing the target function (article eq. 4.32)::

    dM(P) = sum_m | C_m - integral R_m(E) Phi(E; P) dE |.

Stage 2 (article section "Nelineynaya regressiya") refines the best
stochastic point ``P0 = argmin dM`` with a nonlinear least-squares
routine (SciLab ``leastsq`` in the article; ``scipy.optimize.
least_squares`` here).

The port additionally supports an optional free overall scale factor
``phi_scale`` (fitted on a log10 grid) so that the model can match
absolutely calibrated readings; set ``fit_scale=False`` for the exact
7-parameter normalized formulation of the article.

The article's validation criteria (norm of the model spectrum, per-sphere
relative uncertainties with alternating signs, FOM of the fit) are
returned in the ``validation`` entry of the result dictionary.
"""

from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from ._base_unfolder import _build_system, run_unfolding
from ._matrix_utils import compute_log_steps

__all__ = [
    "fission_model",
    "solve_fission_ga",
    "unfold_fission_ga",
    "FISSION_PARAM_NAMES",
    "FISSION_PARAM_BOUNDS",
    "T0_THERMAL",
    "ED_EPITHERMAL",
]

# Fixed constants of the Fission model (article eq. 4.29).
T0_THERMAL = 2.53e-8  # Thermal peak energy (MeV)
ED_EPITHERMAL = 7.07e-8  # Epithermal cutoff parameter (MeV)

# Free parameters of the article's 7-parameter Fission model.
FISSION_PARAM_NAMES = ("a1", "a2", "a3", "b", "beta", "alpha", "TF")

# Article's indicative parameter bounds (section 4, model Fission).
FISSION_PARAM_BOUNDS = {
    "a1": (0.0, 1.0),
    "a2": (0.0, 1.0),
    "a3": (0.0, 1.0),
    "b": (-0.5, 0.5),
    "beta": (1e-4, 1.0),
    "alpha": (0.0, 1.0),
    "TF": (1.0, 2.0),
}

# Boundaries of the fitted parameter vector:
#   [a1, a2, a3, b, beta, alpha, TF] (+ log10(phi_scale) when fit_scale).
_SCALE_NAME = "log10_phi_scale"


# ------------------------------------------------------------------ #
#  Fission model (article eq. 4.29)
# ------------------------------------------------------------------ #


def fission_model(
    E: np.ndarray,
    a1: float,
    a2: float,
    a3: float,
    b: float,
    beta: float,
    alpha: float,
    TF: float,
) -> np.ndarray:
    """Three-fraction Fission model spectrum (article eq. 4.29).

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    a1, a2, a3 : float
        Weights of the thermal, epithermal and fast fractions, [0, 1].
    b : float
        Slope of the epithermal tail, [-1/2, 1/2].
    beta : float
        Right cutoff of the epithermal tail (MeV), (0, 1].
    alpha : float
        Shape parameter of the fast fission peak, [0, 1].
    TF : float
        Position of the fast fission peak (MeV), [1, 2].

    Returns
    -------
    np.ndarray
        Model fluence spectrum per unit lethargy on the grid ``E``.
    """
    E = np.asarray(E, dtype=float)

    thermal = (E / (T0_THERMAL**2)) * np.exp(-E / T0_THERMAL)
    epithermal = (
        (1.0 - np.exp(-((E / ED_EPITHERMAL) ** 2)))
        * np.power(E, b - 1.0)
        * np.exp(-E / beta)
    )
    fast = np.power(E, alpha) * np.exp(-E / TF)

    return a1 * thermal + a2 * epithermal + a3 * fast


def _model_shape(theta: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Evaluate the model shape for the raw parameter vector ``theta``.

    ``theta`` is ``[a1, a2, a3, b, beta, alpha, TF]`` optionally followed
    by ``log10(phi_scale)``.  The returned spectrum is already multiplied
    by ``phi_scale`` (1.0 when the scale is not fitted).
    """
    if len(theta) > 7:
        phi_scale = 10.0 ** theta[7]
    else:
        phi_scale = 1.0
    return phi_scale * fission_model(E, *theta[:7])


def _theta_bounds(fit_scale: bool) -> list[tuple[float, float]]:
    """Bounds of the flat parameter vector used by GA / LM stages."""
    bounds = [FISSION_PARAM_BOUNDS[name] for name in FISSION_PARAM_NAMES]
    if fit_scale:
        bounds.append((-12.0, 12.0))
    return bounds


def _default_theta(fit_scale: bool) -> np.ndarray:
    """Default (mid-range) parameter vector."""
    theta = np.array([0.3, 0.3, 0.4, 0.0, 0.1, 0.5, 1.5], dtype=float)
    if fit_scale:
        theta = np.append(theta, 0.0)  # phi_scale = 1
    return theta


def _theta_from_initial(
    initial_params: dict[str, float] | None, fit_scale: bool
) -> np.ndarray | None:
    """Convert a user parameter dict into the flat vector (or None)."""
    if not initial_params:
        return None
    theta = _default_theta(fit_scale)
    for i, name in enumerate(FISSION_PARAM_NAMES):
        if name in initial_params:
            theta[i] = float(initial_params[name])
    if fit_scale and "phi_scale" in initial_params:
        scale = max(float(initial_params["phi_scale"]), 1e-30)
        theta[7] = np.log10(scale)
    lo = np.array([b[0] for b in _theta_bounds(fit_scale)])
    hi = np.array([b[1] for b in _theta_bounds(fit_scale)])
    return np.clip(theta, lo, hi)


# ------------------------------------------------------------------ #
#  Core two-stage solver
# ------------------------------------------------------------------ #


def _folded_readings(
    theta: np.ndarray,
    A: np.ndarray,
    b_vec: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
) -> np.ndarray:
    """Fold the model spectrum with the response matrix.

    The model returns fluence per unit lethargy; multiplying by the
    logarithmic bin widths converts it to per-bin fluences, which the
    response matrix maps to computed detector readings.
    """
    spectrum = _model_shape(theta, E)
    return A @ (spectrum * ln_steps)


def _residual_vector(theta, A, b_vec, E, ln_steps):
    """Residual vector for the nonlinear least-squares stage."""
    return _folded_readings(theta, A, b_vec, E, ln_steps) - b_vec


def _target(theta, A, b_vec, E, ln_steps):
    """Target function of the stochastic stage (article eq. 4.32)."""
    resid = _folded_readings(theta, A, b_vec, E, ln_steps) - b_vec
    return float(np.sum(np.abs(resid)))


def _run_lm(theta0, A, b_vec, E, ln_steps, fit_scale, lm_method, lm_max_nfev):
    """Stage 2: nonlinear least-squares refinement of a starting point."""
    lo = np.array([b[0] for b in _theta_bounds(fit_scale)])
    hi = np.array([b[1] for b in _theta_bounds(fit_scale)])
    theta0 = np.clip(np.asarray(theta0, dtype=float), lo, hi)

    if lm_method == "lm":
        # Levenberg-Marquardt (as SciLab leastsq) does not support
        # bounds; project the start inside and clip after the fit.
        theta0 = np.clip(theta0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        result = least_squares(
            _residual_vector,
            theta0,
            args=(A, b_vec, E, ln_steps),
            method="lm",
            max_nfev=lm_max_nfev,
        )
        theta = np.clip(result.x, lo, hi)
    else:
        result = least_squares(
            _residual_vector,
            theta0,
            bounds=(lo, hi),
            args=(A, b_vec, E, ln_steps),
            method="trf",
            x_scale="jac",
            max_nfev=lm_max_nfev,
        )
        theta = result.x

    resid = _residual_vector(theta, A, b_vec, E, ln_steps)
    cost = float(np.linalg.norm(resid))
    return theta, cost, int(result.nfev), bool(result.success), result.message


def solve_fission_ga(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    initial_params: dict[str, float] | None = None,
    fit_scale: bool = True,
    ga_popsize: int = 15,
    ga_maxiter: int = 100,
    ga_tol: float = 1e-10,
    lm_method: str = "trf",
    lm_max_nfev: int = 2000,
    random_state: int | None = None,
) -> tuple[np.ndarray, bool, str, int, dict[str, float]]:
    """Two-stage stochastic unfolding with the Fission model.

    Stage 1 searches the parameter hypercube globally with a
    differential-evolution (genetic) algorithm minimizing the L1
    discrepancy of the folded readings (article eq. 4.32).  Stage 2
    refines the best point with a bounded nonlinear least-squares
    routine.  A user-provided ``initial_params`` point is refined as an
    additional stage-2 start, and the better of the two fits is kept.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Natural-logarithmic bin widths (d ln E).
    initial_params : dict, optional
        Optional starting parameter values (keys of
        ``FISSION_PARAM_NAMES``, plus ``phi_scale``); refined as an
        extra stage-2 start.
    fit_scale : bool, optional
        If True (default), an additional free overall scale factor
        ``phi_scale`` is fitted (log10-parameterized), so that the
        model matches absolutely calibrated readings.  If False, the
        exact 7-parameter normalized formulation of the article is
        used.
    ga_popsize : int, optional
        Population multiplier of the genetic algorithm (default: 15).
    ga_maxiter : int, optional
        Maximum number of generations of the genetic algorithm
        (default: 100).
    ga_tol : float, optional
        Convergence tolerance of the genetic algorithm (default: 1e-10).
    lm_method : str, optional
        Stage-2 least-squares method: ``"trf"`` (bounded, default) or
        ``"lm"`` (Levenberg-Marquardt, as in the article's SciLab
        ``leastsq``; parameters are clipped to bounds).
    lm_max_nfev : int, optional
        Maximum function evaluations of one stage-2 run (default: 2000).
    random_state : int, optional
        Seed of the genetic algorithm for reproducibility.

    Returns
    -------
    tuple
        ``(spectrum, success, message, nfev, params)`` where ``spectrum``
        is the unfolded per-bin fluence spectrum (model evaluated per
        unit lethargy, scaled and multiplied by the log-widths of the
        energy bins, matching the convention of the other parametric
        methods), and ``params`` is a dict of the fitted model
        parameters (including ``phi_scale`` and normalized
        ``weight_fractions``).
    """
    A_matrix = np.asarray(A_matrix, dtype=float)
    b_readings = np.asarray(b_readings, dtype=float).ravel()
    E = np.asarray(E, dtype=float)

    bounds = _theta_bounds(fit_scale)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    total_nfev = 0

    # ---- Stage 1: stochastic (genetic) global search ----------------
    ga_result = differential_evolution(
        _target,
        bounds=bounds,
        args=(A_matrix, b_readings, E, log_steps),
        popsize=ga_popsize,
        maxiter=ga_maxiter,
        tol=ga_tol,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=random_state,
        polish=False,
        init="latinhypercube",
        workers=1,
    )
    total_nfev += int(ga_result.nfev)

    candidates = [np.clip(ga_result.x, lo, hi)]

    # Optional extra stage-2 start from user-provided parameters.
    theta_user = _theta_from_initial(initial_params, fit_scale)
    if theta_user is not None:
        candidates.append(theta_user)

    # ---- Stage 2: nonlinear least-squares refinement ----------------
    best_theta, best_cost, best_success, best_message = None, np.inf, False, ""
    for theta0 in candidates:
        theta, cost, nfev, success, message = _run_lm(
            theta0,
            A_matrix,
            b_readings,
            E,
            log_steps,
            fit_scale,
            lm_method,
            lm_max_nfev,
        )
        total_nfev += nfev
        if cost < best_cost:
            best_theta, best_cost = theta, cost
            best_success, best_message = success, message

    spectrum = _model_shape(best_theta, E) * log_steps

    params: dict[str, float] = {
        name: float(best_theta[i]) for i, name in enumerate(FISSION_PARAM_NAMES)
    }
    if fit_scale:
        params["phi_scale"] = float(10.0 ** best_theta[7])
    weight_sum = sum(best_theta[:3])
    if weight_sum > 0:
        params["weight_fractions"] = {
            name: float(best_theta[i] / weight_sum)
            for i, name in enumerate(("a1", "a2", "a3"))
        }
    else:
        params["weight_fractions"] = {"a1": 0.0, "a2": 0.0, "a3": 0.0}
    params["cost"] = best_cost

    return spectrum, best_success, best_message, total_nfev, params


# ------------------------------------------------------------------ #
#  Article's validation criteria
# ------------------------------------------------------------------ #


def _validate_fit(
    computed: np.ndarray,
    measured: np.ndarray,
    spectrum_bins: np.ndarray,
    eps_threshold: float = 0.05,
    norm_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate the article's validation criteria for a fitted solution.

    Implements the checks of the article section "Validatsiya rascheta":

    * per-sphere relative uncertainties ``eps_m = (Cm - Cb_m) / Cm``
      must stay within a few tenths of a percent and alternate their
      signs in random order;
    * the norm of the (normalized) model spectrum must lie close to
      one (checked only when a norm range is supplied).

    Returns a dictionary with the metrics and boolean flags.
    """
    measured = np.asarray(measured, dtype=float)
    computed = np.asarray(computed, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        eps = (computed - measured) / np.where(measured != 0, measured, np.nan)

    finite_eps = eps[np.isfinite(eps)]
    max_eps = float(np.max(np.abs(finite_eps))) if finite_eps.size else np.inf
    fom = float(100.0 * np.sqrt(np.mean(finite_eps**2))) if finite_eps.size else np.inf

    signs = np.sign(finite_eps)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size else 0
    signs_mixed = bool(np.any(signs > 0) and np.any(signs < 0))

    spectrum_norm = float(np.sum(spectrum_bins))

    residuals_ok = bool(np.isfinite(max_eps) and max_eps <= eps_threshold)
    norm_ok: bool | None = None
    if norm_range is not None:
        norm_ok = bool(norm_range[0] <= spectrum_norm <= norm_range[1])

    passed = residuals_ok if norm_ok is None else bool(residuals_ok and norm_ok)

    return {
        "fom_percent": fom,
        "max_relative_uncertainty": max_eps,
        "relative_uncertainties": [float(e) if np.isfinite(e) else None for e in eps],
        "residual_sign_changes": sign_changes,
        "signs_mixed": signs_mixed,
        "spectrum_norm": spectrum_norm,
        "residuals_ok": residuals_ok,
        "norm_ok": norm_ok,
        "eps_threshold": float(eps_threshold),
        "passed": passed,
    }


# ------------------------------------------------------------------ #
#  Detector-level workflow
# ------------------------------------------------------------------ #


def unfold_fission_ga(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    initial_params: dict[str, float] | None = None,
    fit_scale: bool = True,
    ga_popsize: int = 15,
    ga_maxiter: int = 100,
    ga_tol: float = 1e-10,
    lm_method: str = "trf",
    lm_max_nfev: int = 2000,
    eps_threshold: float = 0.05,
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
    """Unfold a neutron spectrum with the Fission-model GA+LM algorithm.

    Port of the article's ``BonnerFinder()``: a stochastic (genetic)
    global search of the seven Fission-model parameters followed by a
    nonlinear least-squares refinement, with the article's validation
    criteria attached to the result.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess (unused by the parametric method).
    initial_params : Optional[Dict[str, float]], optional
        Optional starting parameter values; refined as an extra
        stage-2 start.  Keys: ``a1, a2, a3, b, beta, alpha, TF`` and
        optionally ``phi_scale``.
    fit_scale : bool, optional
        Fit a free overall scale factor (default: True).
    ga_popsize, ga_maxiter, ga_tol :
        Genetic-algorithm controls (see :func:`solve_fission_ga`).
    lm_method : str, optional
        Stage-2 least-squares method (default: ``"trf"``).
    lm_max_nfev : int, optional
        Maximum function evaluations of one stage-2 run.
    eps_threshold : float, optional
        Threshold on the per-sphere relative uncertainty used by the
        validation criteria (default: 0.05).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary with additional keys
        ``model_params`` and ``validation``.
    """
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    log_steps = compute_log_steps(E_MeV, n_energy_bins)
    ln_steps = log_steps * np.log(10)

    holder: dict[str, Any] = {}

    def solve_wrapper(A_mat, b_vec, **kwargs):
        spectrum, success, message, nfev, params = solve_fission_ga(
            A_mat,
            b_vec,
            E_MeV,
            ln_steps,
            initial_params=initial_params,
            fit_scale=fit_scale,
            ga_popsize=ga_popsize,
            ga_maxiter=ga_maxiter,
            ga_tol=ga_tol,
            lm_method=lm_method,
            lm_max_nfev=lm_max_nfev,
            random_state=random_state,
        )
        # Record the clean-fit artifacts only (the Monte-Carlo
        # replicates receive perturbed readings and would overwrite
        # them otherwise).  ``spectrum`` is already in per-bin fluence
        # units, so the folding matches run_unfolding's convention.
        if np.array_equal(np.asarray(b_vec), b):
            computed = A_mat @ spectrum
            norm_range = (0.6, 1.2) if not fit_scale else None
            holder["model_params"] = params
            holder["validation"] = _validate_fit(
                computed,
                b_vec,
                spectrum,
                eps_threshold=eps_threshold,
                norm_range=norm_range,
            )
        return spectrum, nfev, success

    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

    result = run_unfolding(
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
        method_name="fission_ga",
        extra_output={
            "initial_params": initial_params,
            "fit_scale": fit_scale,
            "ga_popsize": ga_popsize,
            "ga_maxiter": ga_maxiter,
            "lm_method": lm_method,
            "T0": T0_THERMAL,
            "Ed": ED_EPITHERMAL,
        },
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

    if "model_params" in holder:
        result["model_params"] = holder["model_params"]
    if "validation" in holder:
        result["validation"] = holder["validation"]

    return result
