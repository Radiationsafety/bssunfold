"""Non-extensive (Tsallis) parametric unfolding.

This module ports the neutron-spectrum reconstruction method of:

    Y. ZhangSun, Z. Li, C. Su, D. Li, S. Guo, Z. Wu, X. Zhang,
    "Determination of Neutron Spectrum for Nuclear Reactor Based on
    Non-extensive Statistical Theory", Applied Radiation and Isotopes
    (2026), DOI 10.1016/j.apradiso.2026.113015.

Motivation and model
--------------------
A reactor neutron field is an open system exchanging particles and
energy with the surrounding medium and does not obey the ergodicity
hypothesis of Boltzmann-Gibbs statistics.  The authors therefore describe
it with the **Tsallis non-extensive entropy** (paper eq. (4))::

    S_q = k * (1 - sum_i p_i**q) / (q - 1),

which reduces to the Boltzmann-Gibbs entropy as ``q -> 1``.  Maximizing
``S_q`` under the normalization, particle-number and average-energy
constraints (paper eq. (5)) yields a *q-exponential* flux density
(paper eq. (6)); multiplying by an absorption influence factor ``E**-n``
gives eq. (7), and the reactor spectrum is represented as a superposition
of three such components (thermal / intermediate / fast, paper eq. (8))::

    phi_k(E) = C_k * exp_q(-u_k(E)) * E**(-n_k),
    u_k(E)   = beta_k * E**(-1/2) + g_k * E,          (E in eV)

with the q-exponential ``exp_q(x) = [1 + (1 - q) x]**(1 / (1 - q))``.
Each component carries five parameters ``(C_k, beta_k, g_k, q_k, n_k)``
and is active on its own energy window, giving the fifteen free
parameters of the paper:

* thermal       : ``1e-4 eV .. 0.5 eV``
* intermediate  : ``0.5 eV .. 1e5 eV``
* fast          : ``1e5 eV .. 2e7 eV``

The component nodes ``0.5 eV`` and ``1e5 eV`` are the paper's ``E1`` and
``E2`` and are *fixed*; the fifteen free parameters are the three
``(C_k, beta_k, g_k, q_k, n_k)`` quintuples.

Fitting procedure (paper section 4)
-----------------------------------
The flux density is folded with the multigroup response to form the
activation rates ``A phi`` (paper eqs. (1)-(2)).  The parameters are
determined by minimizing the sum of squared **relative** deviations of
the activation rates (paper eq. (9))::

    R(theta) = sum_i [ (a_i^cal(theta) - a_i^exp) / a_i^exp ]**2.

The paper solves this nonlinear least-squares problem with the
initial-value-independent global optimizer of the 1stOpt software.  This
port uses the same two-stage strategy as the sibling
:mod:`bssunfold.core.unfold_fission_ga` module: a stochastic global
search (SciPy ``differential_evolution``) followed by a bounded
nonlinear least-squares refinement (``scipy.optimize.least_squares``).
``search="local"`` skips the global stage for speed when a good initial
guess is available (the paper notes an accurate starting spectrum can be
obtained by fitting eq. (8) to a simulated spectrum).

Adaptation to the package forward model
---------------------------------------
The package convention is ``b = A @ phi`` with ``A`` the Bonner-sphere
response matrix that already contains the logarithmic bin widths
``d(ln E)`` (so ``phi`` is the fluence density per unit lethargy and
``A @ phi`` reproduces the readings exactly -- the paper's group-
integrated ``sigma_ij``).  This module therefore folds the model spectrum
directly as ``A @ phi`` without re-applying ``d(ln E)``; ``log_steps`` is
accepted only for signature compatibility with the shared unfolding
engine.

Numerical notes
---------------
* ``q -> 1`` is handled by a dedicated branch (plain exponential) to
  avoid the removable ``0/0`` singularity.
* The q-exponential base ``1 + (1 - q) x`` is clipped to a tiny positive
  value, so the compact-support ``q < 1`` branch yields a zero tail
  instead of ``nan``.
* Amplitudes are parameterized on a ``log10(C)`` scale so the global
  search operates on a well-conditioned simplex.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from ._base_unfolder import _build_system, run_unfolding
from ._parametric_shared import _check_fit_quality

__all__ = [
    "NONEXTENSIVE_PARAM_NAMES",
    "NONEXTENSIVE_PARAM_BOUNDS",
    "E_THERMAL_MAX_EV",
    "E_FAST_MIN_EV",
    "nonextensive_model",
    "fit_model_to_spectrum",
    "solve_nonextensive",
    "unfold_nonextensive",
]

# ------------------------------------------------------------------ #
#  Constants and parameter layout
# ------------------------------------------------------------------ #

#: Thermal/intermediate component node (paper ``E1``), in eV.
E_THERMAL_MAX_EV = 0.5
#: Intermediate/fast component node (paper ``E2``), in eV.
E_FAST_MIN_EV = 1.0e5

_EV_PER_MEV = 1.0e6

#: Component labels (thermal, intermediate, fast).
_COMPONENTS = ("th", "epi", "f")
#: Per-component parameter kinds, in the paper's order.
_PARAM_KINDS = ("log10C", "beta", "g", "q", "n")

#: Canonical flat-vector order of the fifteen parameters.
NONEXTENSIVE_PARAM_NAMES = tuple(
    f"{kind}_{comp}" for comp in _COMPONENTS for kind in _PARAM_KINDS
)

#: Row-wise bounds of the flat parameter vector (same order as
#: :data:`NONEXTENSIVE_PARAM_NAMES`).  ``log10C`` is the base-10 logarithm
#: of the component amplitude; ``q`` is the non-extensivity index.
NONEXTENSIVE_PARAM_BOUNDS = {
    f"{kind}_{comp}": {
        "log10C": (-20.0, 20.0),
        "beta": (-50.0, 50.0),
        "g": (-50.0, 50.0),
        "q": (0.05, 5.0),
        "n": (-5.0, 5.0),
    }[kind]
    for comp in _COMPONENTS
    for kind in _PARAM_KINDS
}

#: Maximum magnitude of a single model bin (overflow guard).
_PHI_MAX = 1.0e30
#: Magnitude at which a single residual is clipped.  The global search
#: probes the whole parameter box where the q-exponential can transiently
#: overflow; a moderate clip keeps the penalty finite and, importantly,
#: keeps SciPy's internal population-variance computation from
#: overflowing on astronomical trial costs.
_RES_CLIP = 1.0e6
#: Tolerance below which ``q`` is treated as the Boltzmann-Gibbs limit 1.
_Q_TOL = 1.0e-8


# ------------------------------------------------------------------ #
#  Model
# ------------------------------------------------------------------ #


def _q_exponential(x: np.ndarray, q: float) -> np.ndarray:
    """Evaluate the Tsallis q-exponential ``exp_q(x)``.

    ``exp_q(x) = [1 + (1 - q) x]**(1 / (1 - q))`` for ``q != 1``, with
    the Boltzmann-Gibbs limit ``exp(x)`` recovered as ``q -> 1``.  The
    base is clipped to a tiny positive value so the compact-support
    ``q < 1`` branch decays to zero instead of producing ``nan``.

    Parameters
    ----------
    x : np.ndarray
        Argument of the q-exponential.
    q : float
        Non-extensivity index.

    Returns
    -------
    np.ndarray
        ``exp_q(x)`` element-wise.
    """
    x = np.asarray(x, dtype=float)
    if abs(q - 1.0) < _Q_TOL:
        return np.exp(np.clip(x, -700.0, 700.0))
    base = np.maximum(1.0 + (1.0 - q) * x, 1.0e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.power(base, 1.0 / (1.0 - q))
    return np.nan_to_num(out, nan=0.0, posinf=_PHI_MAX, neginf=0.0)


def _component(
    E_eV: np.ndarray,
    log10C: float,
    beta: float,
    g: float,
    q: float,
    n: float,
) -> np.ndarray:
    """Single Tsallis component ``C * exp_q(-(beta E^-1/2 + g E)) * E^-n``.

    ``E`` is in eV (matching the paper's energy units).
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = beta * np.power(E_eV, -0.5) + g * E_eV
        phi = (10.0**log10C) * _q_exponential(-u, q) * np.power(E_eV, -n)
    return np.nan_to_num(phi, nan=0.0, posinf=_PHI_MAX, neginf=0.0)


def nonextensive_model(E_MeV: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the three-component non-extensive (Tsallis) spectrum.

    Implements paper eq. (8): each of the thermal, intermediate and fast
    components is a q-exponential (eq. (6)) times an absorption power law
    (eq. (7)), active on its own energy window.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    theta : np.ndarray
        Flat parameter vector of length 15 in the
        :data:`NONEXTENSIVE_PARAM_NAMES` order
        (``[log10C, beta, g, q, n]`` per component, thermal first).

    Returns
    -------
    np.ndarray
        Fluence density per unit lethargy on ``E_MeV`` (non-negative).

    Raises
    ------
    ValueError
        If ``theta`` does not have the expected length.
    """
    E = np.asarray(E_MeV, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    n_params = len(NONEXTENSIVE_PARAM_NAMES)
    if theta.size != n_params:
        raise ValueError(
            f"theta must have {n_params} entries "
            f"({len(_COMPONENTS)} components x {len(_PARAM_KINDS)}), "
            f"got {theta.size}"
        )

    E_eV = E * _EV_PER_MEV
    phi = np.zeros_like(E_eV)

    masks = {
        "th": E_eV < E_THERMAL_MAX_EV,
        "epi": (E_eV >= E_THERMAL_MAX_EV) & (E_eV < E_FAST_MIN_EV),
        "f": E_eV >= E_FAST_MIN_EV,
    }

    n_kinds = len(_PARAM_KINDS)
    for ci, comp in enumerate(_COMPONENTS):
        vals = theta[ci * n_kinds : (ci + 1) * n_kinds]
        mask = masks[comp]
        if np.any(mask):
            phi[mask] = _component(E_eV[mask], *vals)

    return np.maximum(np.nan_to_num(phi, nan=0.0, posinf=_PHI_MAX), 0.0)


# ------------------------------------------------------------------ #
#  Bounds / initial vector helpers
# ------------------------------------------------------------------ #


def _theta_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return the lower/upper bound vectors of the flat parameter vector."""
    lo = np.array([NONEXTENSIVE_PARAM_BOUNDS[name][0]
                   for name in NONEXTENSIVE_PARAM_NAMES])
    hi = np.array([NONEXTENSIVE_PARAM_BOUNDS[name][1]
                   for name in NONEXTENSIVE_PARAM_NAMES])
    return lo, hi


def _default_theta() -> np.ndarray:
    """Neutral starting vector (flat unit-amplitude components, ``q ~ 1``)."""
    defaults = {"log10C": 0.0, "beta": 0.0, "g": 0.0, "q": 1.1, "n": 0.0}
    return np.array(
        [defaults[kind] for _comp in _COMPONENTS for kind in _PARAM_KINDS],
        dtype=float,
    )


def _theta_from_initial(initial_params: dict[str, float] | None) -> np.ndarray:
    """Build a bound-clipped flat vector from a user parameter dict."""
    theta = _default_theta()
    if initial_params:
        for i, name in enumerate(NONEXTENSIVE_PARAM_NAMES):
            if name in initial_params:
                theta[i] = float(initial_params[name])
    lo, hi = _theta_bounds()
    return np.clip(theta, lo, hi)


def fit_model_to_spectrum(
    E_MeV: np.ndarray,
    phi: np.ndarray,
    n_starts: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Curve-fit eq. (8) to a known or prior spectrum.

    Reproduces the verification of the prior information in section 2.2 of
    the paper: the fifteen parameters are obtained by fitting the
    parameterized distribution to a spectrum.  In the unfolding workflow
    this turns an a-priori (e.g. simulated) spectrum into a starting
    parameter vector, as suggested in section 4 of the paper.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    phi : np.ndarray
        Target spectrum (fluence density per unit lethargy) on ``E_MeV``.
    n_starts : int, optional
        Number of multi-start runs (default: 8).
    seed : int, optional
        Random seed for the multi-start sampling.

    Returns
    -------
    np.ndarray
        Fitted fifteen-parameter vector in the
        :data:`NONEXTENSIVE_PARAM_NAMES` order.
    """
    E = np.asarray(E_MeV, dtype=float)
    phi = np.asarray(phi, dtype=float).ravel()
    target = np.log10(np.maximum(phi, 1.0e-30))
    lo, hi = _theta_bounds()

    def residual(theta):
        model = np.maximum(nonextensive_model(E, theta), 1.0e-30)
        return np.log10(model) - target

    starts = [_theta_from_initial(None)]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, n_starts - 1)):
        starts.append(np.clip(rng.uniform(lo, hi), lo, hi))

    best_theta = starts[0]
    best_cost = np.inf
    for theta0 in starts:
        try:
            res = least_squares(
                residual, theta0, bounds=(lo, hi), method="trf", max_nfev=4000
            )
        except ValueError:
            continue
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best_theta = res.x
    return np.clip(best_theta, lo, hi)


def _theta_to_params(theta: np.ndarray, A: np.ndarray, b: np.ndarray,
                     E: np.ndarray, relative: bool) -> dict[str, Any]:
    """Package a fitted vector into a readable parameter dictionary."""
    params: dict[str, Any] = {
        name: float(val) for name, val in zip(NONEXTENSIVE_PARAM_NAMES, theta)
    }
    for comp in _COMPONENTS:
        params[f"C_{comp}"] = float(10.0 ** params[f"log10C_{comp}"])

    computed = A @ nonextensive_model(E, theta)
    b_safe = np.where(np.abs(b) > 0, b, np.nan)
    rel_dev = (computed - b) / b_safe
    finite = np.isfinite(rel_dev)
    params["objective"] = float(np.sum(rel_dev[finite] ** 2))
    params["max_relative_deviation"] = (
        float(np.max(np.abs(rel_dev[finite]))) if np.any(finite) else float("inf")
    )
    params["relative_deviations"] = [
        float(v) if np.isfinite(v) else None for v in rel_dev
    ]
    return params


# ------------------------------------------------------------------ #
#  Objective / local refinement
# ------------------------------------------------------------------ #


def _residual_vector(theta, A, b, E, relative):
    """Residual vector of the fit (relative weights when requested).

    The result is clipped to a finite range: the global search probes the
    whole parameter box, where the q-exponential can transiently overflow;
    the clipped value acts as a large finite penalty for both the scalar
    target and the least-squares stage.
    """
    computed = A @ nonextensive_model(E, theta)
    with np.errstate(over="ignore", invalid="ignore"):
        if relative:
            w = 1.0 / np.maximum(np.abs(b), np.finfo(float).tiny)
            res = (computed - b) * w
        else:
            res = computed - b
    return np.clip(np.nan_to_num(res, nan=0.0), -_RES_CLIP, _RES_CLIP)


def _target(theta, A, b, E, relative):
    """Scalar target of the global stage (paper eq. (9))."""
    res = _residual_vector(theta, A, b, E, relative)
    return float(np.sum(res * res))


def _run_lm(theta0, A, b, E, relative, lm_max_nfev):
    """Bounded nonlinear least-squares refinement of a starting vector."""
    lo, hi = _theta_bounds()
    theta0 = np.clip(np.asarray(theta0, dtype=float), lo, hi)
    result = least_squares(
        _residual_vector,
        theta0,
        bounds=(lo, hi),
        args=(A, b, E, relative),
        method="trf",
        x_scale="jac",
        max_nfev=lm_max_nfev,
    )
    theta = np.clip(result.x, lo, hi)
    cost = _target(theta, A, b, E, relative)
    return theta, cost, int(result.nfev), bool(result.success), result.message


# ------------------------------------------------------------------ #
#  Core solver
# ------------------------------------------------------------------ #


def solve_nonextensive(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray | None = None,
    initial_params: dict[str, float] | None = None,
    initial_spectrum: np.ndarray | None = None,
    search: str = "global",
    ga_popsize: int = 15,
    ga_maxiter: int = 100,
    ga_tol: float = 1e-10,
    ga_polish: bool = True,
    lm_max_nfev: int = 2000,
    relative: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, bool, str, int, dict[str, Any]]:
    """Fit the non-extensive (Tsallis) model to the measured readings.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix ``(m, n)`` (already includes the logarithmic bin
        widths, see the module docstring).
    b_readings : np.ndarray
        Measured readings ``(m,)`` (activation rates).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray, optional
        Accepted for signature compatibility with the shared unfolding
        engine; unused (the response matrix already carries the bin
        widths).
    initial_params : dict, optional
        Starting values for any of the :data:`NONEXTENSIVE_PARAM_NAMES`
        (others keep their defaults).  Used directly by
        ``search="local"`` and as an extra stage-2 start for
        ``search="global"``.
    initial_spectrum : np.ndarray, optional
        A-priori spectrum (fluence density per unit lethargy).  When
        given (and ``initial_params`` is not), its parameters are
        obtained by curve-fitting eq. (8) to it via
        :func:`fit_model_to_spectrum`, providing a physically informed
        starting vector (paper section 4).
    search : {"global", "local"}, optional
        ``"global"`` (default) runs a stochastic global search followed
        by a least-squares refinement; ``"local"`` runs only the
        least-squares stage from the starting vector.
    ga_popsize : int, optional
        Differential-evolution population multiplier (default: 15).
    ga_maxiter : int, optional
        Maximum differential-evolution generations (default: 100).
    ga_tol : float, optional
        Differential-evolution convergence tolerance (default: 1e-10).
    ga_polish : bool, optional
        Polish the differential-evolution result with ``L-BFGS-B``
        (default: True).
    lm_max_nfev : int, optional
        Maximum function evaluations of the least-squares refinement
        (default: 2000).
    relative : bool, optional
        Weight the fit by the relative deviation ``1 / a_i^exp`` (paper
        eq. (9), default: True).  Set False for an unweighted fit.
    seed : int, optional
        Random seed of the differential-evolution stage.

    Returns
    -------
    Tuple[np.ndarray, bool, str, int, dict]
        ``(spectrum, success, message, nfev, params)`` where ``spectrum``
        is the fitted fluence density per unit lethargy and ``params``
        carries the fifteen parameters, the raw amplitudes, the objective
        and the per-reading relative deviations.

    Raises
    ------
    ValueError
        If ``search`` is not ``"global"`` or ``"local"``.
    """
    if search not in ("global", "local"):
        raise ValueError(
            f"search must be 'global' or 'local', got {search!r}"
        )

    A = np.asarray(A_matrix, dtype=float)
    b = np.asarray(b_readings, dtype=float).ravel()
    E = np.asarray(E, dtype=float)

    lo, hi = _theta_bounds()
    theta0 = _theta_from_initial(initial_params)

    # Physically informed start from an a-priori spectrum (paper section 4):
    # fit eq. (8) to the prior and refine from there.
    theta_prior = None
    if initial_params is None and initial_spectrum is not None:
        theta_prior = fit_model_to_spectrum(E, np.asarray(initial_spectrum).ravel())

    nfev = 0
    success = False
    message = ""

    if search == "global":
        de_result = differential_evolution(
            _target,
            bounds=list(zip(lo, hi, strict=True)),
            args=(A, b, E, relative),
            popsize=ga_popsize,
            maxiter=ga_maxiter,
            tol=ga_tol,
            polish=ga_polish,
            seed=seed,
            init="latinhypercube",
        )
        theta_de = np.clip(de_result.x, lo, hi)
        nfev += int(de_result.nfev)

        starts = [theta_de, theta0]
        if theta_prior is not None:
            starts.append(theta_prior)
        best = None
        for theta_start in starts:
            theta_c, cost, n_ev, ok, msg = _run_lm(
                theta_start, A, b, E, relative, lm_max_nfev
            )
            nfev += n_ev
            if best is None or cost < best[1]:
                best = (theta_c, cost, ok, msg)
        theta, cost, success, message = best
    else:
        theta_start = theta_prior if theta_prior is not None else theta0
        theta, cost, nfev, success, message = _run_lm(
            theta_start, A, b, E, relative, lm_max_nfev
        )

    spectrum = nonextensive_model(E, theta)

    residual = _residual_vector(theta, A, b, E, False)
    _check_fit_quality(
        float(np.linalg.norm(residual)),
        b,
        "nonextensive",
        model_descr="3-component Tsallis (non-extensive) model",
    )

    params = _theta_to_params(theta, A, b, E, relative)
    params["cost"] = float(cost)

    if not message:
        message = "converged" if success else "max evaluations reached"

    return spectrum, bool(success), str(message), int(nfev), params


# ------------------------------------------------------------------ #
#  Detector-level workflow
# ------------------------------------------------------------------ #


def unfold_nonextensive(
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
    search: str = "global",
    ga_popsize: int = 15,
    ga_maxiter: int = 100,
    ga_tol: float = 1e-10,
    ga_polish: bool = True,
    lm_max_nfev: int = 2000,
    relative: bool = True,
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
    """Unfold a neutron spectrum with the non-extensive (Tsallis) method.

    Port of the parameterized prior of ZhangSun et al. (2026): the
    spectrum is a three-component q-exponential (paper eq. (8)) whose
    fifteen parameters are fitted to the measured readings by minimizing
    the sum of squared relative deviations (paper eq. (9)).

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
        Optional starting parameter values (keys of
        :data:`NONEXTENSIVE_PARAM_NAMES`).
    search : {"global", "local"}, optional
        Optimization strategy (see :func:`solve_nonextensive`).
    ga_popsize, ga_maxiter, ga_tol, ga_polish :
        Global-search controls (see :func:`solve_nonextensive`).
    lm_max_nfev : int, optional
        Maximum function evaluations of the least-squares refinement.
    relative : bool, optional
        Relative-deviation weighting of the objective (default: True).
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
        ``model_params`` (fitted parameters, objective, per-reading
        relative deviations) and ``energy_nodes_eV``.
    """
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    holder: dict[str, Any] = {}

    def solve_wrapper(A_mat, b_vec, **kwargs):
        spectrum, success, message, nfev, params = solve_nonextensive(
            A_mat,
            b_vec,
            E_MeV,
            initial_params=initial_params,
            initial_spectrum=initial_spectrum,
            search=search,
            ga_popsize=ga_popsize,
            ga_maxiter=ga_maxiter,
            ga_tol=ga_tol,
            ga_polish=ga_polish,
            lm_max_nfev=lm_max_nfev,
            relative=relative,
            seed=random_state,
        )
        # Record artifacts of the clean fit only (Monte-Carlo replicates
        # receive perturbed readings and would overwrite them otherwise).
        if np.array_equal(np.asarray(b_vec), b):
            holder["model_params"] = params
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
        method_name="Tsallis non-extensive",
        extra_output={
            "energy_nodes_eV": (E_THERMAL_MAX_EV, E_FAST_MIN_EV),
            "search": search,
            "relative": relative,
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

    # Attach the fitted parameters (run_unfolding builds ``output`` after
    # the solve; the holder is filled during the clean forward call).
    if "model_params" in holder:
        result["model_params"] = holder["model_params"]

    return result
