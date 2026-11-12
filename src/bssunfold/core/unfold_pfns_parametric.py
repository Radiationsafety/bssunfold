"""Parametric PFNS (prompt-fission neutron spectrum) unfolding.

Port of the parametric spectral families used in:

    N. Jovančević, S. Ilić, D. Knežević, D. Maletić, C. Stieghorst,
    S. Oberstedt, M. Hult, "Artificial neural networks for unfolding of
    neutron spectra from activation measurements", Radiation Physics and
    Chemistry (2026), DOI 10.1016/j.radphyschem.2026.114567.

The ANN paper trains on synthetic spectra drawn from four analytical
PFNS families (Maxwellian, Watt, Koçak, Mannhart-based) with the
parameter ranges of its Table II.  This module exposes the same families
as a *parametric unfolding method*: the spectrum is represented by one
chosen family with a small set of shape parameters (plus a free overall
amplitude/scale) and the parameters are fitted to the measured activation
readings by nonlinear least squares.  This is a low-dimensional,
well-posed parametric method analogous to ``unfold_parametric`` (FRUIT)
but using the PFNS shapes of the article.

Families
--------
* ``"maxwellian"`` : ``sqrt(E) * exp(-E / T)``, ``T in [1.30, 1.55]``.
* ``"watt"``       : ``exp(-E / a) * sinh(sqrt(b * E))``,
  ``a in [0.975, 1.075]``, ``b in [2.80, 3.05]``.
* ``"kocak"``      : Eq. (16) of Koçak et al. (2014), Ann. Nucl. Energy
  70, 82, with parameters ``T in [1.00, 1.20]``,
  ``E_f in [0.48, 0.68]``, ``k in [0.32, 0.44]``.
* ``"mannhart"``   : ``[Phi_M(beta * E)] ** gamma`` with
  ``beta in [0.95, 1.05]``, ``gamma in [0.98, 1.02]``.  The reference
  spectrum ``Phi_M`` is the tabulated 252Cf spectrum ``ISO_ref_Cf252`` from
  the IAEA Compendium (shipped in the repository), used as a proxy for the
  Mannhart recommended evaluation of Ref. [31] of the article.

All shapes are defined up to an overall scale, which is fitted as an
additional free parameter (``scale``) so the model can match the absolute
magnitude of the readings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

from ._base_unfolder import _build_system, run_unfolding

logger = logging.getLogger(__name__)

__all__ = [
    "pfns_model_density",
    "solve_pfns_parametric",
    "unfold_pfns_parametric",
    "unfold_pfns_parametric_impl",
]

# ---------------------------------------------------------------------------
# Family metadata
# ---------------------------------------------------------------------------

FAMILY_PARAM_NAMES: dict[str, list[str]] = {
    "maxwellian": ["T", "scale"],
    "watt": ["a", "b", "scale"],
    "kocak": ["T", "Ef", "k", "scale"],
    "mannhart": ["beta", "gamma", "scale"],
}

FAMILY_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "maxwellian": [(1.30, 1.55), (1e-6, 1e6)],
    "watt": [(0.975, 1.075), (2.80, 3.05), (1e-6, 1e6)],
    "kocak": [(1.00, 1.20), (0.48, 0.68), (0.32, 0.44), (1e-6, 1e6)],
    "mannhart": [(0.95, 1.05), (0.98, 1.02), (1e-6, 1e6)],
}

FAMILY_DEFAULTS: dict[str, list[float]] = {
    "maxwellian": [1.42, 1.0],
    "watt": [1.025, 2.926, 1.0],
    "kocak": [1.116, 0.583, 0.38, 1.0],
    "mannhart": [1.0, 1.0, 1.0],
}


def _validate_family(family: str) -> None:
    if family not in FAMILY_PARAM_NAMES:
        raise ValueError(
            f"Unknown family {family!r}. Choose from "
            f"{sorted(FAMILY_PARAM_NAMES)}."
        )


# ---------------------------------------------------------------------------
# Shape functions (density, up to an overall scale)
# ---------------------------------------------------------------------------


def _maxwellian_shape(E: np.ndarray, T: float) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    out = np.zeros_like(E)
    m = E > 0
    out[m] = np.sqrt(E[m]) * np.exp(-E[m] / float(T))
    return out


def _watt_shape(E: np.ndarray, a: float, b: float) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    out = np.zeros_like(E)
    m = E > 0
    a = float(a)
    b = float(b)
    out[m] = np.exp(-E[m] / a) * np.sinh(np.sqrt(b * E[m]))
    return out


def _kocak_shape(E: np.ndarray, T: float, Ef: float, k: float) -> np.ndarray:
    """Koçak Eq. (16) laboratory spectrum (shape only)."""
    E = np.asarray(E, dtype=float)
    T = float(T)
    Ef = float(Ef)
    k = float(k)
    out = np.zeros_like(E)
    m = E > 0
    if not np.any(m):
        return out
    sqE = np.sqrt(E[m])
    sqEf = np.sqrt(Ef)
    # regularized lower incomplete gamma C(x, a) = gammainc(a, x)
    x1 = (sqE + sqEf) ** 2 / T
    x2 = (sqE - sqEf) ** 2 / T
    a_param = k + 0.5
    term = gammainc(a_param, x1) - gammainc(a_param, x2)
    # constants (gamma ratio, 1/(4 sqrt(Ef T))) are E-independent -> scale
    out[m] = term
    # guard against tiny negatives from numerical error
    out = np.where(out < 0, 0.0, out)
    return out


# Tabulated 252Cf reference spectrum (IAEA Compendium, ISO_ref_Cf252) used as
# a proxy for the Mannhart recommended evaluation [Reich, Mannhart & England,
# IAEA, Vienna, 305 (1989)].  Stored on its native energy grid; interpolated
# onto the detector grid at evaluation time.
_MANNHART_E = np.array(
    [1e-09, 2.15e-09, 4.64e-09, 1e-08, 2.15e-08, 4.64e-08, 1e-07,
     2.15e-07, 4.64e-07, 1e-06, 2.15e-06, 4.64e-06, 1e-05, 2.15e-05,
     4.64e-05, 0.0001, 0.000215, 0.000464, 0.001, 0.00215, 0.00464,
     0.01, 0.0125, 0.0158, 0.0199, 0.0251, 0.0316, 0.0398, 0.0501,
     0.063, 0.0794, 0.1, 0.125, 0.158, 0.199, 0.251, 0.316, 0.398,
     0.501, 0.63, 0.794, 1.0, 1.25, 1.58, 1.99, 2.51, 3.16, 3.98, 5.01,
     6.3, 7.94, 10.0, 15.8, 25.1, 39.8, 63.0, 100.0, 158.0, 251.0, 398.0,
     631.0],
    dtype=float,
)
_MANNHART_PHI = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000922, 0.00129, 0.00179, 0.00248,
     0.00339, 0.00464, 0.00638, 0.00881, 0.0121, 0.0169, 0.0237, 0.0326,
     0.0445, 0.0605, 0.0822, 0.109, 0.145, 0.189, 0.241, 0.3, 0.359, 0.414,
     0.452, 0.462, 0.434, 0.366, 0.271, 0.169, 0.0861, 0.0335, 0.00427,
     0.000431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)


def _mannhart_base(E: np.ndarray) -> np.ndarray:
    """Interpolate the tabulated reference spectrum onto ``E``."""
    E = np.asarray(E, dtype=float)
    return np.interp(E, _MANNHART_E, _MANNHART_PHI, left=0.0, right=0.0)


def _mannhart_shape(E: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    base = _mannhart_base(E * float(beta))
    out = np.zeros_like(E)
    m = base > 0
    out[m] = base[m] ** float(gamma)
    return out


def pfns_model_density(E: np.ndarray, family: str, theta: np.ndarray) -> np.ndarray:
    """Return the fluence density for the given family and parameter vector."""
    _validate_family(family)
    E = np.asarray(E, dtype=float)
    names = FAMILY_PARAM_NAMES[family]
    vals = {n: float(v) for n, v in zip(names, theta)}
    scale = vals.pop("scale")
    if family == "maxwellian":
        shape = _maxwellian_shape(E, vals["T"])
    elif family == "watt":
        shape = _watt_shape(E, vals["a"], vals["b"])
    elif family == "kocak":
        shape = _kocak_shape(E, vals["T"], vals["Ef"], vals["k"])
    elif family == "mannhart":
        shape = _mannhart_shape(E, vals["beta"], vals["gamma"])
    else:  # pragma: no cover
        raise ValueError(family)
    return scale * np.asarray(shape, dtype=float)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def _initial_theta(family: str, initial_params: dict[str, float] | None) -> np.ndarray:
    names = FAMILY_PARAM_NAMES[family]
    theta = np.array(FAMILY_DEFAULTS[family], dtype=float).copy()
    if initial_params:
        for i, n in enumerate(names):
            if n in initial_params:
                theta[i] = float(initial_params[n])
    lo, hi = _bounds_arrays(family)
    return np.clip(theta, lo, hi)


def _bounds_arrays(family: str) -> tuple[np.ndarray, np.ndarray]:
    b = FAMILY_BOUNDS[family]
    lo = np.array([x[0] for x in b], dtype=float)
    hi = np.array([x[1] for x in b], dtype=float)
    return lo, hi


def _residual(theta, A, b, E, family, relative) -> np.ndarray:
    model = pfns_model_density(E, family, theta)
    computed = A @ model
    r = computed - b
    if relative:
        denom = np.where(np.abs(b) > 0, b, 1.0)
        r = r / denom
    return r


def solve_pfns_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray | None = None,
    family: str = "maxwellian",
    initial_params: dict[str, float] | None = None,
    search: str = "local",
    relative: bool = True,
    max_nfev: int = 2000,
    seed: int | None = None,
) -> tuple[np.ndarray, bool, str, int, dict[str, Any]]:
    """Fit a PFNS parametric family to the measured readings."""
    _validate_family(family)
    A = np.asarray(A_matrix, dtype=float)
    b = np.asarray(b_readings, dtype=float).ravel()
    E = np.asarray(E, dtype=float)
    if search not in ("local", "global"):
        raise ValueError(f"search must be 'local' or 'global', got {search!r}")

    lo, hi = _bounds_arrays(family)
    theta0 = _initial_theta(family, initial_params)

    nfev = 0
    best = None

    starts = [theta0]
    if search == "global":
        def obj(theta):
            return float(np.sum(_residual(theta, A, b, E, family, relative) ** 2))

        de = differential_evolution(
            obj,
            bounds=list(zip(lo.tolist(), hi.tolist())),
            seed=seed,
            maxiter=200,
            tol=1e-10,
            polish=True,
            init="latinhypercube",
        )
        starts.insert(0, np.clip(de.x, lo, hi))
        nfev += int(de.nfev)

    for ts in starts:
        try:
            res = least_squares(
                _residual,
                ts,
                args=(A, b, E, family, relative),
                bounds=(lo, hi),
                method="trf",
                max_nfev=max_nfev,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("least_squares failed: %s", exc)
            continue
        nfev += int(res.nfev)
        if best is None or res.cost < best.cost:
            best = res

    if best is None:
        # Fallback: return default shape
        theta = theta0
        success = False
        message = "optimization failed"
    else:
        theta = np.clip(best.x, lo, hi)
        success = bool(best.success)
        message = str(best.message) if best.message else "done"

    spectrum = pfns_model_density(E, family, theta)
    names = FAMILY_PARAM_NAMES[family]
    params = {n: float(v) for n, v in zip(names, theta)}
    params["family"] = family
    params["search"] = search
    params["relative"] = bool(relative)

    # fit quality check
    try:
        from ._parametric_shared import _check_fit_quality as _cfq
        resid = A @ spectrum - b
        _cfq(float(np.linalg.norm(resid)), b, "pfns_parametric")
    except Exception as exc:  # pragma: no cover
        logger.debug("fit-quality check failed: %s", exc)

    if not message:
        message = "converged" if success else "max evaluations reached"
    return spectrum, success, message, int(nfev), params


# ---------------------------------------------------------------------------
# Detector-level workflow
# ---------------------------------------------------------------------------


def unfold_pfns_parametric(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    family: str = "maxwellian",
    initial_params: dict[str, float] | None = None,
    search: str = "local",
    relative: bool = True,
    max_nfev: int = 2000,
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
    _validate_family(family)
    A, b, _ = _build_system(readings, detector_names, sensitivities)
    E = np.asarray(E_MeV, dtype=float)

    holder: dict[str, Any] = {}

    def solve_wrapper(A_mat, b_vec, **kwargs):
        spectrum, success, message, nfev, params = solve_pfns_parametric(
            A_mat,
            b_vec,
            E,
            initial_params=initial_params,
            family=family,
            search=search,
            relative=relative,
            max_nfev=max_nfev,
            seed=random_state,
        )
        if np.array_equal(np.asarray(b_vec), b):
            holder["model_params"] = params
        return spectrum, nfev, success

    x0_default = np.ones(n_energy_bins) * np.mean(b) / max(
        np.mean(A.sum(axis=1)), 1e-30
    )

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
        method_name=f"PFNS parametric ({family})",
        extra_output={"family": family, "search": search, "relative": relative},
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
    return result


# Backwards/alias name used by detector import
unfold_pfns_parametric_impl = unfold_pfns_parametric
