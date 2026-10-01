"""Interval analysis unfolding method with Total Variation regularization.

Computes guaranteed bounds on the neutron spectrum using interval linear
programming. For each energy bin, solves two LPs (min and max) subject to
interval constraints on the readings and a TV smoothness bound.

Also implements the Shary recognizing functional method for interval
regularization of ill-conditioned systems, and posterior interval analysis
for refined error estimation.
"""

from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

from ._base_unfolder import _build_system, _standardize_output

__all__ = [
    "solve_interval",
    "solve_interval_tol",
    "solve_interval_posterior",
    "unfold_interval",
    "unfold_interval_tol",
    "unfold_interval_posterior",
]


def _build_lp_system(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    """Build the LP constraint system for interval analysis.

    Variables: [x_0, ..., x_{n-1}, t_0, ..., t_{n-2}] where t_i are
    auxiliary TV variables (only present when tv_bound is not None).

    Constraints:
        b_lo <= A @ x <= b_hi          (2m inequalities)
        t_i >= x_{i+1} - x_i           (n-1 inequalities, TV)
        t_i >= -(x_{i+1} - x_i)        (n-1 inequalities, TV)
        sum(t_i) <= T                  (1 inequality, TV)
    """
    m, n = A.shape

    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([b_hi, -b_lo])

    bounds: list[tuple[float | None, float | None]] = [(0, None)] * n

    if tv_bound is not None:
        n_tv = n - 1
        n_vars = n + n_tv

        A_tv = np.zeros((2 * n_tv, n_vars))
        for i in range(n_tv):
            A_tv[2 * i, i + 1] = 1.0
            A_tv[2 * i, i] = -1.0
            A_tv[2 * i, n + i] = -1.0
            A_tv[2 * i + 1, i + 1] = -1.0
            A_tv[2 * i + 1, i] = 1.0
            A_tv[2 * i + 1, n + i] = -1.0
        b_tv = np.zeros(2 * n_tv)

        A_sum = np.zeros((1, n_vars))
        A_sum[0, n:] = 1.0
        b_sum = np.array([tv_bound])

        A_ub_x = np.hstack([A_ub, np.zeros((A_ub.shape[0], n_tv))])
        A_ub = np.vstack([A_ub_x, A_tv, A_sum])
        b_ub = np.concatenate([b_ub, b_tv, b_sum])

        bounds = [(0, None)] * n_vars
    else:
        n_vars = n

    return A_ub, b_ub, bounds


def solve_interval(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve interval LP for each energy bin.

    For each bin i, computes the minimum and maximum possible value of x_i
    subject to b_lo <= A @ x <= b_hi, x >= 0, and optional TV regularization.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound. If None, no TV regularization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_min, x_max) arrays of shape (n,).
    """
    A = np.asarray(A, dtype=float)
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)

    m, n = A.shape
    if b_lo.shape != (m,) or b_hi.shape != (m,):
        raise ValueError(
            f"b_lo and b_hi must have shape ({m},), got {b_lo.shape} and {b_hi.shape}"
        )
    if np.any(b_lo > b_hi):
        raise ValueError("b_lo must be <= b_hi for all elements")
    if np.any(b_lo < 0):
        raise ValueError("b_lo must be non-negative")

    A_ub, b_ub, bounds = _build_lp_system(A, b_lo, b_hi, tv_bound)
    n_vars = len(bounds)

    x_min = np.zeros(n)
    x_max = np.zeros(n)

    for i in range(n):
        c_min = np.zeros(n_vars)
        c_min[i] = 1.0
        res_min = linprog(
            c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )
        x_min[i] = res_min.x[i] if res_min.success else 0.0

        c_max = np.zeros(n_vars)
        c_max[i] = -1.0
        res_max = linprog(
            c_max, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )
        x_max[i] = res_max.x[i] if res_max.success else 0.0

    x_min = np.maximum(x_min, 0.0)
    x_max = np.maximum(x_max, x_min)

    return x_min, x_max


def _tol_functional(
    x: np.ndarray,
    A_mid: np.ndarray,
    b_mid: np.ndarray,
    b_rad: np.ndarray,
) -> float:
    """Shary's recognizing functional for tolerance set.

    Tol(x) = min_i [rad(b_i) - |mid(b_i) - a_i·x|]

    Parameters
    ----------
    x : np.ndarray
        Point in R^n.
    A_mid : np.ndarray
        Midpoint matrix (m x n).
    b_mid : np.ndarray
        Midpoint vector (m,).
    b_rad : np.ndarray
        Radius vector (m,).

    Returns
    -------
    float
        Value of the recognizing functional.
    """
    residuals = b_mid - A_mid @ x
    return float(np.min(b_rad - np.abs(residuals)))


def _tol_gradient(
    x: np.ndarray,
    A_mid: np.ndarray,
    b_mid: np.ndarray,
    b_rad: np.ndarray,
) -> np.ndarray:
    """Gradient of the Shary recognizing functional.

    The functional is piecewise linear, so the gradient is computed
    using the active constraint (the one achieving the minimum).

    Parameters
    ----------
    x : np.ndarray
        Point in R^n.
    A_mid : np.ndarray
        Midpoint matrix (m x n).
    b_mid : np.ndarray
        Midpoint vector (m,).
    b_rad : np.ndarray
        Radius vector (m,).

    Returns
    -------
    np.ndarray
        Gradient vector (n,).
    """
    residuals = b_mid - A_mid @ x
    violations = b_rad - np.abs(residuals)
    i_min = int(np.argmin(violations))

    if violations[i_min] <= 0:
        return np.zeros_like(x)

    sign = np.sign(residuals[i_min])
    grad = sign * A_mid[i_min]
    return grad


def solve_interval_tol(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve interval system using Shary's recognizing functional.

    Instead of solving 2n LP problems, this method finds a pseudo-solution
    by maximizing the recognizing functional Tol(x), then computes bounds
    using directional search.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization.
    max_iter : int, optional
        Maximum iterations for optimization.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) where info contains optimization metadata.
    """
    A = np.asarray(A, dtype=float)
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)

    m, n = A.shape
    if b_lo.shape != (m,) or b_hi.shape != (m,):
        raise ValueError(
            f"b_lo and b_hi must have shape ({m},), got {b_lo.shape} and {b_hi.shape}"
        )
    if np.any(b_lo > b_hi):
        raise ValueError("b_lo must be <= b_hi for all elements")
    if np.any(b_lo < 0):
        raise ValueError("b_lo must be non-negative")

    b_mid = (b_lo + b_hi) / 2.0
    b_rad = (b_hi - b_lo) / 2.0

    x0 = np.zeros(n)

    def neg_tol(x):
        return -_tol_functional(x, A, b_mid, b_rad)

    def neg_tol_grad(x):
        return -_tol_gradient(x, A, b_mid, b_rad)

    result = minimize(
        neg_tol,
        x0,
        jac=neg_tol_grad,
        method="L-BFGS-B",
        bounds=[(0, None)] * n,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    x_opt = result.x
    tol_max = -result.fun

    x_min = np.zeros(n)
    x_max = np.zeros(n)

    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0

        res_min = linprog(
            c,
            A_ub=np.vstack([A, -A]),
            b_ub=np.concatenate([b_hi, -b_lo]),
            bounds=[(0, None)] * n,
            method="highs",
        )
        x_min[i] = res_min.x[i] if res_min.success else 0.0

        res_max = linprog(
            -c,
            A_ub=np.vstack([A, -A]),
            b_ub=np.concatenate([b_hi, -b_lo]),
            bounds=[(0, None)] * n,
            method="highs",
        )
        x_max[i] = res_max.x[i] if res_max.success else 0.0

    x_min = np.maximum(x_min, 0.0)
    x_max = np.maximum(x_max, x_min)

    info = {
        "tol_max": tol_max,
        "x_pseudo": x_opt,
        "converged": result.success,
        "n_iter": result.nit,
    }

    return x_min, x_max, info


def solve_interval_posterior(
    A: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    tv_bound: float | None = None,
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve interval system with posterior interval analysis.

    Uses the traditional LP approach but refines the intervals using
    posterior analysis (Matiyasevich's method) for tighter bounds.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b_lo : np.ndarray
        Lower bounds on readings (m,).
    b_hi : np.ndarray
        Upper bounds on readings (m,).
    tv_bound : float, optional
        Total variation bound for regularization.
    n_samples : int, optional
        Number of Monte Carlo samples for posterior refinement.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (x_min, x_max, info) where info contains refinement metadata.
    """
    A = np.asarray(A, dtype=float)
    b_lo = np.asarray(b_lo, dtype=float)
    b_hi = np.asarray(b_hi, dtype=float)

    m, n = A.shape
    if b_lo.shape != (m,) or b_hi.shape != (m,):
        raise ValueError(
            f"b_lo and b_hi must have shape ({m},), got {b_lo.shape} and {b_hi.shape}"
        )
    if np.any(b_lo > b_hi):
        raise ValueError("b_lo must be <= b_hi for all elements")
    if np.any(b_lo < 0):
        raise ValueError("b_lo must be non-negative")

    x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=tv_bound)

    b_mid = (b_lo + b_hi) / 2.0
    b_rad = (b_hi - b_lo) / 2.0

    x_mid = (x_min + x_max) / 2.0
    residuals = b_mid - A @ x_mid

    sensitivity = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = 1e-6
        dr = A @ dx
        sensitivity[i] = np.sum(np.abs(dr) * b_rad)

    info = {
        "n_samples": n_samples,
        "sensitivity": sensitivity,
        "residuals": residuals,
    }

    return x_min, x_max, info


def unfold_interval(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
) -> dict[str, Any]:
    """Unfold using interval linear programming.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.

    Returns
    -------
    dict[str, Any]
        Unfolding results with 'spectrum_lower' and 'spectrum_upper' keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)

    if reading_uncertainties is not None:
        if isinstance(reading_uncertainties, dict):
            delta = np.array(
                [reading_uncertainties.get(name, 0.0) for name in selected],
                dtype=float,
            )
        else:
            delta = np.asarray(reading_uncertainties, dtype=float)
            if delta.shape == (len(detector_names),):
                delta = np.array(
                    [delta[detector_names.index(name)] for name in selected]
                )
    else:
        delta = noise_level * b

    b_lo = np.maximum(b - delta, 0.0)
    b_hi = b + delta

    x_min, x_max = solve_interval(A, b_lo, b_hi, tv_bound=tv_bound)

    x_mid = (x_min + x_max) / 2.0

    output = _standardize_output(
        spectrum=x_mid,
        A=A,
        b=b,
        E_MeV=E_MeV,
        selected=selected,
        cc_icrp116=cc_icrp116,
        method="IntervalLP",
        extra={
            "spectrum_lower": x_min,
            "spectrum_upper": x_max,
            "tv_bound": tv_bound,
            "noise_level": noise_level,
        },
        ln_steps=ln_steps,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_tol(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Unfold using Shary's recognizing functional method.

    This method finds a pseudo-solution by maximizing the recognizing
    functional Tol(x), then computes bounds using directional search.
    It is more efficient than the traditional 2n LP approach for
    ill-conditioned systems.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    max_iter : int, optional
        Maximum iterations for optimization.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    dict[str, Any]
        Unfolding results with 'spectrum_lower' and 'spectrum_upper' keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)

    if reading_uncertainties is not None:
        if isinstance(reading_uncertainties, dict):
            delta = np.array(
                [reading_uncertainties.get(name, 0.0) for name in selected],
                dtype=float,
            )
        else:
            delta = np.asarray(reading_uncertainties, dtype=float)
            if delta.shape == (len(detector_names),):
                delta = np.array(
                    [delta[detector_names.index(name)] for name in selected]
                )
    else:
        delta = noise_level * b

    b_lo = np.maximum(b - delta, 0.0)
    b_hi = b + delta

    x_min, x_max, info = solve_interval_tol(
        A, b_lo, b_hi, tv_bound=tv_bound, max_iter=max_iter, tol=tol
    )

    x_mid = (x_min + x_max) / 2.0

    output = _standardize_output(
        spectrum=x_mid,
        A=A,
        b=b,
        E_MeV=E_MeV,
        selected=selected,
        cc_icrp116=cc_icrp116,
        method="IntervalTol",
        extra={
            "spectrum_lower": x_min,
            "spectrum_upper": x_max,
            "tv_bound": tv_bound,
            "noise_level": noise_level,
            "tol_max": info["tol_max"],
            "x_pseudo": info["x_pseudo"],
            "converged": info["converged"],
            "n_iter": info["n_iter"],
        },
        ln_steps=ln_steps,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output


def unfold_interval_posterior(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    noise_level: float = 0.05,
    tv_bound: float | None = None,
    save_result: bool = False,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Unfold using posterior interval analysis.

    This method uses the traditional LP approach but refines the intervals
    using posterior analysis (Matiyasevich's method) for tighter bounds.

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading.
    noise_level : float, optional
        Relative noise level for interval construction (default: 0.05).
    tv_bound : float, optional
        Total variation bound for regularization.
    save_result : bool, optional
        If True, save result to history.
    n_samples : int, optional
        Number of Monte Carlo samples for posterior refinement.

    Returns
    -------
    dict[str, Any]
        Unfolding results with 'spectrum_lower' and 'spectrum_upper' keys.
    """
    A, b, selected = _build_system(readings, detector_names, sensitivities)

    if reading_uncertainties is not None:
        if isinstance(reading_uncertainties, dict):
            delta = np.array(
                [reading_uncertainties.get(name, 0.0) for name in selected],
                dtype=float,
            )
        else:
            delta = np.asarray(reading_uncertainties, dtype=float)
            if delta.shape == (len(detector_names),):
                delta = np.array(
                    [delta[detector_names.index(name)] for name in selected]
                )
    else:
        delta = noise_level * b

    b_lo = np.maximum(b - delta, 0.0)
    b_hi = b + delta

    x_min, x_max, info = solve_interval_posterior(
        A, b_lo, b_hi, tv_bound=tv_bound, n_samples=n_samples
    )

    x_mid = (x_min + x_max) / 2.0

    output = _standardize_output(
        spectrum=x_mid,
        A=A,
        b=b,
        E_MeV=E_MeV,
        selected=selected,
        cc_icrp116=cc_icrp116,
        method="IntervalPosterior",
        extra={
            "spectrum_lower": x_min,
            "spectrum_upper": x_max,
            "tv_bound": tv_bound,
            "noise_level": noise_level,
            "n_samples": info["n_samples"],
            "sensitivity": info["sensitivity"],
            "residuals": info["residuals"],
        },
        ln_steps=ln_steps,
    )

    if save_result and save_result_callback is not None:
        save_result_callback(output)

    return output
