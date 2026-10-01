"""Interval analysis unfolding method with Total Variation regularization.

Computes guaranteed bounds on the neutron spectrum using interval linear
programming. For each energy bin, solves two LPs (min and max) subject to
interval constraints on the readings and a TV smoothness bound.
"""

from typing import Any

import numpy as np
from scipy.optimize import linprog

from ._base_unfolder import _build_system, _standardize_output

__all__ = ["solve_interval", "unfold_interval"]


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
