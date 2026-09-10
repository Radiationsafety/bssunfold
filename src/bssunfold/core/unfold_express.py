"""Express coarse-to-fine unfolding for Bonner-sphere responses.

The original Express method used threshold reactions.  This adaptation keeps
its piecewise-exponential spectrum model but evaluates the actual Bonner
sphere response functions, so no effective-threshold table is required.
"""

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import compute_log_steps

__all__ = ["solve_express", "unfold_express"]


def _piecewise_exponential(
    E: np.ndarray, boundaries: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Evaluate a spectrum whose log-value is linear between boundaries."""
    return np.exp(np.interp(E, boundaries, np.log(np.maximum(values, 1e-300))))


def solve_express(
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray,
    x0: np.ndarray | None = None,
    n_groups: int = 6,
    interval_boundaries: np.ndarray | None = None,
    max_iterations: int = 3,
    tol_iteration: float = 0.05,
    relative_uncertainty: float = 0.05,
) -> tuple[np.ndarray, int, bool]:
    """Fit a piecewise-exponential spectrum directly to sphere readings."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    E = np.asarray(E, dtype=float)
    if A.ndim != 2 or b.shape != (A.shape[0],) or E.shape != (A.shape[1],):
        raise ValueError("A, b, and E dimensions are inconsistent")
    if np.any(b < 0) or np.any(~np.isfinite(b)) or np.any(np.diff(E) <= 0):
        raise ValueError("Express requires non-negative readings and increasing E")
    if interval_boundaries is None:
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        boundaries = np.linspace(E[0], E[-1], n_groups + 1)
    else:
        boundaries = np.asarray(interval_boundaries, dtype=float)
        if (
            boundaries.ndim != 1
            or boundaries.size < 2
            or np.any(np.diff(boundaries) <= 0)
        ):
            raise ValueError("interval_boundaries must be strictly increasing")
        if boundaries[0] < E[0] or boundaries[-1] > E[-1]:
            raise ValueError("interval_boundaries must lie within E")
    centers = boundaries
    initial = (
        np.ones_like(E)
        if x0 is None
        else np.maximum(np.asarray(x0, dtype=float), 1e-30)
    )
    if initial.shape != E.shape:
        raise ValueError("x0 must match the energy grid")
    guess = np.interp(centers, E, np.log(initial))
    sigma = np.maximum(relative_uncertainty * np.maximum(b, 1e-30), 1e-30)

    def model(log_values: np.ndarray) -> np.ndarray:
        spectrum = _piecewise_exponential(E, boundaries, np.exp(log_values))
        return A @ spectrum

    result = least_squares(
        lambda p: (model(p) - b) / sigma,
        guess,
        max_nfev=max(1, max_iterations) * 100,
    )
    spectrum = _piecewise_exponential(E, boundaries, np.exp(result.x))
    relative_change = np.linalg.norm(model(result.x) - b) / (
        np.linalg.norm(b) + 1e-30
    )
    return spectrum, int(result.nfev), bool(
        result.success or relative_change <= tol_iteration
    )


def unfold_express(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    n_groups: int = 6,
    interval_boundaries: np.ndarray | None = None,
    max_iterations: int = 3,
    tol_iteration: float = 0.05,
    relative_uncertainty: float = 0.05,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold Bonner-sphere readings with a piecewise-exponential model."""
    boundaries = interval_boundaries
    if boundaries is None:
        boundaries = np.linspace(float(E_MeV[0]), float(E_MeV[-1]), n_groups + 1)
    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.ones(n_energy_bins),
        solve_func=make_solve_wrapper(
            lambda A, b, x0=None: solve_express(
                A, b, E_MeV, x0=x0, n_groups=n_groups,
                interval_boundaries=boundaries, max_iterations=max_iterations,
                tol_iteration=tol_iteration,
                relative_uncertainty=relative_uncertainty,
            )
        ),
        solve_kwargs={},
        method_name="Express",
        extra_output={
            "n_groups": int(len(boundaries) - 1),
            "interval_boundaries": np.asarray(boundaries),
            "tol_iteration": float(tol_iteration),
            "log_steps": compute_log_steps(np.asarray(E_MeV), n_energy_bins),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
