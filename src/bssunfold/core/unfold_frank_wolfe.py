"""Frank--Wolfe (conditional gradient) unfolding method.

Solves the fluence-constrained unfolding problem

    min_x  1/2 ||A x - b||^2   s.t.  x >= 0,  sum(x) = total_fluence

by the Frank--Wolfe algorithm (Levitin & Polyak variant of the conditional
gradient method of Frank & Wolfe, 1956).  Every iteration linearizes the
objective and solves the resulting linear minimization oracle (LMO) over the
simplex, which amounts to picking the single "vertex spectrum" bin that
descent wants most.  The iterate stays a convex combination of vertices,
which for spectral unfolding means a non-negative spectrum with the total
fluence preserved exactly at every iteration.

Optional away-steps (Wolfe's away step / Lacoste-Julien & Jaggi 2015)
significantly reduce the "zig-zagging" of plain FW near the optimum.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import estimate_total_fluence

__all__ = ["solve_frank_wolfe", "unfold_frank_wolfe"]


def _lmo_simplex(
    gradient: NDArray[np.float64], total_fluence: float
) -> NDArray[np.float64]:
    """Linear minimization oracle over the simplex: put all mass on argmin."""
    s = np.zeros_like(gradient)
    s[np.argmin(gradient)] = total_fluence
    return s


def solve_frank_wolfe(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    total_fluence: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    away_steps: bool = True,
    line_search: str = "exact",
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with the Frank--Wolfe algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); will be projected onto the simplex.
    total_fluence : float
        Total fluence ``F`` (simplex level), must be positive.
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Duality-gap based stopping tolerance (default: 1e-8).  The Frank--
        Wolfe gap ``<grad, x - s>`` with the LMO vertex ``s`` upper-bounds
        the objective suboptimality and is the natural certificate here.
    away_steps : bool, optional
        Enable Wolfe away-steps for faster local convergence (default: True).
    line_search : str, optional
        ``'exact'`` вЂ” closed-form minimization of the quadratic along the
        segment, ``'backtracking'`` вЂ” Armijo backtracking (default: 'exact').

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    if total_fluence <= 0:
        raise ValueError("total_fluence must be positive")

    F = float(total_fluence)
    x = np.maximum(x0, 0.0)
    x = F * x / x.sum() if x.sum() > 0 else np.full_like(x, F / x.size)

    ATb = A.T @ b
    G = A.T @ A  # n x n Gram matrix; n is small for BSS, precompute once

    converged = False
    iterations = 0
    for k in range(max_iterations):
        grad = G @ x - ATb

        # Frank-Wolfe (duality) gap: <grad, x - s> >= f(x) - f*
        s = _lmo_simplex(grad, F)
        fw_gap = float(grad @ (x - s))
        iterations = k + 1
        if fw_gap <= tolerance * max(1.0, abs(float(grad @ x))):
            converged = True
            break

        # Regular FW step: move from x towards the LMO vertex
        d = s - x
        gamma_max = 1.0

        # Away-step candidate: drop mass from the supported vertex with the
        # largest gradient component.  The step moves towards the remaining
        # support renormalized back onto the simplex, so both endpoints are
        # feasible and any convex combination stays on the simplex.
        if away_steps:
            mask = x > 0
            if np.any(mask) and mask.sum() > 1:
                i_away = int(np.argmax(np.where(mask, grad, -np.inf)))
                x_rest = x.copy()
                x_rest[i_away] = 0.0
                x_rest = F * x_rest / x_rest.sum()
                away_gap = float(grad @ (x - x_rest))  # == -<grad, d_away>
                if away_gap > fw_gap:
                    d = x_rest - x
                    gamma_max = 1.0

        if line_search == "exact":
            g_d = float(grad @ d)
            gd = float(d @ (G @ d))
            if gd <= 0:
                gamma = gamma_max
            else:
                gamma = float(np.clip(-g_d / gd, 0.0, gamma_max))
        else:

            def phi(t, x=x, d=d):
                r = A @ (x + t * d) - b
                return 0.5 * float(r @ r)

            from ._line_search import golden_section_minimize

            t_opt, _ = golden_section_minimize(phi, 0.0, gamma_max)
            gamma = t_opt

        if gamma <= 1e-16:
            # no descent possible along the chosen direction: the FW gap is
            # (numerically) the best achievable decrease -> converged
            converged = True
            break
        x = x + gamma * d
        x = np.maximum(x, 0.0)
        x = F * x / x.sum()

    return x, iterations, converged


def unfold_frank_wolfe(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    total_fluence: float | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    away_steps: bool = True,
    line_search: str = "exact",
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
    """Unfold neutron spectrum using the Frank--Wolfe algorithm.

    The spectrum is constrained to the simplex ``{x >= 0, sum(x) = F}`` where
    ``F`` defaults to the total fluence implied by an initial uniform fit.

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
        Initial spectrum guess (projected onto the simplex).
    total_fluence : Optional[float], optional
        Simplex level ``F``.  If None, estimated from a uniform fit.
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Frank-Wolfe gap tolerance (default: 1e-8).
    away_steps : bool, optional
        Use Wolfe away-steps (default: True).
    line_search : str, optional
        ``'exact'`` or ``'backtracking'`` (default: 'exact').
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        Monte-Carlo variance reduction: ``'none'``, ``'antithetic'``,
        ``'control'`` or ``'both'`` (default: 'none').
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    if total_fluence is None:
        selected = [name for name in detector_names if name in readings]
        A_est = np.array([sensitivities[name] for name in selected], dtype=float)
        b_est = np.array([readings[name] for name in selected], dtype=float)
        # data-driven estimate: NNLS fit total (see estimate_total_fluence)
        total_fluence = estimate_total_fluence(A_est, b_est)

    x0_default = np.full(n_energy_bins, total_fluence / n_energy_bins)

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
        solve_func=make_solve_wrapper(
            solve_frank_wolfe,
            total_fluence=total_fluence,
            max_iterations=max_iterations,
            tolerance=tolerance,
            away_steps=away_steps,
            line_search=line_search,
        ),
        solve_kwargs={},
        method_name="Frank-Wolfe",
        extra_output={
            "total_fluence": total_fluence,
            "away_steps": away_steps,
            "line_search": line_search,
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
