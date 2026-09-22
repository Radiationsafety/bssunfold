"""Subgradient unfolding methods for neutron spectra.

Projected subgradient descent for the nonsmooth unfolding problem

    min_{x >= 0}  f(x) = 1/2 ||A x - b||^2
                  + l1_penalty * ||x||_1
                  + tv_penalty * ||D x||_1

(lecture 8 of the MIPT optimization course, homework 12: subgradient and
adaptive methods for nonsmooth optimization).  A subgradient of ``f`` is

    g = A^T (A x - b) + l1 * s(x) + tv * D^T sign(D x),

with ``s(x)_j = sign(x_j)`` (taking 0 at zero coordinates вЂ” a valid
subgradient choice).  Step-size policies:

- ``'polyak'``     вЂ” Polyak step ``t_k = (f(x_k) - f*) / ||g_k||^2`` with
  ``f*`` estimated by the best objective seen so far (scaled by a shrink
  factor); converges O(1/sqrt(k)) when ``f*`` is known or well estimated;
- ``'diminishing'`` вЂ” square-summable-but-not-summable series
  ``t_k = t0 / (1 + decay * k)``;
- ``'fixed'``      вЂ” constant step ``t0`` (converges to a neighborhood).

For nonsmooth problems the *best* iterate by objective value is returned
(the iterate sequence itself does not need to converge), which is the
standard practice for subgradient schemes.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_subgradient", "unfold_subgradient"]

_STEPS = ("polyak", "diminishing", "fixed")


def _difference_matrix(n: int) -> NDArray[np.float64]:
    """First-order difference operator D (``(D x)_i = x_{i+1} - x_i``)."""
    D = np.zeros((max(n - 1, 0), n))
    idx = np.arange(n - 1)
    D[idx, idx] = -1.0
    D[idx, idx + 1] = 1.0
    return D


def solve_subgradient(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 3000,
    tolerance: float = 1e-8,
    l1_penalty: float = 0.0,
    tv_penalty: float = 0.0,
    step_policy: str = "diminishing",
    step_size: float = 1.0,
    decay: float = 1.0,
    polyak_margin: float = 0.05,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem by projected subgradient descent.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); projected onto the nonnegative orthant.
    max_iterations : int, optional
        Maximum iterations (default: 3000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    l1_penalty : float, optional
        L1 (sparsity) penalty weight (default: 0.0).
    tv_penalty : float, optional
        Total-variation penalty weight ``||D x||_1`` (default: 0.0).
    step_policy : str, optional
        ``'polyak'``, ``'diminishing'`` or ``'fixed'``
        (default: 'diminishing').
    step_size : float, optional
        Base step ``t0`` for 'fixed'/'diminishing' policies (default: 1.0).
    decay : float, optional
        Decay rate of the diminishing step (default: 1.0).
    polyak_margin : float, optional
        Relative shrink of the running-best objective used as the ``f*``
        estimate in the Polyak rule (default: 0.05).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (best solution found, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    if step_policy not in _STEPS:
        raise ValueError(f"step_policy must be one of {_STEPS}")

    n = A.shape[1]
    l1_penalty = max(float(l1_penalty), 0.0)
    tv_penalty = max(float(tv_penalty), 0.0)
    D = _difference_matrix(n) if tv_penalty > 0 else None

    def objective(z: np.ndarray) -> float:
        r = A @ z - b
        val = 0.5 * float(r @ r)
        if l1_penalty:
            val += l1_penalty * float(np.abs(z).sum())
        if D is not None:
            val += tv_penalty * float(np.abs(D @ z).sum())
        return val

    def subgradient(z: np.ndarray) -> np.ndarray:
        g = A.T @ (A @ z - b)
        if l1_penalty:
            g = g + l1_penalty * np.sign(z)
        if D is not None:
            g = g + tv_penalty * (D.T @ np.sign(D @ z))
        return g

    # Scale-aware base step: t0 * ||g(x0)|| ~ ||x||_ref makes one step move
    # the iterate by a reference solution magnitude (||b||/||A||_2).
    x_scale = float(np.linalg.norm(b)) / max(float(np.linalg.norm(A, 2)), 1e-30)
    g0 = subgradient(np.maximum(x0, 0.0))
    g0_norm = float(np.linalg.norm(g0))
    if step_size is None:
        step_size = 1.0
    t0 = float(step_size) * max(x_scale, 1e-300) / max(g0_norm, 1e-300)

    x = np.maximum(x0, 0.0)
    best_x = x.copy()
    best_f = objective(x)
    f_star_est = best_f * max(1.0 - polyak_margin, 0.0)

    converged = False
    iterations = 0
    for k in range(max_iterations):
        g = subgradient(x)
        g_norm_sq = float(g @ g)

        if step_policy == "polyak":
            f_cur = objective(x)
            if f_cur < best_f:
                best_f = f_cur
                best_x = x.copy()
                f_star_est = best_f * max(1.0 - polyak_margin, 0.0)
            if g_norm_sq > 0:
                t = (f_cur - f_star_est) / g_norm_sq
            else:
                t = 0.0
            t = max(t, 0.0)
        elif step_policy == "diminishing":
            t = t0 / (1.0 + decay * k)
        else:
            t = t0

        x_new = np.maximum(x - t * g, 0.0)
        rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-30)
        x = x_new
        iterations = k + 1

        f_new = objective(x)
        if f_new < best_f:
            best_f = f_new
            best_x = x.copy()

        if rel_change < tolerance:
            converged = True
            break

    return best_x, iterations, converged


def unfold_subgradient(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 3000,
    tolerance: float = 1e-8,
    l1_penalty: float = 0.0,
    tv_penalty: float = 0.0,
    step_policy: str = "diminishing",
    step_size: float = 1.0,
    decay: float = 1.0,
    polyak_margin: float = 0.05,
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
    """Unfold neutron spectrum using projected subgradient descent.

    Nonsmooth L1/TV penalties are handled natively via subgradients with
    Polyak / diminishing / fixed step-size policies; the best iterate by
    objective value is returned.

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
        Initial spectrum guess.
    max_iterations : int, optional
        Maximum iterations (default: 3000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    l1_penalty : float, optional
        L1 penalty weight (default: 0.0).
    tv_penalty : float, optional
        TV penalty weight (default: 0.0).
    step_policy : str, optional
        ``'polyak'``, ``'diminishing'`` or ``'fixed'``
        (default: 'diminishing').
    step_size : float, optional
        Base step size (default: 1.0).
    decay : float, optional
        Diminishing-step decay rate (default: 1.0).
    polyak_margin : float, optional
        Relative margin for the Polyak optimal-value estimate
        (default: 0.05).
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
    if step_policy not in _STEPS:
        raise ValueError(f"step_policy must be one of {_STEPS}")

    scale = max(float(np.mean(list(readings.values()))), 1e-30) / max(
        float(np.mean([np.mean(v) for v in sensitivities.values()])), 1e-30
    )
    x0_default = np.full(n_energy_bins, scale / max(n_energy_bins, 1))

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
            solve_subgradient,
            max_iterations=max_iterations,
            tolerance=tolerance,
            l1_penalty=l1_penalty,
            tv_penalty=tv_penalty,
            step_policy=step_policy,
            step_size=step_size,
            decay=decay,
            polyak_margin=polyak_margin,
        ),
        solve_kwargs={},
        method_name="Subgradient Descent",
        extra_output={
            "l1_penalty": l1_penalty,
            "tv_penalty": tv_penalty,
            "step_policy": step_policy,
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
