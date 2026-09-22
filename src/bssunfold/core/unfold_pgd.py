"""Projected gradient descent unfolding method for neutron spectra.

Solves the constrained least-squares unfolding problem

    min_x  1/2 ||A x - b||^2 + reg/2 ||x||^2
    s.t.   x in C,   C = nonnegative orthant | box | simplex

by projected gradient descent: each step performs a gradient step followed by
the Euclidean projection onto the constraint set C.  Supporting the simplex
``C = {x >= 0, sum x = F}`` makes it possible to keep the total fluence fixed
at a physically meaningful value while unfolding вЂ” a constraint the plain
Landweber iteration (which is PGD onto the orthant with a fixed step) cannot
enforce.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_pgd", "project_onto_set", "unfold_pgd"]


def project_onto_set(
    x: NDArray[np.float64],
    constraint: str = "nonnegative",
    total_fluence: float | None = None,
    x_max: float = np.inf,
) -> NDArray[np.float64]:
    """Project a vector onto the constraint set.

    Parameters
    ----------
    x : np.ndarray
        Point to project.
    constraint : str, optional
        One of ``'nonnegative'`` (default), ``'box'`` or ``'simplex'``.
    total_fluence : float, optional
        Total fluence ``F`` for the simplex constraint ``sum(x) = F``.
    x_max : float, optional
        Upper bound for the ``'box'`` constraint (default: inf).

    Returns
    -------
    np.ndarray
        Projected point.
    """
    constraint = constraint.lower()
    if constraint == "nonnegative":
        return np.maximum(x, 0.0)
    if constraint == "box":
        return np.clip(x, 0.0, x_max)
    if constraint == "simplex":
        if total_fluence is None or total_fluence <= 0:
            raise ValueError("simplex projection requires total_fluence > 0")
        return _project_simplex(x, float(total_fluence))
    raise ValueError(
        f"Unknown constraint {constraint!r}; expected 'nonnegative', 'box' or 'simplex'"
    )


def _project_simplex(v: NDArray[np.float64], total: float) -> NDArray[np.float64]:
    """Euclidean projection onto the simplex {x >= 0, sum x = total}."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) + (total - css) > 0)[0][-1]
    theta = (css[rho] - total) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_pgd(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    step_size: float | None = None,
    regularization: float = 0.0,
    constraint: str = "nonnegative",
    total_fluence: float | None = None,
    x_max: float = np.inf,
    backtracking: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem by projected gradient descent.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-6).
    step_size : float, optional
        Fixed gradient step; defaults to ``1 / L`` with ``L = ||A||_2^2 + reg``.
    regularization : float, optional
        Tikhonov (L2) regularization strength (default: 0.0).
    constraint : str, optional
        ``'nonnegative'``, ``'box'`` or ``'simplex'`` (default: 'nonnegative').
    total_fluence : float, optional
        Total fluence for the simplex constraint.
    x_max : float, optional
        Upper bound for the box constraint (default: inf).
    backtracking : bool, optional
        Use Armijo backtracking on the objective when the nominal step fails
        (default: False).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    x = project_onto_set(x0.copy(), constraint, total_fluence, x_max)

    L = np.linalg.norm(A, 2) ** 2 + max(regularization, 0.0)
    if L <= 0:
        return x, 0, False
    t = float(step_size) if step_size is not None else 1.0 / L

    def objective(z: np.ndarray) -> float:
        r = A @ z - b
        return 0.5 * float(r @ r) + 0.5 * regularization * float(z @ z)

    converged = False
    iterations = 0
    for k in range(max_iterations):
        gradient = A.T @ (A @ x - b)
        if regularization:
            gradient += regularization * x

        x_new = x - t * gradient
        if backtracking:
            # Armijo condition on the projected step; shrink t until accepted
            direction = x_new - x
            f0 = objective(x)
            slope = float(gradient @ direction)
            trial = 1.0
            while trial > 1e-12 and (
                objective(x + trial * direction) > f0 + 1e-4 * trial * slope
            ):
                trial *= 0.5
            x_new = x + trial * direction
        x_new = project_onto_set(x_new, constraint, total_fluence, x_max)

        rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-30)
        x = x_new
        iterations = k + 1
        if rel_change < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_pgd(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization: float = 0.0,
    constraint: str = "nonnegative",
    total_fluence: float | None = None,
    x_max: float = np.inf,
    backtracking: bool = False,
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
    """Unfold neutron spectrum using projected gradient descent.

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
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-6).
    regularization : float, optional
        Tikhonov regularization strength (default: 0.0).
    constraint : str, optional
        Constraint set: ``'nonnegative'``, ``'box'`` or ``'simplex'``
        (default: 'nonnegative').
    total_fluence : float, optional
        Total fluence for the simplex constraint (required for 'simplex').
    x_max : float, optional
        Upper bound for the box constraint (default: inf).
    backtracking : bool, optional
        Use Armijo backtracking (default: False).
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
    if constraint == "simplex" and total_fluence is None:
        raise ValueError("constraint='simplex' requires total_fluence")
    x0_default = np.zeros(n_energy_bins)

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
        solve_func=make_solve_wrapper(
            solve_pgd,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
            constraint=constraint,
            total_fluence=total_fluence,
            x_max=x_max,
            backtracking=backtracking,
        ),
        solve_kwargs={},
        method_name="Projected Gradient Descent",
        extra_output={
            "constraint": constraint,
            "total_fluence": total_fluence,
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

    # Duality-gap certificate (Lagrange duality / KKT diagnostics)
    selected = [name for name in detector_names if name in readings]
    A = np.array([sensitivities[name] for name in selected], dtype=float)
    b = np.array([readings[name] for name in selected], dtype=float)
    from ._dual_diagnostics import nnls_duality_gap

    result["duality_gap"] = nnls_duality_gap(A, b, result["spectrum"])
    return result
