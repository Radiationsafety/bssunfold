"""Mirror descent unfolding method for neutron spectra.

Mirror descent (Nemirovski & Yudin, 1979; Beck & Teboulle, 2003) replaces the
Euclidean geometry of gradient descent with a Bregman geometry induced by a
mirror map ``psi``:

    x_{k+1} = argmin_z { <grad f(x_k), z> + (1/eta) D_psi(z, x_k) }

which is equivalent to a mirror step ``grad psi(x_{k+1}) = grad psi(x_k) - eta
grad f(x_k)`` followed by the Bregman projection back onto the feasible set.

For spectral unfolding this is a natural framework: the *entropic* mirror map
``psi(x) = sum x_i log x_i - sum x_i`` produces multiplicative updates that
generalize MLEM / GRAVEL / SAND-II (their iterative schemes are entropic
mirror-descent steps with particular step-size policies), while the *log*
(log-barrier) and *p-norm* maps yield other physically meaningful
non-negative geometries.  The entropy map automatically preserves the total
fluence (iterates stay on the simplex ``{x >= 0, sum x = F}``).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._line_search import golden_section_minimize

__all__ = ["solve_mirror_descent", "unfold_mirror_descent"]

_MIRROR_MAPS = ("entropy", "log", "l2", "pnorm")


def _mirror_next(
    x: NDArray[np.float64],
    gradient: NDArray[np.float64],
    eta: float,
    mirror_map: str,
    total_fluence: float | None,
    p: float,
) -> NDArray[np.float64]:
    """Mirror-descent proximal map applied with step ``eta``.

    ``'entropy'``: multiplicative update ``x * exp(-eta g)``, renormalized to
    the simplex ``{x >= 0, sum x = F}`` (Bregman projection).  ``'log'``:
    log-barrier update ``1/(1/x + eta g)``.  ``'l2'``: plain projected
    gradient step.  ``'pnorm'``: p-norm mirror step ``max(x^{p-1} - eta g,
    0)^{1/(p-1)}``.
    """
    if mirror_map == "entropy":
        log_step = -eta * gradient
        shift = np.max(log_step)  # overflow guard; cancels in normalization
        x_new = x * np.exp(log_step - shift)
        s = x_new.sum()
        if not np.isfinite(s) or s <= 0:
            return x
        return total_fluence * x_new / s
    if mirror_map == "log":
        inv = 1.0 / np.maximum(x, 1e-300) + eta * gradient
        # reject steps that leave the interior of the positive orthant
        if not np.all(np.isfinite(inv)) or np.any(inv <= 0):
            return x
        return 1.0 / inv
    if mirror_map == "l2":
        return np.maximum(x - eta * gradient, 0.0)
    if mirror_map == "pnorm":
        # psi(x) = ||x||_p^p / p  =>  grad psi = x^{p-1} (nonneg domain)
        xp = np.maximum(x, 1e-300) ** (p - 1.0) - eta * gradient
        return np.maximum(xp, 0.0) ** (1.0 / (p - 1.0))
    raise ValueError(f"Unknown mirror map {mirror_map!r}")


def _eta_max(
    gradient: NDArray[np.float64], L: float, mirror_map: str, x: NDArray[np.float64]
) -> float:
    """Upper bracket for the per-iteration line search."""
    g_max = float(np.max(np.abs(gradient)))
    if mirror_map in ("entropy", "log"):
        hi = 4.0 / max(g_max, 1e-300)  # scale-free multiplicative bracket
        if mirror_map == "log":
            # keep 1/x + eta*g strictly positive for every coordinate
            neg = gradient < 0
            if np.any(neg):
                inv_x = 1.0 / np.maximum(x, 1e-300)
                limit = float(np.min(inv_x[neg] / np.abs(gradient[neg])))
                hi = min(hi, 0.25 * limit)
        return hi
    return 2.0 / L if L > 0 else 1.0


def solve_mirror_descent(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    mirror_map: str = "entropy",
    step_size: float | None = None,
    total_fluence: float | None = None,
    regularization: float = 0.0,
    p: float = 3.0,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem by mirror descent.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); must be strictly positive for the 'entropy' and
        'log' mirror maps (add a small floor if needed).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    mirror_map : str, optional
        ``'entropy'`` (simplex, generalizes MLEM/GRAVEL), ``'log'``
        (log-barrier, positive orthant), ``'l2'`` (equals projected gradient)
        or ``'pnorm'`` (nonnegative orthant) (default: 'entropy').
    step_size : float, optional
        Mirror step ``eta``.  If None (default), the step is chosen at every
        iteration by golden-section line search along the mirror trajectory
        (recommended: the entropic geometry makes fixed steps extremely
        slow, just as plain MLEM is slow).
    total_fluence : float, optional
        Simplex level for the entropy map; defaults to ``sum(x0)``.
    regularization : float, optional
        Tikhonov (L2) regularization strength (default: 0.0).
    p : float, optional
        Order of the p-norm mirror map, ``p > 1`` (default: 3.0).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    if mirror_map not in _MIRROR_MAPS:
        raise ValueError(f"mirror_map must be one of {_MIRROR_MAPS}")
    if mirror_map == "pnorm" and p <= 1.0:
        raise ValueError("p-norm mirror map requires p > 1")

    if mirror_map == "entropy":
        F = float(total_fluence) if total_fluence is not None else float(x0.sum())
        if F <= 0:
            raise ValueError("entropy mirror map requires positive total fluence")
        x = np.maximum(x0, 1e-300).copy()
        x = F * x / x.sum()
    else:
        F = None
        max_A = max(float(np.max(A)), 1e-30)
        floor = max(float(np.max(b)) / max_A, 1e-12) / max(A.shape[1], 1)
        x = np.maximum(x0, floor)

    L = np.linalg.norm(A, 2) ** 2 + max(regularization, 0.0)

    def objective(z: np.ndarray) -> float:
        r = A @ z - b
        val = 0.5 * float(r @ r)
        if regularization:
            val += 0.5 * regularization * float(z @ z)
        return val

    converged = False
    iterations = 0
    for k in range(max_iterations):
        gradient = A.T @ (A @ x - b)
        if regularization:
            gradient = gradient + regularization * x

        if step_size is None:
            # Adaptive step: golden-section search along the mirror trajectory
            hi = _eta_max(gradient, L, mirror_map, x)
            if hi <= 0:
                break

            def phi(t: float, x=x, gradient=gradient) -> float:
                return float(objective(_mirror_next(x, gradient, t, mirror_map, F, p)))

            eta_k, _ = golden_section_minimize(phi, 0.0, hi, tolerance=hi * 1e-4)
        else:
            eta_k = float(step_size)

        x_new = _mirror_next(x, gradient, eta_k, mirror_map, F, p)

        rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-300)
        x = x_new
        iterations = k + 1
        if rel_change < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_mirror_descent(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    mirror_map: str = "entropy",
    step_size: float | None = None,
    total_fluence: float | None = None,
    regularization: float = 0.0,
    p: float = 3.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using mirror descent.

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
        Initial spectrum guess (strictly positive for 'entropy'/'log' maps).
    max_iterations : int, optional
        Maximum iterations (default: 2000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    mirror_map : str, optional
        ``'entropy'``, ``'log'``, ``'l2'`` or ``'pnorm'`` (default: 'entropy').
    step_size : Optional[float], optional
        Mirror step size; default ``1 / (||A||^2 + reg)``.
    total_fluence : Optional[float], optional
        Total fluence for the entropy map (default: sum of initial spectrum,
        estimated from a uniform fit when not provided).
    regularization : float, optional
        Tikhonov regularization strength (default: 0.0).
    p : float, optional
        Order of the p-norm mirror map (default: 3.0).
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
    if mirror_map not in _MIRROR_MAPS:
        raise ValueError(f"mirror_map must be one of {_MIRROR_MAPS}")

    selected = [name for name in detector_names if name in readings]
    A_est = np.array([sensitivities[name] for name in selected], dtype=float)
    b_est = np.array([readings[name] for name in selected], dtype=float)

    if mirror_map == "entropy":
        if total_fluence is None:
            mean_response = max(float(np.mean(A_est)), 1e-30)
            total_fluence = float(np.mean(b_est) / mean_response * n_energy_bins)
        x0_default = np.full(n_energy_bins, total_fluence / n_energy_bins)
    else:
        # small positive start so that log/entropy-type maps are well defined
        scale = max(float(np.mean(b_est)), 1e-30) / max(float(np.mean(A_est)), 1e-30)
        x0_default = np.full(n_energy_bins, scale / max(n_energy_bins, 1))

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_mirror_descent,
            max_iterations=max_iterations,
            tolerance=tolerance,
            mirror_map=mirror_map,
            step_size=step_size,
            total_fluence=total_fluence,
            regularization=regularization,
            p=p,
        ),
        solve_kwargs={},
        method_name="Mirror Descent",
        extra_output={
            "mirror_map": mirror_map,
            "total_fluence": total_fluence if mirror_map == "entropy" else None,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        variance_reduction=variance_reduction,
        random_state=random_state,
        save_result=save_result,
    )
