"""IMAXED unfolding method for neutron spectrum reconstruction.

Implements Improved Maximum Entropy Deconvolution (Wong 2024)
using Newton's method with line search (Wolfe conditions) instead of L-BFGS-B.

The algorithm searches for roots of the vector-valued function using Newton's method,
with guaranteed convergence to the optimal solution through line search.

References
----------
Wong, O. (2024). Modernising neutron spectrum unfolding for fusion applications.
PhD Thesis, Sheffield Hallam University. https://shura.shu.ac.uk/36014/
"""

from typing import Any

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_imaxed", "unfold_imaxed"]


def solve_imaxed(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using IMAXED (Improved MAXED).

    Newton iteration in phi-space with Armijo backtracking line search.

    Minimises ``f(phi) = 0.5*(A phi - b)^T S_b (A phi - b)
    + sum_i phi_i*log(phi_i/phi0_i) - phi_i + phi0_i`` where
    ``S_b = diag(1/sigma^2)``, ``sigma = sigma_factor * max(b, eps)``.
    Gradient ``g = A^T S_b (A phi - b) + log(phi/phi0)``,
    Hessian ``H = A^T S_b A + diag(1/phi)``.  ``line_search_tol``
    is the Armijo ``c1`` constant (``0 < c1 < 1``).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Reference (prior) spectrum (n,).
    sigma_factor : float, optional
        Relative measurement uncertainty (default: 0.1).
    max_iterations : int, optional
        Maximum Newton iterations (default: 5000).
    tolerance : float, optional
        Gradient convergence tolerance (default: 1e-8).
    line_search_tol : float, optional
        Armijo line-search constant c1 (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    _m, n = A.shape

    phi_floor = 1e-12

    b_arr = np.asarray(b, dtype=float).ravel()
    b_safe = np.maximum(b_arr, 1e-300)
    sigma = sigma_factor * b_safe
    S_b = np.diag(1.0 / (sigma**2))

    phi_0 = np.maximum(np.asarray(x0, dtype=float).ravel(), 1e-300)
    if phi_0.size != n:
        raise ValueError(f"x0 length {phi_0.size} != n {n}")
    log_phi_0 = np.log(phi_0)

    At_Sb_A = A.T @ S_b @ A

    def _objective(phi: np.ndarray) -> float:
        p = np.maximum(phi, phi_floor)
        residual = A @ p - b_arr
        chi2 = 0.5 * float(residual @ (S_b @ residual))
        kl = float(np.sum(p * (np.log(p + 1e-300) - log_phi_0) - p + phi_0))
        return chi2 + kl

    def _gradient(phi: np.ndarray) -> np.ndarray:
        p = np.maximum(phi, phi_floor)
        residual = A @ p - b_arr
        return A.T @ (S_b @ residual) + np.log(p + 1e-300) - log_phi_0

    def _hessian(phi: np.ndarray) -> np.ndarray:
        p = np.maximum(phi, phi_floor)
        return At_Sb_A + np.diag(1.0 / (p + 1e-300))

    phi = phi_0.copy()
    # Clamp Armijo constant to (0,1) — callers may pass 1e-6..1e-4
    c1 = float(np.clip(line_search_tol, 1e-12, 0.5))

    grad_norm = np.inf
    iteration = 0
    for iteration in range(max_iterations):
        grad = _gradient(phi)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < tolerance:
            break

        Hess = _hessian(phi)
        try:
            delta = np.linalg.solve(Hess, -grad)
        except np.linalg.LinAlgError:
            reg = 1e-6 * float(np.max(np.abs(np.diag(Hess)))) or 1e-12
            Hess_reg = Hess + reg * np.eye(n)
            delta = np.linalg.solve(Hess_reg, -grad)

        # Ensure descent direction
        slope = float(np.dot(grad, delta))
        if slope >= 0:
            delta = -grad
            slope = float(np.dot(grad, delta))

        base = _objective(phi)
        beta = 1.0
        accepted = False
        for _ in range(30):
            trial = np.maximum(phi + beta * delta, phi_floor)
            if _objective(trial) <= base + c1 * beta * slope:
                accepted = True
                break
            beta *= 0.5
        if not accepted:
            beta = 0.01

        phi = np.maximum(phi + beta * delta, phi_floor)
    else:
        # loop exhausted without break — recompute grad norm at last phi
        grad_norm = float(np.linalg.norm(_gradient(phi)))

    iterations = iteration + 1
    converged = bool(grad_norm < tolerance)
    return phi, iterations, converged


def unfold_imaxed(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    sigma_factor: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using the IMAXED algorithm.

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
        Reference spectrum. If None, a flat reference is used.
    sigma_factor : float, optional
        Relative measurement uncertainty (default: 0.1).
    max_iterations : int, optional
        Maximum Newton iterations (default: 5000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-8).
    line_search_tol : float, optional
        Line search tolerance (default: 1e-6).
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
        Unfolding results dictionary.
    """
    if initial_spectrum is not None:
        x0_ref = np.asarray(initial_spectrum, dtype=float)
    else:
        x0_ref = np.ones(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=x0_ref,
        default_initial=np.ones(n_energy_bins),
        solve_func=make_solve_wrapper(
            solve_imaxed,
            sigma_factor=sigma_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
        ),
        solve_kwargs={},
        method_name="IMAXED",
        extra_output={
            "sigma_factor": sigma_factor,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
