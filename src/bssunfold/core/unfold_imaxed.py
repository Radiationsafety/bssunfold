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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

from ..utils.validators import validate_system

__all__ = ["solve_imaxed", "unfold_imaxed"]


def solve_imaxed(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using IMAXED (Improved MAXED).

    Uses gradient-based optimization with cross-entropy regularization
    for stable and reliable convergence.

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
        Maximum iterations (default: 5000).
    tolerance : float, optional
        Gradient convergence tolerance (default: 1e-8).
    line_search_tol : float, optional
        Line search tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A, b, x0 = validate_system(A, b, x0=x0)
    from scipy.optimize import minimize

    m, n = A.shape

    # Measurement uncertainties
    b_safe = np.maximum(b, 1e-300)
    sigma = sigma_factor * b_safe
    S_b = np.diag(1.0 / (sigma**2))  # Inverse covariance matrix

    # Reference spectrum (strictly positive)
    phi_0 = np.maximum(x0, 1e-300)

    def objective_and_gradient(y: np.ndarray):
        """Compute objective and gradient in log-space."""
        x = np.exp(y)

        # Forward model
        Ax = A @ x
        residual = Ax - b

        # Chi-squared term
        chi2 = 0.5 * residual @ S_b @ residual

        # Cross-entropy regularization (relative to prior)
        log_phi_0 = np.log(phi_0)
        entropy = np.sum(x * (np.log(x + 1e-300) - log_phi_0) - x + phi_0)

        # Total objective
        f = chi2 + entropy

        # Gradient
        AT_Sb_res = A.T @ (S_b @ residual)
        grad_x = AT_Sb_res + np.log(x + 1e-300) - log_phi_0

        # Gradient in y-space: df/dy = x * df/dx
        grad_y = x * grad_x

        return f, grad_y

    # Initial point in log-space
    y0 = np.log(phi_0)

    # Optimize using L-BFGS-B
    result = minimize(
        objective_and_gradient,
        y0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "gtol": tolerance, "ftol": 0},
    )

    x_opt = np.exp(result.x)
    iterations = result.nit if hasattr(result, "nit") else 0
    converged = result.success or result.status == 0

    return x_opt, iterations, converged


def unfold_imaxed(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma_factor: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
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
        Save result to history (default: True).
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
