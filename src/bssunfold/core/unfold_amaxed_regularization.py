"""AMAXED-Regularization unfolding method for neutron spectrum reconstruction.

Implements AMAXED with Tikhonov-style regularization (Wong 2024).
Instead of fixing chi-squared and minimizing cross-entropy, this method
simultaneously minimizes both chi-squared and the regularizing function,
providing more stable convergence without requiring manual chi-squared tuning.

The algorithm uses Newton's method with line search for guaranteed convergence
to the optimal solution.

References
----------
Wong, O. (2024). Modernising neutron spectrum unfolding for fusion applications.
PhD Thesis, Sheffield Hallam University. https://shura.shu.ac.uk/36014/
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_amaxed_regularization", "unfold_amaxed_regularization"]


def solve_amaxed_regularization(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.1,
    tau: float = 1.0,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using AMAXED-Regularization.

    Uses Tikhonov-style regularization to simultaneously minimize chi-squared
    and cross-entropy without requiring a fixed target chi-squared value.

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
    tau : float, optional
        Regularization parameter (default: 1.0). Larger values favor
        solutions closer to the prior.
    max_iterations : int, optional
        Maximum Newton iterations (default: 5000).
    tolerance : float, optional
        Gradient convergence tolerance (default: 1e-8).
    line_search_tol : float, optional
        Line search tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    m, n = A.shape

    # Positivity floor for iterates (absolute; avoids inf in Hessian terms)
    phi_floor = 1e-12

    # Reference spectrum (strictly positive)
    phi_0 = np.maximum(np.asarray(x0, dtype=float), 1e-300)
    phi_0_sum = np.sum(phi_0)

    # Normalize reference spectrum
    phi_0_norm = phi_0 / phi_0_sum

    # Work on the normalized simplex: scale the measurements by the same
    # factor so A @ (phi / phi_0_sum) == b / phi_0_sum is reachable. Without
    # this scaling the chi-squared term is evaluated against an inconsistent
    # normalization and the iteration converges to a wrongly scaled solution.
    b_work = np.asarray(b, dtype=float).ravel() / phi_0_sum
    b_safe = np.maximum(b_work, 1e-300)
    sigma = sigma_factor * b_safe
    S_b = np.diag(1.0 / (sigma**2))  # Inverse covariance matrix

    # Initialize solution
    phi_sol = phi_0_norm.copy()

    def objective(phi_sol: np.ndarray):
        """Loss function L = tau * D_KL(phi || phi_0_norm) + chi^2."""
        p = np.maximum(phi_sol, phi_floor)
        residual = A @ p - b_work
        kl = np.sum(p * (np.log(p / phi_0_norm + 1e-300)) - p + phi_0_norm)
        return tau * kl + residual @ S_b @ residual

    def compute_objective_and_gradients(phi_sol: np.ndarray):
        """Compute loss function and gradient for AMAXED-Regularization.

        Loss function: L = tau * D_KL(phi || phi_0) + 0.5 * chi^2

        where D_KL is the Kullback-Leibler divergence.
        """
        phi_sol_safe = np.maximum(phi_sol, phi_floor)

        # Forward model
        R_phi = A @ phi_sol_safe
        residual = R_phi - b_work

        # KL divergence gradient (cross-entropy term)
        # d/dphi [phi * log(phi/phi_0) - phi + phi_0] = log(phi/phi_0)
        kl_grad = np.log(phi_sol_safe / phi_0_norm + 1e-300)

        # Chi-squared gradient
        chi2_grad = 2 * A.T @ (S_b @ residual)

        # Combined gradient
        grad = tau * kl_grad + chi2_grad

        return grad

    def compute_Hessian(phi_sol: np.ndarray):
        """Compute Hessian matrix for Newton's method."""
        phi_sol_safe = np.maximum(phi_sol, phi_floor)

        # KL divergence Hessian: diag(1/phi)
        kl_hess = np.diag(1.0 / (phi_sol_safe + 1e-300))

        # Chi-squared Hessian: 2 * A^T S_b A
        chi2_hess = 2 * A.T @ S_b @ A

        # Combined Hessian
        Hessian = tau * kl_hess + chi2_hess

        return Hessian

    # Newton iteration with backtracking (Armijo) line search
    grad_norm = np.inf

    for iteration in range(max_iterations):
        grad = compute_objective_and_gradients(phi_sol)

        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break

        # Compute Hessian
        Hessian = compute_Hessian(phi_sol)

        # Solve for Newton direction
        try:
            delta_phi = np.linalg.solve(Hessian, -grad)
        except np.linalg.LinAlgError:
            # Add regularization if Hessian is singular
            reg_param = 1e-6 * np.max(np.abs(np.diag(Hessian)))
            Hessian_reg = Hessian + reg_param * np.eye(n)
            delta_phi = np.linalg.solve(Hessian_reg, -grad)

        # Backtracking line search on the objective value
        base_obj = objective(phi_sol)
        slope = float(np.dot(grad, delta_phi))
        beta = 1.0
        accepted = False
        for _ in range(30):
            new_phi = np.maximum(phi_sol + beta * delta_phi, phi_floor)
            if objective(new_phi) <= base_obj + 1e-4 * beta * slope:
                accepted = True
                break
            beta *= 0.5
        if not accepted:
            # Damped fallback step to keep making progress
            beta = 0.01

        # Update solution (positivity enforced via the floor)
        phi_sol = np.maximum(phi_sol + beta * delta_phi, phi_floor)

    # Scale back to original magnitude
    phi_sol = phi_sol * phi_0_sum

    iterations = iteration + 1
    converged = grad_norm < tolerance

    return phi_sol, iterations, converged


def unfold_amaxed_regularization(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma_factor: float = 0.1,
    tau: float = 1.0,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the AMAXED-Regularization algorithm.

    This method combines the advantages of AMAXED with Tikhonov regularization,
    providing stable convergence without requiring manual chi-squared tuning.

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
    tau : float, optional
        Regularization parameter (default: 1.0). Larger values favor
        solutions closer to the prior.
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
            solve_amaxed_regularization,
            sigma_factor=sigma_factor,
            tau=tau,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
        ),
        solve_kwargs={},
        method_name="AMAXED-Regularization",
        extra_output={
            "sigma_factor": sigma_factor,
            "tau": tau,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
