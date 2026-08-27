"""AMAXED unfolding method for neutron spectrum reconstruction.

Implements Alternative Maximum Entropy Deconvolution (Wong 2024)
with reversed cross-entropy definition compared to MAXED.

The algorithm minimizes the Kullback-Leibler divergence while adhering to
the chi-squared constraint using Lagrangian multipliers and Newton's method.

References
----------
Wong, O. (2024). Modernising neutron spectrum unfolding for fusion applications.
PhD Thesis, Sheffield Hallam University. https://shura.shu.ac.uk/36014/
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_amaxed", "unfold_amaxed"]


def solve_amaxed(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.1,
    target_chi2: Optional[float] = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using AMAXED (Alternative MAXED).

    Uses reversed cross-entropy definition with Newton's method and line search.

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
    target_chi2 : float, optional
        Target chi-squared value. If None, automatically determined.
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
    # normalization and the Newton iteration diverges.
    b_work = np.asarray(b, dtype=float).ravel() / phi_0_sum
    b_safe = np.maximum(b_work, 1e-300)
    sigma = sigma_factor * b_safe
    S_b = np.diag(1.0 / (sigma**2))  # Inverse covariance matrix

    # Compute target chi-squared if not provided
    if target_chi2 is None:
        # Expected value of a chi-squared distributed residual vector
        # with m measurements
        target_chi2 = float(m)

    Omega = target_chi2

    # Initialize solution and Lagrange multiplier
    phi_sol = phi_0_norm.copy()
    mu = 1.0  # Initial Lagrange multiplier

    def compute_Lagrangian_gradients(phi_sol: np.ndarray, mu: float):
        """Compute Lagrangian function gradients (Equations C.7, C.8)."""
        phi_sol_safe = np.maximum(phi_sol, phi_floor)
        phi_sol_sum = np.sum(phi_sol_safe)

        # Precompute terms
        a = b_work @ S_b @ A  # Shape (n,)
        R_phi = A @ phi_sol_safe
        residual = R_phi - b_work
        b_term = phi_sol_safe @ (A.T @ S_b @ A)  # Shape (n,)

        # Gradient w.r.t. phi_sol (Equation C.7)
        ones_n = np.ones(n)
        grad_phi = (ones_n / phi_sol_sum - phi_0_norm / phi_sol_safe +
                    2 * mu * (b_term - a))

        # Gradient w.r.t. mu (chi-squared constraint)
        grad_mu = residual @ S_b @ residual - Omega

        return grad_phi, grad_mu

    def compute_Hessian(phi_sol: np.ndarray, mu: float):
        """Compute Hessian matrix (Equation C.8)."""
        phi_sol_safe = np.maximum(phi_sol, phi_floor)
        phi_sol_sum = np.sum(phi_sol_safe)

        # Precompute terms
        a = b_work @ S_b @ A
        b_term = phi_sol_safe @ (A.T @ S_b @ A)

        # Second derivative blocks
        ones_outer = np.outer(np.ones(n), np.ones(n))
        diag_term = np.diag(phi_0_norm / (phi_sol_safe ** 2))

        H_phi_phi = -ones_outer / (phi_sol_sum ** 2) + diag_term + \
                    2 * mu * (A.T @ S_b @ A)

        H_phi_mu = 2 * (b_term - a)
        H_mu_mu = np.array([[0.0]])

        # Assemble full Hessian
        H_top = np.column_stack([H_phi_phi, H_phi_mu])
        H_bottom = np.hstack([H_phi_mu.reshape(1, -1), H_mu_mu])
        Hessian = np.vstack([H_top, H_bottom])

        return Hessian

    # Newton iteration with backtracking line search on the KKT residual
    grad_norm = np.inf

    for iteration in range(max_iterations):
        grad_phi, grad_mu = compute_Lagrangian_gradients(phi_sol, mu)
        state_grad = np.concatenate([grad_phi, [grad_mu]])

        # Check convergence
        grad_norm = np.linalg.norm(state_grad)
        if grad_norm < tolerance:
            break

        # Compute Hessian
        Hessian = compute_Hessian(phi_sol, mu)

        # Solve for Newton direction
        try:
            delta_state = np.linalg.solve(Hessian, -state_grad)
        except np.linalg.LinAlgError:
            # Add regularization if Hessian is singular
            reg_param = 1e-6 * np.max(np.abs(np.diag(Hessian)))
            Hessian_reg = Hessian + reg_param * np.eye(n + 1)
            delta_state = np.linalg.solve(Hessian_reg, -state_grad)

        delta_phi = delta_state[:n]
        delta_mu = delta_state[n]

        # Bind current loop state explicitly so the closure cannot observe
        # later iterations' variables (flake8-bugbear B023).
        def kkt_residual(
            beta,
            phi_cur=phi_sol,
            d_phi=delta_phi,
            mu_cur=mu,
            d_mu=delta_mu,
        ):
            new_phi = np.maximum(phi_cur + beta * d_phi, phi_floor)
            new_mu = mu_cur + beta * d_mu
            g_phi, g_mu = compute_Lagrangian_gradients(new_phi, new_mu)
            return np.sqrt(np.dot(g_phi, g_phi) + g_mu * g_mu)

        # Backtracking line search: accept the step only if it reduces the
        # norm of the KKT residual (sufficient-decrease condition).
        beta = 1.0
        accepted = False
        for _ in range(30):
            if kkt_residual(beta) <= (1.0 - 1e-4 * beta) * grad_norm:
                accepted = True
                break
            beta *= 0.5
        if not accepted:
            # Damped fallback step to keep making progress
            beta = 0.01

        # Update state
        phi_sol = np.maximum(phi_sol + beta * delta_phi, phi_floor)
        mu = mu + beta * delta_mu

    # Scale back to original magnitude
    phi_sol = phi_sol * phi_0_sum

    iterations = iteration + 1
    converged = bool(grad_norm < tolerance)

    return phi_sol, iterations, converged


def unfold_amaxed(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma_factor: float = 0.1,
    target_chi2: Optional[float] = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    line_search_tol: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the AMAXED algorithm.

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
    target_chi2 : float, optional
        Target chi-squared value. If None, automatically determined.
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
            solve_amaxed,
            sigma_factor=sigma_factor,
            target_chi2=target_chi2,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
        ),
        solve_kwargs={},
        method_name="AMAXED",
        extra_output={
            "sigma_factor": sigma_factor,
            "target_chi2": target_chi2,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
