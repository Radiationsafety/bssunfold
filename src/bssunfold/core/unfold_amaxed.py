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

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

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
    from scipy.optimize import root_scalar
    
    m, n = A.shape
    
    # Measurement uncertainties
    b_safe = np.maximum(b, 1e-300)
    sigma = sigma_factor * b_safe
    S_b = np.diag(1.0 / (sigma**2))  # Inverse covariance matrix
    
    # Reference spectrum (strictly positive)
    phi_0 = np.maximum(x0, 1e-300)
    phi_0_sum = np.sum(phi_0)
    
    # Normalize reference spectrum
    phi_0_norm = phi_0 / phi_0_sum
    
    # Compute target chi-squared if not provided
    if target_chi2 is None:
        # Use degrees of freedom as default
        dof = max(m - n, 1)
        target_chi2 = float(dof)
    
    Omega = target_chi2
    
    # Initialize solution and Lagrange multiplier
    phi_sol = phi_0_norm.copy()
    mu = 1.0  # Initial Lagrange multiplier
    
    def compute_Lagrangian_gradients(phi_sol: np.ndarray, mu: float):
        """Compute Lagrangian function gradients (Equations C.7, C.8)."""
        phi_sol_safe = np.maximum(phi_sol, 1e-300)
        phi_sol_sum = np.sum(phi_sol_safe)
        
        # Normalized solution
        phi_sol_norm = phi_sol_safe / phi_sol_sum
        
        # Precompute terms
        a = b @ S_b @ A  # Shape (n,)
        R_phi = A @ phi_sol_safe
        residual = R_phi - b
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
        phi_sol_safe = np.maximum(phi_sol, 1e-300)
        phi_sol_sum = np.sum(phi_sol_safe)
        
        # Precompute terms
        R_phi = A @ phi_sol_safe
        residual = R_phi - b
        a = b @ S_b @ A
        b_term = phi_sol_safe @ (A.T @ S_b @ A)
        
        # Second derivative blocks
        ones_outer = np.outer(np.ones(n), np.ones(n))
        diag_term = np.diag(phi_0_norm / (phi_sol_safe ** 2))
        
        H_phi_phi = -ones_outer / (phi_sol_sum ** 2) + diag_term + \
                    2 * mu * (A.T @ S_b @ A)
        
        H_phi_mu = 2 * (b_term - a)
        H_mu_phi = H_phi_mu.T
        H_mu_mu = np.array([[0.0]])
        
        # Assemble full Hessian
        H_top = np.column_stack([H_phi_phi, H_phi_mu])
        H_bottom = np.column_stack([H_mu_phi.reshape(-1, 1), H_mu_mu])
        Hessian = np.row_stack([H_top, H_bottom])
        
        return Hessian
    
    # Newton iteration with line search
    state_vec = np.concatenate([phi_sol, [mu]])
    
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
        
        # Line search to find optimal step size beta
        def line_search_obj(beta):
            new_phi = phi_sol + beta * delta_phi
            new_mu = mu + beta * delta_mu
            new_phi = np.maximum(new_phi, 1e-300)
            grad_phi_new, grad_mu_new = compute_Lagrangian_gradients(new_phi, new_mu)
            return np.dot(grad_phi_new, delta_phi) + grad_mu_new * delta_mu
        
        # Find beta that makes derivative zero
        try:
            result = root_scalar(
                line_search_obj,
                bracket=[0, 2],
                method='brentq',
                xtol=line_search_tol
            )
            beta = result.root if result.converged else 1.0
        except (ValueError, RuntimeError):
            beta = 1.0
        
        # Ensure beta is in reasonable range
        beta = np.clip(beta, 0.1, 2.0)
        
        # Update state
        phi_sol = phi_sol + beta * delta_phi
        mu = mu + beta * delta_mu
        
        # Ensure positivity
        phi_sol = np.maximum(phi_sol, 1e-300)
    
    # Scale back to original magnitude
    phi_sol = phi_sol * phi_0_sum
    
    iterations = iteration + 1
    converged = grad_norm < tolerance
    
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
