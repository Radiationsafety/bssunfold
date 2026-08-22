"""L-curve and automatic regularization parameter selection methods.

This module provides tools for automatic selection of regularization
parameters using L-curve analysis, discrepancy principle, generalized
cross-validation (GCV), and other criteria commonly used in inverse
problems. These methods are inspired by packages like RegularizationTools
in MATLAB and can be used with Tikhonov, TSVD, and other regularization
methods in bssunfold.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from scipy.linalg import svd
from scipy.optimize import minimize_scalar, brentq
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._matrix_utils import create_derivative_matrix

__all__ = [
    "solve_lcurve_tikhonov",
    "unfold_lcurve_tikhonov",
    "compute_l_curve",
    "find_lcorner",
    "gcv_tikhonov",
    "discrepancy_principle",
]


def _compute_residual_and_solution_norms(
    A: np.ndarray,
    b: np.ndarray,
    L: Optional[np.ndarray] = None,
    alpha_values: Optional[np.ndarray] = None,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute residual and solution norms for a range of alpha values.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    L : np.ndarray, optional
        Regularization matrix. If None, identity is used.
    alpha_values : np.ndarray, optional
        Custom alpha values to evaluate. If None, logarithmically spaced
        values are generated.
    n_points : int, optional
        Number of alpha points to evaluate (default: 50).
    
    Returns
    -------
    tuple
        (alpha_vals, residual_norms, solution_norms)
    """
    m, n = A.shape
    
    if L is None:
        L = np.eye(n)
    
    if alpha_values is None:
        # Generate logarithmically spaced alpha values
        s = svd(A, full_matrices=False, compute_uv=False)
        alpha_min = (s[-1] / s[0]) ** 2 * 0.1
        alpha_max = 10.0
        alpha_vals = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_points)
    else:
        alpha_vals = alpha_values
    
    residual_norms = []
    solution_norms = []
    
    for alpha in alpha_vals:
        # Solve regularized problem: min ||Ax - b||^2 + alpha^2 ||Lx||^2
        if np.allclose(L, np.eye(n)):
            # Standard form Tikhonov
            ATA = A.T @ A + alpha**2 * np.eye(n)
        else:
            # General form using GSVD or transformation to standard form
            # For simplicity, use direct solve
            reg_matrix = A.T @ A + alpha**2 * (L.T @ L)
            try:
                ATA = reg_matrix
            except:
                ATA = reg_matrix
        
        try:
            x = np.linalg.solve(ATA, A.T @ b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(ATA, A.T @ b, rcond=None)[0]
        
        residual_norms.append(np.linalg.norm(A @ x - b))
        solution_norms.append(np.linalg.norm(L @ x))
    
    return alpha_vals, np.array(residual_norms), np.array(solution_norms)


def compute_l_curve(
    A: np.ndarray,
    b: np.ndarray,
    L: Optional[np.ndarray] = None,
    alpha_values: Optional[np.ndarray] = None,
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute the L-curve for Tikhonov regularization.
    
    The L-curve is a log-log plot of the solution norm vs. the residual
    norm for different regularization parameters. The optimal parameter
    is typically at the "corner" of the L-shaped curve.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    L : np.ndarray, optional
        Regularization matrix. If None, identity is used.
    alpha_values : np.ndarray, optional
        Custom alpha values. If None, automatically generated.
    n_points : int, optional
        Number of points (default: 50).
    
    Returns
    -------
    dict
        Dictionary with 'alpha', 'residual_norm', 'solution_norm',
        'log_residual', 'log_solution'.
    """
    alpha_vals, res_norms, sol_norms = _compute_residual_and_solution_norms(
        A, b, L=L, alpha_values=alpha_values, n_points=n_points
    )
    
    return {
        "alpha": alpha_vals,
        "residual_norm": res_norms,
        "solution_norm": sol_norms,
        "log_residual": np.log10(np.maximum(res_norms, 1e-300)),
        "log_solution": np.log10(np.maximum(sol_norms, 1e-300)),
    }


def find_lcorner(
    log_residual: np.ndarray,
    log_solution: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    method: str = "curvature",
) -> Tuple[float, int]:
    """Find the corner of the L-curve.
    
    Parameters
    ----------
    log_residual : np.ndarray
        Logarithm of residual norms.
    log_solution : np.ndarray
        Logarithm of solution norms.
    alpha : np.ndarray, optional
        Alpha values corresponding to the norms.
    method : str, optional
        Method to find corner: 'curvature' (maximum curvature) or
        'triangle' (maximum distance from line) (default: 'curvature').
    
    Returns
    -------
    tuple
        (optimal_alpha, index) where index is the position in the arrays.
    """
    n = len(log_residual)
    if n < 3:
        return alpha[0] if alpha is not None else 1.0, 0
    
    if method == "curvature":
        # Compute curvature using finite differences
        curvature = []
        for i in range(1, n - 1):
            dx1 = log_solution[i] - log_solution[i - 1]
            dy1 = log_residual[i] - log_residual[i - 1]
            dx2 = log_solution[i + 1] - log_solution[i]
            dy2 = log_residual[i + 1] - log_residual[i]
            
            denom = (dx1**2 + dy1**2) ** 1.5 + (dx2**2 + dy2**2) ** 1.5 + 1e-300
            curv = abs(dx1 * dy2 - dx2 * dy1) / denom
            curvature.append(curv)
        
        corner_idx = np.argmax(curvature) + 1
    elif method == "triangle":
        # Find point with maximum distance from line connecting endpoints
        p1 = np.array([log_solution[0], log_residual[0]])
        p2 = np.array([log_solution[-1], log_residual[-1]])
        
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-300:
            return alpha[0] if alpha is not None else 1.0, 0
        
        line_unit = line_vec / line_len
        
        distances = []
        for i in range(n):
            p = np.array([log_solution[i], log_residual[i]])
            vec = p - p1
            dist = abs(np.cross(line_unit, vec))
            distances.append(dist)
        
        corner_idx = np.argmax(distances)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    optimal_alpha = alpha[corner_idx] if alpha is not None else 1.0
    return optimal_alpha, corner_idx


def gcv_tikhonov(
    A: np.ndarray,
    b: np.ndarray,
    L: Optional[np.ndarray] = None,
    alpha_values: Optional[np.ndarray] = None,
    n_points: int = 50,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find optimal alpha using Generalized Cross-Validation (GCV).
    
    GCV minimizes: V(alpha) = ||A x_alpha - b||^2 / [trace(I - A A^#)]^2
    where A^# is the influence matrix.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    L : np.ndarray, optional
        Regularization matrix. If None, identity is used.
    alpha_values : np.ndarray, optional
        Alpha values to search. If None, automatically generated.
    n_points : int, optional
        Number of points (default: 50).
    
    Returns
    -------
    tuple
        (optimal_alpha, alpha_vals, gcv_values)
    """
    m, n = A.shape
    
    if L is None:
        L = np.eye(n)
    
    # Use SVD for efficient computation
    U, s, Vh = svd(A, full_matrices=False)
    
    if alpha_values is None:
        alpha_min = (s[-1] / s[0]) ** 2 * 0.01 if len(s) > 1 else 1e-6
        alpha_max = 10.0
        alpha_vals = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_points)
    else:
        alpha_vals = alpha_values
    
    gcv_values = []
    beta = U.T @ b
    
    for alpha in alpha_vals:
        # Filter factors for Tikhonov
        filter_factors = s**2 / (s**2 + alpha**2)
        
        # Residual
        residual_sq = np.sum((1 - filter_factors) ** 2 * beta[: len(s)] ** 2)
        if len(beta) > len(s):
            residual_sq += np.sum(beta[len(s) :] ** 2)
        
        # Effective degrees of freedom
        eff_dof = np.sum(filter_factors)
        
        # GCV function
        denom = (m - eff_dof) ** 2 if (m - eff_dof) > 0 else 1e-300
        gcv = residual_sq / denom
        gcv_values.append(gcv)
    
    gcv_values = np.array(gcv_values)
    optimal_idx = np.argmin(gcv_values)
    optimal_alpha = alpha_vals[optimal_idx]
    
    return optimal_alpha, alpha_vals, gcv_values


def discrepancy_principle(
    A: np.ndarray,
    b: np.ndarray,
    noise_level: float,
    L: Optional[np.ndarray] = None,
    alpha_values: Optional[np.ndarray] = None,
    n_points: int = 50,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find alpha using the discrepancy principle.
    
    The discrepancy principle selects alpha such that ||A x_alpha - b|| ≈ δ√m
    where δ is the noise level.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    noise_level : float
        Relative noise level (e.g., 0.01 for 1% noise).
    L : np.ndarray, optional
        Regularization matrix. If None, identity is used.
    alpha_values : np.ndarray, optional
        Alpha values to search. If None, automatically generated.
    n_points : int, optional
        Number of points (default: 50).
    
    Returns
    -------
    tuple
        (optimal_alpha, alpha_vals, residual_norms)
    """
    m, n = A.shape
    target_residual = noise_level * np.linalg.norm(b) * np.sqrt(m)
    
    if L is None:
        L = np.eye(n)
    
    alpha_vals, res_norms, _ = _compute_residual_and_solution_norms(
        A, b, L=L, alpha_values=alpha_values, n_points=n_points
    )
    
    # Find alpha where residual crosses target
    diff = np.abs(res_norms - target_residual)
    optimal_idx = np.argmin(diff)
    optimal_alpha = alpha_vals[optimal_idx]
    
    return optimal_alpha, alpha_vals, res_norms


def solve_lcurve_tikhonov(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    alpha_method: str = "lcurve",
    alpha: Optional[float] = None,
    n_points: int = 50,
    noise_level: Optional[float] = None,
) -> np.ndarray:
    """Solve Tikhonov regularization with automatic alpha selection via L-curve.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (not used, for API compatibility).
    L : np.ndarray, optional
        Regularization matrix. If None, identity is used.
    alpha_method : str, optional
        Method for alpha selection: 'lcurve', 'gcv', 'discrepancy',
        'quasiopt' (default: 'lcurve').
    alpha : float, optional
        Fixed alpha value. If provided, overrides automatic selection.
    n_points : int, optional
        Number of points for L-curve evaluation (default: 50).
    noise_level : float, optional
        Noise level for discrepancy principle.
    
    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    m, n = A.shape
    
    if L is None:
        L = np.eye(n)
    
    # Determine optimal alpha
    if alpha is None:
        if alpha_method == "lcurve":
            lcurve_data = compute_l_curve(A, b, L=L, n_points=n_points)
            alpha_opt, _ = find_lcorner(
                lcurve_data["log_residual"],
                lcurve_data["log_solution"],
                alpha=lcurve_data["alpha"],
                method="curvature",
            )
        elif alpha_method == "gcv":
            alpha_opt, _, _ = gcv_tikhonov(A, b, L=L, n_points=n_points)
        elif alpha_method == "discrepancy":
            if noise_level is None:
                noise_level = 0.01  # Default 1% noise
            alpha_opt, _, _ = discrepancy_principle(
                A, b, noise_level, L=L, n_points=n_points
            )
        elif alpha_method == "quasiopt":
            # Quasi-optimality criterion
            alpha_vals, res_norms, sol_norms = _compute_residual_and_solution_norms(
                A, b, L=L, n_points=n_points
            )
            # Find minimum of ||x_alpha - x_{alpha/2}||
            diffs = []
            for i in range(1, len(alpha_vals)):
                # Simplified: use solution norm difference
                diffs.append(abs(sol_norms[i] - sol_norms[i - 1]))
            alpha_opt = alpha_vals[np.argmin(diffs) + 1] if diffs else alpha_vals[0]
        else:
            raise ValueError(f"Unknown alpha_method: {alpha_method}")
    else:
        alpha_opt = alpha
    
    # Solve with optimal alpha
    if np.allclose(L, np.eye(n)):
        ATA = A.T @ A + alpha_opt**2 * np.eye(n)
    else:
        ATA = A.T @ A + alpha_opt**2 * (L.T @ L)
    
    try:
        x = np.linalg.solve(ATA, A.T @ b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(ATA, A.T @ b, rcond=None)[0]
    
    return np.maximum(x, 0)


def unfold_lcurve_tikhonov(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    alpha_method: str = "lcurve",
    alpha: Optional[float] = None,
    n_points: int = 50,
    noise_level: Optional[float] = None,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Tikhonov with L-curve alpha selection.
    
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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess (for API compatibility).
    L : np.ndarray, optional
        Regularization matrix. If None, uses first-derivative for smoothness.
    alpha_method : str, optional
        Alpha selection method: 'lcurve', 'gcv', 'discrepancy', 'quasiopt'.
    alpha : float, optional
        Fixed alpha (overrides automatic selection).
    n_points : int, optional
        Number of L-curve points (default: 50).
    noise_level : float, optional
        Noise level for discrepancy principle.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
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
    # Default L is first derivative for smoothness
    if L is None:
        L = create_derivative_matrix(n_energy_bins, 1).toarray()
    
    x0_default = np.zeros(n_energy_bins)
    
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
            solve_lcurve_tikhonov,
            L=L,
            alpha_method=alpha_method,
            alpha=alpha,
            n_points=n_points,
            noise_level=noise_level,
        ),
        solve_kwargs={},
        method_name=f"LcurveTikhonov_{alpha_method}",
        extra_output={
            "alpha_method": alpha_method,
            "alpha_fixed": alpha,
            "n_points": n_points,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
