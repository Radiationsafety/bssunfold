"""FISTA algorithm for neutron spectrum unfolding.

Implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
for solving regularized least squares problems with constraints.

Based on IRtools IRfista.m by Silvia Gazzola et al.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_config import get_logger
from ..utils.validators import validate_system

logger = get_logger("unfold_fista")


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def _project_nonnegative(x: np.ndarray) -> np.ndarray:
    """Project onto nonnegative orthant."""
    return np.maximum(x, 0)


def _project_box(
    x: np.ndarray, x_min: float = 0.0, x_max: float = np.inf
) -> np.ndarray:
    """Project onto box constraints [x_min, x_max]."""
    return np.clip(x, x_min, x_max)


def unfold_fista(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Optional[callable] = None,
    readings: Optional[Dict[str, float]] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    regularization: float = 0.0,
    l1_penalty: float = 0.0,
    tv_penalty: float = 0.0,
    nonnegativity: bool = True,
    x_min: float = 0.0,
    x_max: float = np.inf,
    noise_level: Optional[float] = None,
    eta: float = 1.01,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using FISTA algorithm.

    The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) is an
    accelerated proximal gradient method that achieves O(1/k^2) convergence
    rate for convex optimization problems. It can handle L1 regularization
    (sparsity), TV regularization, and box constraints.

    Parameters
    ----------
    detector_names : List[str]
        Names of detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Sensitivity matrix as dictionary.
    cc_icrp116 : Dict[str, np.ndarray]
        Dose conversion coefficients.
    save_result_callback : callable, optional
        Callback to save results.
    readings : Dict[str, float], optional
        Detector readings.
    initial_spectrum : np.ndarray, optional
        Initial guess for spectrum.
    max_iterations : int, optional
        Maximum number of iterations (default: 500).
    tolerance : float, optional
        Convergence tolerance (default: 1e-8).
    regularization : float, optional
        Tikhonov regularization parameter (default: 0.0).
    l1_penalty : float, optional
        L1 regularization penalty parameter for sparsity (default: 0.0).
    tv_penalty : float, optional
        Total variation penalty parameter (default: 0.0).
    nonnegativity : bool, optional
        Apply nonnegativity constraints (default: True).
    x_min : float, optional
        Lower bound for solution (default: 0.0).
    x_max : float, optional
        Upper bound for solution (default: inf).
    noise_level : float, optional
        Relative noise level for discrepancy principle stopping.
    eta : float, optional
        Safety factor for discrepancy principle (default: 1.01).
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
    if random_state is not None:
        np.random.seed(random_state)

    # Build system matrix A and measurement vector b
    selected = [name for name in detector_names if name in readings]
    if len(selected) == 0:
        raise ValueError("No valid readings provided")

    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    A, b, _ = validate_system(A, b, max_iterations=max_iterations, tolerance=tolerance)
    _, n_energy = A.shape

    # Initial guess
    if initial_spectrum is None:
        x = np.ones(n_energy) * np.mean(b) / max(np.mean(A), 1e-10)
    else:
        x = initial_spectrum.copy()

    # Ensure nonnegativity if required
    if nonnegativity:
        x = _project_nonnegative(x)

    # FISTA variables
    y = x.copy()
    t = 1.0

    # Precompute Lipschitz constant L = ||A||^2
    # Use power iteration for large matrices
    if n_energy < 100:
        L = np.linalg.norm(A, ord=2) ** 2
    else:
        # Power iteration approximation
        v = np.random.randn(n_energy)
        v = v / np.linalg.norm(v)
        for _ in range(20):
            u = A @ v
            v_new = A.T @ u
            v = v_new / np.linalg.norm(v_new)
        L = np.linalg.norm(A @ v) ** 2 / np.linalg.norm(v) ** 2

    L = max(L, 1e-10)
    step_size = 1.0 / L

    # TV difference matrix
    if tv_penalty > 0:
        D = np.zeros((n_energy - 1, n_energy))
        for i in range(n_energy - 1):
            D[i, i] = -1
            D[i, i + 1] = 1

    # Storage for convergence monitoring
    residuals = []
    obj_values = []

    # Discrepancy principle threshold
    if noise_level is not None:
        discrepancy_threshold = eta * noise_level * np.linalg.norm(b)
    else:
        discrepancy_threshold = None

    logger.info(f"Starting FISTA with L={L:.4e}, step_size={step_size:.4e}")

    for k in range(max_iterations):
        x_old = x.copy()

        # Gradient step: y - (1/L) * A^T (A y - b)
        residual = A @ y - b
        gradient = A.T @ residual

        # Add Tikhonov regularization gradient if needed
        if regularization > 0:
            gradient += regularization * y

        # Add TV regularization gradient if needed
        if tv_penalty > 0:
            grad_tv = D.T @ (D @ y)
            gradient += tv_penalty * grad_tv

        x_temp = y - step_size * gradient

        # Proximal operator
        if l1_penalty > 0:
            # Soft thresholding for L1
            x_temp = _soft_threshold(x_temp, step_size * l1_penalty)

        # Apply constraints
        if nonnegativity:
            x_temp = _project_nonnegative(x_temp)

        if x_max < np.inf:
            x_temp = _project_box(x_temp, x_min, x_max)

        x = x_temp

        # FISTA acceleration step
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        # Monitor convergence
        current_residual = np.linalg.norm(A @ x - b)
        residuals.append(current_residual)

        # Compute objective function value
        data_term = 0.5 * np.sum((A @ x - b) ** 2)
        reg_term = 0.5 * regularization * np.sum(x**2)
        l1_term = l1_penalty * np.sum(np.abs(x))
        tv_term = 0.0
        if tv_penalty > 0:
            tv_term = tv_penalty * np.sum(np.abs(D @ x))
        obj_value = data_term + reg_term + l1_term + tv_term
        obj_values.append(obj_value)

        # Check convergence
        rel_change = np.linalg.norm(x - x_old) / max(
            np.linalg.norm(x_old), 1e-10
        )
        if rel_change < tolerance:
            logger.info(f"FISTA converged at iteration {k + 1}")
            break

        # Discrepancy principle stopping
        if (
            discrepancy_threshold is not None
            and current_residual <= discrepancy_threshold
        ):
            logger.info(f"Discrepancy principle satisfied at iteration {k + 1}")
            break

    # Ensure nonnegativity of final result
    spectrum = np.maximum(x, 0)

    # Compute dose rates
    from .dose_calculation import calculate_dose_rates

    doserates = calculate_dose_rates(spectrum, cc_icrp116)

    # Compute effective readings and residual
    computed_readings = A @ spectrum
    residual = b - computed_readings

    # Build output dictionary
    result = {
        "energy": E_MeV.copy(),
        "spectrum": spectrum.copy(),
        "spectrum_absolute": spectrum.copy(),
        "effective_readings": {
            name: float(val) for name, val in zip(selected, computed_readings)
        },
        "residual": residual.copy(),
        "residual_norm": float(np.linalg.norm(residual)),
        "method": "FISTA",
        "doserates": doserates,
        "iterations": k + 1,
        "final_residual_norm": float(residuals[-1])
        if residuals
        else float(np.linalg.norm(residual)),
        "objective_values": obj_values,
        "parameters": {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "regularization": regularization,
            "l1_penalty": l1_penalty,
            "tv_penalty": tv_penalty,
            "nonnegativity": nonnegativity,
        },
    }

    # Calculate uncertainties if requested
    if calculate_errors:
        rng = np.random.default_rng(random_state)
        spectra_mc = []
        for _ in range(n_montecarlo):
            # Perturb readings based on assumed uncertainty
            b_perturbed = b * (1.0 + 0.05 * rng.standard_normal(len(b)))
            # Run a few FISTA iterations with perturbed data
            x_mc = spectrum.copy()
            for _ in range(min(50, max_iterations)):
                grad = A.T @ (A @ x_mc - b_perturbed)
                if regularization > 0:
                    grad += regularization * x_mc
                x_mc = _project_nonnegative(x_mc - step_size * grad)
                if l1_penalty > 0:
                    x_mc = _soft_threshold(x_mc, step_size * l1_penalty)
            spectra_mc.append(x_mc)

        spectra_mc = np.array(spectra_mc)
        std_spectrum = np.std(spectra_mc, axis=0)
        result["spectrum_uncertainty"] = std_spectrum
        result["calculate_errors"] = True
        result["n_montecarlo"] = n_montecarlo

    # Save result if requested
    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
