"""Hybrid GMRES algorithm for neutron spectrum unfolding.

Implements hybrid GMRES regularization method combining GMRES iteration
with Tikhonov regularization on projected problems.

Based on IRtools IRhybrid_gmres.m by Silvia Gazzola et al.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_config import get_logger

logger = get_logger("unfold_hybrid_gmres")


def _gcv_function(
    lambda_val: float, B_k: np.ndarray, beta: np.ndarray
) -> float:
    """Compute Generalized Cross Validation function value."""
    # Use number of columns in B_k as the dimension of the solution space
    k = B_k.shape[1]
    if lambda_val < 1e-14:
        # Handle near-zero lambda
        try:
            x_lambda = np.linalg.lstsq(B_k, beta[:k], rcond=None)[0]
            residual = beta[:k] - B_k @ x_lambda
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
    else:
        # Add regularization to bidiagonal matrix
        B_reg = np.vstack([B_k, lambda_val * np.eye(k)])
        rhs = np.concatenate([beta, np.zeros(k)])
        try:
            x_lambda = np.linalg.lstsq(B_reg, rhs, rcond=None)[0]
            residual = beta - B_k @ x_lambda
        except (np.linalg.LinAlgError, ValueError):
            return 1e10

    # GCV function: ||A x - b||^2 / (trace(I - A A^+))^2
    numerator = np.sum(residual**2)

    # Approximate trace of influence matrix
    # For small k, compute directly
    if k < 50:
        try:
            H = B_k @ np.linalg.pinv(B_k)
            denominator = (len(beta) - np.trace(H)) ** 2
        except (np.linalg.LinAlgError, ValueError):
            denominator = 1.0
    else:
        # Hutchinson trace estimator approximation
        denominator = max(len(beta) * 0.1, 1.0)

    if denominator < 1e-10:
        return 1e10

    return numerator / denominator


def unfold_hybrid_gmres(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Optional[callable] = None,
    readings: Optional[Dict[str, float]] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    regularization_method: str = "gcv",
    regularization: float = 0.0,
    noise_level: Optional[float] = None,
    eta: float = 1.01,
    reorthogonalization: bool = True,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Hybrid GMRES method.

    The hybrid GMRES method combines the GMRES iterative solver with
    Tikhonov regularization applied to the projected problem at each
    iteration. The regularization parameter is selected automatically
    using GCV or discrepancy principle.

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
        Maximum Krylov dimension (default: 100).
    regularization_method : str, optional
        Method for selecting regularization parameter:
        'gcv', 'modgcv', 'discrep' (default: 'gcv').
    regularization : float, optional
        Fixed regularization parameter (used if not auto-selected).
    noise_level : float, optional
        Relative noise level for discrepancy principle.
    eta : float, optional
        Safety factor for discrepancy principle (default: 1.01).
    reorthogonalization : bool, optional
        Apply full reorthogonalization (default: True).
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

    n_detectors, n_energy = A.shape
    max_krylov = min(max_iterations, n_detectors, n_energy)

    # Initial residual and first basis vector
    if initial_spectrum is None:
        x0 = np.zeros(n_energy)
    else:
        x0 = initial_spectrum.copy()

    r0 = b - A @ x0
    beta = np.linalg.norm(r0)

    if beta < 1e-14:
        logger.warning("Initial residual is nearly zero")
        spectrum = np.maximum(x0, 0)
        computed_readings = A @ spectrum
        from .dose_calculation import calculate_dose_rates

        doserates = calculate_dose_rates(spectrum, cc_icrp116)
        return {
            "energy": E_MeV.copy(),
            "spectrum": spectrum,
            "spectrum_absolute": spectrum,
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, computed_readings)
            },
            "residual": b - computed_readings,
            "residual_norm": 0.0,
            "method": "Hybrid_GMRES",
            "doserates": doserates,
            "iterations": 0,
        }

    v1 = r0 / beta

    # Arnoldi decomposition storage
    V = np.zeros((n_detectors, max_krylov + 1))
    V[:, 0] = v1

    # Storage for solutions and parameters
    gcv_values = []
    reg_params = []
    solution_norms = []
    residual_norms = []
    best_solution = None
    best_gcv = np.inf
    stop_iteration = max_krylov

    logger.info(f"Starting Hybrid GMRES with max_krylov={max_krylov}")

    actual_k = max_krylov
    # For least squares problem min ||A x - b||, we apply GMRES to normal equations
    # A^T A x = A^T b, or use LSQR/Lanczos bidiagonalization approach
    # Here we use Golub-Kahan bidiagonalization for better numerical stability

    # Bidiagonalization: A @ U = V @ B, A.T @ V = U @ B.T
    # where U is n_energy x k, V is n_detectors x k, B is k x k bidiagonal

    U = np.zeros((n_energy, max_krylov + 1))  # Solution space basis
    V_mat = np.zeros((n_detectors, max_krylov + 1))  # Data space basis
    alpha = np.zeros(max_krylov)  # Diagonal of B
    beta = np.zeros(max_krylov + 1)  # Off-diagonal of B

    # Initialize with residual
    r0 = b - A @ x0
    beta[0] = np.linalg.norm(r0)

    if beta[0] < 1e-14:
        logger.warning("Initial residual is nearly zero")
        spectrum = np.maximum(x0, 0)
        computed_readings = A @ spectrum
        from .dose_calculation import calculate_dose_rates

        doserates = calculate_dose_rates(spectrum, cc_icrp116)
        return {
            "energy": E_MeV.copy(),
            "spectrum": spectrum,
            "spectrum_absolute": spectrum,
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, computed_readings)
            },
            "residual": b - computed_readings,
            "residual_norm": 0.0,
            "method": "Hybrid_GMRES",
            "doserates": doserates,
            "iterations": 0,
        }

    V_mat[:, 0] = r0 / beta[0]
    u1 = A.T @ V_mat[:, 0]
    alpha[0] = np.linalg.norm(u1)

    if alpha[0] < 1e-14:
        logger.warning("Breakdown at first iteration")
        spectrum = np.maximum(x0, 0)
        computed_readings = A @ spectrum
        from .dose_calculation import calculate_dose_rates

        doserates = calculate_dose_rates(spectrum, cc_icrp116)
        return {
            "energy": E_MeV.copy(),
            "spectrum": spectrum,
            "spectrum_absolute": spectrum,
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, computed_readings)
            },
            "residual": b - computed_readings,
            "residual_norm": beta[0],
            "method": "Hybrid_GMRES",
            "doserates": doserates,
            "iterations": 0,
        }

    U[:, 0] = u1 / alpha[0]

    for k in range(max_krylov):
        # Golub-Kahan bidiagonalization step
        # v_{k+1} = A @ u_k - alpha_k * v_k
        v_new = A @ U[:, k] - alpha[k] * V_mat[:, k]

        # Orthogonalize against previous V columns
        if reorthogonalization and k > 0:
            for j in range(k):
                v_new -= np.dot(V_mat[:, j], v_new) * V_mat[:, j]

        beta[k + 1] = np.linalg.norm(v_new)

        # Check for breakdown
        if beta[k + 1] < 1e-14:
            logger.info(f"Bidiagonalization breakdown at iteration {k + 1}")
            actual_k = k + 1
            break

        V_mat[:, k + 1] = v_new / beta[k + 1]

        # u_{k+1} = A.T @ v_{k+1} - beta_{k+1} * u_k
        u_new = A.T @ V_mat[:, k + 1] - beta[k + 1] * U[:, k]

        # Orthogonalize against previous U columns
        if reorthogonalization and k > 0:
            for j in range(k):
                u_new -= np.dot(U[:, j], u_new) * U[:, j]

        if k < max_krylov - 1:
            alpha[k + 1] = np.linalg.norm(u_new)

            if alpha[k + 1] < 1e-14:
                logger.info(f"Alpha breakdown at iteration {k + 1}")
                actual_k = k + 1
                break

            U[:, k + 1] = u_new / alpha[k + 1]

        # Current Krylov dimension
        current_k = k + 1

        # Build bidiagonal projection matrix B_k (current_k+1 x current_k)
        # [ alpha_0              ]
        # [ beta_1  alpha_1      ]
        # [   0   beta_2 alpha_2 ]
        # [ ...                ...]
        B_k = np.zeros((current_k + 1, current_k))
        for i in range(current_k):
            B_k[i, i] = alpha[i]
            if i < current_k:
                B_k[i + 1, i] = beta[i + 1]

        # Right-hand side: beta_0 * e1
        rhs_proj = np.zeros(current_k + 1)
        rhs_proj[0] = beta[0]

        # Select regularization parameter
        if regularization_method in ("gcv", "modgcv"):
            # Search for optimal lambda using GCV
            lambda_candidates = np.logspace(-10, 2, 50)
            best_lambda = regularization
            best_gcv_val = np.inf

            for lam in lambda_candidates:
                gcv_val = _gcv_function(lam, B_k, rhs_proj)
                if gcv_val < best_gcv_val:
                    best_gcv_val = gcv_val
                    best_lambda = lam

            lambda_k = best_lambda
            gcv_values.append(best_gcv_val)

            # Check for GCV stabilization (stopping criterion)
            if len(gcv_values) >= 3:
                recent_min = min(gcv_values[-3:])
                if best_gcv_val > recent_min * 1.01:  # GCV minimum increased
                    stop_iteration = current_k
                    logger.info(
                        f"GCV minimum increased at iteration {current_k}"
                    )

        elif regularization_method == "discrep" and noise_level is not None:
            # Discrepancy principle
            threshold = eta * noise_level * np.linalg.norm(b)
            lambda_k = regularization
            # Simple binary search for lambda satisfying discrepancy
            for _ in range(20):
                L_reg = np.vstack([B_k, lambda_k * np.eye(B_k.shape[1])])
                rhs_reg = np.concatenate([rhs_proj, np.zeros(B_k.shape[1])])
                try:
                    y_lambda = np.linalg.lstsq(L_reg, rhs_reg, rcond=None)[0]
                    x_lambda = x0 + U[:, :current_k] @ y_lambda
                    res_norm = np.linalg.norm(A @ x_lambda - b)

                    if res_norm > threshold:
                        lambda_k *= 2
                    else:
                        lambda_k /= 2
                except (np.linalg.LinAlgError, ValueError):
                    break
        else:
            lambda_k = regularization

        reg_params.append(lambda_k)

        # Solve regularized projected problem
        L_reg = np.vstack([B_k, lambda_k * np.eye(B_k.shape[1])])
        rhs_reg = np.concatenate([rhs_proj, np.zeros(B_k.shape[1])])

        try:
            y_lambda = np.linalg.lstsq(L_reg, rhs_reg, rcond=None)[0]
            x_lambda = x0 + U[:, :current_k] @ y_lambda

            # Compute residual norm
            res_norm = np.linalg.norm(A @ x_lambda - b)
            residual_norms.append(res_norm)
            solution_norms.append(np.linalg.norm(x_lambda))

            # Track best solution by GCV
            if best_gcv_val < best_gcv:
                best_gcv = best_gcv_val
                best_solution = x_lambda.copy()

        except np.linalg.LinAlgError:
            logger.warning(f"LinAlgError at iteration {current_k}")
            continue

    # Use best solution or last computed
    if best_solution is None:
        spectrum = (
            np.maximum(x0 + U[:, :actual_k] @ y_lambda, 0)
            if "y_lambda" in locals()
            else np.maximum(x0, 0)
        )
    else:
        spectrum = np.maximum(best_solution, 0)

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
        "method": "Hybrid_GMRES",
        "doserates": doserates,
        "iterations": stop_iteration,
        "regularization_parameters": reg_params,
        "gcv_values": gcv_values,
        "solution_norms": solution_norms,
        "residual_norms_history": residual_norms,
        "parameters": {
            "max_iterations": max_iterations,
            "regularization_method": regularization_method,
            "noise_level": noise_level,
            "eta": eta,
            "reorthogonalization": reorthogonalization,
        },
    }

    # Calculate uncertainties if requested
    if calculate_errors:
        rng = np.random.default_rng(random_state)
        spectra_mc = []
        for _ in range(n_montecarlo):
            b_perturbed = b * (1.0 + 0.05 * rng.standard_normal(len(b)))
            # Quick projection with fixed lambda
            x_mc = np.zeros(n_energy)
            r0_mc = b_perturbed - A @ x_mc
            beta_mc = np.linalg.norm(r0_mc)
            if beta_mc > 1e-14:
                v1_mc = r0_mc / beta_mc
                V_mc = np.zeros((n_detectors, min(10, max_krylov)))
                V_mc[:, 0] = v1_mc
                for j in range(min(9, max_krylov - 1)):
                    w_mc = A.T @ V_mc[:, j]
                    for i in range(j + 1):
                        coef = np.dot(V_mc[:, i], w_mc)
                        w_mc -= coef * V_mc[:, i]
                    norm_w = np.linalg.norm(w_mc)
                    if norm_w > 1e-14:
                        V_mc[:, j + 1] = w_mc / norm_w
                H_mc = V_mc[:, : min(5, max_krylov)]
                try:
                    y_mc = np.linalg.lstsq(
                        H_mc, beta_mc * np.ones(min(5, max_krylov)), rcond=None
                    )[0]
                    x_mc = V_mc[:, : min(5, max_krylov)] @ y_mc
                except (np.linalg.LinAlgError, ValueError):
                    pass
            spectra_mc.append(np.maximum(x_mc, 0))

        spectra_mc = np.array(spectra_mc)
        std_spectrum = np.std(spectra_mc, axis=0)
        result["spectrum_uncertainty"] = std_spectrum
        result["calculate_errors"] = True
        result["n_montecarlo"] = n_montecarlo

    # Save result if requested
    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
