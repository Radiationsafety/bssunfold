"""MAXED unfolding method for neutron spectrum reconstruction.

Implements Maximum Entropy Deconvolution (Reginatto & Goldhagen 1999)
by minimising the primal function in log-space:

    f(x) = -S(x) + ½ Σ_j (b_j - (A@x)_j)² / σ_j²

where S(x) = -Σ_i x_i ln(x_i/x0_i) + Σ_i x_i - Σ_i x0_i is the Shannon
entropy relative to the reference spectrum x0.

The log transform y_i = ln(x_i) ensures positivity and good numerical
conditioning.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_maxed", "unfold_maxed"]


def solve_maxed(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    sigma_factor: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using MAXED (Maximum Entropy Deconvolution).

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
        Larger values → smoother spectrum (weaker data term).
    max_iterations : int, optional
        Maximum L-BFGS-B iterations (default: 5000).
    tolerance : float, optional
        Gradient convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    from scipy.optimize import minimize

    n_det, n_ene = A.shape

    # Measurement uncertainties (relative with floor).
    b_safe = np.maximum(b, 1e-300)
    sigma = sigma_factor * b_safe
    sigma2_inv = 1.0 / (sigma ** 2)

    # Reference spectrum (must be strictly positive).
    phi_0 = np.maximum(x0, 1e-300)
    log_phi_0 = np.log(phi_0)

    # Objective and gradient in log-space: y_i = ln(x_i).
    # f(y) = Σ [e^yi (yi - ln x0_i) - e^yi + x0_i]
    #       + ½ Σ (b_j - Σ A_ji e^yi)² / σ_j²

    def _f_and_g(y: np.ndarray):
        x = np.exp(y)
        folded = A @ x
        residual = b - folded

        # Entropy part of f
        f_ent = np.sum(x * (y - log_phi_0) - x + phi_0)

        # Chi-squared part of f
        f_chi = 0.5 * np.sum(residual ** 2 * sigma2_inv)

        # Gradient: df/dy_i = x_i * [ln(x_i/x0_i) - Aᵀ(r/σ²)_i]
        AT_resid_over_sigma2 = A.T @ (residual * sigma2_inv)
        g = x * (y - log_phi_0 - AT_resid_over_sigma2)

        return f_ent + f_chi, g

    y0 = np.log(phi_0)

    result = minimize(
        _f_and_g, y0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iterations, 'gtol': tolerance, 'ftol': 0},
    )

    x_opt = np.exp(result.x)
    iterations = result.nit if hasattr(result, 'nit') else 0
    converged = result.success or result.status == 0

    return x_opt, iterations, converged


def unfold_maxed(
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
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the MAXED algorithm.

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
        Larger values → smoother spectrum.
    max_iterations : int, optional
        Maximum L-BFGS-B iterations (default: 5000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
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
            solve_maxed,
            sigma_factor=sigma_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="MAXED",
        extra_output={
            "sigma_factor": sigma_factor,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
