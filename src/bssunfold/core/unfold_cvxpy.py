"""CVXPY-based unfolding method for neutron spectrum reconstruction.

This module provides the `unfold_cvxpy` function which wraps the convex
optimization solver for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ..platform_check import get_recommended_solver
from .regularization import select_regularization_parameter
from .unfolding_methods import solve_cvxpy
from ._base_unfolder import run_unfolding


def unfold_cvxpy(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    regularization: float = 1e-4,
    norm: int = 2,
    solver: str = "default",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using convex optimization (cvxpy).

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
        Initial spectrum guess.
    regularization : float, optional
        Regularization parameter (default: 1e-4).
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    solver : str, optional
        Solver to use ('ECOS' or 'default').
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: True).
    regularization_method : str, optional
        Method for selecting regularization parameter.
    noise_var : float, optional
        Noise variance for discrepancy principle.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    if solver == "default":
        solver = get_recommended_solver()

    # Build system for regularization selection
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    # Handle regularization selection
    if regularization_method == "manual":
        alpha = regularization
        selected_lambda = alpha
    elif regularization_method == "cosine":
        if initial_spectrum is None:
            raise ValueError(
                "For 'cosine' method, initial_spectrum must be provided."
            )
        initial_spectrum_norm = np.maximum(initial_spectrum, 0)
        if len(initial_spectrum_norm) == n_energy_bins:
            initial_spectrum_norm = initial_spectrum_norm / np.linalg.norm(initial_spectrum_norm)
        else:
            raise ValueError(
                f"Initial spectrum length ({len(initial_spectrum)}) "
                f"must match number of energy bins ({n_energy_bins})"
            )
        alpha = select_regularization_parameter(
            A, b, method="cosine", initial_spectrum=initial_spectrum_norm
        )
        selected_lambda = alpha
    else:
        selected_lambda = select_regularization_parameter(
            A, b, method=regularization_method, noise_var=noise_var
        )
        alpha = selected_lambda

    def solve_wrapper(A, b, **kwargs):
        # cvxpy doesn't use x0, but we need to accept it for consistency
        kwargs.pop('x0', None)
        x = solve_cvxpy(A, b, alpha, norm, solver)
        return x

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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="cvxpy",
        extra_output={
            "norm": norm,
            "solver": solver,
            "regularization_method": regularization_method,
            "selected_regularization": float(alpha),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
