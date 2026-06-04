"""lmfit-based unfolding method with L1/L2/Elastic net regularization.

This module provides the `unfold_lmfit` function which wraps the lmfit
optimizer for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from .unfolding_methods import solve_lmfit
from ._base_unfolder import run_unfolding


def unfold_lmfit(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using lmfit with L1/L2/Elastic regularization.

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
        Initial spectrum guess. If None, uniform spectrum based on mean readings.
    method : str, optional
        lmfit solver name (leastsq, lbfgsb, etc.), default: "lbfgsb".
    model_name : str, optional
        Regularization model: elastic, lasso, ridge, default: "elastic".
    regularization : float, optional
        L1 regularization strength, default: 1e-4.
    regularization2 : float, optional
        L2 regularization strength for elastic net, default: 1e-4.
    l1_weight : float, optional
        L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5.
    calculate_errors : bool, optional
        Flag to calculate uncertainty via Monte-Carlo, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples for error estimation, default: 100.
    save_result : bool, optional
        If True, save result to internal history, default: True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing unfolding results.
    """
    # Build system for initial spectrum calculation
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    # Store initial spectrum for output
    initial_spec_for_output = None

    def solve_wrapper(A, b, **kwargs):
        nonlocal initial_spec_for_output
        x0 = kwargs.pop('x0')
        initial_spec_for_output = x0.copy()
        x_opt, success, message, nfev = solve_lmfit(
            A, b, x0, method, model_name,
            regularization, regularization2, l1_weight
        )
        return x_opt

    # Default initial spectrum based on mean readings
    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

    result = run_unfolding(
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
        method_name=f"lmfit ({method})",
        extra_output={
            "model_name": model_name,
            "regularization": regularization,
            "regularization2": regularization2 if model_name == "elastic" else None,
            "l1_weight": l1_weight if model_name == "elastic" else None,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    # Add initial spectrum to result if available
    if initial_spec_for_output is not None:
        result["initial_spectrum"] = initial_spec_for_output

    return result
