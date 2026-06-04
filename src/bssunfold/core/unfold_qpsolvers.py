"""QP solvers-based unfolding method with regularization selection.

This module provides the `unfold_qpsolvers` function which wraps the quadratic
programming solver with various regularization selection methods.
"""

import warnings
import numpy as np
from typing import Dict, Optional, Any, List

from .regularization import select_regularization_parameter
from .unfolding_methods import solve_qpsolvers
from ._base_unfolder import run_unfolding


def unfold_qpsolvers(
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
    solver: str = "osqp",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using qpsolvers with regularization selection.

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
        Initial spectrum guess.
    regularization : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    solver : str, optional
        QP solver name, default: 'osqp'.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default: 100.
    save_result : bool, optional
        Save result to history, default: True.
    regularization_method : str, optional
        Method for selecting regularization parameter.
        Options: 'manual', 'cosine', 'gcv', 'lcurve', 'dp'.
    noise_var : float, optional
        Noise variance for discrepancy principle ('dp' method).
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2), default: 0.
    smoothness_weight : float, optional
        Weight for smoothness term, default: 1.0.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
    """
    # Build system for regularization selection
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)
    n = A.shape[1]

    # Select regularization parameter
    if regularization_method == "manual":
        alpha = regularization
        selected_lambda = alpha
    elif regularization_method == "cosine":
        if initial_spectrum is None:
            raise ValueError(
                "For 'cosine' regularization method, "
                "initial_spectrum must be provided."
            )
        if norm != 2:
            warnings.warn(
                f"Cosine regularization selection method assumes L2 "
                f"norm, but norm={norm} was requested. Using L2 for "
                f"selection."
            )
        initial_spectrum_norm = np.maximum(initial_spectrum, 0)
        if len(initial_spectrum_norm) != n_energy_bins:
            raise ValueError(
                f"Initial spectrum length ({len(initial_spectrum)}) "
                f"must match number of energy bins ({n_energy_bins})"
            )
        
        alphas = np.logspace(-9, 2, 100)
        cosine_similarities = []

        for alpha_val in alphas:
            x_temp = solve_qpsolvers(
                A, b, alpha_val, 2, solver,
                x_init=initial_spectrum_norm,
                smoothness_order=smoothness_order,
                smoothness_weight=smoothness_weight,
            )
            if x_temp is not None:
                norm_temp = np.linalg.norm(x_temp)
                if norm_temp > 0:
                    cos_sim = np.dot(x_temp, initial_spectrum_norm) / (norm_temp * np.linalg.norm(initial_spectrum_norm))
                    cosine_similarities.append(cos_sim)
                else:
                    cosine_similarities.append(-1)
            else:
                cosine_similarities.append(-1)

        optimal_idx = int(np.argmax(cosine_similarities))
        selected_lambda = alphas[optimal_idx]
        alpha = selected_lambda
        print(
            f"Selected regularization (method=cosine): "
            f"{selected_lambda:.3e}"
        )
    else:
        if norm != 2:
            warnings.warn(
                f"Automatic regularization selection methods assume L2 "
                f"norm, but norm={norm} was requested. Using L2 for "
                f"selection."
            )
        try:
            selected_lambda = select_regularization_parameter(
                A, b, method=regularization_method, noise_var=noise_var
            )
        except Exception as e:
            raise ValueError(
                f"Regularization selection failed: {e}. "
                "Consider using manual regularization."
            )
        alpha = selected_lambda
        print(
            f"Selected regularization (method={regularization_method}): "
            f"{selected_lambda:.3e}"
        )

    def solve_wrapper(A, b, **kwargs):
        # qpsolvers doesn't use x0, but we need to accept it for consistency
        kwargs.pop('x0', None)
        x = solve_qpsolvers(
            A, b, alpha, norm, solver,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
        )
        if x is None:
            x = np.zeros(A.shape[1])
            warnings.warn("Solution not found, returning zero spectrum.")
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
        method_name=f"qpsolvers_{solver}",
        extra_output={
            "norm": norm,
            "regularization": regularization,
            "regularization_method": regularization_method,
            "selected_regularization": float(selected_lambda),
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
