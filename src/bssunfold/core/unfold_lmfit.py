"""lmfit-based unfolding method with L1/L2/Elastic net regularization.

This module provides the core solve_lmfit solver and the unfold_lmfit
wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = [
    "solve_lmfit",
    "unfold_lmfit",
    "_residual_lasso",
    "_residual_ridge",
    "_residual_elastic",
]


# ---------------------------------------------------------------------------
# Residual functions for lmfit
# ---------------------------------------------------------------------------


def _residual_lasso(params, A, b, regularization, method, m):
    """Lasso (L1) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * np.sqrt(m) * x
        return np.concatenate([residual, reg_residual])
    return np.sum(residual ** 2) + regularization * np.sum(np.abs(x))


def _residual_ridge(params, A, b, regularization, method, m):
    """Ridge (L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * x
        return np.concatenate([residual, reg_residual])
    return np.sum(residual ** 2) + regularization * np.sum(x ** 2)


def _residual_elastic(params, A, b, regularization, regularization2, l1_weight, method, m):
    """Elastic net (L1 + L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        l1_residual = (
            np.sqrt(regularization * l1_weight) * np.sqrt(m) * np.abs(x)
        )
        l2_residual = np.sqrt(regularization2 * (1 - l1_weight)) * x
        reg_residual = np.concatenate([l1_residual, l2_residual])
        return np.concatenate([residual, reg_residual])
    l1_penalty = regularization * l1_weight * np.sum(np.abs(x))
    l2_penalty = regularization2 * (1 - l1_weight) * np.sum(x ** 2)
    return np.sum(residual ** 2) + l1_penalty + l2_penalty


_RESIDUAL_MAP = {
    "lasso": (_residual_lasso, ["regularization"]),
    "ridge": (_residual_ridge, ["regularization"]),
    "elastic": (_residual_elastic, ["regularization", "regularization2", "l1_weight"]),
}


def solve_lmfit(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using lmfit with L1/L2/Elastic regularization.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
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

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        Tuple of (solution, success, message, nfev).
    """
    try:
        import lmfit
    except ImportError as e:
        raise ImportError(
            "lmfit is required for unfold_lmfit. Install with: pip install lmfit"
        ) from e

    m = A.shape[1]

    params = lmfit.Parameters()
    for i in range(m):
        params.add(f"x{i}", value=max(x0[i], 1e-10), min=0.0)

    if model_name not in _RESIDUAL_MAP:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Choose from: elastic, lasso, ridge"
        )

    residual_func, arg_names = _RESIDUAL_MAP[model_name]
    residual_args = {
        "A": A,
        "b": b,
        "method": method,
        "m": m,
    }
    for name in arg_names:
        residual_args[name] = locals()[name]

    result = lmfit.minimize(
        residual_func,
        params,
        args=(A, b, regularization, method, m) if model_name in ("lasso", "ridge")
        else (A, b, regularization, regularization2, l1_weight, method, m),
        method=method,
    )

    spectrum = np.array([result.params[f"x{i}"].value for i in range(m)])
    return spectrum, result.success, result.message, result.nfev


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
    save_result: bool = False,
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
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    initial_spec_for_output = None

    def solve_wrapper(A, b, **kwargs):
        nonlocal initial_spec_for_output
        x0 = kwargs.pop('x0', None)
        if x0 is None:
            x0 = np.ones(A.shape[1]) * np.mean(b) / np.mean(A.sum(axis=1))
        initial_spec_for_output = x0.copy()
        x_opt, success, message, nfev = solve_lmfit(
            A, b, x0, method, model_name,
            regularization, regularization2, l1_weight
        )
        return x_opt, nfev, success

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

    if initial_spec_for_output is not None:
        result["initial_spectrum"] = initial_spec_for_output

    return result
