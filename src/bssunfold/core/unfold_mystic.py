"""Mystic-based unfolding method with regularization selection.

This module provides the core solve_mystic solver and the unfold_mystic
wrapper with various regularization selection methods. The optimization is
performed with the ``mystic`` constrained-optimization framework
(https://pypi.org/project/mystic/), using a direct-search solver on the
penalized least-squares objective.
"""

import warnings
import numpy as np
from typing import Dict, Optional, Any, List

from ._matrix_utils import create_derivative_matrix
from .regularization import select_regularization_parameter
from ._base_unfolder import run_unfolding, make_solve_wrapper, _build_system

__all__ = ["solve_mystic", "unfold_mystic"]

# Supported mystic minimal-interface solvers
_SUPPORTED_SOLVERS = ("fmin", "fmin_powell", "diffev", "diffev2")


def _solver_function(solver: str):
    """Import and return the mystic minimal-interface solver callable."""
    from mystic.solvers import fmin, fmin_powell, diffev, diffev2

    solver_functions = {
        "fmin": fmin,
        "fmin_powell": fmin_powell,
        "diffev": diffev,
        "diffev2": diffev2,
    }
    return solver_functions[solver]


def _nonneg_condition(x: np.ndarray) -> float:
    """Scalar measure of non-negativity violation (x >= 0)."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.maximum(-x, 0.0)))


def _build_bounds(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray]) -> list:
    """Build non-negativity bounds for population-based solvers."""
    n = A.shape[1]
    x0_arr = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    col_norm = float(np.max(np.linalg.norm(A, axis=0))) or 1.0
    scale = float(np.max(np.abs(b))) / max(col_norm, 1e-12)
    ub = np.maximum(2.0 * np.abs(x0_arr), scale)
    ub = np.maximum(ub, 1e-3)
    return [(0.0, float(ub_i)) for ub_i in ub]


def solve_mystic(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    solver: str = "fmin_powell",
    x0: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    maxfun: Optional[int] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
) -> np.ndarray:
    """Solve unfolding problem using mystic.

    Minimizes ``||A x - b||^2 + alpha * ||x||_norm`` with the non-negativity
    constraint ``x >= 0`` imposed via a quadratic penalty.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    norm : int, optional
        Norm type (1 for L1, 2 for L2).
    solver : str, optional
        Mystic solver name: 'fmin', 'fmin_powell', 'diffev' or 'diffev2'
        (default: 'fmin_powell').
    x0 : np.ndarray, optional
        Initial values. Defaults to the zero vector.
    maxiter : int, optional
        Maximum number of solver iterations.
    maxfun : int, optional
        Maximum number of function evaluations.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2).
    smoothness_weight : float, optional
        Weight for the smoothness term.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,). Returns a zero vector if solving failed.
    """
    try:
        from mystic.penalty import quadratic_inequality
    except ImportError as e:
        raise ImportError(
            "mystic is required for unfold_mystic. "
            "Install with: pip install mystic"
        ) from e

    if solver not in _SUPPORTED_SOLVERS:
        warnings.warn(
            f"Solver '{solver}' not supported. "
            f"Available solvers: {_SUPPORTED_SOLVERS}. Using 'fmin_powell'."
        )
        solver = "fmin_powell"

    n = A.shape[1]

    L = None
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)

    def cost(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        residual = A @ x - b
        value = float(np.dot(residual, residual))
        if norm == 2:
            value += alpha * float(np.dot(x, x))
        elif norm == 1:
            value += alpha * float(np.sum(np.abs(x)))
        else:
            raise ValueError(f"Unsupported norm type: {norm}")
        if L is not None:
            value += alpha * smoothness_weight * float(np.dot(L @ x, L @ x))
        return value

    @quadratic_inequality(_nonneg_condition)
    def penalty(x: np.ndarray) -> float:
        return 0.0

    x0_arr = np.zeros(n)
    if x0 is not None:
        x0_arr = np.maximum(np.asarray(x0, dtype=float), 0)

    kw = {"disp": 0, "penalty": penalty}
    if maxiter is not None:
        kw["maxiter"] = maxiter
    if maxfun is not None:
        kw["maxfun"] = maxfun

    try:
        solver_func = _solver_function(solver)
        if solver in ("diffev", "diffev2"):
            kw["bounds"] = _build_bounds(A, b, x0_arr)
        result = solver_func(cost, x0_arr, **kw)
    except Exception as exc:
        warnings.warn(
            f"Mystic solver '{solver}' failed: {exc}. Returning zero vector."
        )
        return np.zeros(n)

    return np.asarray(result, dtype=float)


def unfold_mystic(
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
    solver: str = "fmin_powell",
    maxiter: Optional[int] = 2000,
    maxfun: Optional[int] = 20000,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using mystic with regularization selection.

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
        Mystic solver name, default: 'fmin_powell'.
    maxiter : int, optional
        Maximum number of solver iterations, default: 2000.
    maxfun : int, optional
        Maximum number of function evaluations, default: 20000.
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
    A, b, selected = _build_system(readings, detector_names, sensitivities)

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
        selected_lambda = select_regularization_parameter(
            A, b, method="cosine", initial_spectrum=initial_spectrum_norm
        )
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
            solve_mystic,
            alpha=alpha,
            norm=norm,
            solver=solver,
            maxiter=maxiter,
            maxfun=maxfun,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
        ),
        solve_kwargs={},
        method_name=f"mystic_{solver}",
        extra_output={
            "norm": norm,
            "solver": solver,
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
