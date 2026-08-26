"""Mystic-based unfolding method with regularization selection.

This module provides the core solve_mystic solver and the unfold_mystic
wrapper with various regularization selection methods. The optimization is
performed with the ``mystic`` constrained-optimization framework
(https://pypi.org/project/mystic/), using a direct-search solver on the
penalized least-squares objective.

Additionally, a two-stage hybrid solver (``solve_mystic_hybrid`` / 
``unfold_mystic_hybrid``) is provided that first uses ``diffev2`` for
global exploration and then refines the result with ``fmin_powell`` for
precise local convergence.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ._base_unfolder import _build_system, make_solve_wrapper, run_unfolding
from ._matrix_utils import create_derivative_matrix
from .regularization import select_regularization_parameter

__all__ = [
    "solve_mystic",
    "unfold_mystic",
    "solve_mystic_hybrid",
    "unfold_mystic_hybrid",
]

# Supported mystic minimal-interface solvers
_SUPPORTED_SOLVERS = ("fmin", "fmin_powell", "diffev", "diffev2")


def _solver_function(solver: str):
    """Import and return the mystic minimal-interface solver callable."""
    from mystic.solvers import diffev, diffev2, fmin, fmin_powell

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


def _build_bounds(
    A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray]
) -> list:
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
    A, b, _ = _build_system(readings, detector_names, sensitivities)

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
        print(f"Selected regularization (method=cosine): {selected_lambda:.3e}")
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
            ) from e
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


def solve_mystic_hybrid(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    x0: Optional[np.ndarray] = None,
    global_solver: str = "diffev2",
    local_solver: str = "fmin_powell",
    global_maxiter: Optional[int] = None,
    global_maxfun: Optional[int] = None,
    local_maxiter: Optional[int] = None,
    local_maxfun: Optional[int] = None,
    npop: Optional[int] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
) -> np.ndarray:
    """Two-stage hybrid solver: global search then local refinement.

    Stage 1 runs a population-based solver (default ``diffev2``) with
    automatically derived bounds to locate the basin of the global minimum.
    Stage 2 feeds that result as ``x0`` into a local direct-search solver
    (default ``fmin_powell``) for precise convergence.

    This combines the robustness of global exploration with the accuracy
    of local optimization, which is a widely used practical strategy for
    ill-posed inverse problems like spectrum unfolding.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    x0 : np.ndarray, optional
        Initial values for the global stage. Defaults to the zero vector.
    global_solver : str, optional
        Population-based solver for stage 1. Must be ``'diffev'`` or
        ``'diffev2'`` (default: ``'diffev2'``).
    local_solver : str, optional
        Local solver for stage 2. Must be ``'fmin'`` or ``'fmin_powell'``
        (default: ``'fmin_powell'``).
    global_maxiter : int, optional
        Maximum iterations for the global stage. Defaults to 200.
    global_maxfun : int, optional
        Maximum function evaluations for the global stage.
        Defaults to ``10 * n * 20`` where *n* is the number of energy bins.
    local_maxiter : int, optional
        Maximum iterations for the local stage. Defaults to 2000.
    local_maxfun : int, optional
        Maximum function evaluations for the local stage.
        Defaults to 20000.
    npop : int, optional
        Population size for the global stage. Defaults to
        ``min(10 * n, 200)`` where *n* is the number of energy bins.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2), default: 0.
    smoothness_weight : float, optional
        Weight for the smoothness term, default: 1.0.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,). Returns a zero vector if both stages fail.
    """
    try:
        from mystic.penalty import quadratic_inequality
    except ImportError as e:
        raise ImportError(
            "mystic is required for solve_mystic_hybrid. "
            "Install with: pip install mystic"
        ) from e

    # --- Validate solver choices ---
    _GLOBAL_SOLVERS = ("diffev", "diffev2")
    _LOCAL_SOLVERS = ("fmin", "fmin_powell")

    if global_solver not in _GLOBAL_SOLVERS:
        warnings.warn(
            f"Global solver '{global_solver}' is not population-based. "
            f"Supported: {_GLOBAL_SOLVERS}. Falling back to 'diffev2'."
        )
        global_solver = "diffev2"

    if local_solver not in _LOCAL_SOLVERS:
        warnings.warn(
            f"Local solver '{local_solver}' is not a direct-search solver. "
            f"Supported: {_LOCAL_SOLVERS}. Falling back to 'fmin_powell'."
        )
        local_solver = "fmin_powell"

    n = A.shape[1]

    # --- Derivative matrix for smoothness ---
    L = None
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)

    # --- Objective function (shared by both stages) ---
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
            value += alpha * smoothness_weight * float(
                np.dot(L @ x, L @ x)
            )
        return value

    @quadratic_inequality(_nonneg_condition)
    def penalty(x: np.ndarray) -> float:
        return 0.0

    # --- Default limits ---
    if global_maxiter is None:
        global_maxiter = 200
    if global_maxfun is None:
        global_maxfun = 10 * n * 20
    if local_maxiter is None:
        local_maxiter = 2000
    if local_maxfun is None:
        local_maxfun = 20000
    if npop is None:
        npop = min(10 * n, 200)

    x0_arr = np.zeros(n)
    if x0 is not None:
        x0_arr = np.maximum(np.asarray(x0, dtype=float), 0)

    # ============================================================
    # Stage 1: Global search with population-based solver
    # ============================================================
    global_cost = float("inf")
    x_global = np.zeros(n)

    try:
        solver_func = _solver_function(global_solver)
        bounds = _build_bounds(A, b, x0_arr)
        x_global = np.asarray(
            solver_func(
                cost,
                x0_arr,
                bounds=bounds,
                penalty=penalty,
                maxiter=global_maxiter,
                maxfun=global_maxfun,
                npop=npop,
                disp=0,
            ),
            dtype=float,
        )
        global_cost = cost(x_global)
    except Exception as exc:
        warnings.warn(
            f"Hybrid stage 1 (global, solver='{global_solver}') failed: "
            f"{exc}. Attempting local refinement from zero vector."
        )

    # ============================================================
    # Stage 2: Local refinement with direct-search solver
    # ============================================================
    x_local = np.maximum(x_global, 0.0)  # ensure non-negative start

    try:
        solver_func = _solver_function(local_solver)
        x_local = np.asarray(
            solver_func(
                cost,
                x_local,
                penalty=penalty,
                maxiter=local_maxiter,
                maxfun=local_maxfun,
                disp=0,
            ),
            dtype=float,
        )
        local_cost = cost(x_local)
    except Exception as exc:
        warnings.warn(
            f"Hybrid stage 2 (local, solver='{local_solver}') failed: "
            f"{exc}. Returning global stage result."
        )
        return np.maximum(x_global, 0.0)

    # Return the best of the two stages
    if local_cost <= global_cost:
        return np.maximum(x_local, 0.0)
    else:
        warnings.warn(
            "Hybrid local stage did not improve upon global result. "
            f"Global cost: {global_cost:.6e}, local cost: {local_cost:.6e}. "
            "Returning global stage result."
        )
        return np.maximum(x_global, 0.0)


def unfold_mystic_hybrid(
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
    global_solver: str = "diffev2",
    local_solver: str = "fmin_powell",
    global_maxiter: Optional[int] = None,
    global_maxfun: Optional[int] = None,
    local_maxiter: Optional[int] = None,
    local_maxfun: Optional[int] = None,
    npop: Optional[int] = None,
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
    """Two-stage hybrid unfolding: global search + local refinement.

    Stage 1 uses a population-based solver (``diffev2`` by default) with
    automatically derived bounds to robustly locate the basin of the
    global minimum. Stage 2 feeds that result as ``x0`` into a local
    direct-search solver (``fmin_powell`` by default) for precise final
    convergence. This combines the robustness of global optimization
    with the accuracy of local optimization.

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
        Initial spectrum guess for the global stage.
    regularization : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    global_solver : str, optional
        Population-based solver for stage 1 (``'diffev'`` or ``'diffev2'``),
        default: ``'diffev2'``.
    local_solver : str, optional
        Local solver for stage 2 (``'fmin'`` or ``'fmin_powell'``),
        default: ``'fmin_powell'``.
    global_maxiter : int, optional
        Maximum iterations for the global stage (default: 200).
    global_maxfun : int, optional
        Maximum function evaluations for the global stage.
    local_maxiter : int, optional
        Maximum iterations for the local stage (default: 2000).
    local_maxfun : int, optional
        Maximum function evaluations for the local stage.
    npop : int, optional
        Population size for the global stage.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default: 100.
    save_result : bool, optional
        Save result to history, default: False.
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
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    # --- Regularization parameter selection (same logic as unfold_mystic) ---
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
            f"Selected regularization (method=cosine): {selected_lambda:.3e}"
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
            ) from e
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
            solve_mystic_hybrid,
            alpha=alpha,
            norm=norm,
            global_solver=global_solver,
            local_solver=local_solver,
            global_maxiter=global_maxiter,
            global_maxfun=global_maxfun,
            local_maxiter=local_maxiter,
            local_maxfun=local_maxfun,
            npop=npop,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
        ),
        solve_kwargs={},
        method_name=f"mystic_hybrid_{global_solver}_{local_solver}",
        extra_output={
            "norm": norm,
            "global_solver": global_solver,
            "local_solver": local_solver,
            "regularization": regularization,
            "regularization_method": regularization_method,
            "selected_regularization": float(selected_lambda),
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "global_maxiter": global_maxiter,
            "global_maxfun": global_maxfun,
            "local_maxiter": local_maxiter,
            "local_maxfun": local_maxfun,
            "npop": npop,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
