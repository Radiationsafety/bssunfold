"""QP solvers-based unfolding method with regularization selection.

This module provides the core solve_qpsolvers solver and the unfold_qpsolvers
wrapper with various regularization selection methods.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import csc_matrix

from ._base_unfolder import _build_system, run_unfolding
from ._matrix_utils import build_smoothness_penalty
from ._max_energy import upper_bounds
from .regularization import resolve_regularization_parameter

__all__ = ["solve_qpsolvers", "unfold_qpsolvers"]


def solve_qpsolvers(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    solver: str = "osqp",
    x0: Optional[np.ndarray] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    ub: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Solve unfolding problem using qpsolvers.

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
        QP solver name (default: 'osqp').
    x0 : np.ndarray, optional
        Initial values.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2).
    smoothness_weight : float, optional
        Weight for smoothness term.

    Returns
    -------
    Optional[np.ndarray]
        Unfolded spectrum or None if solving failed.
    """
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as e:
        raise ImportError(
            "qpsolvers is required for unfold_qpsolvers. "
            "Install with: pip install qpsolvers"
        ) from e

    if solver not in available_solvers:
        if "osqp" in available_solvers:
            solver = "osqp"
        elif "ecos" in available_solvers:
            solver = "ecos"
        else:
            warnings.warn(
                f"Solver '{solver}' not available. Available: {available_solvers}"
            )
            return None

    n = A.shape[1]

    P_base = csc_matrix(A.T @ A)
    q_base = -A.T @ b
    pen = build_smoothness_penalty(n, alpha, smoothness_order, smoothness_weight)

    if norm == 2:
        # Zeroth-order Tikhonov (identity) when no derivative penalty is requested.
        P = P_base.copy() + (pen if pen is not None else alpha * csc_matrix(np.eye(n)))

        x = solve_qp(
            P=P,
            q=q_base,
            lb=np.zeros(n),
            ub=ub,
            solver=solver,
            initvals=x0,
            verbose=False,
        )

    elif norm == 1:
        # L1 regularization under the non-negativity constraint x >= 0:
        #   min 0.5 * ||A x - b||^2 + alpha * ||x||_1  s.t.  x >= 0
        # The penalty term alpha * sum(x) is linear, so it shifts q by alpha.
        P = P_base.copy()
        q = q_base + alpha * np.ones(n)
        if pen is not None:
            P += pen

        x = solve_qp(
            P=P,
            q=q,
            lb=np.zeros(n),
            ub=ub,
            solver=solver,
            initvals=x0,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

    if x is None:
        warnings.warn(f"Solver '{solver}' did not find a solution.")
        return None

    x = np.asarray(x)
    if ub is not None:
        x[ub == 0.0] = 0.0
    return x


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
    save_result: bool = False,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    random_state: Optional[int] = None,
    max_neutron_energy: Optional[float] = None,
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

    alpha = resolve_regularization_parameter(
        A,
        b,
        regularization_method,
        regularization,
        n_energy_bins,
        initial_spectrum=initial_spectrum,
        norm=norm,
        noise_var=noise_var,
    )
    selected_lambda = alpha
    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
        x0 = kwargs.pop("x0", None)
        ub = upper_bounds(E_MeV, max_neutron_energy)
        x = solve_qpsolvers(
            A,
            b,
            alpha,
            norm,
            solver,
            x0=x0,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            ub=ub,
        )
        if x is None:
            x = np.zeros(A.shape[1])
            warnings.warn("Solution not found, returning zero spectrum.")
        return x

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
