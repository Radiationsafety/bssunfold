"""QP solvers-based unfolding method with regularization selection.

This module provides the core solve_qpsolvers solver and the unfold_qpsolvers
wrapper with various regularization selection methods.
"""

import warnings
import numpy as np
from typing import Dict, Optional, Any, List

from scipy.sparse import csc_matrix

from ._matrix_utils import create_derivative_matrix
from .regularization import select_regularization_parameter
from ._base_unfolder import run_unfolding
from .basis import SpectralBasis

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

    if norm == 2:
        P = P_base.copy()

        if smoothness_order == 1:
            L = create_derivative_matrix(n, 1)
            P += alpha * smoothness_weight * (L.T @ L)
        elif smoothness_order == 2:
            L = create_derivative_matrix(n, 2)
            P += alpha * smoothness_weight * (L.T @ L)
        else:
            P += alpha * csc_matrix(np.eye(n))

        G = csc_matrix(-np.eye(n))
        h = np.zeros(n)

        x = solve_qp(
            P=P,
            q=q_base,
            G=G,
            h=h,
            solver=solver,
            initvals=x0,
            verbose=False,
        )

    elif norm == 1:
        n_ext = 2 * n
        P_ext = csc_matrix((n_ext, n_ext))
        P_ext[:n, :n] = P_base

        if smoothness_order == 1:
            L = create_derivative_matrix(n, 1)
            P_ext[:n, :n] += alpha * smoothness_weight * (L.T @ L)
        elif smoothness_order == 2:
            L = create_derivative_matrix(n, 2)
            P_ext[:n, :n] += alpha * smoothness_weight * (L.T @ L)

        q_ext = np.zeros(n_ext)
        q_ext[:n] = q_base
        q_ext[n:] = alpha * np.ones(n)

        G_ext = csc_matrix((3 * n, n_ext))
        h_ext = np.zeros(3 * n)

        G_ext[:n, :n] = -csc_matrix(np.eye(n))
        G_ext[n:2 * n, n:] = -csc_matrix(np.eye(n))
        G_ext[2 * n:3 * n, :n] = -csc_matrix(np.eye(n))
        G_ext[2 * n:3 * n, n:] = csc_matrix(np.eye(n))

        x0_ext = None
        if x0 is not None:
            x0_ext = np.concatenate([x0, np.abs(x0)])

        x_ext = solve_qp(
            P=P_ext,
            q=q_ext,
            G=G_ext,
            h=h_ext,
            solver=solver,
            initvals=x0_ext,
            verbose=False,
        )

        if x_ext is None:
            warnings.warn(f"Solver '{solver}' did not find a solution for L1 problem.")
            return None

        x = x_ext[:n] - x_ext[n:]
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

    if x is None:
        warnings.warn(f"Solver '{solver}' did not find a solution.")
        return None

    return np.asarray(x)


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
    basis: Optional[SpectralBasis] = None,
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
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

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
                x0=initial_spectrum_norm,
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

    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
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
        basis=basis,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
