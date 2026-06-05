"""Scipy direct solver-based unfolding method.

This module provides the core solve_scipy_direct solver and the
unfold_scipy_direct_method wrapper for use with the Detector class.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_scipy_direct", "unfold_scipy_direct_method"]


def solve_scipy_direct(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 4000,
    method: str = "cg",
) -> np.ndarray:
    """Solve unfolding problem using scipy sparse linear solvers.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    tolerance : float, optional
        Solver tolerance (default: 1e-8).
    max_iterations : int, optional
        Maximum solver iterations (default: 4000).
    method : str, optional
        Solver method. One of: 'cg', 'cgs', 'bicgstab', 'gmres', 'lgmres',
        'minres', 'gcrotmk', 'qmr', 'tfqmr', 'lsqr', 'lsmr' (default: 'cg').

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    from scipy.sparse.linalg import (
        cg, cgs, bicgstab, gmres, lgmres, minres,
        gcrotmk, qmr, tfqmr, lsqr, lsmr,
    )

    AT_A = A.T @ A
    AT_b = A.T @ b

    solvers = {
        "cg": lambda: cg(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "cgs": lambda: cgs(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "bicgstab": lambda: bicgstab(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "gmres": lambda: gmres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "lgmres": lambda: lgmres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "minres": lambda: minres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "qmr": lambda: qmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "gcrotmk": lambda: gcrotmk(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "tfqmr": lambda: tfqmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
        "lsqr": lambda: lsqr(A, b, atol=tolerance),
        "lsmr": lambda: lsmr(A, b, atol=tolerance, maxiter=max_iterations),
    }

    if method not in solvers:
        raise ValueError(
            f"Unknown solver method '{method}'. "
            f"Choose from: {list(solvers.keys())}"
        )

    x = solvers[method]()[0]
    return np.maximum(x, 0)


def unfold_scipy_direct_method(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 4000,
    method: str = "cg",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using scipy direct solvers.

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
    tolerance : float, optional
        Solver tolerance (default: 1e-8).
    max_iterations : int, optional
        Maximum solver iterations (default: 4000).
    method : str, optional
        Solver method (default: 'cg').
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
            solve_scipy_direct,
            tolerance=tolerance,
            max_iterations=max_iterations,
            method=method,
        ),
        solve_kwargs={},
        method_name=f"Scipy_{method}",
        extra_output={
            "tolerance": tolerance,
            "solver_method": method,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
