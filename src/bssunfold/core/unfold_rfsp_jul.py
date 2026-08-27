"""RFSP-JUL unfolding method for neutron spectrum reconstruction.

This is an independent open-source reimplementation of the RFSP-JUL
algorithm following its published mathematical description (Fischer; the
1981 review of unfolding codes). The original RFSP-JUL code is a
proprietary package. This
implementation is built solely from the published algorithmic description
and does not use or reproduce any proprietary source code.

RFSP-JUL is an iterative, damped least-squares method. At each iteration
``k`` it minimises the functional

    S^{(k)} = sum_i W_i [ (b_i - (A phi)_i) / b_i ]^2
              + sum_j [ (phi_j - phi_prev_j) / phi_prev_j ]^2

where ``phi_prev = phi^{(k-1)}`` (the previous iterate) provides a
Marquardt-style damping that keeps the solution from diverging. Both terms
are quadratic in ``phi``, so the minimiser is found in closed form from the
normal equations (a symmetric positive-definite system solved directly):

    [ sum_i W_i R_i R_i^T / b_i^2  +  diag(1 / phi_prev^2) ] phi
        = sum_i W_i R_i / b_i  +  phi_prev / phi_prev^2
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_rfsp_jul", "unfold_rfsp_jul"]


def solve_rfsp_jul(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-4,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the RFSP-JUL algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,). Also used as the reference iterate
        ``phi_prev`` at the first iteration.
    max_iterations : int, optional
        Maximum number of iterations (default: 200).
    tolerance : float, optional
        Convergence tolerance on the maximum relative spectrum change
        (default: 1e-4).
    weights : np.ndarray, optional
        Per-detector weights ``W_i`` for the residual term. When None, all
        detectors are weighted equally (``W_i = 1``).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    if A.size == 0 or b.size == 0:
        raise ValueError("Response matrix and measurements must be non-empty")
    if np.all(b <= 0):
        raise ValueError("All measurements are zero or negative")

    m, n = A.shape
    if weights is None:
        W = np.ones(m)
    else:
        W = np.asarray(weights, dtype=float)
    W = np.maximum(W, 0.0)

    # Guard the 1/b_i^2 weighting: only strictly positive measurements enter.
    pos = b > 0
    if not np.any(pos):
        raise ValueError("All measurements are zero or negative")

    A_pos = A[pos]
    b_pos = b[pos]
    W_pos = W[pos]
    b_safe_pos = b_pos  # already > 0 here

    # Precompute the (positive) right-hand data term: sum_i W_i R_i / b_i.
    wb_inv = W_pos / (b_safe_pos ** 2)
    rhs_data = A_pos.T @ (wb_inv * b_pos)  # sum_i W_i R_ik / b_i
    # Precompute the (positive) curvature contribution: sum_i W_i R_i R_i^T / b_i^2.
    # Stored as the n x n matrix M[k,j] = sum_i W_i R_ik R_ij / b_i^2.
    M = A_pos.T @ (wb_inv[:, None] * A_pos)

    x = np.maximum(x0, 1e-12).copy()
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        phi_prev = np.maximum(x, 1e-12)
        inv_prev2 = 1.0 / (phi_prev ** 2)

        # Symmetric positive-definite system (regularised on the diagonal).
        lhs = M + np.diag(inv_prev2)
        rhs = rhs_data + inv_prev2 * phi_prev
        x_new = np.linalg.solve(lhs, rhs)
        x_new = np.maximum(x_new, 0.0)

        rel_change = np.max(np.abs(x_new - x) / np.maximum(x, 1e-12))
        x = x_new
        if rel_change < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_rfsp_jul(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-4,
    weights: Optional[np.ndarray] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the RFSP-JUL algorithm.

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
        Initial spectrum guess. If None, a flat spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 200).
    tolerance : float, optional
        Convergence tolerance on maximum relative spectrum change
        (default: 1e-4).
    weights : np.ndarray, optional
        Per-detector weights for the residual term. None => equal weights.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins)

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
            solve_rfsp_jul,
            max_iterations=max_iterations,
            tolerance=tolerance,
            weights=weights,
        ),
        solve_kwargs={},
        method_name="RFSP-JUL",
        extra_output={"tolerance": float(tolerance)},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
