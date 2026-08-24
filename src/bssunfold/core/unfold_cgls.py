"""Conjugate Gradient Least Squares (CGLS) unfolding method.

This module provides a Python port of the CGLS iterative regularization
method for neutron spectrum unfolding.  CGLS applies the conjugate
gradient algorithm implicitly to the normal equations of the least
squares problem ``min ||A x - b||^2``; a regularized solution is obtained
by early termination of the iterations (semi-convergence).  Optionally a
Tikhonov term ``lambda^2 ||L x||^2`` may be added, or a non-negativity
projection applied at every iteration.

The algorithm follows the CGLS implementation of the IR Tools package by
Silvia Gazzola, Per Christian Hansen and James G. Nagy (3-Clause BSD
License) and its Python port in the TRIPs-Py library by Mirjeta Pasha
and Silvia Gazzola (Apache-2.0).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import make_regularization_operator

__all__ = ["solve_cgls", "unfold_cgls"]


def solve_cgls(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
    noise_level: Optional[float] = None,
    regularization: float = 0.0,
    smoothness_order: int = 0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with the CGLS method.

    Applies the conjugate gradient algorithm implicitly to the normal
    equations ``A^T A x = A^T b``.  A regularized solution is obtained
    by stopping the iterations once the normal-equation residual is
    sufficiently small, or (if ``noise_level`` is provided) once the
    discrepancy principle ``||A x - b|| <= eta * noise_level * ||b||``
    is satisfied.  If ``regularization`` is positive, the Tikhonov
    regularized system ``(A^T A + regularization^2 L^T L) x = A^T b`` is
    solved instead.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (default: zero vector).
    max_iterations : int, optional
        Maximum number of CGLS iterations (default: 100).
    tolerance : float, optional
        Relative tolerance on the normal-equation residual norm
        (default: 1e-12).
    noise_level : float, optional
        Relative noise level used for discrepancy-principle stopping
        (default: None).
    regularization : float, optional
        Tikhonov regularization parameter.  ``0.0`` disables the
        Tikhonov term and uses iterative regularization (default: 0.0).
    smoothness_order : int, optional
        Derivative order of the regularization operator L used when
        ``regularization`` is positive: 0 (identity), 1 or 2
        (default: 0).

    Returns
    -------
    tuple
        ``(spectrum, iterations, converged)`` where ``converged`` reports
        whether a stopping criterion was satisfied before reaching the
        maximum number of iterations.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    _, n = A.shape

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n,):
        raise ValueError(
            f"Initial spectrum length ({len(x)}) must match "
            f"number of energy bins ({n})"
        )

    nrmb = float(np.linalg.norm(b))
    if nrmb == 0.0:
        return np.zeros(n), 0, True

    L = None
    if regularization > 0:
        L = make_regularization_operator(n, smoothness_order, identity_for_zero=False)

    r = b - A @ x
    s = A.T @ r
    if L is not None:
        Lx = L @ x
        s = s - regularization * (L.T @ Lx)
    d = s.copy()

    nrmAtb = float(np.linalg.norm(A.T @ b))
    rho = float(np.dot(s, s))

    if noise_level is not None and noise_level >= 0:
        rtol = 1.01 * noise_level * nrmb
    else:
        rtol = None

    iterations = 0
    converged = False

    for k in range(1, max_iterations + 1):
        Ad = A @ d
        if L is not None:
            Ld = L @ d
            normAd2 = float(np.dot(Ad, Ad) + regularization**2 * np.dot(Ld, Ld))
        else:
            normAd2 = float(np.dot(Ad, Ad))

        if normAd2 <= 0.0:
            break

        alpha = rho / normAd2
        x = x + alpha * d
        r = r - alpha * Ad

        if L is not None:
            s = A.T @ r - regularization * (L.T @ (L @ x))
        else:
            s = A.T @ r

        rho_new = float(np.dot(s, s))
        beta = rho_new / rho if rho > 0 else 0.0
        rho = rho_new
        d = s + beta * d

        iterations = k

        ne_res = float(np.linalg.norm(s))
        if rtol is not None:
            res = float(np.linalg.norm(r))
            if res <= rtol:
                converged = True
                break
        if nrmAtb > 0 and ne_res <= tolerance * nrmAtb:
            converged = True
            break
        if ne_res <= tolerance:
            converged = True
            break

    x = np.maximum(x, 0)
    return x, iterations, converged


def unfold_cgls(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
    noise_level: Optional[float] = None,
    regularization: float = 0.0,
    smoothness_order: int = 0,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum with the CGLS method.

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
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tolerance : float, optional
        Relative tolerance on the normal-equation residual (default: 1e-12).
    noise_level : float, optional
        Relative noise level used for discrepancy-principle stopping.
    regularization : float, optional
        Tikhonov regularization parameter (default: 0.0).
    smoothness_order : int, optional
        Derivative order of the regularization operator L used when
        ``regularization`` is positive (default: 0).
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty (default: False).
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
            solve_cgls,
            max_iterations=max_iterations,
            tolerance=tolerance,
            noise_level=noise_level,
            regularization=regularization,
            smoothness_order=smoothness_order,
        ),
        solve_kwargs={},
        method_name="CGLS",
        extra_output={
            "max_iterations": max_iterations,
            "regularization": float(regularization),
            "smoothness_order": int(smoothness_order),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
