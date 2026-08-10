"""FERDOR (Ferret) unfolding method for neutron spectrum reconstruction.

FERDOR is one of the classic neutron-spectrum unfolding codes developed at
Oak Ridge National Laboratory (Burrus, ORNL-4154, 1965; Burrus, ORNL-3743,
"Utilization of a priori information in the interpretation of measured
data").  It belongs to the "Ferret" family of constrained least-squares
unfolding codes and is frequently used as the reference code for Bonner
sphere and NE-213 / BC501A proton-recoil unfolding.

Algorithm
---------
FERDOR seeks a spectrum that reproduces the measured detector readings
within their uncertainties while being as smooth as possible.  This is
realised here as a weighted, noise-constrained least-squares problem with a
second-difference smoothing (regularisation) term:

    phi_hat = argmin { 1/2 * ||Sigma^{-1/2} (A phi - b)||^2
                       + alpha/2 * ||D2 phi||^2 },   phi >= 0,

where ``Sigma`` is the diagonal measurement-covariance matrix derived from
the relative uncertainties, ``D2`` is the second-order finite-difference
operator and ``alpha`` is the smoothing weight.  ``alpha`` is not fixed a
priori: it is adjusted iteratively (bisection on the smoothing weight) so
that the reduced chi-square of the fit,

    chi2 / nu = (1/nu) * sum_i ( (b_i - (A phi)_i) / sigma_i )^2,

equals the ``chi_squared_target`` (default 1.0, i.e. the discrepancy
principle also used by the FERDOR code).  The final spectrum is clipped to
non-negative values.

The solution is obtained by a direct constrained least-squares solve at each
smoothing step, so (like other direct methods) the result does not depend on
the initial guess; ``x0`` is accepted for interface compatibility with the
other ``solve_*`` functions and used only as a fallback if the solve fails.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._matrix_utils import create_derivative_matrix
from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_ferdor", "unfold_ferdor"]


def _solve_weighted_ls(
    ATA: np.ndarray,
    ATb: np.ndarray,
    LTL: np.ndarray,
    alpha: float,
) -> Optional[np.ndarray]:
    """Solve the weighted least-squares system for a given smoothing weight.

    Returns the non-negative solution or None if the linear solve fails.
    """
    P = ATA + alpha * LTL
    try:
        x = np.linalg.solve(P, ATb)
    except np.linalg.LinAlgError:
        try:
            x = np.linalg.lstsq(P, ATb, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
    return np.maximum(x, 0.0)


def solve_ferdor(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    smoothing: float = 1e-3,
    chi_squared_target: float = 1.0,
    relative_uncertainty: float = 0.1,
    sigma: Optional[np.ndarray] = None,
    min_alpha: float = 1e-12,
    max_alpha: float = 1e12,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the FERDOR algorithm.

    The smoothing weight ``alpha`` is adjusted iteratively (bisection) so
    that the reduced chi-square of the fit approaches ``chi_squared_target``
    (discrepancy principle).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,). Accepted for interface compatibility;
        the constrained least-squares solution does not depend on it.
    max_iterations : int, optional
        Maximum number of smoothing-weight adjustment iterations
        (default: 100).
    tolerance : float, optional
        Relative tolerance on the reduced chi-square used to stop the
        bisection (default: 1e-3).
    smoothing : float, optional
        Initial value of the smoothing weight alpha (default: 1e-3).
    chi_squared_target : float, optional
        Target reduced chi-square per degree of freedom (default: 1.0).
    relative_uncertainty : float, optional
        Relative measurement uncertainty used to derive the per-detector
        sigma values when ``sigma`` is not supplied (default: 0.1).
    sigma : np.ndarray, optional
        Explicit per-detector measurement uncertainties (m,). When given,
        overrides ``relative_uncertainty``.
    min_alpha : float, optional
        Lower bound of the smoothing-weight search bracket (default: 1e-12).
    max_alpha : float, optional
        Upper bound of the smoothing-weight search bracket (default: 1e12).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    m, n = A.shape

    if m == 0:
        raise ValueError("Measurement vector b is empty")
    if np.all(b <= 0):
        raise ValueError("FERDOR requires at least one strictly positive measurement")

    if sigma is not None:
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    else:
        sigma = relative_uncertainty * np.maximum(np.abs(b), 1e-12)

    if sigma.shape != (m,):
        raise ValueError(f"sigma must have shape ({m},)")

    Wsqrt = 1.0 / sigma
    Aw = A * Wsqrt[:, None]
    bw = b * Wsqrt

    ATA = Aw.T @ Aw
    ATb = Aw.T @ bw

    L = create_derivative_matrix(n, 2).toarray() if n > 2 else None
    if L is None:
        LTL = np.zeros((n, n))
    else:
        LTL = L.T @ L

    dof = max(m - 1, 1)

    lo, hi = float(min_alpha), float(max_alpha)
    alpha = float(smoothing)
    alpha = min(max(alpha, lo), hi)

    spectrum = np.maximum(x0, 0.0).copy()
    converged = False
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it
        x = _solve_weighted_ls(ATA, ATb, LTL, alpha)
        if x is None:
            break
        spectrum = x

        residual = A @ x - b
        chi2 = float(np.dot(residual / sigma, residual / sigma))
        ratio = chi2 / dof

        if abs(ratio - chi_squared_target) <= tolerance * max(
            abs(chi_squared_target), 1.0
        ):
            converged = True
            break

        if ratio > chi_squared_target:
            hi = alpha
        else:
            lo = alpha

        if hi <= lo:
            converged = True
            break

        new_alpha = np.sqrt(lo * hi)
        if abs(new_alpha - alpha) <= tolerance * max(abs(alpha), 1e-30):
            converged = True
            break
        alpha = new_alpha

    return spectrum, iterations, converged


def unfold_ferdor(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    smoothing: float = 1e-3,
    chi_squared_target: float = 1.0,
    relative_uncertainty: float = 0.1,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the FERDOR algorithm.

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
        Maximum number of smoothing-weight iterations (default: 100).
    tolerance : float, optional
        Relative tolerance on the reduced chi-square (default: 1e-3).
    smoothing : float, optional
        Initial smoothing weight alpha (default: 1e-3).
    chi_squared_target : float, optional
        Target reduced chi-square per degree of freedom (default: 1.0).
    relative_uncertainty : float, optional
        Relative measurement uncertainty for the chi-square criterion
        (default: 0.1).
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
            solve_ferdor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            smoothing=smoothing,
            chi_squared_target=chi_squared_target,
            relative_uncertainty=relative_uncertainty,
        ),
        solve_kwargs={},
        method_name="FERDOR",
        extra_output={
            "chi_squared_target": float(chi_squared_target),
            "relative_uncertainty": float(relative_uncertainty),
            "smoothing": float(smoothing),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
