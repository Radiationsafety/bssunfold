"""SAND-II unfolding method for neutron spectrum reconstruction.

Port of the SAND-II algorithm (Sandia National Laboratories; McElroy et al.
1967 and Griffin, Kelly & VanDenburg 1994, doi:10.2172/10149711) following
the reference NumPy implementation in the BUMS2 package
(https://github.com/larnold34/BUMS2, MIT license).

SAND-II is a multiplicative (geometric-mean) iterative ratio method. Starting
from an initial spectrum x0, each energy bin is corrected by the weighted
geometric mean of the measured-to-calculated count-rate ratios:

    E_i      = sum_j A_ij x_j                     (calculated counts)
    W_ij     = A_ij x_j / E_i                     (bin-to-detector weight)
    x_j     <- x_j * exp( sum_i W_ij ln(b_i / E_i) / sum_i W_ij )

Iteration stops when the chi-square of the fit is not greater than the number
of detectors (``chi_fac=1``) or when the maximum relative change of the
spectrum drops below ``tolerance`` (``chi_fac=0``).
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_sandii", "unfold_sandii"]


def solve_sandii(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    chi_fac: int = 1,
    relative_uncertainty: float = 0.1,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the SAND-II algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    tolerance : float, optional
        Maximum relative spectrum change used when ``chi_fac=0``
        (default: 1e-3).
    chi_fac : int, optional
        Convergence criterion: ``1`` = stop when chi-square of the fit is
        not greater than the number of detectors; ``0`` = stop when the
        maximum relative change of the spectrum is below ``tolerance``
        (default: 1).
    relative_uncertainty : float, optional
        Relative measurement uncertainty used to derive detector sigma values
        when ``sigma`` is not supplied (default: 0.1).
    sigma : np.ndarray, optional
        Explicit per-detector measurement uncertainties (m,). When given,
        overrides ``relative_uncertainty``.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x = np.maximum(x0, 0).copy()

    if sigma is not None:
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    else:
        sigma = relative_uncertainty * np.maximum(b, 1e-12)

    valid = b > 0
    if not np.any(valid):
        raise ValueError("All measurements are zero or negative")

    A_valid = A[valid]
    b_valid = b[valid]
    sigma_valid = sigma[valid]
    m_valid = b_valid.shape[0]
    eps = 1e-12

    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        E = A_valid @ x
        E_safe = np.maximum(E, 1e-300)
        R = b_valid / E_safe
        W = A_valid * (x[None, :] / E_safe[:, None])

        denom = W.sum(axis=0)
        denom_safe = np.where(denom <= eps, eps, denom)
        numer = np.sum(W * np.log(R)[:, None], axis=0)
        x_new = x * np.exp(numer / denom_safe)

        if chi_fac == 1:
            chi2 = np.sum(((b_valid - A_valid @ x_new) / sigma_valid) ** 2)
            if chi2 <= m_valid:
                x = x_new
                converged = True
                break
        else:
            rel = np.abs(x_new - x) / np.maximum(x, eps)
            if np.max(rel) < tolerance:
                x = x_new
                converged = True
                break

        x = x_new

    return x, iterations, converged


def unfold_sandii(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    chi_fac: int = 1,
    relative_uncertainty: float = 0.1,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the SAND-II algorithm.

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
        Maximum number of iterations (default: 50).
    tolerance : float, optional
        Maximum relative spectrum change used when ``chi_fac=0``
        (default: 1e-3).
    chi_fac : int, optional
        Convergence criterion (default: 1, chi-square based).
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
            solve_sandii,
            max_iterations=max_iterations,
            tolerance=tolerance,
            chi_fac=chi_fac,
            relative_uncertainty=relative_uncertainty,
        ),
        solve_kwargs={},
        method_name="SAND-II",
        extra_output={
            "chi_fac": int(chi_fac),
            "relative_uncertainty": float(relative_uncertainty),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
