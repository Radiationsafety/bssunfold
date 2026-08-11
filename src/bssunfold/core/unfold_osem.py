"""OSEM (ordered-subset expectation maximisation) unfolding method.

Port of the ``OSEM`` reconstruction algorithm from PyTomography
(https://github.com/PyTomography/PyTomography, MIT license) adapted to
neutron spectrum unfolding. OSEM generalises MLEM by updating the spectrum
with one subset of detectors at a time, accelerating convergence:

    x^{n+1} = x^n * A_m^T ( b_m / (A_m x^n + eps) ) / ( A_m^T 1 + eps )

where ``m`` indexes the detector subset. With ``n_subsets=1`` the update
reduces exactly to standard MLEM.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_osem", "unfold_osem"]


def solve_osem(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using OSEM (ordered-subset EM).

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
    n_subsets : int, optional
        Number of ordered subsets over the detector readings
        (default: 1, i.e. standard MLEM).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    m, _ = A.shape

    if n_subsets < 1:
        raise ValueError("n_subsets must be >= 1")
    if n_subsets > m:
        raise ValueError(
            f"n_subsets ({n_subsets}) must not exceed the number of "
            f"detectors ({m})"
        )

    eps = 1e-11
    subset_indices = np.array_split(np.arange(m), n_subsets)
    x = np.maximum(x0, 0).copy()
    converged = False
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it
        x_old = x.copy()

        for idx in subset_indices:
            A_sub = A[idx]
            b_sub = b[idx]
            norm = A_sub.sum(axis=0)
            ratio = b_sub / (A_sub @ x + eps)
            correction = A_sub.T @ ratio
            x = np.maximum(x * correction / (norm + eps), 0.0)

        rel = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + eps)
        if rel < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_osem(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the OSEM algorithm.

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
    n_subsets : int, optional
        Number of ordered subsets over the detector readings
        (default: 1, i.e. standard MLEM).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
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
            solve_osem,
            max_iterations=max_iterations,
            n_subsets=n_subsets,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="OSEM",
        extra_output={
            "n_subsets": int(n_subsets),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
