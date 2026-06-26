"""MLEM-STOP unfolding method with J-factor early stopping criterion.

Based on:
    L. Montgomery et al., "A novel MLEM stopping criterion for unfolding
    neutron fluence spectra in radiation therapy",
    Nucl. Instrum. Meth. A 957 (2020) 163400.
    https://doi.org/10.1016/j.nima.2020.163400

Reference C++ implementation:
    https://github.com/kildealab/Neutron-Spectrometry
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_mlem_stop", "unfold_mlem_stop"]


def calculate_j_factor(
    measurements: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Calculate the J-factor indicator from Bouallegue et al.

    J = sum((measurements - estimate)^2) / sum(estimate)
    """
    numerator = np.sum((measurements - estimate) ** 2)
    denominator = np.sum(estimate)
    if denominator <= 0:
        return float('inf')
    return numerator / denominator


def solve_mlem_stop(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 15000,
    cps_crossover: float = 30000.0,
    j_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using MLEM with J-factor stopping criterion.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum iterations (default: 15000).
    cps_crossover : float, optional
        Crossover CPS value for automatic J threshold (default: 30000).
        Used only when j_threshold is None.
    j_threshold : float, optional
        J-factor stopping threshold. If None, computed as
        mean(b) / cps_crossover.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    if j_threshold is None:
        j_threshold = np.mean(b) / cps_crossover

    x = np.maximum(x0.copy(), 1e-10)
    AT = A.T

    for i in range(max_iterations):
        Ax = A @ x
        j_factor = calculate_j_factor(b, Ax)

        if j_factor <= j_threshold:
            return x, i + 1, True

        Ax = np.maximum(Ax, 1e-10)
        ratio = b / Ax
        correction = AT @ ratio
        x = np.maximum(x * correction, 0)

    j_final = calculate_j_factor(b, A @ x)
    converged = j_final <= j_threshold
    return x, max_iterations, converged


def unfold_mlem_stop(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 15000,
    cps_crossover: float = 30000.0,
    j_threshold: Optional[float] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using MLEM-STOP algorithm with J-factor stopping criterion.

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
    max_iterations : int, optional
        Maximum iterations (default: 15000).
    cps_crossover : float, optional
        Crossover CPS value for automatic J threshold (default: 30000).
    j_threshold : float, optional
        J-factor stopping threshold. If None, computed automatically.
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
    x0_default = np.ones(n_energy_bins) * 0.5

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
            solve_mlem_stop,
            max_iterations=max_iterations,
            cps_crossover=cps_crossover,
            j_threshold=j_threshold,
        ),
        solve_kwargs={},
        method_name="MLEM-STOP",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
