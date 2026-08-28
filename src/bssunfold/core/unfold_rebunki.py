"""ReBUNKI unfolding method for neutron spectrum reconstruction.

ReBUNKI (Lacerda et al., 2018) is a modern, open reimplementation of the
BUNKI Bonner-sphere unfolding code.  The original BUNKI was programmed at
the Naval Research Laboratory (Lowry & Johnson, 1984) and uses the SPUNIT
iterative algorithm (a Russian/PNL multisphere unfolding method, RSICC
PSR-266).  The Python version of ReBUNKI supports the SPUNIT algorithm only
(BON31G is available only in the Fortran version), so algorithmically this
module is the SPUNIT method exposed under the ReBUNKI name with the default
settings recommended by the ReBUNKI documentation: iterations run until the
relative change of the solution falls below a ~1% tolerance (bounded by
``max_iterations``), using the three-point SPUNIT smoothing.

This module is a thin wrapper around :func:`bssunfold.core.unfold_bunki.solve_bunki`
which implements the same SPUNIT scheme; the Detector-facing entry point is
``Detector.unfold_rebunki``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from .unfold_bunki import solve_bunki

from ..utils.validators import validate_system

__all__ = ["solve_rebunki", "unfold_rebunki"]


def solve_rebunki(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    smoothing: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 0.01,
    lethargy_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the ReBUNKI (SPUNIT) algorithm.

    This is the SPUNIT iteration of BUNKI/ReBUNKI; see
    :func:`bssunfold.core.unfold_bunki.solve_bunki` for the algorithm
    description.  The default tolerance matches the ~1% relative-error
    convergence recommended for ReBUNKI.

    Parameters
    ----------
    A : np.ndarray
        Lethargy-weighted response matrix (m x n) as built by the Detector
        class.
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    smoothing : float, optional
        Three-point smoothing factor (default: 0.1).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 0.01).
    lethargy_weights : np.ndarray, optional
        Per-bin lethargy widths. Only needed when ``A`` is supplied as a
        per-bin (non-lethargy-weighted) response matrix.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A, b, x0 = validate_system(A, b, x0=x0)
    return solve_bunki(
        A=A,
        b=b,
        x0=x0,
        smoothing=smoothing,
        max_iterations=max_iterations,
        tolerance=tolerance,
        lethargy_weights=lethargy_weights,
    )


def unfold_rebunki(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    smoothing: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 0.01,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the ReBUNKI (SPUNIT) algorithm.

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
    smoothing : float, optional
        Three-point smoothing factor (default: 0.1).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 0.01).
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
            solve_rebunki,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="ReBUNKI",
        extra_output={
            "smoothing": float(smoothing),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
