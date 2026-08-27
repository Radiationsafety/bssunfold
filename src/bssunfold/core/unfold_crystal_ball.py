"""CRYSTAL BALL unfolding method for neutron spectrum reconstruction.

This is an independent open-source reimplementation of the CRYSTAL BALL
algorithm following its published mathematical description (delta-operator
approximation; Kam & Stallmann, and the 1981 review of unfolding codes).
The original CRYSTAL BALL code is a proprietary package. This implementation
is built solely from the
published algorithmic description and does not use or reproduce any
proprietary source code.

CRYSTAL BALL unfolds the spectrum *directly* (without iteration) by
representing the unknown spectrum ``phi`` as a linear combination of the
detector response functions (the rows of the response matrix ``R``):

    phi_j = sum_i alpha_i R_ij

Substituting into the measurement equation ``b_i = sum_j R_ij phi_j`` gives

    b = (R R^T) alpha  =>  alpha = (R R^T + lambda I)^-1 b

and the recovered spectrum is ``phi = R^T alpha``. This is equivalent to
approximating the delta operator ``delta(E - E0)`` by a linear combination
of the integral operators ``int R_i(E) dE``, which is the essence of the
CRYSTAL BALL approach.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_crystal_ball", "unfold_crystal_ball"]


def solve_crystal_ball(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    regularization: float = 0.0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the CRYSTAL BALL algorithm.

    The spectrum is approximated as a linear combination of the detector
    response functions (rows of ``A``). The coefficient vector ``alpha`` is
    obtained from the (regularized) normal equations
    ``(A A^T + lambda I) alpha = b`` and the spectrum reconstructed as
    ``phi = A^T alpha``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Unused by the direct CRYSTAL BALL method; accepted for a uniform
        solver signature.
    regularization : float, optional
        Tikhonov regularization strength ``lambda`` added to the diagonal of
        ``A A^T`` to stabilise the inversion of the (usually ill-conditioned)
        Gram matrix (default: 0.0).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, 1, True). CRYSTAL BALL is a single-step method,
        so ``iterations`` is 1 and ``converged`` is always True.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.size == 0 or b.size == 0:
        raise ValueError("Response matrix and measurements must be non-empty")
    if np.all(b <= 0):
        raise ValueError("All measurements are zero or negative")

    m = A.shape[0]
    # Gram matrix of the detector response functions.
    G = A @ A.T
    if regularization > 0:
        G = G + regularization * np.eye(m)

    # Solve the (regularized) normal equations for the combination weights.
    alpha = np.linalg.solve(G, b)
    spectrum = A.T @ alpha

    return np.maximum(spectrum, 0.0), 1, True


def unfold_crystal_ball(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    regularization: float = 0.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the CRYSTAL BALL algorithm.

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
        Unused; accepted for interface uniformity.
    regularization : float, optional
        Tikhonov regularization strength for the Gram-matrix inversion
        (default: 0.0).
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
            solve_crystal_ball,
            regularization=regularization,
        ),
        solve_kwargs={},
        method_name="CRYSTAL_BALL",
        extra_output={"regularization": float(regularization)},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
