"""Tikhonov regularization with Legendre polynomial basis unfolding.

This module provides the core solve_tikhonov_legendre solver and the
unfold_tikhonov_legendre wrapper for use with the Detector class.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_tikhonov_legendre", "unfold_tikhonov_legendre"]


def _build_legendre_basis(n_energy: int, n_polynomials: int) -> np.ndarray:
    """Build Legendre polynomial basis matrix."""
    x = np.linspace(-1, 1, n_energy)
    basis = np.zeros((n_energy, n_polynomials))
    for i in range(n_polynomials):
        coeffs = np.zeros(i + 1)
        coeffs[-1] = 1.0
        basis[:, i] = Legendre(coeffs)(x)
    return basis


def solve_tikhonov_legendre(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    delta: float = 0.05,
    n_polynomials: int = 15,
) -> np.ndarray:
    """Solve unfolding using Tikhonov regularization with Legendre basis.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    delta : float, optional
        Regularization parameter (default: 0.05).
    n_polynomials : int, optional
        Number of Legendre polynomials (default: 15).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    n_energy = A.shape[1]
    basis = _build_legendre_basis(n_energy, n_polynomials)

    A_proj = A @ basis
    L = create_derivative_matrix(n_polynomials, order=2).toarray()

    n_total = A_proj.shape[0] + L.shape[0]
    M = np.vstack([A_proj, delta * L])
    rhs = np.zeros(n_total)
    rhs[:len(b)] = b

    try:
        c, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(M, rhs, rcond=1e-8)[0]

    spectrum = basis @ c[:n_polynomials]
    return np.maximum(spectrum, 0)


def unfold_tikhonov_legendre(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    delta: float = 0.05,
    n_polynomials: int = 15,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Tikhonov regularization with Legendre basis.

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
        Not used (provided for API compatibility).
    delta : float, optional
        Regularization parameter (default: 0.05).
    n_polynomials : int, optional
        Number of Legendre polynomials (default: 15).
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
            solve_tikhonov_legendre,
            delta=delta,
            n_polynomials=n_polynomials,
        ),
        solve_kwargs={},
        method_name="Tikhonov_Legendre",
        extra_output={
            "delta": delta,
            "n_polynomials": n_polynomials,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
