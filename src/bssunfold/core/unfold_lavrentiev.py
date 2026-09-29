"""Lavrentiev regularization with shift for spectrum unfolding.

This module provides the core ``solve_lavrentiev`` solver and the
``unfold_lavrentiev`` wrapper for use with the Detector class.

The Lavrentiev regularization with shift (also known as shifted Tikhonov
regularization) solves the ill-posed linear system ``A x = b`` by replacing
it with the regularized normal system::

    (A^T A + alpha * I) x = A^T b + alpha * x0,

where ``x0`` is an a-priori guess of the solution (the shift) and
``alpha > 0`` is the regularization parameter.  This is equivalent to
minimizing the functional::

    ||A x - b||^2 + alpha * ||x - x0||^2.

For square matrices the classical Lavrentiev method solves
``(A + alpha * I) x = b + alpha * x0`` directly; the normal-system form
above is the standard generalization that works for rectangular matrices
(m != n) without any zero-padding.
"""

from typing import Any

import numpy as np

from ._base_unfolder import run_unfolding

__all__ = ["solve_lavrentiev", "unfold_lavrentiev"]


def solve_lavrentiev(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Solve unfolding using Lavrentiev regularization with shift.

    Solves the regularized normal system
    ``(A^T A + alpha * I) x = A^T b + alpha * x0``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        A-priori guess of the solution (the shift).  If None, a zero
        vector is used (default: None).
    alpha : float, optional
        Regularization parameter (default: 1.0).

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).

    Raises
    ------
    ValueError
        If ``alpha`` is not positive or if dimensions are incompatible.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    n = A.shape[1]

    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(x0, dtype=float).ravel()
        if len(x0) != n:
            raise ValueError(
                f"x0 length ({len(x0)}) must match number of energy bins ({n})"
            )

    ATA = A.T @ A
    ATb = A.T @ b

    matrix = ATA + alpha * np.eye(n)
    rhs = ATb + alpha * x0

    try:
        x = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(matrix) @ rhs

    return x


def unfold_lavrentiev(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    alpha: float = 1.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using Lavrentiev regularization with shift.

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
        A-priori guess of the solution (the shift).  If None, a zero
        vector is used.
    alpha : float, optional
        Regularization parameter (default: 1.0).
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
    x0_default = np.zeros(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_lavrentiev,
        solve_kwargs={"alpha": alpha},
        method_name="Lavrentiev",
        extra_output={"alpha": alpha},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
        reading_uncertainties=reading_uncertainties,
        reading_covariance=reading_covariance,
        noise_model=noise_model,
        measurement_time=measurement_time,
    )
