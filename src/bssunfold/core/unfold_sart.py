"""SART (simultaneous algebraic reconstruction technique) unfolding.

Port of the ``SART`` reconstruction algorithm from PyTomography
(https://github.com/PyTomography/PyTomography, MIT license) adapted to
neutron spectrum unfolding. SART is a relaxed weighted least-squares
algebraic reconstruction:

    x^{n+1} = x^n + alpha(n)/(A^T 1 + eps) * A^T ( (b - A x^n) / (A 1 + eps) )

The residual is normalised by the forward-projected unit image (``A 1``) and
the update by the back-projected unit image (``A^T 1``); the relaxation
sequence ``alpha(n)`` controls the step size.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ..utils.validators import validate_system

__all__ = ["solve_sart", "unfold_sart"]


def solve_sart(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    relaxation: Optional[Union[float, Callable[[int], float]]] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using SART.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,). The value of the first (lowest-energy)
        bin is held fixed during iteration because the detector response is
        (near-)zero there and cannot constrain it.
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    relaxation : float or callable, optional
        Relaxation sequence ``alpha(n)`` as a constant or as a callable of
        the iteration number. If None, a constant 0.8 is used (default: None).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A, b, x0 = validate_system(A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    if relaxation is None:

        def relaxation_seq(_n):
            return 0.8
    elif callable(relaxation):
        relaxation_seq = relaxation
    else:
        relax_val = float(relaxation)

        def relaxation_seq(_n):
            return relax_val

    eps = 1e-11
    x = np.maximum(x0, 0).copy()
    # The lowest-energy bin has (near-)zero detector response, so the
    # additive update cannot constrain it; hold it at the initial-guess
    # value (0 by default) to avoid a spurious tail at the spectrum edge.
    x0_first = float(x[0])
    norm_back = A.sum(axis=0)  # A^T 1
    norm_forward = A.sum(axis=1)  # A 1
    converged = False
    iterations = 0

    # Pre-allocate work arrays
    Ax = np.empty(b.shape[0])
    residual = np.empty_like(b)
    update = np.empty_like(x)
    x_old = x.copy()
    norm_back_safe = norm_back + eps
    norm_forward_safe = norm_forward + eps

    for it in range(1, max_iterations + 1):
        iterations = it
        x_old_norm = np.linalg.norm(x_old)
        alpha = float(relaxation_seq(it))

        np.dot(A, x, out=Ax)
        np.subtract(b, Ax, out=residual)
        residual /= norm_forward_safe
        np.dot(A.T, residual, out=update)
        x += alpha * update / norm_back_safe
        np.maximum(x, 0.0, out=x)
        x[0] = x0_first

        # Convergence check using the old norm (already computed)
        rel = np.linalg.norm(x - x_old) / (x_old_norm + eps)
        if rel < tolerance:
            converged = True
            break
        np.copyto(x_old, x)

    return x, iterations, converged


def unfold_sart(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    relaxation: Optional[Union[float, Callable[[int], float]]] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the SART algorithm.

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
        Relative change tolerance for early stopping (default: 1e-6).
    relaxation : float or callable, optional
        Relaxation sequence (default: None -> constant 0.8).
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
    x0_default[0] = 0.0

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
            solve_sart,
            max_iterations=max_iterations,
            tolerance=tolerance,
            relaxation=relaxation,
        ),
        solve_kwargs={},
        method_name="SART",
        extra_output={
            "relaxation": relaxation if isinstance(relaxation, float) else None,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
