"""MLEM unfolding method using ODL (Operator Discretization Library).

This module provides the `unfold_mlem_odl` function which wraps the ODL-based
MLEM solver for use with the Detector class.

Requires the 'odl' package to be installed.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding


def unfold_mlem_odl(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using MLEM with ODL (Operator Discretization Library).

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
        Initial spectrum approximation. If None, uniform spectrum is used.
    tolerance : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iterations : int, optional
        Maximum number of iterations. Default is 1000.
    calculate_errors : bool, optional
        Flag for calculating restoration errors. Default is False.
    noise_level : float, optional
        Noise level for error calculation. Default is 0.01.
    n_montecarlo : int, optional
        Number of Monte Carlo samples for error calculation. Default is 100.
    save_result : bool, optional
        If True, save result to internal history. Default is True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing the spectrum restoration results.
    """
    try:
        import odl
    except ImportError as e:
        raise ImportError(
            "odl is required for ODL-based MLEM unfolding. "
            "Install with: pip install odl"
        ) from e

    # Build system
    selected = [name for name in detector_names if name in readings]

    def solve_wrapper(A, b, **kwargs):
        x0 = kwargs.pop('x0', None)
        if x0 is None:
            x0 = np.ones(A.shape[1]) * 0.5
        
        # Create ODL spaces
        meas_end = max(len(b), 1)
        measurement_space = odl.uniform_discr(0, meas_end, len(b))
        spectrum_space = odl.uniform_discr(
            float(np.min(E_MeV)), float(np.max(E_MeV)), n_energy_bins
        )

        # Initialize spectrum
        x = spectrum_space.element(x0)

        # Create operator
        operator = odl.MatrixOperator(
            A, domain=spectrum_space, range=measurement_space
        )

        y = measurement_space.element(b)

        # Run MLEM
        odl.solvers.mlem(operator, x, y, niter=max_iterations)

        # ODL 1.0 requires .data instead of np.asarray()
        x_opt = np.asarray(x.data)
        x_opt = np.maximum(x_opt, 0)
        
        # Return tuple with iterations and converged flag
        return x_opt, max_iterations, True

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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="MLEM (ODL)",
        extra_output={
            "iterations": max_iterations,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
