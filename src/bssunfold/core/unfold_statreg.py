"""Statistical Regularization (Turchin's method) unfolding.

This module provides the solve_statreg solver and unfold_statreg wrapper
using the statreg package (Turchin's method of statistical regularization).
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_statreg", "unfold_statreg"]


def solve_statreg(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    E_MeV: Optional[np.ndarray] = None,
    unfoldermethod: str = "EmpiricalBayes",
    regularization: Optional[float] = None,
    basis_name: str = "CubicSplines",
    boundary: Optional[str] = None,
    derivative_degree: int = 2,
) -> np.ndarray:
    """Solve unfolding problem using Turchin's method of statistical regularization.

    Uses the statreg package with Empirical Bayes or User-specified regularization.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    E_MeV : np.ndarray, optional
        Energy grid for basis construction.
    unfoldermethod : str, optional
        Regularization method: 'EmpiricalBayes' or 'User' (default: 'EmpiricalBayes').
    regularization : float, optional
        Regularization parameter for 'User' method (default: 1e-4).
    basis_name : str, optional
        Basis type: 'CubicSplines' (default).
    boundary : str, optional
        Boundary condition: None or 'dirichlet'.
    derivative_degree : int, optional
        Derivative degree for regularization (1, 2, 3), default: 2.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    try:
        from statreg.model import GaussErrorMatrixUnfolder
        from statreg.basis import CubicSplines
    except ImportError as e:
        raise ImportError(
            "statreg is required for unfold_statreg. "
            "Install with: pip install statreg"
        ) from e

    n_detectors = A.shape[0]
    b_err = b * 0.05

    if E_MeV is not None:
        Emin = np.min(E_MeV)
        if basis_name == "CubicSplines":
            basis = CubicSplines(np.log10(E_MeV / Emin), boundary=boundary)
        else:
            basis = CubicSplines(np.log10(E_MeV / Emin), boundary=boundary)
    else:
        x_vals = np.logspace(-9, 2, A.shape[1])
        Emin = np.min(x_vals)
        basis = CubicSplines(np.log10(x_vals / Emin), boundary=boundary)

    omega = basis.omega(derivative_degree)

    if unfoldermethod == "EmpiricalBayes":
        model = GaussErrorMatrixUnfolder(omega, method=unfoldermethod)
    elif unfoldermethod == "User":
        alpha = 1e-4 if regularization is None else regularization
        model = GaussErrorMatrixUnfolder(omega, method=unfoldermethod, alphas=alpha)
    else:
        raise ValueError(f"Unknown method: {unfoldermethod}")

    result = model.solve(A, b, b_err)

    return np.asarray(result.phi, dtype=float)


def unfold_statreg(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    unfoldermethod: str = "EmpiricalBayes",
    regularization: Optional[float] = None,
    basis_name: str = "CubicSplines",
    boundary: Optional[str] = None,
    derivative_degree: int = 2,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Turchin's statistical regularization.

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
    unfoldermethod : str, optional
        Regularization method (default: 'EmpiricalBayes').
    regularization : float, optional
        Regularization parameter for 'User' method.
    basis_name : str, optional
        Basis type (default: 'CubicSplines').
    boundary : str, optional
        Boundary condition (default: None).
    derivative_degree : int, optional
        Derivative degree (default: 2).
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
            solve_statreg,
            E_MeV=E_MeV,
            unfoldermethod=unfoldermethod,
            regularization=regularization,
            basis_name=basis_name,
            boundary=boundary,
            derivative_degree=derivative_degree,
        ),
        solve_kwargs={},
        method_name="StatReg",
        extra_output={
            "unfoldermethod": unfoldermethod,
            "basis_name": basis_name,
            "derivative_degree": derivative_degree,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
