"""Data validation utilities for bssunfold package.

This module provides functions for validating input data such as
detector readings, energy grids, and spectra.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "validate_readings",
    "validate_energy_grid",
    "validate_spectrum",
    "validate_response_matrix",
    "validate_solver_params",
    "validate_system",
]


def validate_readings(
    readings: Dict[str, float],
    detector_names: List[str],
    allow_zero: bool = True,
) -> Dict[str, float]:
    """Validate detector readings.

    Parameters
    ----------
    readings : Dict[str, float]
        Dictionary of detector readings.
    detector_names : List[str]
        List of valid detector names.
    allow_zero : bool, optional
        If True, zero readings are allowed (default: True).

    Returns
    -------
    Dict[str, float]
        Validated readings dictionary.

    Raises
    ------
    ValueError
        If readings are negative or no valid readings provided.
    TypeError
        If readings is not a dictionary.
    """
    if not isinstance(readings, dict):
        raise TypeError(f"readings must be a dict, got {type(readings)}")

    valid = {}
    for det in detector_names:
        if det in readings:
            val = float(readings[det])
            if np.isnan(val):
                raise ValueError(f"Reading '{det}' is NaN")
            if np.isinf(val):
                raise ValueError(f"Reading '{det}' is infinite")
            if val < 0:
                raise ValueError(f"Reading '{det}' is negative: {val}")
            if val == 0 and not allow_zero:
                raise ValueError(f"Reading '{det}' is zero, which is not allowed")
            valid[det] = val

    if not valid:
        raise ValueError(
            f"No valid detector readings provided. "
            f"Available detectors: {detector_names}"
        )

    return valid


def validate_energy_grid(
    E_MeV: np.ndarray,
    min_points: int = 2,
    Emin: Optional[float] = None,
    Emax: Optional[float] = None,
) -> np.ndarray:
    """Validate energy grid array.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    min_points : int, optional
        Minimum number of energy points (default: 2).
    Emin : float, optional
        Minimum allowed energy. If None, no lower bound.
    Emax : float, optional
        Maximum allowed energy. If None, no upper bound.

    Returns
    -------
    np.ndarray
        Validated energy grid as float64 array.

    Raises
    ------
    ValueError
        If energy grid is invalid (wrong shape, insufficient points, etc.).
    """
    E_MeV = np.asarray(E_MeV, dtype=np.float64)

    if E_MeV.ndim != 1:
        raise ValueError(f"E_MeV must be a 1D array, got {E_MeV.ndim}D")

    if len(E_MeV) < min_points:
        raise ValueError(
            f"Energy grid must have at least {min_points} points, got {len(E_MeV)}"
        )

    if not np.all(E_MeV > 0):
        raise ValueError("All energy values must be positive")

    if not np.all(np.diff(E_MeV) > 0):
        raise ValueError("Energy grid must be strictly increasing")

    if Emin is not None and E_MeV[0] < Emin:
        raise ValueError(f"Minimum energy {E_MeV[0]} is below allowed minimum {Emin}")

    if Emax is not None and E_MeV[-1] > Emax:
        raise ValueError(f"Maximum energy {E_MeV[-1]} is above allowed maximum {Emax}")

    return E_MeV


def validate_spectrum(
    spectrum: np.ndarray,
    E_MeV: np.ndarray,
    allow_negative: bool = False,
) -> np.ndarray:
    """Validate spectrum array against energy grid.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values.
    E_MeV : np.ndarray
        Energy grid.
    allow_negative : bool, optional
        If True, negative spectrum values are allowed (default: False).

    Returns
    -------
    np.ndarray
        Validated spectrum array.

    Raises
    ------
    ValueError
        If spectrum length doesn't match energy grid or contains invalid values.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if spectrum.ndim != 1:
        raise ValueError(f"Spectrum must be 1D array, got {spectrum.ndim}D")

    if len(spectrum) != len(E_MeV):
        raise ValueError(
            f"Spectrum length ({len(spectrum)}) must match "
            f"energy grid length ({len(E_MeV)})"
        )

    if np.any(np.isnan(spectrum)):
        n_nan = int(np.sum(np.isnan(spectrum)))
        raise ValueError(f"Spectrum contains {n_nan} NaN values")

    if np.any(np.isinf(spectrum)):
        n_inf = int(np.sum(np.isinf(spectrum)))
        raise ValueError(f"Spectrum contains {n_inf} infinite values")

    if not allow_negative and np.any(spectrum < 0):
        n_negative = np.sum(spectrum < 0)
        raise ValueError(
            f"Spectrum contains {n_negative} negative values. "
            "Set allow_negative=True to allow negative values."
        )

    return spectrum


def validate_response_matrix(
    A: np.ndarray,
    b: np.ndarray,
    check_rank: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate response matrix and measurement vector.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    check_rank : bool, optional
        If True, check matrix rank (default: False).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Validated A and b arrays.

    Raises
    ------
    ValueError
        If dimensions are incompatible or matrix is rank-deficient.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D array, got {A.ndim}D")

    if A.size == 0:
        raise ValueError("Response matrix A is empty")

    if b.ndim != 1:
        raise ValueError(f"b must be 1D array, got {b.ndim}D")

    if A.shape[0] != len(b):
        raise ValueError(
            f"Number of rows in A ({A.shape[0]}) must match length of b ({len(b)})"
        )

    if np.any(np.isnan(A)) or np.any(np.isnan(b)):
        raise ValueError("Response matrix or measurement vector contains NaN values")

    if np.any(np.isinf(A)) or np.any(np.isinf(b)):
        raise ValueError(
            "Response matrix or measurement vector contains infinite values"
        )

    if check_rank:
        rank = np.linalg.matrix_rank(A)
        if rank < min(A.shape):
            warnings.warn(
                f"Response matrix is rank-deficient: rank={rank}, shape={A.shape}"
            )

    return A, b


def validate_system(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate system matrix, measurement vector, and optional initial guess.

    This is a convenience wrapper used by iterative solvers to perform
    common validation at the start of each ``solve_*`` function.  It checks
    array shapes, dimension compatibility, and optional parameter bounds.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (n,).  If provided, length must match A.shape[1].
    max_iterations : int, optional
        If provided, must be a positive integer.
    tolerance : float, optional
        If provided, must be a positive float.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        Validated (A, b, x0).

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()

    if A.ndim != 2:
        raise ValueError(f"Response matrix A must be 2D, got {A.ndim}D array")
    if A.size == 0:
        raise ValueError("Response matrix A is empty")
    if b.ndim != 1:
        raise ValueError(f"Measurement vector b must be 1D, got {b.ndim}D array")
    if len(b) == 0:
        raise ValueError("Measurement vector b is empty")
    if A.shape[0] != len(b):
        raise ValueError(
            f"Row count of A ({A.shape[0]}) must match length of b ({len(b)})"
        )

    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        if x0.ndim != 1:
            raise ValueError(f"Initial guess x0 must be 1D, got {x0.ndim}D array")
        if len(x0) != A.shape[1]:
            raise ValueError(
                f"Length of x0 ({len(x0)}) must match column count of A ({A.shape[1]})"
            )

    if max_iterations is not None:
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be a positive integer, got {max_iterations!r}"
            )

    if tolerance is not None:
        if not isinstance(tolerance, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"tolerance must be a number, got {type(tolerance).__name__}"
            )
        if float(tolerance) < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    return A, b, x0


def validate_solver_params(
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization_alpha: float = 0.0,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """Validate common solver parameters.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of iterations (must be positive int, default: 1000).
    tolerance : float, optional
        Convergence tolerance (must be positive float, default: 1e-6).
    regularization_alpha : float, optional
        Regularization strength (must be non-negative, default: 0.0).
    noise_level : float, optional
        Relative noise level in [0, 1] range (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (must be positive int, default: 100).
    random_state : int, optional
        Random seed for reproducibility (non-negative int or None).

    Returns
    -------
    Dict[str, Optional[int]]
        Dictionary with validated parameter names and values.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    TypeError
        If any parameter has the wrong type.
    """
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
        raise ValueError(
            f"max_iterations must be a positive integer, got {max_iterations!r}"
        )

    if not isinstance(tolerance, (int, float, np.integer, np.floating)):
        raise TypeError(f"tolerance must be a number, got {type(tolerance).__name__}")
    tolerance = float(tolerance)
    if tolerance <= 0 or tolerance > 1e2:
        raise ValueError(f"tolerance must be in (0, 100] range, got {tolerance}")

    if not isinstance(regularization_alpha, (int, float, np.integer, np.floating)):
        raise TypeError(
            "regularization_alpha must be a number, "
            f"got {type(regularization_alpha).__name__}"
        )
    if float(regularization_alpha) < 0:
        raise ValueError(
            f"regularization_alpha must be non-negative, got {regularization_alpha}"
        )

    if not isinstance(noise_level, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"noise_level must be a number, got {type(noise_level).__name__}"
        )
    noise_level = float(noise_level)
    if noise_level < 0 or noise_level > 1:
        raise ValueError(f"noise_level must be in [0, 1] range, got {noise_level}")

    if not isinstance(n_montecarlo, (int, np.integer)) or n_montecarlo < 0:
        raise ValueError(
            f"n_montecarlo must be a non-negative integer, got {n_montecarlo!r}"
        )

    if random_state is not None:
        if not isinstance(random_state, (int, np.integer)):
            raise TypeError(
                f"random_state must be a non-negative int or None, "
                f"got {type(random_state).__name__}"
            )
        if int(random_state) < 0:
            raise ValueError(f"random_state must be non-negative, got {random_state}")
        random_state = int(random_state)

    return {
        "max_iterations": int(max_iterations),
        "tolerance": tolerance,
        "regularization_alpha": float(regularization_alpha),
        "noise_level": noise_level,
        "n_montecarlo": int(n_montecarlo),
        "random_state": random_state,
    }
