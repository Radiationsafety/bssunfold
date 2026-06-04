"""Data validation utilities for bssunfold package.

This module provides functions for validating input data such as
detector readings, energy grids, and spectra.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List

__all__ = [
    "validate_readings",
    "validate_energy_grid",
    "validate_spectrum",
    "validate_response_matrix",
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
            f"Energy grid must have at least {min_points} points, "
            f"got {len(E_MeV)}"
        )
    
    if not np.all(E_MeV > 0):
        raise ValueError("All energy values must be positive")
    
    if not np.all(np.diff(E_MeV) > 0):
        raise ValueError("Energy grid must be strictly increasing")
    
    if Emin is not None and E_MeV[0] < Emin:
        raise ValueError(
            f"Minimum energy {E_MeV[0]} is below allowed minimum {Emin}"
        )
    
    if Emax is not None and E_MeV[-1] > Emax:
        raise ValueError(
            f"Maximum energy {E_MeV[-1]} is above allowed maximum {Emax}"
        )
    
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
    
    if b.ndim != 1:
        raise ValueError(f"b must be 1D array, got {b.ndim}D")
    
    if A.shape[0] != len(b):
        raise ValueError(
            f"Number of rows in A ({A.shape[0]}) must match "
            f"length of b ({len(b)})"
        )
    
    if check_rank:
        rank = np.linalg.matrix_rank(A)
        if rank < min(A.shape):
            warnings.warn(
                f"Response matrix is rank-deficient: rank={rank}, "
                f"shape={A.shape}"
            )
    
    return A, b
