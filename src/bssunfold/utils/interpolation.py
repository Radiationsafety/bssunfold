"""Interpolation utilities for bssunfold package.

This module provides functions for interpolating spectra onto different
energy grids using PCHIP interpolation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator as pchip
from typing import Union, Dict, Optional, Tuple, Any

__all__ = [
    "interpolate_spectrum",
    "discretize_spectra",
    "resample_to_log_grid",
]


def interpolate_spectrum(
    spectrum: np.ndarray,
    E_from: np.ndarray,
    E_to: np.ndarray,
    fill_value: float = 0.0,
    replace_negative: bool = True,
) -> np.ndarray:
    """Interpolate spectrum from one energy grid to another.
    
    Uses PCHIP interpolation to preserve monotonicity and avoid oscillations.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values on source grid.
    E_from : np.ndarray
        Source energy grid.
    E_to : np.ndarray
        Target energy grid.
    fill_value : float, optional
        Value for extrapolated points (default: 0.0).
    replace_negative : bool, optional
        If True, replace negative interpolated values with 0 (default: True).
    
    Returns
    -------
    np.ndarray
        Interpolated spectrum on target grid.
    """
    # Convert to logarithmic scale
    Emin = np.min(E_from)
    u_from = np.log10(E_from / Emin)
    u_to = np.log10(E_to / Emin)
    
    # Create PCHIP interpolator
    interpolator = pchip(u_from, spectrum)
    
    # Interpolate
    interp_vals = interpolator(u_to)
    
    # Handle extrapolation
    mask_extrapolate = (u_to < u_from.min()) | (u_to > u_from.max())
    interp_vals[mask_extrapolate] = fill_value
    
    # Replace negative values
    if replace_negative:
        interp_vals = np.maximum(interp_vals, 0)
    
    return interp_vals


def discretize_spectra(
    spectra: Union[pd.DataFrame, Dict[str, np.ndarray]],
    target_E_MeV: np.ndarray,
    energy_column: str = "E_MeV",
    Emin: Optional[float] = None,
) -> pd.DataFrame:
    """Discretize spectra onto target energy grid.
    
    Parameters
    ----------
    spectra : Union[pd.DataFrame, Dict[str, np.ndarray]]
        Input spectra. Can be:
        - DataFrame with energy column and spectrum columns
        - dict with 'E_MeV' and spectrum keys
    target_E_MeV : np.ndarray
        Target energy grid.
    energy_column : str, optional
        Name of energy column (default: 'E_MeV').
    Emin : float, optional
        Minimum energy for log scaling. If None, uses min of source grid.
    
    Returns
    -------
    pd.DataFrame
        Discretized spectra with 'E_MeV' column and spectrum columns.
    """
    # Convert to DataFrame if dict
    if isinstance(spectra, dict):
        spectra_df = pd.DataFrame(spectra)
    elif isinstance(spectra, pd.DataFrame):
        spectra_df = spectra.copy()
    else:
        raise TypeError("spectra must be DataFrame or dict")
    
    # Extract source energies and spectra
    if energy_column in spectra_df.columns:
        E_from = spectra_df[energy_column].values
        spec_columns = [c for c in spectra_df.columns if c != energy_column]
    else:
        E_from = spectra_df.iloc[:, 0].values
        spec_columns = spectra_df.columns[1:]
    
    # Determine Emin
    if Emin is None:
        Emin = np.min(E_from)
    
    # Convert to log scale
    u_from = np.log10(E_from / Emin)
    u_to = np.log10(target_E_MeV / Emin)
    
    # Create result DataFrame
    result = pd.DataFrame({"E_MeV": target_E_MeV})
    
    # Interpolate each spectrum
    for col in spec_columns:
        spec_data = spectra_df[col].values
        interpolator = pchip(u_from, spec_data)
        interp_vals = interpolator(u_to)
        
        # Handle extrapolation and negatives
        mask_extrapolate = (u_to < u_from.min()) | (u_to > u_from.max())
        interp_vals[mask_extrapolate] = 0.0
        interp_vals = np.maximum(interp_vals, 0)
        
        result[col] = interp_vals
    
    return result


def resample_to_log_grid(
    spectrum: np.ndarray,
    E_MeV: np.ndarray,
    n_points: Optional[int] = None,
    Emin: Optional[float] = None,
    Emax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample spectrum to uniform logarithmic grid.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum.
    E_MeV : np.ndarray
        Input energy grid.
    n_points : int, optional
        Number of points in output grid. If None, uses input length.
    Emin : float, optional
        Minimum energy. If None, uses min of input grid.
    Emax : float, optional
        Maximum energy. If None, uses max of input grid.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (new_E_MeV, new_spectrum).
    """
    if n_points is None:
        n_points = len(E_MeV)
    
    if Emin is None:
        Emin = np.min(E_MeV)
    if Emax is None:
        Emax = np.max(E_MeV)
    
    # Create uniform log grid
    log_Emin = np.log10(Emin)
    log_Emax = np.log10(Emax)
    log_E_new = np.linspace(log_Emin, log_Emax, n_points)
    E_new = 10 ** log_E_new
    
    # Interpolate spectrum
    new_spectrum = interpolate_spectrum(spectrum, E_MeV, E_new)
    
    return E_new, new_spectrum
