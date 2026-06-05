"""Data conversion utilities for bssunfold package.

This module provides functions for converting between different data formats
(DataFrame, dict, numpy arrays) for response functions and spectra.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Tuple, Any

__all__ = [
    "convert_to_dataframe",
    "convert_to_dict",
    "convert_sensitivities_to_matrix",
    "extract_detector_names",
]


def convert_to_dataframe(
    data: Union[pd.DataFrame, Dict[str, Any]],
    energy_column: str = "E_MeV",
) -> pd.DataFrame:
    """Convert response function data to DataFrame format.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        Input data. Can be:
        - pandas DataFrame (returned as copy)
        - dict with 'E_MeV' key and detector names as keys
    energy_column : str, optional
        Name of energy column (default: 'E_MeV').
    
    Returns
    -------
    pd.DataFrame
        DataFrame with energy column and detector columns.
    
    Raises
    ------
    ValueError
        If dict doesn't contain 'E_MeV' key.
    TypeError
        If data type is not supported.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    
    if isinstance(data, dict):
        if energy_column not in data:
            raise ValueError(f"Dictionary must contain '{energy_column}' key")
        return pd.DataFrame(data)
    
    raise TypeError(
        f"data must be DataFrame or dict, got {type(data)}"
    )


def convert_to_dict(
    data: Union[pd.DataFrame, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Convert response function data to dictionary format.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        Input data. Can be DataFrame or dict.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with column names as keys and numpy arrays as values.
    """
    if isinstance(data, dict):
        return {
            k: np.asarray(v) if not isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }
    
    if isinstance(data, pd.DataFrame):
        return {col: data[col].values for col in data.columns}
    
    raise TypeError(
        f"data must be DataFrame or dict, got {type(data)}"
    )


def convert_sensitivities_to_matrix(
    sensitivities: Union[Dict[str, np.ndarray], np.ndarray],
    E_MeV: np.ndarray,
    detector_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert sensitivities to matrix format.
    
    Parameters
    ----------
    sensitivities : Union[Dict[str, np.ndarray], np.ndarray]
        Sensitivities as dict (detector -> array) or 2D array.
    E_MeV : np.ndarray
        Energy grid.
    detector_names : List[str], optional
        Detector names. If None and sensitivities is dict, uses dict keys.
    
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Tuple of (sensitivity_matrix, detector_names).
        Matrix shape is (n_energy, n_detectors).
    
    Raises
    ------
    ValueError
        If array dimensions don't match energy grid.
    """
    if isinstance(sensitivities, dict):
        if detector_names is None:
            detector_names = list(sensitivities.keys())
        
        n_energy = len(E_MeV)
        n_detectors = len(detector_names)
        
        matrix = np.zeros((n_energy, n_detectors))
        for i, name in enumerate(detector_names):
            sens = sensitivities[name]
            if len(sens) != n_energy:
                raise ValueError(
                    f"Sensitivity for '{name}' has length {len(sens)}, "
                    f"expected {n_energy}"
                )
            matrix[:, i] = sens
        
        return matrix, detector_names
    
    if isinstance(sensitivities, np.ndarray):
        if sensitivities.ndim != 2:
            raise ValueError(
                "sensitivities array must be 2D (n_energy, n_detectors)"
            )
        if sensitivities.shape[0] != len(E_MeV):
            raise ValueError(
                f"Number of rows ({sensitivities.shape[0]}) must match "
                f"energy grid length ({len(E_MeV)})"
            )
        
        if detector_names is None:
            detector_names = [
                f"det_{i}" for i in range(sensitivities.shape[1])
            ]
        
        return sensitivities, detector_names
    
    raise TypeError(
        f"sensitivities must be dict or np.ndarray, got {type(sensitivities)}"
    )


def extract_detector_names(
    data: Union[pd.DataFrame, Dict[str, Any]],
    energy_column: str = "E_MeV",
) -> List[str]:
    """Extract detector names from response function data.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        Response function data.
    energy_column : str, optional
        Name of energy column (default: 'E_MeV').
    
    Returns
    -------
    List[str]
        List of detector names.
    """
    if isinstance(data, pd.DataFrame):
        return [col for col in data.columns if col != energy_column]
    
    if isinstance(data, dict):
        return [key for key in data.keys() if key != energy_column]
    
    raise TypeError(
        f"data must be DataFrame or dict, got {type(data)}"
    )
