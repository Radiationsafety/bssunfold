"""Dose rate calculation module for bssunfold package.

This module provides functions for calculating dose rates from neutron
spectra using ICRP-116 conversion coefficients.
"""

import numpy as np
from typing import Dict, Optional

__all__ = [
    "calculate_dose_rates",
    "get_icrp116_coefficients",
]

# Default ICRP-116 conversion coefficients (will be imported from constants)
ICRP116_COEFFICIENTS: Optional[Dict[str, np.ndarray]] = None


def get_icrp116_coefficients() -> Dict[str, np.ndarray]:
    """Get ICRP-116 conversion coefficients.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of conversion coefficients for different geometries.
    """
    global ICRP116_COEFFICIENTS
    
    if ICRP116_COEFFICIENTS is None:
        try:
            from ..constants import ICRP116_COEFF_EFFECTIVE_DOSE
            ICRP116_COEFFICIENTS = ICRP116_COEFF_EFFECTIVE_DOSE
        except ImportError:
            ICRP116_COEFFICIENTS = {}
    
    return ICRP116_COEFFICIENTS


def calculate_dose_rates(
    spectrum: np.ndarray,
    cc_icrp116: Optional[Dict[str, np.ndarray]] = None,
    dlnE: float = 0.2,
) -> Dict[str, float]:
    """Calculate dose rates using ICRP-116 conversion coefficients.
    
    Uses uniform logarithmic step for integration.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Unfolded neutron spectrum.
    cc_icrp116 : Dict[str, np.ndarray], optional
        ICRP-116 conversion coefficients. If None, uses default coefficients.
    dlnE : float, optional
        Logarithmic energy step for integration (default: 0.2).
    
    Returns
    -------
    Dict[str, float]
        Dictionary of dose rates for different geometries:
        - 'AP': Anterior-Posterior
        - 'PA': Posterior-Anterior
        - 'LLAT': Left Lateral
        - 'RLAT': Right Lateral
        - 'ISO': Isotropic
        - 'ROT': Rotational
        Values are in pico-Sievert per second (pSv/s).
    """
    if cc_icrp116 is None:
        cc_icrp116 = get_icrp116_coefficients()
    
    if not cc_icrp116:
        return {
            geom: 0.0
            for geom in ["AP", "PA", "LLAT", "RLAT", "ISO", "ROT"]
        }
    
    doserates = {}
    for geom, k in cc_icrp116.items():
        if geom != "E_MeV":
            # Ensure arrays have compatible lengths
            min_len = min(len(k), len(spectrum))
            integrand = k[:min_len] * spectrum[:min_len] * dlnE
            dose = np.log(10) * np.sum(integrand)
            doserates[geom] = float(dose)  # pSv/s
    
    return doserates
