"""Dose rate calculation module for bssunfold package.

This module provides functions for calculating dose rates from neutron
spectra using various conversion coefficient datasets (ICRP-116, ICRP-74,
NRB99-2009, etc.).
"""

import logging
import numpy as np
from typing import Dict, Optional

__all__ = [
    "calculate_dose_rates",
    "get_icrp116_coefficients",
    "get_coefficients",
    "interpolate_coefficients",
    "DOSE_COEFFICIENTS_REGISTRY",
]

logger = logging.getLogger(__name__)

# Lazy-loaded ICRP-116 conversion coefficients
ICRP116_COEFFICIENTS: Optional[Dict[str, np.ndarray]] = None

# Registry of available dose conversion coefficient datasets
DOSE_COEFFICIENTS_REGISTRY: Dict[str, Dict[str, np.ndarray]] = {}


def _build_registry() -> None:
    """Build the registry of available dose coefficient datasets."""
    if DOSE_COEFFICIENTS_REGISTRY:
        return

    from ..constants import (
        ICRP116_COEFF_EFFECTIVE_DOSE,
        ICRP74_COEFF_EFFECTIVE_DOSE,
        NRB99_2009_COEFF_EFFECTIVE_DOSE,
        ICRP74_COEFF_OPERATIONAL_QUANTITIES,
    )

    DOSE_COEFFICIENTS_REGISTRY.update({
        "ICRP116": ICRP116_COEFF_EFFECTIVE_DOSE,
        "ICRP74_effective": ICRP74_COEFF_EFFECTIVE_DOSE,
        "NRB99_2009_effective": NRB99_2009_COEFF_EFFECTIVE_DOSE,
        "ICRP74_operational": ICRP74_COEFF_OPERATIONAL_QUANTITIES,
    })


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


def get_coefficients(name: str) -> Dict[str, np.ndarray]:
    """Get dose conversion coefficients by name.

    Parameters
    ----------
    name : str
        Name of the coefficient dataset. Options:

        - ``"ICRP116"``: ICRP-116 effective dose (AP, PA, LLAT, RLAT, ISO, ROT)
        - ``"ICRP74_effective"``: ICRP-74 effective dose (AP, PA, RLAT, ROT, ISO)
        - ``"NRB99_2009_effective"``: NRB99-2009 effective dose (AP, ISO)
        - ``"ICRP74_operational"``: ICRP-74 operational quantities (ADE, PDE0, PDE45, PDE60, PDE75)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'E_MeV' key and geometry/quantity keys.

    Raises
    ------
    ValueError
        If the requested coefficient name is not found.

    Examples
    --------
    >>> from bssunfold import get_coefficients
    >>> cc = get_coefficients("ICRP74_effective")
    >>> print(list(cc.keys()))
    ['E_MeV', 'AP', 'PA', 'RLAT', 'ROT', 'ISO']
    """
    _build_registry()

    if name not in DOSE_COEFFICIENTS_REGISTRY:
        available = list(DOSE_COEFFICIENTS_REGISTRY.keys())
        raise ValueError(
            f"Unknown dose coefficient name: '{name}'. "
            f"Available options: {available}"
        )

    return DOSE_COEFFICIENTS_REGISTRY[name]


def interpolate_coefficients(
    cc: Dict[str, np.ndarray],
    E_target: np.ndarray,
    fill_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Interpolate conversion coefficients to a target energy grid.

    Uses linear interpolation (np.interp). For energy values outside the
    original range, the ``fill_value`` is used (default: 0.0).

    Parameters
    ----------
    cc : Dict[str, np.ndarray]
        Conversion coefficient dictionary with 'E_MeV' key.
    E_target : np.ndarray
        Target energy grid in MeV.
    fill_value : float, optional
        Value to use outside the original energy range (default: 0.0).

    Returns
    -------
    Dict[str, np.ndarray]
        Interpolated conversion coefficients on the target energy grid.

    Examples
    --------
    >>> from bssunfold import get_coefficients, interpolate_coefficients
    >>> cc = get_coefficients("NRB99_2009_effective")
    >>> E_det = np.logspace(-9, 3, 100)  # detector grid
    >>> cc_interp = interpolate_coefficients(cc, E_det)
    """
    E_source = np.asarray(cc["E_MeV"], dtype=float)
    E_target = np.asarray(E_target, dtype=float)

    result = {"E_MeV": E_target.copy()}

    for key, values in cc.items():
        if key == "E_MeV":
            continue
        values_arr = np.asarray(values, dtype=float)
        interpolated = np.interp(E_target, E_source, values_arr)
        # Clamp to fill_value outside the source range
        below = E_target < E_source[0]
        above = E_target > E_source[-1]
        interpolated[below] = fill_value
        interpolated[above] = fill_value
        result[key] = interpolated

    return result


def calculate_dose_rates(
    spectrum: np.ndarray,
    cc_icrp116: Optional[Dict[str, np.ndarray]] = None,
    dlnE: float = 0.2,
) -> Dict[str, float]:
    """Calculate dose rates using conversion coefficients.

    Uses uniform logarithmic step for integration.

    Parameters
    ----------
    spectrum : np.ndarray
        Unfolded neutron spectrum.
    cc_icrp116 : Dict[str, np.ndarray], optional
        Conversion coefficients dictionary. If None, uses ICRP-116 defaults.
        The dictionary must contain an 'E_MeV' key and one or more geometry
        keys (e.g., 'AP', 'PA', 'ISO').
    dlnE : float, optional
        Logarithmic energy step for integration (default: 0.2).

    Returns
    -------
    Dict[str, float]
        Dictionary of dose rates for each geometry/quantity in the
        conversion coefficients. Values are in pico-Sievert per second
        (pSv/s).
    """
    if cc_icrp116 is None:
        cc_icrp116 = get_icrp116_coefficients()

    if not cc_icrp116:
        return {}

    doserates = {}
    for geom, k in cc_icrp116.items():
        if geom != "E_MeV":
            k_arr = np.asarray(k, dtype=float)
            min_len = min(len(k_arr), len(spectrum))
            integrand = k_arr[:min_len] * spectrum[:min_len] * dlnE
            dose = np.log(10) * np.sum(integrand)
            doserates[geom] = float(dose)

    return doserates
