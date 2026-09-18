"""Dose rate calculation module for bssunfold package.

This module provides functions for calculating dose rates from neutron
spectra using various conversion coefficient datasets (ICRP-116, ICRP-74,
NRB99-2009, etc.).
"""

import logging

import numpy as np

__all__ = [
    "calculate_dose_rates",
    "calculate_dose_rates_with_validation",
    "get_icrp116_coefficients",
    "get_coefficients",
    "interpolate_coefficients",
    "DOSE_COEFFICIENTS_REGISTRY",
]

logger = logging.getLogger(__name__)

# Lazy-loaded ICRP-116 conversion coefficients
ICRP116_COEFFICIENTS: dict[str, np.ndarray] | None = None

# Registry of available dose conversion coefficient datasets
DOSE_COEFFICIENTS_REGISTRY: dict[str, dict[str, np.ndarray]] = {}


def _build_registry() -> None:
    """Build the registry of available dose coefficient datasets."""
    if DOSE_COEFFICIENTS_REGISTRY:
        return

    from ..constants import (
        ICRP74_COEFF_EFFECTIVE_DOSE,
        ICRP74_COEFF_OPERATIONAL_QUANTITIES,
        ICRP116_COEFF_EFFECTIVE_DOSE,
        NRB99_2009_COEFF_EFFECTIVE_DOSE,
    )

    DOSE_COEFFICIENTS_REGISTRY.update(
        {
            "ICRP116": ICRP116_COEFF_EFFECTIVE_DOSE,
            "ICRP74_effective": ICRP74_COEFF_EFFECTIVE_DOSE,
            "NRB99_2009_effective": NRB99_2009_COEFF_EFFECTIVE_DOSE,
            "ICRP74_operational": ICRP74_COEFF_OPERATIONAL_QUANTITIES,
        }
    )


def get_icrp116_coefficients() -> dict[str, np.ndarray]:
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


def get_coefficients(name: str) -> dict[str, np.ndarray]:
    """Get dose conversion coefficients by name.

    Parameters
    ----------
    name : str
        Name of the coefficient dataset. Options:

        - ``"ICRP116"``: ICRP-116 effective dose (AP, PA, LLAT, RLAT, ISO, ROT)
        - ``"ICRP74_effective"``: ICRP-74 effective dose (AP, PA, RLAT, ROT, ISO)
        - ``"NRB99_2009_effective"``: NRB99-2009 effective dose (AP, ISO)
        - ``"ICRP74_operational"``: ICRP-74 operational quantities
          (ADE, PDE0, PDE45, PDE60, PDE75)

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
            f"Unknown dose coefficient name: '{name}'. Available options: {available}"
        )

    return DOSE_COEFFICIENTS_REGISTRY[name]


def interpolate_coefficients(
    cc: dict[str, np.ndarray],
    E_target: np.ndarray,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray]:
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


def calculate_dose_rates_with_validation(
    spectrum: np.ndarray,
    E_MeV: np.ndarray,
    cc_icrp116: dict[str, np.ndarray] | None = None,
    dlnE: float = 0.2,
    truncation_threshold: float = 0.01,
) -> dict:
    """Calculate dose rates with energy range validation and warning system.
    
    This function implements AUDIT FIX #3: it checks if the spectrum extends
    beyond the range of dose conversion coefficients and raises warnings or
    errors accordingly. It also separates effective dose (ICRP-116 geometries)
    from operational dose (H*(10) quantities).
    
    Parameters
    ----------
    spectrum : np.ndarray
        Unfolded neutron spectrum (fluence rate per lethargy bin).
    E_MeV : np.ndarray
        Energy grid in MeV corresponding to the spectrum.
    cc_icrp116 : Dict[str, np.ndarray], optional
        Conversion coefficients dictionary. If None, uses ICRP-116 defaults.
    dlnE : float, optional
        Logarithmic energy step for integration (default: 0.2).
    truncation_threshold : float, optional
        Fraction of total fluence above max coefficient energy that triggers
        a warning (default: 0.01 = 1%).
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'dose_rates': combined legacy dict (deprecated)
        - 'effective_dose_rates': ICRP-116 effective dose (AP, PA, ISO, etc.)
        - 'operational_dose_rates': ICRP-74 operational quantities (H*(10))
        - 'truncation_warning': str or None, warning if spectrum exceeds CC range
        - 'extrapolation_warning': str or None, warning about extrapolation
        - 'valid': bool, False if dose calculation is invalid due to truncation
    
    Raises
    ------
    ValueError
        If more than 50% of fluence lies outside coefficient range.
    """
    if cc_icrp116 is None:
        cc_icrp116 = get_icrp116_coefficients()

    if not cc_icrp116:
        return {
            "dose_rates": {},
            "effective_dose_rates": {},
            "operational_dose_rates": {},
            "truncation_warning": None,
            "extrapolation_warning": None,
            "valid": True,
        }

    spec = np.asarray(spectrum, dtype=float)
    E_spec = np.asarray(E_MeV, dtype=float)
    n_spec = len(spec)
    
    # Calculate total fluence for truncation check
    total_fluence = np.sum(spec)
    
    # Separate effective and operational coefficients
    effective_geoms = []
    operational_geoms = []
    
    for key in cc_icrp116:
        if key == "E_MeV":
            continue
        # Operational quantities typically include H*(10), ADE, PDE*, etc.
        if key in ["ADE", "PDE0", "PDE45", "PDE60", "PDE75", "H_star_10", "H*(10)"]:
            operational_geoms.append(key)
        else:
            effective_geoms.append(key)
    
    # Check for energy range mismatch (AUDIT FIX #3)
    truncation_warning = None
    extrapolation_warning = None
    valid = True
    
    for geom_list, dose_type in [
        (effective_geoms, "effective"),
        (operational_geoms, "operational")
    ]:
        if not geom_list:
            continue
            
        # Get energy range from first geometry (all should have same range)
        sample_key = geom_list[0]
        E_cc = np.asarray(cc_icrp116[sample_key], dtype=float)
        # Coefficients may be stored as values; need to get E_MeV from cc dict
        if "E_MeV" in cc_icrp116:
            E_cc_source = np.asarray(cc_icrp116["E_MeV"], dtype=float)
        else:
            # Fallback: assume same length as coefficient array
            E_cc_source = np.arange(len(E_cc))
        
        E_min_cc = E_cc_source[0]
        E_max_cc = E_cc_source[-1]
        
        # Check spectrum range vs coefficient range
        E_min_spec = E_spec[0]
        E_max_spec = E_spec[-1]
        
        # Calculate fluence fraction outside coefficient range
        fluence_below = 0.0
        fluence_above = 0.0
        
        if E_min_spec < E_min_cc:
            mask_below = E_spec < E_min_cc
            fluence_below = np.sum(spec[mask_below]) / max(total_fluence, 1e-30)
        
        if E_max_spec > E_max_cc:
            mask_above = E_spec > E_max_cc
            fluence_above = np.sum(spec[mask_above]) / max(total_fluence, 1e-30)
        
        # Generate warnings
        if fluence_above > truncation_threshold:
            pct = fluence_above * 100
            truncation_warning = (
                f"{dose_type.title()} dose calculation: {pct:.1f}% of fluence lies "
                f"above {E_max_cc:.1f} MeV. Coefficient dataset truncated at {E_max_cc:.1f} MeV. "
                f"Dose calculation may be significantly underestimated."
            )
            if fluence_above > 0.5:
                valid = False
                raise ValueError(
                    f"Critical: {pct:.1f}% of fluence above coefficient maximum "
                    f"({E_max_cc:.1f} MeV). Dose calculation invalid. "
                    f"Use a coefficient dataset with extended energy range."
                )
        
        if E_max_spec > E_max_cc * 1.01 and fluence_above <= truncation_threshold:
            extrapolation_warning = (
                f"Spectrum extends to {E_max_spec:.1f} MeV, but {dose_type} coefficients "
                f"only go to {E_max_cc:.1f} MeV. High-energy tail zeroed in dose calculation."
            )
    
    # Pre-compute constant factor for lethargy integration
    ln10 = np.log(10.0) * dlnE
    
    # Calculate effective dose rates
    effective_doses = {}
    if effective_geoms:
        cc_matrix = np.empty((len(effective_geoms), n_spec))
        for idx, geom in enumerate(effective_geoms):
            k_arr = np.asarray(cc_icrp116[geom], dtype=float)
            min_len = min(len(k_arr), n_spec)
            cc_matrix[idx, :min_len] = k_arr[:min_len]
            if min_len < n_spec:
                cc_matrix[idx, min_len:] = 0.0  # Silent truncation with warning above
        
        doses = cc_matrix @ spec
        doses *= ln10
        effective_doses = {
            geom: float(doses[idx]) for idx, geom in enumerate(effective_geoms)
        }
    
    # Calculate operational dose rates
    operational_doses = {}
    if operational_geoms:
        cc_matrix = np.empty((len(operational_geoms), n_spec))
        for idx, geom in enumerate(operational_geoms):
            k_arr = np.asarray(cc_icrp116[geom], dtype=float)
            min_len = min(len(k_arr), n_spec)
            cc_matrix[idx, :min_len] = k_arr[:min_len]
            if min_len < n_spec:
                cc_matrix[idx, min_len:] = 0.0
        
        doses = cc_matrix @ spec
        doses *= ln10
        operational_doses = {
            geom: float(doses[idx]) for idx, geom in enumerate(operational_geoms)
        }
    
    # Legacy combined dict (deprecated but kept for backward compatibility)
    combined_doses = {**effective_doses, **operational_doses}
    
    return {
        "dose_rates": combined_doses,
        "effective_dose_rates": effective_doses,
        "operational_dose_rates": operational_doses,
        "truncation_warning": truncation_warning,
        "extrapolation_warning": extrapolation_warning,
        "valid": valid,
    }


def calculate_dose_rates(
    spectrum: np.ndarray,
    cc_icrp116: dict[str, np.ndarray] | None = None,
    dlnE: float = 0.2,
) -> dict[str, float]:
    """Calculate dose rates using conversion coefficients.
    
    .. deprecated:: 
        Use :func:`calculate_dose_rates_with_validation` instead for proper
        energy range checking and separation of effective/operational doses.

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
    result = calculate_dose_rates_with_validation(
        spectrum=spectrum,
        E_MeV=np.logspace(-9, 3, len(spectrum)),  # Default assumption
        cc_icrp116=cc_icrp116,
        dlnE=dlnE,
    )
    return result["dose_rates"]
