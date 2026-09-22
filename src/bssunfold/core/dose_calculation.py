"""Dose rate calculation module for bssunfold package.

This module provides functions for calculating dose rates from neutron
spectra using various conversion coefficient datasets (ICRP-116, ICRP-74,
NRB99-2009, etc.).
"""

import logging
import warnings

import numpy as np

__all__ = [
    "calculate_dose_rates",
    "get_icrp116_coefficients",
    "get_coefficients",
    "interpolate_coefficients",
    "DOSE_COEFFICIENTS_REGISTRY",
    "SPECTRUM_DEFINITION",
    "SPECTRUM_UNITS",
    "INTEGRATION_RULE",
    "default_ln_steps",
    "energy_bin_edges",
]

logger = logging.getLogger(__name__)

# Canonical interpretation of the unfolded spectrum vector: the forward
# model is b_j = sum_i R_j(E_i) * phi_i * dlnE_i, i.e. the spectrum holds a
# differential fluence density per unit d(ln E) (lethargy density).
SPECTRUM_DEFINITION = "differential_fluence_per_dlnE"
SPECTRUM_UNITS = "cm^-2 s^-1 (d ln E)^-1"
INTEGRATION_RULE = "rectangular_midpoint_dlnE"

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
    out_of_range: str = "zero",
) -> dict[str, np.ndarray]:
    """Interpolate conversion coefficients to a target energy grid.

    Uses linear interpolation (np.interp). For energy values outside the
    original range, the behaviour is controlled by ``out_of_range``.

    Parameters
    ----------
    cc : Dict[str, np.ndarray]
        Conversion coefficient dictionary with 'E_MeV' key.
    E_target : np.ndarray
        Target energy grid in MeV.
    fill_value : float, optional
        Value to use outside the original energy range (default: 0.0).
    out_of_range : str, optional
        Behaviour for target energies outside ``[E_source[0], E_source[-1]]``:

        - ``"zero"`` (default): fill with ``fill_value`` (historical
          behaviour; silently zeroes the dose contribution of out-of-range
          fluence).
        - ``"nan"``: fill with NaN so that uncovered contributions are
          visible instead of silently dropped.
        - ``"extrapolate"``: linear extrapolation from the two edge points
          (use with care; conversion coefficients are not physical far
          outside their tabulated range).

    Returns
    -------
    Dict[str, np.ndarray]
        Interpolated conversion coefficients on the target energy grid.
        Additionally, the key ``"_out_of_range_mask"`` holds a boolean array
        marking the target bins that fall outside the source energy range
        (True = out of range). Callers may use it to compute the fraction of
        fluence not covered by the coefficient dataset.

    Examples
    --------
    >>> from bssunfold import get_coefficients, interpolate_coefficients
    >>> cc = get_coefficients("NRB99_2009_effective")
    >>> E_det = np.logspace(-9, 3, 100)  # detector grid
    >>> cc_interp = interpolate_coefficients(cc, E_det)
    """
    if out_of_range not in ("zero", "nan", "extrapolate"):
        raise ValueError(
            f"out_of_range must be 'zero', 'nan' or 'extrapolate', "
            f"got {out_of_range!r}"
        )

    E_source = np.asarray(cc["E_MeV"], dtype=float)
    E_target = np.asarray(E_target, dtype=float)

    result = {"E_MeV": E_target.copy()}

    below = E_target < E_source[0]
    above = E_target > E_source[-1]
    out_mask = below | above
    result["_out_of_range_mask"] = out_mask.copy()

    for key, values in cc.items():
        if key == "E_MeV":
            continue
        values_arr = np.asarray(values, dtype=float)
        if out_of_range == "extrapolate":
            interpolated = _extrap_interp(E_source, values_arr, E_target)
        else:
            interpolated = np.interp(E_target, E_source, values_arr)
            fill = np.nan if out_of_range == "nan" else fill_value
            interpolated[below] = fill
            interpolated[above] = fill
        result[key] = interpolated

    return result


def _extrap_interp(
    E_source: np.ndarray,
    values: np.ndarray,
    E_target: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with linear extrapolation beyond the edges."""
    E_target = np.asarray(E_target, dtype=float)
    interpolated = np.interp(E_target, E_source, values)
    below = E_target < E_source[0]
    above = E_target > E_source[-1]
    if below.any():
        slope_lo = (values[1] - values[0]) / (E_source[1] - E_source[0])
        interpolated[below] = values[0] + slope_lo * (E_target[below] - E_source[0])
    if above.any():
        slope_hi = (values[-1] - values[-2]) / (E_source[-1] - E_source[-2])
        interpolated[above] = values[-1] + slope_hi * (E_target[above] - E_source[-1])
    return interpolated


def default_ln_steps(E_MeV: np.ndarray) -> np.ndarray:
    """Per-bin natural-logarithmic widths ``d(ln E)_i`` from bin centers.

    Uses the central-difference convention of
    ``Detector._convert_rf_to_matrix_variable_step``: edge bins get one-sided
    differences, interior bins get central differences of ``log10(E)``
    (rescaled to natural log). Use only when the per-bin widths were not
    stored on the Detector.
    """
    E = np.asarray(E_MeV, dtype=float)
    n = E.size
    log_e = np.log10(E + 1e-15)
    steps = np.zeros(n)
    if n > 1:
        steps[0] = log_e[1] - log_e[0]
        steps[-1] = log_e[-1] - log_e[-2]
    else:
        steps[0] = 1.0
    if n > 2:
        steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    return steps * np.log(10.0)


def energy_bin_edges(E_MeV: np.ndarray) -> np.ndarray:
    """Build energy bin edges from bin centers.

    Interior edges are geometric midpoints ``sqrt(E_i * E_{i+1})`` (constant
    ratio, matching log-spaced grids); the outer edges extend the edge
    spacing symmetrically in log space. Returns ``n + 1`` edges.
    """
    E = np.asarray(E_MeV, dtype=float)
    n = E.size
    edges = np.empty(n + 1, dtype=float)
    if n == 1:
        edges[0] = E[0] / np.sqrt(10.0)
        edges[1] = E[0] * np.sqrt(10.0)
        return edges
    interior = np.sqrt(E[:-1] * E[1:])
    edges[1:-1] = interior
    edges[0] = E[0] * (E[0] / interior[0])
    edges[-1] = E[-1] * (E[-1] / interior[-1])
    return edges


def calculate_dose_rates(
    spectrum: np.ndarray,
    cc_icrp116: dict[str, np.ndarray] | None = None,
    dlnE: float = 0.2,
    dlnE_array: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate dose rates using conversion coefficients.

    The dose rate in each geometry is computed as the rectangular-rule
    integral over the lethargy grid:

        dose = sum_i h(E_i) * phi_i * dlnE_i

    where ``h`` are the (already interpolated) conversion coefficients and
    ``phi_i`` is the spectrum value in bin ``i`` interpreted as a
    differential fluence per unit ``dlnE`` (lethargy density).

    Parameters
    ----------
    spectrum : np.ndarray
        Unfolded neutron spectrum (fluence per unit dlnE per bin).
    cc_icrp116 : Dict[str, np.ndarray], optional
        Conversion coefficients dictionary. If None, uses ICRP-116 defaults.
        The dictionary must contain an 'E_MeV' key and one or more geometry
        keys (e.g., 'AP', 'PA', 'ISO').
    dlnE : float, optional
        Uniform logarithmic energy step for integration (default: 0.2).
        Ignored when ``dlnE_array`` is provided. Kept for backward
        compatibility: only use this when the spectrum is defined on a
        uniform log10 grid with step ``dlnE`` (note: ``dlnE`` is measured
        in *natural* log units here and is multiplied by ln(10), matching
        the historical convention of this function).
    dlnE_array : np.ndarray, optional
        Per-bin natural-logarithmic bin widths ``d(ln E)_i`` matching the
        detector's energy grid. When provided it takes precedence over the
        scalar ``dlnE`` and is the correct choice for non-uniform grids.

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

    # Per-bin lethargy widths take precedence over the scalar step
    if dlnE_array is not None:
        widths = np.asarray(dlnE_array, dtype=float)
        if widths.shape != (len(spectrum),):
            raise ValueError(
                f"dlnE_array shape {widths.shape} does not match spectrum "
                f"length ({len(spectrum)})"
            )
        factor = widths
    else:
        if dlnE != 0.2:
            warnings.warn(
                "calculate_dose_rates: using a scalar uniform dlnE "
                f"({dlnE}); for non-uniform energy grids pass "
                "dlnE_array (per-bin d(ln E) widths) instead.",
                stacklevel=2,
            )
        factor = np.log(10.0) * dlnE

    spec = np.asarray(spectrum, dtype=float)
    n_spec = len(spec)

    # Batch: stack all CC arrays into a matrix and do a single matmul
    geoms = [g for g in cc_icrp116 if g != "E_MeV" and g != "_out_of_range_mask"]
    if not geoms:
        return {}

    cc_matrix = np.empty((len(geoms), n_spec))
    for idx, geom in enumerate(geoms):
        k_arr = np.asarray(cc_icrp116[geom], dtype=float)
        min_len = min(len(k_arr), n_spec)
        cc_matrix[idx, :min_len] = k_arr[:min_len]
        if min_len < n_spec:
            cc_matrix[idx, min_len:] = 0.0

    # Single matrix-vector multiply: (n_geoms x n_spec) @ (n_spec,) -> (n_geoms,)
    # with per-bin lethargy widths folded into the spectrum first
    doses = cc_matrix @ (spec * factor)

    return {geom: float(doses[idx]) for idx, geom in enumerate(geoms)}
