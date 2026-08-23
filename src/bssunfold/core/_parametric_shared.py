"""Shared constants and fit helpers for the parametric unfolding modules.

Extracted from ``unfold_parametric.py`` / ``unfold_parametric2.py`` so the
BON95 family (``_bon95``) and the remaining pipeline code can share them
without circular imports. This module is a leaf: it imports nothing from
the other parametric modules.
"""

import logging
import warnings

import numpy as np

# Fixed constants from the BON95 papers
_Tth = 3.5e-8  # Thermal peak temperature (MeV), = 0.035 eV

# Energy region boundaries for component masking
_THERMAL_MAX_BON95 = 0.1  # MeV — thermal + epithermal dominant
_FAST_MIN_BON95 = 0.1  # MeV — fast component starts

_RESIDUAL_WARN_THRESHOLD = 10.0

logger = logging.getLogger(__name__)


def _check_fit_quality(residual_norm, b_readings, method_name="parametric2"):
    """Emit a warning if the fit residual is large."""
    b_norm = np.linalg.norm(b_readings)
    if b_norm > 0:
        relative_residual = residual_norm / b_norm
        if relative_residual > _RESIDUAL_WARN_THRESHOLD:
            warnings.warn(
                f"{method_name}: large residual "
                f"({residual_norm:.2e} / {b_norm:.2e} = {relative_residual:.1f}x). "
                f"The 4-component BON95 model may not represent this spectrum well.",
                UserWarning,
                stacklevel=3,
            )


def _clean_edge_bins(phi: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Zero out edge bins if anomalously large compared to neighbors.

    The BON95 model divides by E, which can create spikes at the first
    energy bin (tiny E). This function detects and removes such artifacts.

    Parameters
    ----------
    phi : np.ndarray
        Spectrum array.
    factor : float
        Threshold: if edge bin > factor * mean of valid neighbors, zero it.

    Returns
    -------
    np.ndarray
        Cleaned spectrum (new copy).
    """
    phi = np.copy(phi)
    n = len(phi)
    if n < 3:
        return phi

    # Check first bin
    neighbor_mean = np.mean(phi[1:3])
    if neighbor_mean > 0 and phi[0] > factor * neighbor_mean:
        phi[0] = 0.0

    # Check last bin
    neighbor_mean = np.mean(phi[-3:-1])
    if neighbor_mean > 0 and phi[-1] > factor * neighbor_mean:
        phi[-1] = 0.0

    return phi


def _build_measurement_uncertainties(
    b_readings: np.ndarray,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Estimate measurement uncertainties from readings.

    Uses a default relative uncertainty (noise_level) if not provided.
    """
    return np.abs(b_readings) * noise_level + 1e-30
