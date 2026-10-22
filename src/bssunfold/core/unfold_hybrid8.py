"""Hybrid8 start spectrum for iterative unfolding codes.

The "Hybrid8" spectrum is the standard a-priori (start) spectrum of the
MSANDB/MSAND unfolding code family used for cosmic-ray neutron
spectrometry with Bonner spheres at balloon altitudes:

  - G. Simmer, V. Mares, E. Weitzenegger and W. Rühm, "Iterative unfolding for
    Bonner sphere spectrometers using the MSANDB code - Sensitivity analysis and
    dose calculation", Radiat. Meas. 45, 1-9 (2010).
    doi:10.1016/j.radmeas.2009.03.023

The original Hybrid8 start spectrum is a tabulated multi-group spectrum;
this module provides a smooth parametric analogue following the qualitative
description given by S. Barros, V. Mares, R. Bedogni, M. Reginatto,
A. Esposito, I. F. Gonçalves, P. Vaz and W. Rühm, "Comparison of unfolding
codes for neutron spectrometry with Bonner spheres", Radiat. Prot. Dosimetry
161, 46-52 (2014). doi:10.1093/rpd/nct353: four regions — a thermal peak, a
flat epithermal 1/E region, an evaporation peak at 2 MeV, and a cascade peak
at 100 MeV.
It is intended as a physically-shaped ``x0`` for ratio-iteration methods
(SAND-II / MSANDB-style), where the starting spectrum measurably influences
convergence and the regularizing effect of early stopping.

The shape is defined by anchor points on the fluence-per-logarithmic-energy
convention ``E * Phi(E)`` and interpolated linearly in log-log space, which
reproduces the piecewise power-law character (1/E plateau -> evaporation
Maxwellian -> cascade peak) of the described spectrum.
"""

import numpy as np

__all__ = ["HYBRID8_ANCHORS", "hybrid8_start"]

# (E_MeV, relative E*Phi) anchors describing the four Hybrid8 regions:
# thermal rise/plateau below ~0.1 MeV, evaporation peak at 2 MeV,
# epithermal 1/E valley around 30 MeV, cascade peak at 100 MeV,
# and a stiff power-law decrease toward the GeV range.
HYBRID8_ANCHORS: tuple[tuple[float, float], ...] = (
    (1e-9, 0.010),
    (1e-7, 0.030),
    (1e-5, 0.050),
    (1e-3, 0.100),
    (0.020, 0.300),
    (0.100, 0.500),
    (0.500, 0.850),
    (2.000, 1.000),
    (5.000, 0.550),
    (10.000, 0.350),
    (30.000, 0.220),
    (60.000, 0.550),
    (100.000, 1.000),
    (300.000, 0.200),
    (1000.000, 0.030),
    (2000.000, 0.008),
)


def hybrid8_start(
    E_MeV: np.ndarray | list[float],
    total_fluence: float = 1.0,
) -> np.ndarray:
    """Evaluate the Hybrid8-style start spectrum on an energy grid.

    Parameters
    ----------
    E_MeV : array_like
        Energy grid in MeV (bin centers), as used by ``Detector.E_MeV``.
        The returned values are fluence *per bin* (flux-density convention,
        i.e. ``Phi(E)``), obtained from the ``E * Phi(E)`` anchor shape by
        division by ``E``.
    total_fluence : float, optional
        Fluence rate normalized over the grid so that
        ``sum(spectrum * E_MeV) == total_fluence`` (default: 1.0).

    Returns
    -------
    np.ndarray
        Positive spectrum array with the same length as ``E_MeV``.
    """
    if not np.isfinite(total_fluence) or total_fluence <= 0:
        raise ValueError("total_fluence must be a positive finite number")

    E = np.asarray(E_MeV, dtype=float)
    if E.ndim != 1 or E.size == 0:
        raise ValueError("E_MeV must be a non-empty 1D array")
    if np.any(~np.isfinite(E)) or np.any(E <= 0):
        raise ValueError("E_MeV must contain positive finite energies")

    anchors = np.array(HYBRID8_ANCHORS, dtype=float)
    e_phi = np.exp(np.interp(np.log(E), np.log(anchors[:, 0]), np.log(anchors[:, 1])))
    spectrum = e_phi / E
    spectrum *= total_fluence / float(np.sum(spectrum * E))
    return spectrum
