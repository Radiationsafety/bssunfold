"""Helpers for applying a maximum neutron-energy cutoff.

The cutoff is expressed as an upper bound ``ub`` on the per-energy-bin
variables of a quadratic / bounded optimization problem: bins with
``E_MeV > max_neutron_energy`` get ``ub = 0`` (combined with the existing
non-negativity ``lb = 0`` this forces them to zero), while active bins get
``ub = +inf`` (unbounded from above).
"""

from typing import Optional

import numpy as np


def upper_bounds(
    E_MeV: np.ndarray, max_neutron_energy: Optional[float]
) -> np.ndarray:
    """Return the per-bin upper bound array for a QP/optimization solver.

    Active bins (``E_MeV <= max_neutron_energy``) receive ``+inf``; bins above
    the cutoff receive ``0.0``. When ``max_neutron_energy`` is ``None`` all bins
    are unbounded from above (the cutoff is disabled).
    """
    E_MeV = np.asarray(E_MeV, dtype=float)
    ub = np.full(E_MeV.shape[0], np.inf, dtype=float)
    if max_neutron_energy is not None:
        ub[E_MeV > float(max_neutron_energy)] = 0.0
    return ub


def max_energy_mask(
    E_MeV: np.ndarray, max_neutron_energy: Optional[float]
) -> np.ndarray:
    """Boolean mask of energy bins with ``E_MeV <= max_neutron_energy``."""
    E_MeV = np.asarray(E_MeV, dtype=float)
    if max_neutron_energy is None:
        return np.ones(E_MeV.shape[0], dtype=bool)
    return E_MeV <= float(max_neutron_energy)
