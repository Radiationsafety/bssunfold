"""Shared multi-resolution (coarse-to-fine) helpers for unfolding.

These utilities let a coarse unfolding on a reduced energy grid act as a
stable, low-frequency prior that is then prolongated onto the full grid.

Coarsening sums adjacent response-matrix columns so that a coarse spectrum
of bin totals reproduces the same detector readings as the fine one::

    A_coarse[:, k] = sum_{j in bin k} A[:, j]
    A_coarse @ x_coarse == A @ _split_coarse(x_coarse, n)

The coarse spectrum therefore keeps the same units (bin totals) as the fine
one and can be prolongated back without rescaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coarsen_columns(A: np.ndarray, n_coarse: int) -> np.ndarray:
    """Merge adjacent response-matrix columns into ``n_coarse`` bins.

    Columns are summed so that a coarse spectrum of bin totals reproduces
    the same detector readings as the fine one::

        A_coarse[i, k] = sum_{j in bin k} A[i, j]

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    n_coarse : int
        Number of coarse bins (must be <= n).

    Returns
    -------
    np.ndarray
        Coarse response matrix (m x n_coarse).
    """
    m, n = A.shape
    if n_coarse <= 0 or n_coarse > n:
        raise ValueError(f"n_coarse must satisfy 0 < n_coarse <= {n}")
    edges = np.linspace(0, n, n_coarse + 1, dtype=int)
    A_coarse = np.zeros((m, n_coarse), dtype=float)
    for k in range(n_coarse):
        A_coarse[:, k] = np.sum(A[:, edges[k] : edges[k + 1]], axis=1)
    return A_coarse


def _split_coarse(x_coarse: np.ndarray, n: int) -> np.ndarray:
    """Distribute a coarse-bin-total spectrum back onto the fine grid.

    Each coarse-bin total is spread uniformly across the fine bins it
    contains, preserving the total fluence.

    Parameters
    ----------
    x_coarse : np.ndarray
        Coarse spectrum of bin totals (n_coarse,).
    n : int
        Number of fine bins.

    Returns
    -------
    np.ndarray
        Fine-grid spectrum (n,).
    """
    n_coarse = x_coarse.shape[0]
    edges = np.linspace(0, n, n_coarse + 1, dtype=int)
    x = np.zeros(n, dtype=float)
    for k in range(n_coarse):
        lo, hi = edges[k], edges[k + 1]
        width = hi - lo
        if width > 0:
            x[lo:hi] = x_coarse[k] / width
    return x


def _coarse_energy_grid(E: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Geometric-mean energy per coarse bin (arithmetic fallback)."""
    n_coarse = len(edges) - 1
    coarse_E = np.empty(n_coarse, dtype=float)
    for k in range(n_coarse):
        lo, hi = edges[k], edges[k + 1]
        grp = E[lo:hi]
        if grp.size and np.all(grp > 0):
            coarse_E[k] = float(np.exp(np.mean(np.log(grp))))
        elif grp.size:
            coarse_E[k] = float(np.mean(grp))
        else:
            coarse_E[k] = float(E[lo]) if lo < len(E) else 0.0
    return coarse_E


def build_coarse_detector(detector, n_coarse: int):
    """Return a coarse-grid ``Detector`` sharing ``detector``'s readings.

    The coarse response matrix is the column-sum coarsening of the fine one
    (``A_coarse[i, k] = sum_{j in bin k} A[i, j]``) and the coarse energy
    grid is the geometric mean of each group. The coarse detector keeps the
    fine detector's auxiliary setup (dose coefficients, validation, ...) but
    overrides ``E_MeV`` and ``sensitivities`` so its response is exactly the
    coarsened matrix.

    Parameters
    ----------
    detector : Detector
        Original (fine-grid) detector.
    n_coarse : int
        Number of coarse energy bins (``0 < n_coarse <= n_energy_bins``).

    Returns
    -------
    Detector
        Coarse-grid detector with matching detector names.
    """
    from .detector import Detector

    A = np.array([detector.sensitivities[d] for d in detector.detector_names])
    m, n = A.shape
    if n_coarse <= 0 or n_coarse > n:
        raise ValueError(f"n_coarse must satisfy 0 < n_coarse <= {n}")
    edges = np.linspace(0, n, n_coarse + 1, dtype=int)
    A_coarse = _coarsen_columns(A, n_coarse)
    coarse_E = _coarse_energy_grid(detector.E_MeV, edges)

    # Reuse the fine detector's construction so all auxiliary attributes
    # (dose coefficients, validation, ...) stay valid, then switch the grid
    # and the response matrix to the coarsened versions.
    fine_df = pd.DataFrame(
        {
            "E_MeV": detector.E_MeV,
            **{d: detector.sensitivities[d] for d in detector.detector_names},
        }
    )
    coarse = Detector(fine_df)
    coarse.E_MeV = coarse_E
    coarse.sensitivities = {
        d: A_coarse[i].copy() for i, d in enumerate(detector.detector_names)
    }
    return coarse


def prolongate_spectrum(x_coarse: np.ndarray, n_fine: int) -> np.ndarray:
    """Prolongate a coarse-bin-total spectrum onto the fine grid.

    Inverse of :func:`_coarsen_columns` (uniform spread preserving total
    fluence); see :func:`_split_coarse`.

    Parameters
    ----------
    x_coarse : np.ndarray
        Coarse spectrum of bin totals (n_coarse,).
    n_fine : int
        Number of fine energy bins.

    Returns
    -------
    np.ndarray
        Fine-grid spectrum (n_fine,).
    """
    return _split_coarse(x_coarse, n_fine)


__all__ = [
    "_coarsen_columns",
    "_split_coarse",
    "build_coarse_detector",
    "prolongate_spectrum",
]
