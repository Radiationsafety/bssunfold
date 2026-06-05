"""Monte-Carlo uncertainty estimation for unfolding methods.

This module provides a unified function for estimating uncertainty of unfolding
results via Monte-Carlo simulation, eliminating code duplication across the
various unfold_* methods in the Detector class.
"""

import numpy as np
from typing import Callable, Dict, Optional, Any


def monte_carlo_uncertainty(
    func: Callable[..., np.ndarray],
    readings: Dict[str, float],
    noise_level: float,
    n_samples: int,
    n_energy_bins: int,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Estimate unfolding uncertainty via Monte-Carlo simulation.

    Adds Gaussian noise to readings, runs the unfolding function for each
    noisy sample, and computes statistics of the resulting spectra.

    Parameters
    ----------
    func : Callable[..., np.ndarray]
        Unfolding function that takes (readings, **kwargs) and returns
        a spectrum array. The function must accept a 'readings' keyword
        or first positional argument as a Dict[str, float].
    readings : Dict[str, float]
        Original detector readings.
    noise_level : float
        Relative noise level (std of Gaussian noise as fraction of value).
    n_samples : int
        Number of Monte-Carlo samples.
    n_energy_bins : int
        Number of energy bins in the output spectrum.
    random_state : int, optional
        Random seed for reproducibility. If None, uses default numpy RNG.
    **kwargs : Any
        Additional keyword arguments passed to *func*.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'spectrum_uncert_mean': mean spectrum
        - 'spectrum_uncert_std': standard deviation
        - 'spectrum_uncert_min': minimum spectrum
        - 'spectrum_uncert_max': maximum spectrum
        - 'spectrum_uncert_median': median spectrum
        - 'spectrum_uncert_percentile_5': 5th percentile
        - 'spectrum_uncert_percentile_95': 95th percentile
        - 'spectrum_uncert_all': all sample spectra (n_samples x n_energy_bins)
    """
    rng = np.random.default_rng(random_state)
    spectra_samples = np.zeros((n_samples, n_energy_bins))

    for i in range(n_samples):
        noisy_readings = _add_noise(readings, noise_level, rng)
        spectrum = func(noisy_readings, **kwargs)
        spectra_samples[i] = np.asarray(spectrum, dtype=float)

    return {
        "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
        "spectrum_uncert_std": np.std(spectra_samples, axis=0),
        "spectrum_uncert_min": np.min(spectra_samples, axis=0),
        "spectrum_uncert_max": np.max(spectra_samples, axis=0),
        "spectrum_uncert_median": np.median(spectra_samples, axis=0),
        "spectrum_uncert_percentile_5": np.percentile(spectra_samples, 5, axis=0),
        "spectrum_uncert_percentile_95": np.percentile(spectra_samples, 95, axis=0),
        "spectrum_uncert_all": spectra_samples,
    }


def _add_noise(
    readings: Dict[str, float],
    noise_level: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Add Gaussian noise to readings using a provided RNG.

    Parameters
    ----------
    readings : Dict[str, float]
        Original readings.
    noise_level : float
        Relative noise level.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    Dict[str, float]
        Noisy readings.
    """
    return {
        key: value * (1 + rng.normal(loc=0, scale=noise_level))
        for key, value in readings.items()
    }