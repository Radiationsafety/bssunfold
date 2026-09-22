"""Monte-Carlo uncertainty estimation for unfolding methods.

This module provides a unified function for estimating uncertainty of unfolding
results via Monte-Carlo simulation, eliminating code duplication across the
various unfold_* methods in the Detector class.

Variance reduction techniques (lecture 14 of the MIPT optimization course)
are supported:

- **antithetic variates** — every Gaussian noise draw ``eps`` is paired with
  ``-eps``, halving the number of solver calls for the same sample count and
  cancelling the odd (first-order) part of the unfolding response to noise;
- **control variates** — the first-order perturbation response
  ``x_lin = x_base + pinv(A) (b_noisy - b)`` is used as a control with a
  *known* mean (``E[x_lin] = x_base``); per-bin regression coefficients
  ``beta = Cov(x_mc, x_lin) / Var(x_lin)`` remove the linear part of the
  Monte-Carlo fluctuation, leaving only the genuinely non-linear
  uncertainty.  This is the classic "delta-method as control variate"
  construction.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

_VARIANCE_REDUCTION_MODES = ("none", "antithetic", "control", "both")


def _covariance_factor(cov: np.ndarray) -> np.ndarray:
    """Return a factor ``L`` with ``L @ L.T == cov`` (PSD-safe).

    Uses Cholesky when possible; otherwise falls back to an eigendecom-
    position with negative eigenvalues clipped to zero (nearest PSD
    matrix in the eigenvalue sense), which tolerates slightly
    indefinite user-supplied covariance matrices.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"reading_covariance must be a square matrix, got shape {cov.shape}"
        )
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (cov + cov.T))
        w = np.clip(w, 0.0, None)
        return V @ np.diag(np.sqrt(w))


def monte_carlo_uncertainty(
    func: Callable[..., np.ndarray],
    readings: dict[str, float],
    noise_level: float,
    n_samples: int,
    n_energy_bins: int,
    random_state: int | None = None,
    variance_reduction: str = "none",
    response_matrix: np.ndarray | None = None,
    base_spectrum: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Estimate unfolding uncertainty via Monte-Carlo simulation.

    Perturbs the readings, runs the unfolding function for each noisy
    sample, and computes statistics of the resulting spectra.

    The noise model separates the *statistical* part (counting statistics of
    the readings) from the *systematic* part, which are composed additively
    (``V_b = V_stat + V_syst``):

    Statistical part (first applicable wins):

    1. ``noise_model='poisson'`` — the readings are counting data: expected
       counts ``C_i = b_i * T`` (with ``T = measurement_time``, ``T = 1``
       when readings are already counts) are resampled as
       ``C*_i ~ Poisson(C_i)`` and the noisy reading is ``b*_i = C*_i / T``
       (Poisson counting statistics, ``Var(b_i) = b_i / T``). Requires
       non-negative readings. Antithetic pairing does not apply to the
       Poisson draw; ``n_samples`` full samples are generated.
    2. ``reading_uncertainties`` — absolute 1-sigma per reading
       (dict keyed by reading name, or array in reading order); each
       sample is drawn independently per reading:
       ``b_noisy_i = b_i + N(0, sigma_i)``.
    3. ``noise_level`` (legacy, ``noise_model='gaussian'``) — i.i.d.
       *relative* Gaussian noise: ``b_noisy_i = b_i * (1 + N(0, noise_level))``.

    Systematic part (added on top of the statistical part):

    - ``reading_covariance`` — absolute covariance matrix ``V`` (m x m)
      between the readings; each sample receives an additional
      multivariate Gaussian perturbation ``N(0, V)``. Use this to
      propagate correlated systematic effects (shared calibration,
      response-model systematics).

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
        Ignored when ``noise_model='poisson'`` or ``reading_uncertainties``
        is given.
    n_samples : int
        Number of Monte-Carlo samples.
    n_energy_bins : int
        Number of energy bins in the output spectrum.
    random_state : int, optional
        Random seed for reproducibility. If None, uses default numpy RNG.
    variance_reduction : str, optional
        Variance reduction technique: ``'none'`` (default), ``'antithetic'``,
        ``'control'`` or ``'both'``.  The ``'control'``/``'both'`` modes
        require ``response_matrix`` and use the linearized response as a
        control variate with known mean. Antithetic pairing applies to the
        Gaussian statistical models only.
    response_matrix : np.ndarray, optional
        Response matrix ``A`` (m x n) used to build the linearized control
        variate.  Required for ``'control'`` and ``'both'``.
    base_spectrum : np.ndarray, optional
        Nominal spectrum ``x_base`` (mean of the control variate).  If None,
        the sample mean of the raw MC spectra is used as the reference.
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading (see statistical part
        above).
    reading_covariance : np.ndarray, optional
        Absolute covariance matrix between the readings (see systematic
        part above).
    noise_model : str, optional
        Statistical noise model: ``'gaussian'`` (default) or
        ``'poisson'`` (counting statistics; see above).
    measurement_time : float, optional
        Counting time ``T`` used with ``noise_model='poisson'`` when the
        readings are *rates* (counts per unit time); the expected counts
        are ``b_i * T`` and the sampled counts are rescaled back to rates.
        Ignored for the Gaussian models.
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
        - 'noise_model': statistical noise model used
        - 'variance_reduction': technique used (when not 'none')
        - 'variance_reduction_factor': estimated variance ratio
          raw/corrected per bin mean (control modes only)
    """
    if variance_reduction not in _VARIANCE_REDUCTION_MODES:
        raise ValueError(
            f"variance_reduction must be one of {_VARIANCE_REDUCTION_MODES}"
        )
    if noise_model not in ("gaussian", "poisson"):
        raise ValueError(
            f"noise_model must be 'gaussian' or 'poisson', got {noise_model!r}"
        )
    use_antithetic = variance_reduction in ("antithetic", "both")
    use_control = variance_reduction in ("control", "both")
    if use_control and response_matrix is None:
        raise ValueError("variance_reduction='control'/'both' requires response_matrix")

    rng = np.random.default_rng(random_state)

    keys = list(readings.keys())
    values = np.array([readings[k] for k in keys], dtype=float)
    n_readings = len(keys)

    # --- statistical part: poisson > uncertainties > scalar ---------------
    # Either additive deltas (n_eff, m) or multiplicative noise factors.
    stat_deltas: np.ndarray | None = None
    noise_factors: np.ndarray | None = None
    if noise_model == "poisson":
        scale = float(measurement_time) if measurement_time is not None else 1.0
        if scale <= 0:
            raise ValueError(
                f"measurement_time must be positive, got {measurement_time!r}"
            )
        counts = values * scale
        if np.any(counts < 0):
            raise ValueError(
                "noise_model='poisson' requires non-negative readings "
                "(counts cannot be negative); got "
                f"{dict(zip(keys, values))}"
            )
        # No antithetic pairing for a Poisson draw: generate full samples.
        sampled = rng.poisson(counts, size=(n_samples, n_readings))
        stat_deltas = sampled / scale - values[None, :]
        n_effective = n_samples
    elif reading_uncertainties is not None:
        if isinstance(reading_uncertainties, dict):
            missing = [k for k in keys if k not in reading_uncertainties]
            if missing:
                raise ValueError(
                    "reading_uncertainties is missing entries for: " f"{missing}"
                )
            sigma_arr = np.array([float(reading_uncertainties[k]) for k in keys])
        else:
            sigma_arr = np.asarray(reading_uncertainties, dtype=float)
            if sigma_arr.shape != (n_readings,):
                raise ValueError(
                    f"reading_uncertainties must match the number of "
                    f"readings ({n_readings}), got shape {sigma_arr.shape}"
                )
        n_pairs = max(int(np.ceil(n_samples / 2)), 1) if use_antithetic else n_samples
        eps = rng.standard_normal((n_pairs, n_readings)) * sigma_arr[None, :]
        stat_deltas = (
            np.vstack([eps, -eps])[:n_samples] if use_antithetic else eps
        )
        n_effective = stat_deltas.shape[0]
    else:
        if use_antithetic:
            n_pairs = max(int(np.ceil(n_samples / 2)), 1)
            eps = rng.normal(0, noise_level, size=(n_pairs, n_readings))
            noise_factors = np.vstack([1.0 + eps, 1.0 - eps])[:n_samples]
        else:
            noise_factors = 1.0 + rng.normal(
                0, noise_level, size=(n_samples, n_readings)
            )
        n_effective = noise_factors.shape[0]

    # --- systematic part: covariance, added on top ------------------------
    syst_deltas: np.ndarray | None = None
    if reading_covariance is not None:
        factor = _covariance_factor(reading_covariance)
        syst_deltas = rng.standard_normal((n_effective, n_readings)) @ factor.T

    spectra_samples = np.zeros((n_effective, n_energy_bins))
    control_samples = np.zeros((n_effective, n_energy_bins)) if use_control else None

    # Pre-factor for the linearized control variate: pinv(A) maps the
    # reading perturbation onto the spectrum (delta-method Jacobian).
    A_pinv = np.linalg.pinv(response_matrix) if use_control else None
    b_base = values.copy()

    for i in range(n_effective):
        if stat_deltas is not None:
            noisy_values = values + stat_deltas[i]
        else:
            noisy_values = values * noise_factors[i]
        if syst_deltas is not None:
            noisy_values = noisy_values + syst_deltas[i]
        noisy_readings = {k: float(v) for k, v in zip(keys, noisy_values)}
        spectrum = func(noisy_readings, **kwargs)
        spectra_samples[i] = np.asarray(spectrum, dtype=float)
        if use_control:
            delta_b = noisy_values - b_base
            control_samples[i] = A_pinv @ delta_b

    # --- control-variate correction ----------------------------------------
    vr_factor: float | None = None
    if use_control:
        if base_spectrum is None:
            base_spectrum = np.mean(spectra_samples, axis=0)
        control_centered = control_samples  # E[control] = 0 exactly
        raw_centered = spectra_samples - base_spectrum
        var_c = control_samples.var(axis=0)
        cov = (
            (raw_centered * control_centered).mean(axis=0)
            if n_effective > 1
            else np.zeros(n_energy_bins)
        )
        beta = np.divide(cov, var_c, out=np.zeros(n_energy_bins), where=var_c > 0)
        corrected = spectra_samples - beta[None, :] * control_centered
        vr_factor = float(
            np.mean(
                np.divide(
                    spectra_samples.var(axis=0),
                    corrected.var(axis=0),
                    out=np.ones(n_energy_bins),
                    where=corrected.var(axis=0) > 0,
                )
            )
        )
        spectra_samples = corrected

    return {
        "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
        "spectrum_uncert_std": np.std(spectra_samples, axis=0),
        "spectrum_uncert_min": np.min(spectra_samples, axis=0),
        "spectrum_uncert_max": np.max(spectra_samples, axis=0),
        "spectrum_uncert_median": np.median(spectra_samples, axis=0),
        "spectrum_uncert_percentile_5": np.percentile(spectra_samples, 5, axis=0),
        "spectrum_uncert_percentile_95": np.percentile(spectra_samples, 95, axis=0),
        "spectrum_uncert_all": spectra_samples,
        "noise_model": noise_model,
        **(
            {
                "variance_reduction": variance_reduction,
                "variance_reduction_factor": vr_factor,
            }
            if variance_reduction != "none"
            else {}
        ),
    }


def _add_noise(
    readings: dict[str, float],
    noise_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
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
