"""Bayesian MCMC unfolding method using PyMC and NUTS sampler.

This module implements a full Bayesian approach to neutron spectrum unfolding
using Markov Chain Monte Carlo (MCMC) methods, specifically the No-U-Turn
Sampler (NUTS), which is an adaptive variant of Hamiltonian Monte Carlo (HMC).

The spectrum is modelled on the log scale with a smoothness (Ornstein-Uhlenbeck)
prior anchored on a data-driven center (the non-negative least-squares solution,
or a user-supplied ``initial_spectrum``).  This keeps the severely
underdetermined unfolding problem well behaved: the spectrum stays positive,
smooth and bounded in the null space of the response matrix, and the posterior
mean matches deterministic solvers (e.g. ``unfold_cvxpy``) on the IAEA
reference-spectrum database.

The Bayesian framework provides:
- Full posterior distributions for each energy bin
- Uncertainty quantification via credible intervals (HPD intervals)
- Automatic regularization through prior specification
- Hierarchical modeling capabilities for the likelihood noise

Key advantages over point-estimate methods:
1. Uncertainty estimation: 95% credible intervals for each spectral bin
2. Automatic parameter tuning: Hyperparameters can be inferred from data
3. Flexible modeling: Easy incorporation of complex priors (e.g., Gaussian Processes)
4. Robustness: Better handling of ill-posed problems through proper regularization
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import run_unfolding

__all__ = ["solve_bayesian_mcmc", "unfold_mcmc"]

# ---------------------------------------------------------------------------
# Lazy PyMC/ArviZ loading (PEP 562 module __getattr__)
#
# ``pymc``/``arviz`` are heavy (PyTensor JIT compilation machinery, several
# seconds of import time).  Importing them eagerly at module scope made every
# ``import bssunfold`` pay that cost even for users who never touch MCMC.
# Instead they are imported on first attribute access.  The historical
# module-level names (``pm``, ``az``, ``PYMC_AVAILABLE``) keep working: tests
# monkeypatch them directly and the loader caches results in the module
# namespace so repeated lookups are free.
# ---------------------------------------------------------------------------

_pm = None
_az = None
_pymc_checked = False


def _load_pymc() -> Any:
    """Import pymc/arviz on first use; cache result in the module globals.

    Returns
    -------
    Tuple[Optional[Any], Optional[Any]]
        The ``(pm, az)`` modules, or ``(None, None)`` when unavailable.
    """
    global _pm, _az, _pymc_checked
    if not _pymc_checked:
        try:
            import arviz as _az_mod
            import pymc as _pm_mod

            _pm, _az = _pm_mod, _az_mod
        except Exception:
            _pm, _az = None, None
        _pymc_checked = True
    return _pm, _az


def __getattr__(name: str) -> Any:
    if name == "pm":
        pm_mod, _ = _load_pymc()
        globals()["pm"] = pm_mod
        return pm_mod
    if name == "az":
        _, az_mod = _load_pymc()
        globals()["az"] = az_mod
        return az_mod
    if name == "PYMC_AVAILABLE":
        pm_mod, _ = _load_pymc()
        available = pm_mod is not None
        globals()["PYMC_AVAILABLE"] = available
        return available
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _resolve_backends() -> Tuple[Any, Any]:
    """Return the currently active ``(pm, az)`` modules.

    Reads the module namespace first so that externally patched stand-ins
    (e.g. test doubles) take precedence over the lazily imported real
    packages; otherwise triggers :func:`_load_pymc` and caches the result.
    """
    g = globals()
    if "pm" not in g or "az" not in g:
        _load_pymc()
        g.setdefault("pm", _pm)
        g.setdefault("az", _az)
    return g["pm"], g["az"]


def _check_pymc_available() -> bool:
    """Report PyMC availability, honoring an externally patched flag."""
    g = globals()
    if "PYMC_AVAILABLE" in g:
        return bool(g["PYMC_AVAILABLE"])
    pm_mod, _ = _resolve_backends()
    return pm_mod is not None


def _ou_correlation_cholesky(n_bins: int, lengthscale: float) -> np.ndarray:
    """Cholesky factor of the Ornstein-Uhlenbeck correlation matrix.

    The OU correlation ``C[i, j] = exp(-|i - j| / lengthscale)`` yields smooth,
    stationary prior draws that stay bounded in amplitude (unlike a pure random
    walk), which keeps the highly underdetermined unfolding posterior well
    behaved for NUTS.

    Parameters
    ----------
    n_bins : int
        Number of energy bins.
    lengthscale : float
        Correlation length in units of energy bins.

    Returns
    -------
    np.ndarray
        Lower-triangular Cholesky factor of shape (n_bins, n_bins).
    """
    idx = np.arange(n_bins)
    corr = np.exp(-np.abs(idx[:, None] - idx[None, :]) / max(float(lengthscale), 1e-9))
    return np.linalg.cholesky(corr + 1e-9 * np.eye(n_bins))


def _prior_center(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    initial_spectrum: Optional[np.ndarray],
    n_energy: int,
) -> np.ndarray:
    """Data-driven log-space prior center for the spectrum.

    Uses the user-supplied ``initial_spectrum`` when available, otherwise the
    non-negative least-squares solution of ``A @ x = b``.  The center is
    returned on the log scale (``log(max(x, eps))``) so the spectrum prior
    ``f = exp(theta)`` is anchored near a spectrum consistent with the measured
    readings.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    initial_spectrum : Optional[np.ndarray]
        User-provided prior guess (None to use the least-squares center).
    n_energy : int
        Number of energy bins.

    Returns
    -------
    np.ndarray
        Log-scale prior center of shape (n_energy,).
    """
    if initial_spectrum is not None:
        center = np.maximum(np.asarray(initial_spectrum, dtype=float), 0.0)
        if center.ndim != 1 or len(center) != n_energy:
            center = np.zeros(n_energy)
    else:
        center = np.maximum(np.linalg.lstsq(A_matrix, b_readings, rcond=None)[0], 0.0)
    return np.log(np.maximum(center, 1e-6))


def _hpd_interval(samples: np.ndarray, prob: float = 0.95):
    """Compute the shortest (highest posterior density) interval per column.

    ArviZ changed the semantics of ``az.hdi`` between versions (the 2D
    interpretation of the array input and the ``multimodal`` argument both
    differed across releases), so the HPD interval is computed here with pure
    numpy over the sample axis (axis 0) instead of relying on the installed
    ``arviz`` behaviour.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples of shape (n_total_samples, n_energy).
    prob : float, optional
        Credible mass covered by the interval (default: 0.95).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lower and upper HPD bounds, each of shape (n_energy,).
    """
    samples = np.asarray(samples, dtype=float)
    n_total, n_energy = samples.shape
    n_keep = max(int(np.ceil(prob * n_total)), 1)

    sorted_samples = np.sort(samples, axis=0)
    widths = sorted_samples[n_keep - 1 :] - sorted_samples[: n_total - n_keep + 1]
    if widths.shape[0] == 0:
        return sorted_samples[0].copy(), sorted_samples[-1].copy()

    best_idx = np.argmin(widths, axis=0)
    rows = np.arange(n_energy)
    lower = sorted_samples[best_idx, rows]
    upper = sorted_samples[best_idx + n_keep - 1, rows]
    return lower, upper


def _run_nuts_pymc(
    model,
    n_samples: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_state: Optional[int],
    progressbar: bool,
):
    """Run the NUTS sampler in a way that works across PyMC/ArviZ versions.

    ``pm.sample`` changed its signature between PyMC 5 and PyMC 6 (the
    ``return_inferencedata`` argument was dropped once InferenceData became
    the only return type).  We first try the richest argument set and retry
    without the legacy keyword when the installed version rejects it.
    """
    pm, _ = _resolve_backends()
    kwargs = {
        "draws": n_samples,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "random_seed": random_state,
        "progressbar": progressbar,
        "return_inferencedata": True,
    }
    try:
        with model:
            return pm.sample(**kwargs)
    except TypeError:
        kwargs.pop("return_inferencedata", None)
        with model:
            return pm.sample(**kwargs)


def solve_bayesian_mcmc(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    sigma_prior: float = 0.05,
    lambda_prior: float = 0.5,
    lengthscale: float = 3.0,
    n_samples: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.95,
    random_state: Optional[int] = None,
    use_hierarchical: bool = False,
    initial_spectrum: Optional[np.ndarray] = None,
    progressbar: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve unfolding problem using Bayesian MCMC with NUTS sampler.

    The spectrum is modelled on the log scale with a smoothness (Ornstein-
    Uhlenbeck) prior anchored on a data-driven center, which is the standard
    "prior guess" approach for ill-posed unfolding:

    - Prior center: user-supplied ``initial_spectrum`` when given, otherwise the
      non-negative least-squares solution of ``A @ x = b``.
    - Prior: ``f = exp(theta)`` with ``theta ~ MvNormal(mu_prior, s * C_ou)``,
      ``s ~ HalfNormal(lambda_prior)`` and the OU correlation
      ``C_ou[i, j] = exp(-|i - j| / lengthscale)``.  This keeps the spectrum
      positive, smooth and bounded in the null space of the (severely
      underdetermined) response matrix.
    - Likelihood: ``b ~ Normal(A @ f, sigma)`` with a relative noise scale
      ``sigma = sigma_prior * |b|`` (fixed) or estimated hierarchically when
      ``use_hierarchical`` is True.

    The NUTS sampler generates samples from the posterior p(f|b), which are
    used to compute statistics (mean, median, std, HPD intervals).

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV (unused by the model, kept for API consistency).
    log_steps : np.ndarray
        Logarithmic energy steps (unused by the model, kept for API
        consistency; the forward model follows the package convention
        ``b = A @ spectrum``).
    sigma_prior : float, optional
        Relative measurement noise scale (default: 0.05).  With
        ``use_hierarchical=False`` the likelihood noise is fixed at
        ``sigma_prior * |b|``; with ``use_hierarchical=True`` it is the prior
        scale of the estimated relative noise.
    lambda_prior : float, optional
        Prior scale of the spatial amplitude ``s`` of the log-spectrum
        deviations from the prior center (default: 0.5).
    lengthscale : float, optional
        OU correlation length of the smoothness prior, in energy bins
        (default: 3.0).
    n_samples : int, optional
        Number of MCMC samples per chain (default: 2000).
    tune : int, optional
        Number of tuning samples per chain (default: 1000).
    chains : int, optional
        Number of independent MCMC chains (default: 2).
    target_accept : float, optional
        Target acceptance rate for NUTS (default: 0.95).
    random_state : int, optional
        Random seed for reproducibility.
    use_hierarchical : bool, optional
        Estimate the likelihood noise scale from the data instead of fixing it
        (default: False).
    initial_spectrum : np.ndarray, optional
        Prior center guess (n_energy,).  When None, the non-negative
        least-squares solution is used as the center.
    progressbar : bool, optional
        Show sampling progress bar (default: False).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - spectrum: Mean posterior spectrum (n_energy,)
        - stats: Dictionary containing:
            - 'samples': Full MCMC samples (chains * n_samples, n_energy)
            - 'mean': Posterior mean spectrum
            - 'median': Posterior median spectrum
            - 'std': Posterior standard deviation
            - 'hpd_lower': Lower bound of 95% HPD interval
            - 'hpd_upper': Upper bound of 95% HPD interval
            - 'trace': ArviZ InferenceData object
            - 'rhat': R-hat convergence diagnostic
            - 'ess': Effective sample size

    Raises
    ------
    ImportError
        If PyMC or ArviZ is not installed.
    RuntimeError
        If MCMC sampling fails.
    """
    pm, az = _resolve_backends()
    if not _check_pymc_available():
        raise ImportError(
            "PyMC and ArviZ are required for MCMC unfolding. "
            "Install them with: pip install pymc arviz"
        )

    n_detectors, n_energy = A_matrix.shape

    mu_prior = _prior_center(A_matrix, b_readings, initial_spectrum, n_energy)
    L_corr = _ou_correlation_cholesky(n_energy, lengthscale)

    coords = {
        "energy_bin": range(n_energy),
        "detector": range(n_detectors),
    }

    with pm.Model(coords=coords) as model:
        # Spatial amplitude of log-spectrum deviations from the prior center
        s = pm.HalfNormal("s", sigma=lambda_prior)
        # Whitened latent field; theta = mu_prior + s * (L_corr @ z) is a
        # non-centered MvNormal with OU covariance.
        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="energy_bin")
        theta = pm.Deterministic(
            "theta",
            mu_prior + s * pm.math.dot(L_corr, z),
            dims="energy_bin",
        )
        spectrum = pm.Deterministic(
            "spectrum", pm.math.exp(theta), dims="energy_bin"
        )

        # Likelihood noise: fixed relative scale or estimated hierarchically
        b_abs = np.abs(b_readings) + 1e-6
        if use_hierarchical:
            rel_noise = pm.HalfNormal("rel_noise", sigma=sigma_prior)
            sigma = rel_noise * b_abs
        else:
            sigma = sigma_prior * b_abs

        # Forward model (package convention: b = A @ spectrum)
        expected_counts = pm.Deterministic(
            "expected_counts",
            pm.math.matmul(A_matrix, spectrum),
            dims="detector",
        )

        # Likelihood
        pm.Normal(
            "likelihood",
            mu=expected_counts,
            sigma=sigma,
            observed=b_readings,
            dims="detector",
        )

    # Sample from posterior using NUTS
    try:
        trace = _run_nuts_pymc(
            model,
            n_samples=n_samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_state=random_state,
            progressbar=progressbar,
        )
    except Exception as e:
        raise RuntimeError(f"MCMC sampling failed: {str(e)}") from e

    # Extract posterior samples for spectrum
    samples = np.asarray(trace.posterior["spectrum"].values, dtype=float)
    # Reshape to (n_total_samples, n_energy)
    samples = samples.reshape(-1, n_energy)

    # Compute statistics
    mean_spectrum = np.mean(samples, axis=0)
    median_spectrum = np.median(samples, axis=0)
    std_spectrum = np.std(samples, axis=0)

    # Compute 95% HPD intervals per energy bin
    hpd_lower, hpd_upper = _hpd_interval(samples, prob=0.95)

    # Compute convergence diagnostics
    rhat = np.asarray(az.rhat(trace)["spectrum"].values, dtype=float)
    ess = np.asarray(az.ess(trace)["spectrum"].values, dtype=float)

    stats = {
        "samples": samples,
        "mean": mean_spectrum,
        "median": median_spectrum,
        "std": std_spectrum,
        "hpd_lower": hpd_lower,
        "hpd_upper": hpd_upper,
        "trace": trace,
        "rhat": np.ravel(rhat),
        "ess": np.ravel(ess),
        "n_samples_total": samples.shape[0],
        "n_chains": chains,
        "tune_samples": tune,
        "target_accept": target_accept,
        "use_hierarchical": use_hierarchical,
        "lengthscale": lengthscale,
        "prior_center": np.exp(mu_prior),
    }

    return mean_spectrum, stats


def unfold_mcmc(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma_prior: float = 0.05,
    lambda_prior: float = 0.5,
    lengthscale: float = 3.0,
    n_samples: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.95,
    use_hierarchical: bool = False,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
    progressbar: bool = False,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Bayesian MCMC with NUTS sampler.

    This method implements a full Bayesian approach to neutron spectrum
    unfolding using Markov Chain Monte Carlo (MCMC) methods. Unlike traditional
    methods that provide only point estimates, MCMC generates samples from the
    full posterior distribution, enabling comprehensive uncertainty
    quantification.

    The implementation uses the No-U-Turn Sampler (NUTS), an adaptive variant
    of Hamiltonian Monte Carlo (HMC) that automatically tunes its parameters
    for efficient sampling.

    Key Features
    ------------
    1. **Uncertainty Quantification**: Provides 95% credible intervals (HPD)
       for each energy bin, showing where the spectrum is well-constrained vs
       uncertain.

    2. **Automatic Regularization**: Through the log-space smoothness prior,
       the method keeps the underdetermined solution positive and smooth
       without the collapse seen with independent per-bin priors.

    3. **Convergence Diagnostics**: Built-in R-hat and effective sample size
       (ESS) metrics ensure reliable posterior estimates.

    Methodology
    -----------
    The Bayesian model is defined as:

    - Likelihood: b ~ Normal(A @ f, sigma)
      where b is measured readings, A is response matrix, f is spectrum
    - Prior: f = exp(theta) with theta ~ MvNormal(mu_prior, s * C_ou),
      where mu_prior is the log of the data-driven prior center
      (non-negative least-squares solution or user ``initial_spectrum``),
      C_ou is the OU correlation exp(-|i-j|/lengthscale), and
      s ~ HalfNormal(lambda_prior).
    - Hyperpriors (optional): when ``use_hierarchical=True`` the relative
      likelihood noise is estimated as rel_noise ~ HalfNormal(sigma_prior).

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients for dose calculation.
    save_result_callback : callable
        Callback function to save result to history.
    readings : Dict[str, float]
        Detector readings (counts or count rates).
    initial_spectrum : Optional[np.ndarray], optional
        Prior center guess for the spectrum. When None, the non-negative
        least-squares solution of A @ x = b is used as the prior center.
    sigma_prior : float, optional
        Relative likelihood noise scale (default: 0.05). With
        ``use_hierarchical=False`` the noise is fixed at ``sigma_prior * |b|``;
        with ``use_hierarchical=True`` it is the prior scale of the estimated
        relative noise.
    lambda_prior : float, optional
        Prior scale of the log-spectrum spatial amplitude (default: 0.5).
    lengthscale : float, optional
        OU smoothness correlation length in energy bins (default: 3.0).
    n_samples : int, optional
        Number of MCMC samples per chain after tuning (default: 2000).
    tune : int, optional
        Number of tuning (warmup) samples per chain (default: 1000).
    chains : int, optional
        Number of independent MCMC chains (default: 2).
    target_accept : float, optional
        Target acceptance rate for NUTS (default: 0.95).
    use_hierarchical : bool, optional
        Estimate the likelihood noise from the data (default: False).
    calculate_errors : bool, optional
        Calculate additional Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for additional Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of additional Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.
    progressbar : bool, optional
        Show sampling progress bar (default: False).

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary containing:
        - 'energy': Energy grid (MeV)
        - 'spectrum': Mean posterior spectrum
        - 'spectrum_absolute': Same as spectrum (for API consistency)
        - 'spectrum_uncertainty': Standard deviation of posterior
        - 'spectrum_lower': Lower bound of 95% HPD interval
        - 'spectrum_upper': Upper bound of 95% HPD interval
        - 'effective_readings': Computed readings from posterior mean
        - 'residual': Difference between measured and computed readings
        - 'residual_norm': L2 norm of residual
        - 'method': 'MCMC'
        - 'doserates': Dose rates calculated from spectrum
        - 'mcmc_stats': Dictionary with MCMC-specific information (samples,
          median, HPD bounds, rhat, ess, trace and sampling metadata)

    Raises
    ------
    ImportError
        If PyMC or ArviZ is not installed.
    RuntimeError
        If MCMC sampling fails to converge.

    Examples
    --------
    >>> from bssunfold import Detector
    >>> detector = Detector()
    >>> readings = {'sphere_1': 100.5, 'sphere_2': 85.3, ...}
    >>> result = detector.unfold_mcmc(
    ...     readings,
    ...     n_samples=2000,
    ...     chains=2,
    ...     use_hierarchical=True,
    ...     progressbar=True,
    ... )
    >>> spectrum_mean = result['spectrum']
    >>> spectrum_std = result['spectrum_uncertainty']
    >>> rhat = result['mcmc_stats']['rhat']
    >>> print(f"Max R-hat: {rhat.max():.3f}")  # Should be < 1.1

    See Also
    --------
    unfold_bayes : Bayesian iterative unfolding (D'Agostini)
    unfold_bayesian_parametric : Bayesian parametric model with MCMC
    """
    if not _check_pymc_available():
        raise ImportError(
            "PyMC and ArviZ are required for MCMC unfolding. "
            "Install them with: pip install pymc arviz"
        )

    # The main solve is captured in a holder so its MCMC statistics (trace,
    # samples, diagnostics) can be merged into the standardized output.
    holder: Dict[str, Any] = {}

    def _solve_mcmc(A, b, **kwargs):
        x0 = kwargs.get("x0")
        mean_spectrum, stats = solve_bayesian_mcmc(
            A_matrix=A,
            b_readings=b,
            E=E_MeV,
            log_steps=np.ones(n_energy_bins),
            sigma_prior=sigma_prior,
            lambda_prior=lambda_prior,
            lengthscale=lengthscale,
            n_samples=n_samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_state=random_state,
            use_hierarchical=use_hierarchical,
            initial_spectrum=x0 if x0 is not None else initial_spectrum,
            progressbar=progressbar,
        )
        holder.setdefault("stats", stats)
        return mean_spectrum

    result = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.ones(n_energy_bins),
        solve_func=_solve_mcmc,
        solve_kwargs={},
        method_name="MCMC",
        extra_output={
            "sigma_prior": sigma_prior,
            "lambda_prior": lambda_prior,
            "lengthscale": lengthscale,
            "n_samples": n_samples,
            "chains": chains,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=False,
    )

    # Merge the MCMC statistics into the standardized result *before* it is
    # handed to the history callback so saved results carry the full output.
    if "stats" in holder:
        stats = holder["stats"]
        result["spectrum_uncertainty"] = np.array(stats["std"])
        result["spectrum_lower"] = np.maximum(stats["hpd_lower"], 0)
        result["spectrum_upper"] = np.array(stats["hpd_upper"])
        result["mcmc_stats"] = {
            "samples": stats["samples"],
            "median": stats["median"],
            "hpd_lower": stats["hpd_lower"],
            "hpd_upper": stats["hpd_upper"],
            "rhat": stats["rhat"],
            "ess": stats["ess"],
            "n_samples_total": stats["n_samples_total"],
            "n_chains": stats["n_chains"],
            "tune_samples": stats["tune_samples"],
            "target_accept": stats["target_accept"],
            "use_hierarchical": stats["use_hierarchical"],
            "lengthscale": stats["lengthscale"],
            "prior_center": stats["prior_center"],
            "trace": stats["trace"],
        }

    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
