"""Bayesian MCMC unfolding method using PyMC and NUTS sampler.

This module implements a full Bayesian approach to neutron spectrum unfolding
using Markov Chain Monte Carlo (MCMC) methods, specifically the No-U-Turn
Sampler (NUTS), which is an adaptive variant of Hamiltonian Monte Carlo (HMC).

The Bayesian framework provides:
- Full posterior distributions for each energy bin
- Uncertainty quantification via credible intervals (HPD intervals)
- Automatic regularization through prior specification
- Hierarchical modeling capabilities for hyperparameters

Key advantages over point-estimate methods:
1. Uncertainty estimation: 95% credible intervals for each spectral bin
2. Automatic parameter tuning: Hyperparameters can be inferred from data
3. Flexible modeling: Easy incorporation of complex priors (e.g., Gaussian Processes)
4. Robustness: Better handling of ill-posed problems through proper regularization
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    az = None
    pm = None

from ._base_unfolder import run_unfolding
from ._matrix_utils import compute_log_steps

__all__ = ["solve_bayesian_mcmc", "unfold_mcmc"]


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
    sigma_prior: float = 0.1,
    lambda_prior: float = 1.0,
    n_samples: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.8,
    random_state: Optional[int] = None,
    use_hierarchical: bool = False,
    progressbar: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve unfolding problem using Bayesian MCMC with NUTS sampler.

    This function implements a hierarchical Bayesian model for neutron spectrum
    unfolding. The model assumes:

    - Likelihood: b ~ Normal(A @ (f * log_steps), sigma^2)
    - Prior on spectrum: f ~ HalfNormal(lambda) or hierarchical
    - Hyperpriors: sigma ~ HalfCauchy, lambda ~ HalfCauchy (if hierarchical)

    The NUTS sampler generates samples from the posterior distribution p(f|b),
    which are then used to compute statistics (mean, median, std, HPD intervals).

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps for flux-to-counts conversion.
    sigma_prior : float, optional
        Prior scale for measurement noise (default: 0.1).
    lambda_prior : float, optional
        Prior scale for spectrum regularization (default: 1.0).
    n_samples : int, optional
        Number of MCMC samples per chain (default: 2000).
    tune : int, optional
        Number of tuning samples per chain (default: 1000).
    chains : int, optional
        Number of independent MCMC chains (default: 2).
    target_accept : float, optional
        Target acceptance rate for NUTS (default: 0.8).
    random_state : int, optional
        Random seed for reproducibility.
    use_hierarchical : bool, optional
        Use hierarchical priors for hyperparameters (default: False).
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
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC and ArviZ are required for MCMC unfolding. "
            "Install them with: pip install pymc arviz"
        )

    n_detectors, n_energy = A_matrix.shape

    coords = {
        "energy_bin": range(n_energy),
        "detector": range(n_detectors),
    }

    with pm.Model(coords=coords) as model:
        # Priors for spectrum (non-negative)
        if use_hierarchical:
            # Hierarchical prior: learn regularization strength from data
            lambda_hyper = pm.HalfCauchy("lambda_hyper", beta=lambda_prior)
            spectrum = pm.HalfNormal(
                "spectrum", sigma=lambda_hyper, dims="energy_bin"
            )
            # Learn noise level from data
            sigma = pm.HalfCauchy("sigma", beta=sigma_prior)
        else:
            # Fixed priors
            spectrum = pm.HalfNormal(
                "spectrum", sigma=lambda_prior, dims="energy_bin"
            )
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior)

        # Forward model: expected counts
        # Note: We multiply by log_steps to account for flux-to-counts conversion
        expected_counts = pm.Deterministic(
            "expected_counts",
            pm.math.matmul(A_matrix, spectrum * log_steps),
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
        raise RuntimeError(f"MCMC sampling failed: {str(e)}")

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
    sigma_prior: float = 0.1,
    lambda_prior: float = 1.0,
    n_samples: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.8,
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

    2. **Automatic Regularization**: Through hierarchical modeling, the method
       can automatically infer the appropriate regularization strength from
       the data.

    3. **Convergence Diagnostics**: Built-in R-hat and effective sample size
       (ESS) metrics ensure reliable posterior estimates.

    Methodology
    -----------
    The Bayesian model is defined as:

    - Likelihood: b ~ Normal(A @ (f * log_steps), sigma^2)
      where b is measured readings, A is response matrix, f is spectrum
    - Prior: f ~ HalfNormal(lambda), ensures a non-negative spectrum
    - Hyperpriors (optional):
      sigma ~ HalfCauchy(sigma_prior), lambda ~ HalfCauchy(lambda_prior)

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
        Initial spectrum guess (unused in MCMC, but kept for API consistency).
    sigma_prior : float, optional
        Prior scale for measurement noise standard deviation (default: 0.1).
    lambda_prior : float, optional
        Prior scale for spectrum regularization (default: 1.0).
    n_samples : int, optional
        Number of MCMC samples per chain after tuning (default: 2000).
    tune : int, optional
        Number of tuning (warmup) samples per chain (default: 1000).
    chains : int, optional
        Number of independent MCMC chains (default: 2).
    target_accept : float, optional
        Target acceptance rate for NUTS (default: 0.8).
    use_hierarchical : bool, optional
        Use hierarchical priors for hyperparameters (default: False).
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
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC and ArviZ are required for MCMC unfolding. "
            "Install them with: pip install pymc arviz"
        )

    log_steps = compute_log_steps(E_MeV, n_energy_bins)

    # The main solve is captured in a holder so its MCMC statistics (trace,
    # samples, diagnostics) can be merged into the standardized output.
    holder: Dict[str, Any] = {}

    def _solve_mcmc(A, b, **kwargs):
        mean_spectrum, stats = solve_bayesian_mcmc(
            A_matrix=A,
            b_readings=b,
            E=E_MeV,
            log_steps=log_steps,
            sigma_prior=sigma_prior,
            lambda_prior=lambda_prior,
            n_samples=n_samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_state=random_state,
            use_hierarchical=use_hierarchical,
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
            "trace": stats["trace"],
        }

    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result