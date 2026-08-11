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

import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

from ._base_unfolder import run_unfolding, _build_system
from ._matrix_utils import compute_log_steps
from .dose_calculation import calculate_dose_rates

__all__ = ["solve_bayesian_mcmc", "unfold_mcmc"]


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
    
    # Set random seed - PyMC v6 handles seeding through pm.sample() random_seed parameter
    rng = np.random.default_rng(random_state)
    
    # Build PyMC model
    coords = {
        'energy_bin': range(n_energy),
        'detector': range(n_detectors),
    }
    
    with pm.Model(coords=coords) as model:
        # Priors for spectrum (non-negative)
        if use_hierarchical:
            # Hierarchical prior: learn regularization strength from data
            lambda_hyper = pm.HalfCauchy('lambda_hyper', beta=lambda_prior)
            spectrum = pm.HalfNormal('spectrum', sigma=lambda_hyper, dims='energy_bin')
            
            # Learn noise level from data
            sigma = pm.HalfCauchy('sigma', beta=sigma_prior)
        else:
            # Fixed priors
            spectrum = pm.HalfNormal('spectrum', sigma=lambda_prior, dims='energy_bin')
            sigma = pm.HalfNormal('sigma', sigma=sigma_prior)
        
        # Forward model: expected counts
        # Note: We multiply by log_steps to account for flux-to-counts conversion
        expected_counts = pm.Deterministic(
            'expected_counts',
            pm.math.matmul(A_matrix, spectrum * log_steps),
            dims='detector'
        )
        
        # Likelihood
        likelihood = pm.Normal(
            'likelihood',
            mu=expected_counts,
            sigma=sigma,
            observed=b_readings,
            dims='detector'
        )
    
    # Sample from posterior using NUTS
    try:
        with model:
            trace = pm.sample(
                draws=n_samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_state,
                progressbar=progressbar,
                return_inferencedata=True,
                idata_kwargs={
                    'log_likelihood': True,
                    'coords': {'sample': range(n_samples)}
                }
            )
    except Exception as e:
        raise RuntimeError(f"MCMC sampling failed: {str(e)}")
    
    # Extract posterior samples for spectrum
    samples = trace.posterior['spectrum'].values
    # Reshape to (n_total_samples, n_energy)
    samples = samples.reshape(-1, n_energy)
    
    # Compute statistics
    mean_spectrum = np.mean(samples, axis=0)
    median_spectrum = np.median(samples, axis=0)
    std_spectrum = np.std(samples, axis=0)
    
    # Compute 95% HPD intervals using ArviZ (v1.0+ API changes)
    try:
        hpd_results = az.hdi(samples, prob=0.95, multimodal=False)
    except TypeError:
        # Fallback for newer arviz versions that removed 'multimodal' argument
        hpd_results = az.hdi(samples, prob=0.95)
    hpd_lower = hpd_results[:, 0]
    hpd_upper = hpd_results[:, 1]
    
    # Compute convergence diagnostics
    rhat = az.rhat(trace)['spectrum'].values
    ess = az.ess(trace)['spectrum'].values
    
    stats = {
        'samples': samples,
        'mean': mean_spectrum,
        'median': median_spectrum,
        'std': std_spectrum,
        'hpd_lower': hpd_lower,
        'hpd_upper': hpd_upper,
        'trace': trace,
        'rhat': rhat,
        'ess': ess,
        'n_samples_total': samples.shape[0],
        'n_chains': chains,
        'tune_samples': tune,
        'target_accept': target_accept,
        'use_hierarchical': use_hierarchical,
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
    
    This method implements a full Bayesian approach to neutron spectrum unfolding
    using Markov Chain Monte Carlo (MCMC) methods. Unlike traditional methods that
    provide only point estimates, MCMC generates samples from the full posterior
    distribution, enabling comprehensive uncertainty quantification.
    
    The implementation uses the No-U-Turn Sampler (NUTS), an adaptive variant of
    Hamiltonian Monte Carlo (HMC) that automatically tunes its parameters for
    efficient sampling.
    
    Key Features
    ------------
    1. **Uncertainty Quantification**: Provides 95% credible intervals (HPD) for
       each energy bin, showing where the spectrum is well-constrained vs uncertain.
    
    2. **Automatic Regularization**: Through hierarchical modeling, the method can
       automatically infer the appropriate regularization strength from the data,
       eliminating manual tuning of regularization parameters.
    
    3. **Flexible Modeling**: The Bayesian framework allows easy incorporation of:
       - Complex priors (e.g., smoothness via Gaussian Processes)
       - Uncertainty in response matrix
       - Multiple sources of systematic error
    
    4. **Convergence Diagnostics**: Built-in R-hat and effective sample size (ESS)
       metrics ensure reliable posterior estimates.
    
    Methodology
    -----------
    The Bayesian model is defined as:
    
    - Likelihood: b ~ Normal(A @ (f * log_steps), sigma^2)
      where b is measured readings, A is response matrix, f is spectrum
    
    - Prior: f ~ HalfNormal(lambda)
      Ensures non-negative spectrum with regularization
    
    - Hyperpriors (optional):
      sigma ~ HalfCauchy(sigma_prior)
      lambda ~ HalfCauchy(lambda_prior)
    
    The NUTS sampler generates {f^(1), f^(2), ..., f^(N)} samples from p(f|b),
    which are used to compute empirical statistics.
    
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
        Larger values allow more noise; smaller values assume cleaner data.
    lambda_prior : float, optional
        Prior scale for spectrum regularization (default: 1.0).
        Controls the expected magnitude of the spectrum.
    n_samples : int, optional
        Number of MCMC samples per chain after tuning (default: 2000).
        More samples give better statistics but take longer.
    tune : int, optional
        Number of tuning (warmup) samples per chain (default: 1000).
        During tuning, NUTS adapts its step size and mass matrix.
    chains : int, optional
        Number of independent MCMC chains (default: 2).
        Multiple chains help assess convergence.
    target_accept : float, optional
        Target acceptance rate for NUTS (default: 0.8).
        Higher values (e.g., 0.9) give more accurate sampling but slower.
    use_hierarchical : bool, optional
        Use hierarchical priors for hyperparameters (default: False).
        If True, sigma and lambda are inferred from data rather than fixed.
    calculate_errors : bool, optional
        Calculate additional Monte-Carlo errors (default: False).
        Note: MCMC already provides uncertainty, so this is usually not needed.
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
        - 'method': 'mcmc'
        - 'doserates': Dose rates calculated from spectrum
        - 'mcmc_stats': Dictionary with MCMC-specific information:
            - 'samples': Full MCMC samples array
            - 'median': Posterior median spectrum
            - 'hpd_lower': Lower 95% HPD bound
            - 'hpd_upper': Upper 95% HPD bound
            - 'rhat': R-hat convergence diagnostic
            - 'ess': Effective sample size
            - 'n_samples_total': Total number of samples
            - 'n_chains': Number of chains
            - 'tune_samples': Number of tuning samples
            - 'target_accept': Target acceptance rate
            - 'use_hierarchical': Whether hierarchical priors were used
            - 'trace': ArviZ InferenceData object for further analysis
    
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
    ...     progressbar=True
    ... )
    >>> # Access uncertainty
    >>> spectrum_mean = result['spectrum']
    >>> spectrum_std = result['spectrum_uncertainty']
    >>> hpd_lower = result['spectrum_lower']
    >>> hpd_upper = result['spectrum_upper']
    >>> # Check convergence
    >>> rhat = result['mcmc_stats']['rhat']
    >>> print(f"Max R-hat: {rhat.max():.3f}")  # Should be < 1.1
    
    Notes
    -----
    1. **Sampling Time**: MCMC is computationally intensive. Expect 10-60 seconds
       for typical problems with default settings. Reduce n_samples or use fewer
       chains for faster (but less accurate) results.
    
    2. **Convergence**: Always check R-hat values (< 1.1 indicates good convergence)
       and effective sample size (> 100 per chain is recommended).
    
    3. **Hierarchical Mode**: Using use_hierarchical=True is recommended when
       you're unsure about appropriate noise/regularization levels. It adds
       computational cost but provides more robust results.
    
    4. **Comparison with Other Methods**: 
       - vs. Tikhonov: MCMC provides full uncertainty, no manual lambda tuning
       - vs. MLEM: MCMC is more robust for ill-posed problems, provides HPD intervals
       - vs. unfold_bayes: MCMC uses full Bayesian inference, not just point estimates
    
    References
    ----------
    1. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively
       Setting Path Lengths in Hamiltonian Monte Carlo. JMLR, 15(1), 1593-1623.
    
    2. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic
       programming in Python using PyMC3. PeerJ Computer Science, 2, e55.
    
    3. Vehtari, A., et al. (2021). Rank-Normalization, Folding, and Localization:
       An Improved R-hat for Assessing Convergence of MCMC. Bayesian Analysis.
    
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
    
    # Build system
    A, b, selected = _build_system(readings, detector_names, sensitivities)
    
    # Compute log steps
    log_steps = compute_log_steps(E_MeV, n_energy_bins)
    
    # Solve using MCMC
    try:
        mean_spectrum, mcmc_stats = solve_bayesian_mcmc(
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
    except Exception as e:
        raise RuntimeError(f"MCMC unfolding failed: {str(e)}")
    
    # Prepare output
    spectrum_nonneg = np.maximum(mean_spectrum, 0)
    computed_readings = A @ spectrum_nonneg
    residual = b - computed_readings
    
    # Get uncertainty from MCMC samples
    spectrum_std = mcmc_stats['std']
    hpd_lower = mcmc_stats['hpd_lower']
    hpd_upper = mcmc_stats['hpd_upper']
    
    # Build result dictionary
    result = {
        'energy': E_MeV.copy(),
        'spectrum': spectrum_nonneg.copy(),
        'spectrum_absolute': spectrum_nonneg.copy(),
        'spectrum_uncertainty': spectrum_std.copy(),
        'spectrum_lower': np.maximum(hpd_lower, 0),  # Ensure non-negative
        'spectrum_upper': hpd_upper.copy(),
        'effective_readings': {
            name: float(val)
            for name, val in zip(selected, computed_readings)
        },
        'residual': residual.copy(),
        'residual_norm': float(np.linalg.norm(residual)),
        'method': 'mcmc',
        'doserates': calculate_dose_rates(
            spectrum_nonneg, cc_icrp116
        ),
        'mcmc_stats': {
            'samples': mcmc_stats['samples'],
            'median': mcmc_stats['median'],
            'hpd_lower': hpd_lower,
            'hpd_upper': hpd_upper,
            'rhat': mcmc_stats['rhat'],
            'ess': mcmc_stats['ess'],
            'n_samples_total': mcmc_stats['n_samples_total'],
            'n_chains': mcmc_stats['n_chains'],
            'tune_samples': mcmc_stats['tune_samples'],
            'target_accept': mcmc_stats['target_accept'],
            'use_hierarchical': mcmc_stats['use_hierarchical'],
            'trace': mcmc_stats['trace'],
        },
    }
    
    # Add extra output fields
    result['sigma_prior'] = sigma_prior
    result['lambda_prior'] = lambda_prior
    result['n_samples'] = n_samples
    result['chains'] = chains
    
    # Optionally calculate additional Monte-Carlo errors
    if calculate_errors and n_montecarlo > 0:
        # This is redundant since MCMC already provides uncertainty,
        # but included for API consistency
        rng = np.random.default_rng(random_state)
        mc_spectra = []
        for i in range(n_montecarlo):
            noise = rng.normal(0, noise_level * b)
            b_noisy = b + noise
            try:
                mc_spectrum, _ = solve_bayesian_mcmc(
                    A_matrix=A,
                    b_readings=b_noisy,
                    E=E_MeV,
                    log_steps=log_steps,
                    sigma_prior=sigma_prior,
                    lambda_prior=lambda_prior,
                    n_samples=n_samples // 10,  # Fewer samples for MC
                    tune=tune // 10,
                    chains=1,
                    random_state=rng.integers(0, 2**31),
                    progressbar=False,
                )
                mc_spectra.append(mc_spectrum)
            except:
                continue
        
        if mc_spectra:
            mc_spectra = np.array(mc_spectra)
            result['mc_std'] = np.std(mc_spectra, axis=0)
            result['n_mc_samples'] = len(mc_spectra)
    
    # Save result if requested
    if save_result and save_result_callback is not None:
        result_id = save_result_callback(result)
        result['result_id'] = result_id
    
    return result
