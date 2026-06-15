"""Bayesian parametric unfolding method.

This module implements a Bayesian approach to parametric unfolding,
where the parameters of a spectral model are estimated using Bayesian
inference with MCMC sampling.

The parametric model consists of:
- Maxwellian thermal component
- 1/E epithermal component
- Evaporation spectrum for fast neutrons

The Bayesian framework provides:
- Posterior distributions of spectral parameters
- Uncertainty quantification
- Model comparison metrics
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = ["solve_bayesian_parametric", "unfold_bayesian_parametric"]


def _log_prior(params: Dict[str, float]) -> float:
    """Log prior probability for spectral parameters."""
    A_th = params.get('A_th', 0)
    T_th = params.get('T_th', 0.025e-6)
    A_epi = params.get('A_epi', 0)
    A_f = params.get('A_f', 0)
    T_ev = params.get('T_ev', 2.0)

    log_p = 0.0

    if A_th < 0 or A_th > 1e-3:
        return -np.inf
    log_p += -np.log(1e-3)

    if T_th < 1e-9 or T_th > 1e-3:
        return -np.inf
    log_p += -np.log(1e-3 - 1e-9)

    if A_epi < 0 or A_epi > 1e-3:
        return -np.inf
    log_p += -np.log(1e-3)

    if A_f < 0 or A_f > 1e-3:
        return -np.inf
    log_p += -np.log(1e-3)

    if T_ev < 0.1 or T_ev > 20.0:
        return -np.inf
    log_p += -np.log(19.9)

    return log_p


def _log_likelihood(
    params: Dict[str, float],
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    sigma: float,
) -> float:
    """Log likelihood function for the parametric model."""
    from .unfold_fruit_like import parametric_model

    A_th = params['A_th']
    T_th = params['T_th']
    A_epi = params['A_epi']
    A_f = params['A_f']
    T_ev = params['T_ev']

    spectrum = parametric_model(E, A_th, T_th, A_epi, A_f, T_ev)
    spectrum_with_steps = spectrum * log_steps

    computed = A_matrix @ spectrum_with_steps
    residual = b_readings - computed

    log_l = -0.5 * np.sum((residual / sigma) ** 2)
    return log_l


def _log_posterior(
    params: Dict[str, float],
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    sigma: float,
) -> float:
    """Log posterior probability."""
    log_p = _log_prior(params)
    if not np.isfinite(log_p):
        return -np.inf
    log_l = _log_likelihood(params, A_matrix, b_readings, E, log_steps, sigma)
    return log_p + log_l


def _metropolis_hastings(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    sigma: float,
    initial_params: Dict[str, float],
    n_samples: int = 1000,
    burn_in: int = 200,
    proposal_scale: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Metropolis-Hastings MCMC sampling."""
    rng = np.random.default_rng(random_state)

    current = initial_params.copy()
    current_log_p = _log_posterior(current, A_matrix, b_readings, E, log_steps, sigma)

    samples = {k: [] for k in current.keys()}
    n_accepted = 0

    for i in range(n_samples + burn_in):
        proposed = {}
        for k, v in current.items():
            proposed[k] = v + rng.normal(0, proposal_scale * abs(v) + 1e-15)
            if k in ('T_th', 'T_ev'):
                proposed[k] = max(proposed[k], 1e-9)
            elif k.startswith('A_'):
                proposed[k] = max(proposed[k], 0)

        proposed_log_p = _log_posterior(proposed, A_matrix, b_readings, E, log_steps, sigma)

        log_alpha = proposed_log_p - current_log_p
        if np.log(rng.uniform()) < log_alpha:
            current = proposed
            current_log_p = proposed_log_p
            n_accepted += 1

        if i >= burn_in:
            for k in samples:
                samples[k].append(current[k])

    for k in samples:
        samples[k] = np.array(samples[k])

    acceptance_rate = n_accepted / (n_samples + burn_in)

    means = {k: float(np.mean(v)) for k, v in samples.items()}
    return means, samples


def solve_bayesian_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    sigma: float = 0.02,
    initial_params: Optional[Dict[str, float]] = None,
    n_samples: int = 1000,
    burn_in: int = 200,
    proposal_scale: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using Bayesian parametric method.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps.
    sigma : float, optional
        Measurement uncertainty (default: 0.02).
    initial_params : dict, optional
        Initial parameter values.
    n_samples : int, optional
        Number of MCMC samples (default: 1000).
    burn_in : int, optional
        Burn-in samples (default: 200).
    proposal_scale : float, optional
        Proposal scale (default: 0.1).
    random_state : int, optional
        Random seed.

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, n_samples)
    """
    if initial_params is None:
        initial_params = {
            'A_th': 1e-6,
            'T_th': 0.025e-6,
            'A_epi': 1e-6,
            'A_f': 1e-6,
            'T_ev': 2.0,
        }

    try:
        mean_params, _ = _metropolis_hastings(
            A_matrix, b_readings, E, log_steps, sigma,
            initial_params, n_samples, burn_in, proposal_scale, random_state
        )

        from .unfold_fruit_like import parametric_model
        spectrum = parametric_model(
            E,
            mean_params['A_th'],
            mean_params['T_th'],
            mean_params['A_epi'],
            mean_params['A_f'],
            mean_params['T_ev'],
        )
        spectrum = spectrum * log_steps

        return spectrum, True, "Bayesian estimation completed", n_samples
    except Exception as e:
        return np.zeros(len(E)), False, str(e), 0


def unfold_bayesian_parametric(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    sigma: float = 0.02,
    n_samples: int = 1000,
    burn_in: int = 200,
    proposal_scale: float = 0.1,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Bayesian parametric method.

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
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess (unused).
    sigma : float, optional
        Measurement uncertainty (default: 0.02).
    n_samples : int, optional
        Number of MCMC samples (default: 1000).
    burn_in : int, optional
        Burn-in samples (default: 200).
    proposal_scale : float, optional
        Proposal scale (default: 0.1).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: True).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(E_MeV + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if n_energy_bins > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if n_energy_bins > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0

    def solve_wrapper(A_mat, b_vec, **kwargs):
        x_opt, success, message, nfev = solve_bayesian_parametric(
            A_mat, b_vec, E_MeV, log_steps, sigma,
            n_samples=n_samples, burn_in=burn_in,
            proposal_scale=proposal_scale, random_state=random_state
        )
        return x_opt, nfev, success

    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

    result = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="bayesian_parametric",
        extra_output={
            "sigma": sigma,
            "n_samples": n_samples,
            "burn_in": burn_in,
            "proposal_scale": proposal_scale,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
