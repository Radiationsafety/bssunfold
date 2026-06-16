"""FRUIT-like parametric unfolding method.

This module implements a parametric unfolding method inspired by FRUIT
(Fast Real-time Unfolding of neutron spectra with Iterative parameteNization
Technique). The parametric model consists of:
- Maxwellian thermal component
- 1/E epithermal component
- Evaporation spectrum for fast neutrons

Reference: Bedogni et al., Nucl. Instrum. Methods A 580, 1301-1309 (2007)
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = ["solve_fruit_like", "unfold_fruit_like"]


def _maxwellian(E: np.ndarray, T: float, A_th: float) -> np.ndarray:
    """Maxwellian thermal neutron spectrum.

    Phi(E) = A_th * sqrt(E) * exp(-E/T)
    """
    return A_th * np.sqrt(E) * np.exp(-E / T)


def _one_over_e(E: np.ndarray, A_epi: float) -> np.ndarray:
    """1/E epithermal neutron spectrum."""
    return A_epi / (E + 1e-15)


def _evaporation(E: np.ndarray, T_ev: float, A_f: float) -> np.ndarray:
    """Evaporation spectrum for fast neutrons.

    Phi(E) = A_f * exp(-E/T_ev)
    """
    return A_f * np.exp(-E / T_ev)


def parametric_model(
    E: np.ndarray,
    A_th: float,
    T_th: float,
    A_epi: float,
    A_f: float,
    T_ev: float,
    epi_max: float = 0.1,
) -> np.ndarray:
    """Combined parametric neutron spectrum model.

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    A_th : float
        Amplitude of thermal component.
    T_th : float
        Temperature of Maxwellian (MeV), typically ~0.025e-6.
    A_epi : float
        Amplitude of epithermal component.
    A_f : float
        Amplitude of fast component.
    T_ev : float
        Evaporation temperature (MeV), typically ~2 MeV.
    epi_max : float
        Maximum energy of epithermal region (MeV).

    Returns
    -------
    np.ndarray
        Neutron spectrum.
    """
    E = np.asarray(E, dtype=float)
    spectrum = np.zeros_like(E)

    thermal = E < 0.4e-6
    epithermal = (E >= 0.4e-6) & (E < epi_max)
    fast = E >= epi_max

    spectrum[thermal] += _maxwellian(E[thermal], T_th, A_th)
    spectrum[epithermal] += _one_over_e(E[epithermal], A_epi)
    spectrum[fast] += _evaporation(E[fast], T_ev, A_f)

    return spectrum


def _residuals(params, A_matrix, b_readings, E, log_steps):
    """Residual function for lmfit minimization."""
    A_th = params['A_th'].value
    T_th = params['T_th'].value
    A_epi = params['A_epi'].value
    A_f = params['A_f'].value
    T_ev = params['T_ev'].value

    spectrum = parametric_model(E, A_th, T_th, A_epi, A_f, T_ev)
    spectrum_with_steps = spectrum * log_steps

    computed_readings = A_matrix @ spectrum_with_steps
    residual = computed_readings - b_readings

    return residual


def solve_fruit_like(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "leastsq",
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using FRUIT-like parametric model.

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
    initial_params : dict, optional
        Initial parameter values. Keys: A_th, T_th, A_epi, A_f, T_ev.
    method : str, optional
        lmfit solver method (default: "leastsq").

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    try:
        import lmfit
    except ImportError as e:
        raise ImportError(
            "lmfit is required for FRUIT-like unfolding. "
            "Install with: pip install lmfit"
        ) from e

    params = lmfit.Parameters()

    if initial_params is None:
        params.add('A_th', value=1e-6, min=0.0)
        params.add('T_th', value=0.025e-6, min=1e-9, max=1e-3)
        params.add('A_epi', value=1e-6, min=0.0)
        params.add('A_f', value=1e-6, min=0.0)
        params.add('T_ev', value=2.0, min=0.1, max=20.0)
    else:
        for name, value in initial_params.items():
            if name == 'A_th':
                params.add('A_th', value=value, min=0.0)
            elif name == 'T_th':
                params.add('T_th', value=value, min=1e-9, max=1e-3)
            elif name == 'A_epi':
                params.add('A_epi', value=value, min=0.0)
            elif name == 'A_f':
                params.add('A_f', value=value, min=0.0)
            elif name == 'T_ev':
                params.add('T_ev', value=value, min=0.1, max=20.0)

    result = lmfit.minimize(
        _residuals,
        params,
        args=(A_matrix, b_readings, E, log_steps),
        method=method,
    )

    final_params = result.params
    spectrum = parametric_model(
        E,
        final_params['A_th'].value,
        final_params['T_th'].value,
        final_params['A_epi'].value,
        final_params['A_f'].value,
        final_params['T_ev'].value,
    )
    spectrum = spectrum * log_steps

    return spectrum, result.success, result.message, result.nfev


def unfold_fruit_like(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "leastsq",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using FRUIT-like parametric method.

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
        Initial spectrum guess (unused in parametric method).
    initial_params : Optional[Dict[str, float]], optional
        Initial parameter values for the parametric model.
    method : str, optional
        lmfit solver method (default: "leastsq").
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
    ln_steps = log_steps * np.log(10)

    def solve_wrapper(A_mat, b_vec, **kwargs):
        x_opt, success, message, nfev = solve_fruit_like(
            A_mat, b_vec, E_MeV, ln_steps, initial_params, method
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
        method_name="fruit_like",
        extra_output={
            "initial_params": initial_params,
            "method": method,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
