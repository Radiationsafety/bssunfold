"""FRUIT-based parametric unfolding method.

This module implements the parametric neutron spectrum reconstruction
method described in:

  - R. Bedogni et al., "FRUIT: An operational tool for multisphere
    neutron spectrometry in workplaces", Nucl. Instrum. Methods A 580,
    1301-1309 (2007).
  - M.D. Pyshkina et al., "Validation and Verification of the New
    Multisphere Spectrometer Operation", Proc. II Int. Sci.-Tech.
    Conf., Minsk (2021).

The spectrum is represented as a weighted superposition of three
components:

  Thermal   (E < 1e-7 MeV):   (E/T0^2) * exp(-E/T0)
  Epithermal (1e-7 < E < 0.1): [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
  Fast      (E > 0.1 MeV):    E^alpha * exp(-E/beta)

Total: phi_j = P_th * phi_th + P_epi * phi_epi + P_f * phi_f
with constraint: P_th + P_epi + P_f = 1  (P_f = 1 - P_th - P_epi)
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = ["solve_parametric", "unfold_parametric"]

# Fixed constants from the papers / FRUIT code
_T0 = 2.53e-8   # Thermal peak energy (MeV)
_Ed = 7.07e-8   # Epithermal lower boundary parameter (MeV)

# Energy region boundaries (hard-coded per papers)
_THERMAL_MAX = 1e-7    # MeV
_FAST_MIN = 0.1        # MeV


# ------------------------------------------------------------------ #
#  Parametric model
# ------------------------------------------------------------------ #

def _thermal(E: np.ndarray) -> np.ndarray:
    """Thermal neutron component: (E/T0^2) * exp(-E/T0)."""
    return (E / (_T0 ** 2)) * np.exp(-E / _T0)


def _epithermal(E: np.ndarray, b: float, beta_prime: float) -> np.ndarray:
    """Epithermal neutron component.

    [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
    """
    return (1.0 - np.exp(-(_Ed > 0) * (E / _Ed) ** 2)) * E ** (b - 1.0) * np.exp(-E / beta_prime)


def _fast(E: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Fast neutron component: E^alpha * exp(-E/beta)."""
    return E ** alpha * np.exp(-E / beta)


def parametric_model(
    E: np.ndarray,
    b: float,
    beta_prime: float,
    alpha: float,
    beta: float,
    P_th: float,
    P_epi: float,
) -> np.ndarray:
    """Combined three-component parametric neutron spectrum.

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    b : float
        Epithermal rising-slope exponent.
    beta_prime : float
        Epithermal falling-slope characteristic energy (MeV).
    alpha : float
        Fast-neutron power-law exponent.
    beta : float
        Fast-neutron characteristic energy (MeV).
    P_th : float
        Weight of thermal component (0..1).
    P_epi : float
        Weight of epithermal component (0..1).

    Returns
    -------
    np.ndarray
        Neutron spectrum (fluence per energy bin).
    """
    E = np.asarray(E, dtype=float)
    P_f = max(0.0, 1.0 - P_th - P_epi)

    thermal = np.zeros_like(E)
    epithermal = np.zeros_like(E)
    fast = np.zeros_like(E)

    m_th = E < _THERMAL_MAX
    m_epi = (E >= _THERMAL_MAX) & (E < _FAST_MIN)
    m_f = E >= _FAST_MIN

    if np.any(m_th):
        thermal[m_th] = _thermal(E[m_th])
    if np.any(m_epi):
        epithermal[m_epi] = _epithermal(E[m_epi], b, beta_prime)
    if np.any(m_f):
        fast[m_f] = _fast(E[m_f], alpha, beta)

    # Normalize each component to unit sum before weighting
    s_th = thermal.sum()
    s_epi = epithermal.sum()
    s_f = fast.sum()
    if s_th > 0:
        thermal /= s_th
    if s_epi > 0:
        epithermal /= s_epi
    if s_f > 0:
        fast /= s_f

    return P_th * thermal + P_epi * epithermal + P_f * fast


# ------------------------------------------------------------------ #
#  Core solver
# ------------------------------------------------------------------ #

def _residuals(params, A_matrix, b_readings, E, log_steps):
    """Residual function for lmfit minimization."""
    b_val = params['b'].value
    bp_val = params['beta_prime'].value
    alpha_val = params['alpha'].value
    beta_val = params['beta'].value
    P_th_val = params['P_th'].value
    P_epi_val = params['P_epi'].value

    spectrum = parametric_model(E, b_val, bp_val, alpha_val, beta_val, P_th_val, P_epi_val)
    spectrum_with_steps = spectrum * log_steps

    computed = A_matrix @ spectrum_with_steps
    return computed - b_readings


def solve_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "leastsq",
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding using the FRUIT-based parametric model.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps (d(ln E)).
    initial_params : dict, optional
        Initial parameter values.  Keys: b, beta_prime, alpha, beta,
        P_th, P_epi.
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
            "lmfit is required for parametric unfolding. "
            "Install with: pip install lmfit"
        ) from e

    params = lmfit.Parameters()

    defaults = {
        'b':         (1.0,   0.5,  2.0),
        'beta_prime': (0.01, 1e-4, 1.0),
        'alpha':     (0.5,   0.0,  5.0),
        'beta':      (2.0,   0.1,  20.0),
        'P_th':      (0.33,  0.0,  1.0),
        'P_epi':     (0.33,  0.0,  1.0),
    }

    for name, (val, lo, hi) in defaults.items():
        if initial_params and name in initial_params:
            val = initial_params[name]
        params.add(name, value=val, min=lo, max=hi)

    result = lmfit.minimize(
        _residuals,
        params,
        args=(A_matrix, b_readings, E, log_steps),
        method=method,
    )

    fp = result.params
    spectrum = parametric_model(
        E,
        fp['b'].value, fp['beta_prime'].value,
        fp['alpha'].value, fp['beta'].value,
        fp['P_th'].value, fp['P_epi'].value,
    )
    spectrum = spectrum * log_steps

    return spectrum, result.success, result.message, result.nfev


# ------------------------------------------------------------------ #
#  Workflow wrapper
# ------------------------------------------------------------------ #

def unfold_parametric(
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
    """Unfold neutron spectrum using the FRUIT-based parametric method.

    The spectrum is modelled as a weighted superposition of thermal,
    epithermal and fast components (Bedogni FRUIT / Pyshkina B3S).

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
        Keys: b, beta_prime, alpha, beta, P_th, P_epi.
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
        x_opt, success, message, nfev = solve_parametric(
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
        method_name="parametric",
        extra_output={
            "initial_params": initial_params,
            "method": method,
            "T0": _T0,
            "Ed": _Ed,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
