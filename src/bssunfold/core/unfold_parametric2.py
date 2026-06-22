"""BON95-based parametric unfolding method.

This module implements the parametric neutron spectrum reconstruction
method described in:

  - A.V. Sannikov, "BON95, a universal user-independent unfolding code
    for low informative neutron spectrometers", GSF report, Munich, 1995.
  - V.V. Babintsev et al., "Measurement of neutron spectrum at the
    'Neutron' test bench by Bonner spectrometer with activation
    detectors", NRC Kurchatov Institute - IHEP Preprint 2022-4.
  - A.V. Sannikov et al., "Multi-sphere neutron spectrometer based
    on the serial instrument RSU-01", Apparatus, No.1, pp.62-69, 2009.

The spectrum E * Phi(E) is represented as a linear combination of four
components:

  Thermal    (E < 0.1 MeV):  Fth  = Xth^(3/2) * exp(-Xth)
  Epithermal (E < 10 MeV):   Fepi = E^(-b) * (1 - exp(-Xth))
  Intermediate (E < 10 MeV): Fint = (1 - exp(-Xth))
  Fast       (E > 0.1 MeV):  Ff   = Xf^(3/2) * exp(-Xf)

  E * Phi(E) = a1*Fth + a2*Fepi + a3*Fint + a4*Ff

where Xth = E/Tth (Tth = 0.035 eV = 3.5e-8 MeV) and
Xf = (E/Tf)^c.

Free shape parameters (b, Tf, c) are found by grid search.
Linear coefficients (a1..a4) are solved by weighted NLS.

After parametric fitting, the result is refined by directed-divergence
(I-divergence / Itakura-Saito) iterations.
"""

import logging
import warnings
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = [
    "solve_bon95_parametric",
    "directed_divergence_iteration",
    "solve_parametric2",
    "unfold_parametric2",
]

# Fixed constants from the BON95 papers
_Tth = 3.5e-8   # Thermal peak temperature (MeV), = 0.035 eV

# Energy region boundaries for component masking
_THERMAL_MAX_BON95 = 0.1    # MeV — thermal + epithermal dominant
_FAST_MIN_BON95 = 0.1       # MeV — fast component starts

_RESIDUAL_WARN_THRESHOLD = 10.0

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  BON95 parametric model
# ------------------------------------------------------------------ #

def _Fth(E: np.ndarray, Tth: float = _Tth) -> np.ndarray:
    """Thermal component: Xth^(3/2) * exp(-Xth), Xth = E/Tth."""
    Xth = E / Tth
    return Xth ** 1.5 * np.exp(-Xth)


def _Fepi(E: np.ndarray, b: float, Tth: float = _Tth) -> np.ndarray:
    """Epithermal component: E^(-b) * (1 - exp(-Xth))."""
    Xth = E / Tth
    return E ** (-b) * (1.0 - np.exp(-Xth))


def _Fint(E: np.ndarray, Tth: float = _Tth) -> np.ndarray:
    """Intermediate component: (1 - exp(-Xth))."""
    Xth = E / Tth
    return 1.0 - np.exp(-Xth)


def _Ff(E: np.ndarray, Tf: float, c: float) -> np.ndarray:
    """Fast component: Xf^(3/2) * exp(-Xf), Xf = (E/Tf)^c."""
    Xf = (E / Tf) ** c
    return Xf ** 1.5 * np.exp(-Xf)


def bon95_model(
    E: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
) -> np.ndarray:
    """Combined four-component BON95 neutron spectrum.

    Returns E * Phi(E), i.e. the lethargy spectrum.

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    b : float
        Epithermal power-law exponent.
    Tf : float
        Fast peak characteristic energy (MeV).
    c : float
        Fast peak width parameter.
    a1, a2, a3, a4 : float
        Linear weights of thermal, epithermal, intermediate, fast.

    Returns
    -------
    np.ndarray
        Lethargy spectrum E * Phi(E).
    """
    E = np.asarray(E, dtype=float)
    return a1 * _Fth(E) + a2 * _Fepi(E, b) + a3 * _Fint(E) + a4 * _Ff(E, Tf, c)


def bon95_spectrum(
    E: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
) -> np.ndarray:
    """Neutron spectrum Phi(E) from the BON95 model.

    Returns Phi(E) = (a1*Fth + a2*Fepi + a3*Fint + a4*Ff) / E.
    """
    E = np.asarray(E, dtype=float)
    lethargy = bon95_model(E, b, Tf, c, a1, a2, a3, a4)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(E > 0, lethargy / E, 0.0)
    return phi


# ------------------------------------------------------------------ #
#  Weighted NLS for linear coefficients
# ------------------------------------------------------------------ #

def _solve_linear_coefficients(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Solve for optimal linear coefficients a1..a4 given shape params.

    The model is: M_i = sum_j A_i(E_j) * Phi(E_j) * d(ln E)_j
    where Phi(E) = (a1*Fth + a2*Fepi + a3*Fint + a4*Ff) / E.

    This is linear in a1..a4, so we solve the weighted least-squares
    problem: min ||W*(B*a - b)||^2 where B_ij = A_i(E_j)*F_k(E_j)/E_j*d(lnE)_j.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths d(ln E).
    b, Tf, c : float
        Shape parameters.
    weights : np.ndarray, optional
        Diagonal weights (1/sigma_i^2). If None, uniform weights.

    Returns
    -------
    Tuple[np.ndarray, float]
        (coefficients [a1,a2,a3,a4], chi2 value)
    """
    n_det = A_matrix.shape[0]

    # Build the four basis columns: B_ik = sum_j A_i(E_j) * F_k(E_j) / E_j * ln_steps_j
    E_safe = np.where(E > 0, E, 1.0)
    F_cols = np.column_stack([
        _Fth(E),
        _Fepi(E, b),
        _Fint(E),
        _Ff(E, Tf, c),
    ])  # (n_energy, 4)

    # Each column of B: B_i_k = sum_j A_i_j * F_k_j / E_j * ln_steps_j
    # F_cols/E_safe divides each row by corresponding E, then * ln_steps
    weighted_F = F_cols / E_safe[:, np.newaxis] * ln_steps[:, np.newaxis]  # (n_energy, 4)
    B = A_matrix @ weighted_F  # (n_det, 4)

    if weights is None:
        weights = np.ones(n_det)

    W = np.diag(np.sqrt(weights))
    Bw = W @ B
    bw = W @ b_readings

    # Solve via least-squares
    result, res, rank, sv = np.linalg.lstsq(Bw, bw, rcond=None)
    a = np.maximum(result, 0.0)  # enforce non-negativity of coefficients

    # Compute chi2
    residual = B @ a - b_readings
    chi2 = np.mean((residual ** 2) * weights) if np.all(weights > 0) else np.mean(residual ** 2)

    return a, chi2


# ------------------------------------------------------------------ #
#  Grid search + NLS parametric fit
# ------------------------------------------------------------------ #

# Default grid ranges for shape parameters
_DEFAULT_B_RANGE = (0.5, 2.0, 5)      # (min, max, n_points)
_DEFAULT_TF_RANGE = (0.5, 10.0, 5)    # (min MeV, max MeV, n_points)
_DEFAULT_C_RANGE = (0.5, 3.0, 4)      # (min, max, n_points)


def solve_bon95_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    b_meas: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> Tuple[Dict[str, float], float, List[Dict[str, float]]]:
    """Grid search + NLS for the BON95 parametric model.

    Scans over (b, Tf, c) shape parameters, solves for optimal linear
    coefficients (a1..a4) at each grid point via weighted NLS, and
    returns the best result.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    b_range, Tf_range, c_range : tuple
        (min, max, n_points) for each shape parameter.
    b_meas : np.ndarray, optional
        Measurement uncertainties (sigma_i). Used as weights.
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    Tuple[dict, float, list]
        (best_params, best_chi2, top_candidates)
        best_params keys: b, Tf, c, a1, a2, a3, a4
    """
    b_vals = np.linspace(b_range[0], b_range[1], b_range[2])
    Tf_vals = np.linspace(Tf_range[0], Tf_range[1], Tf_range[2])
    c_vals = np.linspace(c_range[0], c_range[1], c_range[2])

    # Weights from measurement uncertainties
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    candidates = []

    for b_val in b_vals:
        for Tf_val in Tf_vals:
            for c_val in c_vals:
                a, chi2 = _solve_linear_coefficients(
                    A_matrix, b_readings, E, ln_steps,
                    b_val, Tf_val, c_val, weights,
                )
                candidates.append({
                    'b': b_val,
                    'Tf': Tf_val,
                    'c': c_val,
                    'a1': a[0],
                    'a2': a[1],
                    'a3': a[2],
                    'a4': a[3],
                    'chi2': chi2,
                })

    # Sort by chi2 (best first)
    candidates.sort(key=lambda x: x['chi2'])
    best = candidates[0]

    return best, best['chi2'], candidates[:top_n]


# ------------------------------------------------------------------ #
#  Directed divergence iteration
# ------------------------------------------------------------------ #

def directed_divergence_iteration(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    phi0: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
    tol_rel: float = 1e-6,
) -> Tuple[np.ndarray, int, float, bool]:
    """Refine spectrum via directed-divergence (I-divergence) iterations.

    Multiplicative update rule (Itakura-Saito / Csiszar-Tusnady):

        phi_{k+1}(E_j) = phi_k(E_j) * numerator / denominator

    where:
        numerator = sum_i [ A_i(E_j) * M_i / M_p_i ]
        denominator = sum_i [ A_i(E_j) ]

    and M_p_i = sum_j A_i(E_j) * phi_k(E_j) * d(ln E)_j is the
    computed reading for detector i.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    phi0 : np.ndarray
        Initial spectrum guess (n_energy,).
    b_meas : np.ndarray, optional
        Measurement uncertainties. If None, uniform weights.
    max_iter : int
        Maximum iterations (default: 200).
    tol_chi2 : float
        Stop when chi2 < tol_chi2 (default: 1.0).
    tol_rel : float
        Stop when relative change in spectrum < tol_rel (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, float, bool]
        (spectrum, n_iterations, final_chi2, converged)
    """
    phi = np.copy(phi0)
    phi = np.maximum(phi, 1e-30)  # avoid division by zero

    n_det = A_matrix.shape[0]
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(n_det)

    # Precompute denominator: sum_i A_i(E_j) for each energy bin
    denom = np.sum(A_matrix, axis=0)  # (n_energy,)
    denom = np.maximum(denom, 1e-30)

    for iteration in range(max_iter):
        # Compute model readings: M_p_i = sum_j A_i_j * phi_j * ln_steps_j
        M_p = A_matrix @ (phi * ln_steps)  # (n_det,)

        # Avoid division by zero in ratio
        M_p_safe = np.maximum(M_p, 1e-30)

        # Chi-squared
        residual = M_p - b_readings
        chi2 = np.mean(residual ** 2 * weights)

        # Check convergence
        if chi2 < tol_chi2:
            return phi, iteration + 1, chi2, True

        # Multiplicative update
        # numerator_j = sum_i [ A_i_j * M_i / M_p_i ]
        ratios = b_readings / M_p_safe  # (n_det,)
        numerator = A_matrix.T @ ratios  # (n_energy,)

        # phi_{k+1} = phi_k * numerator / denominator
        phi_new = phi * numerator / denom
        phi_new = np.maximum(phi_new, 1e-30)

        # Check relative change
        rel_change = np.max(np.abs(phi_new - phi)) / (np.max(phi) + 1e-30)
        phi = phi_new

        if rel_change < tol_rel:
            # Recompute chi2 with updated phi
            phi = _clean_edge_bins(phi)
            M_p_final = A_matrix @ (phi * ln_steps)
            chi2_final = np.mean((M_p_final - b_readings) ** 2 * weights)
            return phi, iteration + 1, chi2_final, True

    # Final chi2
    phi = _clean_edge_bins(phi)
    M_p_final = A_matrix @ (phi * ln_steps)
    chi2_final = np.mean((M_p_final - b_readings) ** 2 * weights)
    converged = chi2_final < tol_chi2

    return phi, max_iter, chi2_final, converged


# ------------------------------------------------------------------ #
#  Full pipeline solver
# ------------------------------------------------------------------ #

def solve_parametric2(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding using the full BON95 parametric pipeline.

    1. Grid search + NLS for parametric fit.
    2. Directed-divergence iteration refinement.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    b_meas : np.ndarray, optional
        Measurement uncertainties for weighted NLS.
    b_range, Tf_range, c_range : tuple
        Grid search ranges for shape parameters.
    max_iter : int
        Max directed-divergence iterations.
    tol_chi2 : float
        Chi-squared convergence threshold.

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    # Step 1: Parametric fit
    best_params, best_chi2, top_candidates = solve_bon95_parametric(
        A_matrix, b_readings, E, ln_steps,
        b_range=b_range, Tf_range=Tf_range, c_range=c_range,
        b_meas=b_meas, top_n=5,
    )

    # Build initial spectrum from best parametric fit
    phi_param = bon95_spectrum(
        E,
        best_params['b'], best_params['Tf'], best_params['c'],
        best_params['a1'], best_params['a2'],
        best_params['a3'], best_params['a4'],
    )

    # Ensure non-negative
    phi_param = np.maximum(phi_param, 0.0)
    phi_param = _clean_edge_bins(phi_param)

    # Step 2: Directed-divergence refinement
    phi_refined, n_iter, chi2_final, converged = directed_divergence_iteration(
        A_matrix, b_readings, E, ln_steps, phi_param,
        b_meas=b_meas, max_iter=max_iter, tol_chi2=tol_chi2,
    )

    # Build final spectrum: Phi(E) * ln_steps for the system matrix
    phi_refined = _clean_edge_bins(phi_refined)
    spectrum = phi_refined * ln_steps

    # Compute residual for fit quality check
    computed = A_matrix @ spectrum
    residual_norm = np.linalg.norm(computed - b_readings)
    _check_fit_quality(residual_norm, b_readings, "parametric2")

    message = (
        f"BON95 parametric fit (chi2={best_chi2:.4f}) + "
        f"DD iteration ({n_iter} iters, chi2={chi2_final:.4f})"
    )
    nfev = len(top_candidates)  # grid evaluations as proxy

    return spectrum, converged, message, nfev


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _check_fit_quality(residual_norm, b_readings, method_name="parametric2"):
    """Emit a warning if the fit residual is large."""
    b_norm = np.linalg.norm(b_readings)
    if b_norm > 0:
        relative_residual = residual_norm / b_norm
        if relative_residual > _RESIDUAL_WARN_THRESHOLD:
            warnings.warn(
                f"{method_name}: large residual "
                f"({residual_norm:.2e} / {b_norm:.2e} = {relative_residual:.1f}x). "
                f"The 4-component BON95 model may not represent this spectrum well.",
                UserWarning,
                stacklevel=3,
            )


def _clean_edge_bins(phi: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Zero out edge bins if anomalously large compared to neighbors.

    The BON95 model divides by E, which can create spikes at the first
    energy bin (tiny E). This function detects and removes such artifacts.

    Parameters
    ----------
    phi : np.ndarray
        Spectrum array.
    factor : float
        Threshold: if edge bin > factor * mean of valid neighbors, zero it.

    Returns
    -------
    np.ndarray
        Cleaned spectrum (new copy).
    """
    phi = np.copy(phi)
    n = len(phi)
    if n < 3:
        return phi

    # Check first bin
    neighbor_mean = np.mean(phi[1:3])
    if neighbor_mean > 0 and phi[0] > factor * neighbor_mean:
        phi[0] = 0.0

    # Check last bin
    neighbor_mean = np.mean(phi[-3:-1])
    if neighbor_mean > 0 and phi[-1] > factor * neighbor_mean:
        phi[-1] = 0.0

    return phi


def _build_measurement_uncertainties(
    b_readings: np.ndarray,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Estimate measurement uncertainties from readings.

    Uses a default relative uncertainty (noise_level) if not provided.
    """
    return np.abs(b_readings) * noise_level + 1e-30


# ------------------------------------------------------------------ #
#  Workflow wrapper
# ------------------------------------------------------------------ #

def unfold_parametric2(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    noise_level: float = 0.05,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the BON95 parametric method.

    Uses the four-component parameterization from Sannikov BON95:
    thermal (Maxwellian), epithermal (1/E), intermediate, and
    fast (evaporation/cascade) components. After parametric fitting,
    the result is refined by directed-divergence iterations.

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
    b_range : tuple
        Grid range for epithermal exponent b: (min, max, n_points).
    Tf_range : tuple
        Grid range for fast peak energy Tf (MeV): (min, max, n_points).
    c_range : tuple
        Grid range for fast peak width c: (min, max, n_points).
    noise_level : float
        Relative uncertainty for measurements (default: 0.05 = 5%).
    max_iter : int
        Max directed-divergence iterations (default: 200).
    tol_chi2 : float
        Chi-squared convergence threshold (default: 1.0).
    calculate_errors : bool
        Calculate Monte-Carlo errors (default: False).
    n_montecarlo : int
        Number of Monte-Carlo samples (default: 100).
    save_result : bool
        Save result to history (default: False).
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
        # Re-estimate uncertainties from the actual readings used
        b_meas_local = _build_measurement_uncertainties(b_vec, noise_level)
        x_opt, success, message, nfev = solve_parametric2(
            A_mat, b_vec, E_MeV, ln_steps,
            b_meas=b_meas_local,
            b_range=b_range, Tf_range=Tf_range, c_range=c_range,
            max_iter=max_iter, tol_chi2=tol_chi2,
        )
        return x_opt, nfev, success

    method_name = "parametric2"
    extra = {
        "b_range": b_range,
        "Tf_range": Tf_range,
        "c_range": c_range,
        "noise_level": noise_level,
        "bon95_Tth": _Tth,
    }

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
        method_name=method_name,
        extra_output=extra,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
