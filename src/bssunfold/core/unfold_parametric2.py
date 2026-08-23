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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from ._bon95 import (
    _DEFAULT_B_RANGE,
    _DEFAULT_C_RANGE,
    _DEFAULT_TF_RANGE,
    bon95_spectrum,
    solve_bon95_combined,
    solve_bon95_cvxpy,
    solve_bon95_parametric,
    solve_bon95_qpsolvers,
)
from ._matrix_utils import compute_log_steps
from ._parametric_shared import (
    _build_measurement_uncertainties,
    _check_fit_quality,
    _clean_edge_bins,
    _Tth,
)

__all__ = [
    "solve_bon95_parametric",
    "solve_bon95_cvxpy",
    "solve_bon95_qpsolvers",
    "solve_bon95_combined",
    "directed_divergence_iteration",
    "solve_parametric2",
    "unfold_parametric2",
]





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
        weights = np.where(b_meas > 0, 1.0 / (b_meas**2), 1.0)
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
        chi2 = np.mean(residual**2 * weights)

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
    optimizer: str = "grid",
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter_qp: int = 50,
    tol_qp: float = 1e-6,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding using the full BON95 parametric pipeline.

    1. Parametric fit using the selected optimizer.
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
    optimizer : str
        Parametric fit optimizer (default: "grid"):
        - ``"grid"``      -- grid search + NLS (default, no extra deps).
        - ``"cvxpy"``     -- SQP via cvxpy.
        - ``"qpsolvers"`` -- SQP via qpsolvers.
        - ``"combined"``  -- grid search + SQP refinement.
    b_range, Tf_range, c_range : tuple
        Grid search ranges for shape parameters (used by "grid" and "combined").
    alpha : float
        Tikhonov regularization for SQP optimizers (default: 1e-4).
    solver_backend : str
        QP backend for SQP optimizers (default: "auto").
    max_iter_qp : int
        Max SQP iterations for QP-based optimizers (default: 50).
    tol_qp : float
        SQP convergence tolerance (default: 1e-6).
    max_iter : int
        Max directed-divergence iterations (default: 200).
    tol_chi2 : float
        Chi-squared convergence threshold (default: 1.0).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    nfev = 0

    # Step 1: Parametric fit using selected optimizer
    if optimizer == "grid":
        best_params, best_chi2, top_candidates = solve_bon95_parametric(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_range=b_range,
            Tf_range=Tf_range,
            c_range=c_range,
            b_meas=b_meas,
            top_n=5,
        )
        phi_param = bon95_spectrum(
            E,
            best_params["b"],
            best_params["Tf"],
            best_params["c"],
            best_params["a1"],
            best_params["a2"],
            best_params["a3"],
            best_params["a4"],
        )
        nfev = len(top_candidates)

    elif optimizer == "cvxpy":
        spectrum_fit, _success, _msg, nfev = solve_bon95_cvxpy(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_meas=b_meas,
            alpha=alpha,
            solver_backend=solver_backend,
            max_iter=max_iter_qp,
            tol=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0  # computed after DD

    elif optimizer == "qpsolvers":
        spectrum_fit, _success, _msg, nfev = solve_bon95_qpsolvers(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_meas=b_meas,
            alpha=alpha,
            solver_backend=solver_backend,
            max_iter=max_iter_qp,
            tol=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0

    elif optimizer == "combined":
        spectrum_fit, _success, _msg, nfev = solve_bon95_combined(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_meas=b_meas,
            alpha=alpha,
            solver_backend=solver_backend,
            max_iter_qp=max_iter_qp,
            tol_qp=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0

    else:
        raise ValueError(
            f"Unknown optimizer: '{optimizer}'. "
            "Choose from 'grid', 'cvxpy', 'qpsolvers', 'combined'."
        )

    # Ensure non-negative
    phi_param = np.maximum(phi_param, 0.0)
    phi_param = _clean_edge_bins(phi_param)

    # Step 2: Directed-divergence refinement
    phi_refined, n_iter, chi2_final, converged = directed_divergence_iteration(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        phi_param,
        b_meas=b_meas,
        max_iter=max_iter,
        tol_chi2=tol_chi2,
    )

    # Build final spectrum: Phi(E) * ln_steps for the system matrix
    phi_refined = _clean_edge_bins(phi_refined)
    spectrum = phi_refined * ln_steps

    # Compute residual for fit quality check
    computed = A_matrix @ spectrum
    residual_norm = np.linalg.norm(computed - b_readings)
    _check_fit_quality(residual_norm, b_readings, "parametric2")

    if optimizer == "grid":
        message = (
            f"BON95 grid fit (chi2={best_chi2:.4f}) + "
            f"DD iteration ({n_iter} iters, chi2={chi2_final:.4f})"
        )
    else:
        message = (
            f"BON95 {optimizer} fit + "
            f"DD iteration ({n_iter} iters, chi2={chi2_final:.4f})"
        )

    return spectrum, converged, message, nfev


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
    optimizer: str = "grid",
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter_qp: int = 50,
    tol_qp: float = 1e-6,
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

    The ``optimizer`` parameter selects the parametric fit backend:

    * ``"grid"``      -- grid search + NLS (default, no extra deps).
    * ``"cvxpy"``     -- SQP via cvxpy.
    * ``"qpsolvers"`` -- SQP via qpsolvers.
    * ``"combined"``  -- grid search + SQP refinement.

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
    optimizer : str
        Parametric fit optimizer (default: "grid").
    b_range : tuple
        Grid range for b: (min, max, n_points). Used by "grid"/"combined".
    Tf_range : tuple
        Grid range for Tf (MeV): (min, max, n_points). Used by "grid"/"combined".
    c_range : tuple
        Grid range for c: (min, max, n_points). Used by "grid"/"combined".
    alpha : float
        Tikhonov regularization for SQP (default: 1e-4).
    solver_backend : str
        QP backend for SQP (default: "auto").
    max_iter_qp : int
        Max SQP iterations (default: 50).
    tol_qp : float
        SQP convergence tolerance (default: 1e-6).
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
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    log_steps = compute_log_steps(E_MeV, n_energy_bins)
    ln_steps = log_steps * np.log(10)

    def solve_wrapper(A_mat, b_vec, **kwargs):
        b_meas_local = _build_measurement_uncertainties(b_vec, noise_level)
        x_opt, success, _message, nfev = solve_parametric2(
            A_mat,
            b_vec,
            E_MeV,
            ln_steps,
            b_meas=b_meas_local,
            optimizer=optimizer,
            b_range=b_range,
            Tf_range=Tf_range,
            c_range=c_range,
            alpha=alpha,
            solver_backend=solver_backend,
            max_iter_qp=max_iter_qp,
            tol_qp=tol_qp,
            max_iter=max_iter,
            tol_chi2=tol_chi2,
        )
        return x_opt, nfev, success

    method_name = "parametric2"
    extra = {
        "optimizer": optimizer,
        "b_range": b_range,
        "Tf_range": Tf_range,
        "c_range": c_range,
        "alpha": alpha,
        "solver_backend": solver_backend,
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
