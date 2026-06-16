"""Statistical Regularization (Turchin's method) unfolding.

Pure numpy/scipy implementation — no external statreg dependency.

Implements Turchin's method of statistical regularisation:
  φ̂ = argmin { ½‖Σ⁻¹⸍²(Aφ−b)‖² + ½ α ‖D₂ φ‖² }

where D₂ is the second-order finite difference operator.
Regularisation parameter α is selected either by the user or automatically
via the L-curve heuristic (maximum curvature).
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_statreg", "unfold_statreg"]


def _build_penalty_matrix(n: int) -> np.ndarray:
    """Build second-order finite difference matrix (n-2 × n)."""
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _lcurve_statreg(
    A_tilde: np.ndarray,
    b_tilde: np.ndarray,
    L: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-8, 1e3),
) -> float:
    """Select α by L-curve corner (maximum curvature).

    Works in the whitened space: A_tilde = Σ⁻¹⸍² A,  b_tilde = Σ⁻¹⸍² b.
    """
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    ATA = A_tilde.T @ A_tilde
    ATb = A_tilde.T @ b_tilde
    LTL = L.T @ L

    residuals = []
    norms = []

    for alpha in alphas:
        P = ATA + alpha * LTL
        try:
            x = np.linalg.solve(P, ATb)
            x = np.maximum(x, 0)
            residuals.append(np.linalg.norm(A_tilde @ x - b_tilde))
            norms.append(np.linalg.norm(L @ x))
        except np.linalg.LinAlgError:
            continue

    if len(residuals) < 3:
        return 1.0

    log_res = np.log(np.maximum(residuals, 1e-300))
    log_norm = np.log(np.maximum(norms, 1e-300))

    p1 = np.array([log_res[0], log_norm[0]])
    p2 = np.array([log_res[-1], log_norm[-1]])
    v = p2 - p1
    edge = np.linalg.norm(v)
    if edge < 1e-300:
        return float(alphas[len(residuals) // 2])

    distances = []
    for i in range(len(residuals)):
        w = np.array([log_res[i], log_norm[i]]) - p1
        d = abs(v[0] * w[1] - v[1] * w[0]) / edge
        distances.append(d)

    idx = int(np.argmax(distances))
    return float(alphas[idx])


def solve_statreg(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    E_MeV: Optional[np.ndarray] = None,
    unfoldermethod: str = "EmpiricalBayes",
    regularization: Optional[float] = None,
    basis_name: str = "CubicSplines",
    boundary: Optional[str] = None,
    derivative_degree: int = 2,
) -> np.ndarray:
    """Solve unfolding problem using Turchin's statistical regularisation.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m × n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    E_MeV : np.ndarray, optional
        Energy grid (n,). Used for log-energy penalty scaling.
    unfoldermethod : str, optional
        Regularisation method: ``'EmpiricalBayes'`` (L-curve, default) or
        ``'User'`` (fixed α).
    regularization : float, optional
        Regularisation parameter α for ``'User'`` method (default: 1e-4).
    basis_name : str, optional
        Ignored (kept for API compatibility).
    boundary : str, optional
        Ignored (kept for API compatibility).
    derivative_degree : int, optional
        Derivative order for penalty. Only 2 is implemented.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    n_ene = A.shape[1]

    L = _build_penalty_matrix(n_ene)

    sigma = np.maximum(b * 0.05, 1e-300)
    sigma_inv = 1.0 / sigma
    A_tilde = A * sigma_inv[:, None]
    b_tilde = b * sigma_inv

    if unfoldermethod == "User":
        alpha = 1e-4 if regularization is None else float(regularization)
    elif unfoldermethod == "EmpiricalBayes":
        alpha = _lcurve_statreg(A_tilde, b_tilde, L)
    else:
        raise ValueError(f"Unknown method: {unfoldermethod}")

    ATA = A_tilde.T @ A_tilde
    ATb = A_tilde.T @ b_tilde
    LTL = L.T @ L

    try:
        x = np.linalg.solve(ATA + alpha * LTL, ATb)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(ATA + alpha * LTL, ATb, rcond=None)[0]

    x = np.maximum(x, 0)
    return x


def unfold_statreg(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    unfoldermethod: str = "EmpiricalBayes",
    regularization: Optional[float] = None,
    basis_name: str = "CubicSplines",
    boundary: Optional[str] = None,
    derivative_degree: int = 2,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Turchin's statistical regularisation.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess.
    unfoldermethod : str, optional
        Regularisation method (default: ``'EmpiricalBayes'``).
    regularization : float, optional
        Regularisation parameter for ``'User'`` method.
    basis_name : str, optional
        Ignored (kept for API compatibility).
    boundary : str, optional
        Ignored (kept for API compatibility).
    derivative_degree : int, optional
        Derivative degree (default: 2).
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
    x0_default = np.zeros(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_statreg,
            E_MeV=E_MeV,
            unfoldermethod=unfoldermethod,
            regularization=regularization,
            basis_name=basis_name,
            boundary=boundary,
            derivative_degree=derivative_degree,
        ),
        solve_kwargs={},
        method_name="StatReg",
        extra_output={
            "unfoldermethod": unfoldermethod,
            "basis_name": basis_name,
            "derivative_degree": derivative_degree,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
