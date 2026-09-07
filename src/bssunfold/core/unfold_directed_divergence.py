"""Directed-divergence unfolding for Bonner-sphere response matrices.

The solver minimizes the Poisson/I-divergence data term with multiplicative
updates.  Optional first- or second-order Tikhonov smoothing is applied as a
non-negative proximal step after each update.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import compute_log_steps, create_derivative_matrix

__all__ = ["solve_directed_divergence", "unfold_directed_divergence"]


def solve_directed_divergence(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 200,
    tol_chi2: float = 1.0,
    tol_rel: float = 1e-6,
    relative_uncertainty: float = 0.05,
    sigma: Optional[np.ndarray] = None,
    smoothness_order: int = 0,
    smoothness_weight: float = 0.0,
) -> Tuple[np.ndarray, int, bool]:
    """Solve a non-negative unfolding problem by directed divergence.

    ``A`` must be the response matrix used by :class:`Detector`; its columns
    already include the energy-bin integration weights.  Smoothing is the
    proximal Tikhonov step ``argmin ||x-y||^2 + alpha ||Lx||^2``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.maximum(np.asarray(x0, dtype=float), 1e-30).copy()
    if A.ndim != 2 or b.ndim != 1 or x.ndim != 1:
        raise ValueError("A, b, and x0 must be one- or two-dimensional arrays")
    if A.shape != (b.size, x.size):
        raise ValueError("Response matrix and vector dimensions are inconsistent")
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(b)):
        raise ValueError("A and b must contain finite values")
    if np.any(b < 0):
        raise ValueError("Directed divergence requires non-negative measurements")
    if max_iterations <= 0 or tol_chi2 < 0 or tol_rel < 0:
        raise ValueError("Iteration limits and tolerances must be non-negative")
    if smoothness_order not in (0, 1, 2):
        raise ValueError("smoothness_order must be 0, 1, or 2")
    if smoothness_weight < 0:
        raise ValueError("smoothness_weight must be non-negative")

    if sigma is None:
        sigma = np.maximum(relative_uncertainty * np.maximum(b, 1e-30), 1e-30)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != b.shape or np.any(sigma <= 0):
            raise ValueError("sigma must be positive and match b")
    weights = 1.0 / sigma**2
    denominator = np.maximum(A.sum(axis=0), 1e-30)
    penalty = None
    if smoothness_order and smoothness_weight:
        L = create_derivative_matrix(x.size, smoothness_order).toarray()
        penalty = smoothness_weight * (L.T @ L)

    for iteration in range(1, max_iterations + 1):
        predicted = np.maximum(A @ x, 1e-30)
        chi2 = float(np.mean(((predicted - b) ** 2) * weights))
        if chi2 <= tol_chi2:
            return x, iteration, True

        update = A.T @ (b / predicted)
        new_x = x * update / denominator
        if penalty is not None:
            new_x = np.linalg.solve(np.eye(x.size) + penalty, new_x)
        new_x = np.maximum(new_x, 1e-30)
        relative_change = np.max(np.abs(new_x - x) / np.maximum(x, 1e-30))
        x = new_x
        if relative_change <= tol_rel:
            return x, iteration, True

    return x, max_iterations, False


def unfold_directed_divergence(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 200,
    tol_chi2: float = 1.0,
    tol_rel: float = 1e-6,
    relative_uncertainty: float = 0.05,
    smoothness_order: int = 0,
    smoothness_weight: float = 0.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold Bonner-sphere readings using directed divergence."""
    steps = compute_log_steps(np.asarray(E_MeV), n_energy_bins)
    default = np.ones(n_energy_bins)
    A = np.array([sensitivities[name] for name in detector_names if name in readings])
    b = np.array([readings[name] for name in detector_names if name in readings])
    scale = np.mean(b) / max(float(np.mean(np.sum(A, axis=1))), 1e-30)
    default *= scale

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=default,
        solve_func=make_solve_wrapper(
            solve_directed_divergence,
            max_iterations=max_iterations,
            tol_chi2=tol_chi2,
            tol_rel=tol_rel,
            relative_uncertainty=relative_uncertainty,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
        ),
        solve_kwargs={},
        method_name="Directed divergence",
        extra_output={
            "tol_chi2": float(tol_chi2),
            "tol_rel": float(tol_rel),
            "smoothness_order": int(smoothness_order),
            "smoothness_weight": float(smoothness_weight),
            "log_steps": steps,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
