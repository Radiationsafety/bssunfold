"""SpecKit-style gradient unfolding with log-smoothness regularisation.

This module implements the gradient-descent spectrum unfolding used by SpecKit
("SpecKit (gradient + log-smoothness)"), adapted to the bssunfold solver API
and made a little more robust than the original heuristic:

* the gradient now contains *both* the data (least-squares) term and the
  log-smoothness regulariser, instead of regularising only the stopping
  criterion;
* the step size is chosen by backtracking line search, so the objective is
  monotonically non-increasing;
* a chi-square / discrepancy stopping rule is retained but expressed in the
  natural sum-of-squares scale.

The solver returns ``(spectrum, iterations, converged)`` like the other
bssunfold core solvers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_speckit", "unfold_speckit"]


def _objective(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    smoothness_lambda: float,
    min_value: float,
) -> float:
    residual = A @ x - b
    data_term = float(np.mean(residual * residual))
    if x.size < 2:
        smooth_term = 0.0
    else:
        logs = np.log(np.maximum(x, min_value))
        diffs = np.diff(logs)
        smooth_term = float(np.mean(diffs * diffs))
    return data_term + float(smoothness_lambda) * smooth_term


def _gradient(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    smoothness_lambda: float,
    min_value: float,
) -> np.ndarray:
    m, n = A.shape
    residual = A @ x - b
    grad = (2.0 / max(m, 1)) * (A.T @ residual)

    if smoothness_lambda != 0.0 and n >= 2:
        logs = np.log(np.maximum(x, min_value))
        diffs = np.diff(logs)  # d_i = log(x_{i+1}) - log(x_i)
        # gradient of mean(d_i^2) wrt. x_j:
        #   boundary terms plus 2*(d_{j-1}-d_j)/((n-1)*x_j) inside.
        smooth_grad = np.zeros(n, dtype=float)
        smooth_grad[0] = -2.0 * diffs[0] / ((n - 1) * max(x[0], min_value))
        if n > 2:
            smooth_grad[1:-1] = (
                2.0 * (diffs[:-1] - diffs[1:])
                / ((n - 1) * np.maximum(x[1:-1], min_value))
            )
        smooth_grad[-1] = 2.0 * diffs[-1] / ((n - 1) * max(x[-1], min_value))
        grad = grad + float(smoothness_lambda) * smooth_grad
    return grad


def solve_speckit(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    learning_rate: float | None = None,
    smoothness_lambda: float = 0.01,
    min_value: float = 1e-12,
    chi_square_stop: bool = True,
) -> tuple[np.ndarray, int, bool]:
    """Solve an unfolding problem by gradient descent with log-smoothness.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    max_iterations : int, optional
        Maximum number of gradient steps (default: 1000).
    tolerance : float, optional
        Relative-change convergence tolerance (default: 1e-6).
    learning_rate : float, optional
        Initial step size. If None, a conservative estimate based on the
        Lipschitz constant of the data term is used.
    smoothness_lambda : float, optional
        Weight of the log-smoothness regulariser (default: 0.01).
    min_value : float, optional
        Lower bound enforced on every spectrum bin (default: 1e-12).
    chi_square_stop : bool, optional
        Stop early once the sum of squared residuals reaches the 95 %
        chi-square level for ``m`` degrees of freedom (default: True).

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(solution, iterations, converged)``.
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    m, n = A.shape

    x = np.maximum(np.asarray(x0, dtype=float).reshape(-1).copy(), min_value)

    # Initial scale: match the measurements in the least-squares sense.
    Ax0 = A @ x
    denom = float(Ax0 @ Ax0)
    if denom > 0.0:
        scale = float(b @ Ax0) / denom
        if np.isfinite(scale) and scale > 0.0:
            x = np.maximum(x * scale, min_value)

    if learning_rate is None:
        # Lipschitz estimate of the (unscaled) data term, plus a safety margin.
        L_data = 2.0 * (np.linalg.norm(A, 2) ** 2) / max(m, 1)
        lr = 1.0 / max(L_data + 1e-12, 1e-12)
    else:
        lr = float(learning_rate)
    if not np.isfinite(lr) or lr <= 0.0:
        lr = 1e-3

    chi2_threshold = None
    if chi_square_stop and m > 0:
        try:
            chi2_threshold = float(chi2.ppf(0.95, m))
        except Exception:  # pragma: no cover - scipy always available here
            chi2_threshold = None

    converged = False
    iterations = 0
    best_loss = _objective(A, b, x, smoothness_lambda, min_value)

    for k in range(max_iterations):
        grad = _gradient(A, b, x, smoothness_lambda, min_value)
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm):
            break

        current_loss = _objective(A, b, x, smoothness_lambda, min_value)

        # Backtracking line search (monotone decrease).
        step = lr
        x_new = None
        for _ in range(60):
            candidate = x - step * grad
            candidate = np.maximum(candidate, min_value)
            cand_loss = _objective(A, b, candidate, smoothness_lambda, min_value)
            if np.isfinite(cand_loss) and cand_loss <= current_loss + 1e-18:
                x_new = candidate
                break
            step *= 0.5
        if x_new is None:
            x_new = np.maximum(x - 1e-12 * grad, min_value)

        rel_change = float(np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-30))
        x = x_new
        iterations = k + 1

        new_loss = _objective(A, b, x, smoothness_lambda, min_value)
        if new_loss < best_loss:
            best_loss = new_loss

        # Chi-square / discrepancy stopping on the data term.
        if chi2_threshold is not None:
            residual_ssq = float(np.sum((A @ x - b) ** 2))
            if residual_ssq <= chi2_threshold:
                converged = True
                break

        if grad_norm <= tolerance:
            converged = True
            break
        if rel_change <= tolerance:
            converged = True
            break

    if not converged:
        iterations = max_iterations

    x = np.maximum(x, 0.0)
    return x, iterations, converged


def unfold_speckit(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    learning_rate: float | None = None,
    smoothness_lambda: float = 0.01,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum by gradient descent with log-smoothness."""
    x0_default = np.ones(n_energy_bins, dtype=float)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_speckit,
            max_iterations=max_iterations,
            tolerance=tolerance,
            learning_rate=learning_rate,
            smoothness_lambda=smoothness_lambda,
        ),
        solve_kwargs={},
        method_name="SpecKit (gradient + log-smoothness)",
        extra_output={
            "learning_rate": learning_rate,
            "smoothness_lambda": smoothness_lambda,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        variance_reduction=variance_reduction,
        random_state=random_state,
        save_result=save_result,
        reading_uncertainties=reading_uncertainties,
        reading_covariance=reading_covariance,
        noise_model=noise_model,
        measurement_time=measurement_time,
    )
