"""Extragradient (Korpelevich) unfolding method for neutron spectra.

Korpelevich's extragradient method (1976) for variational inequalities and
saddle-point problems (lecture 13 of the MIPT optimization course,
homework 20).  The method is applied here to a *robust* unfolding problem
that guards against measurement noise of bounded L2 norm:

    min_{x >= 0}  max_{||y||_2 <= 1}  1/2 ||A x - b||^2 + delta * y^T (A x - b)

which is exactly  ``min_{x>=0} 1/2 ||A x - b||^2 + delta * ||A x - b||_2``
вЂ” the least-squares functional made convex-robust against residual vectors
with ``||delta_b||_2 <= delta`` (``delta = noise_level * ||b||_2``).  The
bilinear saddle form is solved with the two-step extragradient scheme

    prediction:  (x~, y~) = P( x_k - eta * F_x(x_k, y_k),
                               y_k + eta * (A x_k - b) )
    correction:  (x_{k+1}, y_{k+1}) = P( x_k - eta * F_x(x~, y~),
                                         y_k + eta * (A x~ - b) )

with ``F_x(x, y) = A^T (A x - b) + delta A^T y`` and P the product of the
nonnegativity projection (for x) and the unit-ball projection (for y).
The extra gradient evaluation is precisely what restores convergence for
monotone operators where the plain gradient step oscillates.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_extragradient", "unfold_extragradient"]


def _project_ball(v: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    """Euclidean projection onto the L2 ball of the given radius."""
    n = float(np.linalg.norm(v))
    if n > radius:
        return v * (radius / max(n, 1e-300))
    return v


def solve_extragradient(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    noise_level: float = 0.02,
    step_size: float | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the robust unfolding saddle problem by extragradient.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); projected onto the nonnegative orthant.
    max_iterations : int, optional
        Maximum iterations (default: 2000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    noise_level : float, optional
        Relative radius of the noise ball, ``delta = noise_level ||b||_2``
        (default: 0.02).
    step_size : float, optional
        Extragradient step ``eta``.  If None (default), set to
        ``0.9 / L`` with ``L = ||A||_2^2 + delta ||A||_2`` вЂ” an upper bound
        on the Lipschitz constant of the saddle operator.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    noise_level = max(float(noise_level), 0.0)
    delta = noise_level * float(np.linalg.norm(b))

    x = np.maximum(x0, 0.0)
    y = np.zeros(A.shape[0])  # dual noise direction

    norm_A = float(np.linalg.norm(A, 2))
    if step_size is None:
        L = norm_A**2 + delta * norm_A + 1.0
        eta = 0.9 / L
    else:
        eta = float(step_size)

    def F_x(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
        return A.T @ (A @ xv - b) + delta * (A.T @ yv)

    converged = False
    iterations = 0
    for k in range(max_iterations):
        residual = A @ x - b

        # ---- prediction step ------------------------------------------------
        x_tilde = np.maximum(x - eta * F_x(x, y), 0.0)
        y_tilde = _project_ball(y + eta * residual, 1.0)

        # ---- correction step (extra gradient evaluation) --------------------
        residual_t = A @ x_tilde - b
        x_new = np.maximum(x - eta * F_x(x_tilde, y_tilde), 0.0)
        y_new = _project_ball(y + eta * residual_t, 1.0)

        rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-30)
        x, y = x_new, y_new
        iterations = k + 1
        if rel_change < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_extragradient(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    noise_level: float = 0.02,
    step_size: float | None = None,
    calculate_errors: bool = False,
    mc_noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using Korpelevich's extragradient method.

    Solves the robust saddle formulation ``min_{x>=0} 1/2||Ax-b||^2 +
    delta*||Ax-b||_2`` via its bilinear saddle form with the two-step
    extragradient scheme; ``delta = noise_level * ||b||_2`` bounds the
    assumed measurement-noise norm.

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
    max_iterations : int, optional
        Maximum iterations (default: 2000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    noise_level : float, optional
        Relative noise-ball radius (default: 0.02).
    step_size : Optional[float], optional
        Extragradient step; auto from the Lipschitz bound when None.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    mc_noise_level : float, optional
        Noise level for Monte-Carlo uncertainty (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        Monte-Carlo variance reduction: ``'none'``, ``'antithetic'``,
        ``'control'`` or ``'both'`` (default: 'none').
    save_result : bool, optional
        Save result to history (default: False).
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
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_extragradient,
            max_iterations=max_iterations,
            tolerance=tolerance,
            noise_level=noise_level,
            step_size=step_size,
        ),
        solve_kwargs={},
        method_name="Extragradient",
        extra_output={
            "noise_ball_radius": noise_level,
        },
        calculate_errors=calculate_errors,
        noise_level=mc_noise_level,
        n_montecarlo=n_montecarlo,
        variance_reduction=variance_reduction,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
