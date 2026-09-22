"""Coordinate descent unfolding method for neutron spectra.

Block-free coordinate descent for the non-negative least-squares problem

    min_{x >= 0}  1/2 ||A x - b||^2 + l1_penalty * ||x||_1
                  + (l2_penalty/2) * ||x||^2

(lecture 15 of the MIPT optimization course).  Each step updates a single
coordinate in closed form

    x_j <- max(0, (a_j^T r + ||a_j||^2 x_j - l1) / (||a_j||^2 + l2))

where ``r = b - A x`` is the running residual, so every coordinate sweep
costs O(m * n) total with no matrix products вЂ” the same work as one
Landweber iteration, but with exact minimization along each coordinate.
Under non-negativity the L1 prox degenerates to a thresholded maximum,
which keeps the update one line long.

Coordinate order is either cyclic (classical Gauss-Seidel-type CD) or
random (sampled with a seeded RNG, as in random-CD variants of the course).
The method is particularly well suited to Bonner-sphere systems because the
Gram-scale quantities are the *column* norms of A, which are independent of
the conditioning of A^T A.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_coordinate_descent", "unfold_coordinate_descent"]


def solve_coordinate_descent(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    selection: str = "cyclic",
    random_state: int | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem by coordinate descent.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); projected onto the nonnegative orthant.
    max_iterations : int, optional
        Maximum full sweeps over all coordinates (default: 2000).
    tolerance : float, optional
        Relative change tolerance on the spectrum per sweep (default: 1e-8).
    l1_penalty : float, optional
        L1 penalty weight (default: 0.0).
    l2_penalty : float, optional
        Ridge penalty weight (multiplies 1/2 ||x||^2; default: 0.0).
    selection : str, optional
        Coordinate order: ``'cyclic'`` or ``'random'`` (default: 'cyclic').
    random_state : int, optional
        Seed for the random coordinate order.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    selection = selection.lower()
    if selection not in ("cyclic", "random"):
        raise ValueError("selection must be 'cyclic' or 'random'")

    m, n = A.shape
    x = np.maximum(x0, 0.0)
    l1_penalty = max(float(l1_penalty), 0.0)
    l2_penalty = max(float(l2_penalty), 0.0)

    col_sq = np.einsum("ij,ij->j", A, A)  # ||a_j||^2
    residual = b - A @ x

    rng = np.random.default_rng(random_state)

    converged = False
    iterations = 0
    for k in range(max_iterations):
        x_prev = x.copy()
        if selection == "cyclic":
            order = range(n)
        else:
            order = rng.permutation(n)

        for j in order:
            c = col_sq[j]
            if c <= 0:
                continue
            # partial correlation: a_j^T (r + a_j x_j) = a_j^T r + c x_j
            rho_j = float(A[:, j] @ residual) + c * x[j]
            x_new = (rho_j - l1_penalty) / (c + l2_penalty)
            x_new = max(x_new, 0.0)
            delta = x_new - x[j]
            if delta != 0.0:
                residual -= delta * A[:, j]
                x[j] = x_new

        iterations = k + 1
        rel_change = np.linalg.norm(x - x_prev) / max(np.linalg.norm(x), 1e-30)
        if rel_change < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_coordinate_descent(
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
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    selection: str = "cyclic",
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
    """Unfold neutron spectrum using coordinate descent.

    Exact closed-form coordinate minimization of the NNLS objective with
    optional L1/L2 penalties; O(m) per coordinate via a running residual.

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
        Maximum sweeps (default: 2000).
    tolerance : float, optional
        Relative change tolerance (default: 1e-8).
    l1_penalty : float, optional
        L1 penalty weight (default: 0.0).
    l2_penalty : float, optional
        Ridge penalty weight (default: 0.0).
    selection : str, optional
        ``'cyclic'`` or ``'random'`` coordinate order (default: 'cyclic').
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        Monte-Carlo variance reduction: ``'none'``, ``'antithetic'``,
        ``'control'`` or ``'both'`` (default: 'none').
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility (also used by 'random' selection).

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
            solve_coordinate_descent,
            max_iterations=max_iterations,
            tolerance=tolerance,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            selection=selection,
            random_state=random_state,
        ),
        solve_kwargs={},
        method_name="Coordinate Descent",
        extra_output={
            "l1_penalty": l1_penalty,
            "l2_penalty": l2_penalty,
            "coordinate_selection": selection,
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
