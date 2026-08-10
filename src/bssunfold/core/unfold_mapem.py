"""MAP-EM (penalised expectation maximisation) unfolding method.

Port of the ``OSMAPOSL`` (ordered-subset maximum a posteriori, one-step-late)
reconstruction algorithm from PyTomography
(https://github.com/PyTomography/PyTomography, MIT license) applied with a
single subset, i.e. a one-step-late penalised EM update:

    x^{n+1} = x^n * A^T ( b / (A x^n + eps) ) / ( A^T 1 + beta * grad V(x^n) )

where ``V`` is a nearest-neighbour prior over the energy axis. Available
priors (after PyTomography's ``QuadraticPrior``, ``LogCoshPrior`` and
``RelativeDifferencePrior``):

    quadratic            phi0 = 1/4 * ((fr - fs)/delta)^2
    logcosh              phi0 = log cosh((fr - fs)/delta)
    relative_difference  phi0 = (fr - fs)^2 / (fr + fs + gamma|fr - fs| + delta)

Set ``prior='none'`` to recover plain MLEM.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._em_priors import prior_gradient, prior_value

__all__ = ["solve_mapem", "unfold_mapem"]

_PRIORS = ("none", "quadratic", "logcosh", "relative_difference")


def solve_mapem(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    prior: str = "quadratic",
    beta: float = 1e-3,
    prior_delta: float = 1.0,
    gamma: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using penalised EM (OSMAPOSL, one-step-late).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    prior : str, optional
        Prior type: ``'none'``, ``'quadratic'``, ``'logcosh'`` or
        ``'relative_difference'`` (default: ``'quadratic'``).
    beta : float, optional
        Prior weight (default: 1e-3).
    prior_delta : float, optional
        Width parameter of the quadratic/logcosh priors and additive floor of
        the relative-difference prior (default: 1.0).
    gamma : float, optional
        Edge-preservation parameter of the relative-difference prior
        (default: 1.0).
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    prior = str(prior).lower()
    if prior not in _PRIORS:
        raise ValueError(
            f"Unknown prior {prior!r}. Choose from {list(_PRIORS)}"
        )

    eps = 1e-11
    x = np.maximum(x0, 0).copy()
    norm = A.sum(axis=0)
    converged = False
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it
        x_old = x.copy()

        ratio = b / (A @ x + eps)
        correction = A.T @ ratio

        if prior == "none":
            x = np.maximum(x * correction / (norm + eps), 0.0)
        else:
            grad = prior_gradient(x, prior, beta, prior_delta, gamma)
            x = np.maximum(x * correction / (norm + grad + eps), 0.0)

        rel = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + eps)
        if rel < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_mapem(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    prior: str = "quadratic",
    beta: float = 1e-3,
    prior_delta: float = 1.0,
    gamma: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using penalised EM (MAP-EM).

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
        Initial spectrum guess. If None, a flat spectrum is used.
    prior : str, optional
        Prior type: ``'none'``, ``'quadratic'``, ``'logcosh'`` or
        ``'relative_difference'`` (default: ``'quadratic'``).
    beta : float, optional
        Prior weight (default: 1e-3).
    prior_delta : float, optional
        Width parameter of the quadratic/logcosh priors and additive floor of
        the relative-difference prior (default: 1.0).
    gamma : float, optional
        Edge-preservation parameter of the relative-difference prior
        (default: 1.0).
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins)

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
        solve_func=make_solve_wrapper(
            solve_mapem,
            prior=prior,
            beta=beta,
            prior_delta=prior_delta,
            gamma=gamma,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="MAP-EM",
        extra_output={
            "prior": str(prior),
            "beta": float(beta),
            "prior_delta": float(prior_delta),
            "gamma": float(gamma),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=False,
    )

    if str(prior).lower() != "none":
        result["prior_value"] = prior_value(
            result["spectrum"], prior, beta, prior_delta, gamma
        )

    if save_result:
        save_result_callback(result)

    return result
