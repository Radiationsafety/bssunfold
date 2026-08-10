"""BSREM (block sequential regularised expectation maximisation) unfolding.

Port of the ``BSREM`` reconstruction algorithm from PyTomography
(https://github.com/PyTomography/PyTomography, MIT license) adapted to
neutron spectrum unfolding. BSREM is a penalised EM with a user-supplied
relaxation sequence ``alpha(n)`` that guarantees convergence for non-convex
priors, and a floor clamp that prevents spectrum bins from being locked at
zero:

    x^{n+1} = x^n + alpha(n)/(omega_m * A^T 1 + eps)
                  * ( A_m^T ( b_m/(A_m x^n + eps) ) - A_m^T 1 - omega_m * beta * grad V(x^n) )

where ``m`` indexes the detector subset, ``omega_m`` is the subset fraction
and ``V`` is a nearest-neighbour prior over the energy axis (see
:mod:`bssunfold.core.unfold_mapem`). After every sub-iteration the spectrum
is clamped to be at least ``addition_after_iteration``.
"""

import numpy as np
from typing import Callable, Dict, Optional, Any, List, Tuple, Union

from ._base_unfolder import run_unfolding, make_solve_wrapper
from ._em_priors import prior_gradient

__all__ = ["solve_bsrem", "unfold_bsrem"]

_PRIORS = ("none", "quadratic", "logcosh", "relative_difference")


def solve_bsrem(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    prior: str = "none",
    beta: float = 1e-3,
    prior_delta: float = 1.0,
    gamma: float = 1.0,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
    relaxation: Optional[Union[float, Callable[[int], float]]] = None,
    addition_after_iteration: float = 1e-4,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using BSREM.

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
        ``'relative_difference'`` (default: ``'none'``).
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
    n_subsets : int, optional
        Number of ordered subsets over the detector readings
        (default: 1).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    relaxation : float or callable, optional
        Relaxation sequence ``alpha(n)`` as a constant or as a callable of
        the iteration number. If None, a constant 1 is used (default: None).
    addition_after_iteration : float, optional
        Floor value the spectrum is clamped to after every sub-iteration to
        prevent bins being locked at zero (default: 1e-4).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    m, n = A.shape

    if n_subsets < 1:
        raise ValueError("n_subsets must be >= 1")
    if n_subsets > m:
        raise ValueError(
            f"n_subsets ({n_subsets}) must not exceed the number of "
            f"detectors ({m})"
        )

    prior = str(prior).lower()
    if prior not in _PRIORS:
        raise ValueError(
            f"Unknown prior {prior!r}. Choose from {list(_PRIORS)}"
        )

    if relaxation is None:
        def relaxation_seq(_n):
            return 1.0
    elif callable(relaxation):
        relaxation_seq = relaxation
    else:
        relax_val = float(relaxation)

        def relaxation_seq(_n):
            return relax_val

    eps = 1e-11
    subset_indices = np.array_split(np.arange(m), n_subsets)
    norm_all = A.sum(axis=0)
    x = np.maximum(x0, 0).copy()
    converged = False
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it
        x_old = x.copy()
        alpha = float(relaxation_seq(it))

        for idx in subset_indices:
            omega = len(idx) / m
            A_sub = A[idx]
            b_sub = b[idx]
            norm_sub = A_sub.sum(axis=0)

            ratio = b_sub / (A_sub @ x + eps)
            correction = A_sub.T @ ratio
            if prior == "none":
                grad = np.zeros(n)
            else:
                grad = omega * prior_gradient(x, prior, beta, prior_delta, gamma)

            update = correction - norm_sub - grad
            step = np.where(
                omega * norm_all > eps, alpha / (omega * norm_all + eps), 0.0
            )
            x = x + x * step * update
            x[x <= addition_after_iteration] = addition_after_iteration

        rel = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + eps)
        if rel < tolerance:
            converged = True
            break

    return x, iterations, converged


def unfold_bsrem(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    prior: str = "none",
    beta: float = 1e-3,
    prior_delta: float = 1.0,
    gamma: float = 1.0,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
    relaxation: Optional[Union[float, Callable[[int], float]]] = None,
    addition_after_iteration: float = 1e-4,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the BSREM algorithm.

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
        ``'relative_difference'`` (default: ``'none'``).
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
    n_subsets : int, optional
        Number of ordered subsets over the detector readings (default: 1).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    relaxation : float or callable, optional
        Relaxation sequence (default: None -> constant 1).
    addition_after_iteration : float, optional
        Floor value for spectrum bins (default: 1e-4).
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
            solve_bsrem,
            prior=prior,
            beta=beta,
            prior_delta=prior_delta,
            gamma=gamma,
            max_iterations=max_iterations,
            n_subsets=n_subsets,
            tolerance=tolerance,
            relaxation=relaxation,
            addition_after_iteration=addition_after_iteration,
        ),
        solve_kwargs={},
        method_name="BSREM",
        extra_output={
            "prior": str(prior),
            "beta": float(beta),
            "n_subsets": int(n_subsets),
            "addition_after_iteration": float(addition_after_iteration),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
