"""Iterative refinement unfolding method for neutron spectrum reconstruction.

Two-pass method that combines a fast first-pass solver (e.g. MLEM or
Bayes) with a residual-correction second-pass solver (e.g. Landweber
or CGLS).

Algorithm
--------
1. **First pass** – use a fast EM-type method with few iterations to
   capture the gross spectral structure.
2. **Compute residual** ``r = b - A @ x1``.
3. **Second pass** – use a gradient-based method on the residual to
   correct systematic errors: ``x2 = solve_2nd(A, r, 0)``.
4. **Combine** ``x_final = x1 + alpha * x2`` where *alpha* can be
   fixed or selected via discrepancy principle.

The idea is that the first pass captures the gross structure while the
second pass corrects systematic errors that EM-type methods tend to
leave behind.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..logging_config import get_logger
from ._base_unfolder import _build_system

__all__ = ["solve_iterative_refinement", "unfold_iterative_refinement"]

logger = get_logger("unfold_iterative_refinement")


def solve_iterative_refinement(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    first_pass_solver: Optional[Callable] = None,
    second_pass_solver: Optional[Callable] = None,
    first_pass_kwargs: Optional[Dict[str, Any]] = None,
    second_pass_kwargs: Optional[Dict[str, Any]] = None,
    alpha: Optional[float] = None,
    max_alpha_search: int = 20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve unfolding problem using iterative refinement (two-pass method).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (n,).
    first_pass_solver : callable, optional
        Solver for the first pass.  Must accept ``(A, b, x0, **kwargs)``
        and return a 1-D array (or tuple whose first element is the
        solution).  Defaults to ``solve_mlem``.
    second_pass_solver : callable, optional
        Solver for the second pass (residual correction).  Same signature
        as *first_pass_solver*.  Defaults to ``solve_landweber``.
    first_pass_kwargs : dict, optional
        Keyword arguments forwarded to *first_pass_solver*
        (default: ``{max_iterations: 150, tolerance: 1e-4}``).
    second_pass_kwargs : dict, optional
        Keyword arguments forwarded to *second_pass_solver*
        (default: ``{max_iterations: 100, tolerance: 1e-5}``).
    alpha : float, optional
        Blending factor in ``x_final = x1 + alpha * x2``.
        If *None*, selected automatically so that
        ``||A x_final - b||`` is minimised over a line search.
    max_alpha_search : int, optional
        Number of candidate alpha values for line search (default: 20).

    Returns
    -------
    tuple
        ``(spectrum, info)`` with diagnostics.
    """
    n = A.shape[1]
    if x0 is None:
        x0 = np.ones(n) * 0.5

    # Default solvers
    if first_pass_solver is None:
        from .unfold_mlem import solve_mlem
        first_pass_solver = solve_mlem
    if second_pass_solver is None:
        from .unfold_landweber import solve_landweber
        second_pass_solver = solve_landweber

    if first_pass_kwargs is None:
        first_pass_kwargs = {"max_iterations": 150, "tolerance": 1e-4}
    if second_pass_kwargs is None:
        second_pass_kwargs = {"max_iterations": 100, "tolerance": 1e-5}

    # --- First pass ---
    res1 = first_pass_solver(A, b, x0, **first_pass_kwargs)
    x1 = res1[0] if isinstance(res1, tuple) else np.asarray(res1)
    x1 = np.maximum(np.asarray(x1, dtype=float).ravel(), 0)

    # --- Residual ---
    r = b - A @ x1

    # --- Second pass on residual ---
    x0_zero = np.zeros(n)
    res2 = second_pass_solver(A, r, x0_zero, **second_pass_kwargs)
    x2 = res2[0] if isinstance(res2, tuple) else np.asarray(res2)
    x2 = np.asarray(x2, dtype=float).ravel()

    # --- Combine ---
    if alpha is not None:
        # Fixed alpha
        best_alpha = alpha
    else:
        # Line search: minimise ||A(x1 + a*x2) - b||
        candidates = np.linspace(0.0, 2.0, max_alpha_search)
        best_alpha = 0.0
        best_res = np.linalg.norm(A @ x1 - b)
        for a in candidates:
            x_cand = x1 + a * x2
            res_cand = np.linalg.norm(A @ x_cand - b)
            if res_cand < best_res:
                best_res = res_cand
                best_alpha = a

    spectrum = np.maximum(x1 + best_alpha * x2, 0)

    info = {
        "first_pass_residual": float(np.linalg.norm(r)),
        "second_pass_correction_norm": float(np.linalg.norm(x2)),
        "alpha": best_alpha,
        "final_residual": float(np.linalg.norm(A @ spectrum - b)),
    }

    return spectrum, info


def unfold_iterative_refinement(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    first_pass_kwargs: Optional[Dict[str, Any]] = None,
    second_pass_kwargs: Optional[Dict[str, Any]] = None,
    alpha: Optional[float] = None,
    max_alpha_search: int = 20,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using iterative refinement.

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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess.
    first_pass_kwargs : dict, optional
        Keyword arguments for first-pass solver.
    second_pass_kwargs : dict, optional
        Keyword arguments for second-pass solver.
    alpha : float, optional
        Blending factor (None = auto-select).
    max_alpha_search : int, optional
        Number of alpha candidates for line search (default: 20).
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
    if random_state is not None:
        np.random.seed(random_state)

    A, b, _ = _build_system(readings, detector_names, sensitivities)

    x0_default = np.ones(n_energy_bins) * 0.5
    x0 = initial_spectrum if initial_spectrum is not None else x0_default

    from .dose_calculation import calculate_dose_rates

    spectrum, info = solve_iterative_refinement(
        A, b, x0,
        first_pass_kwargs=first_pass_kwargs,
        second_pass_kwargs=second_pass_kwargs,
        alpha=alpha,
        max_alpha_search=max_alpha_search,
    )

    computed_readings = A @ spectrum
    residual = b - computed_readings
    doserates = calculate_dose_rates(spectrum, cc_icrp116)

    result = {
        "energy": E_MeV.copy(),
        "spectrum": spectrum.copy(),
        "spectrum_absolute": spectrum.copy(),
        "effective_readings": {
            name: float(val)
            for name, val in zip(
                [n for n in detector_names if n in readings],
                computed_readings,
            )
        },
        "residual": residual.copy(),
        "residual_norm": float(np.linalg.norm(residual)),
        "method": "IterativeRefinement",
        "doserates": doserates,
        "iterations": 0,
        "parameters": {
            "first_pass_kwargs": first_pass_kwargs,
            "second_pass_kwargs": second_pass_kwargs,
            "alpha": info["alpha"],
            **info,
        },
    }

    # Monte-Carlo uncertainty
    if calculate_errors:
        rng = np.random.default_rng(random_state)
        spectra_mc = []
        for _ in range(n_montecarlo):
            b_pert = b * (1.0 + noise_level * rng.standard_normal(len(b)))
            try:
                x_mc, _ = solve_iterative_refinement(
                    A, np.maximum(b_pert, 0), x0,
                    first_pass_kwargs=first_pass_kwargs,
                    second_pass_kwargs=second_pass_kwargs,
                    alpha=alpha,
                    max_alpha_search=max_alpha_search,
                )
                spectra_mc.append(x_mc)
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.debug(
                    "Monte-Carlo sample failed (%s): %s",
                    exc.__class__.__name__, exc,
                )
                continue
        if spectra_mc:
            result["spectrum_uncertainty"] = np.std(spectra_mc, axis=0)
            result["calculate_errors"] = True
            result["n_montecarlo"] = len(spectra_mc)

    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
