"""Base unfolding workflow for the Detector class.

This module provides a unified workflow function that eliminates code
duplication across all unfold_* methods in the Detector class. It handles:

1. Reading validation
2. System matrix construction (A, b)
3. Initial spectrum normalization
4. Calling the core solver
5. Output standardization
6. Monte-Carlo uncertainty estimation
7. Result saving
"""

import numpy as np
from typing import Callable, Dict, Optional, Any, List, Tuple

from ..logging_config import get_logger
from ._montecarlo import monte_carlo_uncertainty

logger = get_logger("detector")


def make_solve_wrapper(solve_func, **fixed_params):
    """Create a standard solve_wrapper for unfolding methods.

    The wrapped function accepts (A, b, **kwargs), extracts x0 from kwargs,
    and delegates to solve_func(A, b, x0, **fixed_params).

    Parameters
    ----------
    solve_func : callable
        Core solver function with signature (A, b, x0, **params).
    **fixed_params
        Additional keyword arguments forwarded to solve_func.

    Returns
    -------
    callable
        Wrapper compatible with run_unfolding's solve_func interface.
    """
    def wrapper(A, b, **kwargs):
        x0 = kwargs.pop('x0', None)
        return solve_func(A, b, x0=x0, **fixed_params)
    wrapper.__name__ = f"{solve_func.__name__}_wrapper"
    return wrapper


def run_unfolding(
    *,
    # Detector instance data (passed from self)
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Callable[[Dict[str, Any]], str],
    # User-provided inputs
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray],
    default_initial: np.ndarray,
    # Core solver
    solve_func: Callable[..., np.ndarray],
    solve_kwargs: Dict[str, Any],
    # Method metadata
    method_name: str,
    extra_output: Optional[Dict[str, Any]] = None,
    # Monte-Carlo options
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    random_state: Optional[int] = None,
    # Result saving
    save_result: bool = False,
) -> Dict[str, Any]:
    """Run a complete unfolding workflow with unified logic.

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
    save_result_callback : Callable[[Dict[str, Any]], str]
        Callback to save result to history (e.g., Detector._save_result).
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray]
        Initial spectrum guess. If None, uses default_initial.
    default_initial : np.ndarray
        Default initial spectrum when initial_spectrum is None.
    solve_func : Callable[..., np.ndarray]
        Core solver function (e.g., solve_landweber, solve_mlem).
        Must accept (A, b, **solve_kwargs) and return a spectrum array.
    solve_kwargs : Dict[str, Any]
        Keyword arguments for solve_func.
    method_name : str
        Name of the unfolding method for output metadata.
    extra_output : Dict[str, Any], optional
        Additional key-value pairs to include in the output.
    calculate_errors : bool, optional
        If True, run Monte-Carlo uncertainty estimation.
    noise_level : float, optional
        Relative noise level for Monte-Carlo.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples.
    random_state : int, optional
        Random seed for reproducibility.
    save_result : bool, optional
        If True, save result to history.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary.
    """
    # 1. Build system
    A, b, selected = _build_system(readings, detector_names, sensitivities)

    # 2. Normalize initial spectrum
    x0 = _normalize_initial(initial_spectrum, default_initial, n_energy_bins)

    # 3. Solve (solve_func may return spectrum or (spectrum, iterations, converged))
    solve_kwargs_with_x0 = {**solve_kwargs, 'x0': x0}
    solve_result = solve_func(A, b, **solve_kwargs_with_x0)

    # Handle both single return value and tuple returns
    extra_meta = {}
    if isinstance(solve_result, tuple):
        spectrum = solve_result[0]
        # Extract additional metadata from tuple
        if len(solve_result) >= 2:
            extra_meta['iterations'] = int(solve_result[1])
        if len(solve_result) >= 3:
            extra_meta['converged'] = bool(solve_result[2])
    else:
        spectrum = solve_result

    # Merge extra_meta with user-provided extra_output
    if extra_output:
        extra_output = {**extra_output, **extra_meta}
    else:
        extra_output = extra_meta if extra_meta else None

    # 4. Standardize output
    output = _standardize_output(
        spectrum=spectrum,
        A=A,
        b=b,
        E_MeV=E_MeV,
        selected=selected,
        cc_icrp116=cc_icrp116,
        method=method_name,
        extra=extra_output,
    )

    # 5. Monte-Carlo uncertainty
    if calculate_errors:
        _add_montecarlo_uncertainty(
            output=output,
            solve_func=solve_func,
            readings=readings,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            n_energy_bins=n_energy_bins,
            random_state=random_state,
            solve_kwargs=solve_kwargs,
            detector_names=detector_names,
            sensitivities=sensitivities,
            x0=x0,
        )

    # 6. Save result
    if save_result:
        save_result_callback(output)

    return output


def _build_system(
    readings: Dict[str, float],
    detector_names: List[str],
    sensitivities: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build response matrix A and measurement vector b from readings."""
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)
    return A, b, selected


def _normalize_initial(
    initial_spectrum: Optional[np.ndarray],
    default_initial: np.ndarray,
    n_energy_bins: int,
) -> np.ndarray:
    """Return normalized initial spectrum or default."""
    if initial_spectrum is not None:
        spectrum = np.asarray(initial_spectrum, dtype=float)
        if len(spectrum) == n_energy_bins:
            return np.maximum(spectrum, 0)
    return default_initial.copy()


def _standardize_output(
    spectrum: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    selected: List[str],
    cc_icrp116: Dict[str, np.ndarray],
    method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create standardized output dictionary."""
    from .dose_calculation import calculate_dose_rates

    spectrum_nonneg = np.maximum(spectrum, 0)
    computed_readings = A @ spectrum_nonneg
    residual = b - computed_readings

    output = {
        "energy": E_MeV.copy(),
        "spectrum": spectrum_nonneg.copy(),
        "spectrum_absolute": spectrum_nonneg.copy(),
        "effective_readings": {
            name: float(val)
            for name, val in zip(selected, computed_readings)
        },
        "residual": residual.copy(),
        "residual_norm": float(np.linalg.norm(residual)),
        "method": method,
        "doserates": calculate_dose_rates(spectrum_nonneg, cc_icrp116),
    }

    if extra:
        output.update(extra)

    return output


def _add_montecarlo_uncertainty(
    output: Dict[str, Any],
    solve_func: Callable,
    readings: Dict[str, float],
    noise_level: float,
    n_montecarlo: int,
    n_energy_bins: int,
    random_state: Optional[int],
    solve_kwargs: Dict[str, Any],
    detector_names: List[str],
    sensitivities: Dict[str, np.ndarray],
    x0: np.ndarray,
) -> None:
    """Run Monte-Carlo uncertainty and update output dict in-place."""
    logger.info(
        f"Calculating uncertainty with {n_montecarlo} Monte-Carlo samples..."
    )

    def _mc_solver(noisy_readings: Dict[str, float], **kwargs) -> np.ndarray:
        A_noisy, b_noisy, _ = _build_system(
            noisy_readings,
            kwargs["detector_names"],
            kwargs["sensitivities"],
        )
        # Remove extra keys not meant for the solver
        solver_kw = {k: v for k, v in kwargs.items()
                     if k not in ("detector_names", "sensitivities")}
        # Add x0 to solver kwargs
        solver_kw['x0'] = x0
        result = solve_func(A_noisy, b_noisy, **solver_kw)
        # Extract spectrum from tuple if needed
        if isinstance(result, tuple):
            return result[0]
        return result

    mc_kwargs = {
        **solve_kwargs,
        "detector_names": detector_names,
        "sensitivities": sensitivities,
    }

    mc_result = monte_carlo_uncertainty(
        func=_mc_solver,
        readings=readings,
        noise_level=noise_level,
        n_samples=n_montecarlo,
        n_energy_bins=n_energy_bins,
        random_state=random_state,
        **mc_kwargs,
    )

    output.update(mc_result)
    output["montecarlo_samples"] = n_montecarlo
    output["noise_level"] = noise_level
    logger.info("...uncertainty calculation completed.")