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

from collections.abc import Callable
from typing import Any

import numpy as np

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
        x0 = kwargs.pop("x0", None)
        return solve_func(A, b, x0=x0, **kwargs, **fixed_params)

    wrapper.__name__ = f"{solve_func.__name__}_wrapper"
    return wrapper


def run_unfolding(
    *,
    # Detector instance data (passed from self)
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback: Callable[[dict[str, Any]], str],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
    # User-provided inputs
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None,
    default_initial: np.ndarray,
    # Core solver
    solve_func: Callable[..., np.ndarray],
    solve_kwargs: dict[str, Any],
    # Method metadata
    method_name: str,
    extra_output: dict[str, Any] | None = None,
    # Monte-Carlo options
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    random_state: int | None = None,
    # Result saving
    save_result: bool = False,
) -> dict[str, Any]:
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
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths ``d(ln E)_i`` of the energy grid.
        When provided, dose integration uses them instead of a scalar step
        (correct for non-uniform grids).
    reading_uncertainties : dict or np.ndarray, optional
        Absolute 1-sigma uncertainty per reading for the Monte-Carlo
        uncertainty propagation (takes precedence over the relative
        ``noise_level``). Requires ``calculate_errors=True``.
    reading_covariance : np.ndarray, optional
        Absolute covariance matrix between the readings for the
        Monte-Carlo uncertainty propagation (highest precedence over
        ``reading_uncertainties`` and ``noise_level``). Requires
        ``calculate_errors=True``.
    noise_model : str, optional
        Statistical noise model for the Monte-Carlo propagation:
        ``'gaussian'`` (default, relative scalar noise) or ``'poisson'``
        (counting statistics; readings are resampled as Poisson-distributed
        counts, see ``measurement_time``). Requires ``calculate_errors=True``.
    measurement_time : float, optional
        Counting time ``T`` for ``noise_model='poisson'`` when the readings
        are rates; expected counts are ``b_i * T`` and samples are
        rescaled back to rates.
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
    variance_reduction : str, optional
        Monte-Carlo variance reduction technique: ``'none'`` (default),
        ``'antithetic'``, ``'control'`` or ``'both'``.
    random_state : int, optional
        Random seed for reproducibility.
    save_result : bool, optional
        If True, save result to history.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary.
    """
    # 0. Validate inputs before building the system
    if not isinstance(readings, dict) or len(readings) == 0:
        if not isinstance(readings, dict):
            kind = type(readings).__name__
        else:
            kind = "empty dict"
        raise ValueError(f"readings must be a non-empty dict, got {kind}")
    if not isinstance(detector_names, list) or len(detector_names) == 0:
        if not isinstance(detector_names, list):
            kind = type(detector_names).__name__
        else:
            kind = "empty list"
        raise ValueError(f"detector_names must be a non-empty list, got {kind}")
    if not isinstance(n_energy_bins, (int, np.integer)) or n_energy_bins <= 0:
        raise ValueError(
            f"n_energy_bins must be a positive integer, got {n_energy_bins!r}"
        )
    E_MeV_arr = np.asarray(E_MeV)
    if len(E_MeV_arr) != n_energy_bins:
        raise ValueError(
            f"Length of E_MeV ({len(E_MeV_arr)}) must match "
            f"n_energy_bins ({n_energy_bins})"
        )
    if not isinstance(noise_level, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"noise_level must be a number, got {type(noise_level).__name__}"
        )
    noise_level_f = float(noise_level)
    if noise_level_f <= 0 or noise_level_f > 1:
        raise ValueError(f"noise_level must be in (0, 1] range, got {noise_level_f}")
    if not isinstance(n_montecarlo, (int, np.integer)) or n_montecarlo < 0:
        raise ValueError(
            f"n_montecarlo must be a non-negative integer, got {n_montecarlo!r}"
        )

    # 1. Build system
    A, b, selected = _build_system(readings, detector_names, sensitivities)

    # 2. Normalize initial spectrum
    x0 = _normalize_initial(initial_spectrum, default_initial, n_energy_bins)

    # 3. Solve (solve_func may return spectrum or (spectrum, iterations, converged))
    solve_kwargs_with_x0 = {**solve_kwargs, "x0": x0}
    solve_result = solve_func(A, b, **solve_kwargs_with_x0)

    # Handle both single return value and tuple returns
    extra_meta = {}
    if isinstance(solve_result, tuple):
        spectrum = solve_result[0]
        # Extract additional metadata from tuple
        if len(solve_result) >= 2:
            extra_meta["iterations"] = int(solve_result[1])
        if len(solve_result) >= 3:
            extra_meta["converged"] = bool(solve_result[2])
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
        ln_steps=ln_steps,
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
            variance_reduction=variance_reduction,
            response_matrix=A,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
            noise_model=noise_model,
            measurement_time=measurement_time,
        )

    # 6. Save result
    if save_result:
        save_result_callback(output)

    return output


def _build_system(
    readings: dict[str, float],
    detector_names: list[str],
    sensitivities: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build response matrix A and measurement vector b from readings."""
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)
    return A, b, selected


def _normalize_initial(
    initial_spectrum: np.ndarray | None,
    default_initial: np.ndarray,
    n_energy_bins: int,
) -> np.ndarray:
    """Return normalized initial spectrum or default.

    Raises
    ------
    ValueError
        If the provided initial spectrum length does not match the number
        of energy bins (or if it is not one-dimensional).
    """
    if initial_spectrum is not None:
        if isinstance(initial_spectrum, dict):
            initial_spectrum = initial_spectrum.get("spectrum", None)
            if initial_spectrum is None:
                return default_initial.copy()
        spectrum = np.asarray(initial_spectrum, dtype=float)
        if spectrum.ndim != 1 or len(spectrum) != n_energy_bins:
            raise ValueError(
                f"Initial spectrum length ({len(spectrum)}) must match "
                f"number of energy bins ({n_energy_bins})"
            )
        return np.maximum(spectrum, 0)
    return default_initial.copy()


def _standardize_output(
    spectrum: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    selected: list[str],
    cc_icrp116: dict[str, np.ndarray],
    method: str,
    extra: dict[str, Any] | None = None,
    ln_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create standardized output dictionary."""
    from .dose_calculation import (
        INTEGRATION_RULE,
        SPECTRUM_DEFINITION,
        SPECTRUM_UNITS,
        calculate_dose_rates,
        energy_bin_edges,
    )

    spectrum_nonneg = np.maximum(spectrum, 0)
    computed_readings = A @ spectrum_nonneg
    residual = b - computed_readings

    if ln_steps is not None:
        doserates = calculate_dose_rates(
            spectrum_nonneg, cc_icrp116, dlnE_array=ln_steps
        )
        out_mask = cc_icrp116.get("_out_of_range_mask")
        if out_mask is not None:
            mask_arr = np.asarray(out_mask, dtype=bool)
            total = float(np.sum(spectrum_nonneg))
            dose_coverage = (
                float(np.sum(spectrum_nonneg[~mask_arr])) / total
                if total > 0
                else 1.0
            )
        else:
            dose_coverage = 1.0
    else:
        doserates = calculate_dose_rates(spectrum_nonneg, cc_icrp116)
        dose_coverage = 1.0

    output = {
        "energy": E_MeV.copy(),
        "spectrum": spectrum_nonneg.copy(),
        "spectrum_absolute": spectrum_nonneg.copy(),
        "spectrum_definition": SPECTRUM_DEFINITION,
        "spectrum_units": SPECTRUM_UNITS,
        "energy_bin_edges_MeV": energy_bin_edges(E_MeV),
        "integration_rule": INTEGRATION_RULE,
        "effective_readings": {
            name: float(val) for name, val in zip(selected, computed_readings)
        },
        "residual": residual.copy(),
        "residual_norm": float(np.linalg.norm(residual)),
        "method": method,
        "doserates": doserates,
        "dose_coverage_fraction": float(dose_coverage),
    }

    if extra:
        output.update(extra)

    return output


def _add_montecarlo_uncertainty(
    output: dict[str, Any],
    solve_func: Callable,
    readings: dict[str, float],
    noise_level: float,
    n_montecarlo: int,
    n_energy_bins: int,
    random_state: int | None,
    solve_kwargs: dict[str, Any],
    detector_names: list[str],
    sensitivities: dict[str, np.ndarray],
    x0: np.ndarray,
    variance_reduction: str = "none",
    response_matrix: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> None:
    """Run Monte-Carlo uncertainty and update output dict in-place."""
    logger.info(f"Calculating uncertainty with {n_montecarlo} Monte-Carlo samples...")

    def _mc_solver(noisy_readings: dict[str, float], **kwargs) -> np.ndarray:
        A_noisy, b_noisy, _ = _build_system(
            noisy_readings,
            kwargs["detector_names"],
            kwargs["sensitivities"],
        )
        # Remove extra keys not meant for the solver
        solver_kw = {
            k: v
            for k, v in kwargs.items()
            if k not in ("detector_names", "sensitivities")
        }
        # Add x0 to solver kwargs
        solver_kw["x0"] = x0
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
        variance_reduction=variance_reduction,
        response_matrix=response_matrix,
        reading_uncertainties=reading_uncertainties,
        reading_covariance=reading_covariance,
        noise_model=noise_model,
        measurement_time=measurement_time,
        **mc_kwargs,
    )

    output.update(mc_result)
    output["montecarlo_samples"] = n_montecarlo
    output["noise_level"] = noise_level
    logger.info("...uncertainty calculation completed.")
