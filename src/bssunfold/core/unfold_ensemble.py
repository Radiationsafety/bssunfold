"""Ensemble unfolding method for robust neutron spectrum reconstruction.

Combines results from multiple base unfolding methods to reduce variance
and improve robustness against method-specific failures.  Different
unfolding algorithms have different biases; the ensemble exploits this
diversity so that no single failure mode dominates.

Supported combination strategies:
- ``weighted_average``: convex combination of spectra weighted by inverse
  residual or user-supplied weights.
- ``median``: element-wise median across all solutions.
- ``trimmed_mean``: element-wise mean after discarding extreme values.
- ``best_residual``: selects the single solution with the smallest data
  misfit.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..logging_config import get_logger
from ._base_unfolder import _build_system

__all__ = ["solve_ensemble", "unfold_ensemble"]

logger = get_logger("unfold_ensemble")


# Default ensemble members: (display_name, solve_function)
DEFAULT_METHODS: List[Tuple[str, Callable]] = []


def _ensure_default_methods() -> List[Tuple[str, Callable]]:
    """Lazily import default ensemble methods to avoid circular imports."""
    if DEFAULT_METHODS:
        return DEFAULT_METHODS
    from .unfold_bayes import solve_bayes
    from .unfold_cgls import solve_cgls
    from .unfold_gravel import solve_gravel
    from .unfold_landweber import solve_landweber
    from .unfold_mlem import solve_mlem

    DEFAULT_METHODS.extend(
        [
            ("MLEM", solve_mlem),
            ("Bayes", solve_bayes),
            ("Landweber", solve_landweber),
            ("CGLS", solve_cgls),
            ("GRAVEL", solve_gravel),
        ]
    )
    return DEFAULT_METHODS


def _compute_weights_from_residuals(
    spectra: List[np.ndarray],
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Compute inverse-residual weights for ensemble averaging.

    Weight of method *i* is ``1 / ||A x_i - b||``.  Weights are then
    normalised to sum to 1.
    """
    weights = np.zeros(len(spectra))
    for i, x in enumerate(spectra):
        res = np.linalg.norm(A @ x - b)
        weights[i] = 1.0 / (res + 1e-30)
    weights /= weights.sum()
    return weights


def solve_ensemble(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    methods: Optional[List[Tuple[Callable, Dict[str, Any]]]] = None,
    weights: Optional[np.ndarray] = None,
    combination: str = "weighted_average",
    trim_fraction: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve unfolding problem using an ensemble of base methods.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (n,).  Passed to methods that require it; for
        methods that ignore it (e.g. FERDOR, StatReg) it is unused.
    methods : list of (callable, dict), optional
        Each element is ``(solver_func, kwargs_dict)`` where
        ``solver_func(A, b, x0, **kwargs)`` returns a 1-D spectrum array.
        If *None*, the default ensemble of MLEM, Bayes, Landweber, CGLS
        and GRAVEL is used with conservative default parameters.
    weights : np.ndarray, optional
        Per-method weights for ``combination='weighted_average'``.
        If *None*, weights are derived from inverse residuals.
    combination : str, optional
        Combination strategy: ``'weighted_average'``, ``'median'``,
        ``'trimmed_mean'`` or ``'best_residual'`` (default:
        ``'weighted_average'``).
    trim_fraction : float, optional
        Fraction of extreme values to discard per bin for
        ``'trimmed_mean'`` (default: 0.2).

    Returns
    -------
    tuple
        ``(spectrum, info)`` where *info* contains per-method results
        and diagnostics.
    """
    n = A.shape[1]
    if x0 is None:
        x0 = np.ones(n) * 0.5

    if methods is None:
        defaults = _ensure_default_methods()
        methods = [
            (fn, {"max_iterations": 200, "tolerance": 1e-4}) for _, fn in defaults
        ]

    valid_combinations = ("weighted_average", "median", "trimmed_mean", "best_residual")
    if combination not in valid_combinations:
        raise ValueError(
            f"Unknown combination '{combination}'. Choose from {valid_combinations}"
        )

    spectra: List[np.ndarray] = []
    residuals: List[float] = []
    names: List[str] = []

    for idx, (solver, kwargs) in enumerate(methods):
        name = kwargs.get("_name", f"method_{idx}")
        try:
            result = solver(
                A, b, x0, **{k: v for k, v in kwargs.items() if not k.startswith("_")}
            )
            # Handle both (x,) and (x, iters, conv) return types
            if isinstance(result, tuple):
                x_sol = result[0]
            else:
                x_sol = np.asarray(result)
            x_sol = np.maximum(np.asarray(x_sol, dtype=float).ravel(), 0)
            spectra.append(x_sol)
            residuals.append(float(np.linalg.norm(A @ x_sol - b)))
            names.append(name)
        except Exception as exc:
            logger.warning("Ensemble method %s failed: %s", name, exc)

    if not spectra:
        raise RuntimeError("All ensemble methods failed")

    spectra_arr = np.array(spectra)  # (n_methods, n_bins)

    # --- Combination strategies ---
    if combination == "best_residual":
        best_idx = int(np.argmin(residuals))
        spectrum = spectra_arr[best_idx]
        info_str = f"best={names[best_idx]} (res={residuals[best_idx]:.4e})"
    elif combination == "median":
        spectrum = np.median(spectra_arr, axis=0)
        info_str = f"median of {len(spectra)} methods"
    elif combination == "trimmed_mean":
        k = max(1, int(trim_fraction * len(spectra)))
        spectrum = np.mean(
            np.sort(spectra_arr, axis=0)[k:-k] if k < len(spectra) else spectra_arr,
            axis=0,
        )
        info_str = f"trimmed_mean (trim={trim_fraction}) of {len(spectra)} methods"
    else:  # weighted_average
        if weights is None:
            weights = _compute_weights_from_residuals(spectra, A, b)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        spectrum = weights @ spectra_arr
        info_str = f"weighted_average of {len(spectra)} methods"

    info = {
        "combination": combination,
        "n_methods": len(spectra),
        "method_names": names,
        "residuals": residuals,
        "weights": weights.tolist() if weights is not None else None,
        "info_str": info_str,
    }

    return spectrum, info


def unfold_ensemble(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    methods: Optional[List[Tuple[Callable, Dict[str, Any]]]] = None,
    weights: Optional[np.ndarray] = None,
    combination: str = "weighted_average",
    trim_fraction: float = 0.2,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using ensemble method.

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
    methods : list of (callable, dict), optional
        Solver functions and their keyword arguments.
    weights : np.ndarray, optional
        Per-method weights for weighted average.
    combination : str, optional
        Combination strategy (default: ``'weighted_average'``).
    trim_fraction : float, optional
        Trim fraction for trimmed mean (default: 0.2).
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

    spectrum, info = solve_ensemble(
        A,
        b,
        x0,
        methods=methods,
        weights=weights,
        combination=combination,
        trim_fraction=trim_fraction,
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
        "method": "Ensemble",
        "doserates": doserates,
        "iterations": 0,
        "parameters": {
            "combination": combination,
            "n_methods": info["n_methods"],
            "method_names": info["method_names"],
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
                x_mc, _ = solve_ensemble(
                    A,
                    np.maximum(b_pert, 0),
                    x0,
                    methods=methods,
                    weights=weights,
                    combination=combination,
                    trim_fraction=trim_fraction,
                )
                spectra_mc.append(x_mc)
            except Exception:
                continue
        if spectra_mc:
            result["spectrum_uncertainty"] = np.std(spectra_mc, axis=0)
            result["calculate_errors"] = True
            result["n_montecarlo"] = len(spectra_mc)

    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
