"""Per-bin composite (chimera) unfolding method.**Concept**Instead of selecting one best method globally or averaging multiple methods,
this approach selects the **best method independently for each energy bin**,
as determined by calibration against a large set of reference spectra (251 IAEA
Compendium spectra + 20 Monte Carlo reference spectra).  The resulting
spectrum is a *chimera* where different energy bins may originate from
different unfolding methods, each chosen for its historically superior
accuracy in that specific energy range.**Calibration**The per-bin method map is built by :func:`build_bin_method_map`, which:1. Takes a set of reference spectra with known ground truth,
2. Runs every candidate unfolding method on each spectrum,
3. Computes the per-bin relative absolute error,
4. Averages across all reference spectra,
5. For each bin, selects the method with the lowest mean relative error.**Smoothing**To avoid discontinuities at bin boundaries where the selected method
changes, an optional Gaussian smoothing pass is applied.  This preserves
the overall spectral shape while eliminating artificial jumps.
References
----------
- IAEA Compendium of Neutron Spectra and Detector Responses (2001)
- Method performance analysis from bssunfold validation suite.
"""

from __future__ import annotations

import os
import signal
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..logging_config import get_logger
from ..utils.validators import validate_system

logger = get_logger("composite_per_bin")

__all__ = [
    "solve_composite_per_bin",
    "unfold_composite_per_bin",
    "build_bin_method_map",
    "load_bin_method_map",
    "save_bin_method_map",
    "DEFAULT_CANDIDATE_METHODS",
    "DEFAULT_BIN_METHOD_MAP",
]


# ============================================================================
# Default candidate methods (name -> (solve_func, default_kwargs))
# ============================================================================
DEFAULT_CANDIDATE_METHODS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "scipy_direct": ("unfold_scipy_direct_method", {}),
    "cvxpy": ("unfold_cvxpy", {}),
    "qpsolvers": ("unfold_qpsolvers", {}),
    "lmfit": ("unfold_lmfit", {"max_iterations": 200}),
    "landweber": ("unfold_landweber", {"max_iterations": 50}),
    "tsvd": ("unfold_tsvd", {}),
    "kaczmarz": ("unfold_kaczmarz", {"max_iterations": 50}),
    "statreg": ("unfold_statreg", {}),
    "bayes_spline": ("unfold_bayes_spline_regularization", {}),
    "mlem": ("unfold_mlem", {"max_iterations": 200}),
    "gravel": ("unfold_gravel", {"max_iterations": 50}),
    "bayes": ("unfold_bayes", {"max_iterations": 200}),
    "cgls": ("unfold_cgls", {"max_iterations": 50}),
    "tikhonov_legendre": ("unfold_tikhonov_legendre", {}),
    "doroshenko": ("unfold_doroshenko", {"max_iterations": 50}),
}


# ============================================================================
# Pre-computed default map (60 bins, GSF detector set).
# Calibrated on 251 IAEA Compendium + 20 MC reference spectra (271 total),
# with 5 x 10% noise realizations, weighted MAE metric.
# Result: qpsolvers is the best method for all 60 energy bins on GSF.
# Users SHOULD recalibrate for other detector sets using build_bin_method_map().
# ============================================================================
DEFAULT_BIN_METHOD_MAP: List[str] = [
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
    "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers", "qpsolvers",
]


# ============================================================================
# Timeout helper
# ============================================================================

class _MethodTimeout(Exception):
    """Raised when an individual method times out."""


def _timeout_handler(signum, frame):
    raise _MethodTimeout()


def _run_with_timeout(fn: Callable, timeout: float) -> Any:
    """Execute *fn* with a wall-clock timeout (Unix SIGALRM)."""
    if timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(np.ceil(timeout)))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ============================================================================
# Core solver
# ============================================================================


def solve_composite_per_bin(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    methods: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None,
    bin_method_map: Optional[List[str]] = None,
    smooth_sigma: float = 0.0,
    timeout_per_method: float = 30.0,
    fallback_combination: str = "weighted_average",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve unfolding by selecting the best method **per energy bin**.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (n,).
    methods : dict, optional
        ``{name: (solve_func, kwargs_dict)}``.  If *None*, the default
        candidate set from :data:`DEFAULT_CANDIDATE_METHODS` is used
        (lazy-loaded).
    bin_method_map : list of str, optional
        ``[method_for_bin_0, method_for_bin_1, ...]`` with length *n*.
        If *None*, :data:`DEFAULT_BIN_METHOD_MAP` is used (must be
        pre-populated, e.g. via :func:`build_bin_method_map`).
    smooth_sigma : float, optional
        Gaussian smoothing width (in bins) to reduce discontinuities at
        method-switch boundaries.  0.0 = no smoothing (default).
    timeout_per_method : float, optional
        Wall-clock timeout per individual method (seconds, default 30).
    fallback_combination : str, optional
        If no ``bin_method_map`` is available and no default is set,
        fall back to a global combination strategy: ``'weighted_average'``,
        ``'median'``, ``'trimmed_mean'`` or ``'best_residual'``.

    Returns
    -------
    tuple
        ``(spectrum, info)`` where *info* contains diagnostics.
    """
    A, b, x0 = validate_system(A, b, x0=x0)
    n_bins = A.shape[1]

    if x0 is None:
        x0 = np.ones(n_bins) * 0.5

    # ------------------------------------------------------------------
    # Resolve methods
    # ------------------------------------------------------------------
    if methods is None:
        methods = _get_default_solve_methods()

    if not methods:
        raise ValueError(
            "No candidate methods available. Provide 'methods' or ensure "
            "DEFAULT_CANDIDATE_METHODS can be resolved."
        )

    # ------------------------------------------------------------------
    # Run all candidate methods
    # ------------------------------------------------------------------
    spectra: Dict[str, np.ndarray] = {}
    residuals: Dict[str, float] = {}
    failures: Dict[str, str] = {}

    for name, (solver, kwargs) in methods.items():
        try:
            result = _run_with_timeout(
                partial(solver, A, b, x0=x0, **kwargs),
                timeout_per_method,
            )
            if isinstance(result, tuple):
                x_sol = np.asarray(result[0], dtype=float).ravel()
            else:
                x_sol = np.asarray(result, dtype=float).ravel()
            x_sol = np.maximum(x_sol, 0.0)
            # Reject degenerate results
            if np.any(np.isnan(x_sol)) or np.sum(x_sol) <= 0:
                failures[name] = "degenerate output"
                continue
            spectra[name] = x_sol
            residuals[name] = float(np.linalg.norm(A @ x_sol - b))
        except Exception as exc:  # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
            logger.debug("Per-bin composite: method %s failed: %s", name, exc)

    if not spectra:
        raise RuntimeError(
            f"All candidate methods failed: {failures}"
        )

    # ------------------------------------------------------------------
    # Select per-bin values
    # ------------------------------------------------------------------
    method_names = list(spectra.keys())
    stacked = np.array([spectra[m] for m in method_names])  # (n_methods, n_bins)

    use_bin_map = (
        bin_method_map is not None
        and len(bin_method_map) == n_bins
    ) or (
        bin_method_map is None
        and DEFAULT_BIN_METHOD_MAP
        and len(DEFAULT_BIN_METHOD_MAP) == n_bins
    )

    if use_bin_map:
        bmap = bin_method_map if bin_method_map is not None else DEFAULT_BIN_METHOD_MAP
        # Build method->index lookup
        name_to_idx = {name: i for i, name in enumerate(method_names)}

        composite = np.zeros(n_bins)
        bin_source = [""] * n_bins
        n_from_map = 0
        n_from_fallback = 0

        for j in range(n_bins):
            best_name = bmap[j] if j < len(bmap) else ""
            if best_name in name_to_idx:
                composite[j] = stacked[name_to_idx[best_name], j]
                bin_source[j] = best_name
                n_from_map += 1
            else:
                # Fallback: use method with smallest residual for this bin
                # Weighted by inverse residual
                best_idx = _best_bin_index(stacked[:, j], method_names, residuals)
                composite[j] = stacked[best_idx, j]
                bin_source[j] = method_names[best_idx]
                n_from_fallback += 1

        selection_mode = "per_bin_map"
    else:
        # No bin map available — fall back to global combination
        composite, selection_mode = _global_fallback(
            stacked, method_names, residuals, A, b, fallback_combination
        )
        bin_source = [selection_mode] * n_bins
        n_from_map = 0
        n_from_fallback = n_bins

    # ------------------------------------------------------------------
    # Optional Gaussian smoothing
    # ------------------------------------------------------------------
    if smooth_sigma > 0:
        composite = _gaussian_smooth(composite, smooth_sigma)

    info = {
        "selection_mode": selection_mode,
        "n_methods": len(spectra),
        "method_names": method_names,
        "residuals": {m: v for m, v in residuals.items()},
        "failures": failures,
        "bin_source": bin_source,
        "n_from_map": n_from_map,
        "n_from_fallback": n_from_fallback,
        "smooth_sigma": smooth_sigma,
        "composite_residual": float(np.linalg.norm(A @ composite - b)),
    }

    return composite, info


# ============================================================================
# High-level wrapper
# ============================================================================


def unfold_composite_per_bin(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Callable,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    methods: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None,
    bin_method_map: Optional[List[str]] = None,
    smooth_sigma: float = 0.0,
    timeout_per_method: float = 30.0,
    fallback_combination: str = "weighted_average",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using per-bin best-method composite.

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
    methods : dict, optional
        ``{name: (solve_func, kwargs)}`` candidate methods.
    bin_method_map : list of str, optional
        Pre-computed per-bin method selection.
    smooth_sigma : float, optional
        Gaussian smoothing width (default 0 = off).
    timeout_per_method : float, optional
        Per-method timeout in seconds (default 30).
    fallback_combination : str, optional
        Fallback if no bin map is available.
    calculate_errors : bool, optional
        Monte-Carlo uncertainty estimation.
    noise_level : float, optional
        Relative noise for MC (default 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default 100).
    save_result : bool, optional
        Save to history.
    random_state : int, optional
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary.
    """
    from ._base_unfolder import _build_system
    from .dose_calculation import calculate_dose_rates

    if random_state is not None:
        np.random.seed(random_state)

    A, b, _ = _build_system(readings, detector_names, sensitivities)

    x0_default = np.ones(n_energy_bins) * 0.5
    x0 = initial_spectrum if initial_spectrum is not None else x0_default

    spectrum, info = solve_composite_per_bin(
        A, b, x0=x0,
        methods=methods,
        bin_method_map=bin_method_map,
        smooth_sigma=smooth_sigma,
        timeout_per_method=timeout_per_method,
        fallback_combination=fallback_combination,
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
        "method": "CompositePerBin",
        "doserates": doserates,
        "iterations": 0,
        "parameters": {
            "selection_mode": info["selection_mode"],
            "n_methods": info["n_methods"],
            "method_names": info["method_names"],
            "smooth_sigma": info["smooth_sigma"],
            "n_from_map": info["n_from_map"],
            "n_from_fallback": info["n_from_fallback"],
            "bin_source": info["bin_source"],
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
                x_mc, _ = solve_composite_per_bin(
                    A, np.maximum(b_pert, 0), x0,
                    methods=methods,
                    bin_method_map=bin_method_map,
                    smooth_sigma=smooth_sigma,
                    timeout_per_method=timeout_per_method,
                    fallback_combination=fallback_combination,
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


# ============================================================================
# Bin-method map construction (calibration)
# ============================================================================


def build_bin_method_map(
    detector: object,
    reference_spectra: Union["pd.DataFrame", Dict],
    candidate_methods: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
    metric: str = "mae",
    progress: bool = True,
    timeout_per_method: float = 60.0,
) -> List[str]:
    """Calibrate the per-bin best-method map using reference spectra.

    For each reference spectrum the effective readings are computed, each
    candidate method is run, and the per-bin absolute (or relative) error is
    accumulated.  The method with the lowest mean per-bin error is selected
    for each energy bin.

    Parameters
    ----------
    detector : Detector
        A configured :class:`bssunfold.Detector` instance.
    reference_spectra : DataFrame or dict
        Reference spectra.  As a DataFrame: an ``E_MeV`` column plus one
        column per spectrum.  As a dict: ``{name: {"E_MeV": ndarray,
        "Phi": ndarray}}``.
    candidate_methods : dict, optional
        ``{name: (unfold_method_name, kwargs)}``.  If *None*, uses
        :data:`DEFAULT_CANDIDATE_METHODS`.
    metric : str, optional
        Per-bin error metric: ``'mae'`` (mean absolute error),
        ``'mape'`` (mean absolute percentage error), or ``'mse'``
        (mean squared error).  Default: ``'mae'``.
    progress : bool, optional
        Print progress (default True).
    timeout_per_method : float, optional
        Timeout per method per spectrum (default 60s).

    Returns
    -------
    list of str
        ``[best_method_for_bin_0, ..., best_method_for_bin_n]``.
    """
    import pandas as pd

    if candidate_methods is None:
        candidate_methods = DEFAULT_CANDIDATE_METHODS

    # Resolve references
    refs = _as_reference_dict(reference_spectra)
    n_refs = len(refs)
    if n_refs == 0:
        raise ValueError("No reference spectra provided")

    # Determine n_bins from first reference
    first_ref = next(iter(refs.values()))
    n_bins = len(first_ref["Phi"])
    method_names = list(candidate_methods.keys())
    n_methods = len(method_names)

    if progress:
        print(
            f"Calibrating per-bin method map: "
            f"{n_refs} spectra x {n_methods} methods x {n_bins} bins"
        )

    # Accumulate per-bin errors: (n_methods, n_bins)
    error_accum = np.zeros((n_methods, n_bins))
    count_accum = np.zeros(n_bins, dtype=int)

    for spec_idx, (spec_name, ref) in enumerate(refs.items()):
        ref_phi = ref["Phi"]
        ref_E = ref["E_MeV"]

        # Compute effective readings
        readings = detector.get_effective_readings_for_spectra(ref)

        if progress:
            print(f"  [{spec_idx + 1}/{n_refs}] {spec_name}", end="", flush=True)

        for m_idx, (m_name, (unfold_attr, kwargs)) in enumerate(candidate_methods.items()):
            if not hasattr(detector, unfold_attr):
                if progress:
                    print(f"  [SKIP:{m_name}]", end="", flush=True)
                continue
            func = getattr(detector, unfold_attr)
            try:
                result = _run_with_timeout(
                    partial(func, readings=readings, **kwargs),
                    timeout_per_method,
                )
                spec = result.get("spectrum") if isinstance(result, dict) else None
                if spec is None or np.any(np.isnan(spec)) or np.sum(spec) <= 0:
                    continue
                spec = np.asarray(spec, dtype=float)

                # Per-bin error
                if metric == "mape":
                    err = np.abs(spec - ref_phi) / (np.abs(ref_phi) + 1e-30)
                elif metric == "mse":
                    err = (spec - ref_phi) ** 2
                else:  # mae
                    err = np.abs(spec - ref_phi)

                # Handle length mismatch (e.g. 62 bins vs 60)
                min_len = min(len(err), n_bins)
                error_accum[m_idx, :min_len] += err[:min_len]
                count_accum[:min_len] += 1

            except Exception:  # noqa: BLE001
                continue

        if progress:
            print()

    # Average and select best method per bin
    valid_mask = count_accum > 0
    mean_errors = np.full((n_methods, n_bins), np.inf)
    mean_errors[:, valid_mask] = error_accum[:, valid_mask] / count_accum[valid_mask]

    best_per_bin = np.argmin(mean_errors, axis=0)  # (n_bins,)
    bin_method_map = [method_names[idx] if valid_mask[j] else "landweber"
                      for j, idx in enumerate(best_per_bin)]

    if progress:
        print("\nPer-bin method selection:")
        for j in range(n_bins):
            print(f"  Bin {j:3d}: {bin_method_map[j]}")
        # Summary
        from collections import Counter
        counts = Counter(bin_method_map)
        print("\nMethod frequency:")
        for m, c in counts.most_common():
            print(f"  {m:25s}: {c:3d} bins")

    return bin_method_map


def _as_reference_dict(
    reference_spectra: Union["pd.DataFrame", Dict],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Normalize reference spectra into ``{name: {"E_MeV":.., "Phi":..}}``."""
    import pandas as pd

    if isinstance(reference_spectra, dict):
        # Already in dict format
        out = {}
        for name, val in reference_spectra.items():
            if isinstance(val, dict) and "Phi" in val and "E_MeV" in val:
                out[name] = {
                    "E_MeV": np.asarray(val["E_MeV"], dtype=float),
                    "Phi": np.asarray(val["Phi"], dtype=float),
                }
            elif isinstance(val, (np.ndarray, list)):
                # Assume it's just Phi, no E_MeV
                out[name] = {"E_MeV": None, "Phi": np.asarray(val, dtype=float)}
        return out

    if isinstance(reference_spectra, pd.DataFrame):
        df = reference_spectra
        if "E_MeV" not in df.columns:
            raise ValueError("Reference DataFrame must have an 'E_MeV' column")
        E = df["E_MeV"].values.astype(float)
        cols = [c for c in df.columns if c != "E_MeV"]
        return {
            name: {"E_MeV": E.copy(), "Phi": df[name].values.astype(float)}
            for name in cols
        }

    raise TypeError(
        f"reference_spectra must be DataFrame or dict, got {type(reference_spectra)}"
    )


# ============================================================================
# Persistence
# ============================================================================


def save_bin_method_map(
    bin_method_map: List[str],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a bin-method map to a JSON file.

    Parameters
    ----------
    bin_method_map : list of str
        Per-bin method names.
    path : str
        Output file path.
    metadata : dict, optional
        Additional metadata (e.g., n_spectra, metric, detector_set).
    """
    import json

    data = {
        "bin_method_map": bin_method_map,
        "n_bins": len(bin_method_map),
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved bin-method map (%d bins) to %s", len(bin_method_map), path)


def load_bin_method_map(path: str) -> List[str]:
    """Load a bin-method map from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file created by :func:`save_bin_method_map`.

    Returns
    -------
    list of str
        Per-bin method names.
    """
    import json

    with open(path) as f:
        data = json.load(f)
    bmap = data["bin_method_map"]
    logger.info(
        "Loaded bin-method map (%d bins) from %s", len(bmap), path
    )
    return bmap


# ============================================================================
# Internal helpers
# ============================================================================


def _get_default_solve_methods() -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
    """Lazily resolve DEFAULT_CANDIDATE_METHODS to (solve_func, kwargs)."""
    return _import_common_solve_functions()


def _import_common_solve_functions() -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
    """Import common solve_* functions directly."""
    methods = {}
    pairs = [
        ("scipy_direct", "unfold_scipy_direct_method", "solve_scipy_direct", {}),
        ("cvxpy", "unfold_cvxpy", "solve_cvxpy", {}),
        ("qpsolvers", "unfold_qpsolvers", "solve_qpsolvers", {}),
        ("landweber", "unfold_landweber", "solve_landweber", {"max_iterations": 50}),
        ("tsvd", "unfold_tsvd", "solve_tsvd", {}),
        ("kaczmarz", "unfold_kaczmarz", "solve_kaczmarz", {"max_iterations": 50}),
        ("mlem", "unfold_mlem", "solve_mlem", {"max_iterations": 200}),
        ("gravel", "unfold_gravel", "solve_gravel", {"max_iterations": 50}),
        ("bayes", "unfold_bayes", "solve_bayes", {"max_iterations": 200}),
        ("cgls", "unfold_cgls", "solve_cgls", {"max_iterations": 50}),
        ("statreg", "unfold_statreg", "solve_statreg", {}),
        ("lmfit", "unfold_lmfit", "solve_lmfit", {"max_iterations": 200}),
        ("bayes_spline", "unfold_bayes_spline_regularization", "solve_bayes_spline", {}),
        ("tikhonov_legendre", "unfold_tikhonov_legendre", "solve_tikhonov_legendre", {}),
        ("doroshenko", "unfold_doroshenko", "solve_doroshenko", {"max_iterations": 50}),
    ]
    for name, mod_name, func_name, kwargs in pairs:
        try:
            mod = __import__(
                f"bssunfold.core.{mod_name}", fromlist=[func_name]
            )
            if hasattr(mod, func_name):
                methods[name] = (getattr(mod, func_name), kwargs)
        except (ImportError, AttributeError):
            continue
    return methods


def _best_bin_index(
    bin_values: np.ndarray,
    method_names: List[str],
    residuals: Dict[str, float],
) -> int:
    """Select the best method index for a single bin.

    Uses inverse-residual weighting to pick the method whose global
    residual is smallest (proxy for overall reliability).
    """
    if len(bin_values) == 1:
        return 0
    # Weight by inverse residual
    weights = np.array([
        1.0 / (residuals.get(m, 1e30) + 1e-30)
        for m in method_names
    ])
    weights /= weights.sum()
    # Weighted median-like selection: pick the value closest to
    # the weighted average
    weighted_mean = weights @ bin_values
    distances = np.abs(bin_values - weighted_mean)
    # Prefer methods with lower residual
    score = distances / (weights + 1e-30)
    return int(np.argmin(score))


def _global_fallback(
    stacked: np.ndarray,
    method_names: List[str],
    residuals: Dict[str, float],
    A: np.ndarray,
    b: np.ndarray,
    combination: str,
) -> Tuple[np.ndarray, str]:
    """Apply a global combination strategy when no bin map is available."""
    if combination == "best_residual":
        best_name = min(residuals, key=residuals.get)
        idx = method_names.index(best_name)
        return stacked[idx].copy(), f"best_residual({best_name})"

    if combination == "median":
        return np.median(stacked, axis=0), "median"

    if combination == "trimmed_mean":
        k = max(1, int(0.2 * len(method_names)))
        sorted_arr = np.sort(stacked, axis=0)
        if k < len(method_names):
            return np.mean(sorted_arr[k:-k], axis=0), "trimmed_mean"
        return np.mean(stacked, axis=0), "mean"

    # Default: weighted_average by inverse residual
    weights = np.array([
        1.0 / (residuals.get(m, 1e30) + 1e-30)
        for m in method_names
    ])
    weights /= weights.sum()
    return weights @ stacked, "weighted_average"


def _gaussian_smooth(spectrum: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to the spectrum.

    Uses a simple convolution with a truncated Gaussian kernel.
    Non-negative values are preserved.
    """
    if sigma <= 0:
        return spectrum.copy()
    radius = int(3 * sigma) + 1
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Convolve (np.convolve mode='same')
    smoothed = np.convolve(spectrum, kernel, mode="same")
    return np.maximum(smoothed, 0.0)
