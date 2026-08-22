"""Composite (ensemble) multi-method spectrum unfolding.

This module implements an adaptive ensemble approach to neutron spectrum
unfolding. It classifies the unknown spectrum by hardness, selects a pool of
suitable individual methods, runs them, and combines their results with
confidence-weighted averaging. The ensemble reduces sensitivity to individual
method failures and delivers consistent performance across diverse spectra.

References:
-----------
- Wolpert, "Stacked generalization" (1992)
- Reginatto et al., "Sequential Bayesian approach for neutron spectrum unfolding"
"""

import numpy as np
from typing import Dict, Optional, Any, List
import signal

from ..logging_config import get_logger
from ..utils.comparison import cosine_similarity

logger = get_logger("composite")


# Mapping from short method name -> Detector.unfold_* wrapper attribute.
METHOD_DISPATCH: Dict[str, str] = {
    "tsvd": "unfold_tsvd",
    "bayes": "unfold_bayes",
    "cvxpy": "unfold_cvxpy",
    "statreg": "unfold_statreg",
    "lanczos": "unfold_lanczos",
    "mlem": "unfold_mlem",
    "landweber": "unfold_landweber",
    "bayes_spline": "unfold_bayes_spline_regularization",
    "gravel": "unfold_gravel",
    "qpsolvers": "unfold_qpsolvers",
    "hybrid_parametric": "unfold_hybrid_parametric",
    "parametric2": "unfold_parametric2",
    "genetic": "unfold_genetic",
    "interpret": "unfold_interpret",
    "maeo_ensemble": "unfold_maeo",
    "mystic": "unfold_mystic",
    "cs": "unfold_cs",
    "scip": "unfold_scip",
    "docplex": "unfold_docplex",
    "epic": "unfold_epic",
    "kaczmarz": "unfold_kaczmarz",
}

# Default method pools per spectrum-hardness bin (matches the documented table).
DEFAULT_BIN_METHODS: Dict[str, List[str]] = {
    "very_soft": ["tsvd", "bayes", "cvxpy", "statreg", "lanczos"],
    "soft": ["mlem", "landweber", "bayes_spline", "gravel", "qpsolvers"],
    "intermediate": ["cvxpy", "qpsolvers", "hybrid_parametric", "parametric2"],
    "hard": ["genetic", "interpret", "maeo_ensemble", "mystic", "cs"],
    "very_hard": ["scip", "docplex", "epic", "cs", "interpret"],
}

# A curated, fast and robust pool used when no spectrum is supplied for
# classification (e.g. when only readings are available).
GENERAL_METHODS: List[str] = [
    "tsvd",
    "mlem",
    "cvxpy",
    "qpsolvers",
    "bayes_spline",
]

# Optional base weights per method (1.0 == equal contribution).
DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "tsvd": 1.0,
    "bayes": 1.0,
    "cvxpy": 1.0,
    "statreg": 1.0,
    "lanczos": 1.0,
    "mlem": 1.0,
    "landweber": 1.0,
    "bayes_spline": 1.0,
    "gravel": 1.0,
    "qpsolvers": 1.0,
    "hybrid_parametric": 1.0,
    "parametric2": 1.0,
    "genetic": 0.8,
    "interpret": 0.8,
    "maeo_ensemble": 0.8,
    "mystic": 0.8,
    "cs": 0.8,
    "scip": 0.8,
    "docplex": 0.8,
    "epic": 0.8,
    "kaczmarz": 1.0,
}


class _MethodTimeout(Exception):
    """Raised internally when an individual ensemble method times out."""


def _timeout_handler(signum, frame):
    raise _MethodTimeout()


def _run_with_timeout(fn: Any, timeout: float) -> Any:
    """Execute ``fn`` with a per-method wall-clock timeout.

    Uses ``SIGALRM`` on Unix. On platforms without ``SIGALRM`` the timeout is
    ignored and the call runs to completion.
    """
    if timeout is None or timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(np.ceil(timeout)))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def compute_spectrum_features(spectrum: np.ndarray, energy: np.ndarray) -> Dict[str, float]:
    """Compute simple discriminating features of a spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values on the energy grid.
    energy : np.ndarray
        Energy grid (MeV).

    Returns
    -------
    dict
        Features including ``hardness_ratio`` (mean energy in MeV),
        ``total_flux`` and ``entropy``.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    energy = np.asarray(energy, dtype=float)
    total = float(np.sum(spectrum)) + 1e-30

    mean_energy = float(np.sum(spectrum * energy) / total)
    # Normalized entropy of the spectral shape.
    p = spectrum / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p + 1e-30)) / np.log(len(spectrum) + 1e-30))
    peak = float(np.max(spectrum)) if len(spectrum) else 0.0

    return {
        "hardness_ratio": mean_energy,
        "total_flux": total,
        "entropy": entropy,
        "peak": peak,
    }


def classify_spectrum_by_hardness(features: Dict[str, float]) -> str:
    """Classify a spectrum into a hardness bin from its features.

    Thresholds are applied to ``features['hardness_ratio']`` (mean energy, MeV):
    very_soft < 0.1, soft 0.1-0.3, intermediate 0.3-0.5, hard 0.5-1.0,
    very_hard > 1.0.
    """
    hr = features.get("hardness_ratio", 0.5)
    if hr < 0.1:
        return "very_soft"
    elif hr < 0.3:
        return "soft"
    elif hr < 0.5:
        return "intermediate"
    elif hr < 1.0:
        return "hard"
    return "very_hard"


def _confidence_weight(spectrum: np.ndarray, others: List[np.ndarray]) -> float:
    """Confidence of a single solution relative to the ensemble mean."""
    if not others:
        return 1.0
    sims = [cosine_similarity(spectrum, o) for o in others]
    sims = [s for s in sims if np.isfinite(s)]
    if not sims:
        return 1.0
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def unfold_composite(
    detector,
    readings: Dict[str, float],
    n_methods: int = 5,
    timeout_per_method: float = 30.0,
    save_result: bool = False,
    spectrum: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    method_names: Optional[List[str]] = None,
    ensemble_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run an adaptive ensemble of unfolding methods and combine results.

    Parameters
    ----------
    detector : Detector
        Configured Bonner-sphere detector.
    readings : Dict[str, float]
        Detector readings.
    n_methods : int
        Maximum number of methods to combine.
    timeout_per_method : float
        Wall-clock timeout per individual method (seconds).
    save_result : bool
        Persist each method's result to the detector history.
    spectrum : np.ndarray, optional
        Reference/estimated spectrum used to select the method pool by
        hardness. If omitted, a general robust pool is used.
    energy : np.ndarray, optional
        Energy grid for ``spectrum``. Defaults to ``detector.E_MeV``.
    method_names : list, optional
        Explicit list of method short names to run (overrides selection).
    ensemble_weights : dict, optional
        Per-method base weights (defaults to ``DEFAULT_ENSEMBLE_WEIGHTS``).

    Returns
    -------
    dict
        Result dictionary with the combined ``spectrum`` and metadata
        (``successful_methods``, ``consistency``, ``weights``,
        ``individual_spectra``, ``status``, ``message``).
    """
    if energy is None:
        energy = detector.E_MeV
    weights = ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS

    # Select the candidate method pool.
    if method_names:
        candidates = list(method_names)
    elif spectrum is not None:
        features = compute_spectrum_features(spectrum, energy)
        bin_name = classify_spectrum_by_hardness(features)
        candidates = list(DEFAULT_BIN_METHODS.get(bin_name, GENERAL_METHODS))
    else:
        candidates = list(GENERAL_METHODS)

    candidates = candidates[:n_methods]

    individual_spectra: Dict[str, np.ndarray] = {}
    successful_methods: List[str] = []
    messages: Dict[str, str] = {}

    for name in candidates:
        func = getattr(detector, METHOD_DISPATCH.get(name, ""), None)
        if func is None:
            messages[name] = "unknown method"
            continue
        try:
            result = _run_with_timeout(
                lambda: func(readings=readings, save_result=save_result),
                timeout_per_method,
            )
            spec = result.get("spectrum") if isinstance(result, dict) else None
            if spec is None or np.any(np.isnan(spec)) or np.sum(spec) <= 0:
                messages[name] = "invalid output"
                continue
            individual_spectra[name] = np.asarray(spec, dtype=float)
            successful_methods.append(name)
        except Exception as e:  # noqa: BLE001 - skip failing methods
            messages[name] = f"{type(e).__name__}: {e}"

    if not individual_spectra:
        return {
            "spectrum": None,
            "successful_methods": [],
            "consistency": 0.0,
            "weights": {},
            "individual_spectra": {},
            "status": "ERROR",
            "message": f"No method succeeded. Details: {messages}",
            "readings": readings,
            "doserates": None,
        }

    names = list(individual_spectra.keys())
    stacked = np.array([individual_spectra[n] for n in names])

    # Confidence-weighted combination.
    combined = np.zeros_like(stacked[0])
    total_weight = 0.0
    used_weights: Dict[str, float] = {}
    for i, name in enumerate(names):
        others = [stacked[j] for j in range(len(names)) if j != i]
        conf = _confidence_weight(stacked[i], others)
        base = float(weights.get(name, 1.0))
        w = base * conf
        combined += w * stacked[i]
        total_weight += w
        used_weights[name] = w
    if total_weight > 0:
        combined /= total_weight

    # Consistency: mean pairwise cosine similarity among individual solutions.
    consistency = 0.0
    n_pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = cosine_similarity(stacked[i], stacked[j])
            if np.isfinite(s):
                consistency += s
                n_pairs += 1
    consistency = consistency / n_pairs if n_pairs else 0.0

    return {
        "spectrum": combined,
        "successful_methods": successful_methods,
        "consistency": float(consistency),
        "weights": used_weights,
        "individual_spectra": individual_spectra,
        "status": "OK",
        "message": (
            f"Combined {len(successful_methods)}/{len(candidates)} methods "
            f"with consistency {consistency:.3f}"
        ),
        "readings": readings,
        "doserates": None,
    }


__all__ = [
    "METHOD_DISPATCH",
    "DEFAULT_BIN_METHODS",
    "GENERAL_METHODS",
    "DEFAULT_ENSEMBLE_WEIGHTS",
    "unfold_composite",
    "compute_spectrum_features",
    "classify_spectrum_by_hardness",
]
