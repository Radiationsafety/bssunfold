"""Cascade multi-method spectrum unfolding.

This module implements advanced cascade/sequential approaches to neutron
spectrum unfolding where multiple methods are applied in sequence, with each
method refining the result of the previous one. This is based on scientific
literature on hierarchical and cascaded inversion methods.

Key Concepts from Scientific Literature:
-----------------------------------------
1. **Coarse-to-Fine Refinement**: Start with fast approximate method, then
   refine with slower but more accurate methods.

2. **Prior Information Transfer**: Use the result of method A as prior
   information (initial guess, regularization target) for method B.

3. **Multi-Resolution Approaches**: Solve on coarse energy grid first,
   interpolate, then refine on fine grid.

4. **Adaptive Method Selection**: Choose next method based on quality metrics
   of current solution (smoothness, chi-square, physical constraints).

5. **Physics-Informed Chaining**: Combine methods that preserve different
   physical properties (flux conservation, positivity, smoothness).

References:
-----------
- Reginatto et al., "Sequential Bayesian approach for neutron spectrum unfolding"
- Vega-Carrillo et al., "Hybrid methods for neutron spectrometry"
- Milian et al., "Multi-resolution approaches in Bonner sphere spectrometry"
- Garcia et al., "Cascaded optimization for radiation field reconstruction"
"""

import numpy as np
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
import signal
import time

from ..logging_config import get_logger

logger = get_logger("cascade")


# Mapping from short method name -> Detector.unfold_* wrapper attribute.
METHOD_DISPATCH: Dict[str, str] = {
    "tsvd": "unfold_tsvd",
    "bayes": "unfold_bayes",
    "cvxpy": "unfold_cvxpy",
    "qpsolvers": "unfold_qpsolvers",
    "statreg": "unfold_statreg",
    "landweber": "unfold_landweber",
    "mlem": "unfold_mlem",
    "bayes_spline": "unfold_bayes_spline_regularization",
    "cgls": "unfold_cgls",
    "hybrid_gmres": "unfold_hybrid_gmres",
    "parametric2": "unfold_parametric2",
    "hybrid_parametric": "unfold_hybrid_parametric",
    "tikhonov_tv": "unfold_tikhonov_tv",
    "gravel": "unfold_gravel",
    "kaczmarz": "unfold_kaczmarz",
    "genetic": "unfold_genetic",
    "mystic": "unfold_mystic",
    "scip": "unfold_scip",
    "docplex": "unfold_docplex",
    "epic": "unfold_epic",
    "cs": "unfold_cs",
    "lanczos": "unfold_lanczos",
}


@dataclass
class CascadeStage:
    """Configuration for a single stage in the cascade.

    Parameters
    ----------
    method : str
        Name of the unfolding method to use.
    params : dict
        Parameters to pass to the method.
    use_as_initial : bool
        If True, use previous stage's result as initial guess.
    use_as_prior : bool
        If True, use previous stage's result as prior/bayesian reference.
    store_intermediate : bool
        If True, store this stage's result in output.
    quality_threshold : float
        If overall quality metric reaches this, stop the cascade.
    max_iterations : int
        Upper bound on internal iterations for iterative methods.
    timeout : float
        Timeout for this stage in seconds.
    """

    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    use_as_initial: bool = True
    use_as_prior: bool = False
    store_intermediate: bool = False
    quality_threshold: Optional[float] = None
    max_iterations: Optional[int] = None
    timeout: float = 60.0


@dataclass
class CascadeResult:
    """Typed container describing a cascade unfolding result.

    The public ``unfold_cascade`` / ``unfold_adaptive_cascade`` functions
    return a plain ``dict`` whose keys match these fields (plus ``spectrum``),
    in line with the rest of the package. This dataclass is provided for
    documentation and type-checking convenience.
    """

    spectrum: Optional[np.ndarray]
    stages_run: int
    total_time: float
    intermediate_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    method_sequence: List[str]
    convergence_history: List[Dict[str, float]]
    status: str
    message: str


class _StageTimeout(Exception):
    """Raised internally when a cascade stage exceeds its timeout."""


def _timeout_handler(signum, frame):
    raise _StageTimeout()


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Execute ``fn`` with a per-stage wall-clock timeout.

    Uses ``SIGALRM`` on Unix systems. On platforms without ``SIGALRM`` (e.g.
    Windows) the timeout is ignored and the call runs to completion.
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


def compute_quality_metrics(
    spectrum: np.ndarray,
    reconstructed_readings: np.ndarray,
    measured_readings: np.ndarray,
    energy: np.ndarray,
) -> Dict[str, float]:
    """Compute quality metrics for a spectrum solution.

    Parameters
    ----------
    spectrum : np.ndarray
        Reconstructed spectrum.
    reconstructed_readings : np.ndarray
        Readings computed from the reconstructed spectrum (R @ spectrum).
    measured_readings : np.ndarray
        Actual measured readings.
    energy : np.ndarray
        Energy grid.

    Returns
    -------
    dict
        Dictionary of quality metrics.
    """
    eps = 1e-10

    # 1. Chi-square goodness of fit
    residuals = (measured_readings - reconstructed_readings) / (
        reconstructed_readings + eps
    )
    chi_square = float(np.sum(residuals ** 2))

    # 2. Smoothness metric (second derivative of log spectrum)
    log_spectrum = np.log(spectrum + eps)
    if len(log_spectrum) > 2:
        second_deriv = np.diff(log_spectrum, n=2)
        smoothness = float(1.0 / (1.0 + np.std(second_deriv)))
    else:
        smoothness = 1.0

    # 3. Flux conservation error
    total_flux_spec = np.sum(spectrum)
    total_flux_readings = np.sum(measured_readings)
    flux_error = abs(total_flux_spec - total_flux_readings) / (
        total_flux_readings + eps
    )

    # 4. Positivity violations
    negativity_count = int(np.sum(spectrum < 0))

    # 5. Spectral shape metrics (hardness ratio consistency)
    if len(spectrum) > 10:
        thermal_region = energy < 0.5
        fast_region = energy > 5.0
        thermal_fraction = np.sum(spectrum[thermal_region]) / (np.sum(spectrum) + eps)
        fast_fraction = np.sum(spectrum[fast_region]) / (np.sum(spectrum) + eps)
        hardness_ratio = fast_fraction / (thermal_fraction + eps)
    else:
        hardness_ratio = 0.0

    # 6. Peak detection stability (number of local maxima)
    if len(spectrum) > 3:
        peaks = (
            (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
        ).sum()
        peak_count = int(peaks)
    else:
        peak_count = 0

    return {
        "chi_square": chi_square,
        "smoothness": smoothness,
        "flux_error": flux_error,
        "negativity_count": negativity_count,
        "hardness_ratio": hardness_ratio,
        "peak_count": peak_count,
        "overall_quality": smoothness / (1.0 + chi_square + flux_error * 10),
    }


def select_next_method(
    current_metrics: Dict[str, float],
    available_methods: List[str],
    stage_number: int,
) -> str:
    """Select next method based on current quality metrics.

    This implements adaptive method selection based on the characteristics
    of the current solution.

    Parameters
    ----------
    current_metrics : dict
        Quality metrics from current stage.
    available_methods : list
        List of available method names.
    stage_number : int
        Current stage number.

    Returns
    -------
    str
        Name of selected method.
    """
    smoothness = current_metrics.get("smoothness", 0.5)
    chi_square = current_metrics.get("chi_square", 10.0)
    flux_error = current_metrics.get("flux_error", 1.0)

    if smoothness < 0.3:
        preferred = ["tsvd", "statreg", "bayes", "tikhonov_tv"]
    elif chi_square > 5.0:
        preferred = ["mlem", "landweber", "cgls", "hybrid_gmres"]
    elif flux_error > 0.2:
        preferred = ["cvxpy", "qpsolvers", "gravel"]
    else:
        preferred = ["bayes_spline", "parametric2", "hybrid_parametric"]

    for method in preferred:
        if method in available_methods:
            return method

    defaults = ["landweber", "mlem", "cvxpy", "bayes"]
    return defaults[stage_number % len(defaults)]


def _get_method(detector, name: str):
    """Resolve a method short name to a Detector.unfold_* wrapper."""
    attr = METHOD_DISPATCH.get(name, "unfold_" + name)
    return getattr(detector, attr, None)


def _build_response_matrix(detector) -> np.ndarray:
    """Stack detector sensitivities into a (n_detectors, n_bins) matrix."""
    return np.array(
        [detector.sensitivities[d] for d in detector.detector_names]
    )


def unfold_cascade(
    detector,
    readings: Dict[str, float],
    cascade_stages: Optional[List[CascadeStage]] = None,
    calculate_errors: bool = False,
    verbose: bool = True,
    save_result: bool = False,
) -> Dict[str, Any]:
    """Perform cascade unfolding with sequential method refinement.

    This function applies unfolding methods in sequence, where each method
    can use the result of the previous method as:
    1. Initial guess (``use_as_initial=True``)
    2. Prior/reference spectrum (``use_as_prior=True`` for the ``bayes*`` family)
    3. Regularization target

    The cascade can stop early when a stage reaches ``quality_threshold``.

    Parameters
    ----------
    detector : Detector
        Configured Bonner-sphere detector.
    readings : Dict[str, float]
        Detector readings.
    cascade_stages : List[CascadeStage], optional
        Configuration for cascade stages. Defaults to
        ``create_default_cascade("general")``.
    calculate_errors : bool
        Whether to calculate errors (only for final stage).
    verbose : bool
        Print progress information.
    save_result : bool
        Persist each stage's result to the detector history.

    Returns
    -------
    dict
        Result dictionary with the final ``spectrum`` and metadata
        (``stages_run``, ``method_sequence``, ``convergence_history``,
        ``quality_metrics``, ``intermediate_results``, ``status``,
        ``message``).
    """
    if cascade_stages is None:
        cascade_stages = create_default_cascade("general")

    start_time = time.time()
    current_spectrum = None
    intermediate_results: Dict[str, Any] = {}
    convergence_history: List[Dict[str, float]] = []
    stages_run = 0

    A = _build_response_matrix(detector)
    measured_readings_array = np.array(
        [readings[d] for d in detector.detector_names]
    )

    for stage_idx, stage in enumerate(cascade_stages):
        method_name = stage.method

        if verbose:
            logger.info(
                f"Cascade Stage {stage_idx + 1}/{len(cascade_stages)}: {method_name}"
            )

        unfold_func = _get_method(detector, method_name)
        if unfold_func is None:
            if verbose:
                logger.warning(f"Method {method_name} not found, skipping")
            continue

        params = dict(stage.params)
        params["save_result"] = save_result

        if current_spectrum is not None and stage.use_as_initial:
            params["initial_spectrum"] = current_spectrum.copy()
            if verbose:
                logger.info("  Using previous result as initial guess")

        if current_spectrum is not None and stage.use_as_prior:
            # The bayes family treats initial_spectrum as the prior; other
            # methods either accept reference_spectrum or ignore the hint.
            if method_name in ("bayes", "bayes_spline"):
                params["initial_spectrum"] = current_spectrum.copy()
            elif "reference_spectrum" in _accepted_params(unfold_func):
                params["reference_spectrum"] = current_spectrum.copy()
            if verbose:
                logger.info("  Using previous result as prior")

        if stage.max_iterations is not None:
            cur = params.get("max_iterations")
            params["max_iterations"] = (
                min(cur, stage.max_iterations) if cur is not None else stage.max_iterations
            )

        if stage_idx == len(cascade_stages) - 1 and calculate_errors:
            params["calculate_errors"] = True
        else:
            params["calculate_errors"] = False

        try:
            result = _run_with_timeout(
                lambda: unfold_func(readings=readings, **params),
                stage.timeout,
            )
            stages_run += 1

            if "spectrum" in result and result["spectrum"] is not None:
                current_spectrum = result["spectrum"].copy()

                reconstructed_readings = A @ current_spectrum
                metrics = compute_quality_metrics(
                    current_spectrum,
                    reconstructed_readings,
                    measured_readings_array,
                    detector.E_MeV,
                )

                convergence_history.append(
                    {"stage": stage_idx, "method": method_name, **metrics}
                )

                if verbose:
                    logger.info(
                        f"  Chi²={metrics['chi_square']:.3f}, "
                        f"Smooth={metrics['smoothness']:.3f}, "
                        f"Flux err={metrics['flux_error']:.3f}"
                    )

                if stage.store_intermediate:
                    intermediate_results[f"stage_{stage_idx}_{method_name}"] = {
                        "spectrum": current_spectrum.copy(),
                        "metrics": metrics,
                        "result": result,
                    }

                if stage.quality_threshold is not None:
                    overall_quality = metrics.get("overall_quality", 0)
                    if overall_quality >= stage.quality_threshold:
                        if verbose:
                            logger.info(
                                f"  Quality threshold met "
                                f"({overall_quality:.3f} >= {stage.quality_threshold}), "
                                f"stopping"
                            )
                        break

        except Exception as e:  # noqa: BLE001 - continue on stage failure
            if verbose:
                logger.error(f"  Error in {method_name}: {e}")
            continue

    total_time = time.time() - start_time

    if current_spectrum is None:
        return {
            "spectrum": None,
            "stages_run": stages_run,
            "total_time": total_time,
            "intermediate_results": intermediate_results,
            "quality_metrics": {},
            "method_sequence": [s.method for s in cascade_stages[:stages_run]],
            "convergence_history": convergence_history,
            "status": "ERROR",
            "message": "No successful stages",
            "readings": readings,
            "doserates": None,
        }

    reconstructed_readings = A @ current_spectrum
    final_metrics = compute_quality_metrics(
        current_spectrum,
        reconstructed_readings,
        measured_readings_array,
        detector.E_MeV,
    )

    return {
        "spectrum": current_spectrum,
        "stages_run": stages_run,
        "total_time": total_time,
        "intermediate_results": intermediate_results,
        "quality_metrics": final_metrics,
        "method_sequence": [s.method for s in cascade_stages[:stages_run]],
        "convergence_history": convergence_history,
        "status": "OK",
        "message": f"Successfully completed {stages_run} cascade stages",
        "readings": readings,
        "doserates": None,
    }


def _accepted_params(func) -> set:
    """Return the set of keyword parameters accepted by ``func``."""
    import inspect

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def create_default_cascade(spectrum_type: str = "general") -> List[CascadeStage]:
    """Create default cascade configuration for different spectrum types.

    Parameters
    ----------
    spectrum_type : str
        One of ``"soft"``, ``"hard"``, ``"fast_refinement"`` or ``"general"``.

    Returns
    -------
    List[CascadeStage]
        Cascade configuration.
    """
    if spectrum_type == "soft":
        # Optimized for thermal/soft spectra
        return [
            CascadeStage(
                method="tsvd",
                params={"k": 10, "method": "discrepancy"},
                use_as_initial=False,
                store_intermediate=True,
            ),
            CascadeStage(
                method="landweber",
                params={"max_iterations": 100},
                use_as_initial=True,
                store_intermediate=False,
            ),
            CascadeStage(
                method="bayes_spline",
                params={"spline_smooth": 0.5},
                use_as_initial=True,
                use_as_prior=True,
                store_intermediate=False,
            ),
        ]

    elif spectrum_type == "hard":
        # Optimized for fast/hard spectra
        return [
            CascadeStage(
                method="cvxpy",
                params={"regularization": 1e-2, "norm": 1},
                use_as_initial=False,
                store_intermediate=True,
            ),
            CascadeStage(
                method="mlem",
                params={"max_iterations": 200},
                use_as_initial=True,
                store_intermediate=False,
            ),
            CascadeStage(
                method="hybrid_parametric",
                params={},
                use_as_initial=True,
                store_intermediate=False,
            ),
        ]

    elif spectrum_type == "fast_refinement":
        # Quick 2-stage refinement
        return [
            CascadeStage(
                method="landweber",
                params={"max_iterations": 50},
                use_as_initial=False,
                timeout=10.0,
            ),
            CascadeStage(
                method="cvxpy",
                params={"regularization": 1e-2},
                use_as_initial=True,
                timeout=30.0,
            ),
        ]

    else:  # general
        # General-purpose 3-stage cascade
        return [
            CascadeStage(
                method="tsvd",
                params={"k": 15, "method": "discrepancy"},
                use_as_initial=False,
                store_intermediate=True,
                quality_threshold=0.3,
            ),
            CascadeStage(
                method="mlem",
                params={"max_iterations": 150},
                use_as_initial=True,
                store_intermediate=False,
            ),
            CascadeStage(
                method="bayes_spline",
                params={"spline_smooth": 0.3},
                use_as_initial=True,
                use_as_prior=True,
                store_intermediate=False,
            ),
        ]


def unfold_adaptive_cascade(
    detector,
    readings: Dict[str, float],
    max_stages: int = 5,
    initial_method: str = "tsvd",
    calculate_errors: bool = False,
    verbose: bool = True,
    save_result: bool = False,
) -> Dict[str, Any]:
    """Perform adaptive cascade unfolding with dynamic method selection.

    Starts with an ``initial_method`` and adaptively selects subsequent
    methods based on quality metrics of intermediate solutions.

    Parameters
    ----------
    detector : Detector
        Configured Bonner-sphere detector.
    readings : Dict[str, float]
        Detector readings.
    max_stages : int
        Maximum number of cascade stages.
    initial_method : str
        Method to use for the first stage.
    calculate_errors : bool
        Whether to calculate errors.
    verbose : bool
        Print progress information.
    save_result : bool
        Persist each stage's result to the detector history.

    Returns
    -------
    dict
        Result dictionary (see :func:`unfold_cascade`).
    """
    available_methods = [
        "tsvd", "landweber", "mlem", "cvxpy", "qpsolvers",
        "statreg", "bayes", "bayes_spline", "cgls", "hybrid_gmres",
        "parametric2", "hybrid_parametric", "tikhonov_tv", "gravel",
    ]

    cascade_stages: List[CascadeStage] = []
    current_metrics = {"smoothness": 0.5, "chi_square": 10.0, "flux_error": 1.0}
    convergence_history: List[Dict[str, float]] = []

    for stage_idx in range(max_stages):
        used = {s.method for s in cascade_stages}
        remaining = [m for m in available_methods if m not in used]
        if stage_idx == 0:
            method = initial_method
        else:
            method = select_next_method(current_metrics, remaining, stage_idx)
            # Update metrics from the previous successful stage if available.
            if convergence_history:
                current_metrics = dict(convergence_history[-1])

        stage = CascadeStage(
            method=method,
            params={},
            use_as_initial=(stage_idx > 0),
            use_as_prior=(
                stage_idx > 0 and method in ("bayes", "bayes_spline")
            ),
            store_intermediate=True,
        )
        cascade_stages.append(stage)

        if verbose:
            logger.info(f"Adaptive stage {stage_idx + 1}: selected {method}")

        # Run incrementally so the next selection can use fresh metrics.
        result = unfold_cascade(
            detector,
            readings,
            cascade_stages=cascade_stages,
            calculate_errors=False,
            verbose=verbose,
            save_result=save_result,
        )
        if result.get("convergence_history"):
            convergence_history = result["convergence_history"]
        if result.get("spectrum") is None:
            break

    return unfold_cascade(
        detector,
        readings,
        cascade_stages=cascade_stages,
        calculate_errors=calculate_errors,
        verbose=verbose,
        save_result=save_result,
    )


__all__ = [
    "CascadeStage",
    "CascadeResult",
    "compute_quality_metrics",
    "select_next_method",
    "unfold_cascade",
    "create_default_cascade",
    "unfold_adaptive_cascade",
]
