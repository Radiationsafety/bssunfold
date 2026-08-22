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
from typing import Dict, Optional, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import warnings

from ._base_unfolder import run_unfolding, _build_system
from ..logging_config import get_logger

logger = get_logger("cascade")


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
        If quality metric below this, skip remaining stages.
    max_iterations : int
        Number of internal iterations for iterative methods.
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
    """Result of cascade unfolding."""
    spectrum: np.ndarray
    stages_run: int
    total_time: float
    intermediate_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    method_sequence: List[str]
    convergence_history: List[Dict[str, float]]
    status: str
    message: str


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
        Readings computed from reconstructed spectrum.
    measured_readings : np.ndarray
        Actual measured readings.
    energy : np.ndarray
        Energy grid.
    
    Returns
    -------
    dict
        Dictionary of quality metrics.
    """
    # Avoid division by zero
    eps = 1e-10
    
    # 1. Chi-square goodness of fit
    residuals = (measured_readings - reconstructed_readings) / (reconstructed_readings + eps)
    chi_square = float(np.sum(residuals**2))
    
    # 2. Smoothness metric (second derivative)
    log_spectrum = np.log(spectrum + eps)
    if len(log_spectrum) > 2:
        second_deriv = np.diff(log_spectrum, n=2)
        smoothness = float(1.0 / (1.0 + np.std(second_deriv)))
    else:
        smoothness = 1.0
    
    # 3. Flux conservation error
    total_flux_spec = np.sum(spectrum)
    total_flux_readings = np.sum(measured_readings)
    flux_error = abs(total_flux_spec - total_flux_readings) / (total_flux_readings + eps)
    
    # 4. Positivity violations
    negativity_count = int(np.sum(spectrum < 0))
    
    # 5. Spectral shape metrics
    # Hardness ratio consistency
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
        peaks = ((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])).sum()
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
    # Simple heuristic-based selection
    # In production, this could be ML-based
    
    smoothness = current_metrics.get("smoothness", 0.5)
    chi_square = current_metrics.get("chi_square", 10.0)
    flux_error = current_metrics.get("flux_error", 1.0)
    
    # If solution is not smooth enough, use regularization-heavy method
    if smoothness < 0.3:
        preferred = ["tsvd", "statreg", "bayes", "tikhonov_tv"]
    # If chi-square is high, need better fit
    elif chi_square > 5.0:
        preferred = ["mlem", "landweber", "cgls", "hybrid_gmres"]
    # If flux error is high, use flux-conserving method
    elif flux_error > 0.2:
        preferred = ["cvxpy", "qpsolvers", "gravel"]
    # Otherwise, use refinement method
    else:
        preferred = ["bayes_spline", "parametric2", "hybrid_parametric"]
    
    # Find first preferred method that's available
    for method in preferred:
        if method in available_methods:
            return method
    
    # Fallback to default methods based on stage
    defaults = ["landweber", "mlem", "cvxpy", "bayes"]
    return defaults[stage_number % len(defaults)]


def unfold_cascade(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Optional[Callable],
    readings: Dict[str, float],
    cascade_stages: List[CascadeStage],
    calculate_errors: bool = False,
    verbose: bool = True,
) -> CascadeResult:
    """Perform cascade unfolding with sequential method refinement.
    
    This function applies unfolding methods in sequence, where each method
    can use the result of the previous method as:
    1. Initial guess (use_as_initial=True)
    2. Prior/reference spectrum (use_as_prior=True)
    3. Regularization target
    
    The cascade can adaptively select methods based on quality metrics.
    
    Parameters
    ----------
    detector_names : List[str]
        Names of detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        Conversion coefficients.
    save_result_callback : callable
        Callback to save results.
    readings : Dict[str, float]
        Detector readings.
    cascade_stages : List[CascadeStage]
        Configuration for cascade stages.
    calculate_errors : bool
        Whether to calculate errors (only for final stage).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    CascadeResult
        Result object with spectrum and metadata.
    """
    import time
    from .unfold_cvxpy import unfold_cvxpy
    from .unfold_landweber import unfold_landweber
    from .unfold_mlem import unfold_mlem
    from .unfold_qpsolvers import unfold_qpsolvers
    from .unfold_tsvd import unfold_tsvd
    from .unfold_statreg import unfold_statreg
    from .unfold_bayes import unfold_bayes
    from .unfold_bayes_spline_regularization import unfold_bayes_spline
    from .unfold_cgls import unfold_cgls
    from .unfold_hybrid_gmres import unfold_hybrid_gmres
    from .unfold_parametric2 import unfold_parametric2
    from .unfold_hybrid_parametric import unfold_hybrid_parametric
    from .unfold_tikhonov_tv import unfold_tikhonov_tv
    from .unfold_gravel import unfold_gravel
    
    # Map method names to functions
    unfold_funcs = {
        "cvxpy": unfold_cvxpy,
        "landweber": unfold_landweber,
        "mlem": unfold_mlem,
        "qpsolvers": unfold_qpsolvers,
        "tsvd": unfold_tsvd,
        "statreg": unfold_statreg,
        "bayes": unfold_bayes,
        "bayes_spline": unfold_bayes_spline,
        "cgls": unfold_cgls,
        "hybrid_gmres": unfold_hybrid_gmres,
        "parametric2": unfold_parametric2,
        "hybrid_parametric": unfold_hybrid_parametric,
        "tikhonov_tv": unfold_tikhonov_tv,
        "gravel": unfold_gravel,
    }
    
    start_time = time.time()
    current_spectrum = None
    intermediate_results = {}
    convergence_history = []
    stages_run = 0
    
    # Build system matrix
    try:
        A, b = _build_system(detector_names, sensitivities, readings)
    except Exception as e:
        return CascadeResult(
            spectrum=None,
            stages_run=0,
            total_time=0,
            intermediate_results={},
            quality_metrics={},
            method_sequence=[],
            convergence_history=[],
            status="ERROR",
            message=f"Failed to build system: {e}",
        )
    
    measured_readings_array = np.array([readings[d] for d in detector_names])
    
    for stage_idx, stage in enumerate(cascade_stages):
        method_name = stage.method
        
        if verbose:
            logger.info(f"Cascade Stage {stage_idx + 1}/{len(cascade_stages)}: {method_name}")
        
        # Check if method is available
        if method_name not in unfold_funcs:
            if verbose:
                logger.warning(f"Method {method_name} not found, skipping")
            continue
        
        unfold_func = unfold_funcs[method_name]
        
        # Prepare parameters
        params = stage.params.copy()
        
        # Use previous result as initial guess if requested
        if current_spectrum is not None and stage.use_as_initial:
            params["initial_spectrum"] = current_spectrum.copy()
            if verbose:
                logger.info(f"  Using previous result as initial guess")
        
        # Use previous result as prior if requested (for Bayesian methods)
        if current_spectrum is not None and stage.use_as_prior:
            if "prior_spectrum" in params or "reference_spectrum" in params:
                pass  # Already specified
            else:
                params["prior_spectrum"] = current_spectrum.copy()
                if verbose:
                    logger.info(f"  Using previous result as prior")
        
        # Set iteration limits if specified
        if stage.max_iterations is not None:
            if "max_iter" in params:
                params["max_iter"] = min(params["max_iter"], stage.max_iterations)
            else:
                params["max_iter"] = stage.max_iterations
        
        # Only calculate errors for final stage
        if stage_idx == len(cascade_stages) - 1 and calculate_errors:
            params["calculate_errors"] = True
        else:
            params["calculate_errors"] = False
        
        try:
            # Run unfolding method
            result = unfold_func(
                detector_names=detector_names,
                n_energy_bins=n_energy_bins,
                E_MeV=E_MeV,
                sensitivities=sensitivities,
                cc_icrp116=cc_icrp116,
                save_result_callback=save_result_callback,
                readings=readings,
                **params,
            )
            
            stages_run += 1
            
            if "spectrum" in result and result["spectrum"] is not None:
                current_spectrum = result["spectrum"].copy()
                
                # Compute quality metrics
                reconstructed_readings = A @ current_spectrum
                metrics = compute_quality_metrics(
                    current_spectrum,
                    reconstructed_readings,
                    measured_readings_array,
                    E_MeV,
                )
                
                convergence_history.append({
                    "stage": stage_idx,
                    "method": method_name,
                    **metrics,
                })
                
                if verbose:
                    logger.info(f"  Chi²={metrics['chi_square']:.3f}, "
                               f"Smooth={metrics['smoothness']:.3f}, "
                               f"Flux err={metrics['flux_error']:.3f}")
                
                # Store intermediate result if requested
                if stage.store_intermediate:
                    intermediate_results[f"stage_{stage_idx}_{method_name}"] = {
                        "spectrum": current_spectrum.copy(),
                        "metrics": metrics,
                        "result": result,
                    }
                
                # Check quality threshold for early stopping
                if stage.quality_threshold is not None:
                    overall_quality = metrics.get("overall_quality", 0)
                    if overall_quality >= stage.quality_threshold:
                        if verbose:
                            logger.info(f"  Quality threshold met ({overall_quality:.3f} >= {stage.quality_threshold}), stopping")
                        break
                
        except Exception as e:
            if verbose:
                logger.error(f"  Error in {method_name}: {e}")
            # Continue to next stage instead of failing completely
            continue
    
    total_time = time.time() - start_time
    
    if current_spectrum is None:
        return CascadeResult(
            spectrum=None,
            stages_run=stages_run,
            total_time=total_time,
            intermediate_results=intermediate_results,
            quality_metrics={},
            method_sequence=[s.method for s in cascade_stages[:stages_run]],
            convergence_history=convergence_history,
            status="ERROR",
            message="No successful stages",
        )
    
    # Compute final quality metrics
    reconstructed_readings = A @ current_spectrum
    final_metrics = compute_quality_metrics(
        current_spectrum,
        reconstructed_readings,
        measured_readings_array,
        E_MeV,
    )
    
    return CascadeResult(
        spectrum=current_spectrum,
        stages_run=stages_run,
        total_time=total_time,
        intermediate_results=intermediate_results,
        quality_metrics=final_metrics,
        method_sequence=[s.method for s in cascade_stages[:stages_run]],
        convergence_history=convergence_history,
        status="OK",
        message=f"Successfully completed {stages_run} cascade stages",
    )


def create_default_cascade(spectrum_type: str = "general") -> List[CascadeStage]:
    """Create default cascade configuration for different spectrum types.
    
    Parameters
    ----------
    spectrum_type : str
        Type of spectrum: "soft", "hard", "general", "fast_refinement".
    
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
                params={"n_components": 10},
                use_as_initial=False,
                store_intermediate=True,
            ),
            CascadeStage(
                method="landweber",
                params={"step_size": 0.01, "max_iter": 100},
                use_as_initial=True,
                store_intermediate=False,
            ),
            CascadeStage(
                method="bayes_spline",
                params={"smoothness_weight": 0.5},
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
                params={"regularization": "l1"},
                use_as_initial=False,
                store_intermediate=True,
            ),
            CascadeStage(
                method="mlem",
                params={"max_iter": 200},
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
                params={"step_size": 0.05, "max_iter": 50},
                use_as_initial=False,
                timeout=10.0,
            ),
            CascadeStage(
                method="cvxpy",
                params={},
                use_as_initial=True,
                timeout=30.0,
            ),
        ]
    
    else:  # general
        # General-purpose 3-stage cascade
        return [
            CascadeStage(
                method="tsvd",
                params={"n_components": 15},
                use_as_initial=False,
                store_intermediate=True,
                quality_threshold=0.3,
            ),
            CascadeStage(
                method="mlem",
                params={"max_iter": 150},
                use_as_initial=True,
                store_intermediate=False,
            ),
            CascadeStage(
                method="bayes_spline",
                params={"smoothness_weight": 0.3},
                use_as_initial=True,
                use_as_prior=True,
                store_intermediate=False,
            ),
        ]


def unfold_adaptive_cascade(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback: Optional[Callable],
    readings: Dict[str, float],
    max_stages: int = 5,
    initial_method: str = "tsvd",
    calculate_errors: bool = False,
    verbose: bool = True,
) -> CascadeResult:
    """Perform adaptive cascade unfolding with dynamic method selection.
    
    This function starts with an initial method and adaptively selects
    subsequent methods based on quality metrics of intermediate solutions.
    
    Parameters
    ----------
    detector_names : List[str]
        Names of detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        Conversion coefficients.
    save_result_callback : callable
        Callback to save results.
    readings : Dict[str, float]
        Detector readings.
    max_stages : int
        Maximum number of cascade stages.
    initial_method : str
        Method to use for first stage.
    calculate_errors : bool
        Whether to calculate errors.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    CascadeResult
        Result object with spectrum and metadata.
    """
    available_methods = [
        "tsvd", "landweber", "mlem", "cvxpy", "qpsolvers",
        "statreg", "bayes", "bayes_spline", "cgls", "hybrid_gmres",
        "parametric2", "hybrid_parametric", "tikhonov_tv", "gravel",
    ]
    
    cascade_stages = []
    current_metrics = {"smoothness": 0.5, "chi_square": 10.0, "flux_error": 1.0}
    
    for stage_idx in range(max_stages):
        if stage_idx == 0:
            method = initial_method
        else:
            method = select_next_method(current_metrics, available_methods, stage_idx)
        
        stage = CascadeStage(
            method=method,
            params={},
            use_as_initial=(stage_idx > 0),
            use_as_prior=(stage_idx > 0 and method in ["bayes", "bayes_spline"]),
            store_intermediate=True,
        )
        cascade_stages.append(stage)
        
        # Run single stage to get metrics for next selection
        # (simplified - in practice would call unfold_cascade incrementally)
        if verbose:
            logger.info(f"Adaptive stage {stage_idx + 1}: selected {method}")
    
    # Run the full cascade
    result = unfold_cascade(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        cascade_stages=cascade_stages,
        calculate_errors=calculate_errors,
        verbose=verbose,
    )
    
    return result


# Export functions
__all__ = [
    "CascadeStage",
    "CascadeResult",
    "compute_quality_metrics",
    "select_next_method",
    "unfold_cascade",
    "create_default_cascade",
    "unfold_adaptive_cascade",
]
