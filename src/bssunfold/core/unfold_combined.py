"""Combined unfolding method applying multiple methods sequentially.

This module provides the `unfold_combined` function which applies a pipeline
of unfolding methods sequentially, optionally using the result of each method
as the initial guess for the next.
"""

import numpy as np
from typing import Dict, Optional, Any, List

from ..logging_config import get_logger

logger = get_logger("detector")


def unfold_combined(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    pipeline: List[Dict[str, Any]],
    calculate_errors: bool = False,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Combined unfolding method applying multiple methods sequentially.

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
    pipeline : List[Dict[str, Any]]
        List of methods for sequential application. Each dict should contain:
        - 'method': str - method name (e.g., 'cvxpy', 'landweber', 'mlem')
        - 'params': dict - parameters for the method (including optional 'basis')
        - 'use_as_initial': bool (optional) - use result as initial guess
        - 'store_intermediate': bool (optional) - store intermediate result
    calculate_errors : bool, optional
        Flag to calculate errors for the last method.
    verbose : bool, optional
        Flag to print debug information.

    Returns
    -------
    Dict
        Dictionary with unfolding results.
    """
    from .unfold_cvxpy import unfold_cvxpy
    from .unfold_landweber import unfold_landweber
    from .unfold_mlem import unfold_mlem
    from .unfold_mlem_odl import unfold_mlem_odl
    from .unfold_qpsolvers import unfold_qpsolvers
    from .unfold_doroshenko import unfold_doroshenko
    from .unfold_kaczmarz import unfold_kaczmarz
    from .unfold_lmfit import unfold_lmfit

    current_spectrum = None
    intermediate_results = {}
    final_result = None

    if verbose:
        logger.info(f"Combined algorithm, methods = {len(pipeline)}")

    for i, stage in enumerate(pipeline):
        method = stage['method']
        params = stage.get('params', {}).copy()
        use_as_initial = stage.get('use_as_initial', True)
        store_intermediate = stage.get('store_intermediate', False)

        if verbose:
            logger.info(f"Stage {i+1}/{len(pipeline)}: {method}")

        if current_spectrum is not None and use_as_initial:
            params['initial_spectrum'] = current_spectrum.copy()
            if verbose:
                logger.info("Previous result used as initial spectrum")

        # Select the appropriate unfold function
        unfold_funcs = {
            'cvxpy': unfold_cvxpy,
            'landweber': unfold_landweber,
            'mlem': unfold_mlem,
            'mlem_odl': unfold_mlem_odl,
            'qpsolvers': unfold_qpsolvers,
            'doroshenko': unfold_doroshenko,
            'kaczmarz': unfold_kaczmarz,
            'lmfit': unfold_lmfit,
        }

        if method not in unfold_funcs:
            raise ValueError(
                f"Method '{method}' not found. "
                f"Available methods: {list(unfold_funcs.keys())}"
            )

        unfold_func = unfold_funcs[method]

        # Only calculate errors for the last stage if requested
        if i == len(pipeline) - 1 and calculate_errors:
            params['calculate_errors'] = True
        else:
            params['calculate_errors'] = False

        try:
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
        except Exception as e:
            logger.error(f"Error in method {method}: {e}")
            raise

        if 'spectrum' in result:
            current_spectrum = result['spectrum'].copy()
            if verbose:
                logger.info(
                    f"  Spectrum norm: {np.linalg.norm(current_spectrum):.6f}"
                )

        if store_intermediate:
            intermediate_results[f'stage_{i+1}_{method}'] = result.copy()

        final_result = result

    if verbose:
        logger.info("Combined method finished")

    if final_result is None:
        return None

    output = final_result.copy()
    output['pipeline_info'] = {
        'stages': [stage['method'] for stage in pipeline],
        'params': [stage.get('params', {}) for stage in pipeline]
    }

    if intermediate_results:
        output['intermediate_results'] = intermediate_results

    return output
