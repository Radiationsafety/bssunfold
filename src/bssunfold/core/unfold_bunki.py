"""BUNKI (SPUNIT) unfolding method for neutron spectrum reconstruction.

BUNKI is the Bonner-sphere unfolding code developed at the Naval Research
Laboratory (1983) and, like BUNKI-UT, uses the SPUNIT iterative algorithm
(a Russian/PNL multisphere unfolding method, RSICC PSR-266). This module
follows the reference NumPy implementation in the BUMS2 package
(https://github.com/larnold34/BUMS2, MIT license).

SPUNIT operates on the lethargy-weighted response matrix ``aleth`` (which is
exactly what the ``Detector`` builds as ``A``) transformed by the initial
spectrum ``alethnew = aleth * x0`` and on the starting spectrum ``spl = 1``:

    ss_j    = sum_i alethnew_ij / b_i
    spll_j  = sum_i spl_j * alethnew_ij / (ss_j * E_i)     E_i = sum_j alethnew_ij spl_j
    spl     <- 3-point-smoothed spll (bins 0, 1 kept verbatim)

The final spectrum is the inverse transform ``x = spl * x0``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_bunki", "unfold_bunki"]


def solve_bunki(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    smoothing: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    lethargy_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the BUNKI (SPUNIT) algorithm.

    Parameters
    ----------
    A : np.ndarray
        Lethargy-weighted response matrix (m x n) as built by the Detector
        class (see :meth:`bssunfold.Detector`).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    smoothing : float, optional
        Three-point smoothing factor (default: 0.1).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    lethargy_weights : np.ndarray, optional
        Per-bin lethargy widths. Only needed when ``A`` is supplied as a
        per-bin (non-lethargy-weighted) response matrix; the Detector-built
        matrix is already lethargy-weighted so this can be left as None.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n = A.shape[1]

    if lethargy_weights is not None:
        wdleth = np.asarray(lethargy_weights, dtype=float)
        if wdleth.shape != (n,):
            raise ValueError(f"lethargy_weights must have shape ({n},)")
        A = A * wdleth[None, :]

    if np.any(b < 0):
        raise ValueError("BUNKI requires strictly positive measurements")

    if np.any(b == 0):
        # Zero readings carry no usable information for BUNKI and would cause
        # division by zero. Drop those detectors instead of failing the whole
        # unfolding (e.g. purely thermal spectra where the 18-inch spheres
        # have essentially zero response).
        keep = b > 0
        A = A[keep]
        b = b[keep]
        if b.size == 0:
            raise ValueError("BUNKI requires strictly positive measurements")

    m = A.shape[0]

    x0_safe = np.maximum(x0, 0.0)
    # trans_mat: response scaled by the initial spectrum, spl starts at ones.
    aleth = A * x0_safe[None, :]
    spl = np.ones(n)
    bcc = aleth @ spl

    # ss[j] = sum_i aleth[i, j] / b[i] -- vectorized
    inv_b = np.where(b > 0.0, 1.0 / b, 0.0)
    ss = aleth.T @ inv_b

    inv_ss = np.where(ss > 0.0, 1.0 / np.maximum(ss, 1e-37), 0.0)
    denom_s = 1.0 + 2.0 * smoothing

    converged = False
    iterations = 0

    for k in range(1, max_iterations + 1):
        iterations = k

        inv_bcc = np.where(bcc > 0.0, 1.0 / np.maximum(bcc, 1e-37), 0.0)
        spll = spl * (aleth.T @ inv_bcc) * inv_ss
        spll = np.where(spll < 1e-37, 0.0, spll)
        spll = np.where(spl <= 0.0, 0.0, spll)

        # Vectorized 3-point smoothing
        new_spl = spll.copy()
        if n > 2:
            # Interior points: weighted average of neighbors
            new_spl[1:-1] = (
                smoothing * spll[:-2] + spll[1:-1] + smoothing * spll[2:]
            ) / denom_s

        bcc = aleth @ new_spl

        # Convergence check with pre-allocated spl_old
        rel = np.linalg.norm(new_spl - spl) / (np.linalg.norm(spl) + 1e-12)
        spl = new_spl

        if rel < tolerance:
            converged = True
            break

    spectrum = spl * x0_safe
    return spectrum, iterations, converged


def unfold_bunki(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    smoothing: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the BUNKI (SPUNIT) algorithm.

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
    smoothing : float, optional
        Three-point smoothing factor (default: 0.1).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
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
    x0_default[0] = 0.0

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
            solve_bunki,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="BUNKI",
        extra_output={
            "smoothing": float(smoothing),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
