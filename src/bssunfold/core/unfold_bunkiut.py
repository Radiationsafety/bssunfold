"""BUNKI-UT (BON31G) unfolding method for neutron spectrum reconstruction.

BUNKI-UT is the University of Texas modernisation of the BUNKI code
(K. A. Miller et al.) and offers the BON31G and SPUNIT iterative unfolding
algorithms. This module implements the BON31G variant following the
reference NumPy implementation in the BUMS2 package
(https://github.com/larnold34/BUMS2, MIT license) and the original FORTRAN
in ``FORTRAN/BUNKI-UT/bunkiut.f``.

BON31G operates on the lethargy-weighted response matrix ``aleth`` (exactly
the ``A`` built by the ``Detector`` class) transformed by the initial
spectrum ``alethnew = aleth * x0``, with the starting spectrum ``spl = 1``:

    bk_jm   = sum_i alethnew_ij * alethnew_im
    vect_j  = sum_i alethnew_ij * b_i
    ax_j    = sum_m spl_m * bk_jm
    spll_j  = spl_j * vect_j / ax_j
    spl     <- 3-point-smoothed spll (bins 0, 1 kept verbatim)

The final spectrum is the inverse transform ``x = spl * x0``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_bunkiut", "unfold_bunkiut"]


def solve_bunkiut(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    smoothing: float = 0.05,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    lethargy_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the BUNKI-UT (BON31G) algorithm.

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
        Three-point smoothing factor (default: 0.05).
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
    _, n = A.shape

    if lethargy_weights is not None:
        wdleth = np.asarray(lethargy_weights, dtype=float)
        if wdleth.shape != (n,):
            raise ValueError(f"lethargy_weights must have shape ({n},)")
        A = A * wdleth[None, :]

    if np.any(b < 0):
        raise ValueError("BUNKI-UT requires strictly positive measurements")

    if np.any(b == 0):
        # Same guard as BUNKI: drop zero-reading detectors instead of failing.
        keep = b > 0
        A = A[keep]
        b = b[keep]
        if b.size == 0:
            raise ValueError("BUNKI-UT requires strictly positive measurements")

    x0_safe = np.maximum(x0, 0.0)
    # trans_mat: response scaled by the initial spectrum, spl starts at ones.
    aleth = A * x0_safe[None, :]
    spl = np.ones(n)

    # Precompute the symmetric bk matrix and the vect vector once.
    bk = aleth.T @ aleth
    vect = aleth.T @ b

    denom_s = 1.0 + 2.0 * smoothing

    converged = False
    iterations = 0

    for k in range(1, max_iterations + 1):
        iterations = k
        spl_old = spl.copy()

        # ax[j] = sum_m spl[m] * bk[j, m]
        ax = np.maximum(spl @ bk.T, 1e-37)
        spll = spl * vect / ax
        spll = np.where(spll < 1e-37, 0.0, spll)

        new_spl = spll.copy()
        if n > 2:
            for j in range(2, n):
                hi = spll[j + 1] if j + 1 < n else 0.0
                new_spl[j] = (
                    spll[j - 1] * smoothing + spll[j] + hi * smoothing
                ) / denom_s
        new_spl[0] = spll[0]
        if n > 1:
            new_spl[1] = spll[1]

        spl = new_spl

        rel = np.linalg.norm(spl - spl_old) / (np.linalg.norm(spl_old) + 1e-12)
        if rel < tolerance:
            converged = True
            break

    spectrum = spl * x0_safe
    return spectrum, iterations, converged


def unfold_bunkiut(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    smoothing: float = 0.05,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the BUNKI-UT (BON31G) algorithm.

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
        Three-point smoothing factor (default: 0.05).
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
            solve_bunkiut,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="BUNKI-UT",
        extra_output={
            "smoothing": float(smoothing),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
