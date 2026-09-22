"""Truncated SVD (TSVD) unfolding method for neutron spectrum reconstruction.

This module provides the core solve_tsvd solver and the unfold_tsvd
wrapper for use with the Detector class.

The SVD backend is selectable via the ``svd_solver`` parameter, which
maps the R packages ``svd`` (PROPACK / Lanczos-bidiagonalization SVD)
and ``rARPACK`` (ARPACK eigen/SVD solver) onto their SciPy equivalents:

* ``"full"``    -- dense LAPACK SVD (``scipy.linalg.svd``), default;
* ``"arpack"`` -- implicitly restarted Arnoldi/Lanczos (the same ARPACK
  Fortran library wrapped by R's ``rARPACK``/``RSpectra``);
* ``"propack"`` -- Lanczos bidiagonalization with partial reorthogonalization
  (the same PROPACK algorithm as R's ``svd::propack.svd``).

The iterative backends compute only the leading ``k`` singular triplets
and are useful when a fixed truncation ``k`` is known; automatic
k-selection methods require the full singular spectrum and always use
the dense backend.
"""

from typing import Any

import numpy as np
from scipy.linalg import svd

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_tsvd", "unfold_tsvd"]

_VALID_SVD_SOLVERS = ("full", "arpack", "propack")


def _automatic_k_selection(
    s: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    method: str = "discrepancy",
    noise_level: float = None,
) -> int:
    """Automatically select truncation parameter k for TSVD."""
    m, n = A.shape
    max_k = min(m, n)

    if method == "discrepancy":
        if noise_level is None:
            noise_level = s[0] * 1e-3
        U, s_full, Vh = svd(A, full_matrices=False)
        for i in range(1, max_k + 1):
            s_i = s_full[:i]
            U_i = U[:, :i]
            V_i = Vh[:i, :].T
            x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b
            residual = np.linalg.norm(A @ x_i - b)
            if residual <= noise_level * np.sqrt(max(m - i, 1)):
                return i
        return max_k

    if method == "energy":
        energy_threshold = 0.95
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        return int(np.argmax(cumulative_energy >= energy_threshold)) + 1

    if method == "l_curve":
        U, s_full, Vh = svd(A, full_matrices=False)
        V = Vh.T
        residual_norms = []
        solution_norms = []
        for i in range(1, min(len(s_full), n) + 1):
            s_i = s_full[:i]
            U_i = U[:, :i]
            V_i = V[:, :i]
            x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b
            residual_norms.append(np.linalg.norm(A @ x_i - b))
            solution_norms.append(np.linalg.norm(x_i))

        log_res = np.log(np.maximum(residual_norms, 1e-300))
        log_sol = np.log(np.maximum(solution_norms, 1e-300))
        curvature = []
        for i in range(1, len(log_res) - 1):
            dx1 = log_res[i] - log_res[i - 1]
            dy1 = log_sol[i] - log_sol[i - 1]
            dx2 = log_res[i + 1] - log_res[i]
            dy2 = log_sol[i + 1] - log_sol[i]
            curv = abs(dx1 * dy2 - dx2 * dy1) / (
                (dx1**2 + dy1**2) ** 1.5 + 1e-10
            )
            curvature.append(curv)
        if len(curvature) > 0:
            k_idx = np.argmax(curvature) + 1
            return min(k_idx + 1, len(s))
        return len(s) // 2

    if method == "gcv":
        U, s_full, Vh = svd(A, full_matrices=False)
        beta = U.T @ b
        gcv_values = []
        k_values = list(range(1, min(len(s_full), n) + 1))
        for i in k_values:
            residual = np.sum(beta[i:] ** 2)
            eff_params = m - i
            gcv = residual / (eff_params**2) if eff_params > 0 else np.inf
            gcv_values.append(gcv)
        return k_values[np.argmin(gcv_values)]

    if method == "threshold_ratio":
        threshold_ratio = 1e-2
        s_normalized = s / s[0]
        return int(np.sum(s_normalized > threshold_ratio))

    if method == "median_threshold":
        median_s = np.median(s)
        return int(np.sum(s >= median_s))

    if method == "donoho":
        sigma_donoho = 0.05
        n_val = n
        donoho_rcond = 4 / np.sqrt(3) * np.sqrt(n_val) * sigma_donoho
        return int(np.sum(s > donoho_rcond))

    mean_s = np.mean(s)
    return int(np.sum(s >= mean_s))


def _truncated_svd(
    A: np.ndarray,
    k: int | None,
    svd_solver: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute the (truncated) SVD triplets used by the solver.

    Returns ``(U, s, Vh, full_spectrum)``.  For ``svd_solver='full'`` or
    when automatic k-selection is requested (``k is None``) the dense
    LAPACK SVD is returned; otherwise only the leading ``k`` triplets
    are computed with the chosen iterative backend (ARPACK or PROPACK).
    A failing iterative backend degrades to the dense solver with a
    ``RuntimeWarning``.
    """
    if svd_solver not in _VALID_SVD_SOLVERS:
        raise ValueError(
            f"svd_solver must be one of {_VALID_SVD_SOLVERS}, "
            f"got {svd_solver!r}"
        )
    m, n = A.shape
    full_spectrum = svd_solver == "full" or k is None
    if full_spectrum:
        U, s, Vh = svd(A, full_matrices=False)
        return U, s, Vh, True

    k_eff = min(int(k), min(m, n) - 1)
    if k_eff < 1:
        U, s, Vh = svd(A, full_matrices=False)
        return U, s, Vh, True
    try:
        from scipy.sparse.linalg import svds

        U, s, Vh = svds(A, k=k_eff, solver=svd_solver)
        # svds returns ascending singular values -- reverse to descending
        order = np.argsort(s)[::-1]
        return U[:, order], s[order], Vh[order, :], False
    except (ImportError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        import warnings

        warnings.warn(
            f"svd_solver={svd_solver!r} failed ({exc}); "
            "falling back to the dense LAPACK SVD",
            RuntimeWarning,
            stacklevel=2,
        )
        U, s, Vh = svd(A, full_matrices=False)
        return U, s, Vh, True


def solve_tsvd(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    method: str = "discrepancy",
    k: int | None = None,
    threshold: float | None = None,
    noise_level: float | None = None,
    svd_solver: str = "full",
) -> np.ndarray:
    """Solve unfolding problem using Truncated SVD (TSVD).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    method : str, optional
        K-selection method: 'discrepancy', 'l_curve', 'gcv', 'energy',
        'threshold_ratio', 'median_threshold', 'donoho' (default: 'discrepancy').
    k : int, optional
        Fixed number of singular values to keep. Overrides method.
    threshold : float, optional
        Threshold ratio for singular value truncation.
    noise_level : float, optional
        Noise level estimate for discrepancy principle.
    svd_solver : str, optional
        SVD backend: ``'full'`` (dense LAPACK, default), ``'arpack'`` or
        ``'propack'``.  The iterative backends are only used when ``k``
        is fixed; automatic k-selection falls back to the dense solver.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    A, b, _ = validate_system(A, b)
    U, s, Vh, _full_spectrum = _truncated_svd(A, k, svd_solver)
    V = Vh.T

    if k is not None:
        k = min(k, len(s))
    elif threshold is not None:
        k = np.sum(s / s[0] > threshold)
    else:
        k = _automatic_k_selection(s, A, b, method=method, noise_level=noise_level)

    k = max(1, min(k, A.shape[0], A.shape[1]))
    s_k = s[:k]
    U_k = U[:, :k]
    V_k = V[:, :k]

    x = V_k @ np.diag(1.0 / s_k) @ U_k.T @ b
    return np.maximum(x, 0)


def unfold_tsvd(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    method: str = "discrepancy",
    k: int | None = None,
    threshold: float | None = None,
    noise_level: float | None = None,
    svd_solver: str = "full",
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using Truncated SVD (TSVD).

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
        Initial spectrum guess.
    method : str, optional
        K-selection method (default: 'discrepancy').
    k : int, optional
        Fixed truncation parameter.
    threshold : float, optional
        Threshold ratio for truncation.
    noise_level : float, optional
        Noise level estimate.
    svd_solver : str, optional
        SVD backend: ``'full'`` (default), ``'arpack'`` or ``'propack'``.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
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
    x0_default = np.zeros(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_tsvd,
            method=method,
            k=k,
            threshold=threshold,
            noise_level=noise_level,
            svd_solver=svd_solver,
        ),
        solve_kwargs={},
        method_name="TSVD",
        extra_output={
            "k": k,
            "k_method": method,
            "svd_solver": svd_solver,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level or 0.01,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
