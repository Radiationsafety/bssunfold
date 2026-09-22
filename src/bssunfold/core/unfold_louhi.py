"""LOUHI unfolding method (Routti & Sandberg 1980).

Port of the LOUHI78 general purpose unfolding program (J. T. Routti and
V. Sandberg, "General purpose unfolding program LOUHI78 with linear and
nonlinear regressions", Computer Physics Communications 21 (1980) 119-135,
doi:10.1016/0010-4655(80)90021-4).

LOUHI formulates Bonner-sphere unfolding as a constrained weighted
least-squares problem with generalized smoothing:

    chi2(phi) = sum_i [(b_i - (A phi)_i) / sigma_i]^2
                + lambda^2 * sum_k [L (phi - phi0)]_k^2

where ``phi0`` is the default (a-priori) spectrum, ``sigma_i`` the
per-detector measurement uncertainties and ``L`` the smoothing operator.
The quadratic form is minimized by quadratic programming with
non-negativity constraints (LOUHI's LSI step) using Hildreth's iterative
coordinate algorithm, and the smoothing weight ``lambda`` can either be
fixed (linear mode) or adjusted automatically by a nonlinear regression
on the total chi-square so that the data misfit reaches its expected
value (nonlinear mode, LOUHI's smoothing-parameter search).

The statistical error propagation of LOUHI78 (covariance of the free
variables from the inverse Hessian on the active set) is available via
:func:`louhi_covariance`.
"""

from typing import Any

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._line_search import golden_section_minimize

__all__ = [
    "louhi_covariance",
    "louhi_smoothing_matrix",
    "solve_louhi",
    "unfold_louhi",
]

# Smoothing operator orders supported by the generalized smoothing term.
_SMOOTH_ORDERS = (0, 1, 2)

_EPS = 1e-12


def louhi_smoothing_matrix(n: int, smooth_order: int = 1) -> np.ndarray:
    """Build the generalized smoothing operator ``L`` of LOUHI.

    Parameters
    ----------
    n : int
        Number of energy bins.
    smooth_order : int, optional
        Order of the smoothing functional: ``0`` shrinks the solution
        toward the default spectrum (identity operator), ``1`` penalizes
        first differences of the deviation from the default spectrum and
        ``2`` penalizes second differences (default: 1).

    Returns
    -------
    np.ndarray
        The ``(n, n)`` smoothing matrix ``L``.
    """
    if smooth_order not in _SMOOTH_ORDERS:
        raise ValueError(
            f"smooth_order must be one of {_SMOOTH_ORDERS}, got {smooth_order}"
        )
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    identity = np.eye(n)
    if smooth_order == 0 or n == 1:
        return identity
    first = np.zeros((n - 1, n))
    idx = np.arange(n - 1)
    first[idx, idx] = -1.0
    first[idx, idx + 1] = 1.0
    if smooth_order == 1 or n == 2:
        return first
    second = np.zeros((n - 2, n))
    idx2 = np.arange(n - 2)
    second[idx2, idx2] = 1.0
    second[idx2, idx2 + 1] = -2.0
    second[idx2, idx2 + 2] = 1.0
    return second


def _hildreth_qp(
    H: np.ndarray,
    g: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Minimize ``1/2 x' H x - g' x`` subject to ``x >= 0``.

    Hildreth's iterative quadratic-programming algorithm as used by the
    LSI step of LOUHI78: cyclic coordinate minimization with projection
    of every coordinate onto its non-negativity constraint.  The sweep
    stops when the relative change of the quadratic objective between two
    consecutive sweeps drops below ``tolerance`` (the ill-conditioning of
    Bonner-sphere response matrices makes the objective a far more
    robust convergence indicator than the raw iterates).

    Parameters
    ----------
    H : np.ndarray
        Symmetric positive-definite Hessian ``(n, n)``.
    g : np.ndarray
        Linear term ``(n,)``.
    x0 : np.ndarray
        Starting point ``(n,)`` (clipped to ``x >= 0``).
    max_iterations : int, optional
        Maximum number of full coordinate sweeps (default: 100).
    tolerance : float, optional
        Relative objective change for convergence (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        ``(solution, sweeps, converged)``.
    """
    x = np.maximum(np.asarray(x0, dtype=float), 0.0)
    n = x.shape[0]
    diag = np.maximum(np.diag(H), _EPS)
    converged = False
    sweeps = 0
    prev_obj = np.inf
    for sweep in range(1, max_iterations + 1):
        sweeps = sweep
        for j in range(n):
            grad_j = H[j] @ x - g[j]
            x[j] = max(0.0, x[j] - grad_j / diag[j])
        obj = 0.5 * float(x @ (H @ x)) - float(g @ x)
        if abs(prev_obj - obj) <= tolerance * max(1.0, abs(obj)):
            converged = True
            break
        prev_obj = obj
    return x, sweeps, converged


def _validate_inputs(
    A: np.ndarray, b: np.ndarray, sigma: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the measurement system and derive per-detector sigmas."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(
            f"Response matrix must be a non-empty 2-D array, got shape {A.shape}"
        )
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(
            f"Measurement vector shape {b.shape} does not match response "
            f"matrix with {A.shape[0]} rows"
        )
    if not np.any(b > 0):
        raise ValueError(
            "At least one positive measurement is required; "
            "all readings are zero or negative"
        )
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != b.shape:
            raise ValueError(
                f"sigma must have shape {b.shape}, got {sigma.shape}"
            )
        sigma = np.maximum(sigma, _EPS)
    else:
        sigma = 0.1 * np.maximum(b, _EPS)
    return A, b, sigma


def _weighted_normal_equations(
    A: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    x0: np.ndarray,
    L: np.ndarray,
    smoothness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the normal equations of the LOUHI quadratic program.

    Returns ``(H, g)`` for ``min 1/2 x' H x - g' x`` of the objective
    ``chi2_stat + lambda^2 * ||L (x - x0)||^2``.
    """
    w = 1.0 / sigma**2
    ata = A.T @ (A * w[:, None])
    atb = A.T @ (b * w)
    ltl = L.T @ L
    H = 2.0 * (ata + smoothness**2 * ltl)
    g = 2.0 * (atb + smoothness**2 * (ltl @ x0))
    return H, g


def _data_chi2(
    A: np.ndarray, b: np.ndarray, sigma: np.ndarray, x: np.ndarray
) -> float:
    """Weighted chi-square of the data term for spectrum ``x``."""
    resid = (b - A @ x) / sigma
    return float(resid @ resid)


def _smoothness_bracket(
    objective, lo: float = -6.0, hi: float = 6.0, n_scan: int = 25
) -> tuple[float, float]:
    """Bracket the crossing of a monotone objective on ``[lo, hi]``.

    Scans a uniform grid on ``log10(lambda)`` in ``[lo, hi]`` and returns
    the first adjacent pair of grid points where the objective changes
    sign.  Falls back to the full interval when no crossing exists.
    """
    grid = np.linspace(lo, hi, n_scan)
    values = [objective(t) for t in grid]
    for k in range(len(grid) - 1):
        if values[k] == 0.0 or values[k] * values[k + 1] <= 0.0:
            return float(grid[k]), float(grid[k + 1])
    return lo, hi


def solve_louhi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    smoothness: float = 1.0,
    smooth_order: int = 1,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    relative_uncertainty: float = 0.1,
    sigma: np.ndarray | None = None,
    auto_smooth: bool = False,
    chi2_target: float | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with the LOUHI78 algorithm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Default (a-priori) spectrum ``(n,)`` used as the starting point
        and as the reference of the generalized smoothing term.  When
        None, a flat unit spectrum is used.
    smoothness : float, optional
        Smoothing weight ``lambda`` of the generalized smoothing term
        (default: 1.0).  Ignored when ``auto_smooth`` is True.
    smooth_order : int, optional
        Order of the smoothing operator: ``0`` (identity), ``1`` (first
        differences) or ``2`` (second differences); default 1.
    max_iterations : int, optional
        Maximum number of Hildreth coordinate sweeps (default: 500).
    tolerance : float, optional
        Maximum relative change between sweeps for convergence
        (default: 1e-6).
    relative_uncertainty : float, optional
        Relative measurement uncertainty used to derive detector sigma
        values when ``sigma`` is not supplied (default: 0.1).
    sigma : np.ndarray, optional
        Explicit per-detector measurement uncertainties ``(m,)``.  When
        given, overrides ``relative_uncertainty``.
    auto_smooth : bool, optional
        Nonlinear regression mode of LOUHI78: adjust the smoothing
        weight by a golden-section search on ``log10(lambda)`` so that
        the data chi-square reaches ``chi2_target`` (default False).
    chi2_target : float, optional
        Target data chi-square for ``auto_smooth``.  Defaults to the
        number of detectors (the expected value of the chi-square).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        ``(solution spectrum, sweeps used, converged flag)``.
    """
    A, b, sigma = _validate_inputs(A, b, sigma)
    n = A.shape[1]
    if x0 is None:
        x0 = np.ones(n)
    x0 = np.maximum(np.asarray(x0, dtype=float), 0.0)
    if x0.shape[0] != n:
        raise ValueError(
            f"Default spectrum length {x0.shape[0]} does not match the "
            f"number of energy bins {n}"
        )
    if smoothness < 0:
        raise ValueError(f"smoothness must be non-negative, got {smoothness}")
    L = louhi_smoothing_matrix(n, smooth_order)

    def _solve_with(lam: float) -> np.ndarray:
        H, g = _weighted_normal_equations(A, b, sigma, x0, L, lam)
        x, _, _ = _hildreth_qp(H, g, x0, max_iterations, tolerance)
        return x

    lambda_used = float(smoothness)
    target = float(chi2_target) if chi2_target is not None else float(A.shape[0])

    def _misfit(log_lam: float) -> float:
        lam = 10.0**log_lam
        chi2 = _data_chi2(A, b, sigma, _solve_with(lam))
        return abs(chi2 - target)

    if auto_smooth and max_iterations > 0:
        lo, hi = _smoothness_bracket(
            lambda t: _data_chi2(A, b, sigma, _solve_with(10.0**t)) - target
        )
        best_log_lam, _ = golden_section_minimize(_misfit, lo, hi)
        lambda_used = 10.0**best_log_lam

    x, sweeps, converged = _hildreth_qp(
        *_weighted_normal_equations(A, b, sigma, x0, L, lambda_used),
        x0,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return x, sweeps, converged


def louhi_covariance(
    A: np.ndarray,
    sigma: np.ndarray,
    smoothness: float,
    smooth_order: int,
    x0: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Propagate measurement uncertainties for the LOUHI solution.

    Follows the statistical error analysis of LOUHI78: the covariance of
    the free (strictly positive) spectrum bins is the inverse of the
    Hessian restricted to the active set, while constrained bins at zero
    carry no variance in the first-order propagation.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    sigma : np.ndarray
        Per-detector measurement uncertainties ``(m,)``.
    smoothness : float
        Smoothing weight used for the solution.
    smooth_order : int
        Order of the smoothing operator used for the solution.
    x0 : np.ndarray
        Default spectrum used in the smoothing term.
    x : np.ndarray
        LOUHI solution spectrum.

    Returns
    -------
    np.ndarray
        Standard deviations of the spectrum bins ``(n,)``.
    """
    A = np.asarray(A, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    n = A.shape[1]
    L = louhi_smoothing_matrix(n, smooth_order)
    H, _ = _weighted_normal_equations(
        A, np.ones(A.shape[0]), sigma, np.asarray(x0, dtype=float), L, smoothness
    )
    free = np.asarray(x, dtype=float) > 0
    cov = np.zeros((n, n))
    if np.any(free):
        H_free = H[np.ix_(free, free)]
        try:
            cov_free = np.linalg.inv(H_free)
        except np.linalg.LinAlgError:
            cov_free = np.linalg.pinv(H_free)
        cov[np.ix_(free, free)] = cov_free
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def unfold_louhi(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    smoothness: float = 1.0,
    smooth_order: int = 1,
    auto_smooth: bool = False,
    chi2_target: float | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    relative_uncertainty: float = 0.1,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    variance_reduction: str = "none",
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using the LOUHI78 algorithm.

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
        Default (a-priori) spectrum.  If None, a flat spectrum is used.
    smoothness : float, optional
        Smoothing weight ``lambda`` (default: 1.0).
    smooth_order : int, optional
        Smoothing operator order 0/1/2 (default: 1, first differences).
    auto_smooth : bool, optional
        Adjust the smoothing weight automatically to reach
        ``chi2_target`` (default: False).
    chi2_target : Optional[float], optional
        Target data chi-square for ``auto_smooth`` (default: number of
        detectors).
    max_iterations : int, optional
        Maximum number of Hildreth sweeps (default: 500).
    tolerance : float, optional
        Relative objective change per sweep for convergence
        (default: 1e-6).
    relative_uncertainty : float, optional
        Relative measurement uncertainty (default: 0.1).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        MC variance reduction: 'none', 'antithetic', 'control', 'both'
        (default: 'none').
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    default_initial = np.ones(n_energy_bins)

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
        default_initial=default_initial,
        solve_func=make_solve_wrapper(
            solve_louhi,
            smoothness=smoothness,
            smooth_order=smooth_order,
            auto_smooth=auto_smooth,
            chi2_target=chi2_target,
            max_iterations=max_iterations,
            tolerance=tolerance,
            relative_uncertainty=relative_uncertainty,
        ),
        solve_kwargs={},
        method_name="LOUHI",
        extra_output={
            "smoothness": float(smoothness),
            "smooth_order": int(smooth_order),
            "auto_smooth": bool(auto_smooth),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        variance_reduction=variance_reduction,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
