"""Tikhonov regularization with the generalized discrepancy principle.

Python port of the ``alfaFinder()`` method for the energy-non-invariant
apparatus function described in:

    I. N. Ogorodnikov, "Inverse problems of spectroscopy and spectrometry
    in applied research" (obratnye zadachi spektroskopii i
    spektrometrii v prikladnykh issledovaniyakh), Traektoriya
    Issledovaniy -- Chelovek, Priroda, Tekhnologii, no. 2 (10), pp. 42-83
    (2024).  Sections 3 and 5.

The method solves the discretized Fredholm integral equation of the
first kind ``A z = b`` by minimizing the Tikhonov smoothing functional
(article eq. 3.9)::

    M[z] = || A z - b ||^2 + alpha * || z ||_W^2,

where the Sobolev-space norm ``W_2^1`` of the article corresponds, in
its discrete form, to the squared norm of the first difference operator
``L = D1`` (the Euler equation 3.10, ``A*A z + alpha (z - z'') = A* u``,
is the continuous analogue of the normal system
``(A^T A + alpha L^T L) z = A^T b``).

The regularization parameter ``alpha`` is selected by the generalized
discrepancy principle (article eqs. 3.8-3.10): ``alpha*`` is the root of
the generalized discrepancy::

    rho(alpha) = || A z_alpha - b ||^2 - delta^2,

where ``delta`` is the RMS level of the measurement noise.  In the
article the root is found by the Newton and chord methods; here a
bracketing + Brent scheme (which combines bisection, secant/chord and
inverse quadratic interpolation) is used for robustness on log10(alpha).

Like the article's ``alfaFinder()``, the solver reports the achieved
discrepancy and diagnostic status codes when no admissible ``alpha*``
exists (data inconsistent with the supplied ``delta``).
"""

from typing import Any

import numpy as np
from scipy.optimize import brentq

from ._base_unfolder import run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = [
    "generalized_discrepancy",
    "alpha_finder_generalized_discrepancy",
    "solve_tikhonov_sobolev_dp",
    "unfold_tikhonov_sobolev_dp",
]

# Status codes (mirroring the IERR conventions of the article's FFinder).
STATUS_OK = 0  # root found by the (Brent/secant-chord) scheme
STATUS_RHO_POSITIVE = 1  # rho(alpha_min) > 0: delta too small for the data
STATUS_RHO_NEGATIVE = 2  # rho(alpha_max) < 0: delta larger than achievable


def _penalty_matrix(n: int, penalty: str) -> np.ndarray:
    """Build the discrete penalty operator ``L`` (shape (k, n))."""
    if penalty == "sobolev":
        # First difference operator: discrete Sobolev W_2^1 norm.
        return create_derivative_matrix(n, 1).toarray()
    if penalty == "curvature":
        # Second difference operator: discrete W_2^2 norm.
        return create_derivative_matrix(n, 2).toarray()
    if penalty == "identity":
        # Zeroth-order Tikhonov (standard ridge).
        return np.eye(n)
    raise ValueError(
        f"Unsupported penalty: {penalty!r}. "
        "Choose from 'sobolev', 'curvature', 'identity'."
    )


def _solve_regularized(
    N: np.ndarray, K: np.ndarray, rhs: np.ndarray, alpha: float
) -> np.ndarray:
    """Solve the normal system ``(N + alpha K) z = rhs``."""
    matrix = N + alpha * K
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def generalized_discrepancy(
    alpha: float,
    N: np.ndarray,
    K: np.ndarray,
    rhs: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    delta_sq: float,
) -> float:
    """Generalized discrepancy ``rho(alpha)`` (article eq. 3.8).

    Parameters
    ----------
    alpha : float
        Regularization parameter (must be > 0).
    N, K : np.ndarray
        Normal matrix ``A^T A`` and penalty Gram matrix ``L^T L``.
    rhs : np.ndarray
        Right-hand side ``A^T b``.
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    delta_sq : float
        Squared noise level ``delta**2``.

    Returns
    -------
    float
        ``|| A z_alpha - b ||^2 - delta^2``.
    """
    z_alpha = _solve_regularized(N, K, rhs, alpha)
    residual = A @ z_alpha - b
    return float(residual @ residual - delta_sq)


def alpha_finder_generalized_discrepancy(
    A: np.ndarray,
    b: np.ndarray,
    delta: float,
    L: np.ndarray | None = None,
    alpha_range: tuple[float, float] = (1e-10, 1e10),
    max_iter: int = 100,
    xtol: float = 1e-8,
) -> dict[str, Any]:
    """Find ``alpha*`` as the root of the generalized discrepancy.

    Implements the article's step 2 of the regularizing algorithm: the
    optimal regularization parameter ``alpha*``, consistent with the
    data error level, is computed as the root of ``rho(alpha*) = 0``.  The root
    is bracketed on a log10 grid and refined with Brent's method (a
    robust combination of the bisection, chord/secant and inverse
    quadratic methods used in the article).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    delta : float
        RMS noise level; ``alpha*`` satisfies
        ``|| A z_alpha* - b ||^2 = delta^2``.
    L : np.ndarray, optional
        Penalty operator (k x n).  Defaults to the first-difference
        (Sobolev W_2^1) operator.
    alpha_range : tuple, optional
        Search interval for ``alpha`` (default: (1e-10, 1e10)).
    max_iter : int, optional
        Maximum number of root-finder iterations (default: 100).
    xtol : float, optional
        Absolute tolerance on log10(alpha) (default: 1e-8).

    Returns
    -------
    dict
        Dictionary with keys ``alpha``, ``residual_sq``, ``rho``,
        ``status``, ``converged``, ``n_iter`` and ``min_residual_sq`` /
        ``max_residual_sq`` diagnostics.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]

    if L is None:
        L = _penalty_matrix(n, "sobolev")
    L = np.asarray(L, dtype=float)

    delta = float(delta)
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    delta_sq = delta * delta

    N = A.T @ A
    K = L.T @ L
    rhs = A.T @ b

    alpha_min, alpha_max = float(alpha_range[0]), float(alpha_range[1])
    if not (0 < alpha_min < alpha_max):
        raise ValueError(
            f"alpha_range must satisfy 0 < alpha_min < alpha_max, got {alpha_range}"
        )

    def rho_log10(t: float) -> float:
        return generalized_discrepancy(10.0**t, N, K, rhs, A, b, delta_sq)

    n_iter = 0
    rho_min = generalized_discrepancy(alpha_min, N, K, rhs, A, b, delta_sq)
    rho_max = generalized_discrepancy(alpha_max, N, K, rhs, A, b, delta_sq)
    n_iter += 2

    # Residual of the plain least-squares end (alpha -> 0): the minimal
    # achievable data misfit, used for diagnostics.
    z_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    min_residual_sq = float(np.linalg.norm(A @ z_ls - b) ** 2)

    if rho_min > 0:
        # Even the least-regularized solution misfits more than delta:
        # no admissible alpha in the range (article: rho(alpha*) > 0).
        z_alpha = _solve_regularized(N, K, rhs, alpha_min)
        return {
            "alpha": alpha_min,
            "residual_sq": float(np.linalg.norm(A @ z_alpha - b) ** 2),
            "rho": float(rho_min),
            "status": STATUS_RHO_POSITIVE,
            "converged": False,
            "n_iter": n_iter,
            "min_residual_sq": min_residual_sq,
            "max_residual_sq": float(rho_max + delta_sq),
        }

    if rho_max < 0:
        # Even the most-regularized solution fits better than delta:
        # delta is larger than any achievable misfit.
        z_alpha = _solve_regularized(N, K, rhs, alpha_max)
        return {
            "alpha": alpha_max,
            "residual_sq": float(np.linalg.norm(A @ z_alpha - b) ** 2),
            "rho": float(rho_max),
            "status": STATUS_RHO_NEGATIVE,
            "converged": False,
            "n_iter": n_iter,
            "min_residual_sq": min_residual_sq,
            "max_residual_sq": float(rho_max + delta_sq),
        }

    # rho is non-decreasing in alpha: bracket on the log10 grid and
    # refine with Brent's method (chord/secant family).
    t_min, t_max = np.log10(alpha_min), np.log10(alpha_max)
    root, results = brentq(
        rho_log10,
        t_min,
        t_max,
        xtol=xtol,
        rtol=1e-10,
        maxiter=max_iter,
        full_output=True,
    )
    n_iter += int(results.function_calls)
    alpha_star = float(10.0**root)
    z_alpha = _solve_regularized(N, K, rhs, alpha_star)
    residual_sq = float(np.linalg.norm(A @ z_alpha - b) ** 2)

    return {
        "alpha": alpha_star,
        "residual_sq": residual_sq,
        "rho": float(residual_sq - delta_sq),
        "status": STATUS_OK,
        "converged": True,
        "n_iter": n_iter,
        "min_residual_sq": min_residual_sq,
        "max_residual_sq": float(rho_max + delta_sq),
    }


def solve_tikhonov_sobolev_dp(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    noise_level: float = 0.02,
    delta: float | None = None,
    penalty: str = "sobolev",
    alpha_range: tuple[float, float] = (1e-10, 1e10),
    max_iter: int = 100,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding with Tikhonov + generalized discrepancy principle.

    Minimizes ``||A z - b||^2 + alpha ||L z||^2`` with ``alpha*`` chosen
    so that the data misfit matches the noise level
    (``||A z - b||^2 = delta^2``), following the article's
    ``alfaFinder()``.  By default ``L`` is the first-difference operator
    (discrete Sobolev ``W_2^1`` penalty of the article, Euler equation
    3.10) and ``delta = noise_level * ||b||``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess (accepted for API compatibility; not
        used by the linear regularized solver).
    noise_level : float, optional
        Relative noise level used to derive ``delta`` when it is not
        given explicitly (default: 0.02, i.e. 2 % as in the article).
    delta : float, optional
        Explicit RMS noise level; overrides ``noise_level``.
    penalty : str, optional
        Penalty operator: ``"sobolev"`` (first difference, default),
        ``"curvature"`` (second difference) or ``"identity"``.
    alpha_range : tuple, optional
        Search interval for the regularization parameter.
    max_iter : int, optional
        Maximum number of root-finder iterations.

    Returns
    -------
    tuple
        ``(spectrum, iterations, converged)``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    if delta is None:
        delta = float(noise_level) * float(np.linalg.norm(b))
    if delta <= 0:
        raise ValueError(
            f"delta must be positive, got {delta}; "
            "provide delta or a positive noise_level"
        )

    n = A.shape[1]
    L = _penalty_matrix(n, penalty)

    info = alpha_finder_generalized_discrepancy(
        A,
        b,
        delta=delta,
        L=L,
        alpha_range=alpha_range,
        max_iter=max_iter,
    )

    N = A.T @ A
    K = L.T @ L
    rhs = A.T @ b
    spectrum = _solve_regularized(N, K, rhs, info["alpha"])

    return spectrum, int(info["n_iter"]), bool(info["converged"])


def unfold_tikhonov_sobolev_dp(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    noise_level: float = 0.02,
    delta: float | None = None,
    penalty: str = "sobolev",
    alpha_range: tuple[float, float] = (1e-10, 1e10),
    max_iter: int = 100,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum with Tikhonov + discrepancy principle.

    Port of the article's ``alfaFinder()`` for the multisphere data:
    Tikhonov regularization with the discrete Sobolev ``W_2^1`` penalty
    and the regularization parameter selected by the generalized
    discrepancy principle ``||A z - b||^2 = delta^2``.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess (accepted for API compatibility).
    noise_level : float, optional
        Relative noise level used to derive ``delta``
        (default: 0.02, the article's 0-2 % range).
    delta : float, optional
        Explicit RMS noise level; overrides ``noise_level``.
    penalty : str, optional
        Penalty operator: ``"sobolev"``, ``"curvature"`` or
        ``"identity"`` (default: ``"sobolev"``).
    alpha_range : tuple, optional
        Search interval for the regularization parameter.
    max_iter : int, optional
        Maximum number of root-finder iterations (default: 100).
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty (default: False).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary with additional keys ``alpha``,
        ``delta``, ``discrepancy_status`` and ``dp_converged``.
    """
    x0_default = np.zeros(n_energy_bins)

    holder: dict[str, Any] = {}

    b_selected = np.array(
        [readings[name] for name in detector_names if name in readings],
        dtype=float,
    )

    def solve_func(A, b, **kwargs):
        spectrum, n_iter, converged = solve_tikhonov_sobolev_dp(
            A,
            b,
            noise_level=noise_level,
            delta=delta,
            penalty=penalty,
            alpha_range=alpha_range,
            max_iter=max_iter,
        )
        # Record only the clean-fit metadata (Monte-Carlo replicates
        # use perturbed readings and would overwrite them).
        if np.array_equal(np.asarray(b), b_selected):
            used_delta = delta
            if used_delta is None:
                used_delta = float(noise_level) * float(np.linalg.norm(b))
            info = alpha_finder_generalized_discrepancy(
                A,
                b,
                delta=used_delta,
                L=_penalty_matrix(A.shape[1], penalty),
                alpha_range=alpha_range,
                max_iter=max_iter,
            )
            holder["alpha"] = info["alpha"]
            holder["discrepancy_status"] = info["status"]
            holder["dp_converged"] = info["converged"]
            holder["residual_sq"] = info["residual_sq"]
        return spectrum, n_iter, converged

    result = run_unfolding(
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
        solve_func=solve_func,
        solve_kwargs={},
        method_name="TikhonovSobolevDP",
        extra_output={
            "penalty": penalty,
            "delta": float(delta) if delta is not None else None,
            "noise_level": float(noise_level),
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

    for key in ("alpha", "discrepancy_status", "dp_converged", "residual_sq"):
        if key in holder:
            result[key] = holder[key]

    return result
