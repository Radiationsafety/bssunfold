"""NNQP-based unfolding method for neutron spectrum reconstruction.

This module provides a Bonner-sphere-spectrum unfolding method built on the
NNQP (Non-Negative Quadratic Programming) solver of Giovannucci & Pehlevan
(https://github.com/simonsfoundation/NNQP).

NNQP solves, by coordinate descent, the convex program

    minimize    1/2 В· xбµЂ Q x + fбµЂ x
    subject to  x в‰Ґ 0

For BSS unfolding we recast the regularised least-squares problem as NNQP:

    minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
    subject to  x в‰Ґ 0

which has the QP form above with

    Q = AбµЂ A + О± В· LбµЂ L + О±0 В· I            (positive-definite)
    f = в€’ AбµЂ b

The coordinate-descent update for coordinate ``i`` is

    x_i в†ђ max( 0,  в€’(M_i В· x + f_i) / Q_ii )      with M = Q в€’ diag(Q)

which is the classic NNQP update (Giovannucci & Pehlevan, 2016).  The
diagonal entries ``Q_ii`` are always strictly positive because ``Q`` is the
sum of two positive-semidefinite matrices plus ``О±0 В· I``, so the
regularisation floor ``О±0`` (default ``1e-6``) guarantees strict
positive-definiteness even when ``AбµЂA`` is rank-deficient вЂ” which is the
norm for BSS problems (few detectors, many energy bins).

The implementation here is a self-contained NumPy port of the original
``nnqp.py`` (which used ``numba`` for JIT acceleration); we drop the
``numba`` hard-dependency and fall back to a vectorised inner loop that is
fast enough for BSS problems (n в‰І 1000 energy bins).
"""

from typing import Any

import numpy as np

from ..logging_config import get_logger
from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_nnqp", "unfold_nnqp"]

logger = get_logger("detector")


# --------------------------------------------------------------------------- #
# Core NNQP solver                                                             #
# --------------------------------------------------------------------------- #
def _nnqp(
    Q: np.ndarray,
    f: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iterations: int = 10_000,
    random_state: int | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve ``min 0.5 xбµЂQx + fбµЂx s.t. x в‰Ґ 0`` by coordinate descent.

    Pure-NumPy port of ``nnqp.nnqp`` (Giovannucci & Pehlevan, 2016,
    https://github.com/simonsfoundation/NNQP).  The original uses ``numba``;
    we keep the same algorithm but use a vectorised inner update loop so the
    function works without ``numba`` installed.

    Parameters
    ----------
    Q : np.ndarray
        Symmetric positive-definite Hessian ``(n, n)``.
    f : np.ndarray
        Linear term ``(n,)``.
    x0 : np.ndarray, optional
        Initial guess ``(n,)``. If ``None`` a uniform ``[0, 1)`` random
        vector is used (matching the original implementation). A non-negative
        warm-start spectrum can substantially reduce the iteration count.
    tol : float, optional
        Convergence tolerance on the relative change of the active
        components (default 1e-6).
    max_iterations : int, optional
        Iteration cap (default 10 000).
    random_state : int, optional
        Random seed for the initial guess when ``x0`` is ``None``.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(solution, iterations, converged)``.
    """
    Q = np.asarray(Q, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square matrix, got shape {Q.shape}")
    n = Q.shape[0]
    if f.shape[0] != n:
        raise ValueError(
            f"f length ({f.shape[0]}) must match Q size ({n})"
        )

    # Symmetrise defensively (the algorithm relies on the symmetry of Q).
    Q = 0.5 * (Q + Q.T)

    qdg = np.diag(Q).copy()
    # The diagonal must be strictly positive for the coordinate update to be
    # well-defined. If the user passes a rank-deficient Q, we add a tiny
    # floor вЂ” but only here at the lowest level so callers can override.
    bad = qdg <= 0
    if np.any(bad):
        qdg[bad] = np.maximum(qdg[bad], 1e-12)
    Dinv = 1.0 / qdg
    M = Q - np.diag(qdg)  # off-diagonal part

    rng = np.random.default_rng(random_state)
    if x0 is not None:
        x = np.maximum(np.asarray(x0, dtype=float).ravel(), 0.0)
        if x.shape[0] != n:
            raise ValueError(
                f"x0 length ({x.shape[0]}) must match Q size ({n})"
            )
    else:
        x = rng.random(n)

    converged = False
    iterations = 0
    zero_count = 0
    for it in range(max_iterations):
        iterations = it + 1
        x_prev = x.copy()
        # Coordinate-descent sweep вЂ” update in place, picking up the latest
        # values of already-updated coordinates (Gauss-Seidel flavour).
        # Vectorised form of the per-coordinate update
        #   x_i в†ђ max(0, -(M[i,:]В·x + f_i) / Q_ii)
        # Since M has zero diagonal, M[i,:]В·x depends only on the other
        # coordinates, so a Gauss-Seidel sweep is correct.  We do it in a
        # tight Python loop because BSS problems typically have n в‰І 1000.
        for i in range(n):
            dum = Dinv[i] * (-(M[i] @ x) - f[i])
            x[i] = dum if dum > 0 else 0.0

        # Convergence test: relative change on the strictly positive
        # components of x_prev (as in the original NNQP).
        active = x_prev > 0
        if not np.any(active):
            # First iteration can land on all-zero x; give it one more
            # sweep before declaring convergence.
            if zero_count == 0:
                er = 1.0
                zero_count += 1
            else:
                er = 0.0
        else:
            denom = np.maximum(np.abs(x_prev[active]), 1e-300)
            er = float(np.max(np.abs(x[active] - x_prev[active]) / denom))

        if er <= tol:
            converged = True
            break

    return x, iterations, converged


# --------------------------------------------------------------------------- #
# BSS unfolding solver                                                         #
# --------------------------------------------------------------------------- #
def solve_nnqp(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    regularization: float = 1e-4,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    tol: float = 1e-6,
    max_iterations: int = 10_000,
    floor: float = 1e-6,
    random_state: int | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the BSS unfolding problem with NNQP (coordinate descent).

    Recasts the regularised non-negative least-squares problem

        minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
        subject to  x в‰Ґ 0

    as the NNQP ``min 0.5 xбµЂQx + fбµЂx s.t. x в‰Ґ 0`` with

        Q = AбµЂA + О± В· LбµЂL + О±0 В· I       (positive-definite)
        f = в€’ AбµЂ b

    where ``L`` is the finite-difference derivative matrix of order
    ``smoothness_order`` (0 disables the smoothness term вЂ” ``L`` is empty вЂ”
    and ``О±0`` is the diagonal regularisation floor).

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Warm-start spectrum. If ``None`` the NNQP solver uses a uniform
        random initial guess (matching the original implementation).
    regularization : float, optional
        Tikhonov / smoothness regularisation weight (default 1e-4).
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 0.
    smoothness_weight : float, optional
        Weight for the smoothness term (default 1.0).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iterations : int, optional
        Iteration cap (default 10 000).
    floor : float, optional
        Diagonal regularisation floor added to ``Q`` to guarantee strict
        positive-definiteness even when ``AбµЂA`` is rank-deficient
        (default 1e-6).
    random_state : int, optional
        Random seed for the initial guess when ``x0`` is ``None``.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]

    if smoothness_order not in (0, 1, 2):
        raise ValueError(
            f"Unsupported smoothness order: {smoothness_order}. Use 0, 1 or 2."
        )

    # Build Q = A^T A + alpha * L^T L + alpha0 * I
    AtA = A.T @ A
    Q = AtA + floor * np.eye(n)
    if smoothness_order in (1, 2) and regularization > 0:
        L = create_derivative_matrix(n, smoothness_order)
        Q = Q + regularization * smoothness_weight * (L.T @ L)

    f = -(A.T @ b)

    return _nnqp(Q, f, x0=x0, tol=tol, max_iterations=max_iterations,
                 random_state=random_state)


# --------------------------------------------------------------------------- #
# Detector-level wrapper                                                       #
# --------------------------------------------------------------------------- #
def unfold_nnqp(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    regularization: float = 1e-4,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    tol: float = 1e-6,
    max_iterations: int = 10_000,
    floor: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum using NNQP (non-negative QP by coordinate descent).

    Solves

        minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
        subject to  x в‰Ґ 0

    where ``L`` is the finite-difference derivative matrix of order
    ``smoothness_order``, using the coordinate-descent NNQP solver of
    Giovannucci & Pehlevan (https://github.com/simonsfoundation/NNQP).

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray, optional
        Warm-start spectrum. If ``None`` the NNQP solver uses a uniform
        random initial guess.
    regularization : float, optional
        Tikhonov / smoothness regularisation weight (default 1e-4).
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 0.
    smoothness_weight : float, optional
        Weight for the smoothness term (default 1.0).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iterations : int, optional
        Iteration cap (default 10 000).
    floor : float, optional
        Diagonal regularisation floor added to ``Q`` to guarantee strict
        positive-definiteness (default 1e-6).
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default 100.
    save_result : bool, optional
        Save result to history, default False.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
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
            solve_nnqp,
            regularization=regularization,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            tol=tol,
            max_iterations=max_iterations,
            floor=floor,
            random_state=random_state,
        ),
        solve_kwargs={},
        method_name="NNQP",
        extra_output={
            "regularization": regularization,
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "tol": tol,
            "max_iterations": max_iterations,
            "floor": floor,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
