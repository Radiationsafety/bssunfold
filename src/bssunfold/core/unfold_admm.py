"""ADMM unfolding method for neutron spectra.

Alternating Direction Method of Multipliers (Gabay & Mercier, 1976;
Boyd et al., "Distributed Optimization and Statistical Learning via the
Alternating Direction Method of Multipliers", 2011 вЂ” lecture 11 of the
MIPT optimization course, homework 18).

Solves the constrained regularized unfolding problem

    min_x  1/2 ||A x - b||^2
           + l1_penalty * ||x||_1
           + tv_penalty * ||D x||_1
    s.t.   x >= 0

by the *consensus splitting*

    min_{x, z1, z2}  1/2 ||A x - b||^2
                     + l1_penalty ||z1||_1 + tv_penalty ||D z2||_1
                     s.t.  x = z1,  D x = z2,  x >= 0,

whose ADMM iterations are fully decoupled:

- **x-update**: non-negative least squares of the augmented system
  ``[A; sqrt(rho) I; sqrt(rho) D] x ~ [b; sqrt(rho)(z1 - u1);
  sqrt(rho)(z2 - u2)]`` solved exactly by Lawson-Hanson NNLS, so the
  non-negativity constraint is enforced *exactly* at every iteration;
- **z1-update**: element-wise soft thresholding (exact prox of the L1 norm);
- **z2-update**: element-wise soft thresholding of the difference vector
  ``D x + u2`` (exact prox of the 1D total-variation seminorm composed with
  D through the difference splitting вЂ” no inner TV solver needed);
- **u-updates**: scaled dual ascent.

The penalty ``rho`` is adaptively tuned every ``adapt_every`` iterations
following Boyd et al., section 3.4.1, which makes the method robust to the
wide dynamic range of Bonner-sphere count data without hand tuning.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_admm", "soft_threshold", "unfold_admm"]


def soft_threshold(v: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Element-wise soft-thresholding operator (prox of the L1 norm).

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    threshold : float
        Threshold ``t >= 0``; the prox is ``sign(v) * max(|v| - t, 0)``.

    Returns
    -------
    np.ndarray
        Soft-thresholded vector.
    """
    if threshold <= 0:
        return np.asarray(v, dtype=float).copy()
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def _difference_matrix(n: int) -> NDArray[np.float64]:
    """First-order difference operator D (``(D x)_i = x_{i+1} - x_i``)."""
    D = np.zeros((max(n - 1, 0), n))
    idx = np.arange(n - 1)
    D[idx, idx] = -1.0
    D[idx, idx + 1] = 1.0
    return D


def solve_admm(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    l1_penalty: float = 0.0,
    tv_penalty: float = 0.0,
    rho: float | None = None,
    adaptive_rho: bool = True,
    abstol: float = 1e-10,
    reltol: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem by consensus ADMM.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,); projected onto the nonnegative orthant.
    max_iterations : int, optional
        Maximum outer iterations (default: 500).
    tolerance : float, optional
        Relative change tolerance used as an additional stop test
        (default: 1e-6).
    l1_penalty : float, optional
        Weight of the L1 (sparsity) penalty on the spectrum (default: 0.0).
    tv_penalty : float, optional
        Weight of the total-variation penalty ``||D x||_1`` (default: 0.0).
    rho : float, optional
        ADMM penalty parameter.  If None (default), initialized to
        ``||b|| / max(||A||, eps)`` and adapted automatically when
        ``adaptive_rho`` is True.
    adaptive_rho : bool, optional
        Adapt ``rho`` every 10 iterations based on the primal/dual residual
        ratio (default: True).
    abstol : float, optional
        Absolute residual tolerance for the Boyd stopping rule
        (default: 1e-10).
    reltol : float, optional
        Relative residual tolerance for the Boyd stopping rule
        (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    m, n = A.shape

    l1_penalty = max(float(l1_penalty), 0.0)
    tv_penalty = max(float(tv_penalty), 0.0)

    x = np.maximum(x0, 0.0)

    if l1_penalty == 0.0 and tv_penalty == 0.0:
        # Degenerate ADMM: consensus with no nonsmooth terms reduces to a
        # single (augmented) NNLS solve; perform it directly.
        x_opt, _ = nnls(A, b)
        return x_opt, 1, True

    D = _difference_matrix(n)

    if rho is None:
        scale = max(float(np.linalg.norm(b)) / max(np.linalg.norm(A), 1e-30), 1e-12)
        rho = scale * max(float(np.mean(A**2)), 1e-12)
    rho = float(max(rho, 1e-12))

    # Fixed augmented design; only the RHS changes across iterations.
    blocks = [A, np.sqrt(rho) * np.eye(n)]
    if tv_penalty > 0:
        blocks.append(np.sqrt(rho) * D)
    M = np.vstack(blocks)
    M_norm = 1.0  # NNLS handles scaling; no need for normalization

    z1 = x.copy()
    z2 = D @ x
    u1 = np.zeros(n)
    u2 = np.zeros(D.shape[0])

    converged = False
    iterations = 0

    for k in range(max_iterations):
        z1_prev, z2_prev = z1.copy(), z2.copy()

        # ---- x-update: exact NNLS on the augmented system -----------------
        rhs = np.concatenate(
            [
                b,
                np.sqrt(rho) * (z1 - u1),
            ]
            + ([np.sqrt(rho) * (z2 - u2)] if tv_penalty > 0 else [])
        )
        x_new, _ = nnls(M * M_norm, rhs * M_norm, maxiter=10 * n)
        x = x_new

        # ---- z-updates: exact proximal operators --------------------------
        z1 = soft_threshold(x + u1, l1_penalty / rho)
        if tv_penalty > 0:
            z2 = soft_threshold(D @ x + u2, tv_penalty / rho)

        # ---- dual updates --------------------------------------------------
        u1 = u1 + x - z1
        if tv_penalty > 0:
            u2 = u2 + D @ x - z2

        iterations = k + 1

        # ---- Boyd primal/dual residual stopping rule -----------------------
        p = float(np.linalg.norm(x - z1))
        if tv_penalty > 0:
            p = float(np.hypot(p, np.linalg.norm(D @ x - z2)))
        dz1 = z1 - z1_prev
        dz2 = z2 - z2_prev
        s = float(np.linalg.norm(dz1))
        if tv_penalty > 0:
            s = float(np.hypot(s, np.linalg.norm(D.T @ dz2)))
        s *= rho

        n_pri = max(
            float(np.linalg.norm(x)),
            float(np.linalg.norm(z1)),
            float(np.linalg.norm(D @ x)) if tv_penalty > 0 else 0.0,
        )
        n_dual = float(np.linalg.norm(u1)) + (
            float(np.linalg.norm(D.T @ u2)) if tv_penalty > 0 else 0.0
        )
        eps_pri = np.sqrt(M.shape[0]) * abstol + reltol * max(n_pri, 1e-30)
        eps_dual = n * abstol + reltol * max(n_dual, 1e-30) * rho

        if p <= eps_pri and s <= eps_dual:
            converged = True
            break

        # ---- adaptive rho (Boyd et al., sec. 3.4.1) ------------------------
        if adaptive_rho and (k + 1) % 10 == 0:
            if p > 10.0 * s:
                rho_new = 2.0 * rho
            elif s > 10.0 * p:
                rho_new = 0.5 * rho
            else:
                rho_new = rho
            if rho_new != rho:
                # Rescale duals to keep the algorithm state consistent with
                # the new penalty (equivalent to u <- (rho/rho_new) u).
                factor = rho / rho_new
                u1 *= factor
                u2 *= factor
                rho = rho_new
                blocks = [A, np.sqrt(rho) * np.eye(n)]
                if tv_penalty > 0:
                    blocks.append(np.sqrt(rho) * D)
                M = np.vstack(blocks)

    return x, iterations, converged


def unfold_admm(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    l1_penalty: float = 0.0,
    tv_penalty: float = 0.0,
    rho: float | None = None,
    adaptive_rho: bool = True,
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
    """Unfold neutron spectrum using consensus ADMM.

    Supports exact L1 (sparsity) and 1D total-variation penalties together
    with the hard non-negativity constraint; the x-subproblem is solved
    exactly by NNLS, so the physical constraint holds at every iteration.

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
    max_iterations : int, optional
        Maximum outer iterations (default: 500).
    tolerance : float, optional
        Relative change tolerance (default: 1e-6).
    l1_penalty : float, optional
        L1 (sparsity) penalty weight (default: 0.0).
    tv_penalty : float, optional
        Total-variation penalty weight ``||D x||_1`` (default: 0.0).
    rho : Optional[float], optional
        ADMM penalty parameter; adapted automatically when None.
    adaptive_rho : bool, optional
        Adapt rho every 10 iterations (default: True).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    variance_reduction : str, optional
        Monte-Carlo variance reduction: ``'none'``, ``'antithetic'``,
        ``'control'`` or ``'both'`` (default: 'none').
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
            solve_admm,
            max_iterations=max_iterations,
            tolerance=tolerance,
            l1_penalty=l1_penalty,
            tv_penalty=tv_penalty,
            rho=rho,
            adaptive_rho=adaptive_rho,
        ),
        solve_kwargs={},
        method_name="ADMM",
        extra_output={
            "l1_penalty": l1_penalty,
            "tv_penalty": tv_penalty,
            "adaptive_rho": adaptive_rho,
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
