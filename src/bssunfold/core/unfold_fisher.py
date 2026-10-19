"""Minimum Fisher Information (MFI) unfolding.

This module implements spectrum recovery by the **method of minimum
Fisher information**, one of the four reconstruction techniques applied
by Miksuta-Michalik et al., *A feasibility study of neutron spectrum
reconstruction using synthetic activation data for JT-60SA*,
Plasma Phys. Control. Fusion **68** (2026) 105001
(doi:10.1088/1361-6587/acaab2), alongside Tikhonov regularization,
Maximum Entropy and Maximum Likelihood.

Mathematical formulation
------------------------
Given the discrete activation system ``G f = A`` (the paper's eq. 1,
our ``A x = b``), the regularized scheme of the paper's eq. (3)
minimizes a weighted data-misfit plus a solution-dependent penalty.
MFI is obtained by choosing, in eq. (4), the **Fisher information
matrix** as the regularization operator:

    L = Bᵀ W B,

where ``B`` is the first- or second-derivative difference operator and
``W`` is the diagonal weight matrix defined recursively by the current
solution (paper's eqs. (5)-(6)):

    W_ii = 1 / f_i        for f_i > 0,
    W_ii = W_max          for f_i <= 0.

The Fisher information functional ∫ (f′)²/f dE discretizes to the
quadratic form ``fᵀ Bᵀ W B f = Σ_i (Δf_i)² / f_i`` — the penalty
diverges whenever flux is moved through a nearly-empty bin, which is
what makes MFI sharp at spectral edges (the paper's eq. (3) writes the
penalty as ``‖L f‖₂`` with ``L = Bᵀ W B``; the literal reading would
give a quartic form in ``f`` and is not the standard Fisher functional
— we implement the canonical quadratic form, which is the reading used
throughout the tomography literature, cf. Ødstul et al. [4]).

Because ``W`` depends on the solution, the method is **iterative**
(paper, after eq. (6)): start from an initial guess, solve the linear
regularized system for the current weights, rebuild the weights from
the new solution, and repeat until the relative change of the solution
drops below ``tolerance`` or ``max_iterations`` is reached. Each
inner solve is the normal-equation system

    (Aᵀ A + alpha · Bᵀ W B) x = Aᵀ b,

i.e. exactly the paper's eq. (3) with ``λ² = alpha`` and the Fisher
penalty in quadratic form. The derivative operator ``B`` is built by
:func:`bssunfold.core._matrix_utils.create_derivative_matrix`, with
``ln E``-grid-aware rows when the energy grid is supplied.

Properties
----------
* ``alpha = 0`` reduces the method to the plain least-squares solve
  (the penalty disappears; the iteration converges in one step).
* The weight cap ``W_max`` (paper's eq. (6)) is required for
  positivity: bins whose current flux is zero or negative would
  otherwise receive infinite weight. When ``w_max=None`` the cap is
  set adaptively to ``1e-4 · max(f)`` per iteration (flux floor).
* Like the paper's implementation, the method needs a **positive
  initial guess**; the default is a flat spectrum.
* The penalty is scale-sensitive: ``W ~ 1/f`` carries the units of the
  flux, so the meaningful range of ``alpha`` depends on the units of
  the readings. Tune ``alpha`` per data set (the notebook sweeps it).

References
----------
[1] Miksuta-Michalik K., Cubi A., Fabbrì M., Di Giacomo M.
    *A feasibility study of neutron spectrum reconstruction using
    synthetic activation data for JT-60SA.*
    Plasma Phys. Control. Fusion 68 (2026) 105001.
[2] Williamson J.H., et al. *Use of minimum Fisher's information
    criteria to resolve ill-posed tomography problems.*
    IEEE Trans. Med. Imaging (1997); the MFI criterion for
    tomographic inversion.
[3] Reginato T., Goldhagen P., Neumann S. (2002) — MFI context in
    neutron spectrometry unfolding (IAEA NDS-related work).
[4] Odstul H., Mynar J., Odstul T., Alper B., Murari A. and
    Contributors J.E. *Modern numerical methods for plasma tomography
    optimization.* Nucl. Instrum. Methods Phys. Res. A 686 (2012)
    156-161.
"""

from typing import Any

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_fisher_information", "unfold_fisher_information"]


def solve_fisher_information(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    alpha: float = 0.01,
    derivative_order: int = 1,
    E_MeV: np.ndarray | None = None,
    w_max: float | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    damping: float = 1.0,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding by iterative minimum-Fisher-information regularization.

    Minimizes the MFI functional (paper eqs. (3)-(6))

        ||A x - b||^2 + alpha * x^T B^T W B x,
        W_jj = 1 / max(f̄_j, floor),   floor = 1 / w_max,

    where ``f̄_j`` is the mean flux over the bin stencil of the j-th
    finite difference (row of ``B``) — the discretization of the
    Fisher weight ``1/f`` at the location of the difference.
    Solved by fixed-point iteration on the weights ``W``: each step solves the
    linear regularized normal-equation system

        (A^T A + alpha * B^T W B) x = A^T b

    for the current weights, clips the solution to be non-negative,
    optionally damps the update, and rebuilds the weights. The
    iteration stops when the relative change ``||x_new - x|| / ||x||``
    falls below ``tolerance`` or after ``max_iterations`` steps.

    Parameters
    ----------
    A : np.ndarray
        Response matrix of shape ``(m, n)``.
    b : np.ndarray
        Measurement vector of shape ``(m,)``.
    x0 : np.ndarray, optional
        Positive initial guess of shape ``(n,)``. Zero or negative
        components are treated per the paper's eq. (6) (weight capped
        at ``w_max``). If ``None`` or all-zero, a flat unit spectrum
        is used.
    alpha : float, optional
        Regularization strength ``lambda^2`` (default: ``0.01``).
        Must be non-negative. ``alpha = 0`` gives the plain
        least-squares solution.
    derivative_order : {1, 2}, optional
        Order of the difference operator ``B`` (default: ``1``).
    E_MeV : np.ndarray, optional
        Energy grid (MeV) of length ``n``. When given, ``B``
        approximates derivatives w.r.t. ``ln E`` on the (possibly
        non-uniform) log grid (grid-aware penalty); otherwise plain
        unit-spaced finite differences are used.
    w_max : float, optional
        Upper bound of the Fisher weights (paper's eq. (6),
        ``W_max``). When ``None`` (default), the cap is set
        adaptively each iteration to ``1 / (1e-4 * max(x))`` — i.e.
        bins below 1e-4 of the peak flux are floored.
    max_iterations : int, optional
        Maximum number of weight-update iterations (default: ``50``).
        Must be >= 1.
    tolerance : float, optional
        Relative-change convergence threshold (default: ``1e-6``).
        Must be positive.
    damping : float, optional
        Relaxation factor of the solution update,
        ``x <- (1 - damping) * x + damping * x_new`` (default:
        ``1.0``, no damping). Must satisfy ``0 < damping <= 1``.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, n_iterations, converged)`` — the unfolded
        spectrum of shape ``(n,)`` clipped to non-negative values,
        the number of iterations performed, and whether the relative
        change dropped below ``tolerance``.

    Raises
    ------
    ValueError
        If ``alpha`` is negative, ``derivative_order`` is not 1 or 2,
        ``w_max`` is non-positive, ``max_iterations`` < 1,
        ``tolerance`` <= 0, or ``damping`` is outside ``(0, 1]``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}")
    if derivative_order not in (1, 2):
        raise ValueError(
            f"derivative_order must be 1 or 2, got {derivative_order!r}"
        )
    if w_max is not None and w_max <= 0:
        raise ValueError(f"w_max must be positive or None, got {w_max!r}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    if not 0 < damping <= 1:
        raise ValueError(f"damping must satisfy 0 < damping <= 1, got {damping!r}")

    m, n = A.shape

    if alpha == 0.0:
        # No penalty: the MFI fixed point is the plain least-squares
        # solution (minimum-norm for rank-deficient A), clipped to be
        # non-negative. Solve it directly instead of through the
        # (singular) normal-equation system.
        x_ls = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0.0)
        return x_ls, 1, True

    if x0 is None:
        x = np.ones(n)
    else:
        x = np.maximum(np.asarray(x0, dtype=float).ravel(), 0.0)
        if not np.any(x > 0):
            x = np.ones(n)

    B = create_derivative_matrix(n, derivative_order, E_MeV)
    B = np.asarray(B.todense(), dtype=float)
    AtA = A.T @ A
    Atb = A.T @ b
    # Each row of B is one finite difference spanning a few bins; the
    # Fisher weight of that row is 1/f evaluated on the difference,
    # discretized as the mean flux over the row's stencil.
    absB = np.abs(B)
    stencil = absB.sum(axis=1)
    row_flux = (absB @ x) / np.where(stencil > 0, stencil, 1.0)

    converged = False
    n_iter = 0
    for k in range(max_iterations):
        n_iter = k + 1
        # Fisher weights, paper eqs. (5)-(6): w = 1/f for f > 0, capped
        # at w_max for f <= 0 (and, adaptively, for tiny f).
        if w_max is None:
            peak = float(np.max(x))
            floor = 1e-4 * peak if peak > 0 else 1e-4
            w = 1.0 / np.maximum(row_flux, floor)
        else:
            w = np.where(row_flux > 0, 1.0 / np.maximum(row_flux, 1e-300), w_max)
            w = np.minimum(w, w_max)

        K = AtA + alpha * (B.T @ (w[:, None] * B))
        try:
            x_new = np.linalg.solve(K, Atb)
        except np.linalg.LinAlgError:
            x_new = np.linalg.lstsq(K, Atb, rcond=None)[0]
        x_new = np.maximum(x_new, 0.0)

        change = np.linalg.norm(x_new - x)
        denom = np.linalg.norm(x)
        rel = change / denom if denom > 0 else change
        x = (1.0 - damping) * x + damping * x_new
        row_flux = (absB @ x) / np.where(stencil > 0, stencil, 1.0)
        if rel < tolerance:
            converged = True
            break

    return np.maximum(x, 0.0), n_iter, converged


def unfold_fisher_information(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    alpha: float = 0.01,
    derivative_order: int = 1,
    w_max: float | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    damping: float = 1.0,
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
    """Unfold neutron spectrum using minimum Fisher information (MFI).

    Implements the MFI method of Miksuta-Michalik et al. (2026),
    eqs. (3)-(6): Tikhonov-form regularization whose operator is the
    Fisher information matrix ``L = B^T W B`` with solution-dependent
    weights ``W = diag(1/f_i)`` (capped at ``W_max``), solved by
    iterative weight updates from a positive initial guess. See
    :func:`solve_fisher_information` for the precise algorithm and
    parameter semantics.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins on the detector grid.
    E_MeV : np.ndarray
        Energy grid (MeV); used both for the system and for the
        ``ln E``-aware derivative operator.
    sensitivities : Dict[str, np.ndarray]
        Per-detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 ambient-dose-equivalent conversion coefficients.
    save_result_callback : callable
        Callback to save the result to the detector history.
    readings : Dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Per-bin ``d(ln E)`` widths for dose integration.
    initial_spectrum : np.ndarray, optional
        Positive initial guess (default: flat unit spectrum). MFI is
        iterative and requires a starting spectrum, as in the paper.
    alpha : float, optional
        Regularization strength ``lambda^2`` (default: ``0.01``).
    derivative_order : {1, 2}, optional
        Order of the difference operator ``B`` (default: ``1``).
    w_max : float, optional
        Upper bound of the Fisher weights (default: ``None`` —
        adaptive cap at 1e-4 of the current peak flux).
    max_iterations : int, optional
        Maximum weight-update iterations (default: ``50``).
    tolerance : float, optional
        Relative-change convergence threshold (default: ``1e-6``).
    damping : float, optional
        Relaxation factor in ``(0, 1]`` (default: ``1.0``).
    calculate_errors : bool, optional
        Monte-Carlo uncertainty propagation (default: ``False``).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: ``0.01``).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: ``100``).
    save_result : bool, optional
        Save result to detector history (default: ``False``).
    random_state : int, optional
        Random seed for reproducible Monte-Carlo.
    reading_uncertainties : dict or np.ndarray, optional
        Per-detector reading uncertainties.
    reading_covariance : np.ndarray, optional
        Full reading covariance matrix.
    noise_model : str, optional
        Monte-Carlo noise model (default: ``"gaussian"``).
    measurement_time : float, optional
        Measurement time for Poisson noise scaling.

    Returns
    -------
    Dict[str, Any]
        Standard unfolding result dict with extra keys ``alpha``,
        ``derivative_order``, ``w_max``, ``tolerance``, ``damping``
        and, from the solver, ``iterations`` and ``converged``.
    """
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
        default_initial=np.ones(n_energy_bins),
        solve_func=make_solve_wrapper(
            solve_fisher_information,
            alpha=alpha,
            derivative_order=derivative_order,
            E_MeV=np.asarray(E_MeV, dtype=float),
            w_max=w_max,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        ),
        solve_kwargs={},
        method_name="Fisher information (MFI)",
        extra_output={
            "alpha": alpha,
            "derivative_order": derivative_order,
            "w_max": w_max,
            "tolerance": tolerance,
            "damping": damping,
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
