"""Statistical Regularization (Turchin/Vapnik) unfolding — RECONST Fortran port.

Optimized numpy implementation of the STREG1 algorithm from RECONST.FOR.
Solves the system  (B * beta + Omega * alpha) * f = A_vec * beta
with automatic alpha/beta selection.

References
----------
Turchin, V. F., "Statistical regularization method", 1967.
RECONST.FOR — Program for neutron spectrum unfolding by statistical
regularization.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_reconst", "unfold_reconst"]

_AINF = np.array([1.01, 1.01, 0.01, 0.01, 0.0])


def _build_omo_matrix(n: int, pp: float) -> np.ndarray:
    """Build the 5-diagonal smoothing matrix Omega (OMO) in band (5, n) format."""
    XX = np.arange(1.0, n + 2.0)

    AA = np.zeros(n + 2)
    BB = np.zeros(n + 3)
    CC = np.zeros(n + 3)

    for i in range(2, n):
        AA[i] = 1.0 / (XX[i] - XX[i - 1])
        CC[i] = 1.0 / (XX[i - 1] - XX[i - 2])
        BB[i] = -(AA[i] + CC[i])

    OMO = np.zeros((5, n))
    for i in range(n):
        OMO[0, i] = AA[i] * CC[i]
        OMO[1, i] = AA[i] * BB[i] + BB[i + 1] * CC[i + 1]
        OMO[2, i] = AA[i] ** 2 + BB[i + 1] ** 2 + CC[i + 2] ** 2 + pp * (XX[i + 1] - XX[i])

    return OMO


def _omo_to_full(OMO: np.ndarray, n: int) -> np.ndarray:
    """Convert (5, n) band representation to full (n, n) symmetric matrix."""
    Omega = np.zeros((n, n))
    for i in range(n):
        Omega[i, i] = OMO[2, i]
        if i > 0:
            Omega[i, i - 1] = OMO[1, i]
        if i > 1:
            Omega[i, i - 2] = OMO[0, i]
        if i < n - 1:
            Omega[i, i + 1] = OMO[1, i + 1]
        if i < n - 2:
            Omega[i, i + 2] = OMO[0, i + 2]
    return Omega


def _build_system_matrix(
    B: np.ndarray, OMO: np.ndarray, n: int, alpha: float, beta: float
) -> np.ndarray:
    """Build system matrix D = B * beta + Omega * alpha."""
    Omega = _omo_to_full(OMO, n)
    return beta * B + alpha * Omega


def _invert_matrix(D: np.ndarray) -> None:
    """In-place matrix inversion (backward-compat wrapper)."""
    D[:] = _invert_system(D)


def _invert_system(D: np.ndarray) -> np.ndarray:
    """Invert system matrix with fallbacks for singular or ill-conditioned matrices."""
    n = D.shape[0]
    cond = np.linalg.cond(D)
    if cond > 1e12:
        tr = np.trace(D)
        reg = (1e-6 * tr / n) if tr > 0 else 1e-6
        D_reg = D + np.eye(n) * reg
        return np.linalg.inv(D_reg)
    try:
        return np.linalg.inv(D)
    except np.linalg.LinAlgError:
        for reg in [1e-6, 1e-4, 1e-2]:
            D_reg = D + np.eye(n) * reg
            try:
                inv = np.linalg.inv(D_reg)
                if np.all(np.isfinite(inv)):
                    return inv
            except np.linalg.LinAlgError:
                continue
        return np.linalg.pinv(D)


def _reg1(
    B: np.ndarray,
    OMO: np.ndarray,
    A_vec: np.ndarray,
    n: int,
    alpha: float,
    beta: float,
    ich: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build system, invert if ich > 0, return D_inv, FI, SIGMA."""
    D = _build_system_matrix(B, OMO, n, alpha, beta)

    if ich > 0:
        D_inv = _invert_system(D)
        FI = D_inv @ A_vec * beta
        SIGMA = np.sqrt(np.abs(np.diag(D_inv)))
        return D_inv, FI, SIGMA

    return D, None, None


def _compute_omega(
    OMO: np.ndarray, D_inv: np.ndarray, FI: np.ndarray, n: int, alpha: float
) -> float:
    """Compute omega(alpha) functional for alpha selection.

    Uses the original Fortran indexing: sums Omega[i,j] * D_inv[j,i]
    with boundary conditions matching the RECONST algorithm.
    """
    Omega = _omo_to_full(OMO, n)

    # code_trace with boundary handling matching RECONST:
    # rows 0,1: only upper triangle + diagonal
    # row 2: one subdiagonal + diagonal + upper
    # rows 3..n-3: all bands
    # row n-2: all bands except second superdiagonal
    # row n-1: only diagonal
    code_trace = 0.0
    for i in range(n):
        j_start = 0 if i >= 3 else (1 if i == 2 else i)
        j_end = i + 2 if i <= n - 3 else (i + 1 if i == n - 2 else i)
        for j in range(j_start, j_end + 1):
            if abs(i - j) <= 2:
                code_trace += Omega[i, j] * D_inv[j, i]

    fof = FI @ Omega @ FI
    return float(n) / alpha - (code_trace + fof)


def _compute_delta(
    B: np.ndarray,
    D_inv: np.ndarray,
    FI: np.ndarray,
    A_vec: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    n: int,
    m: int,
    beta: float,
) -> float:
    """Compute discrepancy delta(beta) for beta selection."""
    d1 = np.trace(B @ D_inv)
    d2 = FI @ B @ FI
    d3 = A_vec @ FI
    d4 = np.sum((F / S) ** 2)
    delta = d1 + d2 - 2.0 * d3 + d4
    return float(m) / beta - delta


def _def_alpha(
    B: np.ndarray,
    OMO: np.ndarray,
    A_vec: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    n: int,
    m: int,
    alpha: float,
    beta: float,
    omega_init: float,
    ainf: np.ndarray,
) -> float:
    """Find optimal alpha where omega(alpha) = 0."""
    alm = 4.0 ** (1.0 if omega_init >= 0 else -1.0)
    als = omega_init

    for _ in range(50):
        alpha *= alm
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        omega = _compute_omega(OMO, D_inv, FI, n, alpha)
        if omega * als <= 0:
            break

    aln = (alpha + alpha / alm) / 5.0
    alk = 4.0 * aln

    for _ in range(100):
        alpha = (aln + alk) / 2.0
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        omega = _compute_omega(OMO, D_inv, FI, n, alpha)
        if omega < 0:
            alk = alpha
        else:
            aln = alpha
        if alk <= aln * ainf[0]:
            break

    return alpha


def _def_beta(
    B: np.ndarray,
    OMO: np.ndarray,
    A_vec: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    n: int,
    m: int,
    alpha: float,
    beta: float,
    delta_init: float,
    ainf: np.ndarray,
) -> float:
    """Find optimal beta where delta(beta) = 0."""
    betm = 4.0 ** (1.0 if delta_init >= 0 else -1.0)
    bets = delta_init

    for _ in range(50):
        beta *= betm
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        delta = _compute_delta(B, D_inv, FI, A_vec, F, S, n, m, beta)
        if delta * bets <= 0:
            break

    betn = (beta + beta / betm) / 5.0
    betk = 4.0 * betn

    for _ in range(100):
        beta = (betn + betk) / 2.0
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        delta = _compute_delta(B, D_inv, FI, A_vec, F, S, n, m, beta)
        if delta < 0:
            betk = beta
        else:
            betn = beta
        if betk <= betn * ainf[1]:
            break

    return beta


def _streg1(
    AK: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    n: int,
    m: int,
    alpha: float,
    beta: float,
    pp: float,
    ainf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core STREG1 algorithm."""
    sa = np.exp(np.mean(np.log(np.maximum(S, 1e-300))))
    S_norm = S / sa

    # Vectorized construction of B = A^T * diag(1/S_norm^2) * A
    W = AK / S_norm[:, np.newaxis]
    B = W.T @ W

    # Vectorized construction of A_vec = A^T * (F / S_norm^2)
    A_vec = AK.T @ (F / S_norm ** 2)

    OMO = _build_omo_matrix(n, pp)

    if beta > 0.0:
        beta = beta / sa ** 2
        if alpha >= 0.0:
            D_inv, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        else:
            alpha = -alpha
            D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
            omega_init = _compute_omega(OMO, D_inv, FI, n, alpha)
            alpha = _def_alpha(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, omega_init, ainf)
            D_inv, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
    else:
        beta = 1.0 / sa ** 2
        if alpha >= 0.0:
            D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
            delta_init = _compute_delta(B, D_inv, FI, A_vec, F, S_norm, n, m, beta)
            beta = _def_beta(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, delta_init, ainf)
            D_inv, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        else:
            alpha = -alpha
            bet_saved = beta
            D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
            for _ in range(30):
                omega_init = _compute_omega(OMO, D_inv, FI, n, alpha)
                alpha = _def_alpha(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, omega_init, ainf)
                D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
                delta_init = _compute_delta(B, D_inv, FI, A_vec, F, S_norm, n, m, beta)
                beta = _def_beta(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, delta_init, ainf)
                if abs(bet_saved - beta) <= beta * ainf[2]:
                    break
                D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
                bet_saved = beta

            cors = 1.0 / (np.sqrt(beta) * sa)
            sa *= cors
            S_norm *= cors

            D_inv, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)

    FI = np.maximum(FI, 0)

    return FI, SIGMA


def solve_reconst(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    E_MeV: Optional[np.ndarray] = None,
    pp: float = 1e-3,
    alpha: float = -1.0,
    beta: float = 0.0,
    sigma_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve unfolding problem using Turchin's statistical regularization.

    Pure numpy implementation of the RECONST.FOR algorithm (STREG1).
    Solves  (B * beta + Omega * alpha) * f = A_vec * beta.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (M, N).
    b : np.ndarray
        Measurement vector (M,).
    x0 : np.ndarray, optional
        Ignored (API compatibility).
    E_MeV : np.ndarray, optional
        Ignored (API compatibility).
    pp : float, optional
        PP parameter (default: 1e-3).
    alpha : float, optional
        Regularization. >0 fixed, <0 auto (default: -1).
    beta : float, optional
        Data fidelity. >0 fixed, <=0 auto (default: 0).
    sigma_b : np.ndarray, optional
        Measurement uncertainties (M,). If None, sqrt(b) used.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (N,).
    """
    M, N = A.shape
    F = b.copy().astype(np.float64)

    if sigma_b is not None:
        S = np.asarray(sigma_b, dtype=np.float64)
        S = np.maximum(S, 1e-300)
    else:
        S = np.sqrt(np.maximum(F, 1e-10))

    AK = A.astype(np.float64)
    ainf = _AINF.copy()

    FI, _ = _streg1(AK, F, S, N, M, float(alpha), float(beta), float(pp), ainf)

    return np.maximum(FI, 0)


def unfold_reconst(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    pp: float = 1e-3,
    alpha: float = -1.0,
    beta: float = 0.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Turchin's statistical regularization.

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
        Ignored (API compatibility).
    pp : float, optional
        PP parameter (default: 1e-3).
    alpha : float, optional
        Regularization. <0 auto, >0 fixed (default: -1).
    beta : float, optional
        Data fidelity. 0 auto, >0 fixed (default: 0).
    calculate_errors : bool, optional
        Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result (default: True).
    random_state : int, optional
        Random seed.

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
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_reconst,
            E_MeV=E_MeV,
            pp=pp,
            alpha=alpha,
            beta=beta,
        ),
        solve_kwargs={},
        method_name="Reconst",
        extra_output={
            "pp": pp,
            "alpha": alpha,
            "beta": beta,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
