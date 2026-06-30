"""Statistical Regularization (Turchin/Vapnik) unfolding — RECONST Fortran port.

Pure numpy implementation of the STREG1 algorithm from RECONST.FOR.
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
    """Build the 5-diagonal smoothing matrix Omega (OMO).

    Fortran: STREG1 lines 545-561.
    Returns shape (5, n).
    """
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


def _invert_matrix(D: np.ndarray) -> None:
    """In-place partitioned (Banachiewicz) matrix inversion.

    Fortran: INVERS (lines 702-728).
    """
    n = D.shape[0]
    nm = n - 1

    for _ in range(n):
        if D[0, 0] == 0.0:
            raise np.linalg.LinAlgError("D(1,1)=0 in matrix inversion")
        p1 = 1.0 / D[0, 0]
        V1 = D[0, 1:].copy()

        for i in range(nm):
            D[i, -1] = -V1[i] * p1
            y1 = D[i, -1]
            for j in range(i, nm):
                D[i, j] = D[i + 1, j + 1] + V1[j] * y1

        D[-1, -1] = -p1

    for i in range(n):
        for j in range(i, n):
            val = -D[i, j]
            D[i, j] = val
            D[j, i] = val


def _build_system_matrix(
    B: np.ndarray, OMO: np.ndarray, n: int, alpha: float, beta: float
) -> np.ndarray:
    """Build system matrix D = B * beta + Omega * alpha.

    Fortran: REG1 lines 674-689 (without inversion).
    """
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = B[i, j] * beta

    for i in range(n):
        k = i + 2
        if i == n - 1:
            k = i
        elif i == n - 2:
            k = i + 1

        for j in range(i, k + 1):
            jj = i - j
            D[i, j] += OMO[jj + 2, j] * alpha
            if i != j:
                D[j, i] = D[i, j]

    return D


def _reg1(
    B: np.ndarray,
    OMO: np.ndarray,
    A_vec: np.ndarray,
    n: int,
    alpha: float,
    beta: float,
    ich: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build system, invert if ich > 0, return D, FI, SIGMA.

    Fortran: REG1 (lines 669-700).

    Returns
    -------
    D : ndarray
        System matrix (if ich <= 0) or its inverse (if ich > 0).
    FI : ndarray or None
        Solution vector (None if ich <= 0).
    SIGMA : ndarray or None
        Uncertainties (None if ich <= 0).
    """
    D = _build_system_matrix(B, OMO, n, alpha, beta)

    if ich > 0:
        D_inv = D.copy()
        try:
            _invert_matrix(D_inv)
        except np.linalg.LinAlgError:
            D_reg = D.copy()
            D_reg.flat[:: n + 1] += 1e-8
            try:
                _invert_matrix(D_reg)
                D_inv = D_reg
            except np.linalg.LinAlgError:
                D_reg = D + np.eye(n) * 1e-4
                D_inv = np.linalg.inv(D_reg)
        FI = D_inv @ A_vec * beta
        SIGMA = np.sqrt(np.abs(np.diag(D_inv)))
        return D_inv, FI, SIGMA

    return D, None, None


def _compute_omega(
    OMO: np.ndarray, D_inv: np.ndarray, FI: np.ndarray, n: int, alpha: float
) -> float:
    """Compute omega(alpha) functional for alpha selection.

    Fortran: OM1 (lines 730-764).
    D_inv is the inverse of the system matrix (B * beta + Omega * alpha)^-1.
    """
    omega_val = 0.0
    for i_f in range(1, n + 1):
        i = i_f - 1

        if i_f <= 2:
            K = 3
        elif i_f == 3:
            K = 2
        else:
            K = 1

        if i_f <= n - 2:
            J = 5
        elif i_f == n - 1:
            J = 4
        else:
            J = 3

        for kj in range(K, J + 1):
            jj = i_f + kj
            if kj <= 3:
                omo_val = OMO[kj - 1, i]
            elif kj == 4:
                omo_val = OMO[1, i + 1]
            else:
                omo_val = OMO[0, i + 2]
            omega_val += omo_val * D_inv[jj - 4, i]

        omega_val += OMO[2, i] * FI[i] ** 2

        if i_f <= n - 2:
            omega_val += 2.0 * FI[i] * (
                OMO[1, i + 1] * FI[i + 1] + OMO[0, i + 2] * FI[i + 2]
            )
        elif i_f == n - 1:
            omega_val += 2.0 * FI[n - 2] * FI[n - 1] * OMO[1, n - 1]

    omega_val = float(n) / alpha - omega_val
    return omega_val


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
    """Compute discrepancy delta(beta) for beta selection.

    Fortran: DELT1 (lines 794-818).
    D_inv is the inverse of the system matrix.
    """
    delta = 0.0
    V = 0.0
    W = 0.0
    DV = 0.0

    for i in range(n):
        RT_i = 0.0
        T_i = 0.0
        for j in range(n):
            RT_i += B[i, j] * D_inv[j, i]
            T_i += B[i, j] * FI[i] * FI[j]
        delta += RT_i
        V += T_i
        W += A_vec[i] * FI[i]

    for i in range(m):
        DV += (F[i] / S[i]) ** 2

    delta = delta + V - 2.0 * W + DV
    delta = float(m) / beta - delta
    return delta


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
    """Find optimal alpha where omega(alpha) = 0.

    Fortran: DEFALF (lines 640-667).
    """
    alm = 4.0 ** (1.0 if omega_init >= 0 else -1.0)
    als = omega_init

    for _ in range(100):
        alpha *= alm
        _, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        omega = _compute_omega(OMO, D_inv, FI, n, alpha)
        if omega * als <= 0:
            break

    aln = (alpha + alpha / alm) / 5.0
    alk = 4.0 * aln

    for _ in range(200):
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
    """Find optimal beta where delta(beta) = 0.

    Fortran: DEFBET (lines 766-792).
    """
    betm = 4.0 ** (1.0 if delta_init >= 0 else -1.0)
    bets = delta_init

    for _ in range(100):
        beta *= betm
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        delta = _compute_delta(B, D_inv, FI, A_vec, F, S, n, m, beta)
        if delta * bets <= 0:
            break

    betn = (beta + beta / betm) / 5.0
    betk = 4.0 * betn

    for _ in range(200):
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
    """Core STREG1 algorithm.

    Fortran: STREG1 (lines 519-638).
    """
    sa = np.exp(np.mean(np.log(np.maximum(S, 1e-300))))
    S_norm = S / sa

    B = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1):
            s_val = 0.0
            for j in range(m):
                s_val += AK[j, i] * AK[j, k] / S_norm[j] ** 2
            B[i, k] = s_val
            if k != i:
                B[k, i] = s_val

    A_vec = np.zeros(n)
    for i in range(n):
        s_val = 0.0
        for j in range(m):
            s_val += AK[j, i] * F[j] / S_norm[j] ** 2
        A_vec[i] = s_val

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
            for _ in range(50):
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
