"""EPIC Tikhonov regularization unfolding method.

Implements the Equal Posterior Information Condition (EPIC) Tikhonov
regularization for least squares inversion, ported from the EPIC_LS package
(https://github.com/frortega/EPIC_LS). See:

    Ortega-Culaciati, F., Simons, M., Ruiz, J., Rivera, L., & Diaz-Salazar, N.
    (2021). An EPIC Tikhonov regularization: Application to quasi-static fault
    slip inversion. Journal of Geophysical Research: Solid Earth, 126,
    e2020JB021141. https://doi.org/10.1029/2020JB021141

The regularization weights are chosen so that the a posteriori variances of the
model parameters match user-supplied target variances (the EPIC condition).
Once the weights are known, the general linear least squares problem is solved,
optionally under a non-negativity constraint on the model parameters.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, nnls

from ._base_unfolder import run_unfolding, _build_system
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_epic", "unfold_epic"]


def _compute_bounds(
    k_center: float = 0.0, distance: float = 2.0
) -> Tuple[float, float]:
    """Compute bounds for the betas to avoid floating point rounding errors.

    Port of ``beta_bounds.compute_bounds`` from EPIC_LS. Returns the largest
    ``k`` such that ``exp(k_center - k) + exp(k_center + k)`` is still well
    represented in machine precision, keeping some distance from that limit.

    Parameters
    ----------
    k_center : float
        Center of the beta search interval.
    distance : float
        Distance kept from the representability limit.

    Returns
    -------
    Tuple[float, float]
        Lower and upper bound for the betas.
    """
    eps = np.finfo(float).eps
    k_test = 0.0
    for _ in range(1, 1000000):
        k_test += 0.01
        if (
            abs(
                np.exp(k_center - k_test)
                + np.exp(k_center + k_test)
                - np.exp(k_center + k_test)
            )
            < eps
        ):
            break
    k_test -= distance
    return (k_center - k_test, k_center + k_test)


def _default_target_sigmas(
    A: np.ndarray, b: np.ndarray, n: int, sigma_frac: float
) -> np.ndarray:
    """Derive default target sigmas from the naive least-squares solution.

    The scale is taken as ``sigma_frac`` times the largest magnitude of the
    unregularized least-squares solution, falling back to the measurement
    scale when that solution is degenerate.
    """
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    scale = float(np.max(np.abs(x_ls))) if x_ls.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(b))) if b.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    return np.full(n, sigma_frac * scale)


def _build_precision(
    A: np.ndarray, noise_var: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the precision matrix P = A^T inv(Cx) A and the misfit weight Wx.

    With ``noise_var=None`` the misfit covariance is the identity matrix;
    otherwise ``Cx = noise_var * I`` and ``Wx.T @ Wx = inv(Cx)``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    noise_var : float, optional
        Variance of the i.i.d. misfit errors.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Precision matrix (n x n) and misfit weight matrix (m x m).
    """
    m, _ = A.shape
    if noise_var is None:
        P = A.T @ A
        Wx = np.eye(m)
    else:
        if noise_var <= 0:
            raise ValueError(f"noise_var must be positive, got {noise_var}")
        inv_cx = (1.0 / noise_var) * np.eye(m)
        P = A.T @ (inv_cx @ A)
        Wx = np.linalg.cholesky(inv_cx).T
    return P, Wx


def _build_regularization_matrix(n: int, order: int) -> np.ndarray:
    """Build the dense regularization operator H.

    ``order=0`` gives the identity (minimum-norm), ``order=1`` the first
    derivative and ``order=2`` the second derivative operator.
    """
    if order == 0:
        return np.eye(n)
    return create_derivative_matrix(n, order).toarray()


def _calc_epic_ch(
    P: np.ndarray,
    H: np.ndarray,
    target_sigmas: np.ndarray,
    X0: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    LSQpar: Optional[Dict[str, Any]] = None,
    homogeneous_step: bool = True,
    beta_shift_k: float = 0,
    beta_distance: float = 2,
    EPIC_bool: Optional[np.ndarray] = None,
    regularize: Optional[Dict[str, Any]] = None,
):
    """Solve the EPIC condition for the prior variances.

    Port of ``calc_EPIC_Ch`` from EPIC_LS. Returns the scipy ``OptimizeResult``
    whose ``x`` holds the betas (natural logarithms of the reciprocal prior
    variances). ``LSQpar`` may carry solver tuning: ``TolX1/TolFun1/TolG1``
    (homogeneous step), ``TolX2/TolFun2/TolG2`` (heterogeneous step) and
    ``method/loss/verbose`` (defaults ``trf``/``linear``/``0``). ``tr_solver``
    defaults to ``"exact"`` (the upstream EPIC_LS port uses ``"lsmr"``, which
    stalls on small unfolding problems and exhausts ``max_nfev``).
    """
    Nh, Nm = H.shape
    LSQpar = dict(LSQpar or {})

    if Nh > Nm:
        LSQpar.setdefault("TolX1", 1e-6)
        LSQpar.setdefault("TolFun1", 1e-6)
        LSQpar.setdefault("TolG1", 1e-6)
        LSQpar.setdefault("TolX2", 1e-6)
        LSQpar.setdefault("TolFun2", 1e-6)
        LSQpar.setdefault("TolG2", 1e-8)
        LSQpar.setdefault("damp_trf", 1e-3)
    else:
        LSQpar.setdefault("TolX1", 1e-6)
        LSQpar.setdefault("TolFun1", 1e-6)
        LSQpar.setdefault("TolG1", 1e-6)
        LSQpar.setdefault("TolX2", 1e-8)
        LSQpar.setdefault("TolFun2", 1e-8)
        LSQpar.setdefault("TolG2", 1e-10)
        LSQpar.setdefault("damp_trf", 1e-9)

    method = LSQpar.get("method", "trf")
    loss = LSQpar.get("loss", "linear")
    verbose = int(LSQpar.get("verbose", 0))

    bounds = _compute_bounds(beta_shift_k, beta_distance)

    if X0 is None:
        n_unknowns = V.shape[1] if V is not None else Nh
        X0 = np.ones(n_unknowns) * (bounds[0] + bounds[1]) / 2.0
    X0 = np.asarray(X0, dtype=float).reshape(-1)
    target_sigmas = np.asarray(target_sigmas, dtype=float).reshape(-1)
    target_var = target_sigmas**2

    sigma_weight_default = np.exp(np.finfo(float).precision / 4)

    def calc_F(X):
        """EPIC residual vector at the current betas."""
        if V is not None:
            beta = V @ X
        else:
            beta = X
        inv_ch = np.diag(np.exp(beta))
        invA = np.linalg.inv(P + H.T @ (inv_ch @ H))
        F = np.diag(invA)
        if EPIC_bool is not None:
            F = F[EPIC_bool]
        F = (F - target_var) / target_var
        if regularize is not None:
            sigma_weight = regularize.get("sigma_weight", sigma_weight_default)
            F = np.hstack((F, np.exp(beta / 2) / sigma_weight))
        return F

    def calc_JF(X):
        """Jacobian of the EPIC residual at the current betas."""
        if V is not None:
            beta = V @ X
        else:
            beta = X
        E = np.diag(np.exp(beta))
        invA = np.linalg.inv(P + H.T @ (E @ H))
        B = H @ invA
        JF = np.transpose(-1.0 * E @ (B * B))
        if V is not None:
            JF = JF @ V
        if EPIC_bool is not None:
            JF = JF[EPIC_bool, :]
        JF = np.diag(1.0 / target_var) @ JF
        if regularize is not None:
            sigma_weight = regularize.get("sigma_weight", sigma_weight_default)
            JF2 = 0.5 * np.diag(np.exp(beta / 2)) / sigma_weight
            JF = np.vstack((JF, JF2))
        return JF

    if homogeneous_step:

        def calc_F_constant_beta1(x, x0):
            return calc_F(x + x0)

        sol0 = least_squares(
            calc_F_constant_beta1,
            np.array([0.0]),
            jac="2-point",
            method=method,
            args=(X0,),
            verbose=verbose,
            ftol=LSQpar["TolFun1"],
            xtol=LSQpar["TolX1"],
            loss=loss,
            gtol=LSQpar["TolG1"],
            bounds=bounds,
        )
        Xnext = sol0.x + X0
    else:
        Xnext = X0

    tr_solver = LSQpar.get("tr_solver", "exact")
    if tr_solver == "lsmr":
        if Nh > Nm:
            tr_options = {"regularize": True, "damp": 1e-3}
        else:
            tr_options = {
                "regularize": False,
                "damp": LSQpar.get("damp_trf", 1e-9),
            }
    else:
        tr_options = None

    sol = least_squares(
        calc_F,
        Xnext,
        jac=calc_JF,
        method=method,
        verbose=verbose,
        ftol=LSQpar["TolFun2"],
        xtol=LSQpar["TolX2"],
        loss=loss,
        gtol=LSQpar["TolG2"],
        bounds=bounds,
        x_scale="jac",
        tr_solver=tr_solver,
        tr_options=tr_options,
    )

    return sol


def _final_solve(
    A: np.ndarray,
    b: np.ndarray,
    Wx: np.ndarray,
    H: np.ndarray,
    ho: np.ndarray,
    Wh: np.ndarray,
    non_neg: bool,
) -> np.ndarray:
    """Solve the augmented least squares problem.

    Minimizes ``||Wx (A m - b)||^2 + ||Wh (H m - ho)||^2`` by stacking the
    misfit and regularization blocks into an equivalent simple least squares
    problem. Applies non-negativity constraints with ``scipy.optimize.nnls``
    when ``non_neg`` is True. Port of ``LeastSquaresRegNonNeg`` from EPIC_LS.
    """
    WxG = Wx @ A
    Wxd = Wx @ b
    WhH = Wh @ H
    Whho = Wh @ ho

    F = np.vstack([WxG, WhH])
    D = np.concatenate([Wxd, Whho])

    if non_neg:
        x, _ = nnls(F, D)
    else:
        x = np.linalg.lstsq(F, D, rcond=None)[0]

    return np.maximum(np.asarray(x, dtype=float).reshape(-1), 0)


def _epic_weights(
    A: np.ndarray,
    b: np.ndarray,
    target_sigmas: Optional[np.ndarray],
    regularization_order: int,
    noise_var: Optional[float],
    homogeneous_step: bool,
    regularize: Optional[Dict[str, Any]],
    beta_shift_k: float,
    beta_distance: float,
    EPIC_bool: Optional[np.ndarray],
    V: Optional[np.ndarray],
    LSQpar: Optional[Dict[str, Any]],
    sigma_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute the EPIC regularization weights and optimization metadata.

    Returns a tuple ``(Wx, H, Wh, meta)`` where ``Wx`` and ``Wh`` are the
    misfit and prior weight matrices, ``H`` the regularization operator and
    ``meta`` the EPIC optimization status together with the resolved target
    sigmas.
    """
    n = A.shape[1]

    P, Wx = _build_precision(A, noise_var)
    H = _build_regularization_matrix(n, regularization_order)

    if EPIC_bool is not None:
        EPIC_bool = np.asarray(EPIC_bool, dtype=bool).reshape(-1)
        if EPIC_bool.size != n:
            raise ValueError(
                f"EPIC_bool length ({EPIC_bool.size}) must match "
                f"number of energy bins ({n})"
            )
        if target_sigmas is None:
            target_sigmas = _default_target_sigmas(A, b, n, sigma_frac)
        else:
            target_sigmas = np.asarray(target_sigmas, dtype=float).reshape(-1)
        if target_sigmas.size == n:
            target_epic = target_sigmas[EPIC_bool]
        else:
            target_epic = target_sigmas
        expected = int(EPIC_bool.sum())
    else:
        if target_sigmas is None:
            target_sigmas = _default_target_sigmas(A, b, n, sigma_frac)
        else:
            target_sigmas = np.asarray(target_sigmas, dtype=float).reshape(-1)
        target_epic = target_sigmas
        expected = n

    if target_epic.size != expected:
        raise ValueError(
            f"target_sigmas length ({target_epic.size}) must match "
            f"the number of parameters subject to the EPIC ({expected})"
        )
    if not np.all(np.isfinite(target_epic)) or np.any(target_epic <= 0):
        raise ValueError("target_sigmas must be finite and strictly positive")

    sol = _calc_epic_ch(
        P,
        H,
        target_epic,
        X0=None,
        V=V,
        LSQpar=LSQpar,
        homogeneous_step=homogeneous_step,
        beta_shift_k=beta_shift_k,
        beta_distance=beta_distance,
        EPIC_bool=EPIC_bool,
        regularize=regularize,
    )

    beta = np.asarray(sol.x, dtype=float).reshape(-1)
    if V is not None:
        beta = V @ beta
    Wh = np.diag(np.exp(beta / 2))

    meta = {
        "epic_converged": bool(sol.success),
        "epic_cost": float(sol.cost),
        "epic_nfev": int(sol.nfev),
        "beta_min": float(np.min(beta)),
        "beta_max": float(np.max(beta)),
        "target_sigmas": np.asarray(target_epic, dtype=float).tolist(),
    }

    return Wx, H, Wh, meta


def solve_epic(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    target_sigmas: Optional[np.ndarray] = None,
    sigma_frac: float = 0.1,
    regularization_order: int = 1,
    non_neg: bool = True,
    noise_var: Optional[float] = None,
    homogeneous_step: bool = True,
    regularize: Optional[Dict[str, Any]] = None,
    beta_shift_k: float = 0,
    beta_distance: float = 2,
    EPIC_bool: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    LSQpar: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Solve the unfolding problem with EPIC Tikhonov regularization.

    Selects the prior variances of the regularization operator H such that the
    a posteriori variances of the model parameters equal the squared target
    sigmas, then solves the weighted least squares problem.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility).
    target_sigmas : np.ndarray, optional
        Target a posteriori standard deviations of the model parameters.
        If None, derived as ``sigma_frac`` times the magnitude of the naive
        least-squares solution. Must be strictly positive.
    sigma_frac : float, optional
        Fraction used to derive the default target sigmas (default: 0.1).
    regularization_order : int, optional
        Regularization operator order: 0 (identity), 1 (first derivative,
        default) or 2 (second derivative).
    non_neg : bool, optional
        Apply non-negativity constraints to the model parameters (default:
        True).
    noise_var : float, optional
        Variance of the i.i.d. misfit errors used to build Cx (default: None,
        meaning Cx is the identity matrix).
    homogeneous_step : bool, optional
        Run a preliminary homogeneous Ch search (default: True).
    regularize : dict, optional
        If given (can be empty), damp the EPIC weights towards a minimum-norm
        solution. May carry ``sigma_weight``.
    beta_shift_k : float, optional
        Center shift for the beta bounds (default: 0).
    beta_distance : float, optional
        Distance kept from the representability limit (default: 2).
    EPIC_bool : np.ndarray, optional
        Boolean mask of which parameters are subject to the EPIC.
    V : np.ndarray, optional
        Matrix mapping the searched betas to the regularization rows, beta = V @ y (shape (H.shape[0], len(y))).
    LSQpar : dict, optional
        Tuning parameters for the nonlinear least-squares solver.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, _ = A.shape
    if b.size != m:
        raise ValueError(
            f"b length ({b.size}) must match the number of A rows ({m})"
        )
    if regularization_order not in (0, 1, 2):
        raise ValueError(
            "Unsupported regularization_order: "
            f"{regularization_order}. Use 0 (identity), 1 or 2."
        )

    Wx, H, Wh, _ = _epic_weights(
        A=A,
        b=b,
        target_sigmas=target_sigmas,
        regularization_order=regularization_order,
        noise_var=noise_var,
        homogeneous_step=homogeneous_step,
        regularize=regularize,
        beta_shift_k=beta_shift_k,
        beta_distance=beta_distance,
        EPIC_bool=EPIC_bool,
        V=V,
        LSQpar=LSQpar,
        sigma_frac=sigma_frac,
    )

    ho = np.zeros(H.shape[0])
    return _final_solve(A, b, Wx, H, ho, Wh, non_neg)


def unfold_epic(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    target_sigmas: Optional[np.ndarray] = None,
    sigma_frac: float = 0.1,
    regularization_order: int = 1,
    non_neg: bool = True,
    noise_var: Optional[float] = None,
    homogeneous_step: bool = True,
    regularize: Optional[Dict[str, Any]] = None,
    beta_shift_k: float = 0,
    beta_distance: float = 2,
    EPIC_bool: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    LSQpar: Optional[Dict[str, Any]] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a spectrum using EPIC Tikhonov regularization.

    The EPIC weights are computed once from the response matrix and the
    resolved target sigmas (they do not depend on the measurements), and are
    reused by the Monte-Carlo uncertainty propagation.

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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess (unused by this method).
    target_sigmas : np.ndarray, optional
        Target a posteriori standard deviations of the model parameters. If
        None, derived from ``sigma_frac``.
    sigma_frac : float, optional
        Fraction of the naive least-squares magnitude used to derive default
        target sigmas (default: 0.1).
    regularization_order : int, optional
        Regularization operator order: 0 (identity), 1 (first derivative,
        default) or 2 (second derivative).
    non_neg : bool, optional
        Apply non-negativity constraints (default: True).
    noise_var : float, optional
        Variance of the i.i.d. misfit errors (default: None, identity Cx).
    homogeneous_step : bool, optional
        Run a preliminary homogeneous Ch search (default: True).
    regularize : dict, optional
        If given (can be empty), damp the EPIC weights.
    beta_shift_k : float, optional
        Center shift for the beta bounds (default: 0).
    beta_distance : float, optional
        Distance kept from the representability limit (default: 2).
    EPIC_bool : np.ndarray, optional
        Boolean mask of which parameters are subject to the EPIC.
    V : np.ndarray, optional
        Matrix mapping the searched betas to the regularization rows, beta = V @ y (shape (H.shape[0], len(y))).
    LSQpar : dict, optional
        Tuning parameters for the nonlinear least-squares solver.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty (default: False).
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
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    if regularization_order not in (0, 1, 2):
        raise ValueError(
            "Unsupported regularization_order: "
            f"{regularization_order}. Use 0 (identity), 1 or 2."
        )

    Wx, H, Wh, epic_meta = _epic_weights(
        A=A,
        b=b,
        target_sigmas=target_sigmas,
        regularization_order=regularization_order,
        noise_var=noise_var,
        homogeneous_step=homogeneous_step,
        regularize=regularize,
        beta_shift_k=beta_shift_k,
        beta_distance=beta_distance,
        EPIC_bool=EPIC_bool,
        V=V,
        LSQpar=LSQpar,
        sigma_frac=sigma_frac,
    )
    ho = np.zeros(H.shape[0])

    if not epic_meta["epic_converged"]:
        warnings.warn(
            "EPIC nonlinear optimization did not converge fully "
            f"(cost={epic_meta['epic_cost']:.3e}); returning the "
            "best-effort regularized solution.",
            UserWarning,
        )

    def solve_wrapper(A, b, **kwargs):
        kwargs.pop("x0", None)
        return _final_solve(A, b, Wx, H, ho, Wh, non_neg)

    extra_output = {
        "regularization_order": regularization_order,
        "non_neg": bool(non_neg),
    }
    extra_output.update(epic_meta)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.zeros(n_energy_bins),
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="EPIC",
        extra_output=extra_output,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
