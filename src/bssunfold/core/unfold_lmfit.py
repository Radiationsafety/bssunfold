"""lmfit-based unfolding method with L1/L2/Elastic net regularization.

This module provides the core solve_lmfit solver and the unfold_lmfit
wrapper for use with the Detector class.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import _build_system, _normalize_initial, run_unfolding

__all__ = [
    "solve_lmfit",
    "unfold_lmfit",
    "select_regularization_aic_bic",
    "_residual_lasso",
    "_residual_ridge",
    "_residual_elastic",
]


# ---------------------------------------------------------------------------
# Residual functions for lmfit
# ---------------------------------------------------------------------------


def _residual_lasso(params, A, b, regularization, method, m):
    """Lasso (L1) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * np.sqrt(np.abs(x))
        return np.concatenate([residual, reg_residual])
    return np.sum(residual**2) + regularization * np.sum(np.abs(x))


def _residual_ridge(params, A, b, regularization, method, m):
    """Ridge (L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        reg_residual = np.sqrt(regularization) * x
        return np.concatenate([residual, reg_residual])
    return np.sum(residual**2) + regularization * np.sum(x**2)


def _residual_elastic(
    params, A, b, regularization, regularization2, l1_weight, method, m
):
    """Elastic net (L1 + L2) residual function for lmfit."""
    x = np.array([params[f"x{i}"].value for i in range(m)])
    residual = A @ x - b
    if method == "leastsq":
        l1_residual = np.sqrt(regularization * l1_weight) * np.sqrt(np.abs(x))
        l2_residual = np.sqrt(regularization2 * (1 - l1_weight)) * x
        reg_residual = np.concatenate([l1_residual, l2_residual])
        return np.concatenate([residual, reg_residual])
    l1_penalty = regularization * l1_weight * np.sum(np.abs(x))
    l2_penalty = regularization2 * (1 - l1_weight) * np.sum(x**2)
    return np.sum(residual**2) + l1_penalty + l2_penalty


_RESIDUAL_MAP = {
    "lasso": (_residual_lasso, ["regularization"]),
    "ridge": (_residual_ridge, ["regularization"]),
    "elastic": (
        _residual_elastic,
        ["regularization", "regularization2", "l1_weight"],
    ),
}

_AIC_BIC_METHODS = ("aic", "aicc", "bic")


# ---------------------------------------------------------------------------
# AIC/BIC regularization selection
# ---------------------------------------------------------------------------


def _effective_df_ridge(A: np.ndarray, lambda_reg: float) -> float:
    """Effective degrees of freedom for ridge regression.

    ``df = sum(s_i^2 / (s_i^2 + lambda))`` computed from the singular values
    of the response matrix A.
    """
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    return float(np.sum(s**2 / (s**2 + lambda_reg)))


def _effective_df_lasso(
    spectrum: np.ndarray,
    A: np.ndarray,
    lambda_reg: float,
    epsilon: float = 1e-8,
) -> float:
    """Effective degrees of freedom for lasso regression.

    Uses the active set (``|x| > epsilon``) and an SVD-based smooth count on
    the corresponding sub-matrix of A.
    """
    active_set = np.abs(spectrum) > epsilon
    n_active = np.sum(active_set)
    if n_active == 0:
        return 0.0
    A_active = A[:, active_set]
    _, s, _ = np.linalg.svd(A_active, full_matrices=False)
    return float(np.sum(s**2 / (s**2 + lambda_reg * epsilon)))


def _effective_df_elastic(
    spectrum: np.ndarray,
    A: np.ndarray,
    lambda1: float,
    lambda2: float,
    epsilon: float = 1e-8,
) -> float:
    """Effective degrees of freedom for elastic net.

    Approximation ``df ~ df_lasso + 0.5 * sum(lambda2 / (s^2 + lambda2 + eps))``
    computed on the active set.
    """
    df_lasso = _effective_df_lasso(spectrum, A, lambda1, epsilon)
    active_set = np.abs(spectrum) > epsilon
    if np.sum(active_set) == 0:
        return 0.0
    A_active = A[:, active_set]
    _, s, _ = np.linalg.svd(A_active, full_matrices=False)
    l2_shift = np.sum(lambda2 / (s**2 + lambda2 + epsilon))
    return df_lasso + l2_shift * 0.5


def _aic_bic_metrics(
    A: np.ndarray,
    b: np.ndarray,
    spectrum: np.ndarray,
    lambda_reg: float,
    lambda2_reg: float,
    model_name: str,
    l1_weight: float,
) -> Dict[str, float]:
    """Compute AIC/AICc/BIC for a given regularized solution.

    Assumes i.i.d. Gaussian residuals with variance estimated as
    ``sigma2 = ||A x - b||^2 / n_detectors``.

    Returns
    -------
    Dict[str, float]
        Keys: AIC, AICc, BIC, df, log_likelihood, residual_norm, sigma2,
        n_detectors.
    """
    residual = b - A @ spectrum
    residual_norm = float(np.linalg.norm(residual))
    n_detectors = len(b)

    sigma2_hat = residual_norm**2 / n_detectors
    if sigma2_hat <= 0:
        sigma2_hat = np.finfo(float).eps
    log_likelihood = (
        -0.5 * n_detectors * np.log(2 * np.pi * sigma2_hat)
        - 0.5 * residual_norm**2 / sigma2_hat
    )

    if model_name == "ridge":
        df = _effective_df_ridge(A, lambda_reg)
    elif model_name == "lasso":
        df = _effective_df_lasso(spectrum, A, lambda_reg)
    elif model_name == "elastic":
        df = _effective_df_elastic(spectrum, A, lambda_reg, lambda2_reg)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Choose from: elastic, lasso, ridge"
        )

    AIC = -2 * log_likelihood + 2 * df
    BIC = -2 * log_likelihood + df * np.log(n_detectors)

    if n_detectors / df < 40:
        AICc = AIC + 2 * df * (df + 1) / (n_detectors - df - 1 + 1e-10)
    else:
        AICc = AIC

    return {
        "AIC": float(AIC),
        "AICc": float(AICc),
        "BIC": float(BIC),
        "df": float(df),
        "log_likelihood": float(log_likelihood),
        "residual_norm": residual_norm,
        "sigma2": float(sigma2_hat),
        "n_detectors": n_detectors,
    }


def select_regularization_aic_bic(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
    criterion: str = "aic",
    lambda_range: Tuple[float, float] = (1e-6, 1e-1),
    n_lambda: int = 30,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Select the lmfit regularization parameter by an information criterion.

    Sweeps ``n_lambda`` log-spaced regularization candidates, solves the
    lmfit problem for each, scores it with AIC/AICc/BIC (effective degrees of
    freedom and Gaussian likelihood of the data residual), and returns the
    candidate minimizing the chosen criterion.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum used for every candidate solve.
    method : str, optional
        lmfit solver name, default: "lbfgsb".
    model_name : str, optional
        Regularization model: elastic, lasso, ridge, default: "elastic".
    regularization : float, optional
        Manual L1 regularization strength, used as fallback if every
        candidate solve fails, default: 1e-4.
    regularization2 : float, optional
        L2 regularization strength for elastic net, default: 1e-4.
    l1_weight : float, optional
        L1 weight for elastic net, default: 0.5.
    criterion : str, optional
        Information criterion to minimize: 'aic', 'aicc' or 'bic',
        default: 'aic'.
    lambda_range : Tuple[float, float], optional
        Log-spaced range of lambda candidates, default: (1e-6, 1e-1).
    n_lambda : int, optional
        Number of lambda candidates, default: 30.
    verbose : bool, optional
        Print the selection summary, default: True.

    Returns
    -------
    Dict[str, Any]
        Keys: best_lambda, best_lambda2, best_criterion_value, best_index,
        best_df, lambda_candidates, aic_values, aicc_values, bic_values,
        df_values, criterion_used, model_name.

    Raises
    ------
    ValueError
        If criterion is not one of 'aic', 'aicc', 'bic'.
    """
    if criterion not in _AIC_BIC_METHODS:
        raise ValueError(
            f"Unknown criterion: {criterion}. "
            "Choose from: aic, aicc, bic"
        )

    lambda_candidates = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambda
    )

    aic_values = []
    aicc_values = []
    bic_values = []
    df_values = []

    for lam in lambda_candidates:
        if model_name == "elastic":
            lam2 = lam * (1 - l1_weight) / (l1_weight + 1e-10)
        else:
            lam2 = regularization2
        try:
            spectrum, success, _message, _nfev = solve_lmfit(
                A, b, x0, method, model_name, lam, lam2, l1_weight
            )
            if not success:
                raise RuntimeError(f"lmfit solve failed: {_message}")
            metrics = _aic_bic_metrics(
                A, b, spectrum, lam, lam2, model_name, l1_weight
            )
        except Exception:
            aic_values.append(np.inf)
            aicc_values.append(np.inf)
            bic_values.append(np.inf)
            df_values.append(np.nan)
            continue
        aic_values.append(metrics["AIC"])
        aicc_values.append(metrics["AICc"])
        bic_values.append(metrics["BIC"])
        df_values.append(metrics["df"])

    aic_values = np.asarray(aic_values)
    aicc_values = np.asarray(aicc_values)
    bic_values = np.asarray(bic_values)
    df_values = np.asarray(df_values)

    if criterion == "aic":
        crit_values = aic_values
    elif criterion == "aicc":
        crit_values = aicc_values
    else:
        crit_values = bic_values

    finite = np.isfinite(crit_values)
    if not np.any(finite):
        warnings.warn(
            "AIC/BIC selection: all candidate solves failed. "
            "Falling back to the manual regularization parameter.",
            RuntimeWarning,
        )
        best_index = 0
        best_lambda = float(regularization)
        best_lambda2 = float(regularization2)
        best_value = np.inf
        best_df = np.nan
    else:
        best_index = int(np.nanargmin(crit_values))
        best_lambda = float(lambda_candidates[best_index])
        if model_name == "elastic":
            best_lambda2 = (
                best_lambda * (1 - l1_weight) / (l1_weight + 1e-10)
            )
        else:
            best_lambda2 = float(regularization2)
        best_value = float(crit_values[best_index])
        best_df = float(df_values[best_index])

    if verbose:
        print(
            f"Selected regularization (method={criterion}): "
            f"{best_lambda:.3e}"
        )
        if best_df == best_df:
            print(
                f"  best {criterion.upper()} = {best_value:.2f}, "
                f"effective df = {best_df:.1f}"
            )

    return {
        "best_lambda": best_lambda,
        "best_lambda2": best_lambda2,
        "best_criterion_value": best_value,
        "best_index": best_index,
        "best_df": best_df,
        "lambda_candidates": lambda_candidates,
        "aic_values": aic_values,
        "aicc_values": aicc_values,
        "bic_values": bic_values,
        "df_values": df_values,
        "criterion_used": criterion,
        "model_name": model_name,
    }


def solve_lmfit(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding problem using lmfit with L1/L2/Elastic regularization.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial guess (n,).
    method : str, optional
        lmfit solver name (leastsq, lbfgsb, etc.), default: "lbfgsb".
    model_name : str, optional
        Regularization model: elastic, lasso, ridge, default: "elastic".
    regularization : float, optional
        L1 regularization strength, default: 1e-4.
    regularization2 : float, optional
        L2 regularization strength for elastic net, default: 1e-4.
    l1_weight : float, optional
        L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5.

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        Tuple of (solution, success, message, nfev).
    """
    try:
        import lmfit
    except ImportError as e:
        raise ImportError(
            "lmfit is required for unfold_lmfit. Install with: pip install lmfit"
        ) from e

    m = A.shape[1]

    params = lmfit.Parameters()
    for i in range(m):
        params.add(f"x{i}", value=max(x0[i], 1e-10), min=0.0)

    if model_name not in _RESIDUAL_MAP:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Choose from: elastic, lasso, ridge"
        )

    residual_func, arg_names = _RESIDUAL_MAP[model_name]
    residual_args = {
        "A": A,
        "b": b,
        "method": method,
        "m": m,
    }
    for name in arg_names:
        residual_args[name] = locals()[name]

    result = lmfit.minimize(
        residual_func,
        params,
        args=(A, b, regularization, method, m)
        if model_name in ("lasso", "ridge")
        else (A, b, regularization, regularization2, l1_weight, method, m),
        method=method,
    )

    spectrum = np.array([result.params[f"x{i}"].value for i in range(m)])
    return spectrum, result.success, result.message, result.nfev


def unfold_lmfit(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    method: str = "lbfgsb",
    model_name: str = "elastic",
    regularization: float = 1e-4,
    regularization2: float = 1e-4,
    l1_weight: float = 0.5,
    regularization_method: str = "manual",
    lambda_range: Tuple[float, float] = (1e-6, 1e-1),
    n_lambda: int = 30,
    verbose: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using lmfit with L1/L2/Elastic regularization.

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
        Initial spectrum guess. If None, uniform spectrum based on mean readings.
    method : str, optional
        lmfit solver name (leastsq, lbfgsb, etc.), default: "lbfgsb".
    model_name : str, optional
        Regularization model: elastic, lasso, ridge, default: "elastic".
    regularization : float, optional
        L1 regularization strength, default: 1e-4.
    regularization2 : float, optional
        L2 regularization strength for elastic net, default: 1e-4.
    l1_weight : float, optional
        L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5.
    regularization_method : str, optional
        How to choose the regularization parameter. Options: 'manual'
        (use the supplied ``regularization``/``regularization2``), or an
        information criterion 'aic', 'aicc' or 'bic'. For non-manual
        selection the regularization parameter is swept over ``lambda_range``
        and the candidate minimizing the chosen criterion is used.
        Default: 'manual'.
    lambda_range : Tuple[float, float], optional
        Log-spaced range of lambda candidates for information-criterion
        selection, default: (1e-6, 1e-1).
    n_lambda : int, optional
        Number of lambda candidates for information-criterion selection,
        default: 30.
    verbose : bool, optional
        Print the regularization selection summary, default: True.
    calculate_errors : bool, optional
        Flag to calculate uncertainty via Monte-Carlo, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples for error estimation, default: 100.
    save_result : bool, optional
        If True, save result to internal history, default: True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing unfolding results.
    """
    if regularization_method not in ("manual",) + _AIC_BIC_METHODS:
        raise ValueError(
            f"Unknown regularization_method: {regularization_method}. "
            "Choose from: manual, aic, aicc, bic"
        )

    A, b, _ = _build_system(readings, detector_names, sensitivities)

    requested_regularization = regularization
    requested_regularization2 = regularization2

    selection_info = None
    if regularization_method != "manual":
        x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(
            A.sum(axis=1)
        )
        x0_sel = _normalize_initial(initial_spectrum, x0_default, n_energy_bins)
        selection_info = select_regularization_aic_bic(
            A,
            b,
            x0_sel,
            method=method,
            model_name=model_name,
            regularization=regularization,
            regularization2=regularization2,
            l1_weight=l1_weight,
            criterion=regularization_method,
            lambda_range=lambda_range,
            n_lambda=n_lambda,
            verbose=verbose,
        )
        regularization = selection_info["best_lambda"]
        regularization2 = selection_info["best_lambda2"]

    initial_spec_for_output = None

    def solve_wrapper(A, b, **kwargs):
        nonlocal initial_spec_for_output
        x0 = kwargs.pop("x0", None)
        if x0 is None:
            x0 = np.ones(A.shape[1]) * np.mean(b) / np.mean(A.sum(axis=1))
        initial_spec_for_output = x0.copy()
        x_opt, success, _message, nfev = solve_lmfit(
            A,
            b,
            x0,
            method,
            model_name,
            regularization,
            regularization2,
            l1_weight,
        )
        return x_opt, nfev, success

    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

    extra_output = {
        "model_name": model_name,
        "regularization": requested_regularization,
        "regularization2": requested_regularization2
        if model_name == "elastic"
        else None,
        "l1_weight": l1_weight if model_name == "elastic" else None,
        "regularization_method": regularization_method,
        "selected_regularization": float(regularization),
    }
    if model_name == "elastic":
        extra_output["selected_regularization2"] = float(regularization2)
    if selection_info is not None:
        extra_output.update(
            {
                "best_criterion_value": selection_info[
                    "best_criterion_value"
                ],
                "best_df": selection_info["best_df"],
                "criterion_used": selection_info["criterion_used"],
                "aic_bic_path": {
                    "lambda_candidates": selection_info["lambda_candidates"],
                    "aic_values": selection_info["aic_values"],
                    "aicc_values": selection_info["aicc_values"],
                    "bic_values": selection_info["bic_values"],
                    "df_values": selection_info["df_values"],
                },
            }
        )

    result = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name=f"lmfit ({method})",
        extra_output=extra_output,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    if initial_spec_for_output is not None:
        result["initial_spectrum"] = initial_spec_for_output

    return result
