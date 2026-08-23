"""FRUIT-based parametric unfolding method.

This module implements the parametric neutron spectrum reconstruction
method described in:

  - R. Bedogni et al., "FRUIT: An operational tool for multisphere
    neutron spectrometry in workplaces", Nucl. Instrum. Methods A 580,
    1301-1309 (2007).
  - M.D. Pyshkina et al., "Validation and Verification of the New
    Multisphere Spectrometer Operation", Proc. II Int. Sci.-Tech.
    Conf., Minsk (2021).

The spectrum is represented as a weighted superposition of three
components:

  Thermal   (E < 1e-7 MeV):   (E/T0^2) * exp(-E/T0)
  Epithermal (1e-7 < E < 0.1): [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
  Fast      (E > 0.1 MeV):    E^alpha * exp(-E/beta)

Total: phi_j = P_th * phi_th + P_epi * phi_epi + P_f * phi_f
with constraint: P_th + P_epi + P_f = 1  (P_f = 1 - P_th - P_epi)
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from ._base_unfolder import run_unfolding, _build_system
from ._matrix_utils import compute_log_steps
from ._solver_backends import (
    _parse_solver_backend,
    _resolve_cvxpy_solvers,
    _resolve_qpsolver_name,
)

from ._fruit import (
    parametric_model,
    solve_parametric,
    _residuals,  # noqa: F401  (re-exported for test compatibility)
    _get_initial_params,  # noqa: F401  (re-exported for test compatibility)
    _get_param_bounds,
    _clamp_params,
    _compute_jacobian,
    _find_initial_params,
    _gcv_select_alpha,  # noqa: F401  (re-exported for test compatibility)
    _check_fit_quality,
    _T0,
    _Ed,
    _PARAM_NAMES,
    _THERMAL_MAX,  # noqa: F401  (re-exported for test compatibility)
    _FAST_MIN,  # noqa: F401  (re-exported for test compatibility)
)

logger = logging.getLogger(__name__)

__all__ = [
    "solve_parametric",
    "solve_parametric_cvxpy",
    "solve_parametric_qpsolvers",
    "solve_parametric_combined",
    "unfold_parametric",
]

def solve_parametric_cvxpy(
    A_matrix,
    b_readings,
    E,
    log_steps,
    initial_params=None,
    alpha=1e-4,
    solver_backend="auto",
    max_iter=50,
    tol=1e-6,
):
    """Solve parametric unfolding via sequential QP using cvxpy.

    The nonlinear parametric model is linearized at each iteration and
    the resulting QP is solved with cvxpy, including parameter bounds
    and a Tikhonov penalty on the parameter update.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps (d(ln E)).
    initial_params : dict, optional
        Initial parameter values.
    alpha : float, optional
        Regularization weight for parameter penalty (default: 1e-4).
    solver_backend : str, optional
        CVXPY solver backend: "auto", "cvxpy", or "cvxpy:ECOS" etc.
        (default: "auto").
    max_iter : int, optional
        Maximum SQP iterations (default: 50).
    tol : float, optional
        Convergence tolerance on parameter update norm (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for parametric_cvxpy. "
            "Install with: pip install cvxpy"
        ) from e

    _, backend = _parse_solver_backend(solver_backend)
    solvers_to_try = _resolve_cvxpy_solvers(backend)

    # Find good initial params via brute-force scan
    params = _find_initial_params(A_matrix, b_readings, E, log_steps)
    if initial_params:
        params.update(initial_params)
    params = _clamp_params(params, _get_param_bounds())

    n_params = len(_PARAM_NAMES)
    message = ""
    nfev = 0

    for k in range(max_iter):
        spectrum_k = (
            parametric_model(
                E,
                params["b"],
                params["beta_prime"],
                params["alpha"],
                params["beta"],
                params["P_th"],
                params["P_epi"],
            )
            * log_steps
        )

        residual = A_matrix @ spectrum_k - b_readings
        nfev += 1

        if np.linalg.norm(residual) < tol:
            _check_fit_quality(
                np.linalg.norm(residual), b_readings, "parametric_cvxpy"
            )
            message = f"Converged in {k} iterations"
            return spectrum_k, True, message, nfev

        J = _compute_jacobian(E, log_steps, params)
        A_eff = A_matrix @ J

        delta = cp.Variable(n_params)
        data_term = cp.sum_squares(A_eff @ delta + residual)
        penalty_term = alpha * cp.sum_squares(delta)
        objective = cp.Minimize(data_term + penalty_term)

        bounds = _get_param_bounds()
        constraints = []
        for i, name in enumerate(_PARAM_NAMES):
            lo, hi = bounds[name]
            if lo is not None:
                constraints.append(delta[i] >= lo - params[name])
            if hi is not None:
                constraints.append(delta[i] <= hi - params[name])

        problem = cp.Problem(objective, constraints)

        solved = False
        for s in solvers_to_try:
            try:
                problem.solve(solver=s)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    if delta.value is not None:
                        solved = True
                        break
            except Exception as exc:
                logger.debug("CVXPY solver %s failed: %s", s, exc)
                continue

        if not solved:
            message = f"QP subproblem failed at iteration {k}"
            break

        delta_val = np.asarray(delta.value)
        for i, name in enumerate(_PARAM_NAMES):
            params[name] += delta_val[i]
        params = _clamp_params(params, bounds)

        if np.linalg.norm(delta_val) < tol:
            message = f"Converged in {k + 1} iterations"
            return (
                parametric_model(
                    E,
                    params["b"],
                    params["beta_prime"],
                    params["alpha"],
                    params["beta"],
                    params["P_th"],
                    params["P_epi"],
                )
                * log_steps,
                True,
                message,
                nfev,
            )

    spectrum = (
        parametric_model(
            E,
            params["b"],
            params["beta_prime"],
            params["alpha"],
            params["beta"],
            params["P_th"],
            params["P_epi"],
        )
        * log_steps
    )

    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum - b_readings),
        b_readings,
        "parametric_cvxpy",
    )

    if not message:
        message = f"Max iterations ({max_iter}) reached"

    return spectrum, False, message, nfev


# ------------------------------------------------------------------ #
#  qpsolvers-based parametric solver (SQP)
# ------------------------------------------------------------------ #


def solve_parametric_qpsolvers(
    A_matrix,
    b_readings,
    E,
    log_steps,
    initial_params=None,
    alpha=1e-4,
    solver_backend="auto",
    max_iter=50,
    tol=1e-6,
):
    """Solve parametric unfolding via sequential QP using qpsolvers.

    The nonlinear parametric model is linearized at each iteration and
    the resulting QP is solved with qpsolvers, including parameter
    bounds and a Tikhonov penalty on the parameter update.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps (d(ln E)).
    initial_params : dict, optional
        Initial parameter values.
    alpha : float, optional
        Regularization weight (default: 1e-4).
    solver_backend : str, optional
        QP solver backend: "auto", "qpsolvers", or "qpsolvers:osqp" etc.
        (default: "auto").
    max_iter : int, optional
        Maximum SQP iterations (default: 50).
    tol : float, optional
        Convergence tolerance on parameter update norm (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as e:
        raise ImportError(
            "qpsolvers is required for parametric_qpsolvers. "
            "Install with: pip install qpsolvers"
        ) from e

    from scipy.sparse import csc_matrix

    _, backend = _parse_solver_backend(solver_backend)
    solver_name = _resolve_qpsolver_name(backend)

    if solver_name not in available_solvers:
        if "osqp" in available_solvers:
            solver_name = "osqp"
        elif "ecos" in available_solvers:
            solver_name = "ecos"
        else:
            raise ValueError(
                f"Solver '{solver_name}' not available. "
                f"Available: {available_solvers}"
            )

    params = _find_initial_params(A_matrix, b_readings, E, log_steps)
    if initial_params:
        params.update(initial_params)
    params = _clamp_params(params, _get_param_bounds())

    n_params = len(_PARAM_NAMES)
    message = ""
    nfev = 0

    for k in range(max_iter):
        spectrum_k = (
            parametric_model(
                E,
                params["b"],
                params["beta_prime"],
                params["alpha"],
                params["beta"],
                params["P_th"],
                params["P_epi"],
            )
            * log_steps
        )

        residual = A_matrix @ spectrum_k - b_readings
        nfev += 1

        if np.linalg.norm(residual) < tol:
            _check_fit_quality(
                np.linalg.norm(residual), b_readings, "parametric_qpsolvers"
            )
            message = f"Converged in {k} iterations"
            return spectrum_k, True, message, nfev

        J = _compute_jacobian(E, log_steps, params)
        A_eff = A_matrix @ J

        P = csc_matrix(A_eff.T @ A_eff + alpha * np.eye(n_params))
        q = A_eff.T @ residual

        bounds = _get_param_bounds()
        G_rows = []
        h_rows = []
        for i, name in enumerate(_PARAM_NAMES):
            lo, hi = bounds[name]
            if lo is not None:
                row = np.zeros(n_params)
                row[i] = -1.0
                G_rows.append(row)
                h_rows.append(-(lo - params[name]))
            if hi is not None:
                row = np.zeros(n_params)
                row[i] = 1.0
                G_rows.append(row)
                h_rows.append(hi - params[name])

        if G_rows:
            G = csc_matrix(np.vstack(G_rows))
            h = np.array(h_rows)
        else:
            G = csc_matrix(np.zeros((0, n_params)))
            h = np.zeros(0)

        try:
            delta_val = solve_qp(
                P=P,
                q=q,
                G=G,
                h=h,
                solver=solver_name,
                verbose=False,
            )
        except Exception as exc:
            logger.debug(
                "QP solver %s failed at iteration %d: %s", solver_name, k, exc
            )
            message = f"QP subproblem failed at iteration {k}"
            break

        if delta_val is None:
            message = f"QP solver returned None at iteration {k}"
            break

        delta_val = np.asarray(delta_val)
        for i, name in enumerate(_PARAM_NAMES):
            params[name] += delta_val[i]
        params = _clamp_params(params, bounds)

        if np.linalg.norm(delta_val) < tol:
            message = f"Converged in {k + 1} iterations"
            return (
                parametric_model(
                    E,
                    params["b"],
                    params["beta_prime"],
                    params["alpha"],
                    params["beta"],
                    params["P_th"],
                    params["P_epi"],
                )
                * log_steps,
                True,
                message,
                nfev,
            )

    spectrum = (
        parametric_model(
            E,
            params["b"],
            params["beta_prime"],
            params["alpha"],
            params["beta"],
            params["P_th"],
            params["P_epi"],
        )
        * log_steps
    )

    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum - b_readings),
        b_readings,
        "parametric_qpsolvers",
    )

    if not message:
        message = f"Max iterations ({max_iter}) reached"

    return spectrum, False, message, nfev


# ------------------------------------------------------------------ #
#  Combined: lmfit first, then QP refinement
# ------------------------------------------------------------------ #


def solve_parametric_combined(
    A_matrix,
    b_readings,
    E,
    log_steps,
    initial_params=None,
    method="leastsq",
    alpha=1e-4,
    solver_backend="auto",
):
    """Solve parametric unfolding: lmfit first, then QP refinement.

    1. Use lmfit to find the best-fit parametric shape parameters.
    2. Take the resulting spectrum as a starting point and refine it
       with a QP solver (cvxpy or qpsolvers) that adds non-negativity
       and a penalty toward the lmfit solution.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    log_steps : np.ndarray
        Logarithmic energy steps (d(ln E)).
    initial_params : dict, optional
        Initial parameter values for lmfit.
    method : str, optional
        lmfit method (default: "leastsq").
    alpha : float, optional
        Regularization weight for QP refinement (default: 1e-4).
    solver_backend : str, optional
        QP backend for refinement: "auto", "cvxpy", "qpsolvers", etc.
        (default: "auto").

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    # Step 1: lmfit
    spectrum_lmfit, lmfit_success, lmfit_msg, lmfit_nfev = solve_parametric(
        A_matrix,
        b_readings,
        E,
        log_steps,
        initial_params,
        method,
    )

    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_lmfit - b_readings),
        b_readings,
        "parametric_combined(lmfit)",
    )

    # Step 2: QP refinement on the spectrum
    spectrum_init = spectrum_lmfit.copy()
    n = A_matrix.shape[1]

    library, backend = _parse_solver_backend(solver_backend)

    # Auto-detect: try cvxpy first, then qpsolvers
    if library == "auto":
        try:
            import cvxpy  # noqa: F401  # pylint: disable=unused-import

            library = "cvxpy"
        except ImportError:
            library = "qpsolvers"

    if library == "cvxpy":
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(
                "cvxpy is required for combined. Install with: pip install cvxpy"
            ) from e

        solvers_to_try = _resolve_cvxpy_solvers(backend)

        x_var = cp.Variable(n, nonneg=True)
        obj = cp.Minimize(
            cp.sum_squares(A_matrix @ x_var - b_readings)
            + alpha * cp.sum_squares(x_var - spectrum_init)
        )
        problem = cp.Problem(obj)

        refined = None
        for s in solvers_to_try:
            try:
                problem.solve(solver=s)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    if x_var.value is not None:
                        refined = np.asarray(x_var.value)
                        break
            except Exception as exc:
                logger.debug("CVXPY solver %s failed: %s", s, exc)
                continue

        if refined is None:
            return (
                spectrum_lmfit,
                lmfit_success,
                "QP refinement failed",
                lmfit_nfev,
            )

        success = lmfit_success
        message = f"lmfit ({lmfit_msg}) + QP refinement OK"
        return refined * log_steps, success, message, lmfit_nfev

    if library == "qpsolvers":
        try:
            from qpsolvers import available_solvers, solve_qp
        except ImportError as e:
            raise ImportError(
                "qpsolvers is required for combined. Install with: pip install qpsolvers"
            ) from e

        from scipy.sparse import csc_matrix

        qpsolver_name = _resolve_qpsolver_name(backend)
        if qpsolver_name not in available_solvers:
            if "osqp" in available_solvers:
                qpsolver_name = "osqp"
            elif "ecos" in available_solvers:
                qpsolver_name = "ecos"
            else:
                return (
                    spectrum_lmfit,
                    lmfit_success,
                    "No QP solver available",
                    lmfit_nfev,
                )

        P = csc_matrix(A_matrix.T @ A_matrix + alpha * np.eye(n))
        q = -(A_matrix.T @ b_readings + alpha * spectrum_init)

        G = csc_matrix(-np.eye(n))
        h = np.zeros(n)

        x_opt = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            solver=qpsolver_name,
            verbose=False,
        )

        if x_opt is None:
            return (
                spectrum_lmfit,
                lmfit_success,
                "QP refinement failed",
                lmfit_nfev,
            )

        refined = np.asarray(x_opt)
        success = lmfit_success
        message = f"lmfit ({lmfit_msg}) + QP refinement OK"
        return refined * log_steps, success, message, lmfit_nfev

    raise ValueError(
        f"Unknown solver library: '{library}'. Use 'cvxpy' or 'qpsolvers'."
    )


# ------------------------------------------------------------------ #
#  Workflow wrapper
# ------------------------------------------------------------------ #


def unfold_parametric(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "leastsq",
    optimizer: str = "lmfit",
    alpha: float = 1e-4,
    alpha_auto: bool = False,
    solver_backend: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the FRUIT-based parametric method.

    The spectrum is modelled as a weighted superposition of thermal,
    epithermal and fast components (Bedogni FRUIT / Pyshkina B3S).

    The ``optimizer`` parameter selects the backend:

    * ``"lmfit"``     -- classic lmfit least-squares (default).
    * ``"cvxpy"``     -- sequential QP via cvxpy (SQP).
    * ``"qpsolvers"`` -- sequential QP via qpsolvers (SQP).
    * ``"combined"``  -- lmfit first, then QP refinement.

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
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess (unused in parametric method).
    initial_params : Optional[Dict[str, float]], optional
        Initial parameter values for the parametric model.
        Keys: b, beta_prime, alpha, beta, P_th, P_epi.
    method : str, optional
        lmfit solver method (default: "leastsq").
    optimizer : str, optional
        Backend optimizer: "lmfit", "cvxpy", "qpsolvers", or
        "combined" (default: "lmfit").
    alpha : float, optional
        Regularization weight for QP-based optimizers (default: 1e-4).
        Also used as initial alpha for lmfit when alpha_auto is True.
    alpha_auto : bool, optional
        If True, select alpha automatically via GCV for the lmfit
        optimizer (default: False).
    solver_backend : str, optional
        QP solver backend string: "auto", "cvxpy", "cvxpy:ECOS",
        "qpsolvers", "qpsolvers:osqp", etc. (default: "auto").
    max_iter : int, optional
        Max SQP iterations for cvxpy/qpsolvers (default: 50).
    tol : float, optional
        Convergence tolerance for SQP (default: 1e-6).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: True).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    log_steps = compute_log_steps(E_MeV, n_energy_bins)
    ln_steps = log_steps * np.log(10)

    if optimizer == "lmfit":
        # lmfit uses grid scan initialization; small Tikhonov
        # regularization (deviation from initial guess) provides
        # numerical stability for this ill-conditioned problem.
        lmfit_alpha = alpha if alpha_auto else 1e-8

        def solve_wrapper(A_mat, b_vec, **kwargs):
            x_opt, success, _message, nfev = solve_parametric(
                A_mat,
                b_vec,
                E_MeV,
                ln_steps,
                initial_params,
                method,
                alpha=lmfit_alpha,
                alpha_auto=alpha_auto,
            )
            return x_opt, nfev, success

        method_name = "parametric"
        extra = {
            "initial_params": initial_params,
            "lmfit_method": method,
            "alpha_auto": alpha_auto,
            "T0": _T0,
            "Ed": _Ed,
        }

    elif optimizer == "cvxpy":

        def solve_wrapper(A_mat, b_vec, **kwargs):
            x_opt, success, _message, nfev = solve_parametric_cvxpy(
                A_mat,
                b_vec,
                E_MeV,
                ln_steps,
                initial_params=initial_params,
                alpha=alpha,
                solver_backend=solver_backend,
                max_iter=max_iter,
                tol=tol,
            )
            return x_opt, nfev, success

        method_name = "parametric_cvxpy"
        extra = {
            "initial_params": initial_params,
            "optimizer": "cvxpy",
            "alpha": alpha,
            "solver_backend": solver_backend,
            "max_iter": max_iter,
            "tol": tol,
            "T0": _T0,
            "Ed": _Ed,
        }

    elif optimizer == "qpsolvers":

        def solve_wrapper(A_mat, b_vec, **kwargs):
            x_opt, success, _message, nfev = solve_parametric_qpsolvers(
                A_mat,
                b_vec,
                E_MeV,
                ln_steps,
                initial_params=initial_params,
                alpha=alpha,
                solver_backend=solver_backend,
                max_iter=max_iter,
                tol=tol,
            )
            return x_opt, nfev, success

        method_name = "parametric_qpsolvers"
        extra = {
            "initial_params": initial_params,
            "optimizer": "qpsolvers",
            "alpha": alpha,
            "solver_backend": solver_backend,
            "max_iter": max_iter,
            "tol": tol,
            "T0": _T0,
            "Ed": _Ed,
        }

    elif optimizer == "combined":

        def solve_wrapper(A_mat, b_vec, **kwargs):
            x_opt, success, _message, nfev = solve_parametric_combined(
                A_mat,
                b_vec,
                E_MeV,
                ln_steps,
                initial_params=initial_params,
                method=method,
                alpha=alpha,
                solver_backend=solver_backend,
            )
            return x_opt, nfev, success

        method_name = "parametric_combined"
        extra = {
            "initial_params": initial_params,
            "optimizer": "combined",
            "lmfit_method": method,
            "alpha": alpha,
            "solver_backend": solver_backend,
            "T0": _T0,
            "Ed": _Ed,
        }

    else:
        raise ValueError(
            f"Unknown optimizer: '{optimizer}'. "
            "Choose from 'lmfit', 'cvxpy', 'qpsolvers', 'combined'."
        )

    x0_default = np.ones(n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))

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
        method_name=method_name,
        extra_output=extra,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    return result
