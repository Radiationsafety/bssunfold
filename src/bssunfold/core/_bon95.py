"""BON95 parametric unfolding family.

Extracted from ``unfold_parametric2.py`` so the BON95 model, shape helpers
and SQP solvers live in their own module. Shared constants and fit helpers
come from ``_parametric_shared``; solver-backend resolution comes from
``_solver_backends``.
"""


import numpy as np
from typing import Dict, Optional, List, Tuple

from ._parametric_shared import (
    logger,
    _Tth,
    _check_fit_quality,
    _clean_edge_bins,
)
from ._solver_backends import (
    _parse_solver_backend,
    _resolve_cvxpy_solvers,
    _resolve_qpsolver_name,
)
# ------------------------------------------------------------------ #
#  BON95 parametric model
# ------------------------------------------------------------------ #


def _Fth(E: np.ndarray, Tth: float = _Tth) -> np.ndarray:
    """Thermal component: Xth^(3/2) * exp(-Xth), Xth = E/Tth."""
    Xth = E / Tth
    return Xth**1.5 * np.exp(-Xth)


def _Fepi(E: np.ndarray, b: float, Tth: float = _Tth) -> np.ndarray:
    """Epithermal component: E^(-b) * (1 - exp(-Xth))."""
    Xth = E / Tth
    return E ** (-b) * (1.0 - np.exp(-Xth))


def _Fint(E: np.ndarray, Tth: float = _Tth) -> np.ndarray:
    """Intermediate component: (1 - exp(-Xth))."""
    Xth = E / Tth
    return 1.0 - np.exp(-Xth)


def _Ff(E: np.ndarray, Tf: float, c: float) -> np.ndarray:
    """Fast component: Xf^(3/2) * exp(-Xf), Xf = (E/Tf)^c."""
    Xf = (E / Tf) ** c
    return Xf**1.5 * np.exp(-Xf)


def bon95_model(
    E: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
) -> np.ndarray:
    """Combined four-component BON95 neutron spectrum.

    Returns E * Phi(E), i.e. the lethargy spectrum.

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    b : float
        Epithermal power-law exponent.
    Tf : float
        Fast peak characteristic energy (MeV).
    c : float
        Fast peak width parameter.
    a1, a2, a3, a4 : float
        Linear weights of thermal, epithermal, intermediate, fast.

    Returns
    -------
    np.ndarray
        Lethargy spectrum E * Phi(E).
    """
    E = np.asarray(E, dtype=float)
    return a1 * _Fth(E) + a2 * _Fepi(E, b) + a3 * _Fint(E) + a4 * _Ff(E, Tf, c)


def bon95_spectrum(
    E: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
) -> np.ndarray:
    """Neutron spectrum Phi(E) from the BON95 model.

    Returns Phi(E) = (a1*Fth + a2*Fepi + a3*Fint + a4*Ff) / E.
    """
    E = np.asarray(E, dtype=float)
    lethargy = bon95_model(E, b, Tf, c, a1, a2, a3, a4)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(E > 0, lethargy / E, 0.0)
    return phi


# ------------------------------------------------------------------ #
#  Weighted NLS for linear coefficients
# ------------------------------------------------------------------ #


def _solve_linear_coefficients(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b: float,
    Tf: float,
    c: float,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Solve for optimal linear coefficients a1..a4 given shape params.

    The model is: M_i = sum_j A_i(E_j) * Phi(E_j) * d(ln E)_j
    where Phi(E) = (a1*Fth + a2*Fepi + a3*Fint + a4*Ff) / E.

    This is linear in a1..a4, so we solve the weighted least-squares
    problem: min ||W*(B*a - b)||^2 where B_ij = A_i(E_j)*F_k(E_j)/E_j*d(lnE)_j.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths d(ln E).
    b, Tf, c : float
        Shape parameters.
    weights : np.ndarray, optional
        Diagonal weights (1/sigma_i^2). If None, uniform weights.

    Returns
    -------
    Tuple[np.ndarray, float]
        (coefficients [a1,a2,a3,a4], chi2 value)
    """
    n_det = A_matrix.shape[0]

    # Build the four basis columns: B_ik = sum_j A_i(E_j) * F_k(E_j) / E_j * ln_steps_j
    E_safe = np.where(E > 0, E, 1.0)
    F_cols = np.column_stack(
        [
            _Fth(E),
            _Fepi(E, b),
            _Fint(E),
            _Ff(E, Tf, c),
        ]
    )  # (n_energy, 4)

    # Each column of B: B_i_k = sum_j A_i_j * F_k_j / E_j * ln_steps_j
    # F_cols/E_safe divides each row by corresponding E, then * ln_steps
    weighted_F = (
        F_cols / E_safe[:, np.newaxis] * ln_steps[:, np.newaxis]
    )  # (n_energy, 4)
    B = A_matrix @ weighted_F  # (n_det, 4)

    if weights is None:
        weights = np.ones(n_det)

    W = np.diag(np.sqrt(weights))
    Bw = W @ B
    bw = W @ b_readings

    # Solve via least-squares
    result, _, _, _ = np.linalg.lstsq(Bw, bw, rcond=None)
    a = np.maximum(result, 0.0)  # enforce non-negativity of coefficients

    # Compute chi2
    residual = B @ a - b_readings
    chi2 = (
        np.mean((residual**2) * weights)
        if np.all(weights > 0)
        else np.mean(residual**2)
    )

    return a, chi2


# ------------------------------------------------------------------ #
#  Grid search + NLS parametric fit
# ------------------------------------------------------------------ #

# Default grid ranges for shape parameters
_DEFAULT_B_RANGE = (0.5, 2.0, 5)  # (min, max, n_points)
_DEFAULT_TF_RANGE = (0.5, 10.0, 5)  # (min MeV, max MeV, n_points)
_DEFAULT_C_RANGE = (0.5, 3.0, 4)  # (min, max, n_points)


def solve_bon95_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    b_meas: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> Tuple[Dict[str, float], float, List[Dict[str, float]]]:
    """Grid search + NLS for the BON95 parametric model.

    Scans over (b, Tf, c) shape parameters, solves for optimal linear
    coefficients (a1..a4) at each grid point via weighted NLS, and
    returns the best result.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    b_range, Tf_range, c_range : tuple
        (min, max, n_points) for each shape parameter.
    b_meas : np.ndarray, optional
        Measurement uncertainties (sigma_i). Used as weights.
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    Tuple[dict, float, list]
        (best_params, best_chi2, top_candidates)
        best_params keys: b, Tf, c, a1, a2, a3, a4
    """
    b_vals = np.linspace(b_range[0], b_range[1], b_range[2])
    Tf_vals = np.linspace(Tf_range[0], Tf_range[1], Tf_range[2])
    c_vals = np.linspace(c_range[0], c_range[1], c_range[2])

    # Weights from measurement uncertainties
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas**2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    candidates = []

    for b_val in b_vals:
        for Tf_val in Tf_vals:
            for c_val in c_vals:
                a, chi2 = _solve_linear_coefficients(
                    A_matrix,
                    b_readings,
                    E,
                    ln_steps,
                    b_val,
                    Tf_val,
                    c_val,
                    weights,
                )
                candidates.append(
                    {
                        "b": b_val,
                        "Tf": Tf_val,
                        "c": c_val,
                        "a1": a[0],
                        "a2": a[1],
                        "a3": a[2],
                        "a4": a[3],
                        "chi2": chi2,
                    }
                )

    # Sort by chi2 (best first)
    candidates.sort(key=lambda x: x["chi2"])
    best = candidates[0]

    return best, best["chi2"], candidates[:top_n]


# ------------------------------------------------------------------ #
#  Shape parameter bounds and Jacobian for SQP solvers
# ------------------------------------------------------------------ #

_BON95_SHAPE_NAMES = ["b", "Tf", "c"]

_BON95_SHAPE_BOUNDS = {
    "b": (0.5, 2.0),
    "Tf": (0.5, 10.0),
    "c": (0.5, 3.0),
}


def _get_bon95_shape_bounds():
    """Return {name: (lo, hi)} for shape parameters."""
    return dict(_BON95_SHAPE_BOUNDS)


def _clamp_bon95_shape(params, bounds):
    """Clamp shape parameter values to stay within bounds."""
    clamped = dict(params)
    for name, (lo, hi) in bounds.items():
        clamped[name] = max(lo, min(hi, clamped[name]))
    return clamped


def _solve_shape_nls(
    A_matrix,
    b_readings,
    E,
    ln_steps,
    b,
    Tf,
    c,
    weights,
):
    """Solve for optimal (a1..a4) given shape params, return spectrum + chi2."""
    a, chi2 = _solve_linear_coefficients(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        b,
        Tf,
        c,
        weights,
    )
    phi = bon95_spectrum(E, b, Tf, c, a[0], a[1], a[2], a[3])
    phi = np.maximum(phi, 0.0)
    phi = _clean_edge_bins(phi)
    spectrum = phi * ln_steps
    return spectrum, chi2, a


def _compute_bon95_shape_jacobian(
    A_matrix,
    b_readings,
    E,
    ln_steps,
    params,
    weights,
    delta=1e-6,
):
    """Jacobian of spectrum w.r.t. shape params (b, Tf, c).

    At each perturbation, re-solves for optimal (a1..a4) to get the
    correct gradient of the best-fit spectrum.

    Returns
    -------
    J : np.ndarray
        Jacobian matrix (n_energy x 3).
    residual : np.ndarray
        Current residual vector (n_det,).
    """
    bounds = _get_bon95_shape_bounds()

    # Current spectrum and residual
    s0, _, _ = _solve_shape_nls(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        params["b"],
        params["Tf"],
        params["c"],
        weights,
    )
    residual = A_matrix @ s0 - b_readings

    J = np.zeros((len(E), 3))

    for i, name in enumerate(_BON95_SHAPE_NAMES):
        lo, hi = bounds[name]
        p_val = params[name]

        # Forward difference with boundary handling
        d = delta
        if p_val + d > hi:
            d = max(0, hi - p_val) * 0.5
        if d < 1e-15:
            # Backward difference
            d = delta
            if p_val - d >= lo:
                p_pert = dict(params)
                p_pert[name] = p_val - d
                s_pert, _, _ = _solve_shape_nls(
                    A_matrix,
                    b_readings,
                    E,
                    ln_steps,
                    p_pert["b"],
                    p_pert["Tf"],
                    p_pert["c"],
                    weights,
                )
                J[:, i] = (s0 - s_pert) / d
            else:
                J[:, i] = 0.0
            continue

        p_pert = dict(params)
        p_pert[name] = p_val + d
        s_pert, _, _ = _solve_shape_nls(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            p_pert["b"],
            p_pert["Tf"],
            p_pert["c"],
            weights,
        )
        J[:, i] = (s_pert - s0) / d

    return J, residual


# ------------------------------------------------------------------ #
#  Solver backend helpers
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  SQP solver via cvxpy
# ------------------------------------------------------------------ #


def solve_bon95_cvxpy(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    initial_params: Optional[Dict[str, float]] = None,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve BON95 parametric fitting via sequential QP using cvxpy.

    Optimizes shape parameters (b, Tf, c) via SQP. At each iteration,
    the nonlinear model is linearized w.r.t. shape params and the
    resulting QP is solved with cvxpy. The linear coefficients (a1..a4)
    are re-solved by NLS at each step.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    b_meas : np.ndarray, optional
        Measurement uncertainties for weighting.
    initial_params : dict, optional
        Starting shape params {b, Tf, c}. If None, grid scan is used.
    alpha : float
        Tikhonov regularization weight (default: 1e-4).
    solver_backend : str
        CVXPY solver backend (default: "auto").
    max_iter : int
        Maximum SQP iterations (default: 50).
    tol : float
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
            "cvxpy is required for solve_bon95_cvxpy. "
            "Install with: pip install cvxpy"
        ) from e

    _, backend = _parse_solver_backend(solver_backend)
    solvers_to_try = _resolve_cvxpy_solvers(backend)

    # Weights
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas**2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    # Initial params: use grid scan if not provided
    if initial_params is not None:
        params = {k: initial_params[k] for k in _BON95_SHAPE_NAMES}
    else:
        best, _, _ = solve_bon95_parametric(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_meas=b_meas,
            top_n=1,
        )
        params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}

    params = _clamp_bon95_shape(params, _get_bon95_shape_bounds())
    n_params = 3
    message = ""
    nfev = 0

    for k in range(max_iter):
        # Current spectrum via NLS for a1..a4
        spectrum_k, _, _ = _solve_shape_nls(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            params["b"],
            params["Tf"],
            params["c"],
            weights,
        )
        nfev += 1

        residual = A_matrix @ spectrum_k - b_readings
        if np.linalg.norm(residual) < tol:
            _check_fit_quality(
                np.linalg.norm(residual), b_readings, "bon95_cvxpy"
            )
            return spectrum_k, True, f"Converged in {k} iterations", nfev

        # Jacobian w.r.t. shape params
        J, _ = _compute_bon95_shape_jacobian(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            params,
            weights,
        )
        nfev += 2 * n_params  # finite differences
        A_eff = A_matrix @ J  # (n_det x 3) effective forward operator

        # Build and solve QP: min ||A_eff @ delta + residual||^2 + alpha*||delta||^2
        delta = cp.Variable(n_params)
        data_term = cp.sum_squares(A_eff @ delta + residual)
        penalty_term = alpha * cp.sum_squares(delta)
        objective = cp.Minimize(data_term + penalty_term)

        bounds = _get_bon95_shape_bounds()
        constraints = []
        for i, name in enumerate(_BON95_SHAPE_NAMES):
            lo, hi = bounds[name]
            constraints.append(delta[i] >= lo - params[name])
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
        for i, name in enumerate(_BON95_SHAPE_NAMES):
            params[name] += delta_val[i]
        params = _clamp_bon95_shape(params, bounds)

        if np.linalg.norm(delta_val) < tol:
            spectrum_final, _, _ = _solve_shape_nls(
                A_matrix,
                b_readings,
                E,
                ln_steps,
                params["b"],
                params["Tf"],
                params["c"],
                weights,
            )
            _check_fit_quality(
                np.linalg.norm(A_matrix @ spectrum_final - b_readings),
                b_readings,
                "bon95_cvxpy",
            )
            return (
                spectrum_final,
                True,
                f"Converged in {k + 1} iterations",
                nfev,
            )

    # Final spectrum
    spectrum_final, _, _ = _solve_shape_nls(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        params["b"],
        params["Tf"],
        params["c"],
        weights,
    )
    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_final - b_readings),
        b_readings,
        "bon95_cvxpy",
    )
    if not message:
        message = f"Max iterations ({max_iter}) reached"
    return spectrum_final, False, message, nfev


# ------------------------------------------------------------------ #
#  SQP solver via qpsolvers
# ------------------------------------------------------------------ #


def solve_bon95_qpsolvers(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    initial_params: Optional[Dict[str, float]] = None,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve BON95 parametric fitting via sequential QP using qpsolvers.

    Same algorithm as solve_bon95_cvxpy but uses qpsolvers backends
    (OSQP, ECOS, etc.).

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_det x n_energy).
    b_readings : np.ndarray
        Measured readings (n_det,).
    E : np.ndarray
        Energy grid in MeV.
    ln_steps : np.ndarray
        Logarithmic bin widths.
    b_meas : np.ndarray, optional
        Measurement uncertainties for weighting.
    initial_params : dict, optional
        Starting shape params {b, Tf, c}.
    alpha : float
        Tikhonov regularization weight (default: 1e-4).
    solver_backend : str
        QP solver backend (default: "auto").
    max_iter : int
        Maximum SQP iterations (default: 50).
    tol : float
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as e:
        raise ImportError(
            "qpsolvers is required for solve_bon95_qpsolvers. "
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
                f"No QP solver available. Available: {available_solvers}"
            )

    # Weights
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas**2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    # Initial params
    if initial_params is not None:
        params = {k: initial_params[k] for k in _BON95_SHAPE_NAMES}
    else:
        best, _, _ = solve_bon95_parametric(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            b_meas=b_meas,
            top_n=1,
        )
        params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}

    params = _clamp_bon95_shape(params, _get_bon95_shape_bounds())
    n_params = 3
    message = ""
    nfev = 0

    for k in range(max_iter):
        spectrum_k, _, _ = _solve_shape_nls(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            params["b"],
            params["Tf"],
            params["c"],
            weights,
        )
        nfev += 1

        residual = A_matrix @ spectrum_k - b_readings
        if np.linalg.norm(residual) < tol:
            _check_fit_quality(
                np.linalg.norm(residual), b_readings, "bon95_qpsolvers"
            )
            return spectrum_k, True, f"Converged in {k} iterations", nfev

        J, _ = _compute_bon95_shape_jacobian(
            A_matrix,
            b_readings,
            E,
            ln_steps,
            params,
            weights,
        )
        nfev += 2 * n_params
        A_eff = A_matrix @ J

        # QP: min 0.5*delta^T P delta + q^T delta  s.t. G delta <= h
        P = csc_matrix(A_eff.T @ A_eff + alpha * np.eye(n_params))
        q = A_eff.T @ residual

        bounds = _get_bon95_shape_bounds()
        G_rows = []
        h_rows = []
        for i, name in enumerate(_BON95_SHAPE_NAMES):
            lo, hi = bounds[name]
            row_lo = np.zeros(n_params)
            row_lo[i] = -1.0
            G_rows.append(row_lo)
            h_rows.append(-(lo - params[name]))
            row_hi = np.zeros(n_params)
            row_hi[i] = 1.0
            G_rows.append(row_hi)
            h_rows.append(hi - params[name])

        G = csc_matrix(np.vstack(G_rows))
        h = np.array(h_rows)

        try:
            delta_val = solve_qp(
                P=P, q=q, G=G, h=h, solver=solver_name, verbose=False
            )
        except Exception as exc:
            logger.debug(
                "QP solver %s failed at iter %d: %s", solver_name, k, exc
            )
            message = f"QP subproblem failed at iteration {k}"
            break

        if delta_val is None:
            message = f"QP solver returned None at iteration {k}"
            break

        delta_val = np.asarray(delta_val)
        for i, name in enumerate(_BON95_SHAPE_NAMES):
            params[name] += delta_val[i]
        params = _clamp_bon95_shape(params, bounds)

        if np.linalg.norm(delta_val) < tol:
            spectrum_final, _, _ = _solve_shape_nls(
                A_matrix,
                b_readings,
                E,
                ln_steps,
                params["b"],
                params["Tf"],
                params["c"],
                weights,
            )
            _check_fit_quality(
                np.linalg.norm(A_matrix @ spectrum_final - b_readings),
                b_readings,
                "bon95_qpsolvers",
            )
            return (
                spectrum_final,
                True,
                f"Converged in {k + 1} iterations",
                nfev,
            )

    spectrum_final, _, _ = _solve_shape_nls(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        params["b"],
        params["Tf"],
        params["c"],
        weights,
    )
    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_final - b_readings),
        b_readings,
        "bon95_qpsolvers",
    )
    if not message:
        message = f"Max iterations ({max_iter}) reached"
    return spectrum_final, False, message, nfev


# ------------------------------------------------------------------ #
#  Combined: grid search first, then SQP refinement
# ------------------------------------------------------------------ #


def solve_bon95_combined(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter_qp: int = 50,
    tol_qp: float = 1e-6,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve BON95: grid search first, then SQP refinement.

    1. Grid search for best starting (b, Tf, c).
    2. SQP refinement via cvxpy or qpsolvers.

    Parameters
    ----------
    A_matrix, b_readings, E, ln_steps : as usual.
    b_meas : np.ndarray, optional
        Measurement uncertainties.
    alpha : float
        Tikhonov regularization for SQP (default: 1e-4).
    solver_backend : str
        QP backend: "auto", "cvxpy:ECOS", "qpsolvers:osqp", etc.
    max_iter_qp : int
        Max SQP iterations (default: 50).
    tol_qp : float
        SQP convergence tolerance (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    # Step 1: Grid search
    best, _, _ = solve_bon95_parametric(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        b_meas=b_meas,
        top_n=1,
    )
    init_params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}
    nfev_grid = 1

    # Step 2: SQP refinement
    library, _ = _parse_solver_backend(solver_backend)

    if library in ("cvxpy", "auto"):
        try:
            result = solve_bon95_cvxpy(
                A_matrix,
                b_readings,
                E,
                ln_steps,
                b_meas=b_meas,
                initial_params=init_params,
                alpha=alpha,
                solver_backend=solver_backend,
                max_iter=max_iter_qp,
                tol=tol_qp,
            )
            spectrum, success, msg, nfev_qp = result
            return (
                spectrum,
                success,
                f"grid + cvxpy ({msg})",
                nfev_grid + nfev_qp,
            )
        except ImportError:
            if library == "cvxpy":
                raise

    # Fallback to qpsolvers
    result = solve_bon95_qpsolvers(
        A_matrix,
        b_readings,
        E,
        ln_steps,
        b_meas=b_meas,
        initial_params=init_params,
        alpha=alpha,
        solver_backend=solver_backend,
        max_iter=max_iter_qp,
        tol=tol_qp,
    )
    spectrum, success, msg, nfev_qp = result
    return spectrum, success, f"grid + qpsolvers ({msg})", nfev_grid + nfev_qp
