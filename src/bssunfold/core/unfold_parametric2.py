"""BON95-based parametric unfolding method.

This module implements the parametric neutron spectrum reconstruction
method described in:

  - A.V. Sannikov, "BON95, a universal user-independent unfolding code
    for low informative neutron spectrometers", GSF report, Munich, 1995.
  - V.V. Babintsev et al., "Measurement of neutron spectrum at the
    'Neutron' test bench by Bonner spectrometer with activation
    detectors", NRC Kurchatov Institute - IHEP Preprint 2022-4.
  - A.V. Sannikov et al., "Multi-sphere neutron spectrometer based
    on the serial instrument RSU-01", Apparatus, No.1, pp.62-69, 2009.

The spectrum E * Phi(E) is represented as a linear combination of four
components:

  Thermal    (E < 0.1 MeV):  Fth  = Xth^(3/2) * exp(-Xth)
  Epithermal (E < 10 MeV):   Fepi = E^(-b) * (1 - exp(-Xth))
  Intermediate (E < 10 MeV): Fint = (1 - exp(-Xth))
  Fast       (E > 0.1 MeV):  Ff   = Xf^(3/2) * exp(-Xf)

  E * Phi(E) = a1*Fth + a2*Fepi + a3*Fint + a4*Ff

where Xth = E/Tth (Tth = 0.035 eV = 3.5e-8 MeV) and
Xf = (E/Tf)^c.

Free shape parameters (b, Tf, c) are found by grid search.
Linear coefficients (a1..a4) are solved by weighted NLS.

After parametric fitting, the result is refined by directed-divergence
(I-divergence / Itakura-Saito) iterations.
"""

import logging
import warnings
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = [
    "solve_bon95_parametric",
    "solve_bon95_cvxpy",
    "solve_bon95_qpsolvers",
    "solve_bon95_combined",
    "directed_divergence_iteration",
    "solve_parametric2",
    "unfold_parametric2",
]

# Fixed constants from the BON95 papers
_Tth = 3.5e-8   # Thermal peak temperature (MeV), = 0.035 eV

# Energy region boundaries for component masking
_THERMAL_MAX_BON95 = 0.1    # MeV — thermal + epithermal dominant
_FAST_MIN_BON95 = 0.1       # MeV — fast component starts

_RESIDUAL_WARN_THRESHOLD = 10.0

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  BON95 parametric model
# ------------------------------------------------------------------ #

def _Fth(E: np.ndarray, Tth: float = _Tth) -> np.ndarray:
    """Thermal component: Xth^(3/2) * exp(-Xth), Xth = E/Tth."""
    Xth = E / Tth
    return Xth ** 1.5 * np.exp(-Xth)


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
    return Xf ** 1.5 * np.exp(-Xf)


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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    F_cols = np.column_stack([
        _Fth(E),
        _Fepi(E, b),
        _Fint(E),
        _Ff(E, Tf, c),
    ])  # (n_energy, 4)

    # Each column of B: B_i_k = sum_j A_i_j * F_k_j / E_j * ln_steps_j
    # F_cols/E_safe divides each row by corresponding E, then * ln_steps
    weighted_F = F_cols / E_safe[:, np.newaxis] * ln_steps[:, np.newaxis]  # (n_energy, 4)
    B = A_matrix @ weighted_F  # (n_det, 4)

    if weights is None:
        weights = np.ones(n_det)

    W = np.diag(np.sqrt(weights))
    Bw = W @ B
    bw = W @ b_readings

    # Solve via least-squares
    result, res, rank, sv = np.linalg.lstsq(Bw, bw, rcond=None)
    a = np.maximum(result, 0.0)  # enforce non-negativity of coefficients

    # Compute chi2
    residual = B @ a - b_readings
    chi2 = np.mean((residual ** 2) * weights) if np.all(weights > 0) else np.mean(residual ** 2)

    return a, chi2


# ------------------------------------------------------------------ #
#  Grid search + NLS parametric fit
# ------------------------------------------------------------------ #

# Default grid ranges for shape parameters
_DEFAULT_B_RANGE = (0.5, 2.0, 5)      # (min, max, n_points)
_DEFAULT_TF_RANGE = (0.5, 10.0, 5)    # (min MeV, max MeV, n_points)
_DEFAULT_C_RANGE = (0.5, 3.0, 4)      # (min, max, n_points)


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
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    candidates = []

    for b_val in b_vals:
        for Tf_val in Tf_vals:
            for c_val in c_vals:
                a, chi2 = _solve_linear_coefficients(
                    A_matrix, b_readings, E, ln_steps,
                    b_val, Tf_val, c_val, weights,
                )
                candidates.append({
                    'b': b_val,
                    'Tf': Tf_val,
                    'c': c_val,
                    'a1': a[0],
                    'a2': a[1],
                    'a3': a[2],
                    'a4': a[3],
                    'chi2': chi2,
                })

    # Sort by chi2 (best first)
    candidates.sort(key=lambda x: x['chi2'])
    best = candidates[0]

    return best, best['chi2'], candidates[:top_n]


# ------------------------------------------------------------------ #
#  Shape parameter bounds and Jacobian for SQP solvers
# ------------------------------------------------------------------ #

_BON95_SHAPE_NAMES = ["b", "Tf", "c"]

_BON95_SHAPE_BOUNDS = {
    "b":  (0.5, 2.0),
    "Tf": (0.5, 10.0),
    "c":  (0.5, 3.0),
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
    A_matrix, b_readings, E, ln_steps, b, Tf, c, weights,
):
    """Solve for optimal (a1..a4) given shape params, return spectrum + chi2."""
    a, chi2 = _solve_linear_coefficients(
        A_matrix, b_readings, E, ln_steps, b, Tf, c, weights,
    )
    phi = bon95_spectrum(E, b, Tf, c, a[0], a[1], a[2], a[3])
    phi = np.maximum(phi, 0.0)
    phi = _clean_edge_bins(phi)
    spectrum = phi * ln_steps
    return spectrum, chi2, a


def _compute_bon95_shape_jacobian(
    A_matrix, b_readings, E, ln_steps, params, weights, delta=1e-6,
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
        A_matrix, b_readings, E, ln_steps,
        params["b"], params["Tf"], params["c"], weights,
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
                    A_matrix, b_readings, E, ln_steps,
                    p_pert["b"], p_pert["Tf"], p_pert["c"], weights,
                )
                J[:, i] = (s0 - s_pert) / d
            else:
                J[:, i] = 0.0
            continue

        p_pert = dict(params)
        p_pert[name] = p_val + d
        s_pert, _, _ = _solve_shape_nls(
            A_matrix, b_readings, E, ln_steps,
            p_pert["b"], p_pert["Tf"], p_pert["c"], weights,
        )
        J[:, i] = (s_pert - s0) / d

    return J, residual


# ------------------------------------------------------------------ #
#  Solver backend helpers
# ------------------------------------------------------------------ #

def _parse_solver_backend(solver_backend):
    """Parse 'auto', 'cvxpy', 'cvxpy:ECOS', 'qpsolvers:osqp' etc."""
    if solver_backend == "auto":
        return "auto", "default"
    parts = solver_backend.split(":", 1)
    library = parts[0]
    backend = parts[1] if len(parts) > 1 else "default"
    return library, backend


def _resolve_cvxpy_solvers(backend):
    """Return list of cvxpy solvers to try."""
    try:
        import cvxpy as cp
        installed = cp.installed_solvers()
    except ImportError:
        installed = []
    if backend == "default":
        return [s for s in ["ECOS", "SCS", "CLARABEL"] if s in installed] or ["ECOS"]
    fallbacks = [s for s in ["ECOS", "SCS", "CLARABEL"] if s != backend]
    return [backend] + fallbacks


def _resolve_qpsolver_name(backend):
    """Return the qpsolvers backend name to use."""
    if backend != "default":
        return backend
    try:
        from qpsolvers import available_solvers
        if "osqp" in available_solvers:
            return "osqp"
        if "ecos" in available_solvers:
            return "ecos"
    except ImportError:
        pass
    return "osqp"


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
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    # Initial params: use grid scan if not provided
    if initial_params is not None:
        params = {k: initial_params[k] for k in _BON95_SHAPE_NAMES}
    else:
        best, _, _ = solve_bon95_parametric(
            A_matrix, b_readings, E, ln_steps, b_meas=b_meas, top_n=1,
        )
        params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}

    params = _clamp_bon95_shape(params, _get_bon95_shape_bounds())
    n_params = 3
    message = ""
    nfev = 0

    for k in range(max_iter):
        # Current spectrum via NLS for a1..a4
        spectrum_k, chi2_k, _ = _solve_shape_nls(
            A_matrix, b_readings, E, ln_steps,
            params["b"], params["Tf"], params["c"], weights,
        )
        nfev += 1

        residual = A_matrix @ spectrum_k - b_readings
        if np.linalg.norm(residual) < tol:
            _check_fit_quality(np.linalg.norm(residual), b_readings, "bon95_cvxpy")
            return spectrum_k, True, f"Converged in {k} iterations", nfev

        # Jacobian w.r.t. shape params
        J, _ = _compute_bon95_shape_jacobian(
            A_matrix, b_readings, E, ln_steps, params, weights,
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
                A_matrix, b_readings, E, ln_steps,
                params["b"], params["Tf"], params["c"], weights,
            )
            _check_fit_quality(
                np.linalg.norm(A_matrix @ spectrum_final - b_readings),
                b_readings, "bon95_cvxpy",
            )
            return spectrum_final, True, f"Converged in {k + 1} iterations", nfev

    # Final spectrum
    spectrum_final, _, _ = _solve_shape_nls(
        A_matrix, b_readings, E, ln_steps,
        params["b"], params["Tf"], params["c"], weights,
    )
    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_final - b_readings),
        b_readings, "bon95_cvxpy",
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
            raise ValueError(f"No QP solver available. Available: {available_solvers}")

    # Weights
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(A_matrix.shape[0])

    # Initial params
    if initial_params is not None:
        params = {k: initial_params[k] for k in _BON95_SHAPE_NAMES}
    else:
        best, _, _ = solve_bon95_parametric(
            A_matrix, b_readings, E, ln_steps, b_meas=b_meas, top_n=1,
        )
        params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}

    params = _clamp_bon95_shape(params, _get_bon95_shape_bounds())
    n_params = 3
    message = ""
    nfev = 0

    for k in range(max_iter):
        spectrum_k, chi2_k, _ = _solve_shape_nls(
            A_matrix, b_readings, E, ln_steps,
            params["b"], params["Tf"], params["c"], weights,
        )
        nfev += 1

        residual = A_matrix @ spectrum_k - b_readings
        if np.linalg.norm(residual) < tol:
            _check_fit_quality(np.linalg.norm(residual), b_readings, "bon95_qpsolvers")
            return spectrum_k, True, f"Converged in {k} iterations", nfev

        J, _ = _compute_bon95_shape_jacobian(
            A_matrix, b_readings, E, ln_steps, params, weights,
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
            delta_val = solve_qp(P=P, q=q, G=G, h=h, solver=solver_name, verbose=False)
        except Exception as exc:
            logger.debug("QP solver %s failed at iter %d: %s", solver_name, k, exc)
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
                A_matrix, b_readings, E, ln_steps,
                params["b"], params["Tf"], params["c"], weights,
            )
            _check_fit_quality(
                np.linalg.norm(A_matrix @ spectrum_final - b_readings),
                b_readings, "bon95_qpsolvers",
            )
            return spectrum_final, True, f"Converged in {k + 1} iterations", nfev

    spectrum_final, _, _ = _solve_shape_nls(
        A_matrix, b_readings, E, ln_steps,
        params["b"], params["Tf"], params["c"], weights,
    )
    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_final - b_readings),
        b_readings, "bon95_qpsolvers",
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
        A_matrix, b_readings, E, ln_steps, b_meas=b_meas, top_n=1,
    )
    init_params = {"b": best["b"], "Tf": best["Tf"], "c": best["c"]}
    nfev_grid = 1

    # Step 2: SQP refinement
    library, _ = _parse_solver_backend(solver_backend)

    if library == "cvxpy" or library == "auto":
        try:
            result = solve_bon95_cvxpy(
                A_matrix, b_readings, E, ln_steps,
                b_meas=b_meas, initial_params=init_params,
                alpha=alpha, solver_backend=solver_backend,
                max_iter=max_iter_qp, tol=tol_qp,
            )
            spectrum, success, msg, nfev_qp = result
            return spectrum, success, f"grid + cvxpy ({msg})", nfev_grid + nfev_qp
        except ImportError:
            if library == "cvxpy":
                raise

    # Fallback to qpsolvers
    result = solve_bon95_qpsolvers(
        A_matrix, b_readings, E, ln_steps,
        b_meas=b_meas, initial_params=init_params,
        alpha=alpha, solver_backend=solver_backend,
        max_iter=max_iter_qp, tol=tol_qp,
    )
    spectrum, success, msg, nfev_qp = result
    return spectrum, success, f"grid + qpsolvers ({msg})", nfev_grid + nfev_qp

def directed_divergence_iteration(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    phi0: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
    tol_rel: float = 1e-6,
) -> Tuple[np.ndarray, int, float, bool]:
    """Refine spectrum via directed-divergence (I-divergence) iterations.

    Multiplicative update rule (Itakura-Saito / Csiszar-Tusnady):

        phi_{k+1}(E_j) = phi_k(E_j) * numerator / denominator

    where:
        numerator = sum_i [ A_i(E_j) * M_i / M_p_i ]
        denominator = sum_i [ A_i(E_j) ]

    and M_p_i = sum_j A_i(E_j) * phi_k(E_j) * d(ln E)_j is the
    computed reading for detector i.

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
    phi0 : np.ndarray
        Initial spectrum guess (n_energy,).
    b_meas : np.ndarray, optional
        Measurement uncertainties. If None, uniform weights.
    max_iter : int
        Maximum iterations (default: 200).
    tol_chi2 : float
        Stop when chi2 < tol_chi2 (default: 1.0).
    tol_rel : float
        Stop when relative change in spectrum < tol_rel (default: 1e-6).

    Returns
    -------
    Tuple[np.ndarray, int, float, bool]
        (spectrum, n_iterations, final_chi2, converged)
    """
    phi = np.copy(phi0)
    phi = np.maximum(phi, 1e-30)  # avoid division by zero

    n_det = A_matrix.shape[0]
    if b_meas is not None:
        weights = np.where(b_meas > 0, 1.0 / (b_meas ** 2), 1.0)
    else:
        weights = np.ones(n_det)

    # Precompute denominator: sum_i A_i(E_j) for each energy bin
    denom = np.sum(A_matrix, axis=0)  # (n_energy,)
    denom = np.maximum(denom, 1e-30)

    for iteration in range(max_iter):
        # Compute model readings: M_p_i = sum_j A_i_j * phi_j * ln_steps_j
        M_p = A_matrix @ (phi * ln_steps)  # (n_det,)

        # Avoid division by zero in ratio
        M_p_safe = np.maximum(M_p, 1e-30)

        # Chi-squared
        residual = M_p - b_readings
        chi2 = np.mean(residual ** 2 * weights)

        # Check convergence
        if chi2 < tol_chi2:
            return phi, iteration + 1, chi2, True

        # Multiplicative update
        # numerator_j = sum_i [ A_i_j * M_i / M_p_i ]
        ratios = b_readings / M_p_safe  # (n_det,)
        numerator = A_matrix.T @ ratios  # (n_energy,)

        # phi_{k+1} = phi_k * numerator / denominator
        phi_new = phi * numerator / denom
        phi_new = np.maximum(phi_new, 1e-30)

        # Check relative change
        rel_change = np.max(np.abs(phi_new - phi)) / (np.max(phi) + 1e-30)
        phi = phi_new

        if rel_change < tol_rel:
            # Recompute chi2 with updated phi
            phi = _clean_edge_bins(phi)
            M_p_final = A_matrix @ (phi * ln_steps)
            chi2_final = np.mean((M_p_final - b_readings) ** 2 * weights)
            return phi, iteration + 1, chi2_final, True

    # Final chi2
    phi = _clean_edge_bins(phi)
    M_p_final = A_matrix @ (phi * ln_steps)
    chi2_final = np.mean((M_p_final - b_readings) ** 2 * weights)
    converged = chi2_final < tol_chi2

    return phi, max_iter, chi2_final, converged


# ------------------------------------------------------------------ #
#  Full pipeline solver
# ------------------------------------------------------------------ #

def solve_parametric2(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    ln_steps: np.ndarray,
    b_meas: Optional[np.ndarray] = None,
    optimizer: str = "grid",
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter_qp: int = 50,
    tol_qp: float = 1e-6,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding using the full BON95 parametric pipeline.

    1. Parametric fit using the selected optimizer.
    2. Directed-divergence iteration refinement.

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
        Measurement uncertainties for weighted NLS.
    optimizer : str
        Parametric fit optimizer (default: "grid"):
        - ``"grid"``      -- grid search + NLS (default, no extra deps).
        - ``"cvxpy"``     -- SQP via cvxpy.
        - ``"qpsolvers"`` -- SQP via qpsolvers.
        - ``"combined"``  -- grid search + SQP refinement.
    b_range, Tf_range, c_range : tuple
        Grid search ranges for shape parameters (used by "grid" and "combined").
    alpha : float
        Tikhonov regularization for SQP optimizers (default: 1e-4).
    solver_backend : str
        QP backend for SQP optimizers (default: "auto").
    max_iter_qp : int
        Max SQP iterations for QP-based optimizers (default: 50).
    tol_qp : float
        SQP convergence tolerance (default: 1e-6).
    max_iter : int
        Max directed-divergence iterations (default: 200).
    tol_chi2 : float
        Chi-squared convergence threshold (default: 1.0).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    nfev = 0

    # Step 1: Parametric fit using selected optimizer
    if optimizer == "grid":
        best_params, best_chi2, top_candidates = solve_bon95_parametric(
            A_matrix, b_readings, E, ln_steps,
            b_range=b_range, Tf_range=Tf_range, c_range=c_range,
            b_meas=b_meas, top_n=5,
        )
        phi_param = bon95_spectrum(
            E, best_params['b'], best_params['Tf'], best_params['c'],
            best_params['a1'], best_params['a2'], best_params['a3'], best_params['a4'],
        )
        nfev = len(top_candidates)

    elif optimizer == "cvxpy":
        spectrum_fit, success, msg, nfev = solve_bon95_cvxpy(
            A_matrix, b_readings, E, ln_steps,
            b_meas=b_meas, alpha=alpha, solver_backend=solver_backend,
            max_iter=max_iter_qp, tol=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0  # computed after DD

    elif optimizer == "qpsolvers":
        spectrum_fit, success, msg, nfev = solve_bon95_qpsolvers(
            A_matrix, b_readings, E, ln_steps,
            b_meas=b_meas, alpha=alpha, solver_backend=solver_backend,
            max_iter=max_iter_qp, tol=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0

    elif optimizer == "combined":
        spectrum_fit, success, msg, nfev = solve_bon95_combined(
            A_matrix, b_readings, E, ln_steps,
            b_meas=b_meas, alpha=alpha, solver_backend=solver_backend,
            max_iter_qp=max_iter_qp, tol_qp=tol_qp,
        )
        phi_param = spectrum_fit / np.maximum(ln_steps, 1e-30)
        best_chi2 = 0.0

    else:
        raise ValueError(
            f"Unknown optimizer: '{optimizer}'. "
            "Choose from 'grid', 'cvxpy', 'qpsolvers', 'combined'."
        )

    # Ensure non-negative
    phi_param = np.maximum(phi_param, 0.0)
    phi_param = _clean_edge_bins(phi_param)

    # Step 2: Directed-divergence refinement
    phi_refined, n_iter, chi2_final, converged = directed_divergence_iteration(
        A_matrix, b_readings, E, ln_steps, phi_param,
        b_meas=b_meas, max_iter=max_iter, tol_chi2=tol_chi2,
    )

    # Build final spectrum: Phi(E) * ln_steps for the system matrix
    phi_refined = _clean_edge_bins(phi_refined)
    spectrum = phi_refined * ln_steps

    # Compute residual for fit quality check
    computed = A_matrix @ spectrum
    residual_norm = np.linalg.norm(computed - b_readings)
    _check_fit_quality(residual_norm, b_readings, "parametric2")

    if optimizer == "grid":
        message = (
            f"BON95 grid fit (chi2={best_chi2:.4f}) + "
            f"DD iteration ({n_iter} iters, chi2={chi2_final:.4f})"
        )
    else:
        message = (
            f"BON95 {optimizer} fit + "
            f"DD iteration ({n_iter} iters, chi2={chi2_final:.4f})"
        )

    return spectrum, converged, message, nfev


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _check_fit_quality(residual_norm, b_readings, method_name="parametric2"):
    """Emit a warning if the fit residual is large."""
    b_norm = np.linalg.norm(b_readings)
    if b_norm > 0:
        relative_residual = residual_norm / b_norm
        if relative_residual > _RESIDUAL_WARN_THRESHOLD:
            warnings.warn(
                f"{method_name}: large residual "
                f"({residual_norm:.2e} / {b_norm:.2e} = {relative_residual:.1f}x). "
                f"The 4-component BON95 model may not represent this spectrum well.",
                UserWarning,
                stacklevel=3,
            )


def _clean_edge_bins(phi: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Zero out edge bins if anomalously large compared to neighbors.

    The BON95 model divides by E, which can create spikes at the first
    energy bin (tiny E). This function detects and removes such artifacts.

    Parameters
    ----------
    phi : np.ndarray
        Spectrum array.
    factor : float
        Threshold: if edge bin > factor * mean of valid neighbors, zero it.

    Returns
    -------
    np.ndarray
        Cleaned spectrum (new copy).
    """
    phi = np.copy(phi)
    n = len(phi)
    if n < 3:
        return phi

    # Check first bin
    neighbor_mean = np.mean(phi[1:3])
    if neighbor_mean > 0 and phi[0] > factor * neighbor_mean:
        phi[0] = 0.0

    # Check last bin
    neighbor_mean = np.mean(phi[-3:-1])
    if neighbor_mean > 0 and phi[-1] > factor * neighbor_mean:
        phi[-1] = 0.0

    return phi


def _build_measurement_uncertainties(
    b_readings: np.ndarray,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Estimate measurement uncertainties from readings.

    Uses a default relative uncertainty (noise_level) if not provided.
    """
    return np.abs(b_readings) * noise_level + 1e-30


# ------------------------------------------------------------------ #
#  Workflow wrapper
# ------------------------------------------------------------------ #

def unfold_parametric2(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    optimizer: str = "grid",
    b_range: Tuple[float, float, int] = _DEFAULT_B_RANGE,
    Tf_range: Tuple[float, float, int] = _DEFAULT_TF_RANGE,
    c_range: Tuple[float, float, int] = _DEFAULT_C_RANGE,
    alpha: float = 1e-4,
    solver_backend: str = "auto",
    max_iter_qp: int = 50,
    tol_qp: float = 1e-6,
    noise_level: float = 0.05,
    max_iter: int = 200,
    tol_chi2: float = 1.0,
    calculate_errors: bool = False,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using the BON95 parametric method.

    Uses the four-component parameterization from Sannikov BON95:
    thermal (Maxwellian), epithermal (1/E), intermediate, and
    fast (evaporation/cascade) components. After parametric fitting,
    the result is refined by directed-divergence iterations.

    The ``optimizer`` parameter selects the parametric fit backend:

    * ``"grid"``      -- grid search + NLS (default, no extra deps).
    * ``"cvxpy"``     -- SQP via cvxpy.
    * ``"qpsolvers"`` -- SQP via qpsolvers.
    * ``"combined"``  -- grid search + SQP refinement.

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
    optimizer : str
        Parametric fit optimizer (default: "grid").
    b_range : tuple
        Grid range for b: (min, max, n_points). Used by "grid"/"combined".
    Tf_range : tuple
        Grid range for Tf (MeV): (min, max, n_points). Used by "grid"/"combined".
    c_range : tuple
        Grid range for c: (min, max, n_points). Used by "grid"/"combined".
    alpha : float
        Tikhonov regularization for SQP (default: 1e-4).
    solver_backend : str
        QP backend for SQP (default: "auto").
    max_iter_qp : int
        Max SQP iterations (default: 50).
    tol_qp : float
        SQP convergence tolerance (default: 1e-6).
    noise_level : float
        Relative uncertainty for measurements (default: 0.05 = 5%).
    max_iter : int
        Max directed-divergence iterations (default: 200).
    tol_chi2 : float
        Chi-squared convergence threshold (default: 1.0).
    calculate_errors : bool
        Calculate Monte-Carlo errors (default: False).
    n_montecarlo : int
        Number of Monte-Carlo samples (default: 100).
    save_result : bool
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(E_MeV + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if n_energy_bins > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if n_energy_bins > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    ln_steps = log_steps * np.log(10)

    def solve_wrapper(A_mat, b_vec, **kwargs):
        b_meas_local = _build_measurement_uncertainties(b_vec, noise_level)
        x_opt, success, message, nfev = solve_parametric2(
            A_mat, b_vec, E_MeV, ln_steps,
            b_meas=b_meas_local,
            optimizer=optimizer,
            b_range=b_range, Tf_range=Tf_range, c_range=c_range,
            alpha=alpha, solver_backend=solver_backend,
            max_iter_qp=max_iter_qp, tol_qp=tol_qp,
            max_iter=max_iter, tol_chi2=tol_chi2,
        )
        return x_opt, nfev, success

    method_name = "parametric2"
    extra = {
        "optimizer": optimizer,
        "b_range": b_range,
        "Tf_range": Tf_range,
        "c_range": c_range,
        "alpha": alpha,
        "solver_backend": solver_backend,
        "noise_level": noise_level,
        "bon95_Tth": _Tth,
    }

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
