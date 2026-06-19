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
import warnings
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding

__all__ = [
    "solve_parametric",
    "solve_parametric_cvxpy",
    "solve_parametric_qpsolvers",
    "solve_parametric_combined",
    "unfold_parametric",
]

# Fixed constants from the papers / FRUIT code
_T0 = 2.53e-8   # Thermal peak energy (MeV)
_Ed = 7.07e-8   # Epithermal lower boundary parameter (MeV)

# Energy region boundaries (hard-coded per papers)
_THERMAL_MAX = 1e-7    # MeV
_FAST_MIN = 0.1        # MeV

_RESIDUAL_WARN_THRESHOLD = 10.0  # warn when residual norm exceeds this

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Parametric model
# ------------------------------------------------------------------ #

def _thermal(E: np.ndarray) -> np.ndarray:
    """Thermal neutron component: (E/T0^2) * exp(-E/T0)."""
    return (E / (_T0 ** 2)) * np.exp(-E / _T0)


def _epithermal(E: np.ndarray, b: float, beta_prime: float) -> np.ndarray:
    """Epithermal neutron component.

    [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
    """
    return (1.0 - np.exp(-(_Ed > 0) * (E / _Ed) ** 2)) * E ** (b - 1.0) * np.exp(-E / beta_prime)


def _fast(E: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Fast neutron component: E^alpha * exp(-E/beta)."""
    return E ** alpha * np.exp(-E / beta)


def parametric_model(
    E: np.ndarray,
    b: float,
    beta_prime: float,
    alpha: float,
    beta: float,
    P_th: float,
    P_epi: float,
) -> np.ndarray:
    """Combined three-component parametric neutron spectrum.

    Parameters
    ----------
    E : np.ndarray
        Energy grid in MeV.
    b : float
        Epithermal rising-slope exponent.
    beta_prime : float
        Epithermal falling-slope characteristic energy (MeV).
    alpha : float
        Fast-neutron power-law exponent.
    beta : float
        Fast-neutron characteristic energy (MeV).
    P_th : float
        Weight of thermal component.
    P_epi : float
        Weight of epithermal component.

    Returns
    -------
    np.ndarray
        Neutron spectrum (fluence per energy bin).
    """
    E = np.asarray(E, dtype=float)
    P_f = max(0.0, 1.0 - P_th - P_epi)

    thermal = np.zeros_like(E)
    epithermal = np.zeros_like(E)
    fast = np.zeros_like(E)

    m_th = E < _THERMAL_MAX
    m_epi = (E >= _THERMAL_MAX) & (E < _FAST_MIN)
    m_f = E >= _FAST_MIN

    if np.any(m_th):
        thermal[m_th] = _thermal(E[m_th])
    if np.any(m_epi):
        epithermal[m_epi] = _epithermal(E[m_epi], b, beta_prime)
    if np.any(m_f):
        fast[m_f] = _fast(E[m_f], alpha, beta)

    return P_th * thermal + P_epi * epithermal + P_f * fast


# ------------------------------------------------------------------ #
#  Core solver (lmfit)
# ------------------------------------------------------------------ #

def _residuals(params, A_matrix, b_readings, E, log_steps,
               reg_alpha=0.0, initial_param_vec=None):
    """Residual function for lmfit minimization.

    Parameters
    ----------
    reg_alpha : float
        Tikhonov regularization weight.  When > 0, a penalty
        ``sqrt(reg_alpha) * ||p - p0||`` is appended to the residual
        vector, where ``p0`` is the initial parameter guess.
    initial_param_vec : np.ndarray or None
        Reference parameter vector (initial guess) for regularization.
        If None, regularization is applied to raw parameter values.
    """
    b_val = params['b'].value
    bp_val = params['beta_prime'].value
    alpha_val = params['alpha'].value
    beta_val = params['beta'].value
    P_th_val = params['P_th'].value
    P_epi_val = params['P_epi'].value

    spectrum = parametric_model(E, b_val, bp_val, alpha_val, beta_val, P_th_val, P_epi_val)
    spectrum_with_steps = spectrum * log_steps

    computed = A_matrix @ spectrum_with_steps
    residual_data = computed - b_readings

    if reg_alpha > 0:
        param_vec = np.array([b_val, bp_val, alpha_val, beta_val, P_th_val, P_epi_val])
        if initial_param_vec is not None:
            reg_term = np.sqrt(reg_alpha) * (param_vec - initial_param_vec)
        else:
            reg_term = np.sqrt(reg_alpha) * param_vec
        return np.concatenate([residual_data, reg_term])

    return residual_data


def solve_parametric(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray,
    log_steps: np.ndarray,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "leastsq",
    alpha: float = 0.0,
    alpha_auto: bool = False,
    n_restarts: int = 5,
) -> Tuple[np.ndarray, bool, str, int]:
    """Solve unfolding using the FRUIT-based parametric model.

    Uses multi-start optimization: runs lmfit from the top N
    grid-scan starting points and returns the best result.

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
        Initial parameter values.  If None, a grid scan over P_th
        and P_epi is performed automatically.
    method : str, optional
        lmfit solver method (default: "leastsq").
    alpha : float, optional
        Tikhonov regularization weight (default: 0.0).
        When > 0, penalizes deviation from initial guess.
    alpha_auto : bool, optional
        If True, select alpha automatically via GCV (default: False).
    n_restarts : int, optional
        Number of multi-start restarts from top grid-scan points
        (default: 5).

    Returns
    -------
    Tuple[np.ndarray, bool, str, int]
        (spectrum, success, message, nfev)
    """
    try:
        import lmfit
    except ImportError as e:
        raise ImportError(
            "lmfit is required for parametric unfolding. "
            "Install with: pip install lmfit"
        ) from e

    # Grid scan for initial parameters if not provided
    if initial_params is None:
        initial_params = _find_initial_params(A_matrix, b_readings, E, log_steps,
                                               n_grid=7, return_top=n_restarts)

    # GCV-based alpha selection (uses best starting point)
    if alpha_auto:
        best_start = initial_params[0] if isinstance(initial_params, list) else initial_params
        alpha = _gcv_select_alpha(A_matrix, b_readings, E, log_steps, best_start)

    defaults = {
        'b':         (1.0,   0.5,  2.0),
        'beta_prime': (0.01, 1e-4, 1.0),
        'alpha':     (0.5,   0.0,  5.0),
        'beta':      (2.0,   0.1,  20.0),
        'P_th':      (1.0,   0.0,  1.0),
        'P_epi':     (1.0,   0.0,  1.0),
    }

    # Multi-start: collect top starting points from grid scan
    if isinstance(initial_params, list):
        start_points = initial_params[:n_restarts]
    else:
        start_points = [initial_params]

    best_spectrum = None
    best_residual = np.inf
    best_success = False
    best_message = ""
    total_nfev = 0

    for start_params in start_points:
        params = lmfit.Parameters()
        for name, (val, lo, hi) in defaults.items():
            if name in start_params:
                val = start_params[name]
            params.add(name, value=val, min=lo, max=hi)

        initial_param_vec = np.array([
            start_params["b"], start_params["beta_prime"],
            start_params["alpha"], start_params["beta"],
            start_params["P_th"], start_params["P_epi"],
        ])

        result = lmfit.minimize(
            _residuals,
            params,
            args=(A_matrix, b_readings, E, log_steps, alpha, initial_param_vec),
            method=method,
        )

        total_nfev += result.nfev

        fp = result.params
        spectrum = parametric_model(
            E,
            fp['b'].value, fp['beta_prime'].value,
            fp['alpha'].value, fp['beta'].value,
            fp['P_th'].value, fp['P_epi'].value,
        ) * log_steps

        # Evaluate fit quality
        computed = A_matrix @ spectrum
        res = np.linalg.norm(computed - b_readings)

        if res < best_residual:
            best_residual = res
            best_spectrum = spectrum
            best_success = result.success
            best_message = result.message

    return best_spectrum, best_success, best_message, total_nfev


# ------------------------------------------------------------------ #
#  Shared helpers for QP-based parametric solvers
# ------------------------------------------------------------------ #

_PARAM_NAMES = ["b", "beta_prime", "alpha", "beta", "P_th", "P_epi"]

_PARAM_DEFAULTS = {
    "b":         (1.0,   0.5,  2.0),
    "beta_prime": (0.01, 1e-4, 1.0),
    "alpha":     (0.5,   0.0,  5.0),
    "beta":      (2.0,   0.1,  20.0),
    "P_th":      (1.0,   0.0,  1.0),
    "P_epi":     (1.0,   0.0,  1.0),
}


def _get_initial_params(initial_params):
    """Build a flat dict of parameter values from user overrides."""
    params = {}
    for name, (val, _lo, _hi) in _PARAM_DEFAULTS.items():
        if initial_params and name in initial_params:
            val = initial_params[name]
        params[name] = val
    return params


def _get_param_bounds():
    """Return {name: (lo, hi)} bounds."""
    return {name: (lo, hi) for name, (_val, lo, hi) in _PARAM_DEFAULTS.items()}


def _clamp_params(params, bounds):
    """Clamp parameter values to stay within bounds."""
    clamped = dict(params)
    for name, (lo, hi) in bounds.items():
        if lo is not None:
            clamped[name] = max(lo, clamped[name])
        if hi is not None:
            clamped[name] = min(hi, clamped[name])
    return clamped


def _compute_jacobian(E, log_steps, params, delta=1e-8):
    """Numerical Jacobian of (parametric_model * log_steps) w.r.t. params.

    Uses forward finite differences with clamping to keep perturbed
    parameters within bounds.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n_energy, n_params).
    """
    bounds = _get_param_bounds()
    n_params = len(_PARAM_NAMES)
    J = np.zeros((len(E), n_params))

    s0 = parametric_model(
        E, params["b"], params["beta_prime"],
        params["alpha"], params["beta"],
        params["P_th"], params["P_epi"],
    ) * log_steps

    for i, name in enumerate(_PARAM_NAMES):
        lo, hi = bounds[name]
        p_val = params[name]
        d = delta

        # Clamp perturbation to stay within bounds
        if hi is not None and p_val + d > hi:
            d = max(0, hi - p_val) * 0.5
        if lo is not None and p_val + d < lo:
            d = 0.0

        if d < 1e-15:
            # At boundary; use backward difference instead
            d = delta
            if lo is not None and p_val - d >= lo:
                p_pert = p_val - d
                s_pert = parametric_model(
                    E,
                    *(p_pert if n == name else params[n] for n in _PARAM_NAMES),
                ) * log_steps
                J[:, i] = (s0 - s_pert) / d
            else:
                J[:, i] = 0.0
            continue

        p_plus = dict(params)
        p_plus[name] = p_val + d
        s_plus = parametric_model(
            E, p_plus["b"], p_plus["beta_prime"],
            p_plus["alpha"], p_plus["beta"],
            p_plus["P_th"], p_plus["P_epi"],
        ) * log_steps
        J[:, i] = (s_plus - s0) / d

    return J


def _find_initial_params(A_matrix, b_readings, E, log_steps, n_grid=5,
                          return_top=1):
    """Brute-force scan over a small parameter grid to find best starting point.

    Scans P_th and P_epi on a coarse grid (the two parameters that most
    affect the spectral shape), keeps the best residual, and returns the
    full parameter dict.

    Parameters
    ----------
    return_top : int
        If > 1, return a list of the top N starting points sorted by
        residual (best first).
    """
    candidates = []

    p_th_vals = np.linspace(0.0, 1.0, n_grid)
    p_epi_vals = np.linspace(0.0, 1.0, n_grid)

    for p_th in p_th_vals:
        for p_epi in p_epi_vals:
            if p_th + p_epi > 1.0:
                continue
            params = _get_initial_params(None)
            params["P_th"] = p_th
            params["P_epi"] = p_epi

            spectrum = parametric_model(
                E, params["b"], params["beta_prime"],
                params["alpha"], params["beta"],
                params["P_th"], params["P_epi"],
            ) * log_steps

            residual = A_matrix @ spectrum - b_readings
            res_norm = np.linalg.norm(residual)
            candidates.append((res_norm, dict(params)))

    if not candidates:
        return _get_initial_params(None) if return_top == 1 else [_get_initial_params(None)]

    candidates.sort(key=lambda x: x[0])

    if return_top == 1:
        return candidates[0][1]

    return [c[1] for c in candidates[:return_top]]


def _gcv_select_alpha(A_matrix, b_readings, E, log_steps, initial_params,
                       n_coarse=50, n_refine=20):
    """Select Tikhonov regularization alpha via SVD-based GCV with refine.

    Stage 1: coarse search on logspace [1e-8, 1e2].
    Stage 2: refine on linspace [alpha_best/10, alpha_best*10].

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (m x n).
    b_readings : np.ndarray
        Measurement vector (m,).
    E, log_steps : np.ndarray
        Energy grid and log steps.
    initial_params : dict
        Starting parameters for the parametric model.
    n_coarse : int
        Number of coarse alpha candidates.
    n_refine : int
        Number of refine alpha candidates.

    Returns
    -------
    float
        Optimal alpha.
    """
    try:
        from .regularization import compute_svd_components
    except ImportError:
        from bssunfold.core._matrix_utils import compute_svd_components

    # Build effective A: the parametric model is nonlinear, so we
    # linearize around initial_params to get a Jacobian J, then use
    # A_eff = A_matrix @ J as the effective forward operator.
    J = _compute_jacobian(E, log_steps, initial_params)
    A_eff = A_matrix @ J  # (m, n_params)

    m, n = A_eff.shape
    if m < 2 or n < 2:
        return 1e-4

    U, s, Vt, s_sq = compute_svd_components(A_eff)
    UTb = U.T @ b_readings

    def _gcv_value(alpha):
        filt = s_sq / (s_sq + alpha)
        residual_coeff = alpha / (s_sq + alpha)
        residual_sq = np.sum((residual_coeff * UTb) ** 2)
        trace_term = np.sum(filt)
        denom = (m - trace_term) ** 2
        if denom < 1e-30:
            return np.inf
        return residual_sq / denom

    # Stage 1: coarse search
    alphas_coarse = np.logspace(-8, 2, n_coarse)
    gcv_coarse = np.array([_gcv_value(a) for a in alphas_coarse])
    best_idx = int(np.argmin(gcv_coarse))
    alpha_best = alphas_coarse[best_idx]

    # Stage 2: refine around best
    lo = alpha_best / 10.0
    hi = alpha_best * 10.0
    alphas_refine = np.linspace(max(lo, 1e-10), hi, n_refine)
    gcv_refine = np.array([_gcv_value(a) for a in alphas_refine])
    best_idx_r = int(np.argmin(gcv_refine))
    alpha_refined = alphas_refine[best_idx_r]

    return float(alpha_refined)


def _check_fit_quality(residual_norm, b_readings, method_name="parametric"):
    """Emit a warning if the fit residual is large relative to readings."""
    b_norm = np.linalg.norm(b_readings)
    if b_norm > 0:
        relative_residual = residual_norm / b_norm
        if relative_residual > _RESIDUAL_WARN_THRESHOLD:
            warnings.warn(
                f"{method_name}: large residual "
                f"({residual_norm:.2e} / {b_norm:.2e} = {relative_residual:.1f}x). "
                f"The 3-component parametric model may not represent this spectrum well.",
                UserWarning,
                stacklevel=3,
            )


def _parse_solver_backend(solver_backend):
    """Parse a solver_backend string into (library, backend).

    Examples:
        "auto"            -> ("auto", "default")
        "cvxpy"           -> ("cvxpy", "default")
        "cvxpy:ECOS"      -> ("cvxpy", "ECOS")
        "qpsolvers"       -> ("qpsolvers", "auto")
        "qpsolvers:osqp"  -> ("qpsolvers", "osqp")
    """
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
        candidates = [s for s in ["ECOS", "SCS", "CLARABEL"] if s in installed]
        return candidates or ["ECOS"]
    else:
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
#  cvxpy-based parametric solver (SQP)
# ------------------------------------------------------------------ #

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
        spectrum_k = parametric_model(
            E, params["b"], params["beta_prime"],
            params["alpha"], params["beta"],
            params["P_th"], params["P_epi"],
        ) * log_steps

        residual = A_matrix @ spectrum_k - b_readings
        nfev += 1

        if np.linalg.norm(residual) < tol:
            _check_fit_quality(np.linalg.norm(residual), b_readings, "parametric_cvxpy")
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
                    E, params["b"], params["beta_prime"],
                    params["alpha"], params["beta"],
                    params["P_th"], params["P_epi"],
                ) * log_steps,
                True, message, nfev,
            )

    spectrum = parametric_model(
        E, params["b"], params["beta_prime"],
        params["alpha"], params["beta"],
        params["P_th"], params["P_epi"],
    ) * log_steps

    _check_fit_quality(np.linalg.norm(A_matrix @ spectrum - b_readings), b_readings, "parametric_cvxpy")

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
        spectrum_k = parametric_model(
            E, params["b"], params["beta_prime"],
            params["alpha"], params["beta"],
            params["P_th"], params["P_epi"],
        ) * log_steps

        residual = A_matrix @ spectrum_k - b_readings
        nfev += 1

        if np.linalg.norm(residual) < tol:
            _check_fit_quality(np.linalg.norm(residual), b_readings, "parametric_qpsolvers")
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
                P=P, q=q, G=G, h=h,
                solver=solver_name, verbose=False,
            )
        except Exception as exc:
            logger.debug("QP solver %s failed at iteration %d: %s", solver_name, k, exc)
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
                    E, params["b"], params["beta_prime"],
                    params["alpha"], params["beta"],
                    params["P_th"], params["P_epi"],
                ) * log_steps,
                True, message, nfev,
            )

    spectrum = parametric_model(
        E, params["b"], params["beta_prime"],
        params["alpha"], params["beta"],
        params["P_th"], params["P_epi"],
    ) * log_steps

    _check_fit_quality(np.linalg.norm(A_matrix @ spectrum - b_readings), b_readings, "parametric_qpsolvers")

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
        A_matrix, b_readings, E, log_steps, initial_params, method,
    )

    _check_fit_quality(
        np.linalg.norm(A_matrix @ spectrum_lmfit - b_readings),
        b_readings, "parametric_combined(lmfit)",
    )

    # Step 2: QP refinement on the spectrum
    spectrum_init = spectrum_lmfit.copy()
    n = A_matrix.shape[1]

    library, backend = _parse_solver_backend(solver_backend)

    # Auto-detect: try cvxpy first, then qpsolvers
    if library == "auto":
        try:
            import cvxpy  # noqa: F401
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
            return spectrum_lmfit, lmfit_success, "QP refinement failed", lmfit_nfev

        success = lmfit_success
        message = f"lmfit ({lmfit_msg}) + QP refinement OK"
        return refined * log_steps, success, message, lmfit_nfev

    elif library == "qpsolvers":
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
                return spectrum_lmfit, lmfit_success, "No QP solver available", lmfit_nfev

        P = csc_matrix(A_matrix.T @ A_matrix + alpha * np.eye(n))
        q = -(A_matrix.T @ b_readings + alpha * spectrum_init)

        G = csc_matrix(-np.eye(n))
        h = np.zeros(n)

        x_opt = solve_qp(
            P=P, q=q, G=G, h=h,
            solver=qpsolver_name, verbose=False,
        )

        if x_opt is None:
            return spectrum_lmfit, lmfit_success, "QP refinement failed", lmfit_nfev

        refined = np.asarray(x_opt)
        success = lmfit_success
        message = f"lmfit ({lmfit_msg}) + QP refinement OK"
        return refined * log_steps, success, message, lmfit_nfev

    else:
        raise ValueError(f"Unknown solver library: '{library}'. Use 'cvxpy' or 'qpsolvers'.")


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
    selected = [name for name in detector_names if name in readings]
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(E_MeV + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if n_energy_bins > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if n_energy_bins > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    ln_steps = log_steps * np.log(10)

    if optimizer == "lmfit":
        # lmfit uses grid scan initialization; small Tikhonov
        # regularization (deviation from initial guess) provides
        # numerical stability for this ill-conditioned problem.
        lmfit_alpha = alpha if alpha_auto else 1e-8
        def solve_wrapper(A_mat, b_vec, **kwargs):
            x_opt, success, message, nfev = solve_parametric(
                A_mat, b_vec, E_MeV, ln_steps, initial_params, method,
                alpha=lmfit_alpha, alpha_auto=alpha_auto,
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
            x_opt, success, message, nfev = solve_parametric_cvxpy(
                A_mat, b_vec, E_MeV, ln_steps,
                initial_params=initial_params, alpha=alpha,
                solver_backend=solver_backend, max_iter=max_iter, tol=tol,
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
            x_opt, success, message, nfev = solve_parametric_qpsolvers(
                A_mat, b_vec, E_MeV, ln_steps,
                initial_params=initial_params, alpha=alpha,
                solver_backend=solver_backend, max_iter=max_iter, tol=tol,
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
            x_opt, success, message, nfev = solve_parametric_combined(
                A_mat, b_vec, E_MeV, ln_steps,
                initial_params=initial_params, method=method,
                alpha=alpha, solver_backend=solver_backend,
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
