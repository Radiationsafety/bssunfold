"""FRUIT parametric model and nonlinear least-squares fit.

Implements the Bedogni FRUIT three-component neutron spectrum model and
the nonlinear least-squares (NLS) optimizer that fits its parameters to
the measured readings. Split out of ``unfold_parametric.py`` so the
cvxpy / qpsolvers / combined solver backends depend on a single,
well-defined model layer.

Public surface: :func:`parametric_model`, :func:`solve_parametric`.
"""

import logging
import warnings

import numpy as np
from typing import Dict, Optional, Tuple


__all__ = ["parametric_model", "solve_parametric"]
# Fixed constants from the papers / FRUIT code
_T0 = 2.53e-8  # Thermal peak energy (MeV)
_Ed = 7.07e-8  # Epithermal lower boundary parameter (MeV)

# Energy region boundaries (hard-coded per papers)
_THERMAL_MAX = 1e-7  # MeV
_FAST_MIN = 0.1  # MeV

_RESIDUAL_WARN_THRESHOLD = 10.0  # warn when residual norm exceeds this

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Parametric model
# ------------------------------------------------------------------ #


def _thermal(E: np.ndarray) -> np.ndarray:
    """Thermal neutron component: (E/T0^2) * exp(-E/T0)."""
    return (E / (_T0**2)) * np.exp(-E / _T0)


def _epithermal(E: np.ndarray, b: float, beta_prime: float) -> np.ndarray:
    """Epithermal neutron component.

    [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
    """
    return (
        (1.0 - np.exp(-(_Ed > 0) * (E / _Ed) ** 2))
        * E ** (b - 1.0)
        * np.exp(-E / beta_prime)
    )


def _fast(E: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Fast neutron component: E^alpha * exp(-E/beta)."""
    return E**alpha * np.exp(-E / beta)


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


def _residuals(
    params,
    A_matrix,
    b_readings,
    E,
    log_steps,
    reg_alpha=0.0,
    initial_param_vec=None,
):
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
    b_val = params["b"].value
    bp_val = params["beta_prime"].value
    alpha_val = params["alpha"].value
    beta_val = params["beta"].value
    P_th_val = params["P_th"].value
    P_epi_val = params["P_epi"].value

    spectrum = parametric_model(
        E, b_val, bp_val, alpha_val, beta_val, P_th_val, P_epi_val
    )
    spectrum_with_steps = spectrum * log_steps

    computed = A_matrix @ spectrum_with_steps
    residual_data = computed - b_readings

    if reg_alpha > 0:
        param_vec = np.array(
            [b_val, bp_val, alpha_val, beta_val, P_th_val, P_epi_val]
        )
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
        initial_params = _find_initial_params(
            A_matrix, b_readings, E, log_steps, n_grid=7, return_top=n_restarts
        )

    # GCV-based alpha selection (uses best starting point)
    if alpha_auto:
        best_start = (
            initial_params[0]
            if isinstance(initial_params, list)
            else initial_params
        )
        alpha = _gcv_select_alpha(
            A_matrix, b_readings, E, log_steps, best_start
        )

    defaults = {
        "b": (1.0, 0.5, 2.0),
        "beta_prime": (0.01, 1e-4, 1.0),
        "alpha": (0.5, 0.0, 5.0),
        "beta": (2.0, 0.1, 20.0),
        "P_th": (1.0, 0.0, 1.0),
        "P_epi": (1.0, 0.0, 1.0),
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

        initial_param_vec = np.array(
            [
                start_params["b"],
                start_params["beta_prime"],
                start_params["alpha"],
                start_params["beta"],
                start_params["P_th"],
                start_params["P_epi"],
            ]
        )

        result = lmfit.minimize(
            _residuals,
            params,
            args=(A_matrix, b_readings, E, log_steps, alpha, initial_param_vec),
            method=method,
        )

        total_nfev += result.nfev

        fp = result.params
        spectrum = (
            parametric_model(
                E,
                fp["b"].value,
                fp["beta_prime"].value,
                fp["alpha"].value,
                fp["beta"].value,
                fp["P_th"].value,
                fp["P_epi"].value,
            )
            * log_steps
        )

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
    "b": (1.0, 0.5, 2.0),
    "beta_prime": (0.01, 1e-4, 1.0),
    "alpha": (0.5, 0.0, 5.0),
    "beta": (2.0, 0.1, 20.0),
    "P_th": (1.0, 0.0, 1.0),
    "P_epi": (1.0, 0.0, 1.0),
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

    s0 = (
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
                s_pert = (
                    parametric_model(
                        E,
                        *(
                            p_pert if n == name else params[n]
                            for n in _PARAM_NAMES
                        ),
                    )
                    * log_steps
                )
                J[:, i] = (s0 - s_pert) / d
            else:
                J[:, i] = 0.0
            continue

        p_plus = dict(params)
        p_plus[name] = p_val + d
        s_plus = (
            parametric_model(
                E,
                p_plus["b"],
                p_plus["beta_prime"],
                p_plus["alpha"],
                p_plus["beta"],
                p_plus["P_th"],
                p_plus["P_epi"],
            )
            * log_steps
        )
        J[:, i] = (s_plus - s0) / d

    return J


def _find_initial_params(
    A_matrix, b_readings, E, log_steps, n_grid=5, return_top=1
):
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

            residual = A_matrix @ spectrum - b_readings
            res_norm = np.linalg.norm(residual)
            candidates.append((res_norm, dict(params)))

    if not candidates:
        return (
            _get_initial_params(None)
            if return_top == 1
            else [_get_initial_params(None)]
        )

    candidates.sort(key=lambda x: x[0])

    if return_top == 1:
        return candidates[0][1]

    return [c[1] for c in candidates[:return_top]]


def _gcv_select_alpha(
    A_matrix, b_readings, E, log_steps, initial_params, n_coarse=50, n_refine=20
):
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

    U, _, _, s_sq = compute_svd_components(A_eff)
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


# ------------------------------------------------------------------ #
#  cvxpy-based parametric solver (SQP)
# ------------------------------------------------------------------ #


