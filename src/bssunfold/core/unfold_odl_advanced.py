"""Advanced unfolding methods with PDHG and Douglas-Rachford splitting.

This module provides advanced regularization methods using the Primal-Dual
Hybrid Gradient (Chambolle-Pock) and Douglas-Rachford splitting algorithms.
These methods support Total Variation (TV) regularization which better preserves
sharp spectral features compared to standard Tikhonov regularization.

The algorithms follow the Operator Discretization Library (ODL) formulation but
are implemented in pure NumPy for cross-version stability (ODL 1.0's own PDHG /
Douglas-Rachford solvers break on translated data terms). No external solver
 packages are required.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import run_unfolding


from ..utils.validators import validate_system

def _forward_diff_matrix(n: int) -> np.ndarray:
    """1-D forward-difference operator D (shape (n - 1, n))."""
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


def _tv_prox(f: np.ndarray, lam: float, n_iter: int = 200) -> np.ndarray:
    """Proximal operator of lam * ||D x||_1 (1-D total-variation denoising).

    Solves  argmin_x 0.5 ||x - f||^2 + lam ||D x||_1  via Chambolle's dual
    gradient-ascent algorithm.  D is the 1-D forward-difference operator.
    """
    f = np.asarray(f, dtype=float)
    n = f.size
    if n <= 1 or lam <= 0:
        return f.copy()
    Df = np.diff(f)
    # D D^T (tridiagonal) for the forward-difference operator.
    L = 4.0  # largest eigenvalue of D D^T for this 1-D operator
    rho = 1.0 / (L * lam**2)
    p = np.zeros(n - 1)
    for _ in range(n_iter):
        grad = np.zeros(n)
        grad[1:] = p
        grad[:-1] -= p
        p = p + rho * (lam * Df - lam**2 * (grad[1:] - grad[:-1]))
        p = np.clip(p, -1.0, 1.0)
    grad = np.zeros(n)
    grad[1:] = p
    grad[:-1] -= p
    return f - lam * grad


def solve_odl_pdhg(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Primal-Dual Hybrid Gradient (PDHG).

    A NumPy implementation of the Chambolle-Pock primal-dual algorithm
    (the PDHG scheme documented in ODL) for the problem

        min_x  0.5 ||A x - b||^2 + tv_weight ||D x||_1   (+ non-negativity),

    where D is the 1-D forward-difference operator (Total Variation).
    No external ODL solver is required, which keeps the behaviour stable
    across ODL versions (ODL 1.0's own PDHG solver is known to break on
    translated data terms).

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None, uniform spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tau : float, optional
        Primal step size. If None, computed automatically.
    sigma : float, optional
        Dual step size. If None, computed automatically.
    use_tv : bool, optional
        If True, use Total Variation regularization (default: True).
    tv_weight : float, optional
        Weight for TV regularization term (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity constraint (default: True).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple
        (spectrum, iterations, converged)
    """
    A, b, x0 = validate_system(A, b, x0=x0)
    A, b, x0 = validate_system(A, b, x0=x0)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n = A.shape

    if x0 is None:
        x0 = np.ones(n) * 0.5
    x = np.asarray(x0, dtype=float).copy()

    # Build the augmented forward operator  K = [A; w D]  so that the objective
    #     0.5 ||A x - b||^2 + tv_weight ||D x||_1
    # becomes  0.5 ||K x - d||^2   with  d = [b; 0].
    # The primal-dual algorithm then needs only the closed-form dual update of
    # the quadratic data term.
    if use_tv:
        D = _forward_diff_matrix(n)
        K = np.vstack([A, tv_weight * D])
        d = np.concatenate([b, np.zeros(n - 1)])
    else:
        K = A
        d = b

    eig_max = float(np.linalg.eigvalsh(K.T @ K).max())
    op_norm = float(np.sqrt(max(eig_max, 1e-12)))
    if tau is None:
        tau = 0.99 / op_norm
    if sigma is None:
        sigma = 0.99 / op_norm

    b_norm = max(float(np.linalg.norm(b)), 1e-300)
    res_before = float(np.linalg.norm(A @ x - b)) / b_norm

    y = np.zeros(K.shape[0])
    x_bar = x.copy()
    for _ in range(max_iterations):
        z = y + sigma * (K @ x_bar)
        y = (z - sigma * d) / (1.0 + sigma)
        x_new = x - tau * (K.T @ y)
        if nonnegativity:
            x_new = np.maximum(x_new, 0.0)
        x_bar = x_new + (x_new - x)
        x = x_new

    x_opt = np.maximum(x, 0.0) if nonnegativity else x
    finite = bool(np.all(np.isfinite(x_opt)))
    res_after = float(np.linalg.norm(A @ x_opt - b)) / b_norm
    converged = finite and res_after <= res_before * (1.0 + tolerance * 100)

    return x_opt, max_iterations, converged


def solve_odl_douglas_rachford(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Douglas-Rachford splitting.

    Douglas-Rachford is an operator splitting method that can handle
    the sum of two convex functionals. It's particularly effective for
    problems with non-smooth regularizers.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None, uniform spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    use_tv : bool, optional
        If True, use Total Variation regularization (default: True).
    tv_weight : float, optional
        Weight for TV regularization term (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity constraint (default: True).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple
        (spectrum, iterations, converged)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n = A.shape

    if x0 is None:
        x0 = np.ones(n) * 0.5
    x = np.asarray(x0, dtype=float).copy()

    # Douglas-Rachford splitting for  min_x psi1(x) + psi2(x)  with
    #     psi1(x) = 0.5 ||A x - b||^2      (prox = precomputed linear solve)
    #     psi2(x) = tv_weight ||D x||_1    (prox = 1-D TV denoising)
    # (A NumPy implementation; ODL 1.0's own solver is not used for
    #  cross-version stability.)
    gamma = 1.0
    M = np.eye(n) + gamma * (A.T @ A)
    M_fact = np.linalg.cholesky(M)

    def prox_psi1(v):
        rhs = v + gamma * (A.T @ b)
        return np.linalg.solve(M_fact.T, np.linalg.solve(M_fact, rhs))

    def prox_psi2(v):
        if use_tv:
            return _tv_prox(v, gamma * tv_weight)
        return v

    b_norm = max(float(np.linalg.norm(b)), 1e-300)
    res_before = float(np.linalg.norm(A @ x - b)) / b_norm

    y = x.copy()
    z = x.copy()
    for _ in range(max_iterations):
        u = prox_psi1(2.0 * z - y)
        y = prox_psi2(u)
        z = z + y - u

    x_opt = np.maximum(y, 0.0) if nonnegativity else y
    finite = bool(np.all(np.isfinite(x_opt)))
    res_after = float(np.linalg.norm(A @ x_opt - b)) / b_norm
    converged = finite and res_after <= res_before * (1.0 + tolerance * 100)

    return x_opt, max_iterations, converged


def unfold_odl_pdhg(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using ODL PDHG with TV regularization.

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
        Initial spectrum approximation.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tau : float, optional
        Primal step size.
    sigma : float, optional
        Dual step size.
    use_tv : bool, optional
        Use Total Variation regularization (default: True).
    tv_weight : float, optional
        TV regularization weight (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity (default: True).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for error calculation (default: 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default: 100).
    save_result : bool, optional
        Save to history (default: False).
    random_state : int, optional
        Random seed.

    Returns
    -------
    Dict
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins) * 0.5

    def solve_wrapper(A, b, **kwargs):
        x_init = kwargs.pop('x0', initial_spectrum)
        return solve_odl_pdhg(
            A, b, x0=x_init,
            max_iterations=max_iterations,
            tau=tau, sigma=sigma,
            use_tv=use_tv, tv_weight=tv_weight,
            nonnegativity=nonnegativity,
        )

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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="ODL-PDHG",
        extra_output={
            "max_iterations": max_iterations,
            "use_tv": use_tv,
            "tv_weight": tv_weight,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )


def unfold_odl_douglas_rachford(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using ODL Douglas-Rachford splitting with TV regularization.

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
        Initial spectrum approximation.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    use_tv : bool, optional
        Use Total Variation regularization (default: True).
    tv_weight : float, optional
        TV regularization weight (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity (default: True).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for error calculation (default: 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default: 100).
    save_result : bool, optional
        Save to history (default: False).
    random_state : int, optional
        Random seed.

    Returns
    -------
    Dict
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins) * 0.5

    def solve_wrapper(A, b, **kwargs):
        x_init = kwargs.pop('x0', initial_spectrum)
        return solve_odl_douglas_rachford(
            A, b, x0=x_init,
            max_iterations=max_iterations,
            use_tv=use_tv, tv_weight=tv_weight,
            nonnegativity=nonnegativity,
        )

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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="ODL-DouglasRachford",
        extra_output={
            "max_iterations": max_iterations,
            "use_tv": use_tv,
            "tv_weight": tv_weight,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )


__all__ = [
    "solve_odl_pdhg",
    "solve_odl_douglas_rachford",
    "unfold_odl_pdhg",
    "unfold_odl_douglas_rachford",
]
