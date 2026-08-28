"""SCIP-based unfolding method.

This module provides a core ``solve_scip`` solver and the ``unfold_scip``
wrapper that solve the unfolding problem with the SCIP Optimization Suite
(https://www.scipopt.org/) through the ``pyscipopt`` package. SCIP is a
general-purpose optimization solver (MIP/NLP/QP); here it minimizes the
Tikhonov-regularized least-squares objective under non-negativity
constraints.

The ``pyscipopt`` package is an optional dependency imported lazily inside
the function bodies.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from ._matrix_utils import create_derivative_matrix
from ._max_energy import upper_bounds
from .regularization import resolve_regularization_parameter

__all__ = ["solve_scip", "unfold_scip"]


def _import_pyscipopt():
    """Import and return the pyscipopt module, raising a helpful ImportError."""
    try:
        import pyscipopt
    except ImportError as e:
        raise ImportError(
            "pyscipopt is required for unfold_scip. "
            "Install with: pip install pyscipopt"
        ) from e
    return pyscipopt


def solve_scip(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    alpha: float = 1e-4,
    norm: int = 2,
    timeout: float = 10.0,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    nonneg: bool = True,
    random_state: Optional[int] = None,
    ub: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Solve the unfolding problem with the SCIP optimizer.

    Minimizes ``0.5 * ||A x - b||^2 + penalty(x)`` with ``penalty`` given by
    ``alpha * ||x||^2`` (L2), ``alpha * sum(x)`` (L1, exact under ``x >= 0``)
    or a derivative smoothness term, subject to ``x >= 0`` when ``nonneg``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix of size (m, n).
    b : np.ndarray
        Measurement vector of size (m,).
    x0 : np.ndarray, optional
        Initial values, used as a warm start for the solver.
    alpha : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    timeout : float, optional
        Time limit in seconds, default: 10.0.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2), default: 0.
    smoothness_weight : float, optional
        Weight for the smoothness term, default: 1.0.
    nonneg : bool, optional
        Constrain the solution to ``x >= 0``, default: True.
    random_state : int, optional
        Random seed for the solver, for reproducibility.

    Returns
    -------
    Optional[np.ndarray]
        Unfolded spectrum (n,), or None if solving failed.
    """
    pyscipopt = _import_pyscipopt()
    from pyscipopt.recipes.nonlinear import set_nonlinear_objective

    Model = pyscipopt.Model
    quicksum = pyscipopt.quicksum

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("SCIP solver: received ill-formed input.")
    n = A.shape[1]
    m = A.shape[0]

    model = Model("bssunfold_scip")
    model.setParam("limits/time", max(float(timeout), 1e-3))
    model.setParam("display/verblevel", 0)
    if random_state is not None:
        model.setParam("randomization/randomseedshift", int(random_state))

    lb = 0.0 if nonneg else None
    ub_list = None
    if ub is not None:
        ub_list = [None if not np.isfinite(u) else float(u) for u in ub]
    x = [
        model.addVar(
            lb=lb,
            ub=None if ub_list is None else ub_list[i],
            name=f"x{i}",
        )
        for i in range(n)
    ]

    residual = quicksum(
        (b[i] - quicksum(A[i, j] * x[j] for j in range(n))) ** 2
        for i in range(m)
    )
    penalty = _build_penalty(
        x, A, alpha, norm, smoothness_order, smoothness_weight
    )

    set_nonlinear_objective(model, 0.5 * residual + penalty, sense="minimize")

    if x0 is not None:
        x0a = np.asarray(x0, dtype=float)
        if x0a.ndim == 1 and len(x0a) == n:
            sol = model.createSol()
            for j in range(n):
                val = float(x0a[j])
                if nonneg:
                    val = max(val, 0.0)
                model.setSolVal(sol, x[j], val)
            model.addSol(sol)

    model.optimize()
    status = model.getStatus()
    best = model.getBestSol() if model.getNSols() > 0 else None
    if best is None:
        warnings.warn(f"SCIP solver did not find a solution (status={status}).")
        return None
    result = np.array([model.getSolVal(best, xj) for xj in x])
    if ub is not None:
        result[ub == 0.0] = 0.0
    return result


def _build_penalty(
    x,
    A: np.ndarray,
    alpha: float,
    norm: int,
    smoothness_order: int,
    smoothness_weight: float,
):
    """Build the SCIP regularization expression for the objective."""
    from pyscipopt import quicksum

    n = A.shape[1]

    if norm == 2:
        if smoothness_order in (1, 2):
            L = create_derivative_matrix(n, smoothness_order).toarray()
            smooth = (
                alpha
                * smoothness_weight
                * quicksum(
                    (quicksum(L[k, j] * x[j] for j in range(n))) ** 2
                    for k in range(L.shape[0])
                )
            )
            return smooth
        return alpha * quicksum(x[j] ** 2 for j in range(n))

    if norm == 1:
        penalty = alpha * quicksum(x[j] for j in range(n))
        if smoothness_order in (1, 2):
            L = create_derivative_matrix(n, smoothness_order).toarray()
            penalty += (
                alpha
                * smoothness_weight
                * quicksum(
                    (quicksum(L[k, j] * x[j] for j in range(n))) ** 2
                    for k in range(L.shape[0])
                )
            )
        return penalty

    raise ValueError(f"Unsupported norm type: {norm}")


def unfold_scip(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    regularization: float = 1e-4,
    norm: int = 2,
    timeout: float = 10.0,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    nonneg: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    random_state: Optional[int] = None,
    max_neutron_energy: Optional[float] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum using the SCIP optimizer.

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
        Initial spectrum guess.
    regularization : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    timeout : float, optional
        Time limit in seconds, default: 10.0.
    smoothness_order : int, optional
        Smoothness constraint order (0, 1, or 2), default: 0.
    smoothness_weight : float, optional
        Weight for the smoothness term, default: 1.0.
    nonneg : bool, optional
        Constrain the spectrum to be non-negative, default: True.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default: 100.
    save_result : bool, optional
        Save result to history, default: True.
    regularization_method : str, optional
        Method for selecting the regularization parameter
        ('manual', 'cosine', 'lcurve', 'gcv', 'dp').
    noise_var : float, optional
        Noise variance for discrepancy principle ('dp' method).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
    """
    A, b, _ = _build_system(readings, detector_names, sensitivities)

    alpha = resolve_regularization_parameter(
        A,
        b,
        regularization_method,
        regularization,
        n_energy_bins,
        initial_spectrum=initial_spectrum,
        norm=norm,
        noise_var=noise_var,
    )
    selected_lambda = alpha
    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
        x0 = kwargs.pop("x0", None)
        x = solve_scip(
            A,
            b,
            x0=x0,
            alpha=alpha,
            norm=norm,
            timeout=timeout,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            nonneg=nonneg,
            random_state=random_state,
            ub=upper_bounds(E_MeV, max_neutron_energy),
        )
        if x is None:
            x = np.zeros(A.shape[1])
            warnings.warn("Solution not found, returning zero spectrum.")
        return x

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
        method_name="SCIP",
        extra_output={
            "norm": norm,
            "regularization": regularization,
            "regularization_method": regularization_method,
            "selected_regularization": float(selected_lambda),
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "timeout": timeout,
            "nonneg": nonneg,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
