"""docplex (IBM CPLEX) based unfolding method.

This module provides a core ``solve_docplex`` solver and the ``unfold_docplex``
wrapper that solve the unfolding problem with IBM Decision Optimization CPLEX
Modeling for Python (https://pypi.org/project/docplex/). The ``docplex``
package builds the model and the ``cplex`` engine (CPLEX Community Edition)
solves it locally.

Both packages are optional dependencies imported lazily inside the function
bodies.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from ._matrix_utils import create_derivative_matrix
from .regularization import resolve_regularization_parameter

__all__ = ["solve_docplex", "unfold_docplex"]


def _import_docplex():
    """Import and return the docplex model module, raising a helpful error.

    The ``cplex`` engine is required to actually solve the model, so its
    availability is checked here as well.
    """
    try:
        from docplex.mp.model import Model
    except ImportError as e:
        raise ImportError(
            "docplex is required for unfold_docplex. "
            "Install with: pip install docplex cplex"
        ) from e
    try:
        import cplex  # noqa: F401  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError(
            "The CPLEX engine (cplex) is required for unfold_docplex. "
            "Install with: pip install cplex"
        ) from e
    return Model


def solve_docplex(
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
) -> Optional[np.ndarray]:
    """Solve the unfolding problem with CPLEX (docplex).

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
        Initial values (accepted for API compatibility; CPLEX QP has no
        warm start for continuous models).
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
    Model = _import_docplex()

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("docplex solver: received ill-formed input.")
    n = A.shape[1]
    m = A.shape[0]

    mdl = Model(name="bssunfold_docplex")
    mdl.context.solver.log_output = False
    mdl.set_time_limit(max(float(timeout), 1e-3))
    mdl.parameters.threads.set(1)
    if random_state is not None:
        mdl.parameters.randomseed.set(int(random_state))

    lb = 0 if nonneg else None
    x = mdl.continuous_var_list(n, lb=lb, name="x")

    residual = [b[i] - mdl.dot(x, A[i]) for i in range(m)]
    obj = 0.5 * mdl.sum_squares(residual)
    obj += _build_penalty(
        mdl, x, A, alpha, norm, smoothness_order, smoothness_weight
    )
    mdl.minimize(obj)

    sol = mdl.solve()
    if sol is None:
        warnings.warn(
            "CPLEX solver did not find a solution. Returning zero vector."
        )
        return None
    return np.array([sol.get_value(xj) for xj in x])


def _build_penalty(
    mdl, x, A: np.ndarray, alpha, norm, smoothness_order, smoothness_weight
):
    """Build the docplex regularization expression for the objective."""
    n = A.shape[1]

    if norm == 2:
        if smoothness_order in (1, 2):
            L = create_derivative_matrix(n, smoothness_order).toarray()
            return (
                alpha
                * smoothness_weight
                * mdl.sum_squares([mdl.dot(x, L[k]) for k in range(L.shape[0])])
            )
        return alpha * mdl.sum_squares(x)

    if norm == 1:
        penalty = alpha * mdl.sum(x)
        if smoothness_order in (1, 2):
            L = create_derivative_matrix(n, smoothness_order).toarray()
            penalty += (
                alpha
                * smoothness_weight
                * mdl.sum_squares([mdl.dot(x, L[k]) for k in range(L.shape[0])])
            )
        return penalty

    raise ValueError(f"Unsupported norm type: {norm}")


def unfold_docplex(
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
) -> Dict[str, Any]:
    """Unfold a neutron spectrum using CPLEX (docplex).

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
        # x0 is forwarded for API consistency; solve_docplex documents that
        # CPLEX has no warm start for continuous QP models, so it is unused.
        x0 = kwargs.pop("x0", None)
        x = solve_docplex(
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
        method_name="docplex",
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
