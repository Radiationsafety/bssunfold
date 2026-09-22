"""Commercial-license QP solver unfolding methods (license required).

This module exposes five parallel unfolding methods backed by the world's
main commercial optimization engines, driven through their cvxpy
interfaces: ``solve_gurobi``/``unfold_gurobi``, ``solve_mosek``/
``unfold_mosek``, ``solve_cplex``/``unfold_cplex``, ``solve_copt``/
``unfold_copt`` and ``solve_xpress``/``unfold_xpress``.

.. note:: **License required** — every solver here is proprietary:

   =========  ==================  ==========================================
   Solver     Package             License
   =========  ==================  ==========================================
   Gurobi     ``gurobipy``        Commercial / free academic / size-limited
                                  trial (https://www.gurobi.com)
   MOSEK      ``mosek``           Commercial / free full license for students
                                  and academics (https://www.mosek.com)
   CPLEX      ``cplex``           IBM Academic Program / Community Edition /
                                  commercial (https://www.ibm.com/products/
                                  ibm-cplex-studio)
   COPT       ``coptpy``         Commercial / free academic (Gurobi-like)
                                  (https://github.com/COPT-Publication)
   XPRESS     ``xpress``          Commercial / free academic (FICO, formerly
                                  Dash Optimisation) (https://www.fico.com/
                                  products/fico-xpress-optimization)
   =========  ==================  ==========================================

   No license is bundled with bssunfold; the engine is only used if its
   Python package is importable and a valid license is configured on the
   machine.  cvxpy ships the interfaces itself, so only the engine package
   needs installing, e.g. ``pip install bssunfold[gurobi]``.

All five methods solve the same Tikhonov-regularized non-negative least
squares problem as :mod:`unfold_docplex` / :mod:`unfold_qpsolvers`
(``0.5 * ||A x - b||^2 + alpha * ||x||^2`` or the L1 equivalent, optional
first/second-derivative smoothness penalty) via the shared cvxpy QP
backend in :mod:`bssunfold.core._commercial_qp`.  Unlike
:func:`unfold_cvxpy`, there is **no fallback** to open-source solvers: if
the requested engine or its license is missing, the solve warns and the
method returns a zero spectrum.

Existing related method: IBM CPLEX can also be used through the modeling
layer docplex via :func:`unfold_docplex` (that path does not require the
cvxpy CPLEX interface).
"""

import warnings
from functools import partial
from typing import Any

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from ._commercial_qp import (
    COMMERCIAL_SOLVER_ALIASES,
    commercial_solver_info,
    is_commercial_solver_available,
    solve_commercial_qp,
)
from ._max_energy import upper_bounds
from .regularization import resolve_regularization_parameter

__all__ = [
    "COMMERCIAL_SOLVER_ALIASES",
    "commercial_solver_info",
    "is_commercial_solver_available",
    "solve_commercial",
    "solve_copt",
    "solve_cplex",
    "solve_gurobi",
    "solve_mosek",
    "solve_xpress",
    "unfold_commercial",
    "unfold_copt",
    "unfold_cplex",
    "unfold_gurobi",
    "unfold_mosek",
    "unfold_xpress",
]

_COMMERCIAL_EXTRA_PARAMS = (
    "alpha",
    "norm",
    "timeout",
    "smoothness_order",
    "smoothness_weight",
    "nonneg",
    "random_state",
    "ub",
)


def solve_commercial(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    alpha: float = 1e-4,
    norm: int = 2,
    solver: str = "gurobi",
    timeout: float = 10.0,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    nonneg: bool = True,
    random_state: int | None = None,
    ub: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve the unfolding problem with a commercial (license-required) solver.

    General entry point behind :func:`solve_gurobi` and friends: the QP
    ``min 0.5*||A x - b||^2 + penalty(x)`` is handed to the cvxpy interface
    of the requested engine, exclusively (no open-source fallback).

    Parameters
    ----------
    A : np.ndarray
        Response matrix of size (m, n).
    b : np.ndarray
        Measurement vector of size (m,).
    x0 : np.ndarray, optional
        Initial values, forwarded as a warm start when supported.
    alpha : float, optional
        Regularization parameter, default: 1e-4.
    norm : int, optional
        Norm type (1 for L1, 2 for L2), default: 2.
    solver : str, optional
        Commercial solver alias: 'gurobi', 'mosek', 'cplex', 'copt', 'xpress'
        (all license required), default: 'gurobi'.
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
    ub : np.ndarray, optional
        Per-bin upper bounds; non-finite entries mean "unbounded".

    Returns
    -------
    Optional[np.ndarray]
        Unfolded spectrum (n,), or None if the solver/license is unavailable
        or the solve failed.
    """
    return solve_commercial_qp(
        A,
        b,
        alias=solver,
        x0=x0,
        alpha=alpha,
        norm=norm,
        timeout=timeout,
        smoothness_order=smoothness_order,
        smoothness_weight=smoothness_weight,
        nonneg=nonneg,
        random_state=random_state,
        ub=ub,
    )


def unfold_commercial(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
    solver: str = "gurobi",
    initial_spectrum: np.ndarray | None = None,
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
    noise_var: float | None = None,
    random_state: int | None = None,
    max_neutron_energy: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum with a commercial (license-required) solver.

    General entry point behind :func:`unfold_gurobi` and friends.  See the
    module docstring for the license terms of each engine.

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
    solver : str, optional
        Commercial solver alias ('gurobi', 'mosek', 'cplex', 'copt',
        'xpress'; all license required), default: 'gurobi'.
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
        Save result to history, default: False.
    regularization_method : str, optional
        Method for selecting the regularization parameter
        ('manual', 'cosine', 'lcurve', 'gcv', 'dp').
    noise_var : float, optional
        Noise variance for discrepancy principle ('dp' method).
    random_state : int, optional
        Random seed for reproducibility.
    max_neutron_energy : float, optional
        Upper bound on neutron energy for the spectrum support.

    Returns
    -------
    Dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.  The
        metadata carries ``license_required: True`` and the resolved
        ``cvxpy_solver``/``pip_package`` of the engine that was asked for.
    """
    info = commercial_solver_info(solver)  # validates alias
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
    x0_default = np.zeros(n_energy_bins)

    def solve_wrapper(A, b, **kwargs):
        x0 = kwargs.pop("x0", None)
        x = solve_commercial_qp(
            A,
            b,
            alias=solver,
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
        ln_steps=ln_steps,
        reading_uncertainties=reading_uncertainties,
        reading_covariance=reading_covariance,
        noise_model=noise_model,
        measurement_time=measurement_time,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name=solver,
        extra_output={
            "norm": norm,
            "solver": solver,
            "cvxpy_solver": info["cvxpy_solver"],
            "pip_package": info["pip_package"],
            "license_required": True,
            "regularization": regularization,
            "regularization_method": regularization_method,
            "selected_regularization": float(alpha),
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


def _make_solve(alias: str):
    """Build the public ``solve_<alias>`` entry point for one backend."""
    info = commercial_solver_info(alias)
    doc = f"""Solve the unfolding problem with {alias.upper()} (license required).

    Thin pre-bound wrapper over :func:`solve_commercial`; see there for the
    full parameter documentation.  ``{alias}`` is a proprietary commercial
    solver: the ``{info["pip_package"]}`` package with a valid license must
    be installed (``pip install bssunfold[{alias}]``).  No open-source
    fallback is attempted.
    """
    func = partial(solve_commercial, solver=alias)
    func.__name__ = f"solve_{alias}"
    func.__qualname__ = f"solve_{alias}"
    func.__doc__ = doc
    return func


def _make_unfold(alias: str):
    """Build the public ``unfold_<alias>`` entry point for one backend."""
    info = commercial_solver_info(alias)
    doc = f"""Unfold a neutron spectrum using {alias.upper()} (**license required**).

    Pre-bound wrapper over :func:`unfold_commercial` for the
    ``{alias}`` engine: the proprietary ``{info["pip_package"]}`` package
    with a valid license must be installed (``pip install
    bssunfold[{alias}]``).  Solves the same regularized non-negative
    least-squares problem as the open-source methods, but exclusively with
    the licensed QP engine — if the solver or license is missing the method
    warns and returns a zero spectrum (no fallback).

    See :func:`unfold_commercial` for the full parameter documentation.
    """
    func = partial(unfold_commercial, solver=alias)
    func.__name__ = f"unfold_{alias}"
    func.__qualname__ = f"unfold_{alias}"
    func.__doc__ = doc
    return func


solve_gurobi = _make_solve("gurobi")
solve_mosek = _make_solve("mosek")
solve_cplex = _make_solve("cplex")
solve_copt = _make_solve("copt")
solve_xpress = _make_solve("xpress")

unfold_gurobi = _make_unfold("gurobi")
unfold_mosek = _make_unfold("mosek")
unfold_cplex = _make_unfold("cplex")
unfold_copt = _make_unfold("copt")
unfold_xpress = _make_unfold("xpress")
