"""pyoptexplain-coupled QP build/solve and perturbation analyses for interpretation.

Extracted from ``unfold_interpret.py``. Holds the lazy ``pyoptexplain``
namespace loader, the unfolding-QP construction, the solver, and the
robustness/sensitivity/sweep/relaxation helpers. Pure leaf: it imports
nothing from the other interpret modules, so ``unfold_interpret`` can import
from here without a cycle.
"""

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._matrix_utils import create_derivative_matrix

logger = logging.getLogger(__name__)
class _PyOptExplainNamespace:
    """Lazily import the pyoptexplain pieces needed by this module."""

    def __init__(self) -> None:
        self._loaded: Optional[SimpleNamespace] = None

    def load(self) -> SimpleNamespace:
        if self._loaded is not None:
            return self._loaded
        try:
            import pyoptexplain  # noqa: F401  # pylint: disable=unused-import
        except ImportError as exc:
            raise ImportError(
                "The interpretation module requires pyoptexplain. "
                "Install it with: pip install bssunfold[interpret]"
            ) from exc
        from pyoptexplain import (  # noqa: PLC0415
            Analyzer,
            ChangeRHS,
            QuadraticMatrixProblemHandle,
            ScenarioCase,
            SetVariableBounds,
        )
        from pyoptexplain.analysis.workflows.robustness import (  # noqa: PLC0415
            PerturbationPlan,
        )
        from pyoptexplain.core.solvers import SolveParameters  # noqa: PLC0415
        from pyoptexplain.representations.matrix_scenario import (  # noqa: PLC0415
            QuadraticMatrixScenarioRepresentation,
        )

        self._loaded = SimpleNamespace(
            Analyzer=Analyzer,
            ChangeRHS=ChangeRHS,
            QuadraticMatrixProblemHandle=QuadraticMatrixProblemHandle,
            ScenarioCase=ScenarioCase,
            SetVariableBounds=SetVariableBounds,
            PerturbationPlan=PerturbationPlan,
            SolveParameters=SolveParameters,
            QuadraticMatrixScenarioRepresentation=(
                QuadraticMatrixScenarioRepresentation
            ),
        )
        return self._loaded


_pyopt = _PyOptExplainNamespace()


def _require_pyoptexplain() -> SimpleNamespace:
    return _pyopt.load()


def build_interpretation_qp(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    enforce_norm: bool = False,
    norm_value: float = 1.0,
    lower_bound: float = 0.0,
    variable_names: Optional[Sequence[str]] = None,
    equality_name: str = "norm",
) -> Any:
    """Build the pyoptexplain QP handle for the unfolding problem.

    The quadratic program matches the model solved by ``solve_qpsolvers``
    (objective convention ``0.5 * x'Qx + c'x``):

    - ``norm == 2``:  ``Q = A'A + alpha*sw*L'L`` (``smoothness_order`` 1/2) or
      ``Q = A'A + alpha*I``, ``c = -A'b``.
    - ``norm == 1``:  ``Q = A'A + alpha*sw*L'L``, ``c = -A'b + alpha*1``
      (the L1 penalty is linear and exact under ``x >= 0``).

    The spectrum is constrained to ``x >= lower_bound``. When
    ``enforce_norm=True`` the equality ``sum(x) == norm_value`` is added as a
    named block (default ``"norm"``).

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    alpha : float
        Regularization parameter (>= 0).
    norm : int, optional
        Penalty norm, 1 or 2 (default: 2).
    smoothness_order : int, optional
        Smoothness derivative order, 0, 1 or 2 (default: 0).
    smoothness_weight : float, optional
        Weight of the smoothness term (default: 1.0).
    enforce_norm : bool, optional
        Add ``sum(x) == norm_value`` (default: False).
    norm_value : float, optional
        Target total fluence for the norm equality (default: 1.0).
    lower_bound : float, optional
        Shared lower bound on every energy group (default: 0.0).
    variable_names : sequence of str, optional
        Names of the energy-group variables (default: ``E0..E{n-1}``).
    equality_name : str, optional
        Name of the norm equality block (default: ``"norm"``).

    Returns
    -------
    QuadraticMatrixProblemHandle
        A ``pyoptexplain`` problem handle; call
        ``handle.quadratic_representation()`` for the analysis surface.

    Raises
    ------
    ImportError
        If ``pyoptexplain`` is not installed.
    ValueError
        If the arguments are invalid or ``Q`` is not positive semidefinite.
    """
    _require_pyoptexplain()

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError(
            "A must be a 2D (m x n) matrix and b a 1D (m,) vector with "
            "matching numbers of rows."
        )
    n = A.shape[1]
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number.")
    if norm not in (1, 2):
        raise ValueError(f"Unsupported norm type: {norm}. Use 1 or 2.")
    if smoothness_order not in (0, 1, 2):
        raise ValueError(
            f"Unsupported smoothness_order: {smoothness_order}. Use 0, 1 or 2."
        )
    if not np.isfinite(smoothness_weight) or smoothness_weight < 0.0:
        raise ValueError(
            "smoothness_weight must be a finite non-negative number."
        )
    if not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite.")
    if enforce_norm and (not np.isfinite(norm_value)):
        raise ValueError("norm_value must be finite when enforce_norm=True.")

    P = A.T @ A
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)
        P = P + alpha * smoothness_weight * (L.T @ L)
    elif norm == 2:
        P = P + alpha * np.eye(n)

    if norm == 1:
        c = -A.T @ b + alpha * np.ones(n)
    else:
        c = -A.T @ b

    names = (
        [str(name) for name in variable_names]
        if variable_names is not None
        else [f"E{i}" for i in range(n)]
    )
    if len(names) != n:
        raise ValueError(
            f"variable_names must have length {n}; got {len(names)}."
        )

    bounds = [(float(lower_bound), None)] * n
    handle = _pyopt.load().QuadraticMatrixProblemHandle
    if enforce_norm:
        return handle(
            sense="min",
            Q=np.asarray(P, dtype=float),
            c=c,
            A_eq=np.ones((1, n)),
            b_eq=[float(norm_value)],
            bounds=bounds,
            variable_names=names,
            equality_names=[equality_name],
        )
    return handle(
        sense="min",
        Q=np.asarray(P, dtype=float),
        c=c,
        bounds=bounds,
        variable_names=names,
    )


def _make_analyzer(
    handle: Any,
    py: SimpleNamespace,
    tolerance: float,
) -> Any:
    """Construct a pyoptexplain Analyzer over a handle's quadratic surface.

    Accepts either a ``QuadraticMatrixProblemHandle`` (whose analysis surface is
    obtained via ``quadratic_representation()``) or an already-built
    representation such as ``QuadraticMatrixScenarioRepresentation``.
    """
    representation = (
        handle.quadratic_representation()
        if hasattr(handle, "quadratic_representation")
        else handle
    )
    return py.Analyzer(
        representation,
        options=py.SolveParameters(
            feasibility_tolerance=tolerance,
            optimality_tolerance=tolerance,
        ),
    )


def solve_interpret(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int = 2,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    enforce_norm: bool = False,
    norm_value: float = 1.0,
    x0: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    variable_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Solve the unfolding QP through pyoptexplain and return the spectrum.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    alpha : float
        Regularization parameter.
    norm : int, optional
        Penalty norm, 1 or 2 (default: 2).
    smoothness_order : int, optional
        Smoothness derivative order, 0, 1 or 2 (default: 0).
    smoothness_weight : float, optional
        Weight of the smoothness term (default: 1.0).
    enforce_norm : bool, optional
        Add ``sum(x) == norm_value`` (default: False).
    norm_value : float, optional
        Target total fluence for the norm equality (default: 1.0).
    x0 : np.ndarray, optional
        Warm start, accepted for API compatibility (unused).
    tolerance : float, optional
        Solver feasibility/optimality tolerance (default: 1e-8).
    variable_names : sequence of str, optional
        Energy-group variable names.

    Returns
    -------
    np.ndarray
        Unfolded spectrum ``(n,)``.

    Raises
    ------
    RuntimeError
        If the solver does not return a primal solution.
    """
    py = _require_pyoptexplain()
    handle = build_interpretation_qp(
        A,
        b,
        alpha,
        norm=norm,
        smoothness_order=smoothness_order,
        smoothness_weight=smoothness_weight,
        enforce_norm=enforce_norm,
        norm_value=norm_value,
        variable_names=variable_names,
    )
    analyzer = _make_analyzer(handle, py, tolerance)
    result = analyzer.solve()
    if result.primal_solution is None:
        raise RuntimeError(
            f"pyoptexplain solve failed with status {result.status!r}."
        )
    return np.asarray(result.primal_solution, dtype=float)


def _detector_sensitivity(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    build_kwargs: Dict[str, Any],
    py: SimpleNamespace,
    tolerance: float,
    base_x: np.ndarray,
    detector_names: Optional[Sequence[str]],
    deltas: Sequence[float],
) -> List[Dict[str, Any]]:
    """Perturb each detector reading one at a time and re-solve the QP."""
    base_l1 = max(1.0, float(np.sum(np.abs(base_x))))
    rows: List[Dict[str, Any]] = []
    for i in range(b.shape[0]):
        name = str(detector_names[i]) if detector_names is not None else str(i)
        for delta in deltas:
            b2 = b.copy()
            b2[i] = b2[i] * (1.0 + delta)
            handle = build_interpretation_qp(A, b2, alpha, **build_kwargs)
            analyzer = _make_analyzer(handle, py, tolerance)
            result = analyzer.solve()
            x2 = (
                np.asarray(result.primal_solution, dtype=float)
                if result.primal_solution is not None
                else base_x
            )
            change = float(np.sum(np.abs(x2 - base_x)) / base_l1)
            rows.append(
                {
                    "detector": name,
                    "delta": float(delta),
                    "spectrum_change": change,
                    "objective_value": result.objective_value,
                    "status": result.status,
                }
            )
    return rows


def _regularization_sweep(
    A: np.ndarray,
    b: np.ndarray,
    base_alpha: float,
    build_kwargs: Dict[str, Any],
    py: SimpleNamespace,
    tolerance: float,
    base_x: np.ndarray,
    E_MeV: Optional[np.ndarray],
    alphas: Optional[Sequence[float]],
) -> List[Dict[str, Any]]:
    """Solve the QP across a grid of regularization parameters."""
    if alphas is None:
        if base_alpha > 0.0:
            alphas = sorted(
                {
                    base_alpha * 0.1,
                    base_alpha * 0.5,
                    base_alpha,
                    base_alpha * 2.0,
                    base_alpha * 10.0,
                }
            )
        else:
            alphas = (0.0, 1e-5, 1e-4, 1e-3)
    base_l1 = max(1.0, float(np.sum(np.abs(base_x))))
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        handle = build_interpretation_qp(A, b, alpha, **build_kwargs)
        analyzer = _make_analyzer(handle, py, tolerance)
        result = analyzer.solve()
        x = (
            np.asarray(result.primal_solution, dtype=float)
            if result.primal_solution is not None
            else base_x
        )
        peak_index = int(np.argmax(x)) if x.size else 0
        row: Dict[str, Any] = {
            "alpha": float(alpha),
            "status": result.status,
            "objective_value": result.objective_value,
            "residual_norm": float(np.linalg.norm(b - A @ x)),
            "spectrum_l1": float(np.sum(x)),
            "peak": float(np.max(x)) if x.size else None,
            "peak_energy": (
                float(E_MeV[peak_index]) if E_MeV is not None else None
            ),
            "change_from_base": float(np.sum(np.abs(x - base_x)) / base_l1),
        }
        rows.append(row)
    return rows


def _nonnegativity_relaxation(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    build_kwargs: Dict[str, Any],
    py: SimpleNamespace,
    tolerance: float,
    base_x: np.ndarray,
    deltas: Sequence[float],
) -> List[Dict[str, Any]]:
    """Allow small negative values and measure how the solution moves."""
    base_l1 = max(1.0, float(np.sum(np.abs(base_x))))
    rows: List[Dict[str, Any]] = []
    for delta in deltas:
        if delta < 0.0 or not np.isfinite(delta):
            raise ValueError(
                f"nonnegativity relaxation deltas must be >= 0; got {delta}."
            )
        handle = build_interpretation_qp(
            A,
            b,
            alpha,
            lower_bound=-float(delta),
            **build_kwargs,
        )
        analyzer = _make_analyzer(handle, py, tolerance)
        result = analyzer.solve()
        x = (
            np.asarray(result.primal_solution, dtype=float)
            if result.primal_solution is not None
            else base_x
        )
        rows.append(
            {
                "allowed_negative": float(delta),
                "lower_bound": float(-delta),
                "status": result.status,
                "objective_value": result.objective_value,
                "min_spectrum": float(np.min(x)) if x.size else None,
                "change_from_base": float(np.sum(np.abs(x - base_x)) / base_l1),
            }
        )
    return rows
