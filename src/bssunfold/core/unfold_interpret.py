"""pyoptexplain-based interpretation of the unfolding quadratic program.

This module is not just another unfold solver: it *solves* the same quadratic
program that ``unfold_qpsolvers``/``unfold_cvxpy`` use (``min 0.5 x'Qx + c'x``
with ``x >= 0``) through the `pyoptexplain` post-optimality analysis package and
then *interprets* the solution.

The output of :func:`interpret_qp` is an :class:`InterpretationResult` with:

- ``report``  -- a human-readable Markdown report.
- ``metrics`` -- a JSON-friendly dictionary of quantitative diagnostics.
- ``tables``  -- the raw ``pandas.DataFrame`` tables behind the report.

What the interpretation answers:

1. **Solve report** -- solver status, objective value, per-group spectrum,
   residual per detector, active (zeroed) energy groups.
2. **Shadow prices (duals)** -- bound duals for the non-negativity constraint
   (the "price" of each energy group) and, when ``enforce_norm=True``, the dual
   of the ``sum(x) == norm_value`` equality.
3. **Robustness** -- empirical perturbation analysis: how the spectrum moves
   when the measurements (objective ``c = -A'b``) are perturbed by
   ``+/-1%..5%``.
4. **Detector informativeness** -- how much the spectrum changes when *one*
   detector reading at a time is perturbed.
5. **Regularization sweep** -- how the solution changes across a grid of
   regularization parameters ``alpha``.
6. **Non-negativity trust** -- what happens if small negative values are
   allowed (``lower_bound < 0``).
7. **Scenarios** -- pyoptexplain structured what-if cases (norm RHS shifts
   when ``enforce_norm=True``, bound relaxations).
8. **Constraint relaxation** -- ``relaxation_curve`` on the ``norm`` block when
   ``enforce_norm=True``.

``pyoptexplain`` is an optional dependency. Install it with::

    pip install bssunfold[interpret]
"""

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from ._matrix_utils import create_derivative_matrix
from .regularization import (
    cosine_similarity_selection,
    select_regularization_parameter,
)

__all__ = [
    "InterpretationResult",
    "build_interpretation_qp",
    "solve_interpret",
    "interpret_qp",
    "unfold_interpret",
]

logger = logging.getLogger(__name__)

_DEFAULT_RELATIVE_DELTAS = (-0.05, -0.01, 0.01, 0.05)
_DEFAULT_RELAXATION_DELTAS = (0.0, 0.05, 0.1)
_DEFAULT_NONNEG_DELTAS = (0.0, 0.01, 0.05)
_DEFAULT_SENSITIVITY_DELTAS = (0.01, 0.05)


class _PyOptExplainNamespace:
    """Lazily import the pyoptexplain pieces needed by this module."""

    def __init__(self) -> None:
        self._loaded: Optional[SimpleNamespace] = None

    def load(self) -> SimpleNamespace:
        if self._loaded is not None:
            return self._loaded
        try:
            import pyoptexplain  # noqa: F401
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
        raise ValueError("smoothness_weight must be a finite non-negative number.")
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


def _spectrum_columns(
    scenario_df: Any,
    n: int,
    prefix: str = "variable:",
) -> Dict[str, List[float]]:
    """Extract per-scenario spectra from a pyoptexplain scenario frame."""
    spectra: Dict[str, List[float]] = {}
    cols = [f"{prefix}E{i}" for i in range(n)]
    present = [c for c in cols if c in scenario_df.columns]
    for _, row in scenario_df.iterrows():
        name = str(row["scenario"])
        values = [row[c] for c in present]
        spectra[name] = (
            [float(v) for v in values] if all(v is not None for v in values)
            else []
        )
    return spectra


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
        name = (
            str(detector_names[i]) if detector_names is not None else str(i)
        )
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


@dataclass
class InterpretationResult:
    """Output of :func:`interpret_qp`.

    Attributes
    ----------
    spectrum : np.ndarray
        Unfolded spectrum from the pyoptexplain QP solve.
    status : str
        Solver status (e.g. ``"optimal"``).
    objective_value : Optional[float]
        Objective value ``0.5 * x'Qx + c'x``.
    report : str
        Rendered Markdown report.
    metrics : dict
        JSON-friendly dictionary of diagnostics.
    tables : dict
        Raw ``pandas.DataFrame`` tables behind the report.
    """

    spectrum: np.ndarray
    status: str
    objective_value: Optional[float]
    report: str
    metrics: Dict[str, Any]
    tables: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return ``report`` and ``metrics`` as a plain dictionary."""
        return {
            "spectrum": self.spectrum.tolist(),
            "status": self.status,
            "objective_value": self.objective_value,
            "report": self.report,
            "interpretation_metrics": self.metrics,
        }


def _fmt(value: Any, float_format: str = "{:.6g}") -> str:
    """Render one cell value for a Markdown table."""
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (float, np.floating, np.integer)):
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            return "—"
        try:
            return float_format.format(value)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return "…"
    return str(value)


def _df_to_markdown(df: Any, float_format: str = "{:.6g}") -> str:
    """Render a pandas DataFrame as a compact Markdown table."""
    if df is None or len(df) == 0:
        return "_No data._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- "] * len(headers)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [_fmt(row[col], float_format) for col in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interpret_qp(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    *,
    norm: int = 2,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    enforce_norm: bool = False,
    norm_value: float = 1.0,
    E_MeV: Optional[np.ndarray] = None,
    detector_names: Optional[Sequence[str]] = None,
    tolerance: float = 1e-8,
    relative_deltas: Sequence[float] = _DEFAULT_RELATIVE_DELTAS,
    relaxation_deltas: Sequence[float] = _DEFAULT_RELAXATION_DELTAS,
    nonneg_deltas: Sequence[float] = _DEFAULT_NONNEG_DELTAS,
    sensitivity_deltas: Sequence[float] = _DEFAULT_SENSITIVITY_DELTAS,
    regularization_sweep: Optional[Sequence[float]] = None,
    run_robustness: bool = True,
    run_scenarios: bool = True,
    run_detector_sensitivity: bool = True,
    run_regularization_sweep: bool = True,
    run_nonnegativity_relaxation: bool = True,
) -> InterpretationResult:
    """Solve and interpret the unfolding QP with pyoptexplain.

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
    E_MeV : np.ndarray, optional
        Energy grid for the report tables.
    detector_names : sequence of str, optional
        Names of the detector rows of ``A``/``b``.
    tolerance : float, optional
        Solver feasibility/optimality tolerance (default: 1e-8).
    relative_deltas : sequence of float, optional
        Relative perturbations for the robustness analysis (default: -5..5%).
    relaxation_deltas : sequence of float, optional
        RHS deltas for the ``norm`` relaxation curve (default: 0, 0.05, 0.1).
    nonneg_deltas : sequence of float, optional
        Allowed-negative magnitudes to probe (default: 0, 0.01, 0.05).
    sensitivity_deltas : sequence of float, optional
        Per-detector relative perturbations (default: 1%, 5%).
    regularization_sweep : sequence of float, optional
        Explicit alpha grid; default derives a grid around ``alpha``.
    run_robustness : bool, optional
        Run the perturbation robustness analysis (default: True).
    run_scenarios : bool, optional
        Run the pyoptexplain structured scenarios (default: True).
    run_detector_sensitivity : bool, optional
        Run the per-detector sensitivity analysis (default: True).
    run_regularization_sweep : bool, optional
        Run the regularization sweep (default: True).
    run_nonnegativity_relaxation : bool, optional
        Run the non-negativity relaxation analysis (default: True).

    Returns
    -------
    InterpretationResult
        The interpreted solution (spectrum, report, metrics, tables).
    """
    py = _require_pyoptexplain()

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    build_kwargs: Dict[str, Any] = dict(
        norm=norm,
        smoothness_order=smoothness_order,
        smoothness_weight=smoothness_weight,
        enforce_norm=enforce_norm,
        norm_value=norm_value,
    )

    handle = build_interpretation_qp(A, b, alpha, **build_kwargs)
    analyzer = _make_analyzer(handle, py, tolerance)
    result = analyzer.solve()
    if result.primal_solution is None:
        raise RuntimeError(
            f"pyoptexplain solve failed with status {result.status!r}."
        )
    x = np.asarray(result.primal_solution, dtype=float)
    x = np.maximum(x, 0.0) if not enforce_norm else x
    spectrum = x.copy()

    residual = b - A @ x
    residual_norm = float(np.linalg.norm(residual))

    summary_df = analyzer.summary()
    variables_df = analyzer.variables()
    constraints_df = analyzer.constraints()
    binding_df = analyzer.binding_constraints()
    duals_df = analyzer.duals()
    dual_details_df = analyzer.dual_details()
    capabilities_df = analyzer.capabilities()
    quadratic_df = analyzer.quadratic_objective()

    # --- detector diagnostics ------------------------------------------------
    detector_rows: List[Dict[str, Any]] = []
    for i in range(b.shape[0]):
        name = detector_names[i] if detector_names is not None else str(i)
        reading = float(b[i])
        computed = float(A[i] @ x)
        res_i = reading - computed
        detector_rows.append(
            {
                "detector": str(name),
                "reading": reading,
                "effective": computed,
                "residual": res_i,
                "relative_residual": res_i / max(1e-12, abs(reading)),
                "squared_share": float(res_i ** 2) / max(1e-12, residual_norm ** 2),
            }
        )

    # --- duals ---------------------------------------------------------------
    bound_duals: Dict[str, float] = {}
    if len(duals_df) and "name" in duals_df.columns:
        for _, row in duals_df.iterrows():
            name = str(row["name"])
            dual = row.get("dual")
            if name.startswith("E") and "_lower_bound" in name:
                bound_duals[name.split("_lower_bound")[0]] = (
                    float(dual) if dual is not None else 0.0
                )
    norm_dual = None
    if enforce_norm and len(duals_df):
        norm_rows = duals_df[duals_df["name"] == "norm"]
        if len(norm_rows):
            dual = norm_rows.iloc[0].get("dual")
            norm_dual = float(dual) if dual is not None else None

    # --- active groups -------------------------------------------------------
    active_groups: List[int] = []
    zero_groups: List[int] = []
    if len(variables_df) and "name" in variables_df.columns:
        at_lower = (
            variables_df["at_lower_bound"].fillna(False).astype(bool)
            if "at_lower_bound" in variables_df.columns
            else None
        )
        for i, row in variables_df.iterrows():
            name = str(row["name"])
            idx = int(name[1:]) if name.startswith("E") and name[1:].isdigit() else None
            if idx is None:
                continue
            value = float(row.get("value", np.nan)) if row.get("value") is not None else np.nan
            if at_lower is not None and bool(at_lower.iloc[i]):
                active_groups.append(idx)
            if value <= tolerance or np.isclose(value, 0.0, atol=tolerance):
                zero_groups.append(idx)

    # --- robustness ----------------------------------------------------------
    robustness_summary = None
    if run_robustness:
        targets = ["objective"]
        if enforce_norm:
            targets.append("equality_rhs")
        plan = py.PerturbationPlan(
            targets=tuple(targets),
            relative_deltas=tuple(float(d) for d in relative_deltas),
        )
        robustness = analyzer.perturbation_robustness(plan)
        robustness_summary = robustness.summary()

    # --- norm relaxation curve ----------------------------------------------
    relaxation_df = None
    if enforce_norm:
        curve = analyzer.relaxation_curve(
            "norm", [float(d) for d in relaxation_deltas]
        )
        relaxation_df = curve.to_frame()

    # --- scenarios -----------------------------------------------------------
    scenario_df = None
    if run_scenarios:
        scenarios: Dict[str, Any] = {
            "base": py.ScenarioCase(),
        }
        if enforce_norm:
            scenarios["norm_target_-10%"] = py.ScenarioCase(
                (py.ChangeRHS("norm", delta=-0.1 * norm_value),)
            )
            scenarios["norm_target_+10%"] = py.ScenarioCase(
                (py.ChangeRHS("norm", delta=0.1 * norm_value),)
            )
        scenarios["allow_neg_1pct"] = py.ScenarioCase(
            tuple(
                py.SetVariableBounds(f"E{i}", lower=-0.01, upper=None)
                for i in range(n)
            )
        )
        scenarios["allow_neg_5pct"] = py.ScenarioCase(
            tuple(
                py.SetVariableBounds(f"E{i}", lower=-0.05, upper=None)
                for i in range(n)
            )
        )
        scenario_handle = py.QuadraticMatrixScenarioRepresentation.from_matrix(
            handle.quadratic_representation()
        )
        scenario_analyzer = _make_analyzer(scenario_handle, py, tolerance)
        scenario_df = scenario_analyzer.run_scenarios(scenarios)

    # --- detector sensitivity ------------------------------------------------
    sensitivity_rows: List[Dict[str, Any]] = []
    if run_detector_sensitivity:
        sensitivity_rows = _detector_sensitivity(
            A,
            b,
            float(alpha),
            build_kwargs,
            py,
            tolerance,
            x,
            detector_names,
            tuple(float(d) for d in sensitivity_deltas),
        )

    # --- regularization sweep ------------------------------------------------
    sweep_rows: List[Dict[str, Any]] = []
    if run_regularization_sweep:
        sweep_rows = _regularization_sweep(
            A,
            b,
            float(alpha),
            build_kwargs,
            py,
            tolerance,
            x,
            E_MeV,
            regularization_sweep,
        )

    # --- non-negativity relaxation ------------------------------------------
    nonneg_rows: List[Dict[str, Any]] = []
    if run_nonnegativity_relaxation:
        nonneg_rows = _nonnegativity_relaxation(
            A,
            b,
            float(alpha),
            build_kwargs,
            py,
            tolerance,
            x,
            tuple(float(d) for d in nonneg_deltas),
        )

    # --- tables --------------------------------------------------------------
    tables: Dict[str, Any] = {
        "summary": summary_df,
        "variables": variables_df,
        "constraints": constraints_df,
        "binding_constraints": binding_df,
        "duals": duals_df,
        "dual_details": dual_details_df,
        "capabilities": capabilities_df,
        "quadratic_objective": quadratic_df,
        "detectors": _rows_to_frame(detector_rows),
        "detector_sensitivity": _rows_to_frame(sensitivity_rows),
        "regularization_sweep": _rows_to_frame(sweep_rows),
        "nonnegativity_relaxation": _rows_to_frame(nonneg_rows),
    }
    if relaxation_df is not None:
        tables["norm_relaxation"] = relaxation_df
    if scenario_df is not None:
        tables["scenarios"] = scenario_df

    metrics = _build_metrics(
        result=result,
        x=x,
        E_MeV=E_MeV,
        residual_norm=residual_norm,
        Q=np.asarray(handle.quadratic_representation().Q, dtype=float),
        model=dict(
            norm=norm,
            alpha=float(alpha),
            smoothness_order=smoothness_order,
            smoothness_weight=float(smoothness_weight),
            enforce_norm=bool(enforce_norm),
            norm_value=float(norm_value) if enforce_norm else None,
            n_energy_bins=n,
            n_detectors=int(b.shape[0]),
        ),
        active_groups=active_groups,
        zero_groups=zero_groups,
        bound_duals=bound_duals,
        norm_dual=norm_dual,
        detector_rows=detector_rows,
        sensitivity_rows=sensitivity_rows,
        robustness_summary=robustness_summary,
        relaxation_df=relaxation_df,
        nonneg_rows=nonneg_rows,
        scenario_df=scenario_df,
        sweep_rows=sweep_rows,
        capabilities_df=capabilities_df,
    )

    report = _build_report(
        x=x,
        E_MeV=E_MeV,
        residual_norm=residual_norm,
        summary_df=summary_df,
        variables_df=variables_df,
        constraints_df=constraints_df,
        binding_df=binding_df,
        duals_df=duals_df,
        detector_rows=detector_rows,
        sensitivity_rows=sensitivity_rows,
        robustness_summary=robustness_summary,
        relaxation_df=relaxation_df,
        nonneg_rows=nonneg_rows,
        scenario_df=scenario_df,
        sweep_rows=sweep_rows,
        metrics=metrics,
        enforce_norm=enforce_norm,
    )

    return InterpretationResult(
        spectrum=spectrum,
        status=result.status,
        objective_value=result.objective_value,
        report=report,
        metrics=metrics,
        tables=tables,
    )


def _rows_to_frame(rows: List[Dict[str, Any]]):
    """Convert a list of dicts into a DataFrame without pandas at top level."""
    import pandas as pd

    return pd.DataFrame(rows)


def _build_metrics(
    *,
    result: Any,
    x: np.ndarray,
    E_MeV: Optional[np.ndarray],
    residual_norm: float,
    Q: np.ndarray,
    model: Dict[str, Any],
    active_groups: List[int],
    zero_groups: List[int],
    bound_duals: Dict[str, float],
    norm_dual: Optional[float],
    detector_rows: List[Dict[str, Any]],
    sensitivity_rows: List[Dict[str, Any]],
    robustness_summary: Any,
    relaxation_df: Any,
    nonneg_rows: List[Dict[str, Any]],
    scenario_df: Any,
    sweep_rows: List[Dict[str, Any]],
    capabilities_df: Any,
) -> Dict[str, Any]:
    """Assemble the JSON-friendly metrics dictionary."""
    metrics: Dict[str, Any] = {
        "status": result.status,
        "success": bool(result.success),
        "objective_value": result.objective_value,
        "solver_name": result.solver_name,
        "solve_time": result.solve_time,
        "model": model,
        "spectrum": x.tolist(),
        "energy": E_MeV.tolist() if E_MeV is not None else None,
        "residual_norm": residual_norm,
        "condition_number": _safe_cond(Q),
        "active_groups": active_groups,
        "zero_groups": zero_groups,
        "bound_duals": bound_duals,
        "norm_dual": norm_dual,
        "detectors": detector_rows,
        "detector_sensitivity": sensitivity_rows,
        "detector_importance": _detector_importance(sensitivity_rows),
        "nonnegativity_relaxation": nonneg_rows,
        "regularization_sweep": sweep_rows,
        "capabilities": _effective_capabilities(capabilities_df),
    }
    if robustness_summary is not None:
        metrics["robustness"] = _robustness_metrics(robustness_summary)
    if relaxation_df is not None:
        metrics["norm_relaxation"] = _frame_records(relaxation_df)
    if scenario_df is not None:
        metrics["scenarios"] = _scenario_metrics(scenario_df)
    return metrics


def _safe_cond(Q: np.ndarray) -> Optional[float]:
    try:
        value = float(np.linalg.cond(Q))
    except np.linalg.LinAlgError:
        return None
    return value if np.isfinite(value) else None


def _effective_capabilities(capabilities_df: Any) -> List[str]:
    if capabilities_df is None or len(capabilities_df) == 0:
        return []
    if "capability" not in capabilities_df.columns:
        return []
    return [
        str(name)
        for name, effective in zip(
            capabilities_df["capability"], capabilities_df["effective"]
        )
        if bool(effective)
    ]


def _detector_importance(
    sensitivity_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank detectors by their maximum spectral impact."""
    by_name: Dict[str, float] = {}
    for row in sensitivity_rows:
        name = row["detector"]
        by_name[name] = max(by_name.get(name, 0.0), float(row["spectrum_change"]))
    ranked = sorted(by_name.items(), key=lambda item: item[1], reverse=True)
    return [
        {"detector": name, "max_spectrum_change": value}
        for name, value in ranked
    ]


def _robustness_metrics(robustness_summary: Any) -> Dict[str, Any]:
    """Summarize the perturbation-robustness frame into scalar metrics."""
    if robustness_summary is None or len(robustness_summary) == 0:
        return {"cases": 0}
    cases = robustness_summary[robustness_summary["target"] != "base"]
    rows = []
    for _, row in cases.iterrows():
        rows.append(
            {
                "case": str(row["case"]),
                "target": str(row["target"]),
                "magnitude": _float_or_none(row.get("magnitude")),
                "status": str(row.get("status")),
                "objective_change": _float_or_none(row.get("objective_change")),
                "objective_change_relative": _float_or_none(
                    row.get("objective_change_relative")
                ),
                "max_variable_change_relative": _float_or_none(
                    row.get("max_variable_change_relative")
                ),
                "binding_similarity": _float_or_none(
                    row.get("binding_similarity")
                ),
                "regime_changed": bool(row.get("regime_changed", False)),
            }
        )
    deltas = [
        r["max_variable_change_relative"]
        for r in rows
        if r["target"] == "objective" and r["max_variable_change_relative"] is not None
    ]
    return {
        "case_count": len(rows),
        "max_spectrum_change_relative": max(deltas) if deltas else None,
        "cases": rows,
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_records(frame: Any) -> List[Dict[str, Any]]:
    if frame is None or len(frame) == 0:
        return []
    return frame.to_dict(orient="records")


def _scenario_metrics(scenario_df: Any) -> List[Dict[str, Any]]:
    """Extract compact per-scenario metrics (spectra dropped)."""
    if scenario_df is None or len(scenario_df) == 0:
        return []
    rows = []
    for _, row in scenario_df.iterrows():
        rows.append(
            {
                "scenario": str(row.get("scenario")),
                "status": str(row.get("status")),
                "objective_value": _float_or_none(row.get("objective_value")),
                "objective_change": _float_or_none(row.get("objective_change")),
            }
        )
    return rows


def _build_report(
    *,
    x: np.ndarray,
    E_MeV: Optional[np.ndarray],
    residual_norm: float,
    summary_df: Any,
    variables_df: Any,
    constraints_df: Any,
    binding_df: Any,
    duals_df: Any,
    detector_rows: List[Dict[str, Any]],
    sensitivity_rows: List[Dict[str, Any]],
    robustness_summary: Any,
    relaxation_df: Any,
    nonneg_rows: List[Dict[str, Any]],
    scenario_df: Any,
    sweep_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    enforce_norm: bool,
) -> str:
    """Render the Markdown report."""
    sections: List[str] = [
        "# Unfolding interpretation report (pyoptexplain)",
        "",
    ]
    sections.append("## Solve summary")
    sections.append(_df_to_markdown(summary_df))
    sections.append("")

    # spectrum table
    if E_MeV is not None:
        spectrum_table = {
            "energy_MeV": [float(e) for e in E_MeV],
            "spectrum": x.tolist(),
        }
        import pandas as pd

        spec_df = pd.DataFrame(spectrum_table)
    else:
        spec_df = pd.DataFrame({"group": [f"E{i}" for i in range(x.size)], "spectrum": x.tolist()})
    sections.append("## Spectrum")
    sections.append(_df_to_markdown(spec_df))
    sections.append("")

    sections.append("## Active constraints")
    active = metrics["active_groups"]
    if active:
        preview = ", ".join(str(i) for i in active[:12])
        if len(active) > 12:
            preview += f", ... ({len(active)} total)"
        sections.append(
            f"- **{len(active)}** energy groups are pinned at the "
            f"non-negativity bound: {preview}."
        )
    else:
        sections.append("- No energy group is pinned at the non-negativity bound.")
    sections.append("")

    if constraints_df is not None and len(constraints_df):
        sections.append("## Constraint table")
        sections.append(_df_to_markdown(constraints_df))
        sections.append("")
    if binding_df is not None and len(binding_df):
        sections.append("## Binding constraints")
        sections.append(_df_to_markdown(binding_df))
        sections.append("")

    sections.append("## Duals (shadow prices)")
    if len(duals_df):
        sections.append(_df_to_markdown(duals_df))
    else:
        sections.append("_No duals available._")
    sections.append("")

    sections.append("## Detector diagnostics")
    sections.append(_df_to_markdown(_rows_to_frame(detector_rows)))
    sections.append("")

    if sensitivity_rows:
        sections.append("## Detector sensitivity")
        sections.append(_df_to_markdown(_rows_to_frame(sensitivity_rows)))
        sections.append("")

    if sweep_rows:
        sections.append("## Regularization sweep")
        sections.append(_df_to_markdown(_rows_to_frame(sweep_rows)))
        sections.append("")

    if nonneg_rows:
        sections.append("## Non-negativity relaxation")
        sections.append(_df_to_markdown(_rows_to_frame(nonneg_rows)))
        sections.append("")

    if robustness_summary is not None:
        sections.append("## Perturbation robustness")
        sections.append(_df_to_markdown(robustness_summary))
        sections.append("")

    if relaxation_df is not None and len(relaxation_df):
        sections.append("## Norm constraint relaxation")
        sections.append(_df_to_markdown(relaxation_df))
        sections.append("")

    if scenario_df is not None and len(scenario_df):
        section_cols = [
            c
            for c in scenario_df.columns
            if c
            in (
                "scenario",
                "status",
                "objective_value",
                "objective_change",
            )
        ]
        sections.append("## Scenarios")
        sections.append(_df_to_markdown(scenario_df[section_cols]))
        sections.append("")

    sections.append("## Interpretation")
    sections.extend(
        " - " + line for line in _conclusions(metrics, enforce_norm)
    )
    sections.append("")

    return "\n".join(sections)


def _conclusions(
    metrics: Dict[str, Any], enforce_norm: bool
) -> List[str]:
    """Generate natural-language interpretation bullets."""
    lines: List[str] = []

    active = metrics.get("active_groups") or []
    if active:
        lines.append(
            f"The spectrum has no contribution in {len(active)} energy "
            f"group(s) ({', '.join(map(str, active[:10]))}"
            f"{', ...' if len(active) > 10 else ''}): the non-negativity "
            "constraint is active there."
        )
    else:
        lines.append(
            "The non-negativity constraint is inactive for every energy "
            "group: the unconstrained optimum is non-negative."
        )

    importance = metrics.get("detector_importance") or []
    if importance:
        top = importance[0]
        lines.append(
            f"Most informative detector: '{top['detector']}' — perturbing "
            f"its reading by 1-5% moves the spectrum by up to "
            f"{top['max_spectrum_change']:.3f} (relative L1)."
        )

    robustness = metrics.get("robustness") or {}
    max_spec = robustness.get("max_spectrum_change_relative")
    if max_spec is not None:
        if max_spec <= 0.05:
            lines.append(
                f"Robust to measurement noise: a 1-5% perturbation of the "
                f"readings changes the spectrum by at most {max_spec:.1%} "
                "(relative L1)."
            )
        else:
            lines.append(
                f"Sensitive to measurement noise: a 1-5% perturbation of the "
                f"readings changes the spectrum by up to {max_spec:.1%} "
                "(relative L1); consider stronger regularization or better "
                "measurement statistics."
            )

    norm_dual = metrics.get("norm_dual")
    if enforce_norm and norm_dual is not None:
        lines.append(
            f"The shadow price of the total-fluence constraint is "
            f"{norm_dual:.6g}."
        )

    nonneg = metrics.get("nonnegativity_relaxation") or []
    if nonneg:
        worst = max(
            (r["change_from_base"] for r in nonneg if r["status"] == "optimal"),
            default=None,
        )
        if worst is not None:
            if worst <= 0.05:
                lines.append(
                    f"Allowing small negative values changes the spectrum by "
                    f"at most {worst:.1%}: the non-negativity assumption is "
                    "not driving the solution."
                )
            else:
                lines.append(
                    f"Allowing small negative values changes the spectrum by "
                    f"up to {worst:.1%}: the solution leans on the "
                    "non-negativity bound and the model may need revision."
                )

    sweep = metrics.get("regularization_sweep") or []
    if sweep:
        feasible = [r for r in sweep if r["status"] == "optimal"]
        if feasible:
            best = min(
                feasible, key=lambda r: (r["residual_norm"], r["alpha"])
            )
            lines.append(
                f"In the regularization sweep, alpha={best['alpha']:.6g} "
                f"gives the smallest residual norm "
                f"({best['residual_norm']:.6g})."
            )

    return lines


def unfold_interpret(
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
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    enforce_norm: bool = False,
    norm_value: float = 1.0,
    regularization_method: str = "manual",
    noise_var: Optional[float] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
    tolerance: float = 1e-8,
    interpret_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unfold and interpret a neutron spectrum with pyoptexplain.

    The unfolding QP (identical to ``unfold_qpsolvers``) is solved through
    pyoptexplain and the solution is interpreted. The returned dictionary is a
    standard bssunfold result dict with two extra keys:

    - ``report``                 -- Markdown interpretation report.
    - ``interpretation_metrics`` -- JSON-friendly metrics dictionary.

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
        Initial spectrum guess (used by some regularization methods).
    regularization : float, optional
        Regularization parameter (default: 1e-4).
    norm : int, optional
        Penalty norm, 1 or 2 (default: 2).
    smoothness_order : int, optional
        Smoothness derivative order, 0, 1 or 2 (default: 0).
    smoothness_weight : float, optional
        Weight of the smoothness term (default: 1.0).
    enforce_norm : bool, optional
        Add ``sum(x) == norm_value`` (default: False).
    norm_value : float, optional
        Target total fluence (default: 1.0).
    regularization_method : str, optional
        Method for selecting the regularization parameter.
    noise_var : float, optional
        Noise variance for the discrepancy principle ('dp' method).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.
    tolerance : float, optional
        Solver feasibility/optimality tolerance (default: 1e-8). Pyoptexplain's
        backend may fail with ``iteration_limit`` on large problems at the
        strictest tolerance; relax it (e.g. 1e-5) in that case.
    interpret_options : dict, optional
        Extra keyword arguments forwarded to :func:`interpret_qp`.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result plus ``report`` and
        ``interpretation_metrics`` keys.
    """
    _require_pyoptexplain()

    A, b, selected = _build_system(readings, detector_names, sensitivities)

    if regularization_method == "manual":
        alpha = float(regularization)
        selected_lambda = alpha
    elif regularization_method == "cosine":
        if initial_spectrum is None:
            raise ValueError(
                "For 'cosine' regularization method, "
                "initial_spectrum must be provided."
            )
        initial_spectrum_norm = np.maximum(initial_spectrum, 0)
        if len(initial_spectrum_norm) != n_energy_bins:
            raise ValueError(
                f"Initial spectrum length ({len(initial_spectrum)}) "
                f"must match number of energy bins ({n_energy_bins})"
            )
        selected_lambda = cosine_similarity_selection(
            A, b, initial_spectrum_norm, norm=norm
        )
        alpha = float(selected_lambda)
    else:
        try:
            selected_lambda = select_regularization_parameter(
                A, b, method=regularization_method, noise_var=noise_var
            )
        except Exception as exc:
            raise ValueError(
                f"Regularization selection failed: {exc}. "
                "Consider using manual regularization."
            ) from exc
        alpha = float(selected_lambda)

    def solve_wrapper(A, b, **kwargs):
        kwargs.pop("x0", None)
        return solve_interpret(
            A,
            b,
            alpha,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            enforce_norm=enforce_norm,
            norm_value=norm_value,
            tolerance=tolerance,
        )

    output = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.zeros(n_energy_bins),
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="interpret",
        extra_output={
            "norm": norm,
            "regularization": regularization,
            "regularization_method": regularization_method,
            "selected_regularization": float(selected_lambda),
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "enforce_norm": enforce_norm,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )

    options = dict(interpret_options or {})
    options.setdefault("tolerance", tolerance)
    interpretation = interpret_qp(
        A,
        b,
        alpha,
        norm=norm,
        smoothness_order=smoothness_order,
        smoothness_weight=smoothness_weight,
        enforce_norm=enforce_norm,
        norm_value=norm_value,
        E_MeV=E_MeV,
        detector_names=selected,
        **options,
    )
    output["report"] = interpretation.report
    output["interpretation_metrics"] = interpretation.metrics
    output["interpretation_spectrum"] = interpretation.spectrum.tolist()
    return output
