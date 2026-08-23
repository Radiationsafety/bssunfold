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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._base_unfolder import _build_system, run_unfolding
from .regularization import (
    cosine_similarity_selection,
    select_regularization_parameter,
)
from ._interpret_pyopt import (
    _pyopt,  # noqa: F401  # re-exported; tests reset its lazy cache via this attribute
    _require_pyoptexplain,
    build_interpretation_qp,
    _make_analyzer,
    solve_interpret,
    _detector_sensitivity,
    _regularization_sweep,
    _nonnegativity_relaxation,
)
from ._interpret_report import (
    InterpretationResult,
    _rows_to_frame,
    _build_metrics,
    _build_report,
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

    build_kwargs: Dict[str, Any] = {
        "norm": norm,
        "smoothness_order": smoothness_order,
        "smoothness_weight": smoothness_weight,
        "enforce_norm": enforce_norm,
        "norm_value": norm_value,
    }

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
                "squared_share": float(res_i**2) / max(1e-12, residual_norm**2),
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
            idx = (
                int(name[1:])
                if name.startswith("E") and name[1:].isdigit()
                else None
            )
            if idx is None:
                continue
            value = (
                float(row.get("value", np.nan))
                if row.get("value") is not None
                else np.nan
            )
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
        model={
            "norm": norm,
            "alpha": float(alpha),
            "smoothness_order": smoothness_order,
            "smoothness_weight": float(smoothness_weight),
            "enforce_norm": bool(enforce_norm),
            "norm_value": float(norm_value) if enforce_norm else None,
            "n_energy_bins": n,
            "n_detectors": int(b.shape[0]),
        },
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
        # x0 is forwarded for API consistency; solve_interpret documents that
        # the warm start is currently unused.
        x0 = kwargs.pop("x0", None)
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
            x0=x0,
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
