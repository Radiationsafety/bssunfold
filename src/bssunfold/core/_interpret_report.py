"""Report assembly and metric helpers for unfolding interpretation.

Extracted from ``unfold_interpret.py``. Pure leaf: renders the
``InterpretationResult`` dataclass, the Markdown report, and the JSON-friendly
metrics dictionary. No ``pyoptexplain`` dependency; ``pandas`` is imported
lazily inside the functions that need it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


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
        by_name[name] = max(
            by_name.get(name, 0.0), float(row["spectrum_change"])
        )
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
        if r["target"] == "objective"
        and r["max_variable_change_relative"] is not None
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
    import pandas as pd

    if E_MeV is not None:
        spectrum_table = {
            "energy_MeV": [float(e) for e in E_MeV],
            "spectrum": x.tolist(),
        }
        spec_df = pd.DataFrame(spectrum_table)
    else:
        spec_df = pd.DataFrame(
            {"group": [f"E{i}" for i in range(x.size)], "spectrum": x.tolist()}
        )
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
        sections.append(
            "- No energy group is pinned at the non-negativity bound."
        )
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


def _conclusions(metrics: Dict[str, Any], enforce_norm: bool) -> List[str]:
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
            best = min(feasible, key=lambda r: (r["residual_norm"], r["alpha"]))
            lines.append(
                f"In the regularization sweep, alpha={best['alpha']:.6g} "
                f"gives the smallest residual norm "
                f"({best['residual_norm']:.6g})."
            )

    return lines
