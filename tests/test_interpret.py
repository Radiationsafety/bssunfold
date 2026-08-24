"""Tests for the pyoptexplain-based interpretation module.

``pyoptexplain`` is an optional backend installed in the dev group. These tests
cover the QP construction (:func:`build_interpretation_qp`), the core solver
(:func:`solve_interpret`), the full interpretation entry point
(:func:`interpret_qp`), and the two ``Detector``-level wrappers
(``unfold_interpret`` and ``interpret_result``).
"""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("pyoptexplain")

from bssunfold import Detector  # noqa: E402
from tests.conftest import block_import  # noqa: E402


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def small_readings(detector):
    sel = detector.detector_names[:6]
    return {name: float(100 - 10 * i) for i, name in enumerate(sel)}


@pytest.fixture
def system(detector, small_readings):
    A = np.array(
        [detector.sensitivities[name] for name in small_readings], dtype=float
    )
    b = np.array(list(small_readings.values()), dtype=float)
    return A, b


def _reset_pyopt_cache():
    """Force the lazy pyoptexplain namespace to reload."""
    import sys

    ui = sys.modules["bssunfold.core.unfold_interpret"]
    ui._pyopt._loaded = None


def test_core_exports():
    """Interpretation API is exported from bssunfold.core."""
    from bssunfold.core import (  # noqa: PLC0415
        InterpretationResult,
        build_interpretation_qp,
        interpret_qp,
        solve_interpret,
        unfold_interpret,
    )

    assert InterpretationResult is not None
    assert build_interpretation_qp is not None
    assert interpret_qp is not None
    assert solve_interpret is not None
    assert unfold_interpret is not None


def test_detector_methods_exist():
    """The two Detector-level wrappers are available."""
    assert hasattr(Detector, "unfold_interpret")
    assert hasattr(Detector, "interpret_result")


def test_build_interpretation_qp(system):
    """The handle carries the expected QP arrays."""
    from bssunfold.core.unfold_interpret import build_interpretation_qp

    A, b = system
    handle = build_interpretation_qp(
        A, b, 1e-4, variable_names=[f"E{i}" for i in range(A.shape[1])]
    )
    rep = handle.quadratic_representation()
    assert rep.Q.shape == (A.shape[1], A.shape[1])
    assert rep.c.shape == (A.shape[1],)
    assert np.allclose(
        rep.bounds[:, 0]
        if hasattr(rep.bounds, "ndim")
        else [lo for lo, _ in rep.bounds],
        0.0,
    )


def test_build_interpretation_qp_enforce_norm(system):
    """enforce_norm adds a sum(x) == norm_value equality block."""
    from bssunfold.core.unfold_interpret import build_interpretation_qp

    A, b = system
    handle = build_interpretation_qp(
        A, b, 1e-4, enforce_norm=True, norm_value=2.0
    )
    rep = handle.quadratic_representation()
    assert hasattr(rep, "constraint_blocks")
    assert any(
        "norm" in getattr(block, "name", "") for block in rep.constraint_blocks
    )


def test_solve_interpret_returns_array(system, detector):
    """The core solver returns a non-negative spectrum array."""
    from bssunfold.core.unfold_interpret import solve_interpret

    A, b = system
    x = solve_interpret(A, b, 1e-4)
    assert isinstance(x, np.ndarray)
    assert x.shape == (detector.n_energy_bins,)
    assert np.all(x >= -1e-8)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1.0


def test_solve_interpret_norm1(system):
    """norm=1 penalty is accepted."""
    from bssunfold.core.unfold_interpret import solve_interpret

    A, b = system
    x = solve_interpret(A, b, 1e-4, norm=1)
    assert x.shape == (A.shape[1],)


def test_solve_interpret_import_error(system):
    """Missing pyoptexplain raises a helpful ImportError."""
    from bssunfold.core.unfold_interpret import solve_interpret

    A, b = system
    _reset_pyopt_cache()

    try:
        with block_import("pyoptexplain"):
            with pytest.raises(ImportError, match="pyoptexplain"):
                solve_interpret(A, b, 1e-4)
    finally:
        _reset_pyopt_cache()


def test_interpret_qp_full(system, detector):
    """interpret_qp returns a complete InterpretationResult."""
    from bssunfold.core.unfold_interpret import interpret_qp

    A, b = system
    ir = interpret_qp(
        A,
        b,
        1e-4,
        E_MeV=detector.E_MeV,
        detector_names=list(detector.detector_names[:6]),
    )
    assert ir.status == "optimal"
    assert ir.spectrum.shape == (detector.n_energy_bins,)
    assert np.all(ir.spectrum >= -1e-8)
    assert isinstance(ir.report, str)
    assert len(ir.report) > 200
    assert isinstance(ir.metrics, dict)
    assert isinstance(ir.tables, dict)
    assert "energy" in ir.metrics
    assert "spectrum" in ir.metrics
    assert ir.metrics["status"] == "optimal"


def test_interpret_qp_metric_blocks(system, detector):
    """The diagnostic sub-blocks are populated."""
    from bssunfold.core.unfold_interpret import interpret_qp

    A, b = system
    ir = interpret_qp(
        A,
        b,
        1e-4,
        E_MeV=detector.E_MeV,
        detector_names=list(detector.detector_names[:6]),
    )
    metrics = ir.metrics
    assert len(metrics["active_groups"]) > 0
    assert len(metrics["zero_groups"]) > 0
    assert len(metrics["regularization_sweep"]) > 0
    assert len(metrics["nonnegativity_relaxation"]) > 0
    assert len(metrics["detector_sensitivity"]) > 0
    assert metrics["robustness"]["case_count"] > 0
    assert len(metrics["scenarios"]) > 0
    assert set(metrics["scenarios"][0]) >= {"scenario", "status"}
    assert set(metrics["detectors"][0]) >= {
        "detector",
        "reading",
        "effective",
        "residual",
    }
    assert metrics["success"] is True


def test_interpret_qp_enforce_norm(system, detector):
    """enforce_norm adds norm-shift scenarios and a relaxation curve."""
    from bssunfold.core.unfold_interpret import interpret_qp

    A, b = system
    ir = interpret_qp(
        A,
        b,
        1e-4,
        enforce_norm=True,
        norm_value=1.0,
        E_MeV=detector.E_MeV,
        detector_names=list(detector.detector_names[:6]),
    )
    scenarios = [s["scenario"] for s in ir.metrics["scenarios"]]
    assert any("norm_target" in s for s in scenarios)
    assert "norm_relaxation" in ir.tables
    assert ir.metrics["norm_dual"] is not None


def test_interpret_qp_flags_off(system, detector):
    """All diagnostic runs can be disabled for a fast solve."""
    from bssunfold.core.unfold_interpret import interpret_qp

    A, b = system
    ir = interpret_qp(
        A,
        b,
        1e-4,
        run_robustness=False,
        run_scenarios=False,
        run_detector_sensitivity=False,
        run_regularization_sweep=False,
        run_nonnegativity_relaxation=False,
        E_MeV=detector.E_MeV,
        detector_names=list(detector.detector_names[:6]),
    )
    assert ir.status == "optimal"
    assert ir.metrics.get("robustness", {}).get("case_count", 0) == 0
    assert ir.metrics["detector_sensitivity"] == []
    assert ir.metrics["regularization_sweep"] == []
    assert ir.metrics["nonnegativity_relaxation"] == []
    assert ir.metrics.get("scenarios", []) == []


def test_interpret_qp_to_dict(system, detector):
    """InterpretationResult.to_dict is JSON-friendly."""
    from bssunfold.core.unfold_interpret import interpret_qp

    A, b = system
    ir = interpret_qp(
        A,
        b,
        1e-4,
        run_robustness=False,
        run_scenarios=False,
        run_detector_sensitivity=False,
        run_regularization_sweep=False,
        run_nonnegativity_relaxation=False,
        E_MeV=detector.E_MeV,
        detector_names=list(detector.detector_names[:6]),
    )
    d = ir.to_dict()
    assert isinstance(d["spectrum"], list)
    assert isinstance(d["report"], str)
    assert isinstance(d["interpretation_metrics"], dict)


def test_unfold_interpret_basic(detector, small_readings):
    """Detector.unfold_interpret returns a standardized result plus report."""
    result = detector.unfold_interpret(
        small_readings,
        save_result=False,
        interpret_options={"run_scenarios": False},
    )
    assert isinstance(result, dict)
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert "report" in result
    assert len(result["report"]) > 100
    assert "interpretation_metrics" in result
    assert isinstance(result["interpretation_metrics"], dict)
    assert result["interpretation_metrics"]["status"] == "optimal"


def test_unfold_interpret_cosine_requires_initial(detector, small_readings):
    """The cosine regularization method requires an initial spectrum."""
    with pytest.raises(ValueError, match="initial_spectrum"):
        detector.unfold_interpret(
            small_readings, regularization_method="cosine", save_result=False
        )


def _fast_interpret_options():
    return {
        "run_robustness": False,
        "run_scenarios": False,
        "run_detector_sensitivity": False,
        "run_regularization_sweep": False,
        "run_nonnegativity_relaxation": False,
    }


def test_unfold_interpret_cosine(detector, small_readings):
    """The cosine method selects alpha automatically from the initial guess."""
    result = detector.unfold_interpret(
        small_readings,
        regularization_method="cosine",
        initial_spectrum=np.ones(detector.n_energy_bins),
        save_result=False,
        interpret_options=_fast_interpret_options(),
    )
    assert result["interpretation_metrics"]["status"] == "optimal"


def test_unfold_interpret_auto_lcurve(detector, small_readings):
    """Automatic L-curve regularization selection is honored."""
    result = detector.unfold_interpret(
        small_readings,
        regularization_method="lcurve",
        save_result=False,
        interpret_options=_fast_interpret_options(),
    )
    assert result["interpretation_metrics"]["status"] == "optimal"
    assert "selected_regularization" in result
    assert result["selected_regularization"] > 0


def test_unfold_interpret_failed_selection(detector, small_readings):
    """A failing auto-selection raises a ValueError."""
    with patch(
        "bssunfold.core.unfold_interpret.select_regularization_parameter",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(ValueError, match="Regularization selection failed"):
            detector.unfold_interpret(
                small_readings,
                regularization_method="lcurve",
                save_result=False,
            )


def test_unfold_interpret_tolerance_full_lanl():
    """tolerance is forwarded to the solver on the full LANL detector set.

    The strict default (1e-8) makes pyoptexplain's backend report
    ``iteration_limit`` on the 11-sphere RF_LANL problem; relaxing the
    tolerance must yield an optimal solution instead of a RuntimeError.
    """
    import pandas as pd

    from bssunfold import RF_LANL

    det = Detector(RF_LANL)
    ref = pd.read_csv(
        "tests/MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
    )
    readings = det.get_effective_readings_for_spectra(
        ref[["E_MeV", "ISO_ref_Cf252"]]
    )
    result = det.unfold_interpret(
        readings,
        tolerance=1e-5,
        save_result=False,
        interpret_options=_fast_interpret_options(),
    )
    assert result["interpretation_metrics"]["status"] == "optimal"
    assert result["interpretation_metrics"]["success"] is True


def test_unfold_interpret_import_error(detector, small_readings):
    """Missing pyoptexplain raises a helpful ImportError in the wrapper."""
    _reset_pyopt_cache()

    try:
        with block_import("pyoptexplain"):
            with pytest.raises(ImportError, match="pyoptexplain"):
                detector.unfold_interpret(small_readings, save_result=False)
    finally:
        _reset_pyopt_cache()


def test_unfold_interpret_save_result(detector, small_readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_interpret(
        small_readings,
        save_result=True,
        interpret_options={
            "run_robustness": False,
            "run_scenarios": False,
            "run_detector_sensitivity": False,
            "run_regularization_sweep": False,
            "run_nonnegativity_relaxation": False,
        },
    )
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "interpret"


def test_interpret_result(detector, small_readings):
    """interpret_result runs the analysis directly on the readings."""
    result = detector.interpret_result(small_readings)
    for key in ("report", "metrics", "tables", "spectrum"):
        assert key in result
    assert isinstance(result["report"], str)
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["tables"], dict)
    assert result["spectrum"].shape == (detector.n_energy_bins,)
