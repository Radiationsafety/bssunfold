"""Tests for the cascade (sequential) unfolding module."""

import numpy as np
import pandas as pd

from bssunfold import Detector, RF_PTB
from bssunfold.core.unfold_cascade import (
    CascadeStage,
    CascadeResult,
    compute_quality_metrics,
    create_default_cascade,
    unfold_cascade,
    unfold_adaptive_cascade,
)


def _make_detector_and_readings():
    det = Detector(pd.DataFrame(RF_PTB))
    true = np.exp(-det.E_MeV * 3) + 0.2 * np.exp(
        -((det.E_MeV - 1e-1) ** 2) / 1e-3
    )
    readings = {
        d: float(np.dot(det.sensitivities[d], true)) for d in det.detector_names
    }
    return det, readings


def test_cascadestage_defaults():
    stage = CascadeStage(method="tsvd")
    assert stage.method == "tsvd"
    assert stage.use_as_initial is True
    assert stage.use_as_prior is False
    assert stage.timeout == 60.0
    assert stage.quality_threshold is None


def test_create_default_cascade_kinds():
    for kind in ("general", "soft", "hard", "fast_refinement"):
        stages = create_default_cascade(kind)
        assert isinstance(stages, list)
        assert all(isinstance(s, CascadeStage) for s in stages)
        assert len(stages) >= 2


def test_compute_quality_metrics_keys():
    det, readings = _make_detector_and_readings()
    A = np.array([det.sensitivities[d] for d in det.detector_names])
    spec = np.abs(np.random.default_rng(1).normal(1.0, 0.2, det.n_energy_bins)) + 1e-3
    rec = A @ spec
    measured = np.array([readings[d] for d in det.detector_names])
    metrics = compute_quality_metrics(spec, rec, measured, det.E_MeV)
    for key in (
        "chi_square",
        "smoothness",
        "flux_error",
        "negativity_count",
        "hardness_ratio",
        "peak_count",
        "overall_quality",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_unfold_cascade_general_returns_spectrum():
    det, readings = _make_detector_and_readings()
    result = unfold_cascade(
        det, readings, cascade_stages=create_default_cascade("general"), verbose=False
    )
    assert result["status"] == "OK"
    assert result["spectrum"] is not None
    assert np.all(np.isfinite(result["spectrum"]))
    assert len(result["method_sequence"]) >= 1
    assert "convergence_history" in result


def test_unfold_cascade_all_kinds_run():
    det, readings = _make_detector_and_readings()
    for kind in ("general", "soft", "hard", "fast_refinement"):
        result = unfold_cascade(
            det, readings, cascade_stages=create_default_cascade(kind), verbose=False
        )
        assert result["status"] == "OK"
        assert result["spectrum"] is not None
        assert np.all(np.isfinite(result["spectrum"]))


def test_unfold_adaptive_cascade_runs():
    det, readings = _make_detector_and_readings()
    result = unfold_adaptive_cascade(det, readings, max_stages=3, verbose=False)
    assert result["status"] == "OK"
    assert result["spectrum"] is not None
    assert np.all(np.isfinite(result["spectrum"]))


def test_unfold_cascade_unknown_method_skipped():
    det, readings = _make_detector_and_readings()
    stages = [CascadeStage(method="does_not_exist")]
    result = unfold_cascade(det, readings, cascade_stages=stages, verbose=False)
    assert result["status"] == "ERROR"
    assert result["spectrum"] is None


def test_cascaderesult_is_exported_dataclass():
    assert CascadeResult is not None


def test_build_coarse_detector_shape_and_readings():
    from bssunfold.core._multires import (
        build_coarse_detector,
        prolongate_spectrum,
        _coarsen_columns,
    )

    det, readings = _make_detector_and_readings()
    coarse = build_coarse_detector(det, 12)
    assert coarse.n_energy_bins == 12
    assert coarse.detector_names == det.detector_names

    # The coarse detector's response matrix must equal the column-sum
    # coarsening of the fine response matrix (the premise of a coarse
    # pre-solve whose prolongated solution seeds the fine grid).
    A_fine = np.array([det.sensitivities[d] for d in det.detector_names])
    A_coarse = np.array([coarse.sensitivities[d] for d in coarse.detector_names])
    assert np.allclose(A_coarse, _coarsen_columns(A_fine, 12), atol=1e-12)

    # Prolongation preserves total fluence (used as a fine-grid initial guess).
    rng = np.random.default_rng(3)
    x_coarse = np.abs(rng.normal(1.0, 0.3, 12)) + 0.1
    fine_back = prolongate_spectrum(x_coarse, det.n_energy_bins)
    assert fine_back.shape == (det.n_energy_bins,)
    assert np.isclose(np.sum(fine_back), np.sum(x_coarse), atol=1e-9)


def test_prolongate_preserves_fluence():
    from bssunfold.core._multires import prolongate_spectrum

    n = 60
    x_coarse = np.abs(np.random.default_rng(7).normal(1.0, 0.3, 8)) + 0.1
    fine = prolongate_spectrum(x_coarse, n)
    assert fine.shape == (n,)
    # Each fine bin holds a fraction of its coarse total -> sum preserved.
    assert np.isclose(np.sum(fine), np.sum(x_coarse), atol=1e-9)


def test_unfold_cascade_multiresolution_runs():
    det, readings = _make_detector_and_readings()
    result = unfold_cascade(
        det,
        readings,
        cascade_stages=create_default_cascade("general"),
        verbose=False,
        multi_resolution=True,
    )
    assert result["status"] == "OK"
    assert result["spectrum"] is not None
    assert len(result["spectrum"]) == det.n_energy_bins
    assert np.all(np.isfinite(result["spectrum"]))


def test_unfold_cascade_explicit_coarse_stage():
    det, readings = _make_detector_and_readings()
    stages = [
        CascadeStage(method="tsvd", use_as_initial=False, coarse=True, coarse_bins=10),
        CascadeStage(method="landweber", params={"max_iterations": 30}, use_as_initial=True),
    ]
    result = unfold_cascade(det, readings, cascade_stages=stages, verbose=False)
    assert result["status"] in ("OK", "DONE")
    assert result["spectrum"] is not None
    assert len(result["spectrum"]) == det.n_energy_bins

