"""Tests for the composite (ensemble) unfolding module."""

import numpy as np
import pandas as pd
import pytest

from bssunfold import RF_PTB, Detector
from bssunfold.core.unfold_composite import (
    DEFAULT_BIN_METHODS,
    DEFAULT_ENSEMBLE_WEIGHTS,
    METHOD_DISPATCH,
    classify_spectrum_by_hardness,
    compute_spectrum_features,
    unfold_composite,
)


def _make_detector_and_readings():
    det = Detector(pd.DataFrame(RF_PTB))
    true = np.exp(-det.E_MeV * 3) + 0.2 * np.exp(-((det.E_MeV - 1e-1) ** 2) / 1e-3)
    readings = {
        d: float(np.dot(det.sensitivities[d], true)) for d in det.detector_names
    }
    return det, readings


def test_compute_spectrum_features():
    det, _ = _make_detector_and_readings()
    spec = np.exp(-det.E_MeV * 3)
    features = compute_spectrum_features(spec, det.E_MeV)
    assert "hardness_ratio" in features
    assert features["hardness_ratio"] > 0
    assert "entropy" in features
    assert np.isfinite(features["entropy"])


@pytest.mark.parametrize(
    "hr,expected",
    [
        (0.05, "very_soft"),
        (0.2, "soft"),
        (0.4, "intermediate"),
        (0.7, "hard"),
        (2.0, "very_hard"),
    ],
)
def test_classify_spectrum_by_hardness(hr, expected):
    assert classify_spectrum_by_hardness({"hardness_ratio": hr}) == expected


def test_default_bin_methods_and_dispatch_consistent():
    for bin_methods in DEFAULT_BIN_METHODS.values():
        for name in bin_methods:
            assert name in METHOD_DISPATCH
            assert name in DEFAULT_ENSEMBLE_WEIGHTS


def test_unfold_composite_light_methods():
    det, readings = _make_detector_and_readings()
    result = unfold_composite(
        det,
        readings,
        method_names=["tsvd", "mlem", "cvxpy", "qpsolvers", "bayes_spline"],
        timeout_per_method=30.0,
    )
    assert result["status"] == "OK"
    assert result["spectrum"] is not None
    assert np.all(np.isfinite(result["spectrum"]))
    assert len(result["successful_methods"]) >= 1
    assert 0.0 <= result["consistency"] <= 1.0


def test_unfold_composite_classification_pool():
    det, readings = _make_detector_and_readings()
    # Very soft spectrum -> selects from very_soft pool
    soft_spec = np.exp(-det.E_MeV * 50)
    result = unfold_composite(det, readings, spectrum=soft_spec, n_methods=3)
    assert result["status"] == "OK"
    assert result["spectrum"] is not None


def test_unfold_composite_all_methods_fail():
    det, readings = _make_detector_and_readings()
    # None of these map to real methods -> all skipped
    result = unfold_composite(
        det,
        readings,
        method_names=["not_a_method", "also_fake"],
        timeout_per_method=5.0,
    )
    assert result["status"] == "ERROR"
    assert result["spectrum"] is None
    assert result["successful_methods"] == []


def test_unfold_composite_weights_present():
    det, readings = _make_detector_and_readings()
    result = unfold_composite(
        det, readings, method_names=["tsvd", "mlem"], timeout_per_method=30.0
    )
    assert set(result["weights"].keys()) == set(result["successful_methods"])
