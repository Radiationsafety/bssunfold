"""Tests for conditioning diagnostics (core/conditioning.py).

Covers:
- conditioning_diagnostics: known condition numbers, effective rank,
  rank deficiency, ill-conditioning flags, error handling.
- conditioning_summary rendering.
- Integration into the standardized unfolding result (all methods get
  a "conditioning" key; ill-conditioned problems raise a warning).
- Detector.conditioning_report: detector subsets, energy cutoffs,
  validation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from bssunfold import Detector
from bssunfold.core import (
    ILL_THRESHOLD,
    conditioning_diagnostics,
    conditioning_summary,
)
from bssunfold.core.conditioning import _attach_conditioning


class TestConditioningDiagnostics:
    """Unit tests for conditioning_diagnostics."""

    def test_identity_has_cond_one(self):
        diag = conditioning_diagnostics(np.eye(5))
        assert diag["cond_2"] == 1.0
        assert diag["cond_normal"] == 1.0
        assert diag["effective_rank"] == 5
        assert not diag["rank_deficient"]
        assert not diag["ill_conditioned"]
        assert diag["shape"] == (5, 5)

    def test_diagonal_known_condition(self):
        D = np.diag([1.0, 1e-6])
        diag = conditioning_diagnostics(D)
        assert np.isclose(diag["cond_2"], 1e6)
        assert np.isclose(diag["cond_normal"], 1e12)
        # 1e-6 > rcond * 1.0, so both singular values count toward the rank.
        assert diag["effective_rank"] == 2

    def test_ill_conditioned_flag(self):
        D = np.diag([1.0, 1e-11])
        diag = conditioning_diagnostics(D, ill_threshold=1e10)
        assert diag["cond_2"] > 1e10
        assert diag["ill_conditioned"]

    def test_custom_threshold(self):
        D = np.diag([1.0, 1e-3])
        diag = conditioning_diagnostics(D, ill_threshold=1e2)
        assert diag["ill_conditioned"]
        diag2 = conditioning_diagnostics(D, ill_threshold=1e6)
        assert not diag2["ill_conditioned"]

    def test_rank_deficient_matrix(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        diag = conditioning_diagnostics(A)
        assert diag["rank_deficient"]
        assert diag["ill_conditioned"]
        assert diag["effective_rank"] == 1
        assert diag["cond_2"] > 1e15

    def test_zero_matrix(self):
        diag = conditioning_diagnostics(np.zeros((3, 3)))
        assert np.isinf(diag["cond_2"])
        assert np.isinf(diag["cond_normal"])
        assert diag["effective_rank"] == 0
        assert diag["rank_deficient"]
        assert diag["ill_conditioned"]

    def test_rectangular_matrix(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 10))
        diag = conditioning_diagnostics(A)
        assert diag["shape"] == (4, 10)
        assert diag["effective_rank"] <= 4
        assert not diag["rank_deficient"]

    def test_repeated_columns_rank_deficient(self):
        A = np.column_stack([np.ones(6), np.ones(6), np.linspace(0, 1, 6)])
        diag = conditioning_diagnostics(A)
        assert diag["rank_deficient"]
        assert diag["ill_conditioned"]

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            conditioning_diagnostics(np.ones(3))

    def test_singular_values_sorted_descending(self):
        diag = conditioning_diagnostics(np.diag([3.0, 1.0, 2.0]))
        s = diag["singular_values"]
        assert np.all(np.diff(s) <= 0)

    def test_default_ill_threshold_constant(self):
        assert ILL_THRESHOLD == 1e10


class TestConditioningSummary:
    """Human-readable summary rendering."""

    def test_summary_good_matrix(self):
        summary = conditioning_summary(conditioning_diagnostics(np.eye(3)))
        assert "cond_2(A)=1.000e+00" in summary
        assert "effective_rank=3/3" in summary
        assert "ill-conditioned" not in summary

    def test_summary_flags_bad_matrix(self):
        summary = conditioning_summary(
            conditioning_diagnostics(np.array([[1.0, 2.0], [2.0, 4.0]]))
        )
        assert "rank-deficient" in summary
        assert "ill-conditioned" in summary

    def test_summary_infinite_cond(self):
        summary = conditioning_summary(conditioning_diagnostics(np.zeros((2, 2))))
        assert "cond_2(A)=inf" in summary


class TestAttachConditioning:
    """Internal hook attaching diagnostics to result dicts."""

    def test_attaches_key(self):
        out = _attach_conditioning({}, np.eye(2), warn=False)
        assert "conditioning" in out
        assert out["conditioning"]["cond_2"] == 1.0

    def test_warns_on_ill_conditioned(self):
        bad = np.diag([1.0, 1e-13])
        with pytest.warns(UserWarning, match="ill-conditioned"):
            out = _attach_conditioning({}, bad)
        assert out["conditioning"]["ill_conditioned"]

    def test_no_warning_on_good_matrix(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = _attach_conditioning({}, np.eye(4))
        assert not out["conditioning"]["ill_conditioned"]


class TestDetectorConditioningReport:
    """Detector.conditioning_report public API."""

    def test_report_default(self, detector):
        rep = detector.conditioning_report()
        assert rep["shape"] == (len(detector.detector_names), detector.n_energy_bins)
        assert rep["detector_names"] == detector.detector_names
        assert rep["n_energy_bins"] == detector.n_energy_bins
        assert rep["cond_2"] > 0
        assert np.isfinite(rep["cond_2"])

    def test_report_subset(self, detector):
        subset = detector.detector_names[:4]
        rep = detector.conditioning_report(detector_names=subset)
        assert rep["detector_names"] == subset
        assert rep["shape"] == (4, detector.n_energy_bins)

    def test_report_max_energy_cutoff(self, detector):
        rep = detector.conditioning_report(max_neutron_energy=20.0)
        n_active = int((detector.E_MeV <= 20.0).sum())
        assert rep["n_energy_bins"] == n_active
        assert rep["shape"] == (len(detector.detector_names), n_active)

    def test_report_unknown_detector_raises(self, detector):
        with pytest.raises(ValueError, match="Unknown detector names"):
            detector.conditioning_report(detector_names=["no_such_sphere"])

    def test_report_empty_list_raises(self, detector):
        with pytest.raises(ValueError, match="non-empty"):
            detector.conditioning_report(detector_names=[])

    def test_report_conditioning_depends_on_detector_set(self, detector):
        full = detector.conditioning_report()
        subset = detector.conditioning_report(
            detector_names=detector.detector_names[: len(detector.detector_names) - 1]
        )
        # Dropping a detector changes the response matrix, hence its spectrum.
        assert full["cond_2"] != subset["cond_2"]


class TestConditioningIntegration:
    """Conditioning is attached to unfolding results for all methods."""

    @pytest.fixture
    def readings(self, detector):
        return {name: 100.0 for name in detector.detector_names}

    @pytest.mark.parametrize(
        "method_name",
        ["unfold_landweber", "unfold_kaczmarz", "unfold_mlem"],
    )
    def test_result_contains_conditioning(self, detector, readings, method_name):
        result = getattr(detector, method_name)(readings, max_iterations=20)
        assert "conditioning" in result
        diag = result["conditioning"]
        assert diag["cond_2"] > 0
        assert diag["shape"][1] == detector.n_energy_bins
        assert isinstance(diag["ill_conditioned"], bool)

    def test_ill_conditioned_detector_warns(self):
        # Two identical sensitivity columns -> rank-deficient response matrix.
        df = pd.DataFrame(
            {
                "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6],
                "sphere_a": [0.1, 0.2, 0.3, 0.4],
                "sphere_b": [0.1, 0.2, 0.3, 0.4],
            }
        )
        det = Detector(df)
        readings = {"sphere_a": 100.0, "sphere_b": 100.0}
        with pytest.warns(UserWarning, match="ill-conditioned"):
            result = det.unfold_landweber(readings, max_iterations=10)
        assert result["conditioning"]["rank_deficient"]

    def test_conditioning_not_recomputed_in_montecarlo(self, detector, readings):
        # One conditioning SVD describes the whole call, MC samples do not
        # add extra keys or change the diagnostics structure.
        result = detector.unfold_landweber(
            readings, max_iterations=10, calculate_errors=True, n_montecarlo=3
        )
        assert "conditioning" in result
        assert "spectrum_uncert_mean" in result
