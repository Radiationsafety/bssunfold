"""Tests for interval analysis unfolding method."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.bssunfold import RF_GSF, Detector
from src.bssunfold.core.unfold_interval import (
    interval_compatibility_report,
    interval_sev_variativity,
    solve_interval,
    solve_interval_center,
    solve_interval_intvalpy,
    solve_interval_matrix,
    solve_interval_pia,
    solve_interval_posterior,
    solve_interval_regularization,
    solve_interval_tol,
)


@pytest.fixture
def detector():
    df = pd.DataFrame.from_dict(RF_GSF, orient="columns")
    return Detector(df)


@pytest.fixture
def readings():
    return {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}


class TestSolveInterval:
    def test_basic(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max = solve_interval(A, b_lo, b_hi)
        assert x_min.shape == (2,)
        assert x_max.shape == (2,)
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)

    def test_with_tv(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min_no_tv, x_max_no_tv = solve_interval(A, b_lo, b_hi, tv_bound=None)
        x_min_tv, x_max_tv = solve_interval(A, b_lo, b_hi, tv_bound=1.0)
        assert np.all(x_min_tv <= x_max_tv)
        assert np.all(x_min_tv >= 0)

    def test_uncertainty_scaling(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        x_min_narrow, x_max_narrow = solve_interval(A, b - 0.1, b + 0.1)
        x_min_wide, x_max_wide = solve_interval(A, b - 0.5, b + 0.5)
        width_narrow = np.sum(x_max_narrow - x_min_narrow)
        width_wide = np.sum(x_max_wide - x_min_wide)
        assert width_wide >= width_narrow

    def test_invalid_bounds(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([1.5, 1.5])
        b_hi = np.array([0.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be <= b_hi"):
            solve_interval(A, b_lo, b_hi)

    def test_negative_blo(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([-0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be non-negative"):
            solve_interval(A, b_lo, b_hi)


class TestUnfoldInterval:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert "spectrum" in result
        assert result["method"] == "IntervalLP"

    def test_interval_ordering(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_nonnegative(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] >= 0)

    def test_midpoint_in_interval(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert np.all(result["spectrum"] >= result["spectrum_lower"])
        assert np.all(result["spectrum"] <= result["spectrum_upper"])

    def test_uncertainty_scaling(self, detector, readings):
        result_narrow = detector.unfold_interval(readings, noise_level=0.05)
        result_wide = detector.unfold_interval(readings, noise_level=0.2)
        width_narrow = np.sum(
            result_narrow["spectrum_upper"] - result_narrow["spectrum_lower"]
        )
        width_wide = np.sum(
            result_wide["spectrum_upper"] - result_wide["spectrum_lower"]
        )
        assert width_wide >= width_narrow

    def test_tv_regularization(self, detector, readings):
        result_no_tv = detector.unfold_interval(
            readings, noise_level=0.1, tv_bound=None
        )
        result_tv = detector.unfold_interval(readings, noise_level=0.1, tv_bound=1.0)
        assert "tv_bound" in result_no_tv
        assert "tv_bound" in result_tv
        assert result_no_tv["tv_bound"] is None
        assert result_tv["tv_bound"] == 1.0

    def test_reading_uncertainties(self, detector, readings):
        uncertainties = {"3in": 0.01, "5in": 0.02, "10in": 0.02, "18in": 0.005}
        result = detector.unfold_interval(
            readings, reading_uncertainties=uncertainties
        )
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result

    def test_max_neutron_energy(self, detector, readings):
        result_full = detector.unfold_interval(readings, noise_level=0.1)
        result_cut = detector.unfold_interval(
            readings, noise_level=0.1, max_neutron_energy=1.0
        )
        assert len(result_cut["spectrum"]) == len(result_full["spectrum"])
        assert np.all(result_cut["spectrum_lower"] <= result_cut["spectrum_upper"])

    def test_zero_readings(self, detector):
        readings = {"3in": 0.0, "5in": 0.0, "10in": 0.0, "18in": 0.0}
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] >= 0)
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_doserates_present(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert "doserates" in result
        assert len(result["doserates"]) > 0

    def test_effective_readings_present(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert "effective_readings" in result
        assert len(result["effective_readings"]) > 0

    def test_compatibility_report(self, detector, readings):
        result = detector.unfold_interval(
            readings, noise_level=0.1, compatibility_report=True
        )
        assert "compatibility" in result
        rep = result["compatibility"]
        assert "compatible" in rep
        assert "pairwise_jaccard" in rep
        assert "clique_number" in rep
        assert "outlier_candidates" in rep

    def test_compatibility_report_absent_by_default(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert "compatibility" not in result


class TestSolveIntervalTol:
    def test_basic(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max, info = solve_interval_tol(A, b_lo, b_hi)
        assert x_min.shape == (2,)
        assert x_max.shape == (2,)
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)
        assert "tol_max" in info
        assert "x_pseudo" in info
        assert "converged" in info

    def test_tol_functional_properties(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b_lo = np.array([0.5, 0.5])
        b_hi = np.array([1.5, 1.5])
        x_min, x_max, info = solve_interval_tol(A, b_lo, b_hi)
        assert "tol_max" in info
        assert isinstance(info["tol_max"], float)
        assert np.all(x_min <= x_max)

    def test_uncertainty_scaling(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        x_min_narrow, x_max_narrow, _ = solve_interval_tol(A, b - 0.1, b + 0.1)
        x_min_wide, x_max_wide, _ = solve_interval_tol(A, b - 0.5, b + 0.5)
        width_narrow = np.sum(x_max_narrow - x_min_narrow)
        width_wide = np.sum(x_max_wide - x_min_wide)
        assert width_wide >= width_narrow

    def test_invalid_bounds(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([1.5, 1.5])
        b_hi = np.array([0.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be <= b_hi"):
            solve_interval_tol(A, b_lo, b_hi)

    def test_negative_blo(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([-0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be non-negative"):
            solve_interval_tol(A, b_lo, b_hi)


class TestSolveIntervalPosterior:
    def test_basic(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max, info = solve_interval_posterior(A, b_lo, b_hi)
        assert x_min.shape == (2,)
        assert x_max.shape == (2,)
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)
        assert "n_samples" in info
        assert "sensitivity" in info
        assert "residuals" in info

    def test_uncertainty_scaling(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        x_min_narrow, x_max_narrow, _ = solve_interval_posterior(A, b - 0.1, b + 0.1)
        x_min_wide, x_max_wide, _ = solve_interval_posterior(A, b - 0.5, b + 0.5)
        width_narrow = np.sum(x_max_narrow - x_min_narrow)
        width_wide = np.sum(x_max_wide - x_min_wide)
        assert width_wide >= width_narrow

    def test_invalid_bounds(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([1.5, 1.5])
        b_hi = np.array([0.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be <= b_hi"):
            solve_interval_posterior(A, b_lo, b_hi)

    def test_negative_blo(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([-0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be non-negative"):
            solve_interval_posterior(A, b_lo, b_hi)


class TestDetectorUnfoldIntervalTol:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert "spectrum" in result
        assert result["method"] == "IntervalTol"

    def test_interval_ordering(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_nonnegative(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] >= 0)

    def test_midpoint_in_interval(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert np.all(result["spectrum"] >= result["spectrum_lower"])
        assert np.all(result["spectrum"] <= result["spectrum_upper"])

    def test_tol_metadata(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert "tol_max" in result
        assert "x_pseudo" in result
        assert "converged" in result
        assert "n_iter" in result

    def test_uncertainty_scaling(self, detector, readings):
        result_narrow = detector.unfold_interval_tol(readings, noise_level=0.05)
        result_wide = detector.unfold_interval_tol(readings, noise_level=0.2)
        width_narrow = np.sum(
            result_narrow["spectrum_upper"] - result_narrow["spectrum_lower"]
        )
        width_wide = np.sum(
            result_wide["spectrum_upper"] - result_wide["spectrum_lower"]
        )
        assert width_wide >= width_narrow

    def test_doserates_present(self, detector, readings):
        result = detector.unfold_interval_tol(readings, noise_level=0.1)
        assert "doserates" in result
        assert len(result["doserates"]) > 0


class TestDetectorUnfoldIntervalPosterior:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert "spectrum" in result
        assert result["method"] == "IntervalPosterior"

    def test_interval_ordering(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_nonnegative(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] >= 0)

    def test_midpoint_in_interval(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert np.all(result["spectrum"] >= result["spectrum_lower"])
        assert np.all(result["spectrum"] <= result["spectrum_upper"])

    def test_posterior_metadata(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert "n_samples" in result
        assert "sensitivity" in result
        assert "residuals" in result

    def test_uncertainty_scaling(self, detector, readings):
        result_narrow = detector.unfold_interval_posterior(readings, noise_level=0.05)
        result_wide = detector.unfold_interval_posterior(readings, noise_level=0.2)
        width_narrow = np.sum(
            result_narrow["spectrum_upper"] - result_narrow["spectrum_lower"]
        )
        width_wide = np.sum(
            result_wide["spectrum_upper"] - result_wide["spectrum_lower"]
        )
        assert width_wide >= width_narrow

    def test_doserates_present(self, detector, readings):
        result = detector.unfold_interval_posterior(readings, noise_level=0.1)
        assert "doserates" in result
        assert len(result["doserates"]) > 0


class TestSolveIntervalIntvalpy:
    def test_basic(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max, info = solve_interval_intvalpy(A, b_lo, b_hi)
        assert x_min.shape == (2,)
        assert x_max.shape == (2,)
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)
        assert "tol_max" in info
        assert "x_pseudo" in info
        assert "method" in info

    def test_rohn_method(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max, info = solve_interval_intvalpy(A, b_lo, b_hi, method="rohn")
        assert info["method"] == "rohn"
        assert np.all(x_min <= x_max)

    def test_shary_method(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min, x_max, info = solve_interval_intvalpy(A, b_lo, b_hi, method="shary")
        assert info["method"] == "shary"
        assert np.all(x_min <= x_max)

    def test_uncertainty_scaling(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        x_min_narrow, x_max_narrow, _ = solve_interval_intvalpy(A, b - 0.1, b + 0.1)
        x_min_wide, x_max_wide, _ = solve_interval_intvalpy(A, b - 0.5, b + 0.5)
        width_narrow = np.sum(x_max_narrow - x_min_narrow)
        width_wide = np.sum(x_max_wide - x_min_wide)
        assert width_wide >= width_narrow

    def test_invalid_bounds(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([1.5, 1.5])
        b_hi = np.array([0.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be <= b_hi"):
            solve_interval_intvalpy(A, b_lo, b_hi)

    def test_negative_blo(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([-0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="b_lo must be non-negative"):
            solve_interval_intvalpy(A, b_lo, b_hi)


class TestDetectorUnfoldIntervalIntvalpy:
    def test_basic(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert "spectrum" in result
        assert result["method"] == "IntervalIntvalpy"

    def test_interval_ordering(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_nonnegative(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert np.all(result["spectrum_lower"] >= 0)

    def test_midpoint_in_interval(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert np.all(result["spectrum"] >= result["spectrum_lower"])
        assert np.all(result["spectrum"] <= result["spectrum_upper"])

    def test_intvalpy_metadata(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert "tol_max" in result
        assert "x_pseudo" in result
        assert "n_iter" in result
        assert "n_calls" in result
        assert "exit_code" in result
        assert "intvalpy_method" in result

    def test_uncertainty_scaling(self, detector, readings):
        pytest.importorskip("intvalpy")
        result_narrow = detector.unfold_interval_intvalpy(readings, noise_level=0.05)
        result_wide = detector.unfold_interval_intvalpy(readings, noise_level=0.2)
        width_narrow = np.sum(
            result_narrow["spectrum_upper"] - result_narrow["spectrum_lower"]
        )
        width_wide = np.sum(
            result_wide["spectrum_upper"] - result_wide["spectrum_lower"]
        )
        assert width_wide >= width_narrow

    def test_doserates_present(self, detector, readings):
        pytest.importorskip("intvalpy")
        result = detector.unfold_interval_intvalpy(readings, noise_level=0.1)
        assert "doserates" in result
        assert len(result["doserates"]) > 0


class TestSolveIntervalIntvalpyNormalize:
    def test_normalize(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min_raw, x_max_raw, _ = solve_interval_intvalpy(
            A, b_lo, b_hi, normalize=False
        )
        x_min_norm, x_max_norm, _ = solve_interval_intvalpy(
            A, b_lo, b_hi, normalize=True
        )
        sum_raw = np.sum((x_min_raw + x_max_raw) / 2.0)
        sum_norm = np.sum((x_min_norm + x_max_norm) / 2.0)
        x_mid_raw = (x_min_raw + x_max_raw) / 2.0
        target = np.sum(A @ x_mid_raw)
        assert abs(sum_norm - target) < 1e-6
        assert sum_norm != sum_raw

    def test_regularization(self):
        pytest.importorskip("intvalpy")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min_raw, x_max_raw, _ = solve_interval_intvalpy(
            A, b_lo, b_hi, regularization=None
        )
        x_min_reg, x_max_reg, _ = solve_interval_intvalpy(
            A, b_lo, b_hi, regularization=0.1
        )
        assert np.all(x_min_reg <= x_max_reg)
        assert np.all(x_min_reg >= 0)


class TestSolveIntervalPosteriorNormalize:
    def test_normalize(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_lo = np.array([0.5, 1.5])
        b_hi = np.array([1.5, 2.5])
        x_min_raw, x_max_raw, _ = solve_interval_posterior(
            A, b_lo, b_hi, normalize=False
        )
        x_min_norm, x_max_norm, _ = solve_interval_posterior(
            A, b_lo, b_hi, normalize=True
        )
        sum_raw = np.sum((x_min_raw + x_max_raw) / 2.0)
        sum_norm = np.sum((x_min_norm + x_max_norm) / 2.0)
        target = np.sum(A @ ((x_min_raw + x_max_raw) / 2.0))
        assert abs(sum_norm - target) < 1e-6
        assert sum_norm != sum_raw


class TestTolLpBookExample:
    """Validation against Dagesenov et al. 2024, sec. 4.14-4.16 example."""

    # y = a + b*x at x = 1, 2, 3 with y in [1, 2.5], [2, 3], [1.5, 2];
    # the book's max-compatibility line is y = -0.25x + 2.625 with
    # max Tol = 0.125. The second regressor is sign-flipped so the
    # optimum is (2.625, 0.25) >= 0 under the package's x >= 0 convention.
    A = np.array([[1.0, -1.0], [1.0, -2.0], [1.0, -3.0]])
    b_lo = np.array([1.0, 2.0, 1.5])
    b_hi = np.array([2.5, 3.0, 2.0])

    def test_tol_max_and_pseudo_solution(self):
        _, _, info = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        assert abs(info["tol_max"] - 0.125) < 1e-9
        assert np.allclose(info["x_pseudo"], [2.625, 0.25], atol=1e-6)

    def test_generators_active(self):
        _, _, info = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        assert sorted(info["generators"]) == [0, 1, 2]

    def test_bounds_enclose_pseudo_solution(self):
        x_min, x_max, info = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        assert np.all(x_min <= info["x_pseudo"] + 1e-9)
        assert np.all(info["x_pseudo"] <= x_max + 1e-9)


class TestTolWeightsAndProfile:
    """Weighted recognizing functional (tolsolvty-style) and env profile."""

    A = np.array([[1.0, -1.0], [1.0, -2.0], [1.0, -3.0]])
    b_lo = np.array([1.0, 2.0, 1.5])
    b_hi = np.array([2.5, 3.0, 2.0])

    def test_unit_weights_match_default(self):
        _, _, base = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        _, _, ones = solve_interval_tol(
            self.A, self.b_lo, self.b_hi, weights=np.ones(3)
        )
        assert abs(ones["tol_max"] - base["tol_max"]) < 1e-9
        assert np.allclose(ones["x_pseudo"], base["x_pseudo"], atol=1e-6)

    def test_uniform_weights_scale_tol_max(self):
        _, _, base = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        _, _, dbl = solve_interval_tol(
            self.A, self.b_lo, self.b_hi, weights=2.0 * np.ones(3)
        )
        assert abs(dbl["tol_max"] - 2.0 * base["tol_max"]) < 1e-9
        assert np.allclose(dbl["x_pseudo"], base["x_pseudo"], atol=1e-6)

    def test_nonuniform_weights_reweight_reserve(self):
        _, _, base = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        _, _, wn = solve_interval_tol(
            self.A, self.b_lo, self.b_hi, weights=np.array([1.0, 1.0, 0.1])
        )
        # down-weighting a generator lets the unweighted compatibility
        # reserve improve beyond the unweighted optimum
        assert wn["tol_max"] < base["tol_max"]

    def test_generators_profile_sorted_complete(self):
        _, _, info = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        prof = info["generators_profile"]
        m = self.A.shape[0]
        assert prof.shape == (m, 2)
        assert sorted(int(i) for i in prof[:, 0]) == list(range(m))
        assert np.all(np.diff(prof[:, 1]) >= -1e-12)
        assert abs(prof[0, 1] - info["tol_max"]) < 1e-9

    def test_generators_match_profile_minimum(self):
        _, _, info = solve_interval_tol(self.A, self.b_lo, self.b_hi)
        value_of = {int(row[0]): row[1] for row in info["generators_profile"]}
        scale = max(1.0, abs(info["tol_max"]))
        active = sorted(
            i for i, v in value_of.items()
            if v <= info["tol_max"] + 1e-9 * scale
        )
        assert active == sorted(info["generators"])

    def test_detector_weights_dict(self, detector, readings):
        result = detector.unfold_interval_tol(
            readings, noise_level=0.1, weights={"3in": 2.0}
        )
        assert np.isfinite(result["tol_max"])
        assert "generators_profile" in result
        assert result["generators_profile"].shape[0] == len(readings)

    def test_invalid_weights(self):
        with pytest.raises(ValueError, match="weights"):
            solve_interval_tol(self.A, self.b_lo, self.b_hi, weights=np.zeros(3))
        with pytest.raises(ValueError, match="weights"):
            solve_interval_tol(
                self.A, self.b_lo, self.b_hi, weights=np.ones(2)
            )


class TestSolveIntervalCenter:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_lo = np.array([0.5, 1.5])
    b_hi = np.array([1.5, 2.5])

    def test_compatible_zero_widening(self):
        x_min, x_max, info = solve_interval_center(self.A, self.b_lo, self.b_hi)
        assert info["compatible"]
        assert np.allclose(info["epsilon"], 0.0)
        assert info["incompatibility"] == 0.0
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)

    def test_incompatible_minimal_widening(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b_lo = np.array([1.0, 1.0, 5.0])
        b_hi = np.array([2.0, 2.0, 6.0])
        _, _, info = solve_interval_center(A, b_lo, b_hi)
        assert not info["compatible"]
        assert info["incompatibility"] > 0.0
        # only the offending third reading must be widened
        assert info["epsilon"][0] == 0.0
        assert info["epsilon"][1] == 0.0
        assert info["epsilon"][2] == 1.0
        # repaired intervals make the set feasible again
        b_lo_eff, b_hi_eff = info["b_lo_effective"], info["b_hi_effective"]
        x_min, x_max, _ = solve_interval_center(A, b_lo_eff, b_hi_eff)
        # only x=(2, 2) remains feasible: both bins collapse to 2
        assert np.allclose(x_min, 2.0) and np.allclose(x_max, 2.0)

    def test_bounds_match_solve_interval_when_compatible(self):
        cx_min, cx_max, _ = solve_interval_center(self.A, self.b_lo, self.b_hi)
        x_min, x_max = solve_interval(self.A, self.b_lo, self.b_hi)
        assert np.allclose(cx_min, x_min)
        assert np.allclose(cx_max, x_max)

    def test_weights_validation(self):
        with pytest.raises(ValueError, match="weights"):
            solve_interval_center(
                self.A, self.b_lo, self.b_hi, weights=np.array([1.0, -1.0])
            )

    def test_tv_bound(self):
        x_min, x_max, info = solve_interval_center(
            self.A, self.b_lo, self.b_hi, tv_bound=1.0
        )
        assert np.all(x_min <= x_max)
        assert np.all(x_min >= 0)
        assert info["converged"]

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="b_lo must be <= b_hi"):
            solve_interval_center(
                self.A, np.array([1.5, 1.5]), np.array([0.5, 2.5])
            )


class TestSolveIntervalPia:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_lo = np.array([0.5, 1.5])
    b_hi = np.array([1.5, 2.5])

    def test_compatible_zero_distance(self):
        x_pia, e, info = solve_interval_pia(self.A, self.b_lo, self.b_hi)
        assert np.all(x_pia >= 0)
        assert info["max_distance"] < 1e-9
        assert np.all(e >= 0)

    def test_identity_midpoint_within_corridors(self):
        A = np.eye(3)
        b_lo = np.array([1.0, 2.0, 3.0])
        b_hi = np.array([2.0, 3.0, 4.0])
        x_pia, e, info = solve_interval_pia(A, b_lo, b_hi)
        assert info["max_distance"] < 1e-9
        assert np.all(x_pia >= b_lo - 1e-9)
        assert np.all(x_pia <= b_hi + 1e-9)

    def test_incompatible_positive_distance(self):
        A = np.array([[1.0], [1.0], [1.0]])
        b_lo = np.array([1.0, 1.0, 4.0])
        b_hi = np.array([2.0, 2.0, 5.0])
        x_pia, e, info = solve_interval_pia(A, b_lo, b_hi, norm="inf")
        assert info["max_distance"] > 0.0
        assert x_pia[0] >= 0
        assert e[2] == pytest.approx(info["max_distance"], abs=1e-9)

    def test_one_norm_vs_inf_norm(self):
        A = np.array([[1.0], [1.0], [1.0]])
        b_lo = np.array([1.0, 1.0, 4.0])
        b_hi = np.array([2.0, 2.0, 5.0])
        x1, _, i1 = solve_interval_pia(A, b_lo, b_hi, norm="one")
        xi, _, ii = solve_interval_pia(A, b_lo, b_hi, norm="inf")
        assert i1["total_distance"] <= ii["total_distance"] + 1e-9
        assert ii["max_distance"] <= i1["max_distance"] + 1e-9
        assert np.all(x1 >= 0) and np.all(xi >= 0)

    def test_weights_validation(self):
        with pytest.raises(ValueError, match="weights"):
            solve_interval_pia(
                self.A, self.b_lo, self.b_hi, norm="one",
                weights=np.array([0.0, 1.0]),
            )

    def test_norm_validation(self):
        with pytest.raises(ValueError, match="norm"):
            solve_interval_pia(self.A, self.b_lo, self.b_hi, norm="two")

    def test_tv_bound(self):
        x_pia, _, info = solve_interval_pia(
            self.A, self.b_lo, self.b_hi, tv_bound=1.0
        )
        assert info["converged"]
        assert np.all(x_pia >= 0)

    def test_book_data_max_distance(self):
        A = np.array([[1.0, -1.0], [1.0, -2.0], [1.0, -3.0]])
        b_lo = np.array([1.0, 2.0, 1.5])
        b_hi = np.array([2.5, 3.0, 2.0])
        # the data are compatible, so the PIA distance must vanish
        _, _, info = solve_interval_pia(A, b_lo, b_hi)
        assert info["max_distance"] < 1e-9


class TestSolveIntervalMatrix:
    A = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
    b_lo = np.array([3.0, 3.0, 1.0])
    b_hi = np.array([5.0, 7.0, 2.0])

    def test_point_matrix_matches_solve_interval(self):
        x_min, x_max, info = solve_interval_matrix(self.A, self.A, self.b_lo, self.b_hi)
        p_min, p_max = solve_interval(self.A, self.b_lo, self.b_hi)
        assert np.allclose(x_min, p_min, atol=1e-6)
        assert np.allclose(x_max, p_max, atol=1e-6)
        assert info["functional"] == "tol"

    def test_tolerable_inside_united(self):
        A_lo = self.A * 0.9
        A_hi = self.A * 1.1
        t_min, t_max, _ = solve_interval_matrix(A_lo, A_hi, self.b_lo, self.b_hi)
        u_min, u_max, _ = solve_interval_matrix(
            A_lo, A_hi, self.b_lo, self.b_hi, functional="uss"
        )
        assert np.all(u_min <= t_min + 1e-6)
        assert np.all(t_max <= u_max + 1e-6)

    def test_united_widens_vs_point(self):
        A_lo = self.A * 0.9
        A_hi = self.A * 1.1
        u_min, u_max, _ = solve_interval_matrix(
            A_lo, A_hi, self.b_lo, self.b_hi, functional="uss", variativity=False
        )
        p_min, p_max = solve_interval(self.A, self.b_lo, self.b_hi)
        assert np.all(u_min <= p_min + 1e-6)
        assert np.all(u_max >= p_max - 1e-6)

    def test_inner_box_inside_bounds(self):
        A_lo = self.A * 0.95
        A_hi = self.A * 1.05
        lo, hi, info = solve_interval_matrix(
            A_lo, A_hi, self.b_lo, self.b_hi, inner_box=True, variativity=False
        )
        assert info["inner_lower"] is not None
        assert np.all(lo <= info["inner_lower"] + 1e-6)
        assert np.all(info["inner_upper"] <= hi + 1e-6)
        assert np.all(info["inner_lower"] <= info["inner_upper"] + 1e-6)

    def test_tv_bound(self):
        A_lo = self.A * 0.95
        A_hi = self.A * 1.05
        lo, hi, _ = solve_interval_matrix(
            A_lo, A_hi, self.b_lo, self.b_hi, tv_bound=1.0
        )
        assert np.all(lo <= hi)

    def test_functional_validation(self):
        with pytest.raises(ValueError, match="functional"):
            solve_interval_matrix(self.A, self.A, self.b_lo, self.b_hi,
                                  functional="bogus")

    def test_matrix_validation(self):
        with pytest.raises(ValueError, match="A_lo must be <= A_hi"):
            solve_interval_matrix(self.A * 1.1, self.A, self.b_lo, self.b_hi)


class TestIntervalSevVariativity:
    def test_formula(self):
        A = np.eye(2)
        b_lo = np.array([1.0, 1.0])
        b_hi = np.array([3.0, 3.0])
        x_hat = np.array([2.0, 2.0])
        # b' = [2, 2] (formula 4.67), cond2([I; I]) = 1, maxTol = 1
        sev = interval_sev_variativity(A, A, b_lo, b_hi, x_hat, tol_max=1.0)
        expected = np.sqrt(2) * 1.0 * 1.0 * np.linalg.norm(x_hat) / (
            np.linalg.norm([2.0, 2.0])
        )
        assert sev == pytest.approx(expected, rel=1e-9)

    def test_empty_set_returns_zero(self):
        A = np.eye(2)
        sev = interval_sev_variativity(
            A, A, np.array([1.0, 1.0]), np.array([2.0, 2.0]),
            np.array([1.0, 1.0]), tol_max=-0.5,
        )
        assert sev == 0.0

    def test_rank_deficient_returns_inf(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        sev = interval_sev_variativity(
            A, A, np.array([1.0, 2.0]), np.array([2.0, 4.0]),
            np.array([1.0, 1.0]), tol_max=0.5,
        )
        assert sev == float("inf")

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="mode"):
            interval_sev_variativity(
                np.eye(2), np.eye(2), np.ones(2), np.ones(2) + 1,
                np.ones(2), 1.0, mode="bad",
            )


class TestIntervalCompatibilityReport:
    def test_compatible_sample(self):
        rep = interval_compatibility_report(
            np.array([1.0, 2.0, 2.5]), np.array([3.0, 4.0, 2.8])
        )
        assert rep["compatible"]
        assert rep["ji_sample"] > 0
        assert rep["clique_number"] == 3
        assert rep["outlier_candidates"] == []
        assert np.allclose(rep["intersection"], [2.5, 2.8])

    def test_incompatible_sample_flags_outlier(self):
        rep = interval_compatibility_report(
            np.array([1.0, 1.0, 5.0]), np.array([2.0, 2.0, 6.0])
        )
        assert not rep["compatible"]
        assert rep["ji_sample"] < 0
        assert rep["clique_number"] == 2
        assert rep["outlier_candidates"] == [2]

    def test_validation(self):
        with pytest.raises(ValueError):
            interval_compatibility_report(
                np.array([2.0]), np.array([1.0])
            )


class TestPosteriorMonteCarlo:
    def test_mc_envelope_inside_guaranteed_bounds(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
        b_lo = np.array([3.0, 3.0, 1.0])
        b_hi = np.array([5.0, 7.0, 2.0])
        x_min, x_max, info = solve_interval_posterior(
            A, b_lo, b_hi, n_samples=20, random_state=1
        )
        assert info["n_feasible"] >= 2
        assert np.all(info["spectrum_mc_lower"] >= x_min - 1e-6)
        assert np.all(info["spectrum_mc_upper"] <= x_max + 1e-6)
        assert np.all(info["spectrum_mc_std"] >= 0)

    def test_reproducible(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
        b_lo = np.array([3.0, 3.0, 1.0])
        b_hi = np.array([5.0, 7.0, 2.0])
        _, _, i1 = solve_interval_posterior(A, b_lo, b_hi, n_samples=10,
                                           random_state=7)
        _, _, i2 = solve_interval_posterior(A, b_lo, b_hi, n_samples=10,
                                           random_state=7)
        assert np.allclose(i1["spectrum_mc_mean"], i2["spectrum_mc_mean"])


class TestIntervalExtras:
    def test_dose_bounds_contain_midpoint(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        for key in result["doserates"]:
            assert (
                result["doserates_lower"][key]
                <= result["doserates"][key] + 1e-9
            )
            assert (
                result["doserates"][key]
                <= result["doserates_upper"][key] + 1e-9
            )

    def test_width_metrics(self, detector, readings):
        result = detector.unfold_interval(readings, noise_level=0.1)
        assert np.allclose(
            result["width"],
            result["spectrum_upper"] - result["spectrum_lower"],
        )
        assert np.all(result["relative_width"] >= 0)

    def test_tol_drop_infeasible(self, detector, readings):
        result = detector.unfold_interval_tol(
            readings, noise_level=0.001, drop_infeasible=1
        )
        assert "dropped_readings" in result
        assert result["dropped_readings"] == []  # data compatible

    def test_tol_variativity_key(self, detector, readings):
        result = detector.unfold_interval_tol(
            readings, noise_level=0.1, variativity=True
        )
        assert "sev" in result


class TestDetectorUnfoldIntervalCenter:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval_center(readings, noise_level=0.1)
        assert result["method"] == "IntervalCenter"
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])
        assert np.all(result["spectrum_lower"] >= 0)
        assert result["compatible"]
        assert np.allclose(result["epsilon"], 0.0)

    def test_midpoint_in_interval(self, detector, readings):
        result = detector.unfold_interval_center(readings, noise_level=0.1)
        assert np.all(result["spectrum"] >= result["spectrum_lower"] - 1e-9)
        assert np.all(result["spectrum"] <= result["spectrum_upper"] + 1e-9)

    def test_tv_bound(self, detector, readings):
        result = detector.unfold_interval_center(
            readings, noise_level=0.1, tv_bound=5.0
        )
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_doserates_present(self, detector, readings):
        result = detector.unfold_interval_center(readings, noise_level=0.1)
        assert "doserates" in result
        assert "doserates_lower" in result


class TestDetectorUnfoldIntervalPia:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval_pia(readings, noise_level=0.1)
        assert result["method"] == "IntervalPIA"
        assert np.all(result["spectrum"] >= 0)
        assert result["max_distance"] >= 0
        assert np.all(
            result["spectrum_lower"] <= result["spectrum_upper"] + 1e-9
        )

    def test_point_estimate_mirrored(self, detector, readings):
        result = detector.unfold_interval_pia(readings, noise_level=0.1)
        assert np.allclose(result["spectrum_lower"], result["spectrum"])
        assert np.allclose(result["spectrum_upper"], result["spectrum"])

    def test_compute_bounds(self, detector, readings):
        result = detector.unfold_interval_pia(
            readings, noise_level=0.1, compute_bounds=True
        )
        assert np.all(result["spectrum_lower"] <= result["spectrum"] + 1e-9)
        assert np.all(result["spectrum"] <= result["spectrum_upper"] + 1e-9)

    def test_one_norm(self, detector, readings):
        result = detector.unfold_interval_pia(
            readings, noise_level=0.1, norm="one"
        )
        assert result["norm"] == "one"

    def test_tv_bound(self, detector, readings):
        result = detector.unfold_interval_pia(
            readings, noise_level=0.1, tv_bound=5.0
        )
        assert np.all(result["spectrum"] >= 0)


class TestDetectorUnfoldIntervalMatrix:
    def test_requires_uncertainty(self, detector, readings):
        with pytest.raises(ValueError, match="sensitivity_uncertainties"):
            detector.unfold_interval_matrix(readings)

    def test_basic(self, detector, readings):
        result = detector.unfold_interval_matrix(
            readings, sensitivity_uncertainties=0.05
        )
        assert result["method"] == "IntervalMatrix"
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])
        assert np.all(result["spectrum_lower"] >= 0)
        assert "sev" in result
        assert result["functional"] == "tol"

    def test_matches_tol_at_zero_uncertainty(self, detector, readings):
        matrix = detector.unfold_interval_matrix(
            readings, sensitivity_uncertainties=0.0, variativity=False
        )
        point = detector.unfold_interval_tol(readings, noise_level=0.05)
        assert np.allclose(
            matrix["spectrum_upper"], point["spectrum_upper"], atol=1e-6
        )

    def test_uss_widens_vs_point(self, detector, readings):
        uss = detector.unfold_interval_matrix(
            readings, sensitivity_uncertainties=0.05, functional="uss",
            tv_bound=5.0, variativity=False,
        )
        point = detector.unfold_interval(readings, noise_level=0.05,
                                         tv_bound=5.0)
        assert np.all(uss["spectrum_upper"] >= point["spectrum_lower"] - 1e-6)

    def test_explicit_bounds(self, detector, readings):
        from src.bssunfold.core._base_unfolder import _build_system

        A, _, _ = _build_system(
            readings, detector.detector_names,
            {k: v for k, v in detector.sensitivities.items()},
        )
        result = detector.unfold_interval_matrix(
            readings, A_lo=0.95 * A, A_hi=1.05 * A
        )
        assert np.all(np.isfinite(result["spectrum"]))

    def test_invalid_uncertainty(self, detector, readings):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            detector.unfold_interval_matrix(
                readings, sensitivity_uncertainties=1.5
            )


class TestSolveIntervalRegularization:
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([3.0, 3.0])

    def _lo_hi(self):
        return self.b - 0.1, self.b + 0.1

    def test_zero_tau_matches_tol(self):
        b_lo, b_hi = self._lo_hi()
        _, _, reg = solve_interval_regularization(self.A, b_lo, b_hi, tau=0.0)
        _, _, tol = solve_interval_tol(self.A, b_lo, b_hi)
        assert np.allclose(reg["x_pseudo"], tol["x_pseudo"], atol=1e-6)
        assert abs(reg["tol_max"] - tol["tol_max"]) < 1e-9

    def test_shift_inflation_shrinks_tolerable_reserve(self):
        b_lo, b_hi = self._lo_hi()
        _, _, i0 = solve_interval_regularization(self.A, b_lo, b_hi, tau=0.0)
        _, _, i2 = solve_interval_regularization(self.A, b_lo, b_hi, tau=0.2)
        # widening the matrix expands Xi_uni but shrinks Xi_tol (Shary 2017)
        assert i2["tol_max"] < i0["tol_max"]

    def test_uss_inflation_expands_reserve(self):
        b_lo, b_hi = self._lo_hi()
        _, _, i0 = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.0, functional="uss"
        )
        _, _, i2 = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.2, functional="uss"
        )
        assert i2["tol_max"] > i0["tol_max"]

    def test_bounds_enclose_inner_box(self):
        b_lo, b_hi = self._lo_hi()
        x_min, x_max, info = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.05, inner_box=True
        )
        assert np.all(x_min <= x_max)
        assert info["inner_lower"] is not None
        assert np.all(info["inner_lower"] >= x_min - 1e-9)
        assert np.all(info["inner_upper"] <= x_max + 1e-9)

    def test_tau_sweep_monotone(self):
        b_lo, b_hi = self._lo_hi()
        grid = [0.0, 0.05, 0.1, 0.2]
        _, _, info = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.05, tau_grid=grid
        )
        sweep = info["tau_sweep"]
        assert [s["tau"] for s in sweep] == grid
        tols = [s["tol_max"] for s in sweep]
        assert all(b1 >= b2 - 1e-9 for b1, b2 in zip(tols, tols[1:]))
        assert all(np.isfinite(s["residual_inf"]) for s in sweep)

    def test_relative_inflation_radius(self):
        b_lo, b_hi = self._lo_hi()
        _, _, sh = solve_interval_regularization(self.A, b_lo, b_hi, tau=0.2)
        _, _, rel = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.2, inflation="relative"
        )
        # elementwise widening shrinks the reserve more than a shift
        assert rel["tol_max"] < sh["tol_max"]

    def test_invalid_parameters(self):
        b_lo, b_hi = self._lo_hi()
        with pytest.raises(ValueError, match="inflation"):
            solve_interval_regularization(self.A, b_lo, b_hi, inflation="bad")
        with pytest.raises(ValueError, match="tau"):
            solve_interval_regularization(self.A, b_lo, b_hi, tau=-0.1)
        with pytest.raises(ValueError, match="functional"):
            solve_interval_regularization(
                self.A, b_lo, b_hi, functional="ols"
            )

    def test_tv_bound(self):
        b_lo, b_hi = self._lo_hi()
        x_min, x_max, info = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=0.05, tv_bound=2.0
        )
        assert info["converged"]
        assert np.all(np.abs(x_max - x_min) <= 2.0 + 1e-6)


class TestDetectorUnfoldIntervalRegularization:
    def test_basic(self, detector, readings):
        result = detector.unfold_interval_regularization(readings, tau=0.05)
        assert result["method"] == "IntervalRegularization"
        assert "tol_max" in result
        assert "spectrum_lower" in result
        assert "spectrum_upper" in result
        assert np.allclose(result["spectrum"], result["x_pseudo"])

    def test_tau_and_sweep_keys(self, detector, readings):
        result = detector.unfold_interval_regularization(
            readings, tau=0.02, tau_grid=[0.0, 0.02, 0.05]
        )
        assert result["tau"] == 0.02
        assert result["inflation"] == "shift"
        assert len(result["tau_sweep"]) == 3

    def test_relative_inflation(self, detector, readings):
        result = detector.unfold_interval_regularization(
            readings, tau=0.05, inflation="relative"
        )
        assert np.all(np.isfinite(result["spectrum"]))
        assert np.all(result["spectrum_lower"] <= result["spectrum_upper"])

    def test_doserates_present(self, detector, readings):
        result = detector.unfold_interval_regularization(readings, tau=0.05)
        assert "doserates" in result
        assert "doserates_lower" in result

    def test_invalid_inflation(self, detector, readings):
        with pytest.raises(ValueError, match="inflation"):
            detector.unfold_interval_regularization(
                readings, inflation="diag"
            )


class TestIntervalRegularizationSmoothing:
    """Smoothing of the spiky LP argmax-Tol vertex on the max-Tol face."""

    A = np.array(
        [[3.0, 2.0, 1.0, 0.5, 0.2],
         [0.5, 1.5, 2.5, 2.0, 1.0],
         [0.2, 0.8, 1.5, 2.2, 2.6]]
    )
    TAU = 0.1

    def _lo_hi(self):
        mid = self.A @ np.array([1.0, 0.8, 0.6, 0.5, 0.4])
        return mid - 0.2, mid + 0.2

    def test_default_none_keeps_vertex(self):
        b_lo, b_hi = self._lo_hi()
        _, _, info = solve_interval_regularization(self.A, b_lo, b_hi,
                                                   tau=self.TAU)
        assert info["smoothing"] == "none"
        assert "x_smoothed" not in info

    def test_curvature_preserves_tolerance_within_hull(self):
        from src.bssunfold.core.unfold_interval import _tol_functional

        b_lo, b_hi = self._lo_hi()
        A_rad = self.TAU * np.eye(*self.A.shape)
        _, _, v = solve_interval_regularization(self.A, b_lo, b_hi,
                                                tau=self.TAU)
        x_min, x_max, s = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=self.TAU, smoothing="curvature"
        )
        assert s["tol_max"] == v["tol_max"]
        xs = s["x_smoothed"]
        assert np.all(xs >= 0.0)
        tol_at = _tol_functional(
            xs, self.A, (b_lo + b_hi) / 2.0,
            (b_hi - b_lo) / 2.0, A_rad, "tol",
        )
        assert tol_at >= s["tol_max"] - 1e-9
        assert np.all(xs >= x_min - 1e-9)
        assert np.all(xs <= x_max + 1e-9)

    def test_variation_minimizes_tv_on_face(self):
        b_lo, b_hi = self._lo_hi()
        _, _, v = solve_interval_regularization(self.A, b_lo, b_hi,
                                                tau=self.TAU)
        _, _, s = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=self.TAU, smoothing="variation"
        )
        tv_vertex = float(np.sum(np.abs(np.diff(v["x_pseudo"]))))
        tv_smooth = float(np.sum(np.abs(np.diff(s["x_smoothed"]))))
        # the vertex is feasible on the face, so the min-TV point is no worse
        assert tv_smooth <= tv_vertex + 1e-9

    def test_curvature_minimizes_second_differences(self):
        b_lo, b_hi = self._lo_hi()
        _, _, v = solve_interval_regularization(self.A, b_lo, b_hi,
                                                tau=self.TAU)
        _, _, s = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=self.TAU, smoothing="curvature"
        )
        c_vertex = float(np.sum(np.abs(np.diff(v["x_pseudo"], n=2))))
        c_smooth = float(np.sum(np.abs(np.diff(s["x_smoothed"], n=2))))
        assert c_smooth <= c_vertex + 1e-9

    def test_face_slack_relaxes_floor(self):
        b_lo, b_hi = self._lo_hi()
        _, _, s0 = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=self.TAU, smoothing="curvature"
        )
        _, _, s5 = solve_interval_regularization(
            self.A, b_lo, b_hi, tau=self.TAU,
            smoothing="curvature", face_slack=0.5,
        )
        assert s5["face_slack"] == 0.5
        c0 = float(np.sum(np.abs(np.diff(s0["x_smoothed"], n=2))))
        c5 = float(np.sum(np.abs(np.diff(s5["x_smoothed"], n=2))))
        # a relaxed face contains the exact one, so smoothing can only improve
        assert c5 <= c0 + 1e-9

    def test_invalid_smoothing_parameters_raise(self):
        b_lo, b_hi = self._lo_hi()
        with pytest.raises(ValueError, match="smoothing"):
            solve_interval_regularization(
                self.A, b_lo, b_hi, tau=self.TAU, smoothing="face"
            )
        with pytest.raises(ValueError, match="face_slack"):
            solve_interval_regularization(
                self.A, b_lo, b_hi, tau=self.TAU,
                smoothing="curvature", face_slack=1.0,
            )

    def test_detector_wrapper_smoothing(self, detector, readings):
        result = detector.unfold_interval_regularization(
            readings, tau=0.05, smoothing="curvature"
        )
        assert result["smoothing"] == "curvature"
        assert "x_smoothed" in result
        assert np.allclose(result["spectrum"], result["x_smoothed"])
        assert np.all(np.isfinite(result["spectrum"]))
        assert np.all(result["spectrum"] >= 0.0)


class TestTolsolvtyCrossValidation:
    """Exact LP Tol maximization vs published tolsolvty reference values.

    Fixtures from https://github.com/MaximSmolskiy/tolsolvty (test_data/1..7):
    interval systems with the reference maximum ``T`` and argmax ``tau`` of
    the recognizing functional computed by the Shor r-algorithm. All
    reference argmaxes are non-negative, so the package's x >= 0 LP must
    reproduce the unconstrained maximum exactly.
    """

    DATA = pathlib.Path(__file__).parent / "data" / "tolsolvty"
    CASES = list(range(1, 8))

    def _system(self, case):
        d = self.DATA / str(case)
        inf_A = np.loadtxt(d / "inf_A.txt", ndmin=2)
        sup_A = np.loadtxt(d / "sup_A.txt", ndmin=2)
        inf_b = np.loadtxt(d / "inf_b.txt", ndmin=1)
        sup_b = np.loadtxt(d / "sup_b.txt", ndmin=1)
        ref_max = float(np.atleast_1d(np.loadtxt(d / "T.txt"))[0])
        ref_arg = np.loadtxt(d / "tau.txt", ndmin=1)
        return inf_A, sup_A, inf_b, sup_b, ref_max, ref_arg

    @pytest.mark.parametrize("case", CASES)
    def test_tol_max_matches_reference(self, case):
        from src.bssunfold.core.unfold_interval import _tol_max_lp

        inf_A, sup_A, inf_b, sup_b, ref_max, _ = self._system(case)
        lp = _tol_max_lp(
            0.5 * (inf_A + sup_A),
            0.5 * (sup_A - inf_A),
            inf_b,
            sup_b,
            None,
            "tol",
        )
        assert lp["converged"]
        assert abs(lp["tol_max"] - ref_max) < 1e-6

    @pytest.mark.parametrize("case", CASES)
    def test_reference_argmax_attains_reference_value(self, case):
        from src.bssunfold.core.unfold_interval import _tol_functional

        inf_A, sup_A, inf_b, sup_b, ref_max, ref_arg = self._system(case)
        value = _tol_functional(
            ref_arg,
            0.5 * (inf_A + sup_A),
            0.5 * (inf_b + sup_b),
            0.5 * (sup_b - inf_b),
            0.5 * (sup_A - inf_A),
            "tol",
        )
        assert abs(value - ref_max) < 1e-6
