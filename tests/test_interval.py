"""Tests for interval analysis unfolding method."""

import numpy as np
import pandas as pd
import pytest

from src.bssunfold import RF_GSF, Detector
from src.bssunfold.core.unfold_interval import (
    solve_interval,
    solve_interval_intvalpy,
    solve_interval_posterior,
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
