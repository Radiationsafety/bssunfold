"""Tests for the per-bin composite (chimera) unfolding method.

Tests cover:
- solve_composite_per_bin: low-level solver
- unfold_composite_per_bin: high-level wrapper
- Detector.unfold_composite_per_bin: public API
- build_bin_method_map: calibration function
- save/load_bin_method_map: persistence
- Edge cases and error handling
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bssunfold import Detector
from bssunfold.core.unfold_composite_per_bin import (
    DEFAULT_BIN_METHOD_MAP,
    DEFAULT_CANDIDATE_METHODS,
    _gaussian_smooth,
    _global_fallback,
    load_bin_method_map,
    save_bin_method_map,
    solve_composite_per_bin,
)


@pytest.fixture
def detector():
    """Default detector (GSF, 60 bins, 10 detectors)."""
    return Detector()


@pytest.fixture
def sample_readings(detector):
    """Realistic sample readings for testing."""
    # Use a flat spectrum to generate readings
    flat = pd.DataFrame({
        "E_MeV": detector.E_MeV,
        "Phi": np.ones(detector.n_energy_bins) * 0.1,
    })
    return detector.get_effective_readings_for_spectra(flat)


class TestSolveCompositePerBin:
    """Tests for solve_composite_per_bin (low-level)."""

    def test_basic_solve(self, detector, sample_readings):
        """Test basic solve with default settings."""
        A = np.random.rand(8, 60) + 0.1
        b = A @ (np.ones(60) * 0.1) + np.random.randn(8) * 0.01
        b = np.maximum(b, 0)

        spectrum, info = solve_composite_per_bin(A, b)

        assert spectrum.shape == (60,)
        assert np.all(spectrum >= 0)
        assert "selection_mode" in info
        assert "n_methods" in info
        assert info["n_methods"] >= 1

    def test_with_bin_map(self, detector, sample_readings):
        """Test with explicit bin_method_map."""
        A = np.random.rand(8, 60) + 0.1
        x_true = np.ones(60) * 0.1
        b = A @ x_true

        # All bins from landweber
        bin_map = ["landweber"] * 60
        spectrum, info = solve_composite_per_bin(A, b, bin_method_map=bin_map)

        assert spectrum.shape == (60,)
        assert info["selection_mode"] == "per_bin_map"
        assert info["n_from_map"] == 60

    def test_with_unknown_method_in_map(self, detector, sample_readings):
        """Test fallback when bin_map references unknown method."""
        A = np.random.rand(8, 60) + 0.1
        b = A @ (np.ones(60) * 0.1)

        bin_map = ["nonexistent_method"] * 60
        spectrum, info = solve_composite_per_bin(A, b, bin_method_map=bin_map)

        assert spectrum.shape == (60,)
        assert info["n_from_fallback"] == 60

    def test_with_smoothing(self, detector, sample_readings):
        """Test Gaussian smoothing option."""
        A = np.random.rand(8, 60) + 0.1
        b = A @ (np.ones(60) * 0.1)

        spectrum_no_smooth, _ = solve_composite_per_bin(A, b, smooth_sigma=0.0)
        spectrum_smooth, info = solve_composite_per_bin(A, b, smooth_sigma=1.0)

        assert spectrum_smooth.shape == (60,)
        assert info["smooth_sigma"] == 1.0
        # Smoothing should not increase norm dramatically
        assert np.sum(spectrum_smooth) > 0

    def test_custom_methods(self):
        """Test with user-supplied custom methods."""
        A = np.eye(10)
        b = np.ones(10) * 2.0

        def simple_solver(A, b, x0=None, **kw):
            return np.ones(A.shape[1])

        methods = {"simple": (simple_solver, {})}
        spectrum, info = solve_composite_per_bin(A, b, methods=methods)

        assert spectrum.shape == (10,)
        assert info["n_methods"] == 1

    def test_validation_rejects_bad_input(self):
        """Test that validate_system rejects bad A, b."""
        with pytest.raises((ValueError, TypeError)):
            solve_composite_per_bin(None, np.ones(5))

    def test_all_methods_fail(self):
        """Test RuntimeError when all methods fail."""
        A = np.eye(5)
        b = np.ones(5)

        def failing_solver(A, b, x0=None, **kw):
            raise RuntimeError("intentional")

        methods = {"fail": (failing_solver, {})}
        with pytest.raises(RuntimeError, match="All candidate methods failed"):
            solve_composite_per_bin(A, b, methods=methods)

    def test_fallback_combination_modes(self):
        """Test all fallback combination strategies."""
        A = np.eye(5)
        b = np.ones(5) * 2.0

        def ok_solver(A, b, x0=None, **kw):
            return np.ones(A.shape[1]) * 0.5

        methods = {"ok": (ok_solver, {})}
        for mode in ["weighted_average", "median", "trimmed_mean", "best_residual"]:
            spectrum, info = solve_composite_per_bin(
                A, b, methods=methods, fallback_combination=mode
            )
            assert spectrum.shape == (5,)


class TestDetectorUnfoldCompositePerBin:
    """Tests for Detector.unfold_composite_per_bin (high-level API)."""

    def test_basic(self, detector, sample_readings):
        """Test basic usage through Detector."""
        result = detector.unfold_composite_per_bin(
            sample_readings, timeout_per_method=10.0
        )
        assert isinstance(result, dict)
        assert "energy" in result
        assert "spectrum" in result
        assert "method" in result
        assert result["method"] == "CompositePerBin"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_with_smoothing(self, detector, sample_readings):
        """Test with Gaussian smoothing."""
        result = detector.unfold_composite_per_bin(
            sample_readings, smooth_sigma=1.0, timeout_per_method=10.0
        )
        assert result["method"] == "CompositePerBin"
        assert result["parameters"]["smooth_sigma"] == 1.0

    def test_save_result(self, detector, sample_readings):
        """Test save_result option."""
        result = detector.unfold_composite_per_bin(
            sample_readings, save_result=True, timeout_per_method=10.0
        )
        assert result["method"] == "CompositePerBin"


class TestDefaultBinMethodMap:
    """Tests for the DEFAULT_BIN_METHOD_MAP."""

    def test_map_is_not_empty(self):
        """DEFAULT_BIN_METHOD_MAP should be populated."""
        assert len(DEFAULT_BIN_METHOD_MAP) == 60
        assert all(isinstance(m, str) for m in DEFAULT_BIN_METHOD_MAP)

    def test_map_contains_valid_methods(self):
        """All methods in the map should be in DEFAULT_CANDIDATE_METHODS."""
        for m in DEFAULT_BIN_METHOD_MAP:
            assert m in DEFAULT_CANDIDATE_METHODS, f"Unknown method: {m}"


class TestSaveLoadBinMethodMap:
    """Tests for save/load_bin_method_map persistence."""

    def test_save_and_load(self, tmp_path):
        """Test round-trip save/load."""
        test_map = ["cvxpy"] * 30 + ["landweber"] * 30
        path = str(tmp_path / "test_map.json")

        save_bin_method_map(test_map, path, metadata={"test": True})
        assert os.path.exists(path)

        loaded = load_bin_method_map(path)
        assert loaded == test_map

    def test_load_nonexistent_raises(self):
        """Loading non-existent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_bin_method_map("/nonexistent/path.json")


class TestGaussianSmooth:
    """Tests for internal _gaussian_smooth helper."""

    def test_no_smoothing(self):
        """sigma=0 should return copy."""
        x = np.array([1.0, 2.0, 3.0])
        out = _gaussian_smooth(x, 0.0)
        np.testing.assert_array_equal(out, x)

    def test_positive_sigma(self):
        """sigma>0 should smooth."""
        x = np.zeros(20, dtype=float)
        x[10] = 1.0
        out = _gaussian_smooth(x, 1.0)
        assert out.shape == x.shape
        # Peak should be spread but lower
        assert out[10] < x[10]
        # Neighbors should be positive
        assert out[9] > 0
        assert out[11] > 0

    def test_non_negative_output(self):
        """Output should never be negative."""
        x = np.random.randn(100)
        out = _gaussian_smooth(x, 2.0)
        assert np.all(out >= 0)


class TestGlobalFallback:
    """Tests for _global_fallback helper."""

    def test_best_residual(self):
        stacked = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        names = ["a", "b", "c"]
        residuals = {"a": 0.1, "b": 0.01, "c": 1.0}
        A = np.eye(2)
        b = np.ones(2)

        spectrum, mode = _global_fallback(
            stacked, names, residuals, A, b, "best_residual"
        )
        assert mode == "best_residual(b)"
        np.testing.assert_array_equal(spectrum, [3.0, 4.0])

    def test_median(self):
        stacked = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
        names = ["a", "b", "c"]
        residuals = {"a": 1.0, "b": 1.0, "c": 1.0}
        A = np.eye(2)
        b = np.ones(2)

        spectrum, mode = _global_fallback(
            stacked, names, residuals, A, b, "median"
        )
        assert mode == "median"
        np.testing.assert_array_equal(spectrum, [3.0, 4.0])

    def test_weighted_average(self):
        stacked = np.array([[1.0, 2.0], [3.0, 4.0]])
        names = ["a", "b"]
        residuals = {"a": 0.1, "b": 10.0}
        A = np.eye(2)
        b = np.ones(2)

        spectrum, mode = _global_fallback(
            stacked, names, residuals, A, b, "weighted_average"
        )
        assert mode == "weighted_average"
        # 'a' should dominate (lower residual)
        assert spectrum[0] < 2.0  # closer to a's value
