"""Tests for the three classic reimplemented unfolding codes.

These cover CRYSTAL BALL (direct delta-operator method), RFSP-JUL
(iterative damped least squares) and STAY'SL (single-step Bayesian
least squares). All are independent open-source reimplementations of
proprietary codes, built from the published
mathematical descriptions.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import (
    solve_crystal_ball,
    solve_rfsp_jul,
    solve_staysl,
)


@pytest.fixture
def detector() -> Detector:
    """Default Detector instance with default response functions."""
    return Detector()


@pytest.fixture
def readings(detector):
    """Single-detector readings fixture (consistent with other test modules)."""
    return {detector.detector_names[0]: 100.0}


# ============================================================================
# CRYSTAL BALL
# ============================================================================


class TestSolveCrystalBall:
    def test_crystal_ball_basic(self):
        np.random.seed(0)
        A = np.random.rand(6, 12)
        x_true = np.exp(-np.linspace(0, 4, 12))
        b = A @ x_true
        x, iterations, converged = solve_crystal_ball(A, b)
        assert len(x) == 12
        assert iterations == 1
        assert converged is True
        assert np.all(x >= 0)
        # Should reproduce the measurements reasonably in a least-squares sense.
        assert np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-12) < 0.5

    def test_crystal_ball_regularization(self):
        np.random.seed(1)
        A = np.random.rand(4, 20)
        x_true = np.exp(-np.linspace(0, 3, 20))
        b = A @ x_true
        x, _, _ = solve_crystal_ball(A, b, regularization=1e-3)
        assert len(x) == 20
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)

    def test_crystal_ball_all_zero_measurements(self):
        A = np.random.rand(5, 10)
        b = np.zeros(5)
        with pytest.raises(ValueError, match="All measurements are zero"):
            solve_crystal_ball(A, b)


# ============================================================================
# RFSP-JUL
# ============================================================================


class TestSolveRFSPJUL:
    def test_rfsp_jul_basic(self):
        np.random.seed(2)
        A = np.random.rand(6, 12)
        x_true = np.exp(-np.linspace(0, 4, 12))
        b = A @ x_true
        x0 = np.ones(12)
        x, iterations, converged = solve_rfsp_jul(A, b, x0, max_iterations=500)
        assert len(x) == 12
        assert iterations > 0
        assert np.all(x >= 0)
        # RFSP-JUL drives the weighted residual down substantially.
        assert np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-12) < 0.5

    def test_rfsp_jul_converges(self):
        np.random.seed(3)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x0 = np.ones(10)
        x, iterations, converged = solve_rfsp_jul(
            A, b, x0, max_iterations=20000, tolerance=1e-4
        )
        assert converged
        assert iterations <= 20000

    def test_rfsp_jul_custom_weights(self):
        np.random.seed(4)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x0 = np.ones(10)
        weights = np.array([1.0, 2.0, 0.5, 1.0, 1.0])
        x, _, _ = solve_rfsp_jul(A, b, x0, weights=weights)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_rfsp_jul_all_zero_measurements(self):
        A = np.random.rand(5, 10)
        b = np.zeros(5)
        with pytest.raises(ValueError, match="All measurements are zero"):
            solve_rfsp_jul(A, b, np.ones(10))


# ============================================================================
# STAY'SL
# ============================================================================


class TestSolveSTAYSL:
    def test_staysl_basic(self):
        np.random.seed(5)
        A = np.random.rand(6, 12)
        x_true = np.exp(-np.linspace(0, 4, 12))
        b = A @ x_true
        x0 = np.ones(12)
        x, iterations, converged = solve_staysl(A, b, x0)
        assert len(x) == 12
        assert iterations == 1
        assert converged is True
        assert np.all(x >= 0)
        # With a weak prior it should track the measurements.
        assert np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-12) < 0.5

    def test_staysl_strong_prior(self):
        np.random.seed(6)
        A = np.random.rand(6, 12)
        x_true = np.exp(-np.linspace(0, 4, 12))
        b = A @ x_true
        x0 = np.ones(12)
        # A very tight explicit prior covariance forces the posterior mean to
        # stay at the prior (the measurements are then effectively ignored).
        Cx = 1e-8 * np.eye(12)
        x, _, _ = solve_staysl(A, b, x0, Cx=Cx)
        assert np.allclose(x, x0, atol=1e-4)

    def test_staysl_explicit_covariance(self):
        np.random.seed(7)
        A = np.random.rand(5, 10)
        x_true = np.exp(-np.linspace(0, 3, 10))
        b = A @ x_true
        x0 = np.ones(10)
        Cb = np.diag((0.1 * np.abs(b)) ** 2)
        Cx = np.diag((1.0 * x0) ** 2)
        x, _, _ = solve_staysl(A, b, x0, Cb=Cb, Cx=Cx)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_staysl_zero_measurements(self):
        # All-zero measurements with the default relative uncertainty make the
        # measurement covariance collapse (Cb -> 0), a degenerate input that
        # STAY'SL should still handle without crashing or producing NaN/neg.
        A = np.random.rand(5, 10)
        b = np.zeros(5)
        x0 = np.ones(10)
        x, _, _ = solve_staysl(A, b, x0)
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)


# ============================================================================
# Detector-level wrappers
# ============================================================================


class TestUnfoldClassicCodes:
    def test_unfold_crystal_ball(self, detector, readings):
        result = detector.unfold_crystal_ball(readings)
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_crystal_ball_regularization(self, detector, readings):
        result = detector.unfold_crystal_ball(readings, regularization=1e-2)
        assert "spectrum" in result
        assert np.all(np.isfinite(result["spectrum"]))

    def test_unfold_rfsp_jul(self, detector, readings):
        result = detector.unfold_rfsp_jul(readings, max_iterations=100)
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["method"] == "RFSP-JUL"

    def test_unfold_rfsp_jul_no_save(self, detector, readings):
        result = detector.unfold_rfsp_jul(readings, save_result=False)
        assert "spectrum" in result

    def test_unfold_staysl(self, detector, readings):
        result = detector.unfold_staysl(readings)
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["method"] == "STAY'SL"

    def test_unfold_staysl_custom_uncertainty(self, detector, readings):
        result = detector.unfold_staysl(
            readings, relative_uncertainty=0.2, prior_uncertainty=0.5
        )
        assert "spectrum" in result
        assert np.all(np.isfinite(result["spectrum"]))
