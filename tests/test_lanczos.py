"""Tests for the Lanczos-hybrid (Krylov + GCV) unfolding method."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def detector():
    from bssunfold import Detector

    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def all_readings(detector):
    n = detector.n_detectors
    return {
        name: 100.0 * (1 + 0.1 * i)
        for i, name in enumerate(detector.detector_names[:n])
    }


# ============================================================================
# Test solve_lanczos
# ============================================================================


class TestSolveLanczos:
    def test_basic(self):
        from bssunfold.core import solve_lanczos

        rng = np.random.default_rng(42)
        A = rng.random((5, 10))
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x, iterations, converged = solve_lanczos(A, b)
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_lanczos

        rng = np.random.default_rng(1)
        A = rng.random((4, 8))
        b = A @ np.ones(8)
        x1, _, _ = solve_lanczos(A, b)
        x2, _, _ = solve_lanczos(A, b)
        np.testing.assert_allclose(x1, x2)

    def test_small_max_iterations(self):
        from bssunfold.core import solve_lanczos

        rng = np.random.default_rng(3)
        A = rng.random((5, 10))
        b = A @ np.ones(10)
        x, iterations, _ = solve_lanczos(A, b, max_iterations=1)
        assert len(x) == 10
        assert iterations == 1

    def test_discrepancy_early_stop(self):
        from bssunfold.core import solve_lanczos

        rng = np.random.default_rng(4)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, converged = solve_lanczos(A, b, noise_level=1.0)
        assert len(x) == 12
        assert iterations > 0
        assert converged is True

    def test_zero_measurements(self):
        from bssunfold.core import solve_lanczos

        A = np.random.default_rng(5).random((3, 6))
        x, iterations, converged = solve_lanczos(A, np.zeros(3))
        assert np.all(x == 0)
        assert iterations == 0
        assert converged is True

    def test_zero_matrix_alpha_collapse(self):
        from bssunfold.core import solve_lanczos

        x, iterations, converged = solve_lanczos(np.zeros((3, 5)), np.ones(3))
        assert np.all(x == 0)
        assert converged is True

    def test_square_system_beta_collapse(self):
        from bssunfold.core import solve_lanczos

        A = np.eye(5)
        b = np.ones(5)
        x, iterations, converged = solve_lanczos(A, b)
        assert len(x) == 5
        assert iterations == 1
        assert converged is True

    def test_nonfinite_lambda_fallback(self):
        from unittest.mock import patch

        from bssunfold.core import solve_lanczos

        A = np.random.default_rng(6).random((4, 8))
        b = A @ np.ones(8)
        with patch(
            "bssunfold.core.unfold_lanczos._projected_gcv", return_value=0.0
        ):
            x, iterations, _ = solve_lanczos(A, b, regularization=1e-4)
        assert len(x) == 8
        assert iterations > 0
        with patch(
            "bssunfold.core.unfold_lanczos._projected_gcv",
            return_value=np.nan,
        ):
            x2, _, _ = solve_lanczos(A, b, regularization=1e-4)
        np.testing.assert_allclose(x, x2)

    def test_projected_gcv_direct(self):
        from bssunfold.core.unfold_lanczos import _projected_gcv

        B = np.array([[1.0], [0.0]])
        bhat = np.array([1.0, 0.0])
        lam = _projected_gcv(B, bhat, m=10)
        assert lam > 0
        assert np.isfinite(lam)


# ============================================================================
# Test unfold_lanczos
# ============================================================================


class TestUnfoldLanczos:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_lanczos(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert "effective_readings" in result
        assert "residual" in result
        assert result["method"] == "Lanczos"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_single_detector(self, detector, readings):
        result = detector.unfold_lanczos(readings)
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_lanczos(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "Lanczos"
        assert len(detector.results_history) == 1

    def test_no_save(self, detector, readings):
        detector.clear_results()
        detector.unfold_lanczos(readings, save_result=False)
        assert len(detector.results_history) == 0

    def test_initial_spectrum_ignored(self, detector, all_readings):
        x0 = np.ones(detector.n_energy_bins)
        result = detector.unfold_lanczos(
            all_readings, initial_spectrum=x0, save_result=False
        )
        assert "spectrum" in result

    def test_invalid_regularization_method(self, detector, readings):
        with pytest.raises(ValueError, match="Unsupported regularization"):
            detector.unfold_lanczos(readings, regularization_method="manual")

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_lanczos(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result
        assert result["montecarlo_samples"] == 10

    def test_max_iterations(self, detector, all_readings):
        result = detector.unfold_lanczos(
            all_readings, max_iterations=2, save_result=False
        )
        assert "spectrum" in result
        assert "iterations" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_lanczos, unfold_lanczos

        assert callable(solve_lanczos)
        assert callable(unfold_lanczos)
        assert hasattr(Detector, "unfold_lanczos")

    def test_combined_pipeline(self, detector, all_readings):
        result = detector.unfold_combined(
            all_readings,
            pipeline=[{"method": "lanczos", "params": {"save_result": False}}],
            verbose=False,
        )
        assert "spectrum" in result
        assert result["pipeline_info"]["stages"] == ["lanczos"]


# ============================================================================
# Small synthetic detector
# ============================================================================


class TestLanczosSmallDetector:
    def test_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame(
            {
                "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        d = Detector(df)
        result = d.unfold_lanczos(
            {"sphere_1": 1.0, "sphere_2": 2.0}, save_result=False
        )
        assert len(result["spectrum"]) == 5
        assert np.all(np.isfinite(result["spectrum"]))
