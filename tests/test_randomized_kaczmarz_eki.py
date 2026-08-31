"""Tests for Randomized Kaczmarz and Ensemble Kalman Inversion methods."""

from typing import Dict

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core.unfold_eki import solve_eki, unfold_eki
from bssunfold.core.unfold_randomized_kaczmarz import (
    solve_randomized_kaczmarz,
    unfold_randomized_kaczmarz,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    return Detector()


@pytest.fixture(scope="module")
def readings(detector: Detector) -> Dict[str, float]:
    ref = {"E_MeV": detector.E_MeV, "Phi": np.ones(len(detector.E_MeV)) * 1e-4}
    return detector.get_effective_readings_for_spectra(ref)


@pytest.fixture(scope="module")
def A(detector: Detector, readings: Dict[str, float]) -> np.ndarray:
    names = [n for n in detector.detector_names if n in readings]
    return np.array([detector.sensitivities[n] for n in names])


@pytest.fixture(scope="module")
def b(A: np.ndarray) -> np.ndarray:
    return A @ np.ones(A.shape[1]) * 1e-4


@pytest.fixture(scope="module")
def x0(A: np.ndarray) -> np.ndarray:
    return np.zeros(A.shape[1])


# ===========================================================================
#  solve_randomized_kaczmarz
# ===========================================================================

class TestSolveRandomizedKaczmarz:
    """Core solver tests for randomized Kaczmarz."""

    def test_basic(self, A, b, x0):
        spec, iters, conv = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=200, random_state=42,
        )
        assert spec.shape == (A.shape[1],)
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0)
        assert iters > 0

    def test_deterministic(self, A, b, x0):
        r1 = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=100, random_state=42,
        )
        r2 = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=100, random_state=42,
        )
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_different_seeds(self, A, b, x0):
        r1 = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=200, random_state=1,
        )
        r2 = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=200, random_state=2,
        )
        assert not np.allclose(r1[0], r2[0])

    def test_zero_measurements(self, A, x0):
        b_zero = np.zeros(A.shape[0])
        spec, iters, conv = solve_randomized_kaczmarz(
            A, b_zero, x0, max_iterations=50, random_state=0,
        )
        assert spec.shape == (A.shape[1],)

    def test_relaxation(self, A, b, x0):
        spec, _, _ = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=200, omega=0.5, random_state=42,
        )
        assert np.all(spec >= 0)

    def test_convergence_flag(self, A, b, x0):
        _, iters, conv = solve_randomized_kaczmarz(
            A, b, x0, max_iterations=5000, tolerance=1e-4, random_state=42,
        )
        # With enough iterations it should converge
        assert isinstance(conv, bool)


# ===========================================================================
#  unfold_randomized_kaczmarz
# ===========================================================================

class TestUnfoldRandomizedKaczmarz:
    """Detector wrapper tests for randomized Kaczmarz."""

    def test_basic(self, detector, readings):
        result = unfold_randomized_kaczmarz(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            max_iterations=200,
            random_state=42,
        )
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "Randomized Kaczmarz"
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        assert np.all(result["spectrum"] >= 0)

    def test_single_detector(self, detector):
        first = detector.detector_names[0]
        readings = {first: 100.0}
        result = unfold_randomized_kaczmarz(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            max_iterations=100,
            random_state=0,
        )
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        saved = []
        unfold_randomized_kaczmarz(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: saved.append(r),
            readings=readings,
            max_iterations=100,
            save_result=True,
            random_state=0,
        )
        assert len(saved) == 1
        assert "spectrum" in saved[0]

    def test_calculate_errors(self, detector, readings):
        result = unfold_randomized_kaczmarz(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            max_iterations=100,
            calculate_errors=True,
            noise_level=0.05,
            n_montecarlo=5,
            random_state=0,
        )
        assert "spectrum_uncert_mean" in result

    def test_detector_method(self, detector, readings):
        result = detector.unfold_randomized_kaczmarz(
            readings, max_iterations=100, random_state=42, save_result=False,
        )
        assert result["method"] == "Randomized Kaczmarz"
        assert np.all(result["spectrum"] >= 0)


# ===========================================================================
#  solve_eki
# ===========================================================================

class TestSolveEki:
    """Core solver tests for Ensemble Kalman Inversion."""

    def test_basic(self, A, b, x0):
        spec, iters, conv = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10, random_state=42,
        )
        assert spec.shape == (A.shape[1],)
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0)
        assert iters == 10
        assert conv is True

    def test_deterministic(self, A, b, x0):
        r1 = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10, random_state=42,
        )
        r2 = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10, random_state=42,
        )
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_ensemble_size(self, A, b, x0):
        spec, _, _ = solve_eki(
            A, b, x0, n_ensemble=10, n_iterations=5, random_state=0,
        )
        assert spec.shape == (A.shape[1],)

    def test_regularization(self, A, b, x0):
        spec, _, _ = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10,
            regularization=1e-2, random_state=42,
        )
        assert np.all(spec >= 0)

    def test_inflation(self, A, b, x0):
        spec, _, _ = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10,
            inflation=1.1, random_state=42,
        )
        assert np.all(spec >= 0)

    def test_noise_std(self, A, b, x0):
        spec, _, _ = solve_eki(
            A, b, x0, n_ensemble=20, n_iterations=10,
            noise_std=0.1, random_state=42,
        )
        assert np.all(spec >= 0)


# ===========================================================================
#  unfold_eki
# ===========================================================================

class TestUnfoldEki:
    """Detector wrapper tests for EKI."""

    def test_basic(self, detector, readings):
        result = unfold_eki(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            n_ensemble=20,
            n_iterations=10,
            random_state=42,
        )
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "EKI"
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        assert np.all(result["spectrum"] >= 0)

    def test_single_detector(self, detector):
        first = detector.detector_names[0]
        readings = {first: 100.0}
        result = unfold_eki(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            n_ensemble=15,
            n_iterations=5,
            random_state=0,
        )
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        saved = []
        unfold_eki(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: saved.append(r),
            readings=readings,
            n_ensemble=15,
            n_iterations=5,
            save_result=True,
            random_state=0,
        )
        assert len(saved) == 1

    def test_calculate_errors(self, detector, readings):
        result = unfold_eki(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: None,
            readings=readings,
            n_ensemble=15,
            n_iterations=5,
            calculate_errors=True,
            noise_level=0.05,
            n_montecarlo=5,
            random_state=0,
        )
        assert "spectrum_uncert_mean" in result

    def test_detector_method(self, detector, readings):
        result = detector.unfold_eki(
            readings, n_ensemble=20, n_iterations=10,
            random_state=42, save_result=False,
        )
        assert result["method"] == "EKI"
        assert np.all(result["spectrum"] >= 0)


# ===========================================================================
#  Exports
# ===========================================================================

class TestExports:
    """Verify symbols are importable."""

    def test_solve_randomized_kaczmarz_importable(self):
        from bssunfold.core import unfold_randomized_kaczmarz
        assert hasattr(unfold_randomized_kaczmarz, "solve_randomized_kaczmarz")
        assert hasattr(unfold_randomized_kaczmarz, "unfold_randomized_kaczmarz")

    def test_solve_eki_importable(self):
        from bssunfold.core import unfold_eki
        assert hasattr(unfold_eki, "solve_eki")
        assert hasattr(unfold_eki, "unfold_eki")

    def test_detector_has_methods(self):
        d = Detector()
        assert hasattr(d, "unfold_randomized_kaczmarz")
        assert hasattr(d, "unfold_eki")
