"""Tests for the SAND-II, BUNKI, BUNKI-UT, OSEM, MAP-EM, BSREM and SART
unfolding methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bssunfold.utils.interpolation import interpolate_spectrum


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


def _response_matrix(m=5, n=10, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((m, n)) + 0.1


# ============================================================================
# SAND-II
# ============================================================================


class TestSolveSandii:
    def test_basic(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_sandii(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix(4, 8, seed=1)
        b = A @ np.ones(8)
        x1, _, _ = solve_sandii(A, b, np.ones(8))
        x2, _, _ = solve_sandii(A, b, np.ones(8))
        np.testing.assert_allclose(x1, x2)

    def test_chi_fac_zero(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix(6, 12, seed=7)
        b = A @ np.ones(12)
        x, iterations, converged = solve_sandii(
            A, b, np.ones(12), chi_fac=0, tolerance=1e-2
        )
        assert len(x) == 12
        assert iterations > 0
        assert converged is True

    def test_explicit_sigma(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix(6, 12, seed=8)
        b = A @ np.ones(12)
        sigma = np.ones(6) * 0.05
        x, _, _ = solve_sandii(A, b, np.ones(12), sigma=sigma)
        assert len(x) == 12
        assert np.all(np.isfinite(x))

    def test_all_zero_measurements(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix(3, 6, seed=5)
        with pytest.raises(ValueError, match="zero or negative"):
            solve_sandii(A, np.zeros(3), np.ones(6))

    def test_partial_zero_measurements(self):
        from bssunfold.core import solve_sandii

        A = _response_matrix(4, 8, seed=9)
        b = A @ np.ones(8)
        b[0] = 0.0
        x, _, _ = solve_sandii(A, b, np.ones(8))
        assert np.all(np.isfinite(x))


class TestUnfoldSandii:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_sandii(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert "effective_readings" in result
        assert "residual" in result
        assert result["method"] == "SAND-II"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_extra_output(self, detector, all_readings):
        result = detector.unfold_sandii(
            all_readings, chi_fac=0, tolerance=1e-2, save_result=False
        )
        assert result["chi_fac"] == 0
        assert result["relative_uncertainty"] == pytest.approx(0.1)

    def test_single_detector(self, detector, readings):
        result = detector.unfold_sandii(readings, save_result=False)
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_sandii(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "SAND-II"
        assert len(detector.results_history) == 1

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_sandii(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result
        assert result["montecarlo_samples"] == 10

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_sandii, unfold_sandii

        assert callable(solve_sandii)
        assert callable(unfold_sandii)
        assert hasattr(Detector, "unfold_sandii")


# ============================================================================
# BUNKI
# ============================================================================


class TestSolveBunki:
    def test_basic(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_bunki(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix(4, 8, seed=1)
        b = A @ np.ones(8)
        x1, _, _ = solve_bunki(A, b, np.ones(8))
        x2, _, _ = solve_bunki(A, b, np.ones(8))
        np.testing.assert_allclose(x1, x2)

    def test_smoothing_parameter(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix(6, 12, seed=3)
        b = A @ np.ones(12)
        for smoothing in (0.0, 0.1, 0.5):
            x, _, _ = solve_bunki(A, b, np.ones(12), smoothing=smoothing)
            assert len(x) == 12
            assert np.all(np.isfinite(x))

    def test_zero_measurements(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix(3, 6, seed=5)
        with pytest.raises(ValueError, match="positive measurements"):
            solve_bunki(A, np.zeros(3), np.ones(6))

    def test_lethargy_weights_shape_error(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix(4, 8, seed=6)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="lethargy_weights"):
            solve_bunki(A, b, np.ones(8), lethargy_weights=np.ones(5))

    def test_lethargy_weights(self):
        from bssunfold.core import solve_bunki

        A = _response_matrix(4, 8, seed=6)
        b = A @ np.ones(8)
        x, _, _ = solve_bunki(
            A, b, np.ones(8), lethargy_weights=np.ones(8) * 0.5
        )
        assert np.all(np.isfinite(x))


class TestUnfoldBunki:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_bunki(
            all_readings, max_iterations=200, save_result=False
        )
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "BUNKI"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_extra_output(self, detector, all_readings):
        result = detector.unfold_bunki(
            all_readings, smoothing=0.2, max_iterations=200, save_result=False
        )
        assert result["smoothing"] == pytest.approx(0.2)

    def test_single_detector(self, detector, readings):
        result = detector.unfold_bunki(
            readings, max_iterations=200, save_result=False
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_bunki(readings, max_iterations=200, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "BUNKI"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_bunki(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            max_iterations=200,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_bunki, unfold_bunki

        assert callable(solve_bunki)
        assert callable(unfold_bunki)
        assert hasattr(Detector, "unfold_bunki")


# ============================================================================
# BUNKI-UT
# ============================================================================


class TestSolveBunkiut:
    def test_basic(self):
        from bssunfold.core import solve_bunkiut

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_bunkiut(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_bunkiut

        A = _response_matrix(4, 8, seed=1)
        b = A @ np.ones(8)
        x1, _, _ = solve_bunkiut(A, b, np.ones(8))
        x2, _, _ = solve_bunkiut(A, b, np.ones(8))
        np.testing.assert_allclose(x1, x2)

    def test_zero_measurements(self):
        from bssunfold.core import solve_bunkiut

        A = _response_matrix(3, 6, seed=5)
        with pytest.raises(ValueError, match="positive measurements"):
            solve_bunkiut(A, np.zeros(3), np.ones(6))

    def test_lethargy_weights_shape_error(self):
        from bssunfold.core import solve_bunkiut

        A = _response_matrix(4, 8, seed=6)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="lethargy_weights"):
            solve_bunkiut(A, b, np.ones(8), lethargy_weights=np.ones(5))


class TestUnfoldBunkiut:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_bunkiut(
            all_readings, max_iterations=200, save_result=False
        )
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "BUNKI-UT"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_extra_output(self, detector, all_readings):
        result = detector.unfold_bunkiut(
            all_readings, smoothing=0.2, max_iterations=200, save_result=False
        )
        assert result["smoothing"] == pytest.approx(0.2)

    def test_single_detector(self, detector, readings):
        result = detector.unfold_bunkiut(
            readings, max_iterations=200, save_result=False
        )
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_bunkiut(readings, max_iterations=200, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "BUNKI-UT"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_bunkiut(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            max_iterations=200,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_bunkiut, unfold_bunkiut

        assert callable(solve_bunkiut)
        assert callable(unfold_bunkiut)
        assert hasattr(Detector, "unfold_bunkiut")


# ============================================================================
# OSEM
# ============================================================================


class TestSolveOsem:
    def test_basic(self):
        from bssunfold.core import solve_osem

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_osem(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_matches_normalized_mlem_single_subset(self):
        from bssunfold.core import solve_osem

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x_osem, _, _ = solve_osem(A, b, np.ones(10), max_iterations=30)

        eps = 1e-11
        x = np.ones(10)
        norm = A.sum(axis=0)
        for _ in range(30):
            ratio = b / (A @ x + eps)
            correction = A.T @ ratio
            x = x * correction / (norm + eps)
        np.testing.assert_allclose(x_osem, x, atol=1e-8)

    def test_subsets(self):
        from bssunfold.core import solve_osem

        A = _response_matrix(6, 12, seed=2)
        b = A @ np.ones(12)
        for n_subsets in (1, 2, 3):
            x, _, _ = solve_osem(A, b, np.ones(12), n_subsets=n_subsets)
            assert len(x) == 12
            assert np.all(np.isfinite(x))

    def test_invalid_n_subsets_zero(self):
        from bssunfold.core import solve_osem

        A = _response_matrix(4, 8, seed=3)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="n_subsets must be >= 1"):
            solve_osem(A, b, np.ones(8), n_subsets=0)

    def test_invalid_n_subsets_too_large(self):
        from bssunfold.core import solve_osem

        A = _response_matrix(4, 8, seed=4)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="must not exceed"):
            solve_osem(A, b, np.ones(8), n_subsets=5)


class TestUnfoldOsem:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_osem(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "OSEM"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_n_subsets(self, detector, all_readings):
        result = detector.unfold_osem(
            all_readings, n_subsets=3, save_result=False
        )
        assert "spectrum" in result
        assert result["n_subsets"] == 3

    def test_single_detector(self, detector, readings):
        result = detector.unfold_osem(readings, save_result=False)
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_osem(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "OSEM"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_osem(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_osem, unfold_osem

        assert callable(solve_osem)
        assert callable(unfold_osem)
        assert hasattr(Detector, "unfold_osem")


# ============================================================================
# MAP-EM
# ============================================================================


class TestSolveMapem:
    def test_basic(self):
        from bssunfold.core import solve_mapem

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_mapem(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_priors(self):
        from bssunfold.core import solve_mapem

        A = _response_matrix(6, 12, seed=2)
        b = A @ np.ones(12)
        for prior in ("none", "quadratic", "logcosh", "relative_difference"):
            x, _, _ = solve_mapem(A, b, np.ones(12), prior=prior)
            assert len(x) == 12
            assert np.all(np.isfinite(x))

    def test_beta_effect(self):
        from bssunfold.core import solve_mapem

        A = _response_matrix(6, 12, seed=2)
        b = A @ np.ones(12)
        x_none, _, _ = solve_mapem(A, b, np.ones(12), prior="none")
        x_reg, _, _ = solve_mapem(
            A, b, np.ones(12), prior="quadratic", beta=0.1
        )
        assert not np.allclose(x_none, x_reg)

    def test_invalid_prior(self):
        from bssunfold.core import solve_mapem

        A = _response_matrix(4, 8, seed=3)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="Unknown prior"):
            solve_mapem(A, b, np.ones(8), prior="bogus")


class TestUnfoldMapem:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_mapem(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "MAP-EM"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_priors(self, detector, all_readings):
        for prior in ("none", "quadratic", "logcosh", "relative_difference"):
            result = detector.unfold_mapem(
                all_readings, prior=prior, beta=1e-4, save_result=False
            )
            assert "spectrum" in result
            assert result["prior"] == prior

    def test_prior_value(self, detector, all_readings):
        result = detector.unfold_mapem(
            all_readings, prior="quadratic", beta=1e-4, save_result=False
        )
        assert "prior_value" in result
        assert np.isfinite(result["prior_value"])

    def test_single_detector(self, detector, readings):
        result = detector.unfold_mapem(
            readings, prior="none", save_result=False
        )
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_mapem(readings, prior="none", save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "MAP-EM"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_mapem(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_mapem, unfold_mapem

        assert callable(solve_mapem)
        assert callable(unfold_mapem)
        assert hasattr(Detector, "unfold_mapem")


# ============================================================================
# BSREM
# ============================================================================


class TestSolveBsrem:
    def test_basic(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_bsrem(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_priors(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(6, 12, seed=2)
        b = A @ np.ones(12)
        for prior in ("none", "quadratic", "logcosh", "relative_difference"):
            x, _, _ = solve_bsrem(A, b, np.ones(12), prior=prior, beta=1e-4)
            assert np.all(np.isfinite(x))

    def test_subsets(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(6, 12, seed=2)
        b = A @ np.ones(12)
        for n_subsets in (1, 2, 3):
            x, _, _ = solve_bsrem(A, b, np.ones(12), n_subsets=n_subsets)
            assert np.all(np.isfinite(x))

    def test_relaxation_float(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(6, 12, seed=3)
        b = A @ np.ones(12)
        x, _, _ = solve_bsrem(A, b, np.ones(12), relaxation=0.5)
        assert np.all(np.isfinite(x))

    def test_relaxation_callable(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(6, 12, seed=3)
        b = A @ np.ones(12)
        x, _, _ = solve_bsrem(
            A, b, np.ones(12), relaxation=lambda n: 1.0 / (1.0 + n)
        )
        assert np.all(np.isfinite(x))

    def test_addition_floor(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(6, 12, seed=4)
        b = A @ np.ones(12)
        x, _, _ = solve_bsrem(A, b, np.ones(12), addition_after_iteration=1e-3)
        assert np.all(x >= 1e-3)

    def test_invalid_prior(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(4, 8, seed=5)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="Unknown prior"):
            solve_bsrem(A, b, np.ones(8), prior="bogus")

    def test_invalid_n_subsets(self):
        from bssunfold.core import solve_bsrem

        A = _response_matrix(4, 8, seed=6)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="n_subsets must be >= 1"):
            solve_bsrem(A, b, np.ones(8), n_subsets=0)
        with pytest.raises(ValueError, match="must not exceed"):
            solve_bsrem(A, b, np.ones(8), n_subsets=5)


class TestUnfoldBsrem:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_bsrem(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "BSREM"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_priors(self, detector, all_readings):
        for prior in ("none", "quadratic", "logcosh", "relative_difference"):
            result = detector.unfold_bsrem(
                all_readings, prior=prior, beta=1e-4, save_result=False
            )
            assert "spectrum" in result
            assert result["prior"] == prior

    def test_relaxation(self, detector, all_readings):
        result = detector.unfold_bsrem(
            all_readings,
            relaxation=lambda n: 1.0 / (1.0 + n),
            save_result=False,
        )
        assert "spectrum" in result
        assert result["n_subsets"] == 1

    def test_single_detector(self, detector, readings):
        result = detector.unfold_bsrem(readings, save_result=False)
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_bsrem(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "BSREM"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_bsrem(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_bsrem, unfold_bsrem

        assert callable(solve_bsrem)
        assert callable(unfold_bsrem)
        assert hasattr(Detector, "unfold_bsrem")


# ============================================================================
# SART
# ============================================================================


class TestSolveSart:
    def test_basic(self):
        from bssunfold.core import solve_sart

        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 10))
        x, iterations, converged = solve_sart(A, b, np.ones(10))
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_sart

        A = _response_matrix(4, 8, seed=1)
        b = A @ np.ones(8)
        x1, _, _ = solve_sart(A, b, np.ones(8))
        x2, _, _ = solve_sart(A, b, np.ones(8))
        np.testing.assert_allclose(x1, x2)

    def test_relaxation_float(self):
        from bssunfold.core import solve_sart

        A = _response_matrix(6, 12, seed=3)
        b = A @ np.ones(12)
        x, _, _ = solve_sart(A, b, np.ones(12), relaxation=0.5)
        assert np.all(np.isfinite(x))

    def test_relaxation_callable(self):
        from bssunfold.core import solve_sart

        A = _response_matrix(6, 12, seed=3)
        b = A @ np.ones(12)
        x, _, _ = solve_sart(A, b, np.ones(12), relaxation=lambda n: 0.8**n)
        assert np.all(np.isfinite(x))

    def test_zero_measurements(self):
        from bssunfold.core import solve_sart

        A = _response_matrix(3, 6, seed=5)
        x, iterations, converged = solve_sart(A, np.zeros(3), np.ones(6))
        assert np.all(np.isfinite(x))
        assert iterations > 0


class TestUnfoldSart:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_sart(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "SART"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_relaxation(self, detector, all_readings):
        result = detector.unfold_sart(
            all_readings, relaxation=0.5, save_result=False
        )
        assert "spectrum" in result
        assert result["relaxation"] == pytest.approx(0.5)

    def test_single_detector(self, detector, readings):
        result = detector.unfold_sart(readings, save_result=False)
        assert "spectrum" in result

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_sart(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "SART"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_sart(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_sart, unfold_sart

        assert callable(solve_sart)
        assert callable(unfold_sart)
        assert hasattr(Detector, "unfold_sart")


# ============================================================================
# Small synthetic detector
# ============================================================================


class TestSmallDetector:
    def test_sandii_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame(
            {
                "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        d = Detector(df)
        result = d.unfold_sandii(
            {"sphere_1": 1.0, "sphere_2": 2.0}, save_result=False
        )
        assert len(result["spectrum"]) == 5
        assert np.all(np.isfinite(result["spectrum"]))

    def test_osem_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame(
            {
                "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        d = Detector(df)
        result = d.unfold_osem(
            {"sphere_1": 1.0, "sphere_2": 2.0}, save_result=False
        )
        assert len(result["spectrum"]) == 5
        assert np.all(np.isfinite(result["spectrum"]))

    def test_sart_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame(
            {
                "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        d = Detector(df)
        result = d.unfold_sart(
            {"sphere_1": 1.0, "sphere_2": 2.0}, save_result=False
        )
        assert len(result["spectrum"]) == 5
        assert np.all(np.isfinite(result["spectrum"]))

    def test_combined_pipeline(self, detector, all_readings):
        for method in (
            "sandii",
            "bunki",
            "bunkiut",
            "osem",
            "mapem",
            "bsrem",
            "sart",
        ):
            params = {"save_result": False}
            if method in ("bunki", "bunkiut"):
                params["max_iterations"] = 200
            result = detector.unfold_combined(
                all_readings,
                pipeline=[{"method": method, "params": params}],
                verbose=False,
            )
            assert "spectrum" in result
            assert result["pipeline_info"]["stages"] == [method]


class TestFirstBinZero:
    """The default initial spectrum must not pin the first (lowest-energy)
    bin to the flat-guess value of 1.

    Reference IAEA spectra have ``Phi[0] = 0``, but the detector response at
    the lowest energy bin is (near-)zero, so the all-ones initial guess was
    never corrected and the first bin stayed at 1.0. Zeroing the first bin of
    the default initial spectrum fixes it.
    """

    @pytest.mark.parametrize(
        "method",
        [
            "bsrem",
            "mapem",
            "osem",
            "sart",
            "bunki",
            "bunkiut",
            "sandii",
        ],
    )
    @pytest.mark.parametrize(
        "detector_cfg",
        [
            pytest.param("GSF", id="GSF"),
            pytest.param("PTB", id="PTB"),
            pytest.param("LANL", id="LANL"),
        ],
    )
    def test_first_bin_zero(self, detector_cfg, method):
        from bssunfold import RF_LANL, RF_PTB, Detector

        if detector_cfg == "GSF":
            det = Detector()
        elif detector_cfg == "PTB":
            det = Detector(pd.DataFrame(RF_PTB))
        else:
            det = Detector(pd.DataFrame(RF_LANL))

        csv_path = Path(__file__).parent / (
            "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
        )
        df_ref = pd.read_csv(csv_path)
        ref_phi = interpolate_spectrum(
            df_ref["ISO_ref_Cf252"].to_numpy(),
            df_ref["E_MeV"].to_numpy(),
            det.E_MeV,
        )
        A = np.array([det.sensitivities[n] for n in det.detector_names])
        b = A @ ref_phi
        readings = {
            name: float(val) for name, val in zip(det.detector_names, b)
        }

        result = getattr(det, f"unfold_{method}")(
            readings, save_result=False
        )
        first_bin = result["spectrum"][0]

        if method == "bsrem":
            # BSREM floor clamp (addition_after_iteration=1e-4) keeps the
            # first bin at a small nonzero floor value.
            assert first_bin <= 1e-3
        else:
            assert first_bin == 0.0
