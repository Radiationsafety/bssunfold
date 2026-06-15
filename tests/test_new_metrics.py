"""Tests for new EURADOS-style metrics and parametric unfolding methods."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from bssunfold import Detector
from bssunfold.utils.comparison import (
    fluence_difference_percent,
    energy_group_fluence_diff,
    dose_difference_percent,
    fluence_averaged_energy_diff,
    dose_averaged_energy_diff,
    spectral_shape_similarity,
    log_lethargy_correlation,
    peak_location_error,
    peak_width_error,
    dose_weighted_error,
    response_matrix_consistency,
    compare_eurados,
)


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def energy_grid():
    """Logarithmic energy grid from 1e-9 to 20 MeV."""
    return np.logspace(-9, 1, 100)


@pytest.fixture
def identical_spectra(energy_grid):
    """Two identical spectra."""
    E = energy_grid
    spectrum = np.ones_like(E) / len(E)
    return spectrum, spectrum.copy()


@pytest.fixture
def scaled_spectra(energy_grid):
    """Two spectra with different total fluence but same shape."""
    E = energy_grid
    s1 = np.ones_like(E) / len(E)
    s2 = 2.0 * s1
    return s1, s2


@pytest.fixture
def shifted_spectra(energy_grid):
    """Two spectra with shifted peaks."""
    E = energy_grid
    s1 = np.exp(-((np.log10(E) + 3) ** 2) / 0.5)
    s2 = np.exp(-((np.log10(E) + 2.5) ** 2) / 0.5)
    return s1, s2


@pytest.fixture
def detector():
    """Default Detector instance."""
    return Detector()


@pytest.fixture
def sample_readings():
    """Sample detector readings."""
    return {name: float(1.0 + i * 0.1) for i, name in enumerate(Detector().detector_names)}


# ─── Test EURADOS metrics ─────────────────────────────────────────


class TestFluenceDifference:
    def test_identical(self, identical_spectra):
        s1, s2 = identical_spectra
        assert_almost_equal(fluence_difference_percent(s1, s2), 0.0)

    def test_scaled(self, scaled_spectra):
        s1, s2 = scaled_spectra
        assert_almost_equal(fluence_difference_percent(s1, s2), 100.0)

    def test_energy_bins(self, energy_grid):
        s1 = np.ones_like(energy_grid)
        s2 = 2.0 * s1
        bins = np.diff(np.concatenate([energy_grid, [energy_grid[-1] * 10]]))
        assert_almost_equal(fluence_difference_percent(s1, s2, bins[:len(s1)]), 100.0)


class TestEnergyGroupFluenceDiff:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        result = energy_group_fluence_diff(s1, s2, energy_grid)
        assert_almost_equal(result["thermal"], 0.0)
        assert_almost_equal(result["epithermal"], 0.0)
        assert_almost_equal(result["fast"], 0.0)

    def test_thermal_only(self, energy_grid):
        s1 = np.where(energy_grid < 0.4e-6, 1.0, 0.0)
        s2 = 2.0 * s1
        result = energy_group_fluence_diff(s1, s2, energy_grid)
        assert_almost_equal(result["thermal"], 100.0)
        assert_almost_equal(result["epithermal"], 0.0)
        assert_almost_equal(result["fast"], 0.0)


class TestDoseDifference:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(dose_difference_percent(s1, s2, energy_grid), 0.0)


class TestFluenceAveragedEnergy:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(fluence_averaged_energy_diff(s1, s2, energy_grid), 0.0)


class TestDoseAveragedEnergy:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(dose_averaged_energy_diff(s1, s2, energy_grid), 0.0)


class TestSpectralShapeSimilarity:
    def test_identical(self, identical_spectra):
        s1, s2 = identical_spectra
        assert_almost_equal(spectral_shape_similarity(s1, s2), 1.0)

    def test_scaled(self, scaled_spectra):
        s1, s2 = scaled_spectra
        assert_almost_equal(spectral_shape_similarity(s1, s2), 1.0)


class TestLogLethargyCorrelation:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(log_lethargy_correlation(s1, s2, energy_grid), 1.0)


class TestPeakLocationError:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(peak_location_error(s1, s2, energy_grid), 0.0)


class TestPeakWidthError:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(peak_width_error(s1, s2, energy_grid), 0.0)


class TestDoseWeightedError:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        assert_almost_equal(dose_weighted_error(s1, s2, energy_grid), 0.0)


class TestResponseMatrixConsistency:
    def test_perfect(self, detector, sample_readings):
        A = np.array([detector.sensitivities[name] for name in detector.detector_names])
        spectrum = np.ones(detector.n_energy_bins) / detector.n_energy_bins
        readings = A @ spectrum
        chi2 = response_matrix_consistency(spectrum, readings, A)
        assert_almost_equal(chi2, 0.0, decimal=5)


class TestCompareEurados:
    def test_identical(self, identical_spectra, energy_grid):
        s1, s2 = identical_spectra
        result = compare_eurados(s1, s2, energy_grid)
        assert "fluence_difference_percent" in result
        assert "dose_difference_percent" in result
        assert_almost_equal(result["fluence_difference_percent"], 0.0)
        assert_almost_equal(result["spectral_shape_similarity"], 1.0)


# ─── Test parametric unfolding methods ─────────────────────────────


class TestFruitLike:
    def test_import(self):
        from bssunfold.core.unfold_fruit_like import solve_fruit_like, unfold_fruit_like
        assert callable(solve_fruit_like)
        assert callable(unfold_fruit_like)

    def test_parametric_model(self):
        from bssunfold.core.unfold_fruit_like import parametric_model
        E = np.logspace(-9, 1, 50)
        spectrum = parametric_model(E, 1e-6, 0.025e-6, 1e-6, 1e-6, 2.0)
        assert len(spectrum) == len(E)
        assert np.all(spectrum >= 0)

    def test_detector_method(self, detector, sample_readings):
        result = detector.unfold_fruit_like(sample_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert len(result["spectrum"]) == detector.n_energy_bins


class TestHybridParametric:
    def test_import(self):
        from bssunfold.core.unfold_hybrid_parametric import solve_hybrid_parametric, unfold_hybrid_parametric
        assert callable(solve_hybrid_parametric)
        assert callable(unfold_hybrid_parametric)

    def test_detector_method_landweber(self, detector, sample_readings):
        result = detector.unfold_hybrid_parametric(
            sample_readings, refinement_method="landweber", save_result=False
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_detector_method_mlem(self, detector, sample_readings):
        result = detector.unfold_hybrid_parametric(
            sample_readings, refinement_method="mlem", save_result=False
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins


class TestBayesianParametric:
    def test_import(self):
        from bssunfold.core.unfold_bayesian_parametric import solve_bayesian_parametric, unfold_bayesian_parametric
        assert callable(solve_bayesian_parametric)
        assert callable(unfold_bayesian_parametric)

    def test_detector_method(self, detector, sample_readings):
        result = detector.unfold_bayesian_parametric(
            sample_readings, n_samples=100, burn_in=20, save_result=False
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins
