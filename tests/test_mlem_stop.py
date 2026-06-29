"""Tests for unfold_mlem_stop (MLEM with J-factor early stopping)."""

import numpy as np
import pandas as pd
import pytest

from bssunfold import Detector
from bssunfold.core import solve_mlem_stop, unfold_mlem_stop
from bssunfold.constants import RF_GSF


# ---------------------------------------------------------------------------
# Core solver tests
# ---------------------------------------------------------------------------

class TestSolveMlemStop:
    """Tests for the core solve_mlem_stop function."""

    def test_basic(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x_true = np.exp(-np.arange(n_bins) / 3)
        x_true /= x_true.sum()
        b = A @ x_true + rng.normal(0, 0.001, n_det)
        x0 = np.ones(n_bins) / n_bins

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=200, cps_crossover=100.0)
        assert isinstance(sol, np.ndarray)
        assert sol.shape == (n_bins,)
        assert np.all(sol >= 0)
        assert 0 < iters <= 200
        assert isinstance(conv, (bool, np.bool_))

    def test_with_explicit_threshold(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x_true = np.exp(-np.arange(n_bins) / 3)
        x_true /= x_true.sum()
        b = A @ x_true
        x0 = np.ones(n_bins) / n_bins

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=5000, j_threshold=100.0)
        assert conv
        assert iters == 1

    def test_zero_b(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x0 = np.ones(n_bins) / n_bins
        b = np.zeros(n_det)

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=10, cps_crossover=1.0)
        assert sol.shape == (n_bins,)
        assert iters == 10
        assert not conv

    def test_zero_threshold(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x0 = np.ones(n_bins) / n_bins
        b = A @ x0
        # With b = A @ x0 the J factor is 0, so threshold=0 stops immediately
        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=5, j_threshold=0.0)
        assert iters == 1
        assert conv

    def test_high_cps_crossover(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x_true = np.exp(-np.arange(n_bins) / 3)
        x_true /= x_true.sum()
        b = A @ x_true
        x0 = np.ones(n_bins) / n_bins

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=200, cps_crossover=1e10)
        assert not conv

    def test_negative_values(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x0 = np.ones(n_bins) / n_bins
        b = -np.ones(n_det)

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=10, cps_crossover=1.0)
        assert sol.shape == (n_bins,)
        assert np.all(sol >= 0)

    def test_all_same_readings(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        b = np.ones(n_det) * 5.0
        x0 = np.ones(n_bins) / n_bins

        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=200, cps_crossover=10.0)
        assert sol.shape == (n_bins,)
        assert np.all(sol >= 0)

    def test_calculate_j_factor(self):
        from bssunfold.core.unfold_mlem_stop import calculate_j_factor

        meas = np.array([1.0, 2.0, 3.0])
        est = np.array([1.0, 2.0, 3.0])
        assert calculate_j_factor(meas, est) == 0.0

        j = calculate_j_factor(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        assert j > 0

        inf_j = calculate_j_factor(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert inf_j == float('inf')

    def test_convergence_behaviour(self):
        n_bins, n_det = 10, 3
        rng = np.random.default_rng(42)
        A = rng.random((n_det, n_bins)) * 0.1
        x_true = np.exp(-np.arange(n_bins) / 3)
        x_true /= x_true.sum()
        b = A @ x_true + rng.normal(0, 0.001, n_det)
        x0 = np.ones(n_bins) / n_bins

        # More iterations should eventually stop
        sol, iters, conv = solve_mlem_stop(A, b, x0, max_iterations=5000, cps_crossover=10.0)
        assert iters > 0


# ---------------------------------------------------------------------------
# Detector-level unfold tests
# ---------------------------------------------------------------------------

class TestUnfoldMlemStop:
    """Tests for unfold_mlem_stop via the Detector class."""

    @pytest.fixture
    def detector(self):
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        return {detector.detector_names[0]: 100.0}

    def test_basic_unfolding(self, detector, readings):
        result = detector.unfold_mlem_stop(readings, max_iterations=50, cps_crossover=100.0)
        assert 'spectrum' in result
        assert 'energy' in result
        assert result['method'] == 'MLEM-STOP'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_with_multiple_readings(self, detector):
        readings = {}
        for name in detector.detector_names[:3]:
            readings[name] = 50.0
        result = detector.unfold_mlem_stop(readings, max_iterations=100, cps_crossover=100.0)
        assert 'spectrum' in result
        assert result['method'] == 'MLEM-STOP'
        assert len(result['spectrum']) == detector.n_energy_bins

    def test_custom_j_threshold(self, detector, readings):
        result = detector.unfold_mlem_stop(readings, max_iterations=5, j_threshold=1e10)
        assert 'spectrum' in result
        assert result.get('iterations', 0) == 1

    def test_explicit_j_threshold_early_stop(self, detector, readings):
        result = detector.unfold_mlem_stop(readings, max_iterations=100, j_threshold=1e10)
        assert result.get('iterations', 0) == 1

    def test_with_errors(self, detector, readings):
        result = detector.unfold_mlem_stop(
            readings,
            max_iterations=50,
            cps_crossover=100.0,
            calculate_errors=True,
            n_montecarlo=5,
            noise_level=0.05,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result

    def test_doserates(self, detector, readings):
        result = detector.unfold_mlem_stop(readings, max_iterations=50, cps_crossover=100.0)
        assert 'doserates' in result
        assert isinstance(result['doserates'], dict)
        assert len(result['doserates']) > 0

    def test_with_initial_spectrum(self, detector, readings):
        init = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_mlem_stop(
            readings, max_iterations=50, cps_crossover=100.0, initial_spectrum=init,
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_empty_readings(self, detector):
        with pytest.raises(Exception):
            detector.unfold_mlem_stop({}, max_iterations=50)

    def test_invalid_readings(self, detector):
        with pytest.raises(Exception):
            detector.unfold_mlem_stop({'invalid_key': 1.0}, max_iterations=50)

    def test_all_readings_same(self, detector):
        readings = {}
        for name in detector.detector_names:
            readings[name] = 10.0
        result = detector.unfold_mlem_stop(readings, max_iterations=100, cps_crossover=50.0)
        assert 'spectrum' in result

    def test_result_structure(self, detector):
        readings = {detector.detector_names[0]: 100.0, detector.detector_names[1]: 50.0}
        result = detector.unfold_mlem_stop(readings, max_iterations=100, cps_crossover=100.0)
        expected_keys = {'spectrum', 'energy', 'doserates', 'method',
                         'effective_readings', 'residual', 'residual_norm',
                         'spectrum_absolute'}
        assert expected_keys.issubset(result.keys())
        assert result['method'] == 'MLEM-STOP'

    def test_with_small_detector(self):
        df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        det = Detector(df)
        readings = {'sphere_1': 10.0, 'sphere_2': 5.0}
        result = det.unfold_mlem_stop(readings, max_iterations=50, cps_crossover=100.0)
        assert 'spectrum' in result
        assert len(result['spectrum']) == 5


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


def test_unfold_mlem_stop_module_imports():
    assert solve_mlem_stop is not None
    assert unfold_mlem_stop is not None


def test_detector_unfold_mlem_stop(detector, readings):
    result = detector.unfold_mlem_stop(readings, max_iterations=50, cps_crossover=100.0)
    assert 'spectrum' in result
    assert result['method'] == 'MLEM-STOP'


def test_unfold_mlem_stop_with_errors(detector, readings):
    result = detector.unfold_mlem_stop(
        readings,
        max_iterations=50,
        cps_crossover=100.0,
        calculate_errors=True,
        n_montecarlo=5,
        noise_level=0.05,
    )
    assert 'spectrum_uncert_mean' in result
