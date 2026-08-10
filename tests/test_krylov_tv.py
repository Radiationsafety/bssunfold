"""Tests for the CGLS, GKS and Tikhonov-TV unfolding methods."""

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
# CGLS
# ============================================================================


class TestSolveCgls:
    def test_basic(self):
        from bssunfold.core import solve_cgls

        rng = np.random.default_rng(42)
        A = rng.random((5, 10))
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x, iterations, converged = solve_cgls(A, b)
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_cgls

        rng = np.random.default_rng(1)
        A = rng.random((4, 8))
        b = A @ np.ones(8)
        x1, _, _ = solve_cgls(A, b)
        x2, _, _ = solve_cgls(A, b)
        np.testing.assert_allclose(x1, x2)

    def test_zero_measurements(self):
        from bssunfold.core import solve_cgls

        A = np.random.default_rng(5).random((3, 6))
        x, iterations, converged = solve_cgls(A, np.zeros(3))
        assert np.all(x == 0)
        assert iterations == 0
        assert converged is True

    def test_discrepancy_stopping(self):
        from bssunfold.core import solve_cgls

        rng = np.random.default_rng(7)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, converged = solve_cgls(A, b, noise_level=0.5)
        assert len(x) == 12
        assert iterations > 0
        assert converged is True

    def test_tikhonov_regularization(self):
        from bssunfold.core import solve_cgls

        rng = np.random.default_rng(8)
        A = rng.random((5, 10))
        b = A @ np.ones(10)
        x, iterations, _ = solve_cgls(
            A, b, regularization=1e-3, smoothness_order=1
        )
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))

    def test_invalid_smoothness_order(self):
        from bssunfold.core import solve_cgls

        A = np.random.default_rng(9).random((4, 8))
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="smoothness_order"):
            solve_cgls(A, b, regularization=1e-3, smoothness_order=3)

    def test_zero_search_direction(self):
        from bssunfold.core import solve_cgls

        A = np.zeros((3, 5))
        b = np.ones(3)
        x, iterations, _ = solve_cgls(A, b, max_iterations=10)
        assert np.all(x == 0)
        assert iterations == 0

    def test_identity_regularization(self):
        from bssunfold.core import solve_cgls

        rng = np.random.default_rng(21)
        A = rng.random((5, 10))
        b = A @ np.ones(10)
        x, _, _ = solve_cgls(A, b, regularization=1e-3, smoothness_order=0)
        assert np.all(np.isfinite(x))

    def test_initial_spectrum_shape_error(self):
        from bssunfold.core import solve_cgls

        A = np.random.default_rng(10).random((4, 8))
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="Initial spectrum"):
            solve_cgls(A, b, x0=np.ones(5))


class TestUnfoldCgls:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_cgls(all_readings, save_result=False)
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'doserates' in result
        assert 'effective_readings' in result
        assert 'residual' in result
        assert result['method'] == 'CGLS'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_regularization(self, detector, all_readings):
        result = detector.unfold_cgls(
            all_readings,
            regularization=1e-4,
            smoothness_order=2,
            save_result=False,
        )
        assert 'spectrum' in result
        assert result['regularization'] == pytest.approx(1e-4)
        assert result['smoothness_order'] == 2

    def test_single_detector(self, detector, readings):
        result = detector.unfold_cgls(readings, save_result=False)
        assert 'spectrum' in result
        assert len(result['spectrum']) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_cgls(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result['method'] == 'CGLS'
        assert len(detector.results_history) == 1

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_cgls(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert 'spectrum_uncert_mean' in result
        assert result['montecarlo_samples'] == 10

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_cgls, unfold_cgls

        assert callable(solve_cgls)
        assert callable(unfold_cgls)
        assert hasattr(Detector, 'unfold_cgls')


# ============================================================================
# GKS
# ============================================================================


class TestSolveGks:
    def test_basic(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(42)
        A = rng.random((5, 10))
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x, iterations, converged = solve_gks(A, b)
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(1)
        A = rng.random((4, 8))
        b = A @ np.ones(8)
        x1, _, _ = solve_gks(A, b)
        x2, _, _ = solve_gks(A, b)
        np.testing.assert_allclose(x1, x2)

    def test_zero_measurements(self):
        from bssunfold.core import solve_gks

        A = np.random.default_rng(5).random((3, 6))
        x, iterations, converged = solve_gks(A, np.zeros(3))
        assert np.all(x == 0)
        assert iterations == 0
        assert converged is True

    def test_discrepancy_principle(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(11)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, converged = solve_gks(
            A, b, regularization_method='dp', noise_level=0.5
        )
        assert len(x) == 12
        assert iterations > 0
        assert np.all(np.isfinite(x))

    def test_lcurve(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(12)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, converged = solve_gks(
            A, b, regularization_method='lcurve'
        )
        assert len(x) == 12
        assert iterations > 0

    def test_manual(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(13)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, converged = solve_gks(
            A, b, regularization_method='manual', regularization=1e-4
        )
        assert len(x) == 12
        assert iterations > 0

    def test_invalid_method(self):
        from bssunfold.core import solve_gks

        A = np.random.default_rng(14).random((4, 8))
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="regularization method"):
            solve_gks(A, b, regularization_method='bogus')

    def test_invalid_smoothness_order(self):
        from bssunfold.core import solve_gks

        A = np.random.default_rng(22).random((4, 8))
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="smoothness_order"):
            solve_gks(A, b, smoothness_order=3)

    def test_dp_default_noise_level(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(23)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, iterations, _ = solve_gks(A, b, regularization_method='dp')
        assert len(x) == 12
        assert iterations > 0

    def test_alpha_collapse(self):
        from bssunfold.core import solve_gks

        A = np.zeros((3, 5))
        b = np.ones(3)
        x, iterations, converged = solve_gks(A, b)
        assert np.all(x == 0)
        assert converged is True

    def test_nonfinite_lambda_fallback(self):
        from unittest.mock import patch

        from bssunfold.core import solve_gks

        A = np.random.default_rng(24).random((4, 8))
        b = A @ np.ones(8)
        with patch(
            "bssunfold.core.unfold_gks._projected_gcv", return_value=np.nan
        ):
            x, _, _ = solve_gks(A, b, regularization=1e-4)
        assert np.all(np.isfinite(x))

    def test_smoothness_order(self):
        from bssunfold.core import solve_gks

        rng = np.random.default_rng(15)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        for order in (0, 1, 2):
            x, _, _ = solve_gks(A, b, smoothness_order=order)
            assert len(x) == 12

    def test_square_system_collapse(self):
        from bssunfold.core import solve_gks

        A = np.eye(5)
        b = np.ones(5)
        x, iterations, converged = solve_gks(A, b)
        assert len(x) == 5
        assert iterations >= 1
        assert np.all(np.isfinite(x))


class TestUnfoldGks:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_gks(all_readings, save_result=False)
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'doserates' in result
        assert 'effective_readings' in result
        assert 'residual' in result
        assert result['method'] == 'GKS'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_regularization_methods(self, detector, all_readings):
        for method in ('gcv', 'dp', 'lcurve', 'manual'):
            result = detector.unfold_gks(
                all_readings,
                regularization_method=method,
                noise_level=0.01,
                save_result=False,
            )
            assert 'spectrum' in result
            assert result['regularization_method'] == method

    def test_single_detector(self, detector, readings):
        result = detector.unfold_gks(readings, save_result=False)
        assert 'spectrum' in result
        assert len(result['spectrum']) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_gks(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result['method'] == 'GKS'

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_gks(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert 'spectrum_uncert_mean' in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_gks, unfold_gks

        assert callable(solve_gks)
        assert callable(unfold_gks)
        assert hasattr(Detector, 'unfold_gks')


# ============================================================================
# Tikhonov-TV
# ============================================================================


class TestSolveTikhonovTv:
    def test_basic(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(42)
        A = rng.random((5, 10))
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x, iterations, converged = solve_tikhonov_tv(A, b, max_iterations=50)
        assert len(x) == 10
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_deterministic(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(1)
        A = rng.random((4, 8))
        b = A @ np.ones(8)
        x1, _, _ = solve_tikhonov_tv(A, b, max_iterations=30)
        x2, _, _ = solve_tikhonov_tv(A, b, max_iterations=30)
        np.testing.assert_allclose(x1, x2)

    def test_zero_measurements(self):
        from bssunfold.core import solve_tikhonov_tv

        A = np.random.default_rng(5).random((3, 6))
        x, iterations, converged = solve_tikhonov_tv(
            A, np.zeros(3), max_iterations=30
        )
        assert np.all(x == 0)
        assert converged in (True, False)

    def test_types(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(16)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        for type_ in ('TT', 'TV', 'T'):
            x, _, _ = solve_tikhonov_tv(A, b, type_=type_, max_iterations=30)
            assert len(x) == 12
            assert np.all(np.isfinite(x))

    def test_invalid_type(self):
        from bssunfold.core import solve_tikhonov_tv

        A = np.random.default_rng(17).random((4, 8))
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="type_"):
            solve_tikhonov_tv(A, b, type_='BOGUS', max_iterations=10)

    def test_adaptive_beta(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(18)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, _, _ = solve_tikhonov_tv(A, b, beta='adapt', max_iterations=30)
        assert len(x) == 12
        assert np.all(np.isfinite(x))

    def test_epsilon_parameter(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(19)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        noise = 0.01 * np.linalg.norm(b)
        x, _, _ = solve_tikhonov_tv(
            A, b, epsilon=noise**2, max_iterations=30
        )
        assert len(x) == 12
        assert np.all(np.isfinite(x))

    def test_mu_parameter(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(20)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, _, _ = solve_tikhonov_tv(A, b, mu=(10.0, 1.0, 1.0), max_iterations=30)
        assert len(x) == 12

    def test_beta_zero_g2(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(25)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, _, _ = solve_tikhonov_tv(A, b, type_='TV', beta=0.0,
                                    max_iterations=30)
        assert len(x) == 12
        assert np.all(np.isfinite(x))

    def test_singular_system_pinv(self):
        from bssunfold.core import solve_tikhonov_tv

        A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        b = np.array([1.0, 1.0])
        x, _, _ = solve_tikhonov_tv(A, b, max_iterations=10)
        assert len(x) == 3
        assert np.all(np.isfinite(x))

    def test_zscore_empty(self):
        from bssunfold.core.unfold_tikhonov_tv import _zscore_max

        assert _zscore_max(np.array([])) == 0.0

    def test_quadratic_matrix_collapse(self):
        from bssunfold.core import solve_tikhonov_tv

        A = np.eye(4)
        b = np.ones(4)
        x, _, _ = solve_tikhonov_tv(A, b, max_iterations=10)
        assert len(x) == 4
        assert np.all(np.isfinite(x))

    def test_tt_beta_zero_g2_identity(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(26)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        x, _, _ = solve_tikhonov_tv(A, b, type_='TT', beta=0.0,
                                    max_iterations=30)
        assert len(x) == 12
        assert np.all(np.isfinite(x))

    def test_adaptive_beta_non_tt(self):
        from bssunfold.core import solve_tikhonov_tv

        rng = np.random.default_rng(27)
        A = rng.random((6, 12))
        b = A @ np.ones(12)
        for type_ in ('TV', 'T'):
            x, _, _ = solve_tikhonov_tv(A, b, type_=type_, beta='adapt',
                                        max_iterations=30)
            assert len(x) == 12
            assert np.all(np.isfinite(x))


class TestUnfoldTikhonovTv:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_tikhonov_tv(
            all_readings, max_iterations=30, save_result=False
        )
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'doserates' in result
        assert 'effective_readings' in result
        assert 'residual' in result
        assert result['method'] == 'TikhonovTV'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_types(self, detector, all_readings):
        for type_ in ('TT', 'TV', 'T'):
            result = detector.unfold_tikhonov_tv(
                all_readings, type_=type_, max_iterations=30, save_result=False
            )
            assert 'spectrum' in result
            assert result['type_'] == type_

    def test_adaptive_beta(self, detector, all_readings):
        result = detector.unfold_tikhonov_tv(
            all_readings, beta='adapt', max_iterations=30, save_result=False
        )
        assert 'spectrum' in result

    def test_noise_level_epsilon(self, detector, all_readings):
        result = detector.unfold_tikhonov_tv(
            all_readings,
            noise_level=0.01,
            max_iterations=30,
            save_result=False,
        )
        assert 'spectrum' in result
        assert result['epsilon'] is not None

    def test_single_detector(self, detector, readings):
        result = detector.unfold_tikhonov_tv(
            readings, max_iterations=30, save_result=False
        )
        assert 'spectrum' in result
        assert len(result['spectrum']) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_tikhonov_tv(readings, max_iterations=30, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result['method'] == 'TikhonovTV'

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_tikhonov_tv(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            max_iterations=30,
            save_result=False,
        )
        assert 'spectrum_uncert_mean' in result

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import solve_tikhonov_tv, unfold_tikhonov_tv

        assert callable(solve_tikhonov_tv)
        assert callable(unfold_tikhonov_tv)
        assert hasattr(Detector, 'unfold_tikhonov_tv')


# ============================================================================
# Small synthetic detector
# ============================================================================


class TestSmallDetector:
    def test_cgls_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        d = Detector(df)
        result = d.unfold_cgls(
            {'sphere_1': 1.0, 'sphere_2': 2.0}, save_result=False
        )
        assert len(result['spectrum']) == 5
        assert np.all(np.isfinite(result['spectrum']))

    def test_gks_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        d = Detector(df)
        result = d.unfold_gks(
            {'sphere_1': 1.0, 'sphere_2': 2.0}, save_result=False
        )
        assert len(result['spectrum']) == 5
        assert np.all(np.isfinite(result['spectrum']))

    def test_tikhonov_tv_small_detector(self):
        from bssunfold import Detector

        df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        d = Detector(df)
        result = d.unfold_tikhonov_tv(
            {'sphere_1': 1.0, 'sphere_2': 2.0}, max_iterations=30,
            save_result=False,
        )
        assert len(result['spectrum']) == 5
        assert np.all(np.isfinite(result['spectrum']))

    def test_combined_pipeline(self, detector, all_readings):
        for method in ('cgls', 'gks', 'tikhonov_tv'):
            result = detector.unfold_combined(
                all_readings,
                pipeline=[{'method': method, 'params': {'save_result': False}}],
                verbose=False,
            )
            assert 'spectrum' in result
            assert result['pipeline_info']['stages'] == [method]
