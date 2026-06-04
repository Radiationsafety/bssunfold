"""Tests for newly added unfolding methods and features.

These tests verify:
1. unfold_doroshenko - Doroshenko coordinate update method
2. unfold_kaczmarz - Kaczmarz ART algorithm
3. unfold_lmfit - lmfit optimization with L1/L2/Elastic Net
4. unfold_combined - pipeline of multiple methods
5. unfold_mlem_odl - MLEM via ODL
6. plot_response_functions - response function visualization
7. plot_with_uncertainty - spectrum plot with uncertainty
8. compare_regularization_methods - regularization comparison
9. randomization_experiment - randomization experiments
10. Direct calls to solve_* functions from core.unfolding_methods
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def detector():
    """Create a Detector instance with default GSF response functions."""
    from bssunfold import Detector
    det = Detector()
    return det


@pytest.fixture
def readings(detector):
    """Create readings dict using first detector name."""
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def small_detector():
    """Create a small Detector with synthetic data for fast tests."""
    from bssunfold import Detector
    df = pd.DataFrame({
        'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        'sphere_3': [0.01, 0.02, 0.03, 0.04, 0.05],
    })
    return Detector(df)


@pytest.fixture
def small_readings(small_detector):
    """Create readings for small detector."""
    return {name: 10.0 for name in small_detector.detector_names}


# =============================================================================
# unfold_doroshenko tests
# =============================================================================

class TestUnfoldDoroshenko:
    """Tests for unfold_doroshenko method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic Doroshenko unfolding."""
        result = detector.unfold_doroshenko(readings, max_iterations=200)
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'iterations' in result
        assert 'converged' in result
        assert result['method'] == 'Doroshenko'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_with_initial_spectrum(self, detector, readings):
        """Test with custom initial spectrum."""
        initial = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_doroshenko(
            readings, initial_spectrum=initial, max_iterations=200
        )
        assert 'spectrum' in result
        assert len(result['spectrum']) == detector.n_energy_bins

    def test_with_regularization(self, detector, readings):
        """Test with regularization parameter."""
        result = detector.unfold_doroshenko(
            readings, regularization=1e-4, max_iterations=200
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_with_uncertainty(self, detector, readings):
        """Test uncertainty calculation via Monte-Carlo."""
        result = detector.unfold_doroshenko(
            readings,
            max_iterations=200,
            calculate_errors=True,
            n_montecarlo=10,
            noise_level=0.05,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert 'spectrum_uncert_min' in result
        assert 'spectrum_uncert_max' in result
        assert 'spectrum_uncert_median' in result
        assert 'montecarlo_samples' in result
        assert result['montecarlo_samples'] == 10

    def test_save_result_false(self, detector, readings):
        """Test with save_result=False."""
        result = detector.unfold_doroshenko(
            readings, max_iterations=200, save_result=False
        )
        assert 'spectrum' in result
        # Result should not be in history
        assert len(detector.list_results()) == 0

    def test_convergence(self, detector, readings):
        """Test that method converges."""
        result = detector.unfold_doroshenko(
            readings, max_iterations=500, tolerance=1e-4
        )
        assert result['converged'] is True
        assert result['iterations'] <= 500

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.unfold_doroshenko(
            small_readings, max_iterations=200
        )
        assert 'spectrum' in result
        assert len(result['spectrum']) == small_detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)


# =============================================================================
# unfold_kaczmarz tests
# =============================================================================

class TestUnfoldKaczmarz:
    """Tests for unfold_kaczmarz method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic Kaczmarz unfolding."""
        result = detector.unfold_kaczmarz(readings, max_iterations=200)
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'iterations' in result
        assert 'converged' in result
        assert result['method'] == 'Kaczmarz'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_with_omega(self, detector, readings):
        """Test with different relaxation parameter."""
        result = detector.unfold_kaczmarz(
            readings, omega=0.5, max_iterations=200
        )
        assert 'spectrum' in result
        assert result.get('omega') == 0.5

    def test_with_initial_spectrum(self, detector, readings):
        """Test with custom initial spectrum."""
        initial = np.ones(detector.n_energy_bins) * 0.1
        result = detector.unfold_kaczmarz(
            readings, initial_spectrum=initial, max_iterations=200
        )
        assert 'spectrum' in result

    def test_with_uncertainty(self, detector, readings):
        """Test uncertainty calculation."""
        result = detector.unfold_kaczmarz(
            readings,
            max_iterations=200,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert 'montecarlo_samples' in result

    def test_save_result_false(self, detector, readings):
        """Test with save_result=False."""
        result = detector.unfold_kaczmarz(
            readings, max_iterations=200, save_result=False
        )
        assert 'spectrum' in result
        assert len(detector.list_results()) == 0

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.unfold_kaczmarz(
            small_readings, max_iterations=200
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


# =============================================================================
# unfold_lmfit tests
# =============================================================================

class TestUnfoldLmfit:
    """Tests for unfold_lmfit method."""

    def test_basic_elastic(self, detector, readings):
        """Test basic lmfit unfolding with elastic net."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
            regularization2=1e-4,
            l1_weight=0.5,
        )
        assert 'spectrum' in result
        assert 'energy' in result
        assert result['method'].startswith('lmfit')
        assert result['model_name'] == 'elastic'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_lasso_model(self, detector, readings):
        """Test lmfit with L1 (lasso) regularization."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='lasso',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert result['model_name'] == 'lasso'
        assert np.all(result['spectrum'] >= 0)

    def test_ridge_model(self, detector, readings):
        """Test lmfit with L2 (ridge) regularization."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='ridge',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert result['model_name'] == 'ridge'
        assert np.all(result['spectrum'] >= 0)

    def test_with_uncertainty(self, detector, readings):
        """Test uncertainty calculation."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert 'montecarlo_samples' in result

    def test_save_result_false(self, detector, readings):
        """Test with save_result=False."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
            save_result=False,
        )
        assert 'spectrum' in result
        assert len(detector.list_results()) == 0

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.unfold_lmfit(
            small_readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


# =============================================================================
# unfold_combined tests
# =============================================================================

class TestUnfoldCombined:
    """Tests for unfold_combined method."""

    def test_two_stage_pipeline(self, detector, readings):
        """Test combined unfolding with two methods."""
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
            {'method': 'landweber', 'params': {'max_iterations': 50}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert 'spectrum' in result
        assert 'pipeline_info' in result
        assert result['pipeline_info']['stages'] == ['cvxpy', 'landweber']
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_three_stage_pipeline(self, detector, readings):
        """Test combined unfolding with three methods."""
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
            {'method': 'landweber', 'params': {'max_iterations': 50}},
            {'method': 'mlem', 'params': {'max_iterations': 50}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert 'spectrum' in result
        assert len(result['pipeline_info']['stages']) == 3

    def test_with_intermediate_storage(self, detector, readings):
        """Test combined unfolding with intermediate results."""
        pipeline = [
            {
                'method': 'cvxpy',
                'params': {'regularization': 1e-3},
                'store_intermediate': True,
            },
            {
                'method': 'landweber',
                'params': {'max_iterations': 50},
                'store_intermediate': True,
            },
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert 'intermediate_results' in result
        assert 'stage_1_cvxpy' in result['intermediate_results']
        assert 'stage_2_landweber' in result['intermediate_results']

    def test_without_use_as_initial(self, detector, readings):
        """Test combined unfolding without passing result to next stage."""
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
            {
                'method': 'landweber',
                'params': {'max_iterations': 50},
                'use_as_initial': False,
            },
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert 'spectrum' in result

    def test_invalid_method(self, detector, readings):
        """Test that invalid method raises ValueError."""
        pipeline = [
            {'method': 'nonexistent_method', 'params': {}},
        ]
        with pytest.raises(ValueError, match="not found in Detector class"):
            detector.unfold_combined(readings, pipeline=pipeline)

    def test_small_system(self, small_detector, small_readings):
        """Test combined unfolding with small system."""
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
            {'method': 'landweber', 'params': {'max_iterations': 50}},
        ]
        result = small_detector.unfold_combined(
            small_readings, pipeline=pipeline
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


# =============================================================================
# unfold_mlem_odl tests
# =============================================================================

class TestUnfoldMlemOdl:
    """Tests for unfold_mlem_odl method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic MLEM-ODL unfolding."""
        result = detector.unfold_mlem_odl(readings, max_iterations=50)
        assert 'spectrum' in result
        assert 'energy' in result
        assert result['method'] == 'MLEM (ODL)'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_with_initial_spectrum(self, detector, readings):
        """Test with custom initial spectrum."""
        initial = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_mlem_odl(
            readings, initial_spectrum=initial, max_iterations=50
        )
        assert 'spectrum' in result

    def test_with_uncertainty(self, detector, readings):
        """Test uncertainty calculation."""
        result = detector.unfold_mlem_odl(
            readings,
            max_iterations=50,
            calculate_errors=True,
            n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert 'montecarlo_samples' in result

    def test_save_result_false(self, detector, readings):
        """Test with save_result=False."""
        result = detector.unfold_mlem_odl(
            readings, max_iterations=50, save_result=False
        )
        assert 'spectrum' in result
        assert len(detector.list_results()) == 0

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.unfold_mlem_odl(
            small_readings, max_iterations=50
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


# =============================================================================
# plot_response_functions tests
# =============================================================================

class TestPlotResponseFunctions:
    """Tests for plot_response_functions method."""

    def test_plot_creates_figure(self, detector, tmp_path):
        """Test that plot creates a figure."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        save_path = str(tmp_path / "response_functions.png")
        detector.plot_response_functions(
            save_to=save_path, show=False
        )
        assert tmp_path.joinpath("response_functions.png").exists()

    def test_plot_without_save(self, detector):
        """Test plot without saving."""
        import matplotlib
        matplotlib.use('Agg')
        # Should not raise any error
        detector.plot_response_functions(show=False)

    def test_plot_invalid_extension(self, detector, tmp_path):
        """Test that invalid extension raises ValueError."""
        save_path = str(tmp_path / "response_functions.txt")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detector.plot_response_functions(
                save_to=save_path, show=False
            )


# =============================================================================
# plot_with_uncertainty tests
# =============================================================================

class TestPlotWithUncertainty:
    """Tests for plot_with_uncertainty method."""

    def test_plot_without_uncertainty(self, detector, readings, tmp_path):
        """Test plot without uncertainty data."""
        import matplotlib
        matplotlib.use('Agg')

        result = detector.unfold_cvxpy(readings, regularization=1e-3)
        save_path = str(tmp_path / "spectrum.png")
        fig, ax = detector.plot_with_uncertainty(
            result, save_to=save_path, show=False
        )
        assert fig is not None
        assert ax is not None
        assert tmp_path.joinpath("spectrum.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty(self, detector, readings, tmp_path):
        """Test plot with uncertainty data."""
        import matplotlib
        matplotlib.use('Agg')

        result = detector.unfold_doroshenko(
            readings,
            max_iterations=200,
            calculate_errors=True,
            n_montecarlo=10,
        )
        save_path = str(tmp_path / "spectrum_uncert.png")
        fig, ax = detector.plot_with_uncertainty(
            result, save_to=save_path, show=False
        )
        assert fig is not None
        assert ax is not None
        assert tmp_path.joinpath("spectrum_uncert.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_reference(self, detector, readings, tmp_path):
        """Test plot with reference spectrum."""
        import matplotlib
        matplotlib.use('Agg')

        result = detector.unfold_cvxpy(readings, regularization=1e-3)
        reference = {
            'E_MeV': result['energy'],
            'Phi': result['spectrum'] * 1.1,  # Slightly different
        }
        save_path = str(tmp_path / "spectrum_ref.png")
        fig, ax = detector.plot_with_uncertainty(
            result, reference_spectrum=reference,
            save_to=save_path, show=False
        )
        assert fig is not None
        assert ax is not None
        assert tmp_path.joinpath("spectrum_ref.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# compare_regularization_methods tests
# =============================================================================

class TestCompareRegularizationMethods:
    """Tests for compare_regularization_methods method."""

    def test_basic_comparison(self, detector, readings):
        """Test basic regularization comparison."""
        result = detector.compare_regularization_methods(readings)
        assert isinstance(result, dict)
        # Returns keys: lcurve, dp, gcv, all_data, selected
        assert 'lcurve' in result
        assert 'dp' in result
        assert 'gcv' in result
        assert 'selected' in result

    def test_comparison_with_noise_var(self, detector, readings):
        """Test comparison with specified noise variance."""
        result = detector.compare_regularization_methods(
            readings, noise_var=0.01
        )
        assert isinstance(result, dict)
        assert 'selected' in result

    def test_comparison_with_plot(self, detector, readings, tmp_path):
        """Test comparison with plot generation."""
        import matplotlib
        matplotlib.use('Agg')

        save_path = str(tmp_path / "reg_comparison.png")
        result = detector.compare_regularization_methods(
            readings, plot=True, plot_path=save_path
        )
        assert isinstance(result, dict)
        # Plot may or may not be saved depending on implementation
        if tmp_path.joinpath("reg_comparison.png").exists():
            assert tmp_path.joinpath("reg_comparison.png").stat().st_size > 0

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.compare_regularization_methods(
            small_readings
        )
        assert isinstance(result, dict)
        assert 'selected' in result


# =============================================================================
# randomization_experiment tests
# =============================================================================

class TestRandomizationExperiment:
    """Tests for randomization_experiment method."""

    def test_basic_experiment(self, detector, readings):
        """Test basic randomization experiment."""
        result = detector.randomization_experiment(
            readings, n_samples=5
        )
        assert isinstance(result, dict)
        # Returns keys for each method: lcurve, dp, gcv, lcurve_full
        # Each contains: lambdas, mean, std, median, min, max, cv
        assert 'lcurve' in result or 'dp' in result or 'gcv' in result

    def test_experiment_with_noise_var(self, detector, readings):
        """Test with specified noise variance."""
        result = detector.randomization_experiment(
            readings, noise_var=0.01, n_samples=5
        )
        assert isinstance(result, dict)
        # Should have at least one method result
        assert len(result) > 0

    def test_small_system(self, small_detector, small_readings):
        """Test with small synthetic system."""
        result = small_detector.randomization_experiment(
            small_readings, n_samples=5
        )
        assert isinstance(result, dict)
        assert len(result) > 0


# =============================================================================
# Direct function calls from core.unfolding_methods
# =============================================================================

class TestDirectSolveFunctions:
    """Tests for direct solve_* function calls."""

    def test_solve_doroshenko_direct(self):
        """Test solve_doroshenko function directly."""
        from bssunfold.core.unfolding_methods import solve_doroshenko

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, n_iter, converged = solve_doroshenko(
            A, b, x0, max_iterations=500, tolerance=1e-6
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)
        assert converged is True
        assert n_iter <= 500

    def test_solve_kaczmarz_direct(self):
        """Test solve_kaczmarz function directly."""
        from bssunfold.core.unfolding_methods import solve_kaczmarz

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.zeros(10)

        x_opt, n_iter, converged = solve_kaczmarz(
            A, b, x0, max_iterations=500, tolerance=1e-6
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)
        assert converged is True

    def test_solve_lmfit_direct(self):
        """Test solve_lmfit function directly."""
        from bssunfold.core.unfolding_methods import solve_lmfit

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, success, message, nfev = solve_lmfit(
            A, b, x0, method='lbfgsb', model_name='elastic',
            regularization=1e-4, regularization2=1e-4, l1_weight=0.5
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)
        assert success is True

    def test_solve_lmfit_lasso(self):
        """Test solve_lmfit with lasso regularization."""
        from bssunfold.core.unfolding_methods import solve_lmfit

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, success, message, nfev = solve_lmfit(
            A, b, x0, method='lbfgsb', model_name='lasso',
            regularization=1e-4
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)
        assert success is True

    def test_solve_lmfit_ridge(self):
        """Test solve_lmfit with ridge regularization."""
        from bssunfold.core.unfolding_methods import solve_lmfit

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, success, message, nfev = solve_lmfit(
            A, b, x0, method='lbfgsb', model_name='ridge',
            regularization=1e-4
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)
        assert success is True

    def test_solve_doroshenko_with_x0(self):
        """Test solve_doroshenko with initial guess."""
        from bssunfold.core.unfolding_methods import solve_doroshenko

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, n_iter, converged = solve_doroshenko(
            A, b, x0=x0, max_iterations=500, tolerance=1e-6
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)

    def test_solve_kaczmarz_with_omega(self):
        """Test solve_kaczmarz with relaxation parameter."""
        from bssunfold.core.unfolding_methods import solve_kaczmarz

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.zeros(10)

        x_opt, n_iter, converged = solve_kaczmarz(
            A, b, x0, omega=0.5, max_iterations=500, tolerance=1e-6
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)


# =============================================================================
# Edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_doroshenko_zero_readings(self, detector):
        """Test Doroshenko with zero readings."""
        readings = {detector.detector_names[0]: 0.0}
        result = detector.unfold_doroshenko(
            readings, max_iterations=100
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_kaczmarz_zero_readings(self, detector):
        """Test Kaczmarz with zero readings."""
        readings = {detector.detector_names[0]: 0.0}
        result = detector.unfold_kaczmarz(
            readings, max_iterations=100
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_lmfit_zero_readings(self, detector):
        """Test lmfit with zero readings."""
        readings = {detector.detector_names[0]: 0.0}
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_combined_empty_pipeline(self, detector, readings):
        """Test combined with empty pipeline."""
        pipeline = []
        result = detector.unfold_combined(readings, pipeline=pipeline)
        # Empty pipeline returns None (no stages to execute)
        assert result is None

    def test_doroshenko_negative_readings(self, detector):
        """Test Doroshenko with negative readings (should handle gracefully)."""
        readings = {detector.detector_names[0]: -10.0}
        with pytest.raises((ValueError, RuntimeError)):
            detector.unfold_doroshenko(readings, max_iterations=100)

    def test_results_history_after_multiple_unfolds(self, detector, readings):
        """Test that results history accumulates correctly."""
        detector.clear_results()
        assert len(detector.list_results()) == 0

        detector.unfold_doroshenko(
            readings, max_iterations=100, save_result=True
        )
        assert len(detector.list_results()) == 1

        detector.unfold_kaczmarz(
            readings, max_iterations=100, save_result=True
        )
        assert len(detector.list_results()) == 2

        detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
            save_result=True,
        )
        assert len(detector.list_results()) == 3

        # Verify we can retrieve results using list_results() keys
        keys = detector.list_results()
        result1 = detector.get_result(keys[0])
        assert result1['method'] == 'Doroshenko'

        result2 = detector.get_result(keys[1])
        assert result2['method'] == 'Kaczmarz'

        result3 = detector.get_result(keys[2])
        assert result3['method'].startswith('lmfit')