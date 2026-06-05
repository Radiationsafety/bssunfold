"""Tests for newly added unfolding methods.

Simplified tests for doroshenko, kaczmarz, lmfit, combined, and mlem_odl methods.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def detector():
    """Create a Detector instance with default GSF response functions."""
    from bssunfold import Detector
    return Detector()


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
    })
    return Detector(df)


class TestUnfoldDoroshenko:
    """Tests for unfold_doroshenko method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic Doroshenko unfolding."""
        result = detector.unfold_doroshenko(readings, max_iterations=200)
        assert 'spectrum' in result
        assert 'energy' in result
        assert result['method'] == 'Doroshenko'
        assert len(result['spectrum']) == detector.n_energy_bins
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


class TestUnfoldKaczmarz:
    """Tests for unfold_kaczmarz method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic Kaczmarz unfolding."""
        result = detector.unfold_kaczmarz(readings, max_iterations=200)
        assert 'spectrum' in result
        assert result['method'] == 'Kaczmarz'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_with_omega(self, detector, readings):
        """Test with different relaxation parameter."""
        result = detector.unfold_kaczmarz(
            readings, omega=0.5, max_iterations=200
        )
        assert 'spectrum' in result


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
        assert result['model_name'] == 'elastic'
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

    def test_invalid_method(self, detector, readings):
        """Test that invalid method raises ValueError."""
        pipeline = [
            {'method': 'nonexistent_method', 'params': {}},
        ]
        with pytest.raises(ValueError):
            detector.unfold_combined(readings, pipeline=pipeline)


class TestUnfoldMlemOdl:
    """Tests for unfold_mlem_odl method."""

    def test_basic_unfolding(self, detector, readings):
        """Test basic MLEM-ODL unfolding."""
        result = detector.unfold_mlem_odl(readings, max_iterations=50)
        assert 'spectrum' in result
        assert result['method'] == 'MLEM (ODL)'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)


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


class TestPlotting:
    """Tests for plotting methods."""

    def test_plot_response_functions(self, detector, tmp_path):
        """Test that plot creates a figure."""
        import matplotlib
        matplotlib.use('Agg')

        save_path = str(tmp_path / "response_functions.png")
        detector.plot_response_functions(
            save_to=save_path, show=False
        )
        assert tmp_path.joinpath("response_functions.png").exists()

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
