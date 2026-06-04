"""Comprehensive tests for bssunfold package.

This file contains all essential tests for the refactored bssunfold package.
Run with: pytest tests/test_all.py -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ============================================================================
# Basic Import Tests
# ============================================================================

class TestImports:
    """Test that all modules can be imported."""

    def test_main_imports(self):
        """Test main package imports."""
        from bssunfold import Detector
        from bssunfold import (
            ICRP116_COEFF_EFFECTIVE_DOSE,
            RF_GSF,
            RF_PTB,
            RF_LANL,
        )
        assert Detector is not None
        assert isinstance(ICRP116_COEFF_EFFECTIVE_DOSE, dict)
        assert "E_MeV" in RF_GSF

    def test_core_imports(self):
        """Test core module imports."""
        from bssunfold.core import (
            Detector,
            solve_cvxpy,
            solve_landweber,
            solve_mlem,
            solve_qpsolvers,
            solve_doroshenko,
            solve_kaczmarz,
            solve_lmfit,
            unfold_cvxpy,
            unfold_landweber,
            unfold_mlem,
            unfold_qpsolvers,
            unfold_doroshenko,
            unfold_kaczmarz,
            unfold_lmfit,
            unfold_mlem_odl,
            unfold_combined,
        )
        assert Detector is not None
        assert solve_cvxpy is not None
        assert unfold_cvxpy is not None

    def test_platform_check(self):
        """Test platform check functions."""
        from bssunfold.platform_check import (
            is_windows,
            get_available_solvers,
            get_recommended_solver,
        )
        
        solvers = get_available_solvers()
        assert isinstance(solvers, dict)
        assert 'ecos' in solvers
        
        solver = get_recommended_solver()
        assert isinstance(solver, str)


# ============================================================================
# Detector Tests
# ============================================================================

class TestDetector:
    """Test Detector class."""

    @pytest.fixture
    def detector(self):
        """Create default detector."""
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def small_detector(self):
        """Create small detector for fast tests."""
        from bssunfold import Detector
        df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'sphere_1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sphere_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        return Detector(df)

    @pytest.fixture
    def readings(self, detector):
        """Create sample readings."""
        return {detector.detector_names[0]: 100.0}

    def test_creation(self, detector):
        """Test detector creation."""
        assert detector.n_detectors > 0
        assert detector.n_energy_bins > 0

    def test_string_repr(self, detector):
        """Test string representations."""
        str_repr = str(detector)
        assert 'Detector' in str_repr
        assert 'energy bins' in str_repr

    def test_unfold_cvxpy(self, detector, readings):
        """Test unfold_cvxpy method."""
        result = detector.unfold_cvxpy(readings, regularization=1e-3)
        assert 'spectrum' in result
        assert 'energy' in result
        assert result['method'] == 'cvxpy'
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_landweber(self, detector, readings):
        """Test unfold_landweber method."""
        result = detector.unfold_landweber(readings, max_iterations=100)
        assert 'spectrum' in result
        assert result['method'] == 'Landweber'

    def test_unfold_mlem(self, detector, readings):
        """Test unfold_mlem method."""
        result = detector.unfold_mlem(readings, max_iterations=100)
        assert 'spectrum' in result
        assert result['method'] == 'MLEM'

    def test_unfold_qpsolvers(self, detector, readings):
        """Test unfold_qpsolvers method."""
        result = detector.unfold_qpsolvers(
            readings, regularization=1e-3, solver='osqp'
        )
        assert 'spectrum' in result
        assert 'qpsolvers' in result['method']

    def test_unfold_doroshenko(self, detector, readings):
        """Test unfold_doroshenko method."""
        result = detector.unfold_doroshenko(readings, max_iterations=100)
        assert 'spectrum' in result
        assert result['method'] == 'Doroshenko'

    def test_unfold_kaczmarz(self, detector, readings):
        """Test unfold_kaczmarz method."""
        result = detector.unfold_kaczmarz(readings, max_iterations=100)
        assert 'spectrum' in result
        assert result['method'] == 'Kaczmarz'

    def test_unfold_lmfit(self, detector, readings):
        """Test unfold_lmfit method."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='elastic',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert 'lmfit' in result['method']

    def test_unfold_combined(self, detector, readings):
        """Test unfold_combined method."""
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
            {'method': 'landweber', 'params': {'max_iterations': 50}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert result is not None
        assert 'spectrum' in result
        assert 'pipeline_info' in result

    def test_unfold_mlem_odl(self, detector, readings):
        """Test unfold_mlem_odl method."""
        result = detector.unfold_mlem_odl(readings, max_iterations=50)
        assert 'spectrum' in result
        assert result['method'] == 'MLEM (ODL)'

    def test_results_history(self, detector, readings):
        """Test results history functionality."""
        detector.unfold_cvxpy(readings, save_result=True)
        assert len(detector.list_results()) >= 1
        assert detector.current_result is not None
        
        detector.clear_results()
        assert len(detector.list_results()) == 0

    def test_effective_readings(self, detector):
        """Test get_effective_readings_for_spectra."""
        spectrum = np.ones(detector.n_energy_bins)
        spectrum_df = pd.DataFrame({
            'E_MeV': detector.E_MeV,
            'Phi': spectrum
        })
        readings = detector.get_effective_readings_for_spectra(spectrum_df)
        assert isinstance(readings, dict)
        assert len(readings) == detector.n_detectors

    def test_discretize_spectra(self, detector):
        """Test discretize_spectra method."""
        E_coarse = np.logspace(-9, -1, 20)
        spectrum_dict = {
            'E_MeV': E_coarse,
            'Phi': np.ones(20)
        }
        discretized = detector.discretize_spectra(spectrum_dict)
        assert 'E_MeV' in discretized.columns
        assert 'Phi' in discretized.columns
        assert len(discretized) == detector.n_energy_bins


# ============================================================================
# Monte-Carlo Uncertainty Tests
# ============================================================================

class TestMonteCarlo:
    """Test Monte-Carlo uncertainty estimation."""

    @pytest.fixture
    def detector(self):
        """Create default detector."""
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        """Create sample readings."""
        return {detector.detector_names[0]: 100.0}

    def test_cvxpy_with_uncertainty(self, detector, readings):
        """Test unfold_cvxpy with Monte-Carlo uncertainty."""
        result = detector.unfold_cvxpy(
            readings,
            regularization=1e-2,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result

    def test_landweber_with_uncertainty(self, detector, readings):
        """Test unfold_landweber with Monte-Carlo uncertainty."""
        result = detector.unfold_landweber(
            readings,
            max_iterations=50,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'montecarlo_samples' in result

    def test_doroshenko_with_uncertainty(self, detector, readings):
        """Test unfold_doroshenko with Monte-Carlo uncertainty."""
        result = detector.unfold_doroshenko(
            readings,
            max_iterations=100,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result


# ============================================================================
# Direct Function Tests
# ============================================================================

class TestDirectFunctions:
    """Test direct function calls from core modules."""

    def test_solve_doroshenko(self):
        """Test solve_doroshenko function."""
        from bssunfold.core.unfolding_methods import solve_doroshenko

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, n_iter, converged = solve_doroshenko(
            A, b, x0, max_iterations=500
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)

    def test_solve_kaczmarz(self):
        """Test solve_kaczmarz function."""
        from bssunfold.core.unfolding_methods import solve_kaczmarz

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.zeros(10)

        x_opt, n_iter, converged = solve_kaczmarz(
            A, b, x0, max_iterations=500
        )
        assert x_opt.shape == (10,)
        assert np.all(x_opt >= 0)

    def test_solve_lmfit(self):
        """Test solve_lmfit function."""
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


# ============================================================================
# Plotting Tests
# ============================================================================

class TestPlotting:
    """Test plotting methods."""

    @pytest.fixture
    def detector(self):
        """Create default detector."""
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        """Create sample readings."""
        return {detector.detector_names[0]: 100.0}

    def test_plot_response_functions(self, detector, tmp_path):
        """Test plot_response_functions."""
        import matplotlib
        matplotlib.use('Agg')

        save_path = str(tmp_path / "response_functions.png")
        detector.plot_response_functions(save_to=save_path, show=False)
        assert tmp_path.joinpath("response_functions.png").exists()

    def test_plot_with_uncertainty(self, detector, readings, tmp_path):
        """Test plot_with_uncertainty."""
        import matplotlib
        matplotlib.use('Agg')

        result = detector.unfold_doroshenko(
            readings,
            max_iterations=100,
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


# ============================================================================
# lmfit Model Tests
# ============================================================================

class TestLmfitModels:
    """Test lmfit with different regularization models."""

    @pytest.fixture
    def detector(self):
        """Create default detector."""
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        """Create sample readings."""
        return {detector.detector_names[0]: 100.0}

    def test_elastic_model(self, detector, readings):
        """Test elastic net model."""
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

    def test_lasso_model(self, detector, readings):
        """Test lasso model."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='lasso',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert result['model_name'] == 'lasso'

    def test_ridge_model(self, detector, readings):
        """Test ridge model."""
        result = detector.unfold_lmfit(
            readings,
            method='lbfgsb',
            model_name='ridge',
            regularization=1e-4,
        )
        assert 'spectrum' in result
        assert result['model_name'] == 'ridge'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
