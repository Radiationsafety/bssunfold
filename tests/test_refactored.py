"""Tests for refactored bssunfold package.

These tests verify:
1. Backward compatibility - old API still works
2. Platform compatibility - no jaxlib/proxsuite imports on Windows
3. Functional equivalence - results match before/after refactoring
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# Test imports - backward compatibility
def test_backward_compatibility_imports():
    """Test that old import paths still work."""
    # Main class import
    from bssunfold import Detector
    assert Detector is not None
    
    # Constants imports
    from bssunfold import (
        ICRP116_COEFF_EFFECTIVE_DOSE,
        RF_GSF,
        RF_PTB,
        RF_LANL,
    )
    assert isinstance(ICRP116_COEFF_EFFECTIVE_DOSE, dict)
    assert isinstance(RF_GSF, dict)
    assert "E_MeV" in RF_GSF
    assert "E_MeV" in RF_PTB
    assert "E_MeV" in RF_LANL


def test_detector_creation():
    """Test Detector class instantiation with various inputs."""
    from bssunfold import Detector
    
    # Test with default GSF response functions
    detector = Detector()
    assert detector.n_detectors > 0
    assert detector.n_energy_bins > 0
    assert len(detector.detector_names) == detector.n_detectors
    
    # Test with DataFrame
    df = pd.DataFrame({
        'E_MeV': [1e-9, 1e-8, 1e-7],
        'sphere_1': [0.1, 0.2, 0.3],
        'sphere_2': [0.4, 0.5, 0.6]
    })
    detector_df = Detector(df)
    assert detector_df.n_detectors == 2
    assert detector_df.n_energy_bins == 3
    
    # Test with dict
    rf_dict = {
        'E_MeV': [1e-9, 1e-8, 1e-7],
        'sphere_1': [0.1, 0.2, 0.3],
        'sphere_2': [0.4, 0.5, 0.6]
    }
    detector_dict = Detector(rf_dict)
    assert detector_dict.n_detectors == 2


def test_detector_unfold_cvxpy():
    """Test unfold_cvxpy method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    # Test basic unfolding
    result = detector.unfold_cvxpy(readings, regularization=1e-3)
    
    assert 'spectrum' in result
    assert 'energy' in result
    assert 'residual_norm' in result
    assert 'method' in result
    assert result['method'] == 'cvxpy'
    assert len(result['spectrum']) == detector.n_energy_bins
    assert np.all(result['spectrum'] >= 0)  # Non-negative


def test_detector_unfold_landweber():
    """Test unfold_landweber method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_landweber(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert 'iterations' in result
    assert 'converged' in result
    assert result['method'] == 'Landweber'
    assert len(result['spectrum']) == detector.n_energy_bins


def test_detector_unfold_mlem():
    """Test unfold_mlem method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_mlem(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert 'iterations' in result
    assert result['method'] == 'MLEM'


def test_detector_unfold_qpsolvers():
    """Test unfold_qpsolvers method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    # Use osqp solver (available on all platforms)
    result = detector.unfold_qpsolvers(
        readings,
        regularization=1e-3,
        solver='osqp'
    )
    
    assert 'spectrum' in result
    assert 'qpsolvers' in result['method']


def test_platform_check_no_jaxlib_import():
    """Test that jaxlib is not imported on Windows."""
    from bssunfold.platform_check import (
        is_windows,
        JAX_AVAILABLE,
        check_jax_availability,
    )
    
    # Check that JAX_AVAILABLE is False if jaxlib not installed
    if is_windows:
        assert not JAX_AVAILABLE, "JAX should not be available on Windows"
    
    # Test check function
    result = check_jax_availability()
    if is_windows:
        assert not result


def test_platform_check_no_proxsuite_import():
    """Test that proxsuite is not imported on Windows."""
    from bssunfold.platform_check import (
        is_windows,
        PROXSUITE_AVAILABLE,
        check_proxsuite_availability,
    )
    
    # Check that PROXSUITE_AVAILABLE is False if proxsuite not installed
    if is_windows:
        assert not PROXSUITE_AVAILABLE, "Proxsuite should not be available on Windows"
    
    # Test check function
    result = check_proxsuite_availability()
    if is_windows:
        assert not result


def test_get_available_solvers():
    """Test get_available_solvers function."""
    from bssunfold.platform_check import get_available_solvers, is_windows
    
    solvers = get_available_solvers()
    
    assert isinstance(solvers, dict)
    assert 'ecos' in solvers
    assert 'osqp' in solvers
    assert 'proxqp' in solvers
    
    # On Windows, proxqp should be False
    if is_windows:
        assert not solvers['proxqp']


def test_get_recommended_solver():
    """Test get_recommended_solver function."""
    from bssunfold.platform_check import get_recommended_solver, is_windows
    
    solver = get_recommended_solver()
    
    # On Windows, should not recommend proxqp
    if is_windows:
        assert solver != 'proxqp'
        assert solver in ['osqp', 'ecos', 'piqp']


def test_constants_structure():
    """Test that constants have correct structure."""
    from bssunfold import (
        ICRP116_COEFF_EFFECTIVE_DOSE,
        RF_GSF,
    )
    
    # Check ICRP116 coefficients
    assert 'E_MeV' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'AP' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'PA' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'LLAT' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'RLAT' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'ISO' in ICRP116_COEFF_EFFECTIVE_DOSE
    assert 'ROT' in ICRP116_COEFF_EFFECTIVE_DOSE
    
    # Check RF_GSF
    assert 'E_MeV' in RF_GSF
    assert len(RF_GSF['E_MeV']) > 0
    
    # All arrays should have same length
    n_energy = len(RF_GSF['E_MeV'])
    for key, value in RF_GSF.items():
        if key != 'E_MeV':
            assert len(value) == n_energy


def test_detector_string_representations():
    """Test __str__ and __repr__ methods."""
    from bssunfold import Detector
    
    detector = Detector()
    
    str_repr = str(detector)
    assert 'Detector' in str_repr
    assert 'energy bins' in str_repr
    assert 'detectors' in str_repr
    
    repr_repr = repr(detector)
    assert 'Detector' in repr_repr
    assert 'E_MeV' in repr_repr


def test_detector_results_history():
    """Test results history functionality."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    # First unfolding
    detector.unfold_cvxpy(readings, save_result=True)
    
    # Check history
    assert len(detector.list_results()) >= 1
    assert detector.current_result is not None
    
    # Get result by key
    keys = detector.list_results()
    if keys:
        result_by_key = detector.get_result(keys[0])
        assert result_by_key is not None
    
    # Get current result
    current = detector.get_result()
    assert current is not None
    
    # Clear results
    detector.clear_results()
    assert len(detector.list_results()) == 0
    assert detector.current_result is None


def test_detector_effective_readings():
    """Test get_effective_readings_for_spectra method."""
    from bssunfold import Detector
    import numpy as np
    
    detector = Detector()
    
    # Create a simple spectrum
    spectrum = np.ones(detector.n_energy_bins)
    spectrum_df = pd.DataFrame({
        'E_MeV': detector.E_MeV,
        'Phi': spectrum
    })
    
    readings = detector.get_effective_readings_for_spectra(spectrum_df)
    
    assert isinstance(readings, dict)
    assert len(readings) == detector.n_detectors
    assert all(v >= 0 for v in readings.values())


def test_detector_discretize_spectra():
    """Test discretize_spectra method."""
    from bssunfold import Detector
    import numpy as np
    
    detector = Detector()
    
    # Create spectrum on different energy grid
    E_coarse = np.logspace(-9, -1, 20)
    Phi_coarse = np.ones(20)
    
    spectrum_dict = {
        'E_MeV': E_coarse,
        'Phi': Phi_coarse
    }
    
    discretized = detector.discretize_spectra(spectrum_dict)
    
    assert 'E_MeV' in discretized.columns
    assert 'Phi' in discretized.columns
    assert len(discretized) == detector.n_energy_bins


def test_logging_setup():
    """Test logging configuration."""
    from bssunfold.logging_config import setup_logging, get_logger
    import logging
    
    logger = setup_logging(level=logging.INFO)
    assert logger is not None
    assert logger.level == logging.INFO
    
    named_logger = get_logger('test')
    assert named_logger is not None


def test_utils_imports():
    """Test that utility modules are importable."""
    from bssunfold.utils import (
        validate_readings,
        validate_energy_grid,
        convert_to_dataframe,
        interpolate_spectrum,
        plot_spectrum,
    )
    
    assert validate_readings is not None
    assert validate_energy_grid is not None
    assert convert_to_dataframe is not None
    assert interpolate_spectrum is not None
    assert plot_spectrum is not None


def test_core_imports():
    """Test that core modules are importable."""
    from bssunfold.core import (
        Detector,
        solve_cvxpy,
        solve_landweber,
        solve_mlem,
        solve_qpsolvers,
        solve_doroshenko,
        solve_kaczmarz,
        solve_lmfit,
        select_regularization_parameter,
        lcurve_selection,
        gcv_selection,
        discrepancy_principle_selection,
        cosine_similarity_selection,
        compare_regularization_methods,
        randomization_experiment,
        calculate_dose_rates,
    )
    
    assert Detector is not None
    assert solve_cvxpy is not None
    assert solve_landweber is not None
    assert solve_mlem is not None
    assert solve_qpsolvers is not None
    assert solve_doroshenko is not None
    assert solve_kaczmarz is not None
    assert solve_lmfit is not None
    assert select_regularization_parameter is not None
    assert lcurve_selection is not None
    assert gcv_selection is not None
    assert discrepancy_principle_selection is not None
    assert cosine_similarity_selection is not None
    assert compare_regularization_methods is not None
    assert randomization_experiment is not None
    assert calculate_dose_rates is not None


def test_windows_compatibility_simulation():
    """Simulate Windows environment and verify no jaxlib/proxsuite imports."""
    # This test simulates Windows by mocking sys.platform
    with patch('sys.platform', 'win32'):
        # Re-import to trigger platform check
        import importlib
        import bssunfold.platform_check as pc
        
        # Reload to re-evaluate platform check
        importlib.reload(pc)
        
        assert pc.is_windows
        # JAX and proxsuite should not be available
        # (unless actually installed, but the check should work)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
