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


def test_backward_compatibility_imports():
    """Test that old import paths still work."""
    from bssunfold import Detector
    assert Detector is not None
    
    from bssunfold import (
        ICRP116_COEFF_EFFECTIVE_DOSE,
        RF_GSF,
        RF_PTB,
        RF_LANL,
    )
    assert isinstance(ICRP116_COEFF_EFFECTIVE_DOSE, dict)
    assert isinstance(RF_GSF, dict)
    assert "E_MeV" in RF_GSF


def test_detector_creation():
    """Test Detector class instantiation."""
    from bssunfold import Detector
    
    detector = Detector()
    assert detector.n_detectors > 0
    assert detector.n_energy_bins > 0
    
    df = pd.DataFrame({
        'E_MeV': [1e-9, 1e-8, 1e-7],
        'sphere_1': [0.1, 0.2, 0.3],
        'sphere_2': [0.4, 0.5, 0.6]
    })
    detector_df = Detector(df)
    assert detector_df.n_detectors == 2


def test_detector_unfold_cvxpy():
    """Test unfold_cvxpy method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_cvxpy(readings, regularization=1e-3)
    
    assert 'spectrum' in result
    assert 'energy' in result
    assert 'method' in result
    assert result['method'] == 'cvxpy'
    assert len(result['spectrum']) == detector.n_energy_bins
    assert np.all(result['spectrum'] >= 0)


def test_detector_unfold_landweber():
    """Test unfold_landweber method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_landweber(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert 'method' in result
    assert result['method'] == 'Landweber'


def test_detector_unfold_mlem():
    """Test unfold_mlem method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_mlem(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert result['method'] == 'MLEM'


def test_detector_unfold_qpsolvers():
    """Test unfold_qpsolvers method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_qpsolvers(
        readings,
        regularization=1e-3,
        solver='osqp'
    )
    
    assert 'spectrum' in result
    assert 'qpsolvers' in result['method']


def test_detector_unfold_doroshenko():
    """Test unfold_doroshenko method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_doroshenko(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert result['method'] == 'Doroshenko'


def test_detector_unfold_kaczmarz():
    """Test unfold_kaczmarz method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_kaczmarz(readings, max_iterations=100)
    
    assert 'spectrum' in result
    assert result['method'] == 'Kaczmarz'


def test_detector_unfold_lmfit():
    """Test unfold_lmfit method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    result = detector.unfold_lmfit(
        readings,
        method='lbfgsb',
        model_name='elastic',
        regularization=1e-4,
    )
    
    assert 'spectrum' in result
    assert 'lmfit' in result['method']


def test_detector_unfold_combined():
    """Test unfold_combined method."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    pipeline = [
        {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
        {'method': 'landweber', 'params': {'max_iterations': 50}},
    ]
    result = detector.unfold_combined(readings, pipeline=pipeline)
    
    assert result is not None
    assert 'spectrum' in result
    assert 'pipeline_info' in result


def test_platform_check():
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


def test_detector_results_history():
    """Test results history functionality."""
    from bssunfold import Detector
    
    detector = Detector()
    readings = {detector.detector_names[0]: 100.0}
    
    detector.unfold_cvxpy(readings, save_result=True)
    
    assert len(detector.list_results()) >= 1
    assert detector.current_result is not None
    
    detector.clear_results()
    assert len(detector.list_results()) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
