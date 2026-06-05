"""Tests for new unfolding methods added from addons/."""

import pytest
import numpy as np
from unittest.mock import patch
import builtins


# ============================================================================
# Test solve_gravel
# ============================================================================

class TestSolveGravel:
    def test_gravel_basic(self):
        from bssunfold.core import solve_gravel
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x0 = np.ones(10) / 10
        x, iterations, converged = solve_gravel(A, b, x0, max_iterations=200)
        assert len(x) == 10
        assert iterations > 0
        assert converged
        assert np.all(x >= 0)

    def test_gravel_all_zero_measurements(self):
        from bssunfold.core import solve_gravel
        A = np.random.rand(5, 10)
        b = np.zeros(5)
        x0 = np.ones(10)
        with pytest.raises(ValueError, match="All measurements are zero or negative"):
            solve_gravel(A, b, x0)

    def test_gravel_tolerance_zero(self):
        from bssunfold.core import solve_gravel
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 4, 10))
        x0 = np.ones(10)
        x, iterations, converged = solve_gravel(A, b, x0, tolerance=0, max_iterations=5)
        assert iterations == 5
        assert not converged


# ============================================================================
# Test solve_maxed
# ============================================================================

class TestSolveMaxed:
    def test_maxed_basic(self):
        from bssunfold.core import solve_maxed
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 4, 10))
        x0 = np.ones(10)
        x, iterations, converged = solve_maxed(
            A, b, x0, max_iterations=100, tolerance=1e-3
        )
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_tikhonov_legendre
# ============================================================================

class TestSolveTikhonovLegendre:
    def test_tikhonov_legendre_basic(self):
        from bssunfold.core import solve_tikhonov_legendre
        np.random.seed(42)
        A = np.random.rand(5, 20)
        b = np.random.rand(5)
        x = solve_tikhonov_legendre(A, b, delta=0.1, n_polynomials=10)
        assert len(x) == 20
        assert np.all(x >= 0)

    def test_tikhonov_legendre_default_params(self):
        from bssunfold.core import solve_tikhonov_legendre
        np.random.seed(42)
        A = np.random.rand(3, 15)
        b = np.random.rand(3)
        x = solve_tikhonov_legendre(A, b)
        assert len(x) == 15
        assert np.all(x >= 0)


# ============================================================================
# Test solve_scipy_direct
# ============================================================================

class TestSolveScipyDirect:
    def test_scipy_direct_cg(self):
        from bssunfold.core import solve_scipy_direct
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(A, b, method="cg", tolerance=1e-6, max_iterations=1000)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_lsqr(self):
        from bssunfold.core import solve_scipy_direct
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(A, b, method="lsqr", tolerance=1e-6)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_gmres(self):
        from bssunfold.core import solve_scipy_direct
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(A, b, method="gmres", tolerance=1e-6)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_unknown_method(self):
        from bssunfold.core import solve_scipy_direct
        A = np.random.rand(3, 5)
        b = np.random.rand(3)
        with pytest.raises(ValueError, match="Unknown solver method"):
            solve_scipy_direct(A, b, method="unknown")


# ============================================================================
# Test solve_tsvd
# ============================================================================

class TestSolveTsvd:
    def test_tsvd_basic(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, k=5)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_fixed_k(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, k=3)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_threshold(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, threshold=0.1)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_discrepancy(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="discrepancy")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_lcurve(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="l_curve")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_gcv(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="gcv")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_energy(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="energy")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_median(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="median_threshold")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_donoho(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="donoho")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_default(self):
        from bssunfold.core import solve_tsvd
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="nonexistent")
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_bayes (pyunfold dependency)
# ============================================================================

class TestSolveBayes:
    def test_bayes_basic(self):
        from bssunfold.core import solve_bayes
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10) * 10
        x = solve_bayes(A, b, max_iterations=10, tolerance=1)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_bayes_no_prior(self):
        from bssunfold.core import solve_bayes
        np.random.seed(42)
        A = np.random.rand(4, 8)
        b = A @ np.ones(8) * 10
        x = solve_bayes(A, b, max_iterations=5, tolerance=10)
        assert len(x) == 8
        assert np.all(x >= 0)


# ============================================================================
# Test solve_bayes_spline (pyunfold dependency)
# ============================================================================

class TestSolveBayesSpline:
    def test_bayes_spline_basic(self):
        from bssunfold.core import solve_bayes_spline
        np.random.seed(42)
        A = np.random.rand(5, 15)
        b = A @ np.ones(15) * 10
        x = solve_bayes_spline(
            A, b, max_iterations=10, tolerance=1,
            spline_degree=1, spline_smooth=0.1,
        )
        assert len(x) == 15
        assert np.all(x >= 0)

    def test_bayes_spline_no_prior(self):
        from bssunfold.core import solve_bayes_spline
        np.random.seed(42)
        A = np.random.rand(4, 10)
        b = A @ np.ones(10) * 10
        x = solve_bayes_spline(
            A, b, max_iterations=5, tolerance=10,
            spline_degree=1, spline_smooth=1.0,
        )
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_statreg (statreg dependency)
# ============================================================================

class TestSolveStatreg:
    def test_statreg_raises_import_error(self):
        from bssunfold.core import solve_statreg
        A = np.random.rand(3, 5)
        b = np.random.rand(3)
        with pytest.raises(ImportError, match="statreg is required"):
            solve_statreg(A, b)

    def test_statreg_import_error_message(self):
        from bssunfold.core import solve_statreg
        A = np.eye(3)
        b = np.ones(3)
        with pytest.raises(ImportError) as excinfo:
            solve_statreg(A, b)
        assert "statreg" in str(excinfo.value)
        assert "pip install statreg" in str(excinfo.value)


# ============================================================================
# Test unfold_* wrappers via Detector class
# ============================================================================

@pytest.fixture
def detector():
    from bssunfold import Detector
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


class TestUnfoldGravel:
    def test_unfold_gravel_basic(self, detector, readings):
        result = detector.unfold_gravel(readings, max_iterations=10, tolerance=1e-3)
        assert 'spectrum' in result
        assert 'doserates' in result
        assert 'residual' in result
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_gravel_with_initial(self, detector, readings):
        initial = np.ones(detector.n_energy_bins)
        result = detector.unfold_gravel(
            readings, initial_spectrum=initial, max_iterations=10
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_gravel_empty_readings(self, detector):
        with pytest.raises(ValueError):
            detector.unfold_gravel({})

    def test_unfold_gravel_no_save(self, detector, readings):
        result = detector.unfold_gravel(readings, max_iterations=5, save_result=False)
        assert 'spectrum' in result


class TestUnfoldMaxed:
    def test_unfold_maxed_basic(self, detector, readings):
        result = detector.unfold_maxed(
            readings, max_iterations=50, tolerance=0.1
        )
        assert 'spectrum' in result
        assert 'doserates' in result
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_maxed_with_reference(self, detector, readings):
        initial = np.ones(detector.n_energy_bins)
        result = detector.unfold_maxed(
            readings, initial_spectrum=initial,
            max_iterations=50, tolerance=0.1
        )
        assert 'spectrum' in result

    def test_unfold_maxed_no_save(self, detector, readings):
        result = detector.unfold_maxed(
            readings, max_iterations=20, save_result=False
        )
        assert 'spectrum' in result


class TestUnfoldTikhonovLegendre:
    def test_unfold_tikhonov_legendre_basic(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(readings, delta=0.1, n_polynomials=8)
        assert 'spectrum' in result
        assert 'doserates' in result
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_tikhonov_legendre_default(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(readings)
        assert 'spectrum' in result

    def test_unfold_tikhonov_legendre_no_save(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(
            readings, delta=0.1, save_result=False
        )
        assert 'spectrum' in result


class TestUnfoldScipyDirect:
    def test_unfold_scipy_direct_cg(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="cg")
        assert 'spectrum' in result
        assert 'doserates' in result

    def test_unfold_scipy_direct_lsqr(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="lsqr")
        assert 'spectrum' in result

    def test_unfold_scipy_direct_gmres(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="gmres")
        assert 'spectrum' in result

    def test_unfold_scipy_direct_lsmr(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="lsmr")
        assert 'spectrum' in result

    def test_unfold_scipy_direct_minres(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="minres")
        assert 'spectrum' in result

    def test_unfold_scipy_direct_no_save(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, save_result=False)
        assert 'spectrum' in result


class TestUnfoldTsvd:
    def test_unfold_tsvd_basic(self, detector, readings):
        result = detector.unfold_tsvd(readings, k=5)
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_tsvd_auto_discrepancy(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="discrepancy")
        assert 'spectrum' in result

    def test_unfold_tsvd_auto_gcv(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="gcv")
        assert 'spectrum' in result

    def test_unfold_tsvd_auto_lcurve(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="l_curve")
        assert 'spectrum' in result

    def test_unfold_tsvd_no_save(self, detector, readings):
        result = detector.unfold_tsvd(readings, k=3, save_result=False)
        assert 'spectrum' in result


class TestUnfoldBayes:
    def test_unfold_bayes_basic(self, detector, readings):
        result = detector.unfold_bayes(readings, max_iterations=10, tolerance=1)
        assert 'spectrum' in result
        assert 'doserates' in result

    def test_unfold_bayes_spline_basic(self, detector, readings):
        result = detector.unfold_bayes_spline_regularization(
            readings, max_iterations=10, tolerance=1,
            spline_degree=1, spline_smooth=0.1,
        )
        assert 'spectrum' in result
        assert 'doserates' in result


# ============================================================================
# Test solve_statreg and unfold_statreg (statreg dependency, scipy-incompatible)
# ============================================================================

class TestSolveStatreg:
    def test_statreg_import_error_with_mock(self):
        from bssunfold.core import solve_statreg
        A = np.random.rand(3, 5)
        b = np.random.rand(3)
        orig = builtins.__import__
        def mock_import(name, *a, **kw):
            if name == 'statreg' or name.startswith('statreg.'):
                raise ImportError
            return orig(name, *a, **kw)
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="statreg is required"):
                solve_statreg(A, b)


class TestUnfoldStatreg:
    def test_unfold_statreg_import_error_with_mock(self, detector, readings):
        orig = builtins.__import__
        def mock_import(name, *a, **kw):
            if name == 'statreg' or name.startswith('statreg.'):
                raise ImportError
            return orig(name, *a, **kw)
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="statreg is required"):
                detector.unfold_statreg(readings)


# ============================================================================
# Test MC with new methods
# ============================================================================

class TestNewMethodsWithMC:
    def test_gravel_with_errors(self, detector, readings):
        result = detector.unfold_gravel(
            readings, max_iterations=5, calculate_errors=True,
            n_montecarlo=3, random_state=42
        )
        assert 'spectrum_uncert_std' in result

    def test_tsvd_with_errors(self, detector, readings):
        result = detector.unfold_tsvd(
            readings, k=3, calculate_errors=True,
            n_montecarlo=3, random_state=42
        )
        assert 'spectrum_uncert_std' in result

    def test_scipy_direct_with_errors(self, detector, readings):
        result = detector.unfold_scipy_direct_method(
            readings, method="cg", calculate_errors=True,
            n_montecarlo=3, random_state=42
        )
        assert 'spectrum_uncert_std' in result


# ============================================================================
# Verify solve_* functions are in __all__
# ============================================================================

class TestModuleExports:
    def test_solve_gravel_exported(self):
        from bssunfold.core import solve_gravel
        assert callable(solve_gravel)

    def test_solve_maxed_exported(self):
        from bssunfold.core import solve_maxed
        assert callable(solve_maxed)

    def test_solve_tikhonov_legendre_exported(self):
        from bssunfold.core import solve_tikhonov_legendre
        assert callable(solve_tikhonov_legendre)

    def test_solve_bayes_exported(self):
        from bssunfold.core import solve_bayes
        assert callable(solve_bayes)

    def test_solve_bayes_spline_exported(self):
        from bssunfold.core import solve_bayes_spline
        assert callable(solve_bayes_spline)

    def test_solve_statreg_exported(self):
        from bssunfold.core import solve_statreg
        assert callable(solve_statreg)

    def test_solve_scipy_direct_exported(self):
        from bssunfold.core import solve_scipy_direct
        assert callable(solve_scipy_direct)

    def test_solve_tsvd_exported(self):
        from bssunfold.core import solve_tsvd
        assert callable(solve_tsvd)

    def test_unfold_gravel_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_gravel')

    def test_unfold_maxed_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_maxed')

    def test_unfold_tikhonov_legendre_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_tikhonov_legendre')

    def test_unfold_bayes_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_bayes')

    def test_unfold_bayes_spline_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_bayes_spline_regularization')

    def test_unfold_statreg_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_statreg')

    def test_unfold_scipy_direct_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_scipy_direct_method')

    def test_unfold_tsvd_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_tsvd')
