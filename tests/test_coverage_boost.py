"""Comprehensive coverage boost tests.

Targets specific uncovered lines identified by pytest-cov to push
coverage from 83% to 91%+.
"""

import sys
import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch, MagicMock
from numpy.testing import assert_array_almost_equal


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create a default Detector instance."""
    from bssunfold import Detector
    return Detector()


@pytest.fixture
def simple_response():
    """Simple response matrix, readings, energy grid for unit tests."""
    n_det = 4
    n_energy = 10
    np.random.seed(42)
    E = np.logspace(-6, 1, n_energy)
    A = np.random.rand(n_det, n_energy) * 1e-3
    x_true = np.ones(n_energy)
    b = A @ x_true
    return A, b, E


@pytest.fixture
def numba_disabled():
    """Context manager that patches NUMBA_AVAILABLE to False."""
    import bssunfold.core._numba_jit as nj
    orig = nj.NUMBA_AVAILABLE
    nj.NUMBA_AVAILABLE = False
    yield
    nj.NUMBA_AVAILABLE = orig


# ============================================================================
# 1. Solver fallback paths (doroshenko, gravel, kaczmarz, mlem)
#    These are pure Python paths that run when NUMBA_AVAILABLE = False.
# ============================================================================

class TestSolverFallbackPaths:
    """Force NUMBA_AVAILABLE=False to test pure-Python fallback paths."""

    def _mock_numba_and_clear(self):
        """Return context manager that forces pure-Python fallback."""
        import sys
        import types
        from unittest.mock import MagicMock

        saved = sys.modules.get("bssunfold.core._numba_jit")
        mock_numba = types.ModuleType("bssunfold.core._numba_jit")
        mock_numba.NUMBA_AVAILABLE = False
        # Provide stubs for any JIT functions that might be looked up
        for name in ["_doroshenko_inner", "_gravel_inner", "_kaczmarz_inner",
                      "_mlem_inner", "_compute_log_steps_jit", "_dose_weighted_mse_jit"]:
            setattr(mock_numba, name, MagicMock(side_effect=Exception("should not be called")))
        sys.modules["bssunfold.core._numba_jit"] = mock_numba

        class _Ctx:
            def __exit__(self, *a):
                if saved is not None:
                    sys.modules["bssunfold.core._numba_jit"] = saved
                else:
                    sys.modules.pop("bssunfold.core._numba_jit", None)
        return _Ctx()

    def test_doroshenko_fallback(self):
        from bssunfold.core.unfold_doroshenko import solve_doroshenko
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x0 = np.ones(10)
        ctx = self._mock_numba_and_clear()
        try:
            x, iters, conv = solve_doroshenko(A, b, x0, max_iterations=50, tolerance=1e-6)
        finally:
            ctx.__exit__()
        assert x.shape == (10,)
        assert iters > 0

    def test_gravel_fallback(self):
        from bssunfold.core.unfold_gravel import solve_gravel
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5) * 1.0
        x0 = np.ones(10) * 0.5
        ctx = self._mock_numba_and_clear()
        try:
            x, iters, conv = solve_gravel(A, b, x0, max_iterations=20, tolerance=1e-8)
        finally:
            ctx.__exit__()
        assert x.shape == (10,)
        assert iters > 0

    def test_kaczmarz_fallback(self):
        from bssunfold.core.unfold_kaczmarz import solve_kaczmarz
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x0 = np.ones(10) * 0.1
        ctx = self._mock_numba_and_clear()
        try:
            x, iters, conv = solve_kaczmarz(A, b, x0, max_iterations=100, tolerance=1e-6)
        finally:
            ctx.__exit__()
        assert x.shape == (10,)

    def test_mlem_fallback(self):
        from bssunfold.core.unfold_mlem import solve_mlem
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5) * 1.0
        x0 = np.ones(10) * 0.5
        ctx = self._mock_numba_and_clear()
        try:
            x, iters, conv = solve_mlem(A, b, x0, max_iterations=50, tolerance=1e-6)
        finally:
            ctx.__exit__()
        assert x.shape == (10,)
        assert iters > 0


# ============================================================================
# 2. platform_check.py uncovered lines (44-46, 63-64, 93-95)
# ============================================================================

class TestPlatformCheck:
    def test_check_jax_availability_import_error(self):
        import bssunfold.platform_check as pc
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("mocked jax")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            result = pc.check_jax_availability()
        assert result is False
        assert pc.JAX_AVAILABLE is False

    def test_check_proxsuite_availability_import_error(self):
        import bssunfold.platform_check as pc
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "proxsuite":
                raise ImportError("mocked proxsuite")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            result = pc.check_proxsuite_availability()
        assert result is False
        assert pc.PROXSUITE_AVAILABLE is False

    def test_check_qpsolvers_extra_import_error(self):
        import bssunfold.platform_check as pc
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "qpsolvers":
                raise ImportError("mocked qpsolvers")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            result = pc.check_qpsolvers_extra_availability()
        assert result is False
        assert pc.QPSOLVERS_EXTRA_AVAILABLE is False

    def test_check_qpsolvers_extra_no_extra_solvers(self):
        import bssunfold.platform_check as pc
        mock_mod = MagicMock()
        mock_mod.available_solvers = ["cvxopt"]
        with patch.dict("sys.modules", {"qpsolvers": mock_mod}):
            result = pc.check_qpsolvers_extra_availability()
        assert result is False

    def test_get_recommended_solver_no_proxsuite(self):
        import bssunfold.platform_check as pc
        orig_prox = pc.PROXSUITE_AVAILABLE
        orig_win = pc.is_windows
        pc.PROXSUITE_AVAILABLE = False
        pc.is_windows = False
        try:
            result = pc.get_recommended_solver()
        finally:
            pc.PROXSUITE_AVAILABLE = orig_prox
            pc.is_windows = orig_win
        assert result == "osqp"

    def test_get_recommended_solver_windows(self):
        import bssunfold.platform_check as pc
        orig_prox = pc.PROXSUITE_AVAILABLE
        orig_win = pc.is_windows
        pc.is_windows = True
        pc.PROXSUITE_AVAILABLE = True
        try:
            result = pc.get_recommended_solver()
        finally:
            pc.PROXSUITE_AVAILABLE = orig_prox
            pc.is_windows = orig_win
        assert result == "osqp"

    def test_get_recommended_solver_proxsuite_unix(self):
        import bssunfold.platform_check as pc
        orig_prox = pc.PROXSUITE_AVAILABLE
        orig_win = pc.is_windows
        pc.is_windows = False
        # Mock proxsuite import to succeed so check_proxsuite_availability sets True
        import builtins
        orig_import = builtins.__import__
        mock_proxsuite = MagicMock()
        def mock_import(name, *args, **kwargs):
            if name == "proxsuite":
                return mock_proxsuite
            return orig_import(name, *args, **kwargs)
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                result = pc.get_recommended_solver()
        finally:
            pc.PROXSUITE_AVAILABLE = orig_prox
            pc.is_windows = orig_win
        assert result == "proxqp"

    def test_get_available_solvers_all_checked(self):
        import bssunfold.platform_check as pc
        orig_prox = pc.PROXSUITE_AVAILABLE
        orig_win = pc.is_windows
        orig_jax = pc.JAX_AVAILABLE
        orig_qp = pc.QPSOLVERS_EXTRA_AVAILABLE
        # Mock the individual check functions to set specific values
        try:
            with patch.object(pc, "check_jax_availability", return_value=True), \
                 patch.object(pc, "check_proxsuite_availability", return_value=True), \
                 patch.object(pc, "check_qpsolvers_extra_availability", return_value=True):
                pc.JAX_AVAILABLE = True
                pc.PROXSUITE_AVAILABLE = True
                pc.QPSOLVERS_EXTRA_AVAILABLE = True
                solvers = pc.get_available_solvers()
        finally:
            pc.PROXSUITE_AVAILABLE = orig_prox
            pc.is_windows = orig_win
            pc.JAX_AVAILABLE = orig_jax
            pc.QPSOLVERS_EXTRA_AVAILABLE = orig_qp
        assert "ecos" in solvers
        assert "osqp" in solvers
        assert "proxqp" in solvers
        assert solvers["proxqp"] is True
        assert solvers["osqp"] is True
        assert solvers["jaxqp"] is True


# ============================================================================
# 3. comparison.py uncovered lines
# ============================================================================

class TestComparisonUncovered:
    def test_fluence_difference_zero_total(self):
        from bssunfold.utils.comparison import fluence_difference_percent
        s1 = np.zeros(5)
        s2 = np.ones(5)
        result = fluence_difference_percent(s1, s2)
        assert result == 0.0

    def test_fluence_difference_with_energy_bins(self):
        from bssunfold.utils.comparison import fluence_difference_percent
        s1 = np.ones(5) * 2.0
        s2 = np.ones(5) * 3.0
        bins = np.ones(5)
        result = fluence_difference_percent(s1, s2, energy_bins=bins)
        assert result == 50.0

    def test_fluence_difference_with_energy_bins_zero_total(self):
        from bssunfold.utils.comparison import fluence_difference_percent
        s1 = np.zeros(5)
        s2 = np.ones(5)
        bins = np.ones(5)
        result = fluence_difference_percent(s1, s2, energy_bins=bins)
        assert result == 0.0

    def test_energy_group_fluence_diff_length_mismatch(self):
        from bssunfold.utils.comparison import energy_group_fluence_diff
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            energy_group_fluence_diff(s1, s2, e)

    def test_energy_group_fluence_diff_no_thermal(self):
        from bssunfold.utils.comparison import energy_group_fluence_diff
        e = np.array([0.5, 1.0, 5.0])
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.1, 2.1, 3.1])
        result = energy_group_fluence_diff(s1, s2, e)
        assert "thermal" in result
        assert result["thermal"] == 0.0

    def test_dose_difference_length_mismatch(self):
        from bssunfold.utils.comparison import dose_difference_percent
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            dose_difference_percent(s1, s2, e)

    def test_dose_difference_zero_dose(self):
        from bssunfold.utils.comparison import dose_difference_percent
        s1 = np.zeros(5)
        s2 = np.ones(5)
        e = np.logspace(-6, 1, 5)
        result = dose_difference_percent(s1, s2, e)
        assert result == 0.0

    def test_fluence_averaged_energy_diff_length_mismatch(self):
        from bssunfold.utils.comparison import fluence_averaged_energy_diff
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            fluence_averaged_energy_diff(s1, s2, e)

    def test_fluence_averaged_energy_diff_zero_s1(self):
        from bssunfold.utils.comparison import fluence_averaged_energy_diff
        s1 = np.zeros(5)
        s2 = np.ones(5)
        e = np.logspace(-6, 1, 5)
        result = fluence_averaged_energy_diff(s1, s2, e)
        assert result == 0.0

    def test_dose_averaged_energy_diff_length_mismatch(self):
        from bssunfold.utils.comparison import dose_averaged_energy_diff
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            dose_averaged_energy_diff(s1, s2, e)

    def test_dose_averaged_energy_diff_zero_weight(self):
        from bssunfold.utils.comparison import dose_averaged_energy_diff
        s1 = np.zeros(5)
        s2 = np.ones(5)
        e = np.logspace(-6, 1, 5)
        result = dose_averaged_energy_diff(s1, s2, e)
        assert result == 0.0

    def test_spectral_shape_similarity_zero_sum(self):
        from bssunfold.utils.comparison import spectral_shape_similarity
        s1 = np.zeros(5)
        s2 = np.ones(5)
        result = spectral_shape_similarity(s1, s2)
        assert result == 0.0

    def test_spectral_shape_similarity_zero_norm(self):
        from bssunfold.utils.comparison import spectral_shape_similarity
        s1 = np.zeros(5)
        s2 = np.zeros(5)
        result = spectral_shape_similarity(s1, s2)
        assert result == 0.0

    def test_log_lethargy_correlation_length_mismatch(self):
        from bssunfold.utils.comparison import log_lethargy_correlation
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            log_lethargy_correlation(s1, s2, e)

    def test_log_lethargy_correlation_zero_std(self):
        from bssunfold.utils.comparison import log_lethargy_correlation
        # Both spectra are constant => log_e * const => std of lethargy may not be zero
        # Use truly zero spectra to force zero std
        s1 = np.zeros(5)
        s2 = np.zeros(5)
        e = np.logspace(-6, 1, 5)
        result = log_lethargy_correlation(s1, s2, e)
        assert result == 0.0

    def test_peak_location_error_length_mismatch(self):
        from bssunfold.utils.comparison import peak_location_error
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            peak_location_error(s1, s2, e)

    def test_peak_location_error_zero_peak(self):
        from bssunfold.utils.comparison import peak_location_error
        s1 = np.zeros(5)
        s2 = np.ones(5)
        e = np.logspace(-6, 1, 5)
        result = peak_location_error(s1, s2, e)
        assert result == 0.0

    def test_peak_width_error_length_mismatch(self):
        from bssunfold.utils.comparison import peak_width_error
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            peak_width_error(s1, s2, e)

    def test_peak_width_error_zero_max(self):
        from bssunfold.utils.comparison import peak_width_error
        s1 = np.zeros(5)
        s2 = np.ones(5)
        e = np.logspace(-6, 1, 5)
        result = peak_width_error(s1, s2, e)
        assert result == 0.0

    def test_dose_weighted_error_length_mismatch(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.ones(5)
        s2 = np.ones(5)
        e = np.ones(3)
        with pytest.raises(ValueError):
            dose_weighted_error(s1, s2, e)

    def test_dose_weighted_error_zero_total_weight(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.zeros(5)
        s2 = np.zeros(5)
        e = np.logspace(-6, 1, 5)
        result = dose_weighted_error(s1, s2, e)
        assert result == 0.0

    def test_response_matrix_consistency_no_mask(self):
        from bssunfold.utils.comparison import response_matrix_consistency
        s = np.ones(3)
        r = np.zeros(2)
        A = np.ones((2, 3))
        result = response_matrix_consistency(s, r, A)
        assert result == 0.0

    def test_compare_spectra_unknown_metric_string(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_spectra(s1, s2, metrics="nonexistent_metric")

    def test_compare_spectra_unknown_metric_in_list(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_spectra(s1, s2, metrics=["kl_divergence", "bad_metric"])

    def test_compare_spectra_eurados_metric_without_energy(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        result = compare_spectra(s1, s2, metrics="dose_difference_percent")
        assert np.isnan(result["dose_difference_percent"])

    def test_compare_spectra_response_matrix_consistency(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        e = np.logspace(-6, 1, 5)
        A = np.random.RandomState(42).rand(3, 5)
        r1 = A @ s1
        r2 = A @ s2
        result = compare_spectra(s1, s2, metrics="response_matrix_consistency",
                                 energy=e, readings1=r1, readings2=r2,
                                 response_matrix=A)
        assert "response_matrix_consistency_ref" in result

    def test_compare_spectra_eurados_with_energy(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(10)
        s2 = np.ones(10) * 1.1
        e = np.logspace(-6, 1, 10)
        result = compare_spectra(s1, s2, energy=e)
        assert "dose_difference_percent" in result
        assert "fluence_averaged_energy_diff" in result

    def test_compare_spectra_fluence_difference_percent_in_list(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        result = compare_spectra(s1, s2, metrics=["fluence_difference_percent"])
        assert "fluence_difference_percent" in result

    def test_compare_spectra_energy_group_in_list(self):
        from bssunfold.utils.comparison import compare_spectra
        e = np.logspace(-6, 1, 5)
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        result = compare_spectra(s1, s2, metrics=["energy_group_fluence_diff"], energy=e)
        assert any(k.startswith("energy_group_fluence_diff_") for k in result)

    def test_compare_spectra_response_matrix_in_list(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        e = np.logspace(-6, 1, 5)
        A = np.ones((3, 5))
        r = A @ s1
        result = compare_spectra(s1, s2, metrics=["response_matrix_consistency"],
                                 energy=e, readings1=r, response_matrix=A)
        assert "response_matrix_consistency_ref" in result

    def test_compare_spectra_exception_in_simple_metric(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        # Patch at the module level where it's used
        with patch("bssunfold.utils.comparison._METRIC_FUNCTIONS",
                   {"kl_divergence": lambda s1, s2: 1/0}):
            result = compare_spectra(s1, s2, metrics="kl_divergence")
        assert np.isnan(result["kl_divergence"])

    def test_compare_spectra_exception_in_eurados_metric(self):
        from bssunfold.utils.comparison import compare_spectra
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        e = np.logspace(-6, 1, 5)
        with patch("bssunfold.utils.comparison._METRIC_FUNCTIONS_WITH_PARAMS",
                   {"dose_difference_percent": lambda s1, s2, e, cc: 1/0}):
            result = compare_spectra(s1, s2, metrics="dose_difference_percent", energy=e)
        assert np.isnan(result["dose_difference_percent"])

    def test_compare_multiple(self):
        from bssunfold.utils.comparison import compare_multiple
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        s3 = np.ones(5) * 3
        result = compare_multiple([s1, s2, s3], metrics=["kl_divergence"])
        assert len(result) == 2  # 2 pairwise comparisons against reference

    def test_compare_multiple_with_labels(self):
        from bssunfold.utils.comparison import compare_multiple
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        result = compare_multiple([s1, s2], labels=["A", "B"], metrics=["kl_divergence"])
        assert len(result) == 1  # 1 pairwise comparison


# ============================================================================
# 4. detector.py uncovered lines (2246-2285)
# ============================================================================

class TestDetectorUncovered:
    def test_compare_dict_with_readings(self, detector):
        """Test dict input with readings/response_matrix."""
        s1 = np.ones(detector.n_energy_bins) * 0.01
        s2 = np.ones(detector.n_energy_bins) * 0.02
        r1 = np.ones(len(detector.detector_names))
        rm = np.ones((len(detector.detector_names), detector.n_energy_bins))
        result = detector.compare(
            {"spectrum": s1, "readings": r1, "response_matrix": rm},
            {"spectrum": s2, "readings": r1 * 2},
        )
        assert result is not None

    def test_compare_dict_with_phi_key(self, detector):
        """Test dict with 'Phi' key instead of 'spectrum'."""
        s1 = {"Phi": np.ones(detector.n_energy_bins) * 0.01}
        s2 = {"Phi": np.ones(detector.n_energy_bins) * 0.02}
        result = detector.compare(s1, s2)
        assert result is not None

    def test_compare_ndim_check(self, detector):
        """Test spectrum is not 1-D."""
        s1 = np.ones((3, 3))
        s2 = np.ones(detector.n_energy_bins) * 0.02
        with pytest.raises(ValueError, match="must be 1-D"):
            detector.compare(s1, s2)

    def test_compare_dict_no_spectrum_or_phi(self, detector):
        """Test dict without 'spectrum' or 'Phi' key."""
        s1 = {"bad_key": np.ones(detector.n_energy_bins)}
        s2 = np.ones(detector.n_energy_bins) * 0.02
        with pytest.raises(ValueError, match="no 'spectrum' or 'Phi' key"):
            detector.compare(s1, s2)


# ============================================================================
# 5. unfold_hybrid_parametric.py uncovered lines (29-52, 71, 94, 155-157, 172)
# ============================================================================

class TestHybridParametric:
    def test_parametric_initial_guess_zero_counts(self):
        from bssunfold.core.unfold_hybrid_parametric import _parametric_initial_guess
        E = np.logspace(-6, 1, 10)
        readings = {"det1": 0.0, "det2": 0.0}
        result = _parametric_initial_guess(E, readings, ["det1", "det2"],
                                            {"det1": np.ones(10), "det2": np.ones(10)})
        assert result.shape == (10,)
        assert np.all(result > 0)

    def test_parametric_initial_guess_normal(self):
        from bssunfold.core.unfold_hybrid_parametric import _parametric_initial_guess
        E = np.logspace(-6, 1, 10)
        readings = {"det1": 1.0, "det2": 2.0}
        sens = {"det1": np.ones(10), "det2": np.ones(10)}
        result = _parametric_initial_guess(E, readings, ["det1", "det2"], sens)
        assert result.shape == (10,)
        assert np.sum(result) > 0

    def test_landweber_convergence(self):
        from bssunfold.core.unfold_hybrid_parametric import _landweber_iteration
        np.random.seed(42)
        A = np.random.rand(3, 5) * 0.01
        x_true = np.ones(5)
        b = A @ x_true
        x0 = np.ones(5) * 0.1
        x, n_iter = _landweber_iteration(x0, A, b, step_size=0.01, max_iter=100, tolerance=1e-6)
        assert x.shape == (5,)
        assert n_iter > 0

    def test_mlem_convergence(self):
        from bssunfold.core.unfold_hybrid_parametric import _mlem_iteration
        np.random.seed(42)
        A = np.random.rand(3, 5) * 0.01
        x_true = np.ones(5)
        b = A @ x_true
        x0 = np.ones(5) * 0.1
        x, n_iter = _mlem_iteration(x0, A, b, max_iter=100, tolerance=1e-6)
        assert x.shape == (5,)
        assert n_iter > 0

    def test_solve_hybrid_unknown_method(self):
        from bssunfold.core.unfold_hybrid_parametric import solve_hybrid_parametric
        A = np.random.rand(3, 10) * 0.01
        b = np.ones(3)
        E = np.logspace(-6, 1, 10)
        ln = np.ones(10)
        with pytest.raises(ValueError, match="Unknown refinement method"):
            solve_hybrid_parametric(A, b, E, ln, refinement_method="bogus")


# ============================================================================
# 6. unfold_fruit_like.py uncovered lines (138-139, 153-163)
# ============================================================================

class TestFruitLike:
    def test_solve_fruit_like_with_initial_params(self):
        from bssunfold.core.unfold_fruit_like import solve_fruit_like
        np.random.seed(42)
        A = np.random.rand(5, 20) * 1e-3
        E = np.logspace(-6, 1, 20)
        ln_steps = np.ones(20) * 0.5
        b = A @ (np.ones(20) * ln_steps)
        init_params = {"A_th": 1e-6, "T_th": 0.025e-6, "A_epi": 1e-6,
                        "A_f": 1e-6, "T_ev": 2.0}
        spectrum, success, msg, nfev = solve_fruit_like(A, b, E, ln_steps,
                                                         initial_params=init_params)
        assert spectrum.shape == (20,)

    def test_solve_fruit_like_missing_lmfit(self):
        from bssunfold.core.unfold_fruit_like import solve_fruit_like
        A = np.random.rand(3, 5)
        b = np.ones(3)
        E = np.logspace(-6, 1, 5)
        ln = np.ones(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "lmfit":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="lmfit is required"):
                solve_fruit_like(A, b, E, ln)


# ============================================================================
# 7. unfold_bayesian_parametric.py uncovered lines (37, 41, 45, 49, 53, 97, 220-221)
# ============================================================================

class TestBayesianParametric:
    def test_log_prior_out_of_bounds(self):
        from bssunfold.core.unfold_bayesian_parametric import _log_prior
        assert _log_prior({"A_th": -1.0}) == -np.inf
        assert _log_prior({"T_th": 0.0}) == -np.inf
        assert _log_prior({"A_epi": -1.0}) == -np.inf
        assert _log_prior({"A_f": -1.0}) == -np.inf
        assert _log_prior({"T_ev": 0.0}) == -np.inf

    def test_log_prior_valid(self):
        from bssunfold.core.unfold_bayesian_parametric import _log_prior
        result = _log_prior({"A_th": 1e-6, "T_th": 0.025e-6, "A_epi": 1e-6,
                              "A_f": 1e-6, "T_ev": 2.0})
        assert np.isfinite(result)

    def test_log_posterior_invalid_prior(self):
        from bssunfold.core.unfold_bayesian_parametric import _log_posterior
        A = np.random.rand(3, 5)
        b = np.ones(3)
        E = np.logspace(-6, 1, 5)
        ln = np.ones(5)
        params = {"A_th": -1.0, "T_th": 0.025e-6, "A_epi": 1e-6,
                  "A_f": 1e-6, "T_ev": 2.0}
        result = _log_posterior(params, A, b, E, ln, 0.02)
        assert result == -np.inf

    def test_solve_bayesian_short_run(self):
        from bssunfold.core.unfold_bayesian_parametric import solve_bayesian_parametric
        np.random.seed(42)
        A = np.random.rand(5, 20) * 1e-3
        b = np.ones(5) * 1e-3
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        spectrum, success, msg, nfev = solve_bayesian_parametric(
            A, b, E, ln, n_samples=5, burn_in=2, random_state=42
        )
        assert spectrum.shape == (20,)


# ============================================================================
# 8. unfold_cvxpy.py uncovered lines (31-32, 52, 171-172)
# ============================================================================

class TestCvxpy:
    def test_solve_cvxpy_import_error(self):
        from bssunfold.core.unfold_cvxpy import _solve_cvxpy_problem
        A = np.random.rand(3, 5)
        b = np.ones(3)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="cvxpy is required"):
                _solve_cvxpy_problem(A, b, alpha=1e-4)

    def test_solve_cvxpy_all_solvers_fail(self):
        from bssunfold.core.unfold_cvxpy import _solve_cvxpy_problem
        A = np.random.rand(3, 5)
        b = np.ones(3)
        import cvxpy as cp
        # Mock cp.Problem.solve to always fail
        with patch.object(cp, "Problem") as MockProblem:
            mock_inst = MagicMock()
            mock_inst.status = "infeasible"
            mock_inst.value = None
            MockProblem.return_value = mock_inst
            # Also need to mock Variable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _solve_cvxpy_problem(A, b, alpha=1e-4)
            assert np.allclose(result, 0.0)


# ============================================================================
# 9. unfold_combined.py uncovered lines (129-131)
# ============================================================================

class TestCombined:
    def test_combined_invalid_method(self, detector):
        from bssunfold.core.unfold_combined import unfold_combined
        pipeline = [{"method": "nonexistent_method"}]
        readings = {name: 1.0 for name in detector.detector_names}
        with pytest.raises(ValueError, match="not found"):
            unfold_combined(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector.cc_icrp116,
                save_result_callback=lambda x: None,
                readings=readings,
                pipeline=pipeline,
            )

    def test_combined_two_stage(self, detector):
        from bssunfold.core.unfold_combined import unfold_combined
        readings = {name: 1.0 for name in detector.detector_names}
        pipeline = [
            {"method": "landweber", "params": {"max_iterations": 5}},
            {"method": "mlem", "params": {"max_iterations": 5},
             "use_as_initial": True, "store_intermediate": True},
        ]
        result = unfold_combined(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            pipeline=pipeline,
            verbose=True,
        )
        assert "spectrum" in result
        assert "pipeline_info" in result
        assert "intermediate_results" in result


# ============================================================================
# 10. unfold_qpsolvers.py uncovered lines
# ============================================================================

class TestQpsolvers:
    def test_solve_qpsolvers_import_error(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        A = np.random.rand(3, 5)
        b = np.ones(3)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "qpsolvers":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="qpsolvers is required"):
                solve_qpsolvers(A, b, alpha=1e-4)

    def test_solve_qpsolvers_unavailable_solver(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        A = np.random.rand(3, 5)
        b = np.ones(3)
        # Mock the available_solvers and solve_qp at the qpsolvers package level
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "qpsolvers":
                mod = MagicMock()
                mod.available_solvers = []
                mod.solve_qp = MagicMock(return_value=None)
                return mod
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = solve_qpsolvers(A, b, alpha=1e-4, solver="bogus")
        assert result is None

    def test_solve_qpsolvers_l1_norm(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5)
        result = solve_qpsolvers(A, b, alpha=1e-4, norm=1)
        assert result is not None
        assert result.shape == (10,)

    def test_solve_qpsolvers_l1_with_x0(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5)
        x0 = np.ones(10) * 0.5
        result = solve_qpsolvers(A, b, alpha=1e-4, norm=1, x0=x0)
        assert result is not None

    def test_solve_qpsolvers_l2_smoothness_order1(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5)
        result = solve_qpsolvers(A, b, alpha=1e-4, norm=2, smoothness_order=1)
        assert result is not None

    def test_solve_qpsolvers_l2_smoothness_order2(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10) * 0.01
        b = np.ones(5)
        result = solve_qpsolvers(A, b, alpha=1e-4, norm=2, smoothness_order=2)
        assert result is not None

    def test_solve_qpsolvers_unsupported_norm(self):
        from bssunfold.core.unfold_qpsolvers import solve_qpsolvers
        A = np.random.rand(3, 5)
        b = np.ones(3)
        with pytest.raises(ValueError, match="Unsupported norm type"):
            solve_qpsolvers(A, b, alpha=1e-4, norm=3)

    def test_unfold_qpsolvers_cosine_method(self, detector):
        from bssunfold.core.unfold_qpsolvers import unfold_qpsolvers
        readings = {name: 1.0 for name in detector.detector_names}
        init_spec = np.ones(detector.n_energy_bins)
        result = unfold_qpsolvers(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            initial_spectrum=init_spec,
            regularization_method="cosine",
            norm=2,
        )
        assert "spectrum" in result

    def test_unfold_qpsolvers_cosine_no_initial_raises(self, detector):
        from bssunfold.core.unfold_qpsolvers import unfold_qpsolvers
        readings = {name: 1.0 for name in detector.detector_names}
        with pytest.raises(ValueError, match="initial_spectrum must be provided"):
            unfold_qpsolvers(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector.cc_icrp116,
                save_result_callback=lambda x: None,
                readings=readings,
                regularization_method="cosine",
                norm=2,
            )


# ============================================================================
# 11. regularization.py fallback paths
# ============================================================================

class TestRegularizationFallbacks:
    def test_lcurve_fallback(self):
        from bssunfold.core.regularization import _lcurve_fallback
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        alpha = _lcurve_fallback(A, b, n_alphas=20)
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_gcv_fallback(self):
        from bssunfold.core.regularization import _gcv_fallback
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        alpha = _gcv_fallback(A, b, n_alphas=20)
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_dp_fallback(self):
        from bssunfold.core.regularization import _dp_fallback
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        alpha = _dp_fallback(A, b, noise_var=0.01, n_alphas=20)
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_lcurve_selection_fallback(self):
        from bssunfold.core.regularization import lcurve_selection
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pytikhonov":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            alpha = lcurve_selection(A, b, n_alphas=20)
        assert alpha > 0

    def test_gcv_selection_fallback(self):
        from bssunfold.core.regularization import gcv_selection
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pytikhonov":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            alpha = gcv_selection(A, b, n_alphas=20)
        assert alpha > 0

    def test_dp_selection_fallback(self):
        from bssunfold.core.regularization import discrepancy_principle_selection
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pytikhonov":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            alpha = discrepancy_principle_selection(A, b, n_alphas=20)
        assert alpha > 0

    def test_cosine_selection_zero_norm(self):
        from bssunfold.core.regularization import cosine_similarity_selection
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.ones(5)
        initial = np.ones(10) * 1e-30  # tiny but non-zero
        alpha = cosine_similarity_selection(A, b, initial, n_alphas=10)
        assert isinstance(alpha, float)

    def test_compare_regularization_methods_import_error(self):
        from bssunfold.core.regularization import compare_regularization_methods
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pytikhonov":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pytikhonov is required"):
                compare_regularization_methods(A, b)

    def test_randomization_experiment_import_error(self):
        from bssunfold.core.regularization import randomization_experiment
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pytikhonov":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pytikhonov is required"):
                randomization_experiment(A, b)


# ============================================================================
# 12. unfold_lmfit.py uncovered lines (107-108, 222)
# ============================================================================

class TestLmfit:
    def test_solve_lmfit_import_error(self):
        from bssunfold.core.unfold_lmfit import solve_lmfit
        A = np.random.rand(3, 5)
        b = np.ones(3)
        x0 = np.ones(5)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "lmfit":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="lmfit is required"):
                solve_lmfit(A, b, x0)

    def test_solve_lmfit_invalid_model(self):
        from bssunfold.core.unfold_lmfit import solve_lmfit
        A = np.random.rand(3, 5)
        b = np.ones(3)
        x0 = np.ones(5)
        with pytest.raises(ValueError, match="Unknown model_name"):
            solve_lmfit(A, b, x0, model_name="bogus")


# ============================================================================
# 13. unfold_parametric.py uncovered lines
# ============================================================================

class TestParametric:
    def test_parametric_import_error(self):
        from bssunfold.core.unfold_parametric import solve_parametric
        A = np.random.rand(5, 20)
        b = np.ones(5)
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "lmfit":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="lmfit is required"):
                solve_parametric(A, b, E, ln)

    def test_parametric_cvxpy_import_error(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A = np.random.rand(5, 20)
        b = np.ones(5)
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="cvxpy is required"):
                solve_parametric_cvxpy(A, b, E, ln)

    def test_parametric_qpsolvers_import_error(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A = np.random.rand(5, 20)
        b = np.ones(5)
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "qpsolvers":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="qpsolvers is required"):
                solve_parametric_qpsolvers(A, b, E, ln)

    def test_parametric_combined_cvxpy_import_error(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A = np.random.rand(5, 20)
        b = np.ones(5)
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name in ("cvxpy", "lmfit"):
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                solve_parametric_combined(A, b, E, ln, solver_backend="cvxpy")

    def test_parametric_combined_qpsolvers_import_error(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A = np.random.rand(5, 20)
        b = np.ones(5)
        E = np.logspace(-6, 1, 20)
        ln = np.ones(20)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name in ("qpsolvers", "lmfit"):
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                solve_parametric_combined(A, b, E, ln, solver_backend="qpsolvers")


# ============================================================================
# 14. _base_unfolder.py uncovered line (208)
# ============================================================================

class TestBaseUnfolder:
    def test_run_unfolding_with_calculate_errors(self, detector):
        readings = {name: 1.0 for name in detector.detector_names}
        x0 = np.ones(detector.n_energy_bins)

        def dummy_solver(A, b, **kwargs):
            return np.ones(A.shape[1]), A.shape[1], True

        from bssunfold.core._base_unfolder import run_unfolding
        result = run_unfolding(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            initial_spectrum=None,
            default_initial=x0,
            solve_func=dummy_solver,
            solve_kwargs={},
            method_name="test",
            extra_output={},
            calculate_errors=True,
            noise_level=0.01,
            n_montecarlo=5,
            random_state=42,
            save_result=False,
        )
        assert "spectrum" in result


# ============================================================================
# 15. Unfold via Detector interface with fallback solvers
# ============================================================================

class TestDetectorFallbackSolvers:
    def _mock_numba_and_clear(self):
        """Return context manager that forces pure-Python fallback."""
        import sys
        import types
        from unittest.mock import MagicMock

        saved = sys.modules.get("bssunfold.core._numba_jit")
        mock_numba = types.ModuleType("bssunfold.core._numba_jit")
        mock_numba.NUMBA_AVAILABLE = False
        for name in ["_doroshenko_inner", "_gravel_inner", "_kaczmarz_inner",
                      "_mlem_inner", "_compute_log_steps_jit", "_dose_weighted_mse_jit"]:
            setattr(mock_numba, name, MagicMock(side_effect=Exception("should not be called")))
        sys.modules["bssunfold.core._numba_jit"] = mock_numba

        class _Ctx:
            def __exit__(self, *a):
                if saved is not None:
                    sys.modules["bssunfold.core._numba_jit"] = saved
                else:
                    sys.modules.pop("bssunfold.core._numba_jit", None)
        return _Ctx()

    def _make_readings(self, detector):
        """Generate dummy readings dict."""
        return {name: 1.0 for name in detector.detector_names}

    def test_detector_doroshenko_fallback(self, detector):
        ctx = self._mock_numba_and_clear()
        try:
            readings = self._make_readings(detector)
            result = detector.unfold_doroshenko(readings, max_iterations=20)
        finally:
            ctx.__exit__()
        assert result is not None
        assert "spectrum" in result

    def test_detector_gravel_fallback(self, detector):
        ctx = self._mock_numba_and_clear()
        try:
            readings = self._make_readings(detector)
            result = detector.unfold_gravel(readings, max_iterations=10)
        finally:
            ctx.__exit__()
        assert result is not None

    def test_detector_kaczmarz_fallback(self, detector):
        ctx = self._mock_numba_and_clear()
        try:
            readings = self._make_readings(detector)
            result = detector.unfold_kaczmarz(readings, max_iterations=20)
        finally:
            ctx.__exit__()
        assert result is not None

    def test_detector_mlem_fallback(self, detector):
        ctx = self._mock_numba_and_clear()
        try:
            readings = self._make_readings(detector)
            result = detector.unfold_mlem(readings, max_iterations=20)
        finally:
            ctx.__exit__()
        assert result is not None


# ============================================================================
# 16. _numba_jit.py fallback paths via comparison.py
# ============================================================================

class TestNumbaFallbackViaComparison:
    def _mock_numba_and_clear(self):
        import sys
        import types
        from unittest.mock import MagicMock

        saved = sys.modules.get("bssunfold.core._numba_jit")
        mock_numba = types.ModuleType("bssunfold.core._numba_jit")
        mock_numba.NUMBA_AVAILABLE = False
        for name in ["_doroshenko_inner", "_gravel_inner", "_kaczmarz_inner",
                      "_mlem_inner", "_compute_log_steps_jit", "_dose_weighted_mse_jit"]:
            setattr(mock_numba, name, MagicMock(side_effect=Exception("should not be called")))
        sys.modules["bssunfold.core._numba_jit"] = mock_numba

        class _Ctx:
            def __exit__(self, *a):
                if saved is not None:
                    sys.modules["bssunfold.core._numba_jit"] = saved
                else:
                    sys.modules.pop("bssunfold.core._numba_jit", None)
        return _Ctx()

    def test_compute_log_steps_fallback(self):
        from bssunfold.utils.comparison import _compute_log_steps
        E = np.logspace(-6, 1, 10)
        ctx = self._mock_numba_and_clear()
        try:
            result = _compute_log_steps(E)
        finally:
            ctx.__exit__()
        assert result.shape == (10,)
        assert np.all(result > 0)

    def test_dose_weighted_error_fallback(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.ones(5)
        s2 = np.ones(5) * 2
        e = np.logspace(-6, 1, 5)
        ctx = self._mock_numba_and_clear()
        try:
            result = dose_weighted_error(s1, s2, e)
        finally:
            ctx.__exit__()
        assert isinstance(result, float)


# ============================================================================
# 17. unfold_parametric2.py uncovered lines
# ============================================================================

class TestParametric2:
    def test_bon95_qpsolvers_import_error(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A = np.random.rand(3, 10)
        b = np.ones(3)
        E = np.logspace(-6, 1, 10)
        ln = np.ones(10)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "qpsolvers":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="qpsolvers is required"):
                solve_bon95_qpsolvers(A, b, E, ln)

    def test_bon95_cvxpy_import_error(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_cvxpy
        A = np.random.rand(3, 10)
        b = np.ones(3)
        E = np.logspace(-6, 1, 10)
        ln = np.ones(10)
        import builtins
        orig_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("mocked")
            return orig_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="cvxpy is required"):
                solve_bon95_cvxpy(A, b, E, ln)


# ============================================================================
# 18. unfold_lmfit.py line 222 path (x0 from kwargs)
# ============================================================================

class TestLmfitX0FromKwargs:
    def test_unfold_lmfit_with_explicit_x0(self, detector):
        from bssunfold.core.unfold_lmfit import unfold_lmfit
        readings = {name: 1.0 for name in detector.detector_names}
        result = unfold_lmfit(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            initial_spectrum=np.ones(detector.n_energy_bins),
        )
        assert "spectrum" in result


# ============================================================================
# 19. Parametric SQP solver paths (cvxpy, qpsolvers, combined)
# ============================================================================

class TestParametricSqpSolvers:
    """Test solve_parametric_cvxpy, solve_parametric_qpsolvers, solve_parametric_combined
    with real data to exercise the full SQP code paths."""

    def _make_data(self, n_det=6, n_energy=15):
        np.random.seed(42)
        E = np.logspace(-6, 1, n_energy)
        A = np.random.rand(n_det, n_energy) * 1e-3
        b = A @ np.ones(n_energy)
        log_steps = np.ones(n_energy) * 0.1
        return A, b, E, log_steps

    def test_cvxpy_convergence(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        spectrum, success, msg, nfev = solve_parametric_cvxpy(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=30, tol=1e-4
        )
        assert spectrum.shape == (15,)

    def test_cvxpy_max_iter_reached(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_cvxpy(
            A, b, E, log_steps, alpha=1e-4, max_iter=1, tol=1e-10
        )
        assert not success
        assert "Max iterations" in msg

    def test_cvxpy_specific_backend(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_cvxpy(
            A, b, E, log_steps, alpha=1e-4, solver_backend="cvxpy:ECOS", max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_cvxpy_all_solvers_fail(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        with patch("cvxpy.Problem") as MockProblem:
            mock_prob = MagicMock()
            mock_prob.status = "infeasible"
            mock_prob.solve.side_effect = Exception("fail")
            MockProblem.return_value = mock_prob
            spectrum, success, msg, nfev = solve_parametric_cvxpy(
                A, b, E, log_steps, alpha=1e-4, max_iter=2
            )
            assert not success
            assert "QP subproblem failed" in msg

    def test_qpsolvers_convergence(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        spectrum, success, msg, nfev = solve_parametric_qpsolvers(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=30, tol=1e-4
        )
        assert spectrum.shape == (15,)

    def test_qpsolvers_max_iter_reached(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_qpsolvers(
            A, b, E, log_steps, alpha=1e-4, max_iter=1, tol=1e-10
        )
        assert not success
        assert "Max iterations" in msg

    def test_qpsolvers_specific_backend(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_qpsolvers(
            A, b, E, log_steps, alpha=1e-4, solver_backend="qpsolvers:osqp", max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_qpsolvers_all_fail(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        with patch("qpsolvers.solve_qp", side_effect=Exception("QP fail")):
            spectrum, success, msg, nfev = solve_parametric_qpsolvers(
                A, b, E, log_steps, alpha=1e-4, max_iter=3, tol=1e-10
            )
            assert not success
            assert "QP subproblem failed" in msg

    def test_qpsolvers_returns_none(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        with patch("qpsolvers.solve_qp", return_value=None):
            spectrum, success, msg, nfev = solve_parametric_qpsolvers(
                A, b, E, log_steps, alpha=1e-4, max_iter=2, tol=1e-10
            )
            assert not success
            assert "None" in msg

    def test_combined_cvxpy(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_combined(
            A, b, E, log_steps, alpha=1e-4, solver_backend="cvxpy"
        )
        assert spectrum.shape == (15,)

    def test_combined_qpsolvers(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_combined(
            A, b, E, log_steps, alpha=1e-4, solver_backend="qpsolvers"
        )
        assert spectrum.shape == (15,)

    def test_combined_auto_cvxpy(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric_combined(
            A, b, E, log_steps, alpha=1e-4, solver_backend="auto"
        )
        assert spectrum.shape == (15,)
        assert "QP refinement" in msg

    def test_combined_cvxpy_refinement_fails(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        with patch("cvxpy.Problem") as MockProblem:
            mock_prob = MagicMock()
            mock_prob.status = "infeasible"
            mock_prob.solve.side_effect = Exception("fail")
            MockProblem.return_value = mock_prob
            spectrum, success, msg, nfev = solve_parametric_combined(
                A, b, E, log_steps, alpha=1e-4, solver_backend="cvxpy"
            )
            assert "QP refinement failed" in msg

    def test_combined_qpsolvers_refinement_fails(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        with patch("qpsolvers.solve_qp", return_value=None):
            spectrum, success, msg, nfev = solve_parametric_combined(
                A, b, E, log_steps, alpha=1e-4, solver_backend="qpsolvers"
            )
            assert "QP refinement failed" in msg

    def test_combined_unknown_library(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        with pytest.raises(ValueError, match="Unknown solver library"):
            solve_parametric_combined(A, b, E, log_steps, solver_backend="numpyro")

    def test_gcv_select_alpha(self):
        from bssunfold.core.unfold_parametric import _gcv_select_alpha
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        alpha = _gcv_select_alpha(A, b, E, log_steps, init, n_coarse=5, n_refine=3)
        assert alpha > 0

    def test_gcv_select_alpha_small_matrix(self):
        from bssunfold.core.unfold_parametric import _gcv_select_alpha
        A = np.random.rand(2, 2) * 0.01
        b = np.ones(2) * 0.01
        E = np.logspace(-6, 1, 2)
        log_steps = np.ones(2) * 0.1
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        alpha = _gcv_select_alpha(A, b, E, log_steps, init, n_coarse=3, n_refine=2)
        assert alpha > 0

    def test_parse_solver_backend_auto(self):
        from bssunfold.core.unfold_parametric import _parse_solver_backend
        assert _parse_solver_backend("auto") == ("auto", "default")

    def test_parse_solver_backend_colon(self):
        from bssunfold.core.unfold_parametric import _parse_solver_backend
        assert _parse_solver_backend("cvxpy:ECOS") == ("cvxpy", "ECOS")
        assert _parse_solver_backend("qpsolvers") == ("qpsolvers", "default")

    def test_resolve_cvxpy_solvers_default(self):
        from bssunfold.core.unfold_parametric import _resolve_cvxpy_solvers
        solvers = _resolve_cvxpy_solvers("default")
        assert len(solvers) > 0

    def test_resolve_cvxpy_solvers_specific(self):
        from bssunfold.core.unfold_parametric import _resolve_cvxpy_solvers
        solvers = _resolve_cvxpy_solvers("ECOS")
        assert solvers[0] == "ECOS"

    def test_resolve_qpsolver_name_default(self):
        from bssunfold.core.unfold_parametric import _resolve_qpsolver_name
        name = _resolve_qpsolver_name("default")
        assert name in ("osqp", "ecos")

    def test_resolve_qpsolver_name_specific(self):
        from bssunfold.core.unfold_parametric import _resolve_qpsolver_name
        name = _resolve_qpsolver_name("osqp")
        assert name == "osqp"

    def test_parametric_model(self):
        from bssunfold.core.unfold_parametric import parametric_model
        E = np.logspace(-6, 1, 15)
        result = parametric_model(E, 1.0, 0.1, 0.5, 2.0, 0.5, 0.5)
        assert result.shape == (15,)
        assert np.all(result >= 0)

    def test_clamp_params(self):
        from bssunfold.core.unfold_parametric import _clamp_params, _get_param_bounds
        bounds = _get_param_bounds()
        params = {'b': 0.01, 'beta_prime': 0.001, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        clamped = _clamp_params(params, bounds)
        for name, (lo, hi) in bounds.items():
            if lo is not None:
                assert clamped[name] >= lo
            if hi is not None:
                assert clamped[name] <= hi

    def test_get_initial_params(self):
        from bssunfold.core.unfold_parametric import _get_initial_params
        result = _get_initial_params({'b': 2.0})
        assert result['b'] == 2.0
        assert 'P_th' in result

    def test_get_param_bounds(self):
        from bssunfold.core.unfold_parametric import _get_param_bounds
        bounds = _get_param_bounds()
        assert 'b' in bounds
        assert 'P_th' in bounds

    def test_compute_jacobian_at_boundary(self):
        from bssunfold.core.unfold_parametric import _compute_jacobian
        E = np.logspace(-6, 1, 15)
        log_steps = np.ones(15) * 0.1
        # P_th and P_epi at upper boundary (1.0) to trigger backward diff
        params = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 1.0, 'P_epi': 1.0}
        J = _compute_jacobian(E, log_steps, params)
        assert J.shape == (15, 6)

    def test_compute_jacobian_at_lower_boundary(self):
        from bssunfold.core.unfold_parametric import _compute_jacobian
        E = np.logspace(-6, 1, 15)
        log_steps = np.ones(15) * 0.1
        # P_th and P_epi at lower boundary (0.0) to trigger backward diff
        params = {'b': 0.5, 'beta_prime': 1e-4, 'alpha': 0.0, 'beta': 0.1, 'P_th': 0.0, 'P_epi': 0.0}
        J = _compute_jacobian(E, log_steps, params)
        assert J.shape == (15, 6)

    def test_unfold_parametric_alpha_auto(self, detector):
        from bssunfold.core.unfold_parametric import unfold_parametric
        readings = {name: 1.0 for name in detector.detector_names}
        result = unfold_parametric(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            optimizer="lmfit",
            alpha_auto=True,
        )
        assert "spectrum" in result

    def test_parametric_cvxpy_initial_params_update(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        init = {'b': 2.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.3, 'P_epi': 0.7}
        spectrum, success, msg, nfev = solve_parametric_cvxpy(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_parametric_qpsolvers_initial_params_update(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        init = {'b': 2.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.3, 'P_epi': 0.7}
        spectrum, success, msg, nfev = solve_parametric_qpsolvers(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_parametric_combined_initial_params(self):
        from bssunfold.core.unfold_parametric import solve_parametric_combined
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.3, 'P_epi': 0.7}
        spectrum, success, msg, nfev = solve_parametric_combined(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, solver_backend="cvxpy"
        )
        assert spectrum.shape == (15,)

    def test_parametric_cvxpy_converge_in_loop(self):
        from bssunfold.core.unfold_parametric import solve_parametric_cvxpy
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        spectrum, success, msg, nfev = solve_parametric_cvxpy(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=50, tol=1e-2
        )
        assert spectrum.shape == (15,)

    def test_parametric_qpsolvers_converge_in_loop(self):
        from bssunfold.core.unfold_parametric import solve_parametric_qpsolvers
        A, b, E, log_steps = self._make_data()
        init = {'b': 1.0, 'beta_prime': 0.1, 'alpha': 0.5, 'beta': 2.0, 'P_th': 0.5, 'P_epi': 0.5}
        spectrum, success, msg, nfev = solve_parametric_qpsolvers(
            A, b, E, log_steps, initial_params=init, alpha=1e-4, max_iter=50, tol=1e-2
        )
        assert spectrum.shape == (15,)

    def test_check_fit_quality_ok(self):
        from bssunfold.core.unfold_parametric import _check_fit_quality
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_fit_quality(0.01, np.ones(5), "test")

    def test_check_fit_quality_warns(self):
        from bssunfold.core.unfold_parametric import _check_fit_quality
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fit_quality(100.0, np.ones(5), "test")
            assert len(w) > 0

    def test_unfold_parametric_cvxpy(self, detector):
        from bssunfold.core.unfold_parametric import unfold_parametric
        readings = {name: 1.0 for name in detector.detector_names}
        result = unfold_parametric(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            optimizer="cvxpy",
            alpha=1e-4,
            max_iter=10,
        )
        assert "spectrum" in result

    def test_unfold_parametric_qpsolvers(self, detector):
        from bssunfold.core.unfold_parametric import unfold_parametric
        readings = {name: 1.0 for name in detector.detector_names}
        result = unfold_parametric(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            optimizer="qpsolvers",
            alpha=1e-4,
            max_iter=10,
        )
        assert "spectrum" in result

    def test_unfold_parametric_combined(self, detector):
        from bssunfold.core.unfold_parametric import unfold_parametric
        readings = {name: 1.0 for name in detector.detector_names}
        result = unfold_parametric(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector.cc_icrp116,
            save_result_callback=lambda x: None,
            readings=readings,
            optimizer="combined",
            alpha=1e-4,
        )
        assert "spectrum" in result

    def test_unfold_parametric_unknown_optimizer(self, detector):
        from bssunfold.core.unfold_parametric import unfold_parametric
        readings = {name: 1.0 for name in detector.detector_names}
        with pytest.raises(ValueError, match="Unknown optimizer"):
            unfold_parametric(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector.cc_icrp116,
                save_result_callback=lambda x: None,
                readings=readings,
                optimizer="bogus",
            )

    def test_residuals_with_alpha(self):
        from bssunfold.core.unfold_parametric import _residuals
        import lmfit
        E = np.logspace(-6, 1, 15)
        log_steps = np.ones(15) * 0.1
        A = np.random.rand(6, 15) * 1e-3
        b = np.ones(6) * 0.1
        params = lmfit.Parameters()
        params.add('b', value=1.0, min=0.5, max=2.0)
        params.add('beta_prime', value=0.1, min=1e-4, max=1.0)
        params.add('alpha', value=0.5, min=0.0, max=5.0)
        params.add('beta', value=2.0, min=0.1, max=20.0)
        params.add('P_th', value=0.5, min=0.0, max=1.0)
        params.add('P_epi', value=0.5, min=0.0, max=1.0)
        initial_param_vec = np.array([1.0, 0.1, 0.5, 2.0, 0.5, 0.5])
        result = _residuals(params, A, b, E, log_steps, reg_alpha=0.01, initial_param_vec=initial_param_vec)
        assert result.shape[0] == 6 + 6

    def test_residuals_no_alpha(self):
        from bssunfold.core.unfold_parametric import _residuals
        import lmfit
        E = np.logspace(-6, 1, 15)
        log_steps = np.ones(15) * 0.1
        A = np.random.rand(6, 15) * 1e-3
        b = np.ones(6) * 0.1
        params = lmfit.Parameters()
        params.add('b', value=1.0, min=0.5, max=2.0)
        params.add('beta_prime', value=0.1, min=1e-4, max=1.0)
        params.add('alpha', value=0.5, min=0.0, max=5.0)
        params.add('beta', value=2.0, min=0.1, max=20.0)
        params.add('P_th', value=0.5, min=0.0, max=1.0)
        params.add('P_epi', value=0.5, min=0.0, max=1.0)
        result = _residuals(params, A, b, E, log_steps, reg_alpha=0.0, initial_param_vec=None)
        assert result.shape == (6,)

    def test_find_initial_params(self):
        from bssunfold.core.unfold_parametric import _find_initial_params
        A, b, E, log_steps = self._make_data()
        result = _find_initial_params(A, b, E, log_steps, n_grid=3, return_top=1)
        assert isinstance(result, dict)
        assert 'P_th' in result


# ============================================================================
# 20. Parametric2 SQP solver paths and full pipeline
# ============================================================================

class TestParametric2SqpSolvers:
    """Test solve_bon95_cvxpy, solve_bon95_qpsolvers, solve_bon95_combined
    and solve_parametric2 with different optimizers."""

    def _make_data(self, n_det=6, n_energy=15):
        np.random.seed(42)
        E = np.logspace(-6, 1, n_energy)
        A = np.random.rand(n_det, n_energy) * 1e-3
        b = A @ np.ones(n_energy)
        ln_steps = np.ones(n_energy) * 0.1
        return A, b, E, ln_steps

    def test_bon95_cvxpy_convergence(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_cvxpy
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_cvxpy(
            A, b, E, ln_steps, alpha=1e-4, max_iter=20, tol=1e-4
        )
        assert spectrum.shape == (15,)

    def test_bon95_cvxpy_max_iter(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_cvxpy
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_cvxpy(
            A, b, E, ln_steps, alpha=1e-4, max_iter=1, tol=1e-10
        )
        assert not success

    def test_bon95_cvxpy_all_fail(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_cvxpy
        A, b, E, ln_steps = self._make_data()
        with patch("cvxpy.Problem") as MockProblem:
            mock_prob = MagicMock()
            mock_prob.status = "infeasible"
            mock_prob.solve.side_effect = Exception("fail")
            MockProblem.return_value = mock_prob
            spectrum, success, msg, nfev = solve_bon95_cvxpy(
                A, b, E, ln_steps, alpha=1e-4, max_iter=2
            )
            assert not success
            assert "QP subproblem failed" in msg

    def test_bon95_qpsolvers_convergence(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_qpsolvers(
            A, b, E, ln_steps, alpha=1e-4, max_iter=20, tol=1e-4
        )
        assert spectrum.shape == (15,)

    def test_bon95_qpsolvers_max_iter(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_qpsolvers(
            A, b, E, ln_steps, alpha=1e-4, max_iter=1, tol=1e-10
        )
        assert not success

    def test_bon95_qpsolvers_with_initial(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        init = {'b': 1.0, 'Tf': 0.5, 'c': 1.0}
        spectrum, success, msg, nfev = solve_bon95_qpsolvers(
            A, b, E, ln_steps, initial_params=init, alpha=1e-4, max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_bon95_qpsolvers_all_fail(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        with patch("qpsolvers.solve_qp", side_effect=Exception("QP fail")):
            spectrum, success, msg, nfev = solve_bon95_qpsolvers(
                A, b, E, ln_steps, alpha=1e-4, max_iter=3, tol=1e-10
            )
            assert not success

    def test_bon95_qpsolvers_returns_none(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        with patch("qpsolvers.solve_qp", return_value=None):
            spectrum, success, msg, nfev = solve_bon95_qpsolvers(
                A, b, E, ln_steps, alpha=1e-4, max_iter=2, tol=1e-10
            )
            assert not success
            assert "None" in msg

    def test_bon95_combined(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_combined
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_combined(
            A, b, E, ln_steps, alpha=1e-4, max_iter_qp=5, tol_qp=1e-4
        )
        assert spectrum.shape == (15,)

    def test_bon95_combined_cvxpy(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_combined
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_combined(
            A, b, E, ln_steps, alpha=1e-4, solver_backend="cvxpy", max_iter_qp=5
        )
        assert spectrum.shape == (15,)

    def test_bon95_combined_qpsolvers(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_combined
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_combined(
            A, b, E, ln_steps, alpha=1e-4, solver_backend="qpsolvers", max_iter_qp=5
        )
        assert spectrum.shape == (15,)

    def test_bon95_combined_cvxpy_fail_fallback(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_combined
        A, b, E, ln_steps = self._make_data()
        with patch("cvxpy.Problem") as MockProblem:
            mock_prob = MagicMock()
            mock_prob.status = "infeasible"
            mock_prob.solve.side_effect = Exception("fail")
            MockProblem.return_value = mock_prob
            spectrum, success, msg, nfev = solve_bon95_combined(
                A, b, E, ln_steps, alpha=1e-4, solver_backend="cvxpy", max_iter_qp=2
            )
            assert spectrum.shape == (15,)

    def test_solve_parametric2_grid(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric2(
            A, b, E, ln_steps, optimizer="grid",
            max_iter=10, tol_chi2=100.0
        )
        assert spectrum.shape == (15,)

    def test_solve_parametric2_cvxpy(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric2(
            A, b, E, ln_steps, optimizer="cvxpy", alpha=1e-4,
            max_iter_qp=5, max_iter=10, tol_chi2=100.0
        )
        assert spectrum.shape == (15,)

    def test_solve_parametric2_qpsolvers(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric2(
            A, b, E, ln_steps, optimizer="qpsolvers", alpha=1e-4,
            max_iter_qp=5, max_iter=10, tol_chi2=100.0
        )
        assert spectrum.shape == (15,)

    def test_solve_parametric2_combined(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric2(
            A, b, E, ln_steps, optimizer="combined", alpha=1e-4,
            max_iter_qp=5, max_iter=10, tol_chi2=100.0
        )
        assert spectrum.shape == (15,)

    def test_solve_parametric2_unknown(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            solve_parametric2(A, b, E, ln_steps, optimizer="bogus")

    def test_directed_divergence_converge(self):
        from bssunfold.core.unfold_parametric2 import directed_divergence_iteration
        A, b, E, ln_steps = self._make_data()
        phi0 = np.ones(15) / 15
        phi, n_iter, chi2, conv = directed_divergence_iteration(
            A, b, E, ln_steps, phi0, max_iter=50, tol_chi2=100.0
        )
        assert phi.shape == (15,)

    def test_directed_divergence_rel_change(self):
        from bssunfold.core.unfold_parametric2 import directed_divergence_iteration
        A, b, E, ln_steps = self._make_data()
        phi0 = np.ones(15) / 15
        phi, n_iter, chi2, conv = directed_divergence_iteration(
            A, b, E, ln_steps, phi0, max_iter=50, tol_rel=1.0
        )
        assert phi.shape == (15,)

    def test_bon95_parametric(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_parametric
        A, b, E, ln_steps = self._make_data()
        best, chi2, top = solve_bon95_parametric(A, b, E, ln_steps, top_n=2)
        assert 'b' in best
        assert 'Tf' in best

    def test_bon95_combined_cvxpy_refinement_fails(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_combined
        A, b, E, ln_steps = self._make_data()
        with patch("cvxpy.Problem") as MockProblem:
            mock_prob = MagicMock()
            mock_prob.status = "infeasible"
            mock_prob.solve.side_effect = Exception("fail")
            MockProblem.return_value = mock_prob
            spectrum, success, msg, nfev = solve_bon95_combined(
                A, b, E, ln_steps, alpha=1e-4, solver_backend="cvxpy", max_iter_qp=2
            )
            # cvxpy fails, should fallback to qpsolvers
            assert spectrum.shape == (15,)

    def test_bon95_qpsolvers_specific_backend(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_qpsolvers
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_qpsolvers(
            A, b, E, ln_steps, alpha=1e-4, solver_backend="qpsolvers:osqp", max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_bon95_cvxpy_specific_backend(self):
        from bssunfold.core.unfold_parametric2 import solve_bon95_cvxpy
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_bon95_cvxpy(
            A, b, E, ln_steps, alpha=1e-4, solver_backend="cvxpy:ECOS", max_iter=5
        )
        assert spectrum.shape == (15,)

    def test_directed_divergence_with_meas(self):
        from bssunfold.core.unfold_parametric2 import directed_divergence_iteration
        A, b, E, ln_steps = self._make_data()
        phi0 = np.ones(15) / 15
        phi, n_iter, chi2, conv = directed_divergence_iteration(
            A, b, E, ln_steps, phi0, b_meas=b, max_iter=50, tol_chi2=100.0
        )
        assert phi.shape == (15,)

    def test_solve_parametric2_cvxpy_with_meas(self):
        from bssunfold.core.unfold_parametric2 import solve_parametric2
        A, b, E, ln_steps = self._make_data()
        spectrum, success, msg, nfev = solve_parametric2(
            A, b, E, ln_steps, optimizer="cvxpy", alpha=1e-4,
            b_meas=b, max_iter_qp=5, max_iter=10, tol_chi2=100.0
        )
        assert spectrum.shape == (15,)

    def test_parse_solver_backend(self):
        from bssunfold.core.unfold_parametric2 import _parse_solver_backend
        assert _parse_solver_backend("auto") == ("auto", "default")
        assert _parse_solver_backend("cvxpy:ECOS") == ("cvxpy", "ECOS")
        assert _parse_solver_backend("qpsolvers") == ("qpsolvers", "default")

    def test_resolve_cvxpy_solvers(self):
        from bssunfold.core.unfold_parametric2 import _resolve_cvxpy_solvers
        solvers = _resolve_cvxpy_solvers("default")
        assert len(solvers) > 0
        solvers = _resolve_cvxpy_solvers("ECOS")
        assert solvers[0] == "ECOS"

    def test_resolve_qpsolver_name(self):
        from bssunfold.core.unfold_parametric2 import _resolve_qpsolver_name
        name = _resolve_qpsolver_name("default")
        assert name in ("osqp", "ecos")
        assert _resolve_qpsolver_name("osqp") == "osqp"

    def test_bon95_shape_bounds(self):
        from bssunfold.core.unfold_parametric2 import _get_bon95_shape_bounds
        bounds = _get_bon95_shape_bounds()
        assert 'b' in bounds
        assert 'Tf' in bounds

    def test_clamp_bon95_shape(self):
        from bssunfold.core.unfold_parametric2 import _clamp_bon95_shape, _get_bon95_shape_bounds
        bounds = _get_bon95_shape_bounds()
        params = {'b': 0.001, 'Tf': 0.001, 'c': 0.001}
        clamped = _clamp_bon95_shape(params, bounds)
        for name, (lo, hi) in bounds.items():
            if lo is not None:
                assert clamped[name] >= lo
            if hi is not None:
                assert clamped[name] <= hi


# ============================================================================
# 21. Comparison.py remaining edge cases
# ============================================================================

class TestComparisonEdgeCases:
    """Cover remaining comparison.py edge cases."""

    def _mock_numba_fallback(self):
        """Force numba fallback for comparison.py paths."""
        import sys
        import types
        saved = sys.modules.get("bssunfold.core._numba_jit")
        mock_numba = types.ModuleType("bssunfold.core._numba_jit")
        mock_numba.NUMBA_AVAILABLE = False
        sys.modules["bssunfold.core._numba_jit"] = mock_numba

        class _Ctx:
            def __exit__(self, *a):
                if saved is not None:
                    sys.modules["bssunfold.core._numba_jit"] = saved
                else:
                    sys.modules.pop("bssunfold.core._numba_jit", None)
        return _Ctx()

    def test_compute_log_steps_numba_fallback(self):
        from bssunfold.utils.comparison import _compute_log_steps
        E = np.logspace(-6, 1, 10)
        ctx = self._mock_numba_fallback()
        try:
            result = _compute_log_steps(E)
        finally:
            ctx.__exit__()
        assert result.shape == (10,)

    def test_extract_cc_array_dict_preferred(self):
        from bssunfold.utils.comparison import _extract_cc_array
        E = np.logspace(-6, 1, 10)
        cc = {"AP": np.ones(10), "ISO": np.ones(10) * 2}
        result = _extract_cc_array(cc, E, preferred_geom="AP")
        np.testing.assert_array_equal(result, np.ones(10))

    def test_extract_cc_array_dict_any_key(self):
        from bssunfold.utils.comparison import _extract_cc_array
        E = np.logspace(-6, 1, 10)
        cc = {"ISO": np.ones(10) * 2}
        result = _extract_cc_array(cc, E, preferred_geom="AP")
        np.testing.assert_array_equal(result, np.ones(10) * 2)

    def test_extract_cc_array_dict_empty(self):
        from bssunfold.utils.comparison import _extract_cc_array
        E = np.logspace(-6, 1, 10)
        cc = {"E_MeV": E}
        result = _extract_cc_array(cc, E, preferred_geom="AP")
        np.testing.assert_array_equal(result, np.ones(10))

    def test_entropy_difference_different(self):
        from bssunfold.utils.comparison import entropy_difference_percent
        p = np.array([0.1, 0.3, 0.6])
        q = np.array([0.6, 0.3, 0.1])
        result = entropy_difference_percent(p, q)
        assert isinstance(result, float)

    def test_peak_location_zero_peak(self):
        from bssunfold.utils.comparison import peak_location_error
        s1 = np.array([1.0, 0.0, 0.0])
        s2 = np.array([1.0, 2.0, 3.0])
        E = np.array([0.0, 2.0, 3.0])
        result = peak_location_error(s1, s2, E)
        assert result == 0.0

    def test_peak_width_no_half_max(self):
        from bssunfold.utils.comparison import peak_width_error
        s1 = np.array([0.0, 0.0, 0.0])
        s2 = np.array([1.0, 2.0, 3.0])
        E = np.array([1.0, 2.0, 3.0])
        result = peak_width_error(s1, s2, E)
        assert result == 0.0

    def test_dose_weighted_zero_weights(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.array([1.0, 1.0, 1.0])
        s2 = np.array([1.0, 1.0, 1.0])
        E = np.array([1.0, 2.0, 3.0])
        cc = {"AP": np.array([0.0, 0.0, 0.0])}
        result = dose_weighted_error(s1, s2, E, cc_icrp116=cc)
        assert result == 0.0

    def test_dose_weighted_numba_fallback(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        E = np.logspace(-6, 1, 3)
        ctx = self._mock_numba_fallback()
        try:
            result = dose_weighted_error(s1, s2, E)
        finally:
            ctx.__exit__()
        assert result >= 0

    def test_spectral_shape_zero_norm(self):
        from bssunfold.utils.comparison import spectral_shape_similarity
        s1 = np.array([0.0, 0.0, 0.0])
        s2 = np.array([0.0, 0.0, 0.0])
        result = spectral_shape_similarity(s1, s2)
        assert result == 0.0

    def test_peak_width_fwhm_zero(self):
        from bssunfold.utils.comparison import peak_width_error
        s1 = np.array([0.0, 0.0, 0.0])
        s2 = np.array([0.0, 0.0, 0.0])
        E = np.array([1.0, 2.0, 3.0])
        result = peak_width_error(s1, s2, E)
        assert result == 0.0

    def test_dose_weighted_zero_total_weight(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        E = np.array([1e-20, 2e-20, 3e-20])
        cc = {"AP": np.array([1e-30, 1e-30, 1e-30])}
        result = dose_weighted_error(s1, s2, E, cc_icrp116=cc)
        assert result == 0.0

    def test_extract_cc_array_ndarray(self):
        from bssunfold.utils.comparison import _extract_cc_array
        E = np.logspace(-6, 1, 10)
        cc = np.ones(10) * 2.0
        result = _extract_cc_array(cc, E)
        np.testing.assert_array_equal(result, cc)

    def test_dose_weighted_no_cc(self):
        from bssunfold.utils.comparison import dose_weighted_error
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        E = np.logspace(-6, 1, 3)
        result = dose_weighted_error(s1, s2, E, cc_icrp116=None)
        assert result >= 0
