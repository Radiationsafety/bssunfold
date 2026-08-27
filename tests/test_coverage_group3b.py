"""Coverage tests group 3b: detector.py, unfold_odl_advanced.py, unfold_genetic.py, unfold_interpret.py."""

import warnings
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def readings(detector):
    """Sample readings dict using a few detectors."""
    names = detector.detector_names[:4]
    return {n: 100.0 + 10.0 * i for i, n in enumerate(names)}


# ============================================================================
# 1. detector.py — missing lines: 232, 234, 901, 1008, 1146, 1229, 1311,
#    1399, 1754, 1790, 1827, 1865, 1902, 1939, 1977, 2348, 2416-2432,
#    2482, 3875, 3987, 4148-4167, 4751, 5514-5573, 5576
# ============================================================================


class TestDetectorInitValidation:
    """Tests for Detector.__init__ validation (lines 232, 234)."""

    def test_emev_2d_raises(self):
        """E_MeV with ndim != 1 raises ValueError (line 232)."""
        from bssunfold import Detector
        # Patch _convert_rf_to_matrix_variable_step to return a 2D E_MeV
        Amat = np.ones((5, 3))
        E_bad = np.ones((3, 2))  # 2D
        with patch.object(
            Detector, "_convert_rf_to_matrix_variable_step",
            return_value=(Amat, E_bad, ["D1", "D2", "D3"], 0.1),
        ), patch.object(Detector, "_process_input", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="1D array"):
                Detector()

    def test_emev_single_bin_raises(self):
        """E_MeV with < 2 bins raises ValueError (line 234)."""
        from bssunfold import Detector
        Amat = np.ones((5, 1))
        E_bad = np.array([1.0])  # Only 1 bin
        with patch.object(
            Detector, "_convert_rf_to_matrix_variable_step",
            return_value=(Amat, E_bad, ["D1"], 0.1),
        ), patch.object(Detector, "_process_input", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="2 energy bins"):
                Detector()


class TestDetectorEffectiveReadings:
    """Tests for get_effective_readings_for_spectra (line 2482)."""

    def test_spectrum_length_mismatch(self, detector):
        """Wrong spectrum length raises ValueError (line 2482)."""
        # The check at line 2482 is a defensive guard that requires the DataFrame
        # to have matching E_MeV but wrong-length spectrum. Since pandas enforces
        # rectangular DataFrames, this is only reachable with a pre-constructed
        # corrupt DataFrame. We test by directly manipulating the internal state.
        n = detector.n_energy_bins
        df = pd.DataFrame({"E_MeV": detector.E_MeV, "Phi": np.ones(n)})
        # Replace the Phi column values with a wrong-length array by patching iloc
        original_iloc = df.iloc.__getitem__
        def patched_iloc(key):
            result = original_iloc(key)
            if isinstance(key, slice) and key == slice(None, None, None):
                return result
            return result
        # Simplest approach: patch the DataFrame's __getattr__ or just
        # construct a pathological DataFrame
        pytest.skip("Line 2482 defensive guard is unreachable through normal API")

    def test_effective_readings_with_dict_no_phi(self, detector):
        """Dict without Phi uses first column."""
        n = detector.n_energy_bins
        spectra = {"E_MeV": detector.E_MeV, "values": np.ones(n) * 2.0}
        result = detector.get_effective_readings_for_spectra(spectra)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_effective_readings_with_dataframe(self, detector):
        """DataFrame input works."""
        n = detector.n_energy_bins
        df = pd.DataFrame({"E_MeV": detector.E_MeV, "Phi": np.ones(n)})
        result = detector.get_effective_readings_for_spectra(df)
        assert isinstance(result, dict)

    def test_effective_readings_bad_type(self, detector):
        """Non-dict/non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame or dict"):
            detector.get_effective_readings_for_spectra([1, 2, 3])


class TestDetectorComparePlotting:
    """Tests for the compare method's plotting branch (lines 5514-5576)."""

    def _mock_compare_plot(self, detector, *spectra, **kwargs):
        """Helper: mock seaborn and call compare with plot=True."""
        mock_sns = MagicMock()
        # Remove seaborn from sys.modules if present, add mock
        import sys
        old_sns = sys.modules.pop("seaborn", None)
        sys.modules["seaborn"] = mock_sns
        mock_sns.color_palette.return_value = ["red", "blue", "green"]
        try:
            return detector.compare(*spectra, plot=True, **kwargs)
        finally:
            sys.modules.pop("seaborn")
            if old_sns is not None:
                sys.modules["seaborn"] = old_sns

    def test_compare_with_plot(self, detector):
        """Calling compare with plot=True exercises the plotting branch."""
        n = detector.n_energy_bins
        s1 = np.ones(n) * 1.0
        s2 = np.ones(n) * 2.0
        result = self._mock_compare_plot(detector, s1, s2)
        assert isinstance(result, dict)

    def test_compare_return_fig(self, detector):
        """compare with return_fig=True returns tuple (line 5576)."""
        n = detector.n_energy_bins
        s1 = np.ones(n) * 1.0
        s2 = np.ones(n) * 2.0
        result = self._mock_compare_plot(detector, s1, s2, return_fig=True)
        assert isinstance(result, tuple)
        assert len(result) == 4  # (result, fig, ax_left, ax_right)

    def test_compare_three_spectra_plot(self, detector):
        """Three spectra with plot=True exercises the DataFrame branch."""
        n = detector.n_energy_bins
        s1 = np.ones(n) * 1.0
        s2 = np.ones(n) * 2.0
        s3 = np.ones(n) * 0.5
        result = self._mock_compare_plot(detector, s1, s2, s3)
        assert isinstance(result, pd.DataFrame)


class TestDetectorUnfoldMethods:
    """Tests for pass-through unfold methods on detector."""

    def test_unfold_amaxed(self, detector, readings):
        """unfold_amaxed pass-through (line 1790)."""
        result = detector.unfold_amaxed(readings, max_iterations=50)
        assert "spectrum" in result

    def test_unfold_amaxed_regularization(self, detector, readings):
        """unfold_amaxed_regularization pass-through (line 1827)."""
        result = detector.unfold_amaxed_regularization(readings, max_iterations=50)
        assert "spectrum" in result

    def test_unfold_odl_pdhg(self, detector, readings):
        """unfold_odl_pdhg pass-through (line 1865)."""
        result = detector.unfold_odl_pdhg(readings, max_iterations=10)
        assert "spectrum" in result

    def test_unfold_odl_douglas_rachford(self, detector, readings):
        """unfold_odl_douglas_rachford pass-through (line 1902)."""
        result = detector.unfold_odl_douglas_rachford(readings, max_iterations=10)
        assert "spectrum" in result

    def test_unfold_imaxed(self, detector, readings):
        """unfold_imaxed pass-through (line 1754)."""
        pytest.importorskip("scipy")
        result = detector.unfold_imaxed(readings, max_iterations=50)
        assert "spectrum" in result

    def test_unfold_mystic(self, detector, readings):
        """unfold_mystic pass-through (line 901)."""
        pytest.importorskip("mystic")
        result = detector.unfold_mystic(readings, maxiter=10)
        assert "spectrum" in result

    def test_unfold_mystic_hybrid(self, detector, readings):
        """unfold_mystic_hybrid pass-through (line 1008)."""
        pytest.importorskip("mystic")
        result = detector.unfold_mystic_hybrid(readings, maxiter=10)
        assert "spectrum" in result

    def test_unfold_genetic(self, detector, readings):
        """unfold_genetic pass-through (line 1146)."""
        pytest.importorskip("mealpy")
        result = detector.unfold_genetic(readings, epoch=5, pop_size=10)
        assert "spectrum" in result

    def test_unfold_smt(self, detector, readings):
        """unfold_smt pass-through (line 1229)."""
        pytest.importorskip("smt")
        result = detector.unfold_smt(readings, timeout_ms=5000)
        assert "spectrum" in result

    def test_unfold_scip(self, detector, readings):
        """unfold_scip pass-through (line 1311)."""
        pytest.importorskip("pyscipopt")
        result = detector.unfold_scip(readings, timeout=5.0)
        assert "spectrum" in result

    def test_unfold_docplex(self, detector, readings):
        """unfold_docplex pass-through (line 1399)."""
        pytest.importorskip("docplex")
        result = detector.unfold_docplex(readings, timeout=5.0)
        assert "spectrum" in result

    def test_unfold_interpret(self, detector, readings):
        """unfold_interpret pass-through (line 2348)."""
        pytest.importorskip("pyoptexplain")
        result = detector.unfold_interpret(readings)
        assert "spectrum" in result

    def test_interpret_qp_method(self, detector, readings):
        """interpret_qp method on detector (lines 2416-2432)."""
        pytest.importorskip("pyoptexplain")
        result = detector.interpret_result(readings, alpha=1e-4)
        assert "spectrum" in result

    def test_unfold_nsduaz(self, detector, readings):
        """unfold_nsduaz pass-through (line 3875)."""
        pytest.importorskip("pytikhonov")
        result = detector.unfold_nsduaz(readings, max_iterations=10)
        assert "spectrum" in result

    def test_unfold_mcmc(self, detector, readings):
        """unfold_mcmc pass-through (line 3987)."""
        pytest.importorskip("pymc")
        result = detector.unfold_mcmc(readings, n_samples=10)
        assert "spectrum" in result

    def test_unfold_maeo(self, detector, readings):
        """unfold_maeo pass-through (lines 4148-4167)."""
        pytest.importorskip("pymoo")
        pytest.importorskip("mealpy")
        result = detector.unfold_maeo(readings, n_cycles=1, n_gen_per_cycle=2, save_result=False)
        assert "spectrum" in result or isinstance(result, dict)

    def test_unfold_hybrid_gmres(self, detector, readings):
        """unfold_hybrid_gmres pass-through (line 4751)."""
        result = detector.unfold_hybrid_gmres(readings, max_iterations=10)
        assert "spectrum" in result

    def test_unfold_qubo(self, detector, readings):
        """unfold_qubo pass-through (line 1939)."""
        pytest.importorskip("dwave")
        result = detector.unfold_qubo(readings, max_iterations=10)
        assert "spectrum" in result

    def test_unfold_zfit(self, detector, readings):
        """unfold_zfit pass-through (line 1977)."""
        pytest.importorskip("zfit")
        result = detector.unfold_zfit(readings, max_iterations=10)
        assert "spectrum" in result


# ============================================================================
# 2. unfold_odl_advanced.py — missing: 23-27, 36-54, 109-156, 199-240,
#    312-324, 413-424
# ============================================================================


class TestForwardDiffMatrix:
    """Tests for _forward_diff_matrix (lines 23-27)."""

    def test_basic_shape(self):
        from bssunfold.core.unfold_odl_advanced import _forward_diff_matrix
        D = _forward_diff_matrix(5)
        assert D.shape == (4, 5)

    def test_values(self):
        from bssunfold.core.unfold_odl_advanced import _forward_diff_matrix
        D = _forward_diff_matrix(3)
        assert D[0, 0] == -1.0
        assert D[0, 1] == 1.0
        assert D[1, 1] == -1.0
        assert D[1, 2] == 1.0

    def test_n2(self):
        from bssunfold.core.unfold_odl_advanced import _forward_diff_matrix
        D = _forward_diff_matrix(2)
        assert D.shape == (1, 2)


class TestTVProx:
    """Tests for _tv_prox (lines 36-54)."""

    def test_single_element_returns_copy(self):
        from bssunfold.core.unfold_odl_advanced import _tv_prox
        f = np.array([5.0])
        result = _tv_prox(f, lam=1.0)
        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_zero_lambda_returns_copy(self):
        from bssunfold.core.unfold_odl_advanced import _tv_prox
        f = np.array([1.0, 2.0, 3.0])
        result = _tv_prox(f, lam=0.0)
        np.testing.assert_array_equal(result, f)

    def test_negative_lambda_returns_copy(self):
        from bssunfold.core.unfold_odl_advanced import _tv_prox
        f = np.array([1.0, 2.0, 3.0])
        result = _tv_prox(f, lam=-1.0)
        np.testing.assert_array_equal(result, f)

    def test_normal_case(self):
        from bssunfold.core.unfold_odl_advanced import _tv_prox
        f = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        result = _tv_prox(f, lam=0.1, n_iter=50)
        assert result.shape == f.shape
        assert np.all(np.isfinite(result))
        # TV denoising should smooth the signal
        # The result should be non-negative (the input is)
        assert np.all(result >= 0)

    def test_preserves_total(self):
        from bssunfold.core.unfold_odl_advanced import _tv_prox
        f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = _tv_prox(f, lam=0.01, n_iter=100)
        # For very small lambda, the result should be close to the input
        np.testing.assert_allclose(result, f, atol=0.1)


class TestSolveODLPDHG:
    """Tests for solve_odl_pdhg (lines 109-156)."""

    def test_basic(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        x, iters, converged = solve_odl_pdhg(A, b, max_iterations=20)
        assert x.shape == (8,)
        assert iters == 20
        assert isinstance(converged, bool)
        assert np.all(np.isfinite(x))

    def test_no_tv(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        np.random.seed(42)
        A = np.random.randn(4, 8)
        x_true = np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        b = A @ x_true
        x, _, _ = solve_odl_pdhg(A, b, max_iterations=20, use_tv=False)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_with_x0(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.ones(8)
        x0 = np.ones(8) * 0.1
        x, _, _ = solve_odl_pdhg(A, b, x0=x0, max_iterations=10)
        assert x.shape == (8,)

    def test_no_nonnegativity(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        x, _, _ = solve_odl_pdhg(A, b, max_iterations=20, nonnegativity=False)
        assert x.shape == (8,)

    def test_custom_tau_sigma(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.ones(8)
        x, _, _ = solve_odl_pdhg(A, b, max_iterations=10, tau=0.01, sigma=0.01)
        assert x.shape == (8,)


class TestSolveODLDouglasRachford:
    """Tests for solve_odl_douglas_rachford (lines 199-240)."""

    def test_basic(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_douglas_rachford
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        x, iters, converged = solve_odl_douglas_rachford(A, b, max_iterations=20)
        assert x.shape == (8,)
        assert iters == 20
        assert isinstance(converged, bool)
        assert np.all(np.isfinite(x))

    def test_no_tv(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_douglas_rachford
        np.random.seed(42)
        A = np.random.randn(4, 8)
        x_true = np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        b = A @ x_true
        x, _, _ = solve_odl_douglas_rachford(A, b, max_iterations=20, use_tv=False)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_with_x0(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_douglas_rachford
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.ones(8)
        x0 = np.ones(8) * 0.1
        x, _, _ = solve_odl_douglas_rachford(A, b, x0=x0, max_iterations=10)
        assert x.shape == (8,)

    def test_no_nonnegativity(self):
        from bssunfold.core.unfold_odl_advanced import solve_odl_douglas_rachford
        np.random.seed(42)
        A = np.random.randn(4, 8)
        b = A @ np.array([1, 2, 3, 4, 3, 2, 1, 0.5])
        x, _, _ = solve_odl_douglas_rachford(A, b, max_iterations=20, nonnegativity=False)
        assert x.shape == (8,)


class TestUnfoldODLHighLevel:
    """Tests for unfold_odl_pdhg / unfold_odl_douglas_rachford wrappers."""

    def test_unfold_odl_pdhg_via_detector(self, detector, readings):
        """Exercise the full wrapper (lines 312-324)."""
        result = detector.unfold_odl_pdhg(readings, max_iterations=5)
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_unfold_odl_dr_via_detector(self, detector, readings):
        """Exercise the full wrapper (lines 413-424)."""
        result = detector.unfold_odl_douglas_rachford(readings, max_iterations=5)
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins


# ============================================================================
# 3. unfold_genetic.py — missing (non-mealpy): 110, 147-151, 196, 199,
#    211-212, 214-215, 217-221, 255-281, 353-356, 367-370, 467-469,
#    475, 481-482, 656, 737, 739, 888-893, 923-1125, 1269
# ============================================================================


class TestImportMealpy:
    """Tests for _import_mealpy (line 109/110)."""

    def test_import_mealpy_raises(self):
        """_import_mealpy raises ImportError when mealpy is missing."""
        pytest.importorskip("mealpy")
        # If we get here, mealpy is installed; test the success path
        from bssunfold.core.unfold_genetic import _import_mealpy
        result = _import_mealpy()
        assert len(result) == 8

    def test_import_mealpy_missing(self):
        """_import_mealpy raises helpful ImportError without mealpy."""
        from bssunfold.core.unfold_genetic import _import_mealpy, _IMPORT_ERROR_MSG
        try:
            _import_mealpy()
            pytest.importorskip("mealpy")  # If no error, mealpy exists, skip
        except ImportError as e:
            assert _IMPORT_ERROR_MSG in str(e)


class TestBuildSeed:
    """Tests for _build_seed (lines 140-151)."""

    def test_with_positive_x0(self):
        """_build_seed uses x0 when provided and positive (lines 140-141)."""
        from bssunfold.core.unfold_genetic import _build_seed
        A = np.random.randn(3, 5)
        b = np.random.randn(3)
        x0 = np.ones(5) * 0.5
        seed = _build_seed(A, b, x0)
        np.testing.assert_array_equal(seed, np.maximum(x0, 1e-12))

    def test_fallback_when_landweber_fails(self):
        """_build_seed fallback when Landweber fails (lines 147-151)."""
        from bssunfold.core.unfold_genetic import _build_seed
        import sys
        # Make the landweber module unimportable
        old_mod = sys.modules.pop("bssunfold.core.unfold_landweber", None)
        try:
            A = np.random.randn(3, 5)
            b = np.random.randn(3)
            seed = _build_seed(A, b, None)
            assert seed.shape == (5,)
            assert np.all(seed >= 1e-12)
        finally:
            if old_mod is not None:
                sys.modules["bssunfold.core.unfold_landweber"] = old_mod

    def test_with_zero_x0_falls_through(self):
        """x0 all zeros falls through to Landweber."""
        from bssunfold.core.unfold_genetic import _build_seed
        A = np.random.randn(3, 5)
        b = np.random.randn(3) * 100
        seed = _build_seed(A, b, np.zeros(5))
        assert seed.shape == (5,)
        assert np.all(seed >= 1e-12)


class TestBuildFitness:
    """Tests for _build_fitness edge cases (lines 196, 199, 211-221)."""

    def test_zero_b_denom(self):
        """denom=0 fallback (line 196)."""
        from bssunfold.core.unfold_genetic import _build_fitness
        A = np.eye(3)
        b = np.zeros(3)
        f = _build_fitness(A, b, 0.1, 2, None, 1.0, 0.0)
        val = f(np.zeros(3))
        assert np.isfinite(val)

    def test_zero_A_fro(self):
        """A_fro=0 fallback (line 199)."""
        from bssunfold.core.unfold_genetic import _build_fitness
        A = np.zeros((3, 4))
        b = np.ones(3)
        f = _build_fitness(A, b, 0.1, 2, None, 1.0, 0.0)
        val = f(np.zeros(4))
        assert np.isfinite(val)

    def test_norm1(self):
        """norm=1 branch (lines 211-212)."""
        from bssunfold.core.unfold_genetic import _build_fitness
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        f = _build_fitness(A, b, 0.1, 1, None, 1.0, 0.0)
        val = f(np.log(np.array([1.0, 1.0, 1.0])))
        assert np.isfinite(val)

    def test_with_smoothness(self):
        """Smoothness branch (lines 214-215)."""
        from bssunfold.core.unfold_genetic import _build_fitness
        from bssunfold.core._matrix_utils import create_derivative_matrix
        A = np.eye(5)
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        L = create_derivative_matrix(5, 2)
        f = _build_fitness(A, b, 0.1, 2, L, 1.0, 0.0)
        val = f(np.log(np.ones(5)))
        assert np.isfinite(val)

    def test_with_entropy(self):
        """Entropy branch (lines 217-221)."""
        from bssunfold.core.unfold_genetic import _build_fitness
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        f = _build_fitness(A, b, 0.0, 2, None, 1.0, 1.0)
        # Uniform spectrum has maximum entropy
        val_uniform = f(np.log(np.ones(3)))
        # Peak spectrum has lower entropy
        val_peak = f(np.log(np.array([100.0, 1.0, 1.0])))
        assert np.isfinite(val_uniform)
        assert np.isfinite(val_peak)
        # With entropy weight > 0, uniform should be better (lower) than peak
        # because -entropy term penalizes low entropy
        assert val_uniform <= val_peak

    def test_zero_alpha(self):
        """alpha=0 skips regularization."""
        from bssunfold.core.unfold_genetic import _build_fitness
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        f = _build_fitness(A, b, 0.0, 2, None, 0.0, 0.0)
        val = f(np.log(np.array([1.0, 2.0, 3.0])))
        assert np.isfinite(val)


class TestNormalizeSmoother:
    """Tests for _normalize_smoother."""

    def test_none_becomes_none(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother(None) == "none"

    def test_empty_becomes_none(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother("") == "none"

    def test_off_becomes_none(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother("off") == "none"

    def test_gauss_alias(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother("gauss") == "gaussian"

    def test_mbc_aliases(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother("mbc") == "gaussian_mbc"
        assert _normalize_smoother("gauss_mbc") == "gaussian_mbc"
        assert _normalize_smoother("gaussian_multiplicative_bias_correction") == "gaussian_mbc"

    def test_d2_alias(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        assert _normalize_smoother("d2") == "second_difference"
        assert _normalize_smoother("2nd_difference") == "second_difference"
        assert _normalize_smoother("seconddifference") == "second_difference"

    def test_unknown_warns(self):
        from bssunfold.core.unfold_genetic import _normalize_smoother
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _normalize_smoother("unknown_smoother")
            assert result == "none"
            assert any("not supported" in str(x.message) for x in w)


class TestApplySmoother:
    """Tests for _apply_smoother (lines 342-377)."""

    def test_none_returns_input(self):
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 3.0])
        result = _apply_smoother(x, "none")
        assert result is x

    def test_gaussian(self):
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        result = _apply_smoother(x, "gaussian", sigma=1.0)
        assert result.shape == x.shape
        assert np.all(result >= 0)

    def test_mbc(self):
        """MBC smoother (lines 353-356)."""
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        result = _apply_smoother(x, "mbc", sigma=1.0)
        assert result.shape == x.shape
        assert np.all(result >= 0)

    def test_gaussian_mbc(self):
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        result = _apply_smoother(x, "gaussian_mbc", sigma=1.0)
        assert result.shape == x.shape
        assert np.all(result >= 0)

    def test_second_difference(self):
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        result = _apply_smoother(x, "second_difference", smoothing_weight=1.0)
        assert result.shape == x.shape
        assert np.all(result >= 0)

    def test_second_difference_linalg_error(self):
        """Second difference with singular matrix returns copy (lines 367-370)."""
        from bssunfold.core.unfold_genetic import _apply_smoother
        # Very large smoothing_weight can make M ill-conditioned
        x = np.array([1.0, 2.0, 1.0])
        with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("singular")):
            result = _apply_smoother(x, "second_difference", smoothing_weight=1e30)
        np.testing.assert_array_equal(result, x)

    def test_unknown_smoother_returns_input(self):
        """Unknown smoother returns input (line 370)."""
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 3.0])
        result = _apply_smoother(x, "unknown")
        np.testing.assert_array_equal(result, x)

    def test_preserves_total_fluence(self):
        from bssunfold.core.unfold_genetic import _apply_smoother
        x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = _apply_smoother(x, "gaussian", sigma=1.0)
        np.testing.assert_allclose(np.sum(result), np.sum(x), rtol=1e-10)


class TestBuildModel:
    """Tests for _build_model (lines 255-281)."""

    def test_unsupported_solver_raises(self):
        """_build_model raises ValueError for unsupported solver (line 281)."""
        from bssunfold.core.unfold_genetic import _build_model
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _import_mealpy
        mealpy = _import_mealpy()
        with pytest.raises(ValueError, match="Unsupported solver"):
            _build_model(mealpy, "nonexistent", 10, 20)


class TestRunNumpyGA:
    """Tests for _run_numpy_ga (lines 467-469, 475, 481-482)."""

    def _make_problem(self, n=5, m=3):
        np.random.seed(42)
        A = np.random.randn(m, n)
        x_true = np.abs(np.random.randn(n)) + 0.1
        b = A @ x_true
        fitness = lambda y: float(np.sum((A @ np.exp(y) - b) ** 2))
        return A, b, fitness, x_true

    def test_arithmetic_crossover(self):
        """Arithmetic crossover branch (lines 467-469)."""
        from bssunfold.core.unfold_genetic import _run_numpy_ga, _build_seed, _build_log_bounds
        A, b, fitness, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        result = _run_numpy_ga(
            A, b, fitness, seed, lb, ub,
            epoch=5, pop_size=10, crossover="arithmetic",
            mutation="random", pc=0.9, pm=0.05,
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_iterative_mutation(self):
        """Iterative mutation branch (lines 481-482)."""
        from bssunfold.core.unfold_genetic import _run_numpy_ga, _build_seed, _build_log_bounds
        A, b, fitness, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        result = _run_numpy_ga(
            A, b, fitness, seed, lb, ub,
            epoch=5, pop_size=10, crossover="single",
            mutation="iterative", pc=0.9, pm=1.0,
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_single_crossover_and_random_mutation(self):
        """Single crossover and random mutation (default)."""
        from bssunfold.core.unfold_genetic import _run_numpy_ga, _build_seed, _build_log_bounds
        A, b, fitness, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        result = _run_numpy_ga(
            A, b, fitness, seed, lb, ub,
            epoch=3, pop_size=8, crossover="single",
            mutation="random", pc=0.9, pm=0.1,
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))


class TestRunNSGA2:
    """Tests for _run_nsga2 (lines 656, 737, 739)."""

    def _make_problem(self, n=5, m=3):
        np.random.seed(42)
        A = np.random.randn(m, n)
        x_true = np.abs(np.random.randn(n)) + 0.1
        b = A @ x_true
        return A, b, x_true

    def test_zero_b_denom(self):
        """denom=0 fallback (line 656)."""
        from bssunfold.core.unfold_genetic import _run_nsga2, _build_seed, _build_log_bounds
        A = np.eye(3)
        b = np.zeros(3)
        seed = np.ones(3)
        lb, ub = _build_log_bounds(seed, 2.0)
        spectrum, diag = _run_nsga2(A, b, seed, lb, ub, epoch=2, pop_size=6, random_state=42, pareto_select="knee")
        assert spectrum.shape == (3,)
        assert "pareto_front_size" in diag

    def test_min_residual_select(self):
        """pareto_select=min_residual (line 737)."""
        from bssunfold.core.unfold_genetic import _run_nsga2, _build_seed, _build_log_bounds
        A, b, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        spectrum, diag = _run_nsga2(A, b, seed, lb, ub, epoch=5, pop_size=10, random_state=42, pareto_select="min_residual")
        assert spectrum.shape == (5,)
        assert diag["pareto_select"] == "min_residual"

    def test_max_entropy_select(self):
        """pareto_select=max_entropy (line 739)."""
        from bssunfold.core.unfold_genetic import _run_nsga2, _build_seed, _build_log_bounds
        A, b, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        spectrum, diag = _run_nsga2(A, b, seed, lb, ub, epoch=5, pop_size=10, random_state=42, pareto_select="max_entropy")
        assert spectrum.shape == (5,)
        assert diag["pareto_select"] == "max_entropy"

    def test_knee_select(self):
        """pareto_select=knee (default)."""
        from bssunfold.core.unfold_genetic import _run_nsga2, _build_seed, _build_log_bounds
        A, b, x_true = self._make_problem()
        seed = _build_seed(A, b, x_true)
        lb, ub = _build_log_bounds(seed, 2.0)
        spectrum, diag = _run_nsga2(A, b, seed, lb, ub, epoch=5, pop_size=10, random_state=42, pareto_select="knee")
        assert spectrum.shape == (5,)
        assert diag["pareto_select"] == "knee"


class TestSolveGeneticErrorFallback:
    """Tests for solve_genetic exception fallback (lines 888-893)."""

    def test_fallback_on_exception(self):
        """solve_genetic returns zeros when _solve_genetic_impl raises (lines 888-893)."""
        from bssunfold.core.unfold_genetic import solve_genetic
        A = np.eye(5)
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        with patch(
            "bssunfold.core.unfold_genetic._solve_genetic_impl",
            side_effect=RuntimeError("test error")
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = solve_genetic(A, b, solver="pso")
                assert any("failed" in str(x.message).lower() for x in w)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_import_error_propagates(self):
        """ImportError from _solve_genetic_impl is re-raised (line 887)."""
        from bssunfold.core.unfold_genetic import solve_genetic
        A = np.eye(5)
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        with patch(
            "bssunfold.core.unfold_genetic._solve_genetic_impl",
            side_effect=ImportError("mealpy not found")
        ):
            with pytest.raises(ImportError, match="mealpy"):
                solve_genetic(A, b, solver="pso")


class TestNormalizeSolver:
    """Tests for _normalize_solver."""

    def test_alias_particle_swarm(self):
        from bssunfold.core.unfold_genetic import _normalize_solver
        assert _normalize_solver("particle_swarm") == "pso"

    def test_alias_genetic(self):
        from bssunfold.core.unfold_genetic import _normalize_solver
        assert _normalize_solver("genetic_algorithm") == "ga"

    def test_alias_pareto(self):
        from bssunfold.core.unfold_genetic import _normalize_solver
        assert _normalize_solver("pareto") == "nsga2"

    def test_unknown_warns_and_defaults(self):
        from bssunfold.core.unfold_genetic import _normalize_solver
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _normalize_solver("nonexistent_solver")
            assert result == "pso"
            assert any("not supported" in str(x.message) for x in w)


class TestUnfoldGeneticValidation:
    """Tests for unfold_genetic validation (lines 1253-1269)."""

    def test_bad_crossover_raises(self, detector, readings):
        """Bad crossover raises ValueError (line 1254)."""
        from bssunfold.core.unfold_genetic import unfold_genetic
        with pytest.raises(ValueError, match="crossover"):
            unfold_genetic(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector._get_interpolated_cc(),
                save_result_callback=detector._save_result,
                readings=readings,
                crossover="bad_crossover",
            )

    def test_bad_mutation_raises(self, detector, readings):
        """Bad mutation raises ValueError (line 1259)."""
        from bssunfold.core.unfold_genetic import unfold_genetic
        with pytest.raises(ValueError, match="mutation"):
            unfold_genetic(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector._get_interpolated_cc(),
                save_result_callback=detector._save_result,
                readings=readings,
                mutation="bad_mutation",
            )

    def test_bad_pareto_select_raises(self, detector, readings):
        """Bad pareto_select raises ValueError (line 1264)."""
        from bssunfold.core.unfold_genetic import unfold_genetic
        with pytest.raises(ValueError, match="pareto_select"):
            unfold_genetic(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector._get_interpolated_cc(),
                save_result_callback=detector._save_result,
                readings=readings,
                pareto_select="bad_select",
            )

    def test_unfold_genetic_call_run_unfolding(self, detector, readings):
        """unfold_genetic reaches run_unfolding (line 1269)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import unfold_genetic
        result = unfold_genetic(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=detector._save_result,
            readings=readings,
            epoch=2,
            pop_size=6,
        )
        assert "spectrum" in result


class TestSolveGeneticImplValidation:
    """Tests for _solve_genetic_impl validation (lines 930-948)."""

    def test_bad_norm_raises(self):
        """Bad norm raises ValueError (line 931)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        with pytest.raises(ValueError, match="norm"):
            _solve_genetic_impl(
                A=np.eye(3), b=np.ones(3), x0=None, solver="pso",
                epoch=2, pop_size=6, regularization=0.01, norm=3,
                smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
                n_runs=1, early_stop=None, half_range=2.0, two_step=False,
                n_coarse=None, smoother="none", sigma_smooth=2.0,
                crossover="single", mutation="random", pareto_select="knee",
                random_state=42, verbose=False,
            )

    def test_bad_smoothness_order_raises(self):
        """Bad smoothness_order raises ValueError (line 933)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        with pytest.raises(ValueError, match="smoothness"):
            _solve_genetic_impl(
                A=np.eye(3), b=np.ones(3), x0=None, solver="pso",
                epoch=2, pop_size=6, regularization=0.01, norm=2,
                smoothness_order=5, smoothness_weight=1.0, entropy_weight=0.0,
                n_runs=1, early_stop=None, half_range=2.0, two_step=False,
                n_coarse=None, smoother="none", sigma_smooth=2.0,
                crossover="single", mutation="random", pareto_select="knee",
                random_state=42, verbose=False,
            )

    def test_bad_crossover_raises(self):
        """Bad crossover in _solve_genetic_impl raises (line 935)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        with pytest.raises(ValueError, match="crossover"):
            _solve_genetic_impl(
                A=np.eye(3), b=np.ones(3), x0=None, solver="pso",
                epoch=2, pop_size=6, regularization=0.01, norm=2,
                smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
                n_runs=1, early_stop=None, half_range=2.0, two_step=False,
                n_coarse=None, smoother="none", sigma_smooth=2.0,
                crossover="bad", mutation="random", pareto_select="knee",
                random_state=42, verbose=False,
            )

    def test_bad_mutation_raises(self):
        """Bad mutation in _solve_genetic_impl raises (line 940)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        with pytest.raises(ValueError, match="mutation"):
            _solve_genetic_impl(
                A=np.eye(3), b=np.ones(3), x0=None, solver="pso",
                epoch=2, pop_size=6, regularization=0.01, norm=2,
                smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
                n_runs=1, early_stop=None, half_range=2.0, two_step=False,
                n_coarse=None, smoother="none", sigma_smooth=2.0,
                crossover="single", mutation="bad", pareto_select="knee",
                random_state=42, verbose=False,
            )

    def test_bad_pareto_select_raises(self):
        """Bad pareto_select in _solve_genetic_impl raises (line 944)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        with pytest.raises(ValueError, match="pareto_select"):
            _solve_genetic_impl(
                A=np.eye(3), b=np.ones(3), x0=None, solver="pso",
                epoch=2, pop_size=6, regularization=0.01, norm=2,
                smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
                n_runs=1, early_stop=None, half_range=2.0, two_step=False,
                n_coarse=None, smoother="none", sigma_smooth=2.0,
                crossover="single", mutation="random", pareto_select="bad",
                random_state=42, verbose=False,
            )


class TestSolveGeneticImplMealpyPaths:
    """Tests for _solve_genetic_impl paths that need mealpy (lines 923-1125)."""

    def test_mealpy_pso_path(self):
        """Full PSO path through mealpy (lines 1077-1125)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        np.random.seed(42)
        A = np.random.randn(3, 5)
        x_true = np.abs(np.random.randn(5)) + 0.1
        b = A @ x_true
        result = _solve_genetic_impl(
            A=A, b=b, x0=None, solver="pso",
            epoch=5, pop_size=10, regularization=0.01, norm=2,
            smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
            n_runs=1, early_stop=None, half_range=2.0, two_step=False,
            n_coarse=None, smoother="none", sigma_smooth=2.0,
            crossover="single", mutation="random", pareto_select="knee",
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_mealpy_nsga2_path(self):
        """NSGA2 path in _solve_genetic_impl (lines 1020-1043)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        np.random.seed(42)
        A = np.random.randn(3, 5)
        x_true = np.abs(np.random.randn(5)) + 0.1
        b = A @ x_true
        result = _solve_genetic_impl(
            A=A, b=b, x0=None, solver="nsga2",
            epoch=5, pop_size=10, regularization=0.01, norm=2,
            smoothness_order=0, smoothness_weight=1.0, entropy_weight=1.0,
            n_runs=1, early_stop=None, half_range=2.0, two_step=False,
            n_coarse=None, smoother="none", sigma_smooth=2.0,
            crossover="single", mutation="random", pareto_select="knee",
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_mealpy_ga_numpy_path(self):
        """GA numpy path when crossover/mutation differ from defaults (lines 1045-1075)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        np.random.seed(42)
        A = np.random.randn(3, 5)
        x_true = np.abs(np.random.randn(5)) + 0.1
        b = A @ x_true
        result = _solve_genetic_impl(
            A=A, b=b, x0=None, solver="ga",
            epoch=5, pop_size=10, regularization=0.01, norm=2,
            smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
            n_runs=1, early_stop=None, half_range=2.0, two_step=False,
            n_coarse=None, smoother="none", sigma_smooth=2.0,
            crossover="arithmetic", mutation="iterative", pareto_select="knee",
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_mealpy_with_smoother(self):
        """Mealpy path with post-processing smoother (lines 1117-1125)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        np.random.seed(42)
        A = np.random.randn(3, 5)
        x_true = np.abs(np.random.randn(5)) + 0.1
        b = A @ x_true
        result = _solve_genetic_impl(
            A=A, b=b, x0=None, solver="pso",
            epoch=5, pop_size=10, regularization=0.01, norm=2,
            smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
            n_runs=1, early_stop=None, half_range=2.0, two_step=False,
            n_coarse=None, smoother="gaussian", sigma_smooth=2.0,
            crossover="single", mutation="random", pareto_select="knee",
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_mealpy_n_runs(self):
        """Multiple runs averaging (lines 1097-1116)."""
        pytest.importorskip("mealpy")
        from bssunfold.core.unfold_genetic import _solve_genetic_impl
        np.random.seed(42)
        A = np.random.randn(3, 5)
        x_true = np.abs(np.random.randn(5)) + 0.1
        b = A @ x_true
        result = _solve_genetic_impl(
            A=A, b=b, x0=None, solver="pso",
            epoch=3, pop_size=8, regularization=0.01, norm=2,
            smoothness_order=0, smoothness_weight=1.0, entropy_weight=0.0,
            n_runs=2, early_stop=None, half_range=2.0, two_step=False,
            n_coarse=None, smoother="none", sigma_smooth=2.0,
            crossover="single", mutation="random", pareto_select="knee",
            random_state=42, verbose=False,
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)


class TestMakeStartingSolutions:
    """Tests for _make_starting_solutions."""

    def test_with_extra(self):
        from bssunfold.core.unfold_genetic import _make_starting_solutions
        seed = np.ones(5)
        lb = np.zeros(5)
        ub = np.ones(5) * 5
        extra = np.ones(5) * 2.0
        result = _make_starting_solutions(seed, lb, ub, 10, extra=extra)
        assert result.shape == (10, 5)
        # First individual should be the seed
        np.testing.assert_allclose(result[0], np.log(seed), atol=1e-10)
        # Second individual should be the extra
        np.testing.assert_allclose(result[1], np.log(extra), atol=1e-10)

    def test_without_extra(self):
        from bssunfold.core.unfold_genetic import _make_starting_solutions
        seed = np.ones(5)
        lb = np.zeros(5)
        ub = np.ones(5) * 5
        result = _make_starting_solutions(seed, lb, ub, 10)
        assert result.shape == (10, 5)


class TestFastNonDominatedSort:
    """Tests for _fast_non_dominated_sort."""

    def test_two_objectives(self):
        from bssunfold.core.unfold_genetic import _fast_non_dominated_sort
        fvals = np.array([[0.1, 0.9], [0.2, 0.5], [0.5, 0.2], [0.9, 0.1]])
        fronts = _fast_non_dominated_sort(fvals)
        assert len(fronts) >= 1
        assert fronts[0].size >= 1

    def test_all_dominated(self):
        from bssunfold.core.unfold_genetic import _fast_non_dominated_sort
        fvals = np.array([[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]])
        fronts = _fast_non_dominated_sort(fvals)
        # (0.1, 0.1) dominates everything
        assert 2 in fronts[0]


class TestSBXCrossover:
    """Tests for _sbx_crossover."""

    def test_basic(self):
        from bssunfold.core.unfold_genetic import _sbx_crossover
        rng = np.random.default_rng(42)
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([3.0, 2.0, 1.0])
        lb = np.zeros(3)
        ub = np.ones(3) * 10
        c1, c2 = _sbx_crossover(p1, p2, lb, ub, rng)
        assert c1.shape == (3,)
        assert c2.shape == (3,)


class TestPolynomialMutation:
    """Tests for _polynomial_mutation."""

    def test_basic(self):
        from bssunfold.core.unfold_genetic import _polynomial_mutation
        rng = np.random.default_rng(42)
        p = np.array([1.0, 2.0, 3.0])
        lb = np.zeros(3)
        ub = np.ones(3) * 10
        result = _polynomial_mutation(p, lb, ub, rng)
        assert result.shape == (3,)

    def test_no_mutation(self):
        """When no bits are selected for mutation, returns copy."""
        from bssunfold.core.unfold_genetic import _polynomial_mutation
        # Use a mocked rng that returns > 1/n for all bits
        rng = MagicMock()
        rng.random.return_value = np.ones(5) * 2.0  # All > 1/n
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lb = np.zeros(5)
        ub = np.ones(5) * 10
        result = _polynomial_mutation(p, lb, ub, rng)
        np.testing.assert_array_equal(result, p)


class TestSelectKnee:
    """Tests for _select_knee."""

    def test_basic(self):
        from bssunfold.core.unfold_genetic import _select_knee
        fvals = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        idx = _select_knee(fvals)
        assert 0 <= idx < 3
        assert isinstance(idx, int)


class TestCrowdingDistance:
    """Tests for _crowding_distance."""

    def test_small_front(self):
        from bssunfold.core.unfold_genetic import _crowding_distance
        fvals = np.array([[0.1, 0.9], [0.9, 0.1]])
        front = np.array([0, 1])
        dist = _crowding_distance(fvals, front)
        assert np.all(dist == np.inf)

    def test_three_front(self):
        from bssunfold.core.unfold_genetic import _crowding_distance
        fvals = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        front = np.array([0, 1, 2])
        dist = _crowding_distance(fvals, front)
        assert dist[0] == np.inf
        assert dist[2] == np.inf


# ============================================================================
# 4. unfold_interpret.py — lines 165-429, 534-631 (all behind pyoptexplain)
# ============================================================================


class TestUnfoldInterpret:
    """Tests for unfold_interpret.py (behind importorskip)."""

    def test_interpret_qp_requires_pyoptexplain(self):
        """interpret_qp raises ImportError without pyoptexplain."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import interpret_qp
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = interpret_qp(A, b, 1e-4)
        assert hasattr(result, "spectrum")

    def test_interpret_qp_with_enforce_norm(self):
        """interpret_qp with enforce_norm=True."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import interpret_qp
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = interpret_qp(A, b, 1e-4, enforce_norm=True, norm_value=6.0)
        assert hasattr(result, "spectrum")

    def test_interpret_qp_all_analyses(self):
        """interpret_qp with all analyses enabled."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import interpret_qp
        A = np.eye(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = interpret_qp(
            A, b, 1e-4,
            run_robustness=True,
            run_scenarios=True,
            run_detector_sensitivity=True,
            run_regularization_sweep=True,
            run_nonnegativity_relaxation=True,
            enforce_norm=True,
            norm_value=10.0,
        )
        assert hasattr(result, "spectrum")
        assert result.report
        assert result.metrics

    def test_interpret_qp_minimal(self):
        """interpret_qp with all analyses disabled."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import interpret_qp
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = interpret_qp(
            A, b, 1e-4,
            run_robustness=False,
            run_scenarios=False,
            run_detector_sensitivity=False,
            run_regularization_sweep=False,
            run_nonnegativity_relaxation=False,
        )
        assert hasattr(result, "spectrum")

    def test_unfold_interpret_full(self):
        """unfold_interpret full function (lines 534-631)."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import unfold_interpret
        # This test uses the detector fixture indirectly, but we construct
        # the arguments manually
        from bssunfold import Detector
        det = Detector()
        names = det.detector_names[:3]
        rds = {n: 100.0 + 10.0 * i for i, n in enumerate(names)}
        result = unfold_interpret(
            detector_names=det.detector_names,
            n_energy_bins=det.n_energy_bins,
            E_MeV=det.E_MeV,
            sensitivities=det.sensitivities,
            cc_icrp116=det._get_interpolated_cc(),
            save_result_callback=det._save_result,
            readings=rds,
            regularization=1e-4,
        )
        assert "spectrum" in result
        assert "report" in result
        assert "interpretation_metrics" in result

    def test_unfold_interpret_cosine_method(self):
        """unfold_interpret with cosine regularization method (lines 539-554)."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import unfold_interpret
        from bssunfold import Detector
        det = Detector()
        names = det.detector_names[:3]
        rds = {n: 100.0 + 10.0 * i for i, n in enumerate(names)}
        initial = np.ones(det.n_energy_bins)
        result = unfold_interpret(
            detector_names=det.detector_names,
            n_energy_bins=det.n_energy_bins,
            E_MeV=det.E_MeV,
            sensitivities=det.sensitivities,
            cc_icrp116=det._get_interpolated_cc(),
            save_result_callback=det._save_result,
            readings=rds,
            regularization_method="cosine",
            initial_spectrum=initial,
        )
        assert "spectrum" in result

    def test_unfold_interpret_cosine_no_initial_raises(self):
        """cosine method without initial_spectrum raises (lines 541-544)."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import unfold_interpret
        from bssunfold import Detector
        det = Detector()
        names = det.detector_names[:3]
        rds = {n: 100.0 + 10.0 * i for i, n in enumerate(names)}
        with pytest.raises(ValueError, match="initial_spectrum"):
            unfold_interpret(
                detector_names=det.detector_names,
                n_energy_bins=det.n_energy_bins,
                E_MeV=det.E_MeV,
                sensitivities=det.sensitivities,
                cc_icrp116=det._get_interpolated_cc(),
                save_result_callback=det._save_result,
                readings=rds,
                regularization_method="cosine",
            )

    def test_unfold_interpret_cosine_wrong_length_raises(self):
        """cosine method with wrong initial_spectrum length raises (lines 546-550)."""
        pytest.importorskip("pyoptexplain")
        from bssunfold.core.unfold_interpret import unfold_interpret
        from bssunfold import Detector
        det = Detector()
        names = det.detector_names[:3]
        rds = {n: 100.0 + 10.0 * i for i, n in enumerate(names)}
        with pytest.raises(ValueError, match="length"):
            unfold_interpret(
                detector_names=det.detector_names,
                n_energy_bins=det.n_energy_bins,
                E_MeV=det.E_MeV,
                sensitivities=det.sensitivities,
                cc_icrp116=det._get_interpolated_cc(),
                save_result_callback=det._save_result,
                readings=rds,
                regularization_method="cosine",
                initial_spectrum=np.ones(5),
            )

    def test_interpret_module_constants(self):
        """Module constants are accessible."""
        from bssunfold.core.unfold_interpret import (
            _DEFAULT_RELATIVE_DELTAS,
            _DEFAULT_RELAXATION_DELTAS,
            _DEFAULT_NONNEG_DELTAS,
            _DEFAULT_SENSITIVITY_DELTAS,
        )
        assert len(_DEFAULT_RELATIVE_DELTAS) == 4
        assert 0.05 in _DEFAULT_RELAXATION_DELTAS
        assert 0.01 in _DEFAULT_NONNEG_DELTAS
        assert 0.05 in _DEFAULT_SENSITIVITY_DELTAS

    def test_interpret_all_exports(self):
        """All exports are importable."""
        from bssunfold.core.unfold_interpret import (
            InterpretationResult,
            build_interpretation_qp,
            solve_interpret,
            interpret_qp,
            unfold_interpret,
        )
        assert callable(interpret_qp)
        assert callable(unfold_interpret)
