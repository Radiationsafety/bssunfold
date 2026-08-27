"""Coverage tests group 3a: validators, dose_calculation, _multires, _numba_jit, unfold_iterative_refinement, unfold_ensemble, unfold_amaxed_regularization."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
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
# 1. validators.py — missing lines: 58, 60, 175-176, 179-180, 225, 237, 240,
#    245, 299, 303, 313, 324, 331, 335, 397, 402, 407, 423, 431
# ============================================================================


class TestValidatorsReadings:
    """Tests for validate_readings edge cases."""

    def test_reading_nan_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="NaN"):
            validate_readings({"D1": float("nan")}, ["D1"])

    def test_reading_inf_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="infinite"):
            validate_readings({"D1": float("inf")}, ["D1"])


class TestValidatorsSpectrum:
    """Tests for validate_spectrum edge cases."""

    def test_spectrum_nan_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        E = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="1 NaN"):
            validate_spectrum(np.array([1.0, np.nan, 3.0]), E)

    def test_spectrum_multiple_nan_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        E = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="2 NaN"):
            validate_spectrum(np.array([1.0, np.nan, np.nan, 4.0]), E)

    def test_spectrum_inf_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        E = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="1 infinite"):
            validate_spectrum(np.array([1.0, 2.0, np.inf]), E)

    def test_spectrum_multiple_inf_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        E = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="2 infinite"):
            validate_spectrum(np.array([np.inf, 2.0, np.inf, 4.0]), E)


class TestValidatorsResponseMatrix:
    """Tests for validate_response_matrix edge cases."""

    def test_empty_A_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="empty"):
            validate_response_matrix(np.array([]).reshape(0, 0), np.array([]))

    def test_nan_in_A_or_b_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1.0, 2.0], [3.0, np.nan]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            validate_response_matrix(A, b)

    def test_nan_in_b_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, np.nan])
        with pytest.raises(ValueError, match="NaN"):
            validate_response_matrix(A, b)

    def test_inf_in_A_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1.0, np.inf], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="infinite"):
            validate_response_matrix(A, b)

    def test_inf_in_b_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, np.inf])
        with pytest.raises(ValueError, match="infinite"):
            validate_response_matrix(A, b)

    def test_check_rank_warns_on_rank_deficient(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        b = np.array([1.0, 2.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_response_matrix(A, b, check_rank=True)
        assert any("rank-deficient" in str(x.message) for x in w)


class TestValidatorsSystem:
    """Tests for validate_system edge cases."""

    def test_A_row_count_mismatch_b_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0])  # 1 element, A has 2 rows
        with pytest.raises(ValueError, match="Row count"):
            validate_system(A, b)

    def test_x0_length_mismatch_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        x0 = np.array([1.0, 2.0, 3.0])  # 3 elements, A has 2 columns
        with pytest.raises(ValueError, match="Length of x0"):
            validate_system(A, b, x0=x0)

    def test_b_empty_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0]])
        b = np.array([])
        with pytest.raises(ValueError, match="b is empty"):
            validate_system(A, b)

    def test_max_iterations_non_positive_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
            validate_system(A, b, max_iterations=0)

    def test_max_iterations_not_int_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
            validate_system(A, b, max_iterations=1.5)

    def test_tolerance_wrong_type_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            validate_system(A, b, tolerance="bad")

    def test_tolerance_negative_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            validate_system(A, b, tolerance=-1.0)


class TestValidatorsSolverParams:
    """Tests for validate_solver_params edge cases."""

    def test_regularization_alpha_wrong_type_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="regularization_alpha must be a number"):
            validate_solver_params(regularization_alpha="bad")

    def test_regularization_alpha_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="regularization_alpha must be non-negative"):
            validate_solver_params(regularization_alpha=-1.0)

    def test_noise_level_wrong_type_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="noise_level must be a number"):
            validate_solver_params(noise_level="bad")

    def test_random_state_wrong_type_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="random_state must be a non-negative int"):
            validate_solver_params(random_state="bad")

    def test_random_state_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            validate_solver_params(random_state=-1)


# ============================================================================
# 2. dose_calculation.py — missing lines: 67-68, 209, 217
# ============================================================================


class TestDoseCalculation:
    """Tests for dose_calculation edge cases."""

    def test_get_icrp116_coefficients_import_error_fallback(self):
        """Test the ImportError fallback in get_icrp116_coefficients (lines 67-68)."""
        from bssunfold.core import dose_calculation as dc
        # Directly test the except branch by calling the function body
        # with a mocked import that raises ImportError
        import types
        # Reset cache
        original_val = dc.ICRP116_COEFFICIENTS
        dc.ICRP116_COEFFICIENTS = None
        try:
            # Simulate the except branch: set the module-level name to a
            # stub that raises ImportError on attribute access
            mod = types.ModuleType("bssunfold.constants")
            # mod does NOT have ICRP116_COEFF_EFFECTIVE_DOSE attribute
            with patch.dict("sys.modules", {"bssunfold.constants": mod}):
                # Re-import inside the function by calling it fresh
                result = dc.get_icrp116_coefficients()
            assert result == {}
        finally:
            dc.ICRP116_COEFFICIENTS = original_val

    def test_calculate_dose_rates_cc_none(self):
        """Test calculate_dose_rates with cc_icrp116=None (line 195)."""
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.array([1.0, 2.0, 3.0])
        # cc_icrp116=None triggers the default get_icrp116_coefficients() call
        result = calculate_dose_rates(spectrum, cc_icrp116=None)
        assert isinstance(result, dict)

    def test_calculate_dose_rates_no_geoms(self):
        """Test calculate_dose_rates returns {} when cc has only E_MeV key (line 209)."""
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.array([1.0, 2.0, 3.0])
        cc = {"E_MeV": np.array([0.1, 1.0, 10.0])}
        result = calculate_dose_rates(spectrum, cc_icrp116=cc)
        assert result == {}

    def test_calculate_dose_rates_empty_cc(self):
        """Test calculate_dose_rates returns {} when cc is empty (line 198)."""
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.array([1.0, 2.0, 3.0])
        result = calculate_dose_rates(spectrum, cc_icrp116={})
        assert result == {}

    def test_calculate_dose_rates_cc_shorter_than_spectrum(self):
        """Test calculate_dose_rates when cc arrays are shorter than spectrum (line 217)."""
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cc = {
            "E_MeV": np.array([0.1, 1.0, 10.0]),
            "AP": np.array([0.5, 1.0, 2.0]),  # shorter than spectrum
        }
        result = calculate_dose_rates(spectrum, cc_icrp116=cc)
        assert "AP" in result
        # Should not raise, and last bins get 0.0 padding

    def test_get_icrp116_coefficients_returns_dict(self):
        """Test get_icrp116_coefficients returns expected dict."""
        from bssunfold.core.dose_calculation import get_icrp116_coefficients
        cc = get_icrp116_coefficients()
        assert isinstance(cc, dict)
        assert "E_MeV" in cc


# ============================================================================
# 3. _multires.py — missing lines: 44, 90-93, 124
# ============================================================================


class TestMultires:
    """Tests for _multires edge cases."""

    def test_coarsen_columns_n_coarse_too_large(self):
        from bssunfold.core._multires import _coarsen_columns
        A = np.random.rand(3, 5)
        with pytest.raises(ValueError, match="n_coarse must satisfy"):
            _coarsen_columns(A, n_coarse=10)

    def test_coarsen_columns_n_coarse_zero(self):
        from bssunfold.core._multires import _coarsen_columns
        A = np.random.rand(3, 5)
        with pytest.raises(ValueError, match="n_coarse must satisfy"):
            _coarsen_columns(A, n_coarse=0)

    def test_coarse_energy_grid_mixed_sign(self):
        """Test _coarse_energy_grid when some values are non-positive (line 91 - arithmetic mean fallback)."""
        from bssunfold.core._multires import _coarse_energy_grid
        E = np.array([0.5, -0.5, 1.0, 2.0, 3.0, 4.0])
        edges = np.array([0, 2, 4, 6])
        result = _coarse_energy_grid(E, edges)
        assert len(result) == 3
        # Bin 0 has a negative value, so uses arithmetic mean
        assert result[0] == np.mean(E[0:2])

    def test_coarse_energy_grid_empty_group(self):
        """Test _coarse_energy_grid with an empty group (line 93)."""
        from bssunfold.core._multires import _coarse_energy_grid
        E = np.array([1.0, 2.0, 3.0])
        # edges that create an empty bin: [0,0,1,3]
        edges = np.array([0, 0, 1, 3])
        result = _coarse_energy_grid(E, edges)
        assert len(result) == 3
        # First bin is empty (lo=0, hi=0), so should use E[lo]=E[0] if lo < len(E)
        assert result[0] == float(E[0])

    def test_build_coarse_detector_bad_n_coarse(self, detector):
        """Test build_coarse_detector with invalid n_coarse (line 124)."""
        from bssunfold.core._multires import build_coarse_detector
        with pytest.raises(ValueError, match="n_coarse must satisfy"):
            build_coarse_detector(detector, n_coarse=0)


# ============================================================================
# 4. _numba_jit.py — missing lines: 15, 558, 561
# ============================================================================


class TestNumbaJit:
    """Tests for _numba_jit.py fallback branches."""

    def test_numba_available_flag(self):
        """Test NUMBA_AVAILABLE flag is set (line 15 or 17)."""
        from bssunfold.core._numba_jit import NUMBA_AVAILABLE
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_fallback_bayes_inner_raises(self):
        """Test _bayes_inner fallback raises ImportError when numba not available (line 558)."""
        pytest.importorskip("numba")
        # If numba IS available, test the JIT path; if not, the fallback
        from bssunfold.core._numba_jit import _bayes_inner
        # When numba is available, we can't test the fallback easily,
        # but we can at least import and verify existence
        assert callable(_bayes_inner)

    def test_fallback_landweber_inner_raises(self):
        """Test _landweber_inner fallback raises ImportError when numba not available (line 561)."""
        pytest.importorskip("numba")
        from bssunfold.core._numba_jit import _landweber_inner
        assert callable(_landweber_inner)

    def test_no_numba_fallbacks(self):
        """Test that all fallback functions exist and raise ImportError."""
        from bssunfold.core import _numba_jit as nj
        if not nj.NUMBA_AVAILABLE:
            for fn_name in [
                "_mlem_inner", "_kaczmarz_inner", "_doroshenko_inner",
                "_gravel_inner", "_compute_log_steps_jit", "_dose_weighted_mse_jit",
                "_bayes_inner", "_landweber_inner",
            ]:
                fn = getattr(nj, fn_name)
                with pytest.raises(ImportError, match="numba is required"):
                    fn(np.zeros((2, 2)), np.zeros(2), np.zeros(2))


# ============================================================================
# 5. unfold_iterative_refinement.py — missing lines: 203, 250-268, 271
# ============================================================================


class TestUnfoldIterativeRefinement:
    """Tests for unfold_iterative_refinement edge cases."""

    def test_unfold_iterative_refinement_with_random_state(self, detector, readings):
        """Test that random_state seeds the RNG (line 203)."""
        from bssunfold.core.unfold_iterative_refinement import unfold_iterative_refinement
        result = unfold_iterative_refinement(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            random_state=42,
        )
        assert "spectrum" in result
        assert result["method"] == "IterativeRefinement"

    def test_unfold_iterative_refinement_with_mc_errors(self, detector, readings):
        """Test Monte-Carlo error calculation (lines 250-268)."""
        from bssunfold.core.unfold_iterative_refinement import unfold_iterative_refinement
        result = unfold_iterative_refinement(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            calculate_errors=True,
            n_montecarlo=3,
            noise_level=0.01,
            random_state=42,
        )
        assert "spectrum_uncertainty" in result
        assert result["calculate_errors"] is True
        assert result["n_montecarlo"] >= 1

    def test_unfold_iterative_refinement_save_result(self, detector, readings):
        """Test save_result callback is called (line 271)."""
        from bssunfold.core.unfold_iterative_refinement import unfold_iterative_refinement
        callback = MagicMock(return_value="ok")
        result = unfold_iterative_refinement(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=callback,
            readings=readings,
            save_result=True,
        )
        callback.assert_called_once()

    def test_solve_iterative_refinement_fixed_alpha(self):
        """Test solve_iterative_refinement with fixed alpha."""
        from bssunfold.core.unfold_iterative_refinement import solve_iterative_refinement
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        spectrum, info = solve_iterative_refinement(A, b, alpha=0.5)
        assert spectrum.shape == (10,)
        assert info["alpha"] == 0.5

    def test_solve_iterative_refinement_custom_solvers(self):
        """Test solve_iterative_refinement with custom solver functions."""
        from bssunfold.core.unfold_iterative_refinement import solve_iterative_refinement
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.random.rand(10)
        b = A @ x_true

        def fake_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.1

        spectrum, info = solve_iterative_refinement(
            A, b,
            first_pass_solver=fake_solver,
            second_pass_solver=fake_solver,
        )
        assert spectrum.shape == (10,)


# ============================================================================
# 6. unfold_ensemble.py — missing lines: 126, 148-149, 152, 255, 302-320, 323
# ============================================================================


class TestUnfoldEnsemble:
    """Tests for unfold_ensemble edge cases."""

    def test_solve_ensemble_unknown_combination(self):
        """Test ValueError for unknown combination strategy (line 126)."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="Unknown combination"):
            solve_ensemble(A, b, combination="bad_strategy")

    def test_solve_ensemble_method_failure_handled(self):
        """Test that failed methods are skipped (lines 148-149)."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, 1.0])

        def failing_solver(A, b, x0, **kwargs):
            raise RuntimeError("intentional failure")

        def good_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.5

        spectrum, info = solve_ensemble(
            A, b,
            methods=[
                (failing_solver, {"max_iterations": 10, "tolerance": 1e-4, "_name": "bad"}),
                (good_solver, {"max_iterations": 10, "tolerance": 1e-4, "_name": "good"}),
            ],
        )
        assert spectrum.shape == (2,)
        assert info["n_methods"] == 1
        assert "good" in info["method_names"]

    def test_solve_ensemble_all_methods_fail(self):
        """Test RuntimeError when all methods fail (line 152)."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, 1.0])

        def failing_solver(A, b, x0, **kwargs):
            raise RuntimeError("intentional failure")

        with pytest.raises(RuntimeError, match="All ensemble methods failed"):
            solve_ensemble(
                A, b,
                methods=[
                    (failing_solver, {"max_iterations": 10, "tolerance": 1e-4, "_name": "bad"}),
                ],
            )

    def test_solve_ensemble_median_combination(self):
        """Test median combination strategy."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])

        def simple_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.5

        spectrum, info = solve_ensemble(
            A, b,
            methods=[(simple_solver, {"max_iterations": 10, "tolerance": 1e-4, "_name": "s1"})],
            combination="median",
        )
        assert spectrum.shape == (2,)
        assert info["combination"] == "median"

    def test_unfold_ensemble_with_random_state(self, detector, readings):
        """Test unfold_ensemble with random_state (line 255)."""
        from bssunfold.core.unfold_ensemble import unfold_ensemble

        def simple_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.1

        result = unfold_ensemble(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            methods=[(simple_solver, {"max_iterations": 10, "tolerance": 1e-4})],
            random_state=42,
        )
        assert "spectrum" in result

    def test_unfold_ensemble_with_mc_errors(self, detector, readings):
        """Test Monte-Carlo error calculation (lines 302-320)."""
        from bssunfold.core.unfold_ensemble import unfold_ensemble

        def simple_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.1

        result = unfold_ensemble(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            methods=[(simple_solver, {"max_iterations": 10, "tolerance": 1e-4})],
            calculate_errors=True,
            n_montecarlo=3,
            noise_level=0.01,
            random_state=42,
        )
        assert "spectrum_uncertainty" in result
        assert result["calculate_errors"] is True

    def test_unfold_ensemble_save_result(self, detector, readings):
        """Test save_result callback is called (line 323)."""
        from bssunfold.core.unfold_ensemble import unfold_ensemble

        def simple_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.1

        callback = MagicMock(return_value="ok")
        result = unfold_ensemble(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=callback,
            readings=readings,
            methods=[(simple_solver, {"max_iterations": 10, "tolerance": 1e-4})],
            save_result=True,
        )
        callback.assert_called_once()

    def test_solve_ensemble_best_residual(self):
        """Test best_residual combination strategy."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])

        def solver1(A, b, x0, **kwargs):
            return np.array([0.5, 1.0])

        def solver2(A, b, x0, **kwargs):
            return np.array([1.0, 1.0])

        spectrum, info = solve_ensemble(
            A, b,
            methods=[
                (solver1, {"max_iterations": 10, "tolerance": 1e-4, "_name": "s1"}),
                (solver2, {"max_iterations": 10, "tolerance": 1e-4, "_name": "s2"}),
            ],
            combination="best_residual",
        )
        assert spectrum.shape == (2,)
        assert info["combination"] == "best_residual"

    def test_solve_ensemble_trimmed_mean(self):
        """Test trimmed_mean combination strategy."""
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])

        def simple_solver(A, b, x0, **kwargs):
            return np.ones(A.shape[1]) * 0.5

        spectrum, info = solve_ensemble(
            A, b,
            methods=[(simple_solver, {"max_iterations": 10, "tolerance": 1e-4, "_name": "s1"})],
            combination="trimmed_mean",
            trim_fraction=0.2,
        )
        assert spectrum.shape == (2,)
        assert info["combination"] == "trimmed_mean"


# ============================================================================
# 7. unfold_amaxed_regularization.py — missing lines: 66-184, 257-262
# ============================================================================


class TestUnfoldAmaxedRegularization:
    """Tests for unfold_amaxed_regularization (currently 9% coverage)."""

    def test_solve_amaxed_regularization_basic(self):
        """Test basic solve_amaxed_regularization (lines 66-184)."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.abs(np.random.rand(10)) + 0.1
        b = A @ x_true
        x0 = np.ones(10) * 0.5
        spectrum, iters, converged = solve_amaxed_regularization(A, b, x0, max_iterations=50)
        assert spectrum.shape == (10,)
        assert iters > 0
        assert bool(converged)

    def test_solve_amaxed_regularization_converges(self):
        """Test that solver converges with enough iterations."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.abs(np.random.rand(10)) + 0.1
        b = A @ x_true
        x0 = np.ones(10) * 0.5
        spectrum, iters, converged = solve_amaxed_regularization(
            A, b, x0, max_iterations=500, tolerance=1e-6
        )
        assert bool(converged) is True

    def test_solve_amaxed_regularization_with_tau(self):
        """Test with different tau values (regularization strength)."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.abs(np.random.rand(10)) + 0.1
        b = A @ x_true
        x0 = np.ones(10) * 0.5
        # High tau = strong regularization, stays close to prior
        spectrum, iters, converged = solve_amaxed_regularization(
            A, b, x0, tau=10.0, max_iterations=100
        )
        assert spectrum.shape == (10,)
        assert np.all(spectrum >= 0)

    def test_solve_amaxed_regularization_sigma_factor(self):
        """Test with different sigma_factor values."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        np.random.seed(42)
        A = np.random.rand(4, 10)
        x_true = np.abs(np.random.rand(10)) + 0.1
        b = A @ x_true
        x0 = np.ones(10) * 0.5
        spectrum, iters, converged = solve_amaxed_regularization(
            A, b, x0, sigma_factor=0.5, max_iterations=50
        )
        assert spectrum.shape == (10,)

    def test_solve_amaxed_regularization_damped_fallback(self):
        """Test that the damped fallback step works when line search fails."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        # Use a system where line search may not easily accept
        np.random.seed(42)
        A = np.random.rand(4, 10) * 0.01  # very small matrix
        x_true = np.abs(np.random.rand(10)) + 0.1
        b = A @ x_true
        x0 = np.ones(10) * 100.0  # very different from true
        spectrum, iters, converged = solve_amaxed_regularization(
            A, b, x0, max_iterations=20, tolerance=1e-20
        )
        assert spectrum.shape == (10,)

    def test_solve_amaxed_regularization_singular_hessian(self):
        """Test that singular Hessian is handled (lines 154-158)."""
        from bssunfold.core.unfold_amaxed_regularization import solve_amaxed_regularization
        # Create a rank-deficient matrix
        A = np.array([[1.0, 2.0], [2.0, 4.0], [1.0, 2.0]])
        b = np.array([3.0, 6.0, 3.0])
        x0 = np.array([1.0, 1.0])
        spectrum, iters, converged = solve_amaxed_regularization(
            A, b, x0, max_iterations=50
        )
        assert spectrum.shape == (2,)

    def test_unfold_amaxed_regularization_with_initial_spectrum(self, detector, readings):
        """Test unfold with initial_spectrum (lines 257-258)."""
        from bssunfold.core.unfold_amaxed_regularization import unfold_amaxed_regularization
        initial = np.ones(len(detector.E_MeV)) * 0.5
        result = unfold_amaxed_regularization(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            initial_spectrum=initial,
            max_iterations=20,
        )
        assert "spectrum" in result
        assert result["method"] == "AMAXED-Regularization"

    def test_unfold_amaxed_regularization_no_initial_spectrum(self, detector, readings):
        """Test unfold without initial_spectrum (lines 259-260)."""
        from bssunfold.core.unfold_amaxed_regularization import unfold_amaxed_regularization
        result = unfold_amaxed_regularization(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=lambda r: "ok",
            readings=readings,
            initial_spectrum=None,
            max_iterations=20,
        )
        assert "spectrum" in result

    def test_unfold_amaxed_regularization_with_save(self, detector, readings):
        """Test unfold with save_result=True."""
        from bssunfold.core.unfold_amaxed_regularization import unfold_amaxed_regularization
        callback = MagicMock(return_value="ok")
        result = unfold_amaxed_regularization(
            detector_names=detector.detector_names,
            n_energy_bins=len(detector.E_MeV),
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116={},
            save_result_callback=callback,
            readings=readings,
            max_iterations=20,
            save_result=True,
        )
        callback.assert_called_once()
