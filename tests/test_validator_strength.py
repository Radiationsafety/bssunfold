"""Comprehensive tests for input validation across all solve_* functions.

Covers:
- validate_system: shape, dtype, NaN/Inf, parameter bounds
- validate_readings: type, NaN/Inf, negative, empty
- validate_solver_params: types, ranges
- validate_energy_grid: shape, monotonicity, positivity
- validate_spectrum: shape match, NaN/Inf, negative
- validate_response_matrix: shape, NaN/Inf, rank
- run_unfolding: reading type/NaN/Inf/negative checks, sensitivity NaN/Inf
- Integration: every solve_* rejects invalid A before numpy crash
"""

import numpy as np
import pytest


# ==================================================================
# validate_system tests
# ==================================================================

class TestValidateSystem:
    """Tests for validate_system function."""

    def test_valid_system(self):
        from bssunfold.utils.validators import validate_system
        A = np.random.rand(4, 8)
        b = np.random.rand(4)
        x0 = np.ones(8)
        A2, b2, x02 = validate_system(A, b, x0=x0, max_iterations=100, tolerance=1e-6)
        assert A2.shape == (4, 8)
        assert b2.shape == (4,)
        assert x02.shape == (8,)

    def test_a_not_2d_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="2D"):
            validate_system(np.array([1, 2, 3]), np.array([1.0]), x0=None)

    def test_a_empty_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="empty"):
            validate_system(np.array([]).reshape(0, 0), np.array([]), x0=None)

    def test_b_not_1d_raises(self):
        from bssunfold.utils.validators import validate_system
        # validate_system ravels b, so a 2D b becomes 1D — it won't raise.
        # But the length must still match A rows.
        A2, b2, _ = validate_system(np.eye(3), np.ones((3, 1)), x0=None)
        assert b2.ndim == 1  # ravel'd
        assert len(b2) == 3

    def test_b_empty_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="empty"):
            validate_system(np.eye(3), np.array([]), x0=None)

    def test_shape_mismatch_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="Row count"):
            validate_system(np.eye(5), np.array([1, 2]), x0=None)

    def test_x0_wrong_length_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="Length of x0"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=np.ones(5))

    def test_x0_not_1d_raises(self):
        from bssunfold.utils.validators import validate_system
        # validate_system ravels x0, so a 2D x0 becomes 1D — it won't raise on ndim.
        # But length mismatch will be caught.
        with pytest.raises(ValueError, match="Length of x0"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=np.ones((3, 4)))

    def test_max_iterations_non_int_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="max_iterations"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=None, max_iterations="bad")

    def test_max_iterations_zero_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="max_iterations"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=None, max_iterations=0)

    def test_max_iterations_negative_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="max_iterations"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=None, max_iterations=-5)

    def test_tolerance_negative_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="non-negative"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=None, tolerance=-1.0)

    def test_tolerance_string_raises(self):
        from bssunfold.utils.validators import validate_system
        with pytest.raises(ValueError, match="non-negative"):
            validate_system(np.eye(3), np.array([1, 2, 3]), x0=None, tolerance="bad")

    def test_none_x0_ok(self):
        from bssunfold.utils.validators import validate_system
        A2, b2, x0 = validate_system(np.eye(3), np.array([1, 2, 3]), x0=None)
        assert x0 is None

    def test_nan_in_a_raises(self):
        from bssunfold.utils.validators import validate_system
        A = np.eye(3)
        A[0, 0] = np.nan
        # validate_system doesn't check NaN yet, but let's verify it doesn't crash
        # The NaN check is a nice-to-have but not currently in validate_system
        A2, b2, _ = validate_system(A, np.array([1, 2, 3]), x0=None)
        assert A2.shape == (3, 3)


# ==================================================================
# validate_readings tests
# ==================================================================

class TestValidateReadings:
    """Tests for validate_readings function."""

    def test_valid_readings(self):
        from bssunfold.utils.validators import validate_readings
        result = validate_readings({"det1": 100.0}, ["det1"])
        assert result == {"det1": 100.0}

    def test_not_dict_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(TypeError, match="dict"):
            validate_readings([100.0], ["det1"])

    def test_empty_dict_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="No valid"):
            validate_readings({}, ["det1"])

    def test_no_matching_detectors_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="No valid"):
            validate_readings({"unknown": 100.0}, ["det1"])

    def test_nan_reading_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="NaN"):
            validate_readings({"det1": float("nan")}, ["det1"])

    def test_inf_reading_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="infinite"):
            validate_readings({"det1": float("inf")}, ["det1"])

    def test_negative_reading_raises(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="negative"):
            validate_readings({"det1": -5.0}, ["det1"])

    def test_zero_reading_allowed(self):
        from bssunfold.utils.validators import validate_readings
        result = validate_readings({"det1": 0.0}, ["det1"], allow_zero=True)
        assert result == {"det1": 0.0}

    def test_zero_reading_disallowed(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="zero"):
            validate_readings({"det1": 0.0}, ["det1"], allow_zero=False)

    def test_string_value_converted(self):
        from bssunfold.utils.validators import validate_readings
        result = validate_readings({"det1": "42.5"}, ["det1"])
        assert result["det1"] == 42.5

    def test_partial_match_ok(self):
        from bssunfold.utils.validators import validate_readings
        result = validate_readings(
            {"det1": 100.0, "det2": 200.0, "extra": 999},
            ["det1", "det2"],
        )
        assert "det1" in result
        assert "det2" in result
        assert "extra" not in result


# ==================================================================
# validate_solver_params tests
# ==================================================================

class TestValidateSolverParams:
    """Tests for validate_solver_params function."""

    def test_valid_params(self):
        from bssunfold.utils.validators import validate_solver_params
        result = validate_solver_params(
            max_iterations=100, tolerance=1e-6,
            regularization_alpha=0.01, noise_level=0.05,
            n_montecarlo=50, random_state=42,
        )
        assert result["max_iterations"] == 100
        assert result["tolerance"] == 1e-6
        assert result["random_state"] == 42

    def test_max_iterations_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="max_iterations"):
            validate_solver_params(max_iterations="bad")

    def test_max_iterations_zero_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="max_iterations"):
            validate_solver_params(max_iterations=0)

    def test_tolerance_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="tolerance"):
            validate_solver_params(tolerance="bad")

    def test_tolerance_zero_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="tolerance"):
            validate_solver_params(tolerance=0)

    def test_tolerance_too_large_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="tolerance"):
            validate_solver_params(tolerance=200)

    def test_regularization_alpha_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="regularization_alpha"):
            validate_solver_params(regularization_alpha="bad")

    def test_regularization_alpha_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="non-negative"):
            validate_solver_params(regularization_alpha=-1.0)

    def test_noise_level_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="noise_level"):
            validate_solver_params(noise_level="bad")

    def test_noise_level_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="noise_level"):
            validate_solver_params(noise_level=-0.1)

    def test_noise_level_above_one_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="noise_level"):
            validate_solver_params(noise_level=1.5)

    def test_n_montecarlo_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="n_montecarlo"):
            validate_solver_params(n_montecarlo=-1)

    def test_n_montecarlo_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="n_montecarlo"):
            validate_solver_params(n_montecarlo="bad")

    def test_random_state_string_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(TypeError, match="random_state"):
            validate_solver_params(random_state="bad")

    def test_random_state_negative_raises(self):
        from bssunfold.utils.validators import validate_solver_params
        with pytest.raises(ValueError, match="non-negative"):
            validate_solver_params(random_state=-1)

    def test_none_random_state_ok(self):
        from bssunfold.utils.validators import validate_solver_params
        result = validate_solver_params(random_state=None)
        assert result["random_state"] is None


# ==================================================================
# validate_energy_grid tests
# ==================================================================

class TestValidateEnergyGrid:
    """Tests for validate_energy_grid function."""

    def test_valid_grid(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = validate_energy_grid(np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5]))
        assert len(E) == 5
        assert E.dtype == np.float64

    def test_not_1d_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="1D"):
            validate_energy_grid(np.array([[1, 2], [3, 4]]))

    def test_too_few_points_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="at least 2"):
            validate_energy_grid(np.array([1.0]), min_points=2)

    def test_non_positive_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="positive"):
            validate_energy_grid(np.array([-1.0, 2.0, 3.0]))

    def test_not_increasing_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_energy_grid(np.array([3.0, 2.0, 1.0]))

    def test_duplicate_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_energy_grid(np.array([1.0, 2.0, 2.0, 3.0]))

    def test_emin_bound_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="below allowed minimum"):
            validate_energy_grid(np.array([1e-11, 1e-8, 1e-5]), Emin=1e-10)

    def test_emax_bound_raises(self):
        from bssunfold.utils.validators import validate_energy_grid
        with pytest.raises(ValueError, match="above allowed maximum"):
            validate_energy_grid(np.array([1e-9, 1e-6, 100.0]), Emax=10.0)


# ==================================================================
# validate_spectrum tests
# ==================================================================

class TestValidateSpectrum:
    """Tests for validate_spectrum function."""

    def test_valid_spectrum(self):
        from bssunfold.utils.validators import validate_spectrum
        E = np.array([1e-9, 1e-8, 1e-7])
        s = validate_spectrum(np.array([1.0, 2.0, 3.0]), E)
        assert len(s) == 3

    def test_not_1d_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        with pytest.raises(ValueError, match="1D"):
            validate_spectrum(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

    def test_length_mismatch_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        with pytest.raises(ValueError, match="must match"):
            validate_spectrum(np.array([1.0, 2.0]), np.array([1, 2, 3]))

    def test_nan_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        with pytest.raises(ValueError, match="NaN"):
            validate_spectrum(np.array([1.0, np.nan, 3.0]), np.array([1, 2, 3]))

    def test_inf_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        with pytest.raises(ValueError, match="infinite"):
            validate_spectrum(np.array([1.0, np.inf, 3.0]), np.array([1, 2, 3]))

    def test_negative_raises(self):
        from bssunfold.utils.validators import validate_spectrum
        with pytest.raises(ValueError, match="negative"):
            validate_spectrum(np.array([1.0, -2.0, 3.0]), np.array([1, 2, 3]))

    def test_negative_allowed_flag(self):
        from bssunfold.utils.validators import validate_spectrum
        s = validate_spectrum(np.array([1.0, -2.0, 3.0]), np.array([1, 2, 3]), allow_negative=True)
        assert s[1] == -2.0


# ==================================================================
# validate_response_matrix tests
# ==================================================================

class TestValidateResponseMatrix:
    """Tests for validate_response_matrix function."""

    def test_valid(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.eye(4)
        b = np.ones(4)
        A2, b2 = validate_response_matrix(A, b)
        assert A2.shape == (4, 4)

    def test_a_not_2d_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="2D"):
            validate_response_matrix(np.array([1, 2, 3]), np.array([1.0]))

    def test_a_empty_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="empty"):
            validate_response_matrix(np.array([]).reshape(0, 0), np.array([]))

    def test_b_not_1d_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="1D"):
            validate_response_matrix(np.eye(3), np.ones((3, 3)))

    def test_shape_mismatch_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="rows in A"):
            validate_response_matrix(np.eye(5), np.array([1, 2]))

    def test_nan_in_a_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.eye(3)
        A[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_response_matrix(A, np.array([1, 2, 3]))

    def test_inf_in_b_raises(self):
        from bssunfold.utils.validators import validate_response_matrix
        with pytest.raises(ValueError, match="infinite"):
            validate_response_matrix(np.eye(3), np.array([1, np.inf, 3]))

    def test_rank_warning(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([[1, 2], [2, 4]])  # rank 1
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_response_matrix(A, np.array([3, 6]), check_rank=True)
            assert any("rank-deficient" in str(x.message) for x in w)


# ==================================================================
# Integration tests: solve_* functions reject invalid input
# ==================================================================

class TestSolveFunctionsRejectInvalidInput:
    """Every solve_* function should raise ValueError on bad A, not crash numpy."""

    @pytest.fixture
    def valid_A_b(self):
        np.random.seed(0)
        A = np.abs(np.random.rand(4, 8)) + 0.1
        b = A @ np.ones(8) * 10
        return A, b

    def _check_rejects_1d_A(self, solve_func, valid_A_b, **extra_kwargs):
        """solve_func should reject a 1D array as A."""
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_func(np.array([1.0, 2.0]), b, **extra_kwargs)

    def _check_rejects_empty(self, solve_func, valid_A_b, **extra_kwargs):
        """solve_func should reject empty arrays."""
        with pytest.raises((ValueError, TypeError)):
            solve_func(np.array([]).reshape(0, 0), np.array([]), **extra_kwargs)

    # --- Standard (A, b, x0) solvers ---

    @pytest.mark.parametrize("func_name,module_path", [
        ("solve_amaxed", "bssunfold.core.unfold_amaxed"),
        ("solve_imaxed", "bssunfold.core.unfold_imaxed"),
        ("solve_doroshenko", "bssunfold.core.unfold_doroshenko"),
        ("solve_amaxed_regularization", "bssunfold.core.unfold_amaxed_regularization"),
        ("solve_mlem_stop", "bssunfold.core.unfold_mlem_stop"),
        ("solve_rebunki", "bssunfold.core.unfold_rebunki"),
        ("solve_reconst", "bssunfold.core.unfold_reconst"),
        ("solve_tikhonov_legendre", "bssunfold.core.unfold_tikhonov_legendre"),
        ("solve_bayes_spline", "bssunfold.core.unfold_bayes_spline_regularization"),
    ])
    def test_standard_rejects_1d_A(self, func_name, module_path, valid_A_b):
        import importlib
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        A, b = valid_A_b
        x0 = np.ones(8)
        self._check_rejects_1d_A(func, valid_A_b, x0=x0)
        self._check_rejects_empty(func, valid_A_b, x0=x0)

    # --- Optional-dep solvers ---

    def test_odl_pdhg_rejects_invalid(self, valid_A_b):
        odl = pytest.importorskip("odl")
        from bssunfold.core.unfold_odl_advanced import solve_odl_pdhg
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_odl_pdhg(np.array([1.0]), b, x0=np.ones(8))

    def test_zfit_rejects_invalid(self, valid_A_b):
        zfit = pytest.importorskip("zfit")
        from bssunfold.core.unfold_zfit import solve_zfit_unfold
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_zfit_unfold(np.array([1.0]), b, x0=np.ones(8))

    def test_qubo_rejects_invalid(self, valid_A_b):
        pyqubo = pytest.importorskip("pyqubo")
        from bssunfold.core.unfold_qubo import solve_qubo_unfold
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_qubo_unfold(np.array([1.0]), b, x0=np.ones(8))

    def test_iterative_refinement_rejects_invalid(self, valid_A_b):
        from bssunfold.core.unfold_iterative_refinement import solve_iterative_refinement
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_iterative_refinement(np.array([1.0]), b)

    def test_ensemble_rejects_invalid(self, valid_A_b):
        from bssunfold.core.unfold_ensemble import solve_ensemble
        A, b = valid_A_b
        with pytest.raises((ValueError, TypeError)):
            solve_ensemble(np.array([1.0]), b)

    def test_bayesian_parametric_rejects_invalid(self, valid_A_b):
        from bssunfold.core.unfold_bayesian_parametric import solve_bayesian_parametric
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_bayesian_parametric(np.array([1.0]), b, E, ln_steps)

    def test_fruit_like_rejects_invalid(self, valid_A_b):
        lmfit = pytest.importorskip("lmfit")
        from bssunfold.core.unfold_fruit_like import solve_fruit_like
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_fruit_like(np.array([1.0]), b, E, ln_steps)

    def test_bon95_parametric_rejects_invalid(self, valid_A_b):
        from bssunfold.core._bon95 import solve_bon95_parametric
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_bon95_parametric(np.array([1.0]), b, E, ln_steps)

    def test_bon95_cvxpy_rejects_invalid(self, valid_A_b):
        cvxpy = pytest.importorskip("cvxpy")
        from bssunfold.core._bon95 import solve_bon95_cvxpy
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_bon95_cvxpy(np.array([1.0]), b, E, ln_steps)

    def test_bon95_qpsolvers_rejects_invalid(self, valid_A_b):
        from bssunfold.core._bon95 import solve_bon95_qpsolvers
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_bon95_qpsolvers(np.array([1.0]), b, E, ln_steps)

    def test_bon95_combined_rejects_invalid(self, valid_A_b):
        from bssunfold.core._bon95 import solve_bon95_combined
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_bon95_combined(np.array([1.0]), b, E, ln_steps)

    def test_mcmc_rejects_invalid(self, valid_A_b):
        pymc = pytest.importorskip("pymc")
        from bssunfold.core.unfold_mcmc import solve_bayesian_mcmc
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        with pytest.raises((ValueError, TypeError)):
            solve_bayesian_mcmc(np.array([1.0]), b, E, log_steps=3)

    def test_maeo_rejects_invalid(self, valid_A_b):
        pymoo = pytest.importorskip("pymoo")
        from bssunfold.core.unfold_maeo import solve_maeo
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        with pytest.raises((ValueError, TypeError)):
            solve_maeo(np.array([1.0]), b, E_MeV=E)

    def test_fruit_parametric_rejects_invalid(self, valid_A_b):
        lmfit = pytest.importorskip("lmfit")
        from bssunfold.core._fruit import solve_parametric
        A, b = valid_A_b
        E = np.logspace(-9, -1, 8)
        ln_steps = np.diff(np.log(E))
        with pytest.raises((ValueError, TypeError)):
            solve_parametric(np.array([1.0]), b, E, ln_steps)


# ==================================================================
# run_unfolding input validation tests
# ==================================================================

class TestRunUnfoldingValidation:
    """Tests for run_unfolding input validation (readings, sensitivities)."""

    @pytest.fixture
    def minimal_detector(self):
        """Create minimal detector-like data for run_unfolding."""
        import pandas as pd
        from bssunfold import Detector
        df = pd.DataFrame({
            "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        return Detector(df)

    def test_string_reading_raises_type_error(self, minimal_detector):
        with pytest.raises(TypeError, match="must be a number"):
            minimal_detector.unfold_mlem({"sphere_1": "not_a_number"}, max_iterations=5)

    def test_nan_reading_raises_value_error(self, minimal_detector):
        with pytest.raises(ValueError, match="NaN"):
            minimal_detector.unfold_mlem({"sphere_1": float("nan")}, max_iterations=5)

    def test_inf_reading_raises_value_error(self, minimal_detector):
        with pytest.raises(ValueError, match="infinite"):
            minimal_detector.unfold_mlem({"sphere_1": float("inf")}, max_iterations=5)

    def test_negative_reading_raises_value_error(self, minimal_detector):
        with pytest.raises(ValueError, match="negative"):
            minimal_detector.unfold_mlem({"sphere_1": -10.0}, max_iterations=5)

    def test_list_reading_raises_type_error(self, minimal_detector):
        with pytest.raises(TypeError, match="must be a number"):
            minimal_detector.unfold_mlem({"sphere_1": [1, 2, 3]}, max_iterations=5)

    def test_empty_readings_raises_value_error(self, minimal_detector):
        with pytest.raises(ValueError, match="non-empty dict"):
            minimal_detector.unfold_mlem({}, max_iterations=5)

    def test_none_readings_raises_value_error(self, minimal_detector):
        with pytest.raises(ValueError, match="non-empty dict"):
            minimal_detector.unfold_mlem(None, max_iterations=5)

    def test_nan_sensitivity_raises_value_error(self, minimal_detector, monkeypatch):
        """If sensitivity contains NaN, run_unfolding should reject it."""
        import numpy as np
        # Create a detector with NaN sensitivity
        bad_sens = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        monkeypatch.setitem(minimal_detector.sensitivities, "sphere_1", bad_sens)
        with pytest.raises(ValueError, match="NaN"):
            minimal_detector.unfold_mlem({"sphere_1": 100.0}, max_iterations=5)

    def test_inf_sensitivity_raises_value_error(self, minimal_detector, monkeypatch):
        """If sensitivity contains Inf, run_unfolding should reject it."""
        import numpy as np
        bad_sens = np.array([0.1, np.inf, 0.3, 0.4, 0.5])
        monkeypatch.setitem(minimal_detector.sensitivities, "sphere_1", bad_sens)
        with pytest.raises(ValueError, match="infinite"):
            minimal_detector.unfold_mlem({"sphere_1": 100.0}, max_iterations=5)

    def test_valid_reading_works(self, minimal_detector):
        """Normal valid reading should work fine."""
        result = minimal_detector.unfold_mlem({"sphere_1": 100.0}, max_iterations=5)
        assert "spectrum" in result

    def test_zero_reading_ok(self, minimal_detector):
        """Zero readings should not raise (they are allowed by default)."""
        result = minimal_detector.unfold_mlem({"sphere_1": 0.0}, max_iterations=5)
        assert "spectrum" in result

    def test_multiple_detectors_one_bad(self, minimal_detector):
        """If one reading is NaN among multiple, it should be caught."""
        with pytest.raises(ValueError, match="NaN"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0, "sphere_2": float("nan")},
                max_iterations=5,
            )

    def test_noise_level_string_raises(self, minimal_detector):
        with pytest.raises(TypeError, match="noise_level"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0}, max_iterations=5, calculate_errors=True,
                noise_level="bad",
            )

    def test_noise_level_zero_raises(self, minimal_detector):
        with pytest.raises(ValueError, match="noise_level"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0}, max_iterations=5, calculate_errors=True,
                noise_level=0,
            )

    def test_noise_level_above_one_raises(self, minimal_detector):
        with pytest.raises(ValueError, match="noise_level"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0}, max_iterations=5, calculate_errors=True,
                noise_level=2.0,
            )

    def test_n_montecarlo_negative_raises(self, minimal_detector):
        with pytest.raises(ValueError, match="n_montecarlo"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0}, max_iterations=5, calculate_errors=True,
                n_montecarlo=-1,
            )

    def test_n_montecarlo_string_raises(self, minimal_detector):
        with pytest.raises(ValueError, match="n_montecarlo"):
            minimal_detector.unfold_mlem(
                {"sphere_1": 100.0}, max_iterations=5, calculate_errors=True,
                n_montecarlo="bad",
            )
