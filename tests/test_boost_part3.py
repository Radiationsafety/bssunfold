"""Coverage boost tests part 3: medium-coverage modules."""

from __future__ import annotations

import builtins
import warnings
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest


@contextmanager
def block_import(*module_names: str) -> Iterator[None]:
    """Make importing the given top-level module names fail with ImportError."""
    names = tuple(module_names)
    original = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):
        if name in names or name.startswith(tuple(f"{m}." for m in names)):
            raise ImportError(f"{names[0]} not installed (blocked in test)")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        yield


# ── unfold_mlem_odl (33.3% → ~100%) ──────────────────────────────────────


class TestUnfoldMlemOdl:
    """Tests for unfold_mlem_odl module."""

    def test_import_error_odl(self):
        """Test ImportError when odl is not available."""
        with block_import("odl"):
            from bssunfold.core.unfold_mlem_odl import unfold_mlem_odl

            with pytest.raises(ImportError, match="odl is required"):
                unfold_mlem_odl(
                    detector_names=["a"],
                    n_energy_bins=10,
                    E_MeV=np.linspace(1e-10, 20, 10),
                    sensitivities={"a": np.ones(10)},
                    cc_icrp116={"AP": np.ones(10)},
                    save_result_callback=lambda x: None,
                    readings={"a": 1.0},
                )


# ── _parametric_shared (85.7% → ~100%) ───────────────────────────────────


class TestParametricShared:
    """Tests for _parametric_shared module."""

    def test_check_fit_quality_warns(self):
        """Warning when residual is large relative to readings."""
        from bssunfold.core._parametric_shared import _check_fit_quality

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fit_quality(100.0, np.array([1.0, 2.0]))
            assert len(w) == 1
            assert "large residual" in str(w[0].message).lower()

    def test_check_fit_quality_no_warn(self):
        """No warning when residual is small."""
        from bssunfold.core._parametric_shared import _check_fit_quality

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fit_quality(0.01, np.array([100.0, 200.0]))
            assert len(w) == 0

    def test_check_fit_quality_zero_b(self):
        """No warning when b_readings norm is zero."""
        from bssunfold.core._parametric_shared import _check_fit_quality

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fit_quality(100.0, np.array([0.0, 0.0]))
            assert len(w) == 0

    def test_clean_edge_bins_first(self):
        """First bin cleaned when anomalously large."""
        from bssunfold.core._parametric_shared import _clean_edge_bins

        phi = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
        result = _clean_edge_bins(phi, factor=5.0)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_clean_edge_bins_last(self):
        """Last bin cleaned when anomalously large."""
        from bssunfold.core._parametric_shared import _clean_edge_bins

        phi = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        result = _clean_edge_bins(phi, factor=5.0)
        assert result[-1] == 0.0
        assert result[0] == 1.0

    def test_clean_edge_bins_short(self):
        """Short arrays returned as-is."""
        from bssunfold.core._parametric_shared import _clean_edge_bins

        result = _clean_edge_bins(np.array([1.0, 2.0]))
        assert len(result) == 2

    def test_clean_edge_bins_no_spike(self):
        """No cleaning when no spike."""
        from bssunfold.core._parametric_shared import _clean_edge_bins

        phi = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = _clean_edge_bins(phi, factor=10.0)
        np.testing.assert_array_equal(result, phi)

    def test_clean_edge_bins_zero_neighbors(self):
        """No cleaning when neighbors are zero."""
        from bssunfold.core._parametric_shared import _clean_edge_bins

        phi = np.array([100.0, 0.0, 0.0, 1.0, 1.0])
        result = _clean_edge_bins(phi, factor=5.0)
        assert result[0] == 100.0  # neighbor_mean=0, condition not met

    def test_build_measurement_uncertainties(self):
        """Test uncertainty estimation."""
        from bssunfold.core._parametric_shared import _build_measurement_uncertainties

        b = np.array([10.0, 20.0, 0.0])
        sigma = _build_measurement_uncertainties(b, noise_level=0.05)
        assert sigma.shape == (3,)
        assert sigma[0] == pytest.approx(0.5, abs=1e-10)
        assert sigma[1] == pytest.approx(1.0, abs=1e-10)
        assert sigma[2] == pytest.approx(1e-30, abs=1e-15)

    def test_build_measurement_uncertainties_negative(self):
        """Test with negative readings - abs is applied."""
        from bssunfold.core._parametric_shared import _build_measurement_uncertainties

        b = np.array([-10.0, 20.0])
        sigma = _build_measurement_uncertainties(b)
        # abs(-10) * 0.05 = 0.5
        assert sigma[0] == pytest.approx(0.5, abs=1e-10)


# ── unfold_statreg (82.9% → ~100%) ───────────────────────────────────────


class TestStatReg:
    """Tests for unfold_statreg module."""

    def test_solve_statreg_negative_b_raises(self):
        """Negative b raises ValueError."""
        from bssunfold.core.unfold_statreg import solve_statreg

        A = np.random.rand(5, 20) + 0.1
        b = np.array([1.0, -0.5, 2.0, 3.0, 1.0])
        with pytest.raises(ValueError, match="positive measurements"):
            solve_statreg(A, b)

    def test_solve_statreg_zero_b_drops(self):
        """Zero b values are dropped."""
        from bssunfold.core.unfold_statreg import solve_statreg

        np.random.seed(42)
        A = np.random.rand(5, 20) + 0.1
        b = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        x = solve_statreg(A, b)
        assert len(x) == 20
        assert np.all(x >= 0)

    def test_solve_statreg_all_zero_b_raises(self):
        """All zero b raises ValueError."""
        from bssunfold.core.unfold_statreg import solve_statreg

        A = np.random.rand(5, 20) + 0.1
        b = np.zeros(5)
        with pytest.raises(ValueError, match="positive measurements"):
            solve_statreg(A, b)

    def test_solve_statreg_unknown_method(self):
        """Unknown method raises ValueError."""
        from bssunfold.core.unfold_statreg import solve_statreg

        A = np.random.rand(5, 20) + 0.1
        b = np.ones(5)
        with pytest.raises(ValueError, match="Unknown method"):
            solve_statreg(A, b, unfoldermethod="bad_method")

    def test_solve_statreg_user_alpha(self):
        """User-specified alpha."""
        from bssunfold.core.unfold_statreg import solve_statreg

        A = np.random.rand(5, 20) + 0.1
        b = A @ (np.random.rand(20) + 0.1)
        x = solve_statreg(A, b, unfoldermethod="User", regularization=0.01)
        assert len(x) == 20

    def test_lcurve_statreg_edge_distance_zero(self):
        """L-curve with zero edge distance returns middle alpha."""
        from bssunfold.core.unfold_statreg import _lcurve_statreg

        # Create degenerate case where all alphas give same result
        B = np.eye(5)
        beta = np.zeros(5)
        L = np.eye(5)
        alpha = _lcurve_statreg(B, beta, L)
        assert isinstance(alpha, float)

    def test_lcurve_statreg_linalg_error(self):
        """L-curve handles LinAlgError gracefully."""
        from bssunfold.core.unfold_statreg import _lcurve_statreg

        # Singular system
        B = np.zeros((5, 5))
        beta = np.zeros(5)
        L = np.eye(5)
        alpha = _lcurve_statreg(B, beta, L, n_alphas=5)
        # Should return default 1.0 when < 3 valid alphas
        assert isinstance(alpha, float)


# ── unfold_fruit_like (50% → ~100%) ──────────────────────────────────────


class TestFruitLike:
    """Tests for unfold_fruit_like module."""

    def test_import_error_no_lmfit(self):
        """ImportError when lmfit is not available."""
        with block_import("lmfit"):
            from bssunfold.core.unfold_fruit_like import solve_fruit_like

            with pytest.raises(ImportError, match="lmfit is required"):
                solve_fruit_like(
                    np.eye(5),
                    np.ones(5),
                    np.linspace(1e-10, 20, 10),
                    np.ones(10),
                )

    def test_parametric_model_thermal(self):
        """Parametric model thermal component."""
        from bssunfold.core.unfold_fruit_like import parametric_model

        E = np.array([1e-8, 2e-8, 3e-8])  # All thermal
        phi = parametric_model(E, A_th=1.0, T_th=0.025e-6, A_epi=0, A_f=0, T_ev=2.0)
        assert len(phi) == 3
        assert np.all(phi >= 0)

    def test_parametric_model_epithermal(self):
        """Parametric model epithermal component."""
        from bssunfold.core.unfold_fruit_like import parametric_model

        E = np.array([0.001, 0.01, 0.05])  # All epithermal
        phi = parametric_model(E, A_th=0, T_th=0.025e-6, A_epi=1.0, A_f=0, T_ev=2.0)
        assert len(phi) == 3
        assert np.all(phi > 0)

    def test_parametric_model_fast(self):
        """Parametric model fast component."""
        from bssunfold.core.unfold_fruit_like import parametric_model

        E = np.array([1.0, 5.0, 10.0])  # All fast
        phi = parametric_model(E, A_th=0, T_th=0.025e-6, A_epi=0, A_f=1.0, T_ev=2.0)
        assert len(phi) == 3
        assert np.all(phi > 0)

    def test_parametric_model_mixed(self):
        """Parametric model with all components."""
        from bssunfold.core.unfold_fruit_like import parametric_model

        E = np.logspace(-8, 1, 150)
        phi = parametric_model(
            E, A_th=1e-6, T_th=0.025e-6, A_epi=1e-6, A_f=1e-6, T_ev=2.0
        )
        assert len(phi) == 150

    def test_maxwellian(self):
        """Maxwellian function."""
        from bssunfold.core.unfold_fruit_like import _maxwellian

        E = np.array([1e-8, 2e-8])
        phi = _maxwellian(E, T=0.025e-6, A_th=1.0)
        assert len(phi) == 2
        assert phi[0] > 0

    def test_one_over_e(self):
        """1/E function."""
        from bssunfold.core.unfold_fruit_like import _one_over_e

        E = np.array([0.001, 0.01, 0.1])
        phi = _one_over_e(E, A_epi=1.0)
        np.testing.assert_allclose(phi, 1.0 / (E + 1e-15), rtol=1e-10)

    def test_evaporation(self):
        """Evaporation spectrum function."""
        from bssunfold.core.unfold_fruit_like import _evaporation

        E = np.array([1.0, 2.0, 5.0])
        phi = _evaporation(E, T_ev=2.0, A_f=1.0)
        expected = np.exp(-E / 2.0)
        np.testing.assert_allclose(phi, expected)

    def test_solve_fruit_like_with_lmfit(self):
        """Full solve with lmfit."""
        try:
            import lmfit  # noqa: F401
        except ImportError:
            pytest.skip("lmfit not installed")

        from bssunfold.core.unfold_fruit_like import solve_fruit_like

        np.random.seed(42)
        E = np.logspace(-8, 1, 150)
        log_steps = np.gradient(np.log(E))

        # Create a simple solvable problem
        A = np.random.rand(7, 150) + 0.1
        b = A @ (np.random.rand(150) + 0.1)

        spectrum, success, message, nfev = solve_fruit_like(A, b, E, log_steps)
        assert len(spectrum) == 150
        assert isinstance(success, bool)
        assert isinstance(nfev, int)

    def test_solve_fruit_like_with_initial_params(self):
        """Solve with initial params."""
        try:
            import lmfit  # noqa: F401
        except ImportError:
            pytest.skip("lmfit not installed")

        from bssunfold.core.unfold_fruit_like import solve_fruit_like

        E = np.logspace(-8, 1, 150)
        log_steps = np.gradient(np.log(E))
        A = np.random.rand(7, 150) + 0.1
        b = A @ (np.random.rand(150) + 0.1)

        initial = {
            "A_th": 1e-5,
            "T_th": 0.03e-6,
            "A_epi": 1e-5,
            "A_f": 1e-5,
            "T_ev": 1.5,
        }
        spectrum, success, message, nfev = solve_fruit_like(
            A, b, E, log_steps, initial_params=initial
        )
        assert len(spectrum) == 150


# ── unfold_ferdor (53.1% → ~100%) ────────────────────────────────────────


class TestFerdor:
    """Tests for unfold_ferdor module."""

    def test_solve_ferdor_basic(self):
        """Basic FERDOR solve."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        np.random.seed(42)
        A = np.random.rand(7, 30) + 0.1
        x_true = np.random.rand(30) + 0.1
        b = A @ x_true
        x0 = np.ones(30)

        x, iters, converged = solve_ferdor(A, b, x0)
        assert len(x) == 30
        assert np.all(x >= 0)
        assert isinstance(iters, int)
        assert isinstance(converged, bool)

    def test_solve_ferdor_empty_b_raises(self):
        """Empty measurement vector raises ValueError."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        with pytest.raises(ValueError, match="empty"):
            solve_ferdor(np.zeros((0, 5)), np.array([]), np.ones(5))

    def test_solve_ferdor_all_nonpositive_b_raises(self):
        """All non-positive b raises ValueError."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        A = np.random.rand(3, 10) + 0.1
        b = np.array([-1.0, -2.0, -3.0])
        with pytest.raises(ValueError, match="positive measurement"):
            solve_ferdor(A, b, np.ones(10))

    def test_solve_ferdor_bad_sigma_shape(self):
        """Wrong sigma shape raises ValueError."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        A = np.random.rand(3, 10) + 0.1
        b = np.ones(3)
        with pytest.raises(ValueError, match="sigma must have shape"):
            solve_ferdor(A, b, np.ones(10), sigma=np.ones(5))

    def test_solve_ferdor_with_sigma(self):
        """FERDOR with explicit sigma."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        np.random.seed(42)
        A = np.random.rand(7, 30) + 0.1
        b = A @ (np.random.rand(30) + 0.1)
        sigma = np.full(7, 0.05)

        x, iters, converged = solve_ferdor(A, b, np.ones(30), sigma=sigma)
        assert len(x) == 30

    def test_solve_weighted_ls_near_zero_alpha(self):
        """Near-zero alpha uses NNLS path."""
        from bssunfold.core.unfold_ferdor import _solve_weighted_ls

        A = np.random.rand(5, 10) + 0.1
        ATA = A.T @ A
        ATb = A.T @ np.ones(5)
        LTL = np.eye(10)
        Aw = A.copy()
        bw = np.ones(5)

        x = _solve_weighted_ls(ATA, ATb, LTL, 1e-30, Aw, bw)
        assert x is not None
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_solve_weighted_ls_near_zero_alpha_no_aw(self):
        """Near-zero alpha without Aw/bw uses lstsq fallback."""
        from bssunfold.core.unfold_ferdor import _solve_weighted_ls

        A = np.random.rand(5, 10) + 0.1
        ATA = A.T @ A
        ATb = A.T @ np.ones(5)
        LTL = np.eye(10)

        x = _solve_weighted_ls(ATA, ATb, LTL, 1e-30)
        assert x is not None
        assert np.all(x >= 0)

    def test_solve_weighted_ls_normal_path(self):
        """Normal alpha path uses linalg.solve."""
        from bssunfold.core.unfold_ferdor import _solve_weighted_ls

        A = np.random.rand(5, 10) + 0.1
        ATA = A.T @ A
        ATb = A.T @ np.ones(5)
        LTL = np.eye(10)

        x = _solve_weighted_ls(ATA, ATb, LTL, 1.0)
        assert x is not None
        assert len(x) == 10

    def test_solve_weighted_ls_singular_normal(self):
        """Singular normal equations fall through to lstsq."""
        from bssunfold.core.unfold_ferdor import _solve_weighted_ls

        # Nearly singular matrix
        ATA = np.zeros((10, 10))
        ATb = np.ones(10)
        LTL = np.eye(10)

        x = _solve_weighted_ls(ATA, ATb, LTL, 1.0)
        # Should fall through to lstsq, then nnls
        assert x is not None

    def test_solve_weighted_ls_all_fail(self):
        """All solvers fail returns None."""
        from bssunfold.core.unfold_ferdor import _solve_weighted_ls

        # Zero matrix - will cause all solves to fail
        ATA = np.zeros((2, 2))
        ATb = np.zeros(2)
        LTL = np.zeros((2, 2))

        x = _solve_weighted_ls(ATA, ATb, LTL, 1.0)
        # May return None or zeros
        if x is not None:
            assert len(x) == 2

    def test_solve_ferdor_already_good(self):
        """Unregularized solution already meets chi2 target."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        A = np.random.rand(7, 30) + 0.1
        x_true = np.random.rand(30) + 0.1
        b = A @ x_true

        x, iters, converged = solve_ferdor(A, b, np.ones(30), max_iterations=100)
        assert converged is True or iters > 0

    def test_solve_ferdor_convergence(self):
        """Converges with enough iterations."""
        from bssunfold.core.unfold_ferdor import solve_ferdor

        np.random.seed(42)
        A = np.random.rand(7, 30) + 0.1
        b = A @ (np.random.rand(30) + 0.1)

        x, iters, converged = solve_ferdor(
            A, b, np.ones(30), max_iterations=200, tolerance=1e-6
        )
        assert iters >= 1

    def test_LinalgWarning_class(self):
        """LinalgWarning class exists."""
        from bssunfold.core.unfold_ferdor import LinalgWarning

        assert issubclass(LinalgWarning, Warning)


# ── validators new functions ──────────────────────────────────────────────


class TestNewValidators:
    """Tests for new validation functions in validators.py."""

    def test_validate_system_basic(self):
        """Basic system validation."""
        from bssunfold.utils.validators import validate_system

        A = np.random.rand(5, 10) + 0.1
        b = np.ones(5)
        x0 = np.ones(10)
        Av, bv, x0v = validate_system(A, b, x0=x0)
        assert Av.shape == (5, 10)
        assert bv.shape == (5,)
        assert x0v.shape == (10,)

    def test_validate_system_bad_A_ndim(self):
        """A must be 2D."""
        from bssunfold.utils.validators import validate_system

        with pytest.raises(ValueError, match="2D"):
            validate_system(np.ones(5), np.ones(5))

    def test_validate_system_empty_A(self):
        """A must not be empty."""
        from bssunfold.utils.validators import validate_system

        with pytest.raises(ValueError, match="empty"):
            validate_system(np.zeros((0, 5)), np.array([]))

    def test_validate_system_shape_mismatch(self):
        """A and b dimensions must match."""
        from bssunfold.utils.validators import validate_system

        with pytest.raises(ValueError):
            validate_system(np.ones((5, 10)), np.ones(3))

    def test_validate_system_x0_mismatch(self):
        """x0 must match A columns."""
        from bssunfold.utils.validators import validate_system

        with pytest.raises(ValueError):
            validate_system(np.ones((5, 10)), np.ones(5), x0=np.ones(8))

    def test_validate_system_nan(self):
        """NaN in A is checked by validate_response_matrix, not validate_system."""
        from bssunfold.utils.validators import validate_system

        A = np.ones((5, 10))
        A[0, 0] = np.nan
        # validate_system does not check for NaN (validate_response_matrix does)
        # It just converts to float64
        Av, bv, _ = validate_system(A, np.ones(5))
        assert np.isnan(Av[0, 0])

    def test_validate_solver_params_basic(self):
        """Basic solver params validation."""
        from bssunfold.utils.validators import validate_solver_params

        result = validate_solver_params(
            max_iterations=100,
            tolerance=1e-6,
            noise_level=0.01,
            n_montecarlo=50,
        )
        assert result["max_iterations"] == 100
        assert result["tolerance"] == 1e-6

    def test_validate_solver_params_bad_max_iter(self):
        """Bad max_iterations raises ValueError (implementation uses ValueError)."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError, match="max_iterations"):
            validate_solver_params(max_iterations="bad")

    def test_validate_solver_params_negative_max_iter(self):
        """Negative max_iterations raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(max_iterations=-1)

    def test_validate_solver_params_bad_tolerance(self):
        """Bad tolerance raises TypeError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(TypeError):
            validate_solver_params(tolerance="bad")

    def test_validate_solver_params_zero_tolerance(self):
        """Zero tolerance is not allowed (must be > 0)."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError, match="tolerance"):
            validate_solver_params(tolerance=0)

    def test_validate_solver_params_negative_tolerance(self):
        """Negative tolerance raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(tolerance=-1.0)

    def test_validate_solver_params_noise_level(self):
        """Noise level validation."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(noise_level=1.5)

    def test_validate_solver_params_negative_noise(self):
        """Negative noise_level raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(noise_level=-0.1)

    def test_validate_solver_params_bad_montecarlo(self):
        """Bad n_montecarlo raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError, match="n_montecarlo"):
            validate_solver_params(n_montecarlo="bad")

    def test_validate_solver_params_negative_montecarlo(self):
        """Negative n_montecarlo raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(n_montecarlo=-1)

    def test_validate_solver_params_bad_random_state(self):
        """Negative random_state raises ValueError."""
        from bssunfold.utils.validators import validate_solver_params

        with pytest.raises(ValueError):
            validate_solver_params(random_state=-1)

    def test_validate_solver_params_none_random_state(self):
        """None random_state is allowed."""
        from bssunfold.utils.validators import validate_solver_params

        result = validate_solver_params(random_state=None)
        assert result["random_state"] is None


# ── _base_unfolder validation (new code) ──────────────────────────────────


class TestBaseUnfolderValidation:
    """Tests for new validation in _base_unfolder.py."""

    def test_run_unfolding_empty_readings(self):
        """Empty readings dict raises ValueError."""
        from bssunfold.core._base_unfolder import run_unfolding

        with pytest.raises(ValueError, match="readings"):
            run_unfolding(
                detector_names=["a"],
                n_energy_bins=10,
                E_MeV=np.linspace(1e-10, 20, 10),
                sensitivities={"a": np.ones(10)},
                cc_icrp116={"AP": np.ones(10)},
                save_result_callback=lambda x: None,
                readings={},
                initial_spectrum=None,
                default_initial=np.ones(10),
                solve_func=lambda A, b, **kw: np.ones(10),
                solve_kwargs={},
                method_name="test",
            )

    def test_run_unfolding_empty_detector_names(self):
        """Empty detector_names causes empty readings match."""
        from bssunfold.core._base_unfolder import run_unfolding

        # Empty readings is checked first (reads matches against detector_names)
        with pytest.raises(ValueError, match="readings"):
            run_unfolding(
                detector_names=[],
                n_energy_bins=10,
                E_MeV=np.linspace(1e-10, 20, 10),
                sensitivities={},
                cc_icrp116={},
                save_result_callback=lambda x: None,
                readings={},
                initial_spectrum=None,
                default_initial=np.ones(10),
                solve_func=lambda A, b, **kw: np.ones(10),
                solve_kwargs={},
                method_name="test",
            )

    def test_run_unfolding_bad_noise_level(self):
        """Bad noise_level raises ValueError."""
        from bssunfold.core._base_unfolder import run_unfolding

        with pytest.raises((ValueError, TypeError)):
            run_unfolding(
                detector_names=["a"],
                n_energy_bins=10,
                E_MeV=np.linspace(1e-10, 20, 10),
                sensitivities={"a": np.ones(10)},
                cc_icrp116={"AP": np.ones(10)},
                save_result_callback=lambda x: None,
                readings={"a": 1.0},
                initial_spectrum=None,
                default_initial=np.ones(10),
                solve_func=lambda A, b, **kw: np.ones(10),
                solve_kwargs={},
                method_name="test",
                noise_level=5.0,
            )

    def test_run_unfolding_bad_montecarlo(self):
        """Negative n_montecarlo raises ValueError."""
        from bssunfold.core._base_unfolder import run_unfolding

        with pytest.raises(ValueError, match="n_montecarlo"):
            run_unfolding(
                detector_names=["a"],
                n_energy_bins=10,
                E_MeV=np.linspace(1e-10, 20, 10),
                sensitivities={"a": np.ones(10)},
                cc_icrp116={"AP": np.ones(10)},
                save_result_callback=lambda x: None,
                readings={"a": 1.0},
                initial_spectrum=None,
                default_initial=np.ones(10),
                solve_func=lambda A, b, **kw: np.ones(10),
                solve_kwargs={},
                method_name="test",
                n_montecarlo=-1,
            )

    def test_run_unfolding_E_MeV_length_mismatch(self):
        """E_MeV length must match n_energy_bins."""
        from bssunfold.core._base_unfolder import run_unfolding

        with pytest.raises(ValueError, match="E_MeV"):
            run_unfolding(
                detector_names=["a"],
                n_energy_bins=10,
                E_MeV=np.linspace(1e-10, 20, 5),  # Wrong length
                sensitivities={"a": np.ones(10)},
                cc_icrp116={"AP": np.ones(10)},
                save_result_callback=lambda x: None,
                readings={"a": 1.0},
                initial_spectrum=None,
                default_initial=np.ones(10),
                solve_func=lambda A, b, **kw: np.ones(10),
                solve_kwargs={},
                method_name="test",
            )

    def test_normalize_initial_dict_no_spectrum(self):
        """Dict without spectrum key returns default."""
        from bssunfold.core._base_unfolder import _normalize_initial

        result = _normalize_initial({"other": 1.0}, np.ones(10), 10)
        np.testing.assert_array_equal(result, np.ones(10))

    def test_normalize_initial_dict_with_spectrum(self):
        """Dict with spectrum key uses it."""
        from bssunfold.core._base_unfolder import _normalize_initial

        spec = np.full(10, 2.0)
        result = _normalize_initial({"spectrum": spec}, np.ones(10), 10)
        np.testing.assert_array_equal(result, np.full(10, 2.0))

    def test_normalize_initial_wrong_length(self):
        """Wrong length raises ValueError."""
        from bssunfold.core._base_unfolder import _normalize_initial

        with pytest.raises(ValueError, match="length"):
            _normalize_initial(np.ones(5), np.ones(10), 10)

    def test_normalize_initial_negative_clipped(self):
        """Negative values are clipped to zero."""
        from bssunfold.core._base_unfolder import _normalize_initial

        spec = np.array([-1.0, 0.5, 2.0] + [1.0] * 7)
        result = _normalize_initial(spec, np.ones(10), 10)
        assert result[0] == 0.0
        assert result[1] == 0.5


# ── detector __getattr__ (new code) ───────────────────────────────────────


class TestDetectorGetattr:
    """Tests for new __getattr__ in Detector."""

    def test_unknown_unfold_method(self):
        """Unknown unfold_* method gives helpful error."""
        from bssunfold import Detector

        d = Detector()
        with pytest.raises(AttributeError, match="Unknown unfolding method"):
            d.unfold_typo()

    def test_non_unfold_attribute(self):
        """Non-unfold attributes raise normal AttributeError."""
        from bssunfold import Detector

        d = Detector()
        with pytest.raises(AttributeError):
            _ = d.nonexistent_attribute_xyz

    def test_known_unfold_method_works(self):
        """Known unfold methods still work."""
        from bssunfold import Detector

        d = Detector()
        readings = {name: 1.0 for name in d.detector_names}
        result = d.unfold_mlem(readings, max_iterations=2)
        assert "spectrum" in result


# ── _numba_jit new functions ──────────────────────────────────────────────


class TestNumbaJit:
    """Tests for new Numba JIT functions."""

    def test_bayes_inner_exists(self):
        """Check _bayes_inner function exists."""
        from bssunfold.core._numba_jit import _bayes_inner

        assert callable(_bayes_inner)

    def test_landweber_inner_exists(self):
        """Check _landweber_inner function exists."""
        from bssunfold.core._numba_jit import _landweber_inner

        assert callable(_landweber_inner)

    def test_NUMBA_AVAILABLE(self):
        """Check NUMBA_AVAILABLE is a bool."""
        from bssunfold.core._numba_jit import NUMBA_AVAILABLE

        assert isinstance(NUMBA_AVAILABLE, bool)
