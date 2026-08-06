"""Tests for the EPIC Tikhonov regularization unfolding method.

Covers the ``solve_epic`` core solver, the internal helpers
(``_calc_epic_ch``, ``_compute_bounds``, ``_default_target_sigmas``,
``_build_regularization_matrix``) and the ``unfold_epic`` wrapper exposed both
on the ``Detector`` class and as a module-level function, including its
registration in ``unfold_combined`` pipelines.
"""

import numpy as np
import pytest

from bssunfold import Detector


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[i]: float(100 - 15 * i) for i in range(5)}


def _synthetic_problem(n=8):
    """Well-posed overdetermined system with a smooth non-negative solution."""
    rng = np.random.default_rng(7)
    A = rng.uniform(0.05, 1.0, size=(n + 2, n))
    x_true = np.abs(rng.uniform(0.5, 2.0, size=n))
    x_true = np.sort(x_true)[::-1]
    b = A @ x_true
    return A, b, x_true


class TestSolveEpic:
    def test_recovers_smooth_solution(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, x_true = _synthetic_problem()
        x = solve_epic(A, b)
        assert x.shape == (A.shape[1],)
        assert np.all(np.isfinite(x))
        assert np.all(x >= -1e-6)
        rel_residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert rel_residual < 0.2

    def test_basic_shapes_and_nonneg(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, regularization_order=1)
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_regularization_order_0(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x0 = solve_epic(A, b, regularization_order=0)
        x1 = solve_epic(A, b, regularization_order=1)
        assert x0.shape == x1.shape == (8,)
        assert np.all(np.isfinite(x0))

    def test_regularization_order_2(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, regularization_order=2)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_invalid_regularization_order(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="Unsupported regularization_order"):
            solve_epic(A, b, regularization_order=3)

    def test_b_length_mismatch(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, _, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="must match the number of A rows"):
            solve_epic(A, np.ones(3))

    def test_non_neg_false_allows_negative(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, non_neg=False)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_explicit_target_sigmas(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, target_sigmas=np.full(A.shape[1], 0.5))
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_invalid_target_sigmas(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="finite and strictly positive"):
            solve_epic(A, b, target_sigmas=np.zeros(A.shape[1]))

    def test_wrong_target_sigmas_length(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="must match"):
            solve_epic(A, b, target_sigmas=np.full(3, 0.5))

    def test_noise_var(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, noise_var=1.0)
        assert x.shape == (8,)

    def test_noise_var_zero_raises(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="noise_var must be positive"):
            solve_epic(A, b, noise_var=0.0)

    def test_epic_bool_mask(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        mask = np.zeros(A.shape[1], dtype=bool)
        mask[:4] = True
        x = solve_epic(A, b, EPIC_bool=mask)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_change_of_variables(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        n = A.shape[1]
        # V maps the searched betas to the Nh regularization rows.
        V = np.ones((n - 1, 1))
        x = solve_epic(A, b, V=V)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_change_of_variables_multicolumn(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        n = A.shape[1]
        rng = np.random.default_rng(3)
        V = rng.uniform(0.5, 1.5, size=(n - 1, 2))
        x = solve_epic(A, b, V=V)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_homogeneous_step_off(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, homogeneous_step=False)
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_regularize_damping(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x = solve_epic(A, b, regularize={})
        assert x.shape == (8,)
        assert np.all(np.isfinite(x))

    def test_lsqpar_tr_solver_exact_and_lsmr(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        x_exact = solve_epic(A, b, LSQpar={"tr_solver": "exact"})
        x_lsmr = solve_epic(A, b, LSQpar={"tr_solver": "lsmr"})
        assert x_exact.shape == x_lsmr.shape == (8,)
        assert np.allclose(x_exact, x_lsmr, atol=1e-3)


class TestEpicHelpers:
    def test_calc_epic_ch_converges(self):
        from bssunfold.core.unfold_epic import (
            _build_precision,
            _build_regularization_matrix,
            _calc_epic_ch,
            _default_target_sigmas,
        )

        A, b, _ = _synthetic_problem()
        P, _ = _build_precision(A, None)
        H = _build_regularization_matrix(A.shape[1], 1)
        ts = _default_target_sigmas(A, b, A.shape[1], 0.1)
        sol = _calc_epic_ch(P, H, ts)
        assert sol.success
        assert sol.cost < 0.1
        assert sol.x.shape == (H.shape[0],)

    def test_compute_bounds_symmetric_and_positive(self):
        from bssunfold.core.unfold_epic import _compute_bounds

        lo, hi = _compute_bounds()
        assert lo < 0 < hi
        assert abs(lo) == pytest.approx(abs(hi))
        # exp bounds must stay representable
        assert np.exp(lo) > 0 and np.isfinite(np.exp(hi))

    def test_default_target_sigmas_shape(self):
        from bssunfold.core.unfold_epic import _default_target_sigmas

        A, b, _ = _synthetic_problem()
        ts = _default_target_sigmas(A, b, A.shape[1], 0.1)
        assert ts.shape == (A.shape[1],)
        assert np.all(ts > 0)
        assert np.all(np.isfinite(ts))

    def test_build_regularization_matrix_orders(self):
        from bssunfold.core.unfold_epic import _build_regularization_matrix

        n = 8
        assert _build_regularization_matrix(n, 0).shape == (n, n)
        assert _build_regularization_matrix(n, 1).shape == (n - 1, n)
        assert _build_regularization_matrix(n, 2).shape == (n - 2, n)

    def test_build_precision_identity_and_noise(self):
        from bssunfold.core.unfold_epic import _build_precision

        A, b, _ = _synthetic_problem()
        P, Wx = _build_precision(A, None)
        assert P.shape == (A.shape[1], A.shape[1])
        assert np.allclose(P, A.T @ A)
        assert Wx.shape == (A.shape[0], A.shape[0])
        Pv, Wv = _build_precision(A, 2.0)
        assert np.allclose(Pv, A.T @ (0.5 * A))
        assert np.allclose(Wv.T @ Wv, 0.5 * np.eye(A.shape[0]))


class TestDerivativeMatrixRegression:
    def test_order1_operator_values(self):
        """order-1 derivative rows must be [-1, 1] at the right positions."""
        from bssunfold.core._matrix_utils import create_derivative_matrix

        L = create_derivative_matrix(6, 1).toarray()
        expected = np.array(
            [
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0],
                [0, 0, -1, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 1],
            ]
        )
        assert np.array_equal(L, expected)

    def test_order1_full_rank(self):
        """The fixed operator must have full row rank (regression test)."""
        from bssunfold.core._matrix_utils import create_derivative_matrix

        for n in (4, 6, 10):
            L = create_derivative_matrix(n, 1).toarray()
            assert np.linalg.matrix_rank(L) == n - 1

    def test_order2_values(self):
        from bssunfold.core._matrix_utils import create_derivative_matrix

        L = create_derivative_matrix(5, 2).toarray()
        assert L[0, 0] == 1
        assert L[0, 1] == -2
        assert L[0, 2] == 1
        assert np.linalg.matrix_rank(L) == 3


class TestUnfoldEpicDetector:
    def test_basic(self, detector, readings):
        result = detector.unfold_epic(readings, save_result=False)
        assert isinstance(result, dict)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "EPIC"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert result["epic_converged"] is True
        assert result["regularization_order"] == 1

    def test_spectrum_absolute_present(self, detector, readings):
        result = detector.unfold_epic(readings, save_result=False)
        assert "spectrum_absolute" in result
        assert len(result["spectrum_absolute"]) == detector.n_energy_bins

    def test_montecarlo_errors(self, detector, readings):
        result = detector.unfold_epic(
            readings,
            calculate_errors=True,
            n_montecarlo=5,
            noise_level=0.05,
            random_state=42,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "spectrum_uncert_min" in result
        assert "spectrum_uncert_max" in result
        assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins

    def test_save_result(self, detector, readings):
        detector.results_history = {}
        detector.unfold_epic(readings, save_result=True)
        assert len(detector.results_history) == 1
        assert detector.results_history[list(detector.results_history)[0]]["method"] == "EPIC"

    def test_regularization_order_2(self, detector, readings):
        result = detector.unfold_epic(
            readings, regularization_order=2, save_result=False
        )
        assert result["regularization_order"] == 2
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_non_neg_false(self, detector, readings):
        result = detector.unfold_epic(readings, non_neg=False, save_result=False)
        assert result["non_neg"] is False
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_module_level_unfold_epic(self, detector, readings):
        from bssunfold.core.unfold_epic import unfold_epic

        result = unfold_epic(
            detector_names=detector.detector_names[:5],
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities={
                n: detector.sensitivities[n] for n in detector.detector_names[:5]
            },
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=detector._save_result,
            readings=readings,
            save_result=False,
        )
        assert result["method"] == "EPIC"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_combined_pipeline(self, detector, readings):
        pipeline = [
            {"method": "landweber", "params": {"max_iterations": 50}},
            {"method": "epic", "params": {"regularization_order": 1}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline, verbose=False)
        assert result["pipeline_info"]["stages"] == ["landweber", "epic"]
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_invalid_order_raises(self, detector, readings):
        with pytest.raises(ValueError, match="Unsupported regularization_order"):
            detector.unfold_epic(readings, regularization_order=5, save_result=False)


class TestUnfoldEpicExports:
    def test_core_exports(self):
        from bssunfold.core import solve_epic as core_solve
        from bssunfold.core import unfold_epic as core_unfold

        assert callable(core_solve)
        assert callable(core_unfold)


class TestEpicEdgeCases:
    def test_default_target_sigmas_degenerate_lstsq(self):
        """Zero design matrix: fall back to the measurement scale."""
        from bssunfold.core.unfold_epic import _default_target_sigmas

        A = np.zeros((4, 5))
        b = np.array([1.0, 2.0, 3.0, 4.0])
        ts = _default_target_sigmas(A, b, 5, 0.1)
        assert np.all(ts == pytest.approx(0.1 * 4.0))

    def test_default_target_sigmas_fully_degenerate(self):
        """Zero design matrix and zero readings: fall back to scale 1.0."""
        from bssunfold.core.unfold_epic import _default_target_sigmas

        A = np.zeros((3, 4))
        ts = _default_target_sigmas(A, np.zeros(3), 4, 0.5)
        assert np.all(ts == pytest.approx(0.5))

    def test_epic_bool_wrong_length_raises(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        with pytest.raises(ValueError, match="EPIC_bool length"):
            solve_epic(A, b, EPIC_bool=np.array([True, False]))

    def test_epic_bool_with_full_target_sigmas(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        mask = np.zeros(A.shape[1], dtype=bool)
        mask[:4] = True
        x = solve_epic(A, b, EPIC_bool=mask, target_sigmas=np.full(A.shape[1], 0.5))
        assert x.shape == (A.shape[1],)
        assert np.all(np.isfinite(x))

    def test_epic_bool_with_short_target_sigmas(self):
        from bssunfold.core.unfold_epic import solve_epic

        A, b, _ = _synthetic_problem()
        mask = np.zeros(A.shape[1], dtype=bool)
        mask[:4] = True
        x = solve_epic(A, b, EPIC_bool=mask, target_sigmas=np.full(4, 0.5))
        assert x.shape == (A.shape[1],)
        assert np.all(np.isfinite(x))

    def test_calc_epic_ch_overdetermined_h(self):
        """Nh > Nm branch: damped path must still run to completion."""
        from bssunfold.core.unfold_epic import _calc_epic_ch

        A, b, _ = _synthetic_problem()
        P = A.T @ A
        H = np.eye(A.shape[1])[:6]  # 6 rows > 4 columns used below
        H = np.random.default_rng(1).normal(size=(6, 4))
        sol = _calc_epic_ch(P[:4, :4], H, np.full(4, 0.5))
        assert sol.x.shape == (H.shape[0],)
        assert np.all(np.isfinite(sol.x))

    def test_calc_epic_ch_overdetermined_h_lsmr(self):
        """Nh > Nm with lsmr tr_solver exercises the damped tr_options."""
        from bssunfold.core.unfold_epic import _calc_epic_ch

        rng = np.random.default_rng(2)
        P = np.eye(4)
        H = rng.normal(size=(6, 4))
        sol = _calc_epic_ch(P, H, np.full(4, 0.5), LSQpar={"tr_solver": "lsmr"})
        assert sol.x.shape == (H.shape[0],)
        assert np.all(np.isfinite(sol.x))

    def test_warning_when_epic_not_converged(self, detector, readings):
        import importlib
        from unittest.mock import patch

        epic_mod = importlib.import_module("bssunfold.core.unfold_epic")

        class FakeMeta:
            def __init__(self):
                self.epic_converged = False
                self.epic_cost = 1e3
                self.epic_nfev = 1
                self.beta_min = 0.0
                self.beta_max = 1.0
                self.target_sigmas = []

        n = detector.n_energy_bins
        with patch.object(
            epic_mod, "_epic_weights",
            return_value=(
                np.eye(5),
                np.eye(n)[: n - 1],
                np.eye(n - 1),
                vars(FakeMeta()),
            ),
        ):
            with pytest.warns(UserWarning, match="did not converge"):
                result = detector.unfold_epic(readings, save_result=False)
        assert result["epic_converged"] is False

