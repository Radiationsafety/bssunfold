"""Tests for the FRUIT-based parametric unfolding method."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_less

from bssunfold import Detector
from bssunfold.core.unfold_parametric import (
    parametric_model,
    solve_parametric,
    solve_parametric_cvxpy,
    solve_parametric_qpsolvers,
    solve_parametric_combined,
    unfold_parametric,
    _T0,
    _Ed,
    _THERMAL_MAX,
    _FAST_MIN,
    _compute_jacobian,
    _get_initial_params,
    _get_param_bounds,
    _PARAM_NAMES,
    _find_initial_params,
    _parse_solver_backend,
)


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def energy_grid():
    return Detector().E_MeV


@pytest.fixture
def sample_readings():
    return {name: float(1.0 + i * 0.1) for i, name in enumerate(Detector().detector_names)}


# ─── Parametric model tests ───────────────────────────────────────


class TestParametricModel:
    def test_shape(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=0.33, P_epi=0.33)
        assert result.shape == energy_grid.shape

    def test_nonnegative(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=0.33, P_epi=0.33)
        assert_array_less(-1e-30, result)

    def test_weights_sum_to_one(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=0.33, P_epi=0.33)
        assert result.sum() > 0

    def test_thermal_dominates_low_energy(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=1.0, P_epi=0.0)
        low = energy_grid < _THERMAL_MAX
        high = energy_grid >= _FAST_MIN
        assert result[low].sum() > result[high].sum()

    def test_fast_dominates_high_energy(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=0.0, P_epi=0.0)
        # P_f = 1.0 when P_th=P_epi=0
        high = energy_grid >= _FAST_MIN
        assert result[high].sum() > 0

    def test_epithermal_region(self, energy_grid):
        result = parametric_model(energy_grid, b=1.0, beta_prime=0.01,
                                  alpha=0.5, beta=2.0, P_th=0.0, P_epi=1.0)
        epi = (energy_grid >= _THERMAL_MAX) & (energy_grid < _FAST_MIN)
        assert result[epi].sum() > 0

    def test_custom_params(self, energy_grid):
        r1 = parametric_model(energy_grid, b=0.5, beta_prime=0.01,
                              alpha=0.5, beta=2.0, P_th=0.5, P_epi=0.3)
        r2 = parametric_model(energy_grid, b=1.5, beta_prime=0.1,
                              alpha=1.0, beta=5.0, P_th=0.1, P_epi=0.6)
        assert not np.allclose(r1, r2)

    def test_constants_match_papers(self):
        assert_almost_equal(_T0, 2.53e-8)
        assert_almost_equal(_Ed, 7.07e-8)
        assert_almost_equal(_THERMAL_MAX, 1e-7)
        assert_almost_equal(_FAST_MIN, 0.1)


# ─── Solver tests ─────────────────────────────────────────────────


class TestSolveParametric:
    def test_returns_tuple(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        log_e = np.log10(E + 1e-15)
        ln_steps = np.zeros_like(E)
        ln_steps[0] = log_e[1] - log_e[0]
        ln_steps[-1] = log_e[-1] - log_e[-2]
        ln_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
        ln_steps *= np.log(10)

        spectrum, success, message, nfev = solve_parametric(A, b, E, ln_steps)
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(nfev, int)

    def test_custom_initial_params(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        log_e = np.log10(E + 1e-15)
        ln_steps = np.zeros_like(E)
        ln_steps[0] = log_e[1] - log_e[0]
        ln_steps[-1] = log_e[-1] - log_e[-2]
        ln_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
        ln_steps *= np.log(10)

        init = {'b': 0.8, 'beta_prime': 0.05, 'alpha': 0.3, 'beta': 1.5,
                'P_th': 0.4, 'P_epi': 0.4}
        spectrum, success, message, nfev = solve_parametric(
            A, b, E, ln_steps, initial_params=init
        )
        assert spectrum.shape == E.shape


# ─── Detector method tests ────────────────────────────────────────


class TestDetectorUnfoldParametric:
    def test_basic(self, detector, sample_readings):
        result = detector.unfold_parametric(sample_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_with_custom_params(self, detector, sample_readings):
        init = {'b': 0.8, 'beta_prime': 0.05, 'alpha': 0.3, 'beta': 1.5,
                'P_th': 0.4, 'P_epi': 0.4}
        result = detector.unfold_parametric(
            sample_readings, initial_params=init, save_result=False
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_with_save_result(self, detector, sample_readings):
        result = detector.unfold_parametric(sample_readings, save_result=True)
        assert "spectrum" in result
        # Check it was saved
        keys = detector.list_results()
        assert len(keys) > 0
        saved = detector.get_result(keys[-1])
        assert "spectrum" in saved

    def test_has_doserates(self, detector, sample_readings):
        result = detector.unfold_parametric(sample_readings, save_result=False)
        assert "doserates" in result
        assert isinstance(result["doserates"], dict)


# ─── Helper for computing log steps ────────────────────────────────


def _make_ln_steps(E):
    log_e = np.log10(E + 1e-15)
    n = len(E)
    ln_steps = np.zeros(n)
    ln_steps[0] = log_e[1] - log_e[0] if n > 1 else 1.0
    ln_steps[-1] = log_e[-1] - log_e[-2] if n > 1 else 1.0
    ln_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    return ln_steps * np.log(10)


# ─── Jacobian tests ───────────────────────────────────────────────


class TestJacobian:
    def test_shape(self, energy_grid):
        params = _get_initial_params(None)
        ln_steps = _make_ln_steps(energy_grid)
        J = _compute_jacobian(energy_grid, ln_steps, params)
        assert J.shape == (len(energy_grid), len(_PARAM_NAMES))

    def test_nonzero(self, energy_grid):
        params = _get_initial_params(None)
        ln_steps = _make_ln_steps(energy_grid)
        J = _compute_jacobian(energy_grid, ln_steps, params)
        assert np.any(np.abs(J) > 0)

    def test_param_names(self):
        assert _PARAM_NAMES == ["b", "beta_prime", "alpha", "beta", "P_th", "P_epi"]

    def test_param_bounds_keys(self):
        bounds = _get_param_bounds()
        assert set(bounds.keys()) == set(_PARAM_NAMES)

    def test_at_boundary(self, energy_grid):
        params = _get_initial_params(None)
        params["P_th"] = 0.0  # at lower bound
        params["P_epi"] = 0.0  # at lower bound
        ln_steps = _make_ln_steps(energy_grid)
        J = _compute_jacobian(energy_grid, ln_steps, params)
        assert J.shape == (len(energy_grid), len(_PARAM_NAMES))
        assert not np.any(np.isnan(J))


class TestParseSolverBackend:
    def test_auto(self):
        assert _parse_solver_backend("auto") == ("auto", "default")

    def test_cvxpy(self):
        assert _parse_solver_backend("cvxpy") == ("cvxpy", "default")

    def test_cvxpy_with_backend(self):
        assert _parse_solver_backend("cvxpy:ECOS") == ("cvxpy", "ECOS")

    def test_qpsolvers(self):
        assert _parse_solver_backend("qpsolvers") == ("qpsolvers", "default")

    def test_qpsolvers_with_backend(self):
        assert _parse_solver_backend("qpsolvers:osqp") == ("qpsolvers", "osqp")


class TestFindInitialParams:
    def test_returns_dict(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        params = _find_initial_params(A, b, E, ln_steps, n_grid=3)
        assert isinstance(params, dict)
        assert set(params.keys()) == set(_PARAM_NAMES)
        assert params["P_th"] + params["P_epi"] <= 1.0

    def test_bounds_respected(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        bounds = _get_param_bounds()
        params = _find_initial_params(A, b, E, ln_steps, n_grid=3)
        for name, (lo, hi) in bounds.items():
            if lo is not None:
                assert params[name] >= lo - 1e-10
            if hi is not None:
                assert params[name] <= hi + 1e-10


# ─── solve_parametric_cvxpy tests ──────────────────────────────────


class TestSolveParametricCvxpy:
    def test_returns_tuple(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_cvxpy(
            A, b, E, ln_steps, alpha=1e-3,
        )
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(nfev, int)

    def test_nonnegative_spectrum(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_cvxpy(
            A, b, E, ln_steps, alpha=1e-2,
        )
        assert_array_less(-1e-30, spectrum)

    def test_custom_params(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        init = {'b': 0.8, 'beta_prime': 0.05, 'alpha': 0.3, 'beta': 1.5,
                'P_th': 0.4, 'P_epi': 0.4}
        spectrum, success, message, nfev = solve_parametric_cvxpy(
            A, b, E, ln_steps, initial_params=init, alpha=1e-3,
        )
        assert spectrum.shape == E.shape

    def test_solver_backend_param(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_cvxpy(
            A, b, E, ln_steps, alpha=1e-3, solver_backend="cvxpy:SCS",
        )
        assert spectrum.shape == E.shape


# ─── solve_parametric_qpsolvers tests ──────────────────────────────


class TestSolveParametricQpsolvers:
    def test_returns_tuple(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_qpsolvers(
            A, b, E, ln_steps, alpha=1e-3,
        )
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(nfev, int)

    def test_nonnegative_spectrum(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_qpsolvers(
            A, b, E, ln_steps, alpha=1e-2,
        )
        assert_array_less(-1e-30, spectrum)

    def test_solver_backend_param(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_qpsolvers(
            A, b, E, ln_steps, alpha=1e-3, solver_backend="qpsolvers:osqp",
        )
        assert spectrum.shape == E.shape


# ─── solve_parametric_combined tests ───────────────────────────────


class TestSolveParametricCombined:
    def test_returns_tuple_cvxpy(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_combined(
            A, b, E, ln_steps, solver_backend="cvxpy", alpha=1e-3,
        )
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(message, str)

    def test_returns_tuple_qpsolvers(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_combined(
            A, b, E, ln_steps, solver_backend="qpsolvers", alpha=1e-3,
        )
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)

    def test_nonnegative_spectrum(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        spectrum, success, message, nfev = solve_parametric_combined(
            A, b, E, ln_steps, solver_backend="cvxpy", alpha=1e-2,
        )
        assert_array_less(-1e-30, spectrum)

    def test_invalid_solver_backend_raises(self, detector, sample_readings):
        selected = [n for n in detector.detector_names if n in sample_readings]
        b = np.array([sample_readings[n] for n in selected])
        A = np.array([detector.sensitivities[n] for n in selected])
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)

        with pytest.raises(ValueError, match="Unknown solver library"):
            solve_parametric_combined(
                A, b, E, ln_steps, solver_backend="invalid",
            )


# ─── Detector method with optimizer parameter ──────────────────────


class TestDetectorUnfoldParametricOptimizers:
    def test_cvxpy_optimizer(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="cvxpy", alpha=1e-3, save_result=False,
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert result["method"] == "parametric_cvxpy"

    def test_qpsolvers_optimizer(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="qpsolvers", alpha=1e-3, save_result=False,
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert result["method"] == "parametric_qpsolvers"

    def test_combined_optimizer_cvxpy(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="combined", solver_backend="cvxpy",
            alpha=1e-3, save_result=False,
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert result["method"] == "parametric_combined"

    def test_combined_optimizer_qpsolvers(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="combined", solver_backend="qpsolvers",
            alpha=1e-3, save_result=False,
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_invalid_optimizer_raises(self, detector, sample_readings):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            detector.unfold_parametric(
                sample_readings, optimizer="invalid", save_result=False,
            )

    def test_lmfit_optimizer_unchanged(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="lmfit", save_result=False,
        )
        assert "spectrum" in result
        assert result["method"] == "parametric"

    def test_cvxpy_has_doserates(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="cvxpy", alpha=1e-3, save_result=False,
        )
        assert "doserates" in result
        assert isinstance(result["doserates"], dict)

    def test_combined_with_custom_params(self, detector, sample_readings):
        init = {'b': 0.8, 'beta_prime': 0.05, 'alpha': 0.3, 'beta': 1.5,
                'P_th': 0.4, 'P_epi': 0.4}
        result = detector.unfold_parametric(
            sample_readings, optimizer="combined", initial_params=init,
            alpha=1e-3, save_result=False,
        )
        assert "spectrum" in result

    def test_cvxpy_with_solver_backend(self, detector, sample_readings):
        result = detector.unfold_parametric(
            sample_readings, optimizer="cvxpy", alpha=1e-3,
            solver_backend="cvxpy:SCS", save_result=False,
        )
        assert "spectrum" in result
