"""Tests for the BON95 parametric unfolding method (unfold_parametric2)."""

import numpy as np
import pytest
from numpy.testing import assert_array_less, assert_allclose

from bssunfold import Detector
from bssunfold.core.unfold_parametric2 import (
    _Fth,
    _Fepi,
    _Fint,
    _Ff,
    bon95_model,
    bon95_spectrum,
    _solve_linear_coefficients,
    solve_bon95_parametric,
    solve_bon95_cvxpy,
    solve_bon95_qpsolvers,
    solve_bon95_combined,
    directed_divergence_iteration,
    solve_parametric2,
    unfold_parametric2,
    _Tth,
    _build_measurement_uncertainties,
    _clean_edge_bins,
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


def _make_ln_steps(E):
    """Compute log-energy bin widths (same as in production code)."""
    n = len(E)
    log_steps = np.zeros(n)
    log_e = np.log10(E + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if n > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if n > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    return log_steps * np.log(10)


# ─── Component function tests ─────────────────────────────────────


class TestBon95Components:
    def test_Fth_shape(self, energy_grid):
        result = _Fth(energy_grid)
        assert result.shape == energy_grid.shape

    def test_Fth_nonnegative(self, energy_grid):
        result = _Fth(energy_grid)
        assert_array_less(-1e-30, result)

    def test_Fth_peak_near_Tth(self):
        E_peak = np.linspace(1e-10, 1e-6, 1000)
        F = _Fth(E_peak)
        peak_idx = np.argmax(F)
        # Peak should be near Tth = 3.5e-8 MeV
        assert abs(E_peak[peak_idx] - _Tth) / _Tth < 0.5

    def test_Fepi_shape(self, energy_grid):
        result = _Fepi(energy_grid, b=1.0)
        assert result.shape == energy_grid.shape

    def test_Fepi_nonnegative(self, energy_grid):
        result = _Fepi(energy_grid, b=1.0)
        assert_array_less(-1e-30, result)

    def test_Fint_shape(self, energy_grid):
        result = _Fint(energy_grid)
        assert result.shape == energy_grid.shape

    def test_Fint_range(self, energy_grid):
        result = _Fint(energy_grid)
        assert_array_less(-1e-10, result)
        assert_array_less(result, 1.0 + 1e-10)

    def test_Ff_shape(self, energy_grid):
        result = _Ff(energy_grid, Tf=2.0, c=1.0)
        assert result.shape == energy_grid.shape

    def test_Ff_nonnegative(self, energy_grid):
        result = _Ff(energy_grid, Tf=2.0, c=1.0)
        assert_array_less(-1e-30, result)


# ─── Combined model tests ─────────────────────────────────────────


class TestBon95Model:
    def test_shape(self, energy_grid):
        result = bon95_model(energy_grid, b=1.0, Tf=2.0, c=1.0,
                             a1=1.0, a2=1.0, a3=1.0, a4=1.0)
        assert result.shape == energy_grid.shape

    def test_nonnegative(self, energy_grid):
        result = bon95_model(energy_grid, b=1.0, Tf=2.0, c=1.0,
                             a1=1.0, a2=1.0, a3=1.0, a4=1.0)
        assert_array_less(-1e-30, result)

    def test_thermal_only(self, energy_grid):
        result = bon95_model(energy_grid, b=1.0, Tf=2.0, c=1.0,
                             a1=1.0, a2=0.0, a3=0.0, a4=0.0)
        low = energy_grid < 1e-6
        high = energy_grid > 1.0
        assert result[low].sum() > result[high].sum()

    def test_fast_only(self, energy_grid):
        result = bon95_model(energy_grid, b=1.0, Tf=2.0, c=1.0,
                             a1=0.0, a2=0.0, a3=0.0, a4=1.0)
        low = energy_grid < 1e-6
        high = energy_grid > 1.0
        assert result[high].sum() > result[low].sum()

    def test_spectrum_shape(self, energy_grid):
        result = bon95_spectrum(energy_grid, b=1.0, Tf=2.0, c=1.0,
                                a1=1.0, a2=1.0, a3=1.0, a4=1.0)
        assert result.shape == energy_grid.shape

    def test_spectrum_nonnegative(self, energy_grid):
        result = bon95_spectrum(energy_grid, b=1.0, Tf=2.0, c=1.0,
                                a1=1.0, a2=1.0, a3=1.0, a4=1.0)
        assert_array_less(-1e-30, result)


# ─── Linear coefficient solver tests ──────────────────────────────


class TestLinearCoefficients:
    def test_returns_four_coefficients(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        a, chi2 = _solve_linear_coefficients(A, b, E, ln_steps, b=1.0, Tf=2.0, c=1.0)
        assert a.shape == (4,)
        assert isinstance(chi2, float)

    def test_coefficients_nonnegative(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        a, _ = _solve_linear_coefficients(A, b, E, ln_steps, b=1.0, Tf=2.0, c=1.0)
        assert_array_less(-1e-10, a)


# ─── Grid search parametric fit tests ─────────────────────────────


class TestSolveBon95Parametric:
    def test_returns_best_params(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        best, chi2, top = solve_bon95_parametric(
            A, b, E, ln_steps,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'b' in best
        assert 'Tf' in best
        assert 'c' in best
        assert 'a1' in best
        assert 'a2' in best
        assert 'a3' in best
        assert 'a4' in best
        assert isinstance(chi2, float)
        assert len(top) <= 5

    def test_chi2_finite(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        best, chi2, _ = solve_bon95_parametric(
            A, b, E, ln_steps,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert np.isfinite(chi2)


# ─── Directed divergence iteration tests ──────────────────────────


class TestDirectedDivergence:
    def test_converges_on_easy_problem(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        # Start from a reasonable initial guess
        phi0 = np.ones(len(E)) * 0.01

        phi, n_iter, chi2, converged = directed_divergence_iteration(
            A, b, E, ln_steps, phi0, max_iter=200,
        )
        assert phi.shape == E.shape
        assert_array_less(-1e-30, phi)
        assert n_iter > 0
        assert np.isfinite(chi2)

    def test_improves_chi2(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        phi0 = np.ones(len(E)) * 0.01
        M_p0 = A @ (phi0 * ln_steps)
        chi2_0 = np.mean((M_p0 - b) ** 2)

        phi, _, chi2_final, _ = directed_divergence_iteration(
            A, b, E, ln_steps, phi0, max_iter=100,
        )
        assert chi2_final <= chi2_0 + 1e-10


# ─── Full pipeline solver tests ───────────────────────────────────


class TestSolveParametric2:
    def test_returns_spectrum(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, success, message, nfev = solve_parametric2(
            A, b, E, ln_steps,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(message, str)
        assert isinstance(nfev, int)

    def test_nonnegative_output(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, _, _, _ = solve_parametric2(
            A, b, E, ln_steps,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert_array_less(-1e-30, spectrum)


# ─── Detector method integration tests ────────────────────────────


class TestDetectorUnfoldParametric2:
    def test_basic_call(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in result
        assert 'doserates' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_save_result(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            save_result=True,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in result

    def test_with_custom_ranges(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.5, 2.0, 2),
            Tf_range=(0.5, 8.0, 2),
            c_range=(0.5, 2.5, 2),
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)


# ─── Comparison tests (parametric vs parametric2) ─────────────────


class TestParametricComparison:
    """Compare FRUIT parametric (unfold_parametric) with BON95 (unfold_parametric2)."""

    def test_both_produce_results(self, detector, sample_readings):
        r1 = detector.unfold_parametric(readings=sample_readings)
        r2 = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in r1
        assert 'spectrum' in r2
        assert r1['spectrum'].shape == r2['spectrum'].shape

    def test_both_nonnegative(self, detector, sample_readings):
        r1 = detector.unfold_parametric(readings=sample_readings)
        r2 = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert_array_less(-1e-30, r1['spectrum'])
        assert_array_less(-1e-30, r2['spectrum'])

    def test_both_compute_doserates(self, detector, sample_readings):
        r1 = detector.unfold_parametric(readings=sample_readings)
        r2 = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'doserates' in r1
        assert 'doserates' in r2
        assert len(r1['doserates']) > 0
        assert len(r2['doserates']) > 0

    def test_residual_finite(self, detector, sample_readings):
        r2 = detector.unfold_parametric2(
            readings=sample_readings,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        # Residual should be present and finite
        if 'residual' in r2:
            assert np.all(np.isfinite(r2['residual']))


# ─── Measurement uncertainty tests ────────────────────────────────


class TestMeasurementUncertainties:
    def test_shape(self):
        b = np.array([1.0, 2.0, 3.0])
        result = _build_measurement_uncertainties(b, noise_level=0.05)
        assert result.shape == b.shape

    def test_positive(self):
        b = np.array([1.0, 2.0, 3.0])
        result = _build_measurement_uncertainties(b, noise_level=0.05)
        assert_array_less(0.0, result)

    def test_scales_with_noise_level(self):
        b = np.array([1.0, 2.0, 3.0])
        r1 = _build_measurement_uncertainties(b, noise_level=0.01)
        r2 = _build_measurement_uncertainties(b, noise_level=0.10)
        assert_allclose(r2, r1 * 10.0, rtol=1e-10)


# ─── Edge-bin cleaning tests ──────────────────────────────────────


class TestCleanEdgeBins:
    def test_no_change_for_normal_spectrum(self):
        phi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _clean_edge_bins(phi)
        assert_allclose(result, phi)

    def test_zeros_first_bin_if_spike(self):
        phi = np.array([100.0, 1.0, 2.0, 3.0, 4.0])
        result = _clean_edge_bins(phi, factor=10.0)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_zeros_last_bin_if_spike(self):
        phi = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _clean_edge_bins(phi, factor=10.0)
        assert result[-1] == 0.0
        assert result[-2] == 4.0

    def test_does_not_modify_input(self):
        phi = np.array([100.0, 1.0, 2.0, 3.0, 4.0])
        original = phi.copy()
        _clean_edge_bins(phi, factor=10.0)
        assert_allclose(phi, original)

    def test_short_array_unchanged(self):
        phi = np.array([100.0, 1.0])
        result = _clean_edge_bins(phi)
        assert_allclose(result, phi)

    def test_all_zeros_unchanged(self):
        phi = np.zeros(10)
        result = _clean_edge_bins(phi)
        assert_allclose(result, phi)


# ─── Dict initial_spectrum tests ──────────────────────────────────


class TestDictInitialSpectrum:
    def test_unfold_parametric2_accepts_dict_initial_spectrum(self, detector, sample_readings):
        # Get a spectrum from a previous run to use as initial
        r1 = detector.unfold_parametric(readings=sample_readings)
        # Pass the entire result dict as initial_spectrum
        result = detector.unfold_parametric2(
            readings=sample_readings,
            initial_spectrum=r1,
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_unfold_parametric2_accepts_array_initial_spectrum(self, detector, sample_readings):
        r1 = detector.unfold_parametric(readings=sample_readings)
        result = detector.unfold_parametric2(
            readings=sample_readings,
            initial_spectrum=r1['spectrum'],
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in result


# ─── SQP solver tests ─────────────────────────────────────────────


class TestSolveBon95Cvxpy:
    def test_returns_spectrum(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, success, message, nfev = solve_bon95_cvxpy(
            A, b, E, ln_steps,
            max_iter=10,
        )
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)
        assert isinstance(message, str)

    def test_nonnegative_output(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, _, _, _ = solve_bon95_cvxpy(A, b, E, ln_steps, max_iter=10)
        assert_array_less(-1e-30, spectrum)


class TestSolveBon95Qpsolvers:
    def test_returns_spectrum(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, success, message, nfev = solve_bon95_qpsolvers(
            A, b, E, ln_steps,
            max_iter=10,
        )
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)

    def test_nonnegative_output(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, _, _, _ = solve_bon95_qpsolvers(A, b, E, ln_steps, max_iter=10)
        assert_array_less(-1e-30, spectrum)


class TestSolveBon95Combined:
    def test_returns_spectrum(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, success, message, nfev = solve_bon95_combined(
            A, b, E, ln_steps,
            max_iter_qp=10,
        )
        assert spectrum.shape == E.shape
        assert isinstance(success, bool)

    def test_nonnegative_output(self, detector, sample_readings):
        E = detector.E_MeV
        ln_steps = _make_ln_steps(E)
        selected = list(sample_readings.keys())
        A = np.array([detector.sensitivities[n] for n in selected])
        b = np.array([sample_readings[n] for n in selected])

        spectrum, _, _, _ = solve_bon95_combined(A, b, E, ln_steps, max_iter_qp=10)
        assert_array_less(-1e-30, spectrum)


# ─── Optimizer parameter tests ────────────────────────────────────


class TestOptimizerParam:
    def test_grid_optimizer_default(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            optimizer="grid",
            b_range=(0.8, 1.5, 3),
            Tf_range=(1.0, 5.0, 3),
            c_range=(0.8, 2.0, 3),
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_cvxpy_optimizer(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            optimizer="cvxpy",
            max_iter_qp=10,
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_qpsolvers_optimizer(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            optimizer="qpsolvers",
            max_iter_qp=10,
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_combined_optimizer(self, detector, sample_readings):
        result = detector.unfold_parametric2(
            readings=sample_readings,
            optimizer="combined",
            max_iter_qp=10,
        )
        assert 'spectrum' in result
        assert result['spectrum'].shape == (detector.n_energy_bins,)

    def test_invalid_optimizer_raises(self, detector, sample_readings):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            detector.unfold_parametric2(
                readings=sample_readings,
                optimizer="invalid",
            )

    def test_all_optimizers_nonnegative(self, detector, sample_readings):
        for opt in ["grid", "cvxpy", "qpsolvers", "combined"]:
            result = detector.unfold_parametric2(
                readings=sample_readings,
                optimizer=opt,
                max_iter_qp=10,
                b_range=(0.8, 1.5, 3),
                Tf_range=(1.0, 5.0, 3),
                c_range=(0.8, 2.0, 3),
            )
            assert_array_less(-1e-30, result['spectrum'], err_msg=f"Failed for optimizer={opt}")
