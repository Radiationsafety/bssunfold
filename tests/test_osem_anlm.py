"""Tests for the OSEM-ANLM unfolding method.

Covers the two-stage asymptotic non-local means (ANLM) filter, the noise
estimator and the OSEM-ANLM solver (Jamaati et al. 2026, Scientific
Reports, https://doi.org/10.1038/s41598-026-70607-1) at both the core
solver level and through the ``Detector`` wrapper.
"""

import numpy as np
import pytest

from bssunfold.core import (
    anlm_filter_1d,
    estimate_noise_1d,
    solve_osem,
    solve_osem_anlm,
)


@pytest.fixture
def detector():
    from bssunfold import Detector

    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def all_readings(detector):
    n = detector.n_detectors
    return {
        name: 100.0 * (1 + 0.1 * i)
        for i, name in enumerate(detector.detector_names[:n])
    }


def _response_matrix(m=6, n=30, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((m, n)) + 0.1


# ============================================================================
# ANLM filter (1D)
# ============================================================================


class TestAnlmFilter1d:
    def test_identity_for_unit_search_window(self):
        x = np.abs(np.sin(np.linspace(0, 3, 41))) + 0.1
        np.testing.assert_array_equal(anlm_filter_1d(x, search_window=1), x)

    def test_constant_signal_unchanged(self):
        x = np.full(60, 3.7)
        np.testing.assert_allclose(anlm_filter_1d(x), x)

    def test_output_properties(self):
        rng = np.random.default_rng(0)
        x = np.abs(rng.standard_normal(80)) + 0.2
        f = anlm_filter_1d(x)
        assert f.shape == x.shape
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0)

    def test_smooths_white_noise(self):
        rng = np.random.default_rng(1)
        x = 5.0 + rng.standard_normal(300)
        f = anlm_filter_1d(x, log_space=False)
        rough_before = np.abs(np.diff(x, 2)).mean()
        rough_after = np.abs(np.diff(f, 2)).mean()
        assert rough_after < 0.9 * rough_before

    def test_smooths_noisy_plateau_log_space(self):
        rng = np.random.default_rng(2)
        x = np.full(120, 10.0) * (1 + 0.1 * rng.standard_normal(120))
        f = anlm_filter_1d(x, log_space=True)
        assert np.abs(np.log(f)).std() < np.abs(np.log(x)).std()

    def test_single_bin(self):
        np.testing.assert_array_equal(anlm_filter_1d(np.array([3.0])), [3.0])

    def test_two_bins_finite(self):
        f = anlm_filter_1d(np.array([3.0, 1.0]))
        assert np.all(np.isfinite(f))

    def test_zero_input_finite(self):
        f = anlm_filter_1d(np.zeros(20))
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0)

    def test_explicit_h_changes_smoothing(self):
        rng = np.random.default_rng(3)
        x = np.full(100, 7.0) * (1 + 0.2 * rng.standard_normal(100))
        f_small = anlm_filter_1d(x, h=0.01)
        f_large = anlm_filter_1d(x, h=0.5)
        # A larger filter parameter must smooth the plateau more.
        assert np.abs(np.log(f_large)).std() <= np.abs(np.log(f_small)).std()

    def test_invalid_search_window(self):
        with pytest.raises(ValueError, match="search_window"):
            anlm_filter_1d(np.ones(10), search_window=0)

    def test_invalid_similarity_window(self):
        with pytest.raises(ValueError, match="similarity_window"):
            anlm_filter_1d(np.ones(10), similarity_window=-1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            anlm_filter_1d(np.ones(10), alpha=0.0)

    def test_invalid_h(self):
        with pytest.raises(ValueError, match="h must be"):
            anlm_filter_1d(np.ones(10), h=0.0)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            anlm_filter_1d(np.array([]))


class TestEstimateNoise1d:
    def test_white_noise_estimate(self):
        rng = np.random.default_rng(4)
        x = 2.0 + 0.3 * rng.standard_normal(2000)
        sigma = estimate_noise_1d(x)
        assert 0.15 < sigma < 0.6

    def test_linear_ramp_near_zero(self):
        x = np.linspace(0.0, 10.0, 500)
        assert estimate_noise_1d(x) < 1e-8

    def test_too_short_signals(self):
        assert estimate_noise_1d(np.array([1.0])) == 0.0
        assert estimate_noise_1d(np.array([1.0, 2.0])) == 0.0

    def test_scales_with_noise_level(self):
        rng = np.random.default_rng(5)
        base = 2.0 + rng.standard_normal(1500)
        loud = 2.0 + 5.0 * rng.standard_normal(1500)
        assert estimate_noise_1d(loud) > 2.0 * estimate_noise_1d(base)


# ============================================================================
# Core solver
# ============================================================================


class TestSolveOsemAnlm:
    def test_basic(self):
        A = _response_matrix()
        b = A @ np.exp(-np.linspace(0, 3, 30))
        x, iterations, converged = solve_osem_anlm(A, b, np.ones(30))
        assert len(x) == 30
        assert iterations > 0
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)
        assert converged in (True, False)

    def test_identity_filter_matches_plain_osem(self):
        A = _response_matrix(6, 24, seed=2)
        b = A @ np.ones(24)
        x_ref, it_ref, conv_ref = solve_osem(A, b, np.ones(24), max_iterations=25)
        x, it, conv = solve_osem_anlm(
            A, b, np.ones(24), max_iterations=25, search_window=1
        )
        np.testing.assert_allclose(x, x_ref, atol=1e-12)
        assert it == it_ref
        assert conv == conv_ref

    def test_post_mode_equals_osem_then_filter(self):
        A = _response_matrix(5, 20, seed=6)
        rng = np.random.default_rng(7)
        b = (A @ np.exp(-np.linspace(0, 2, 20))) * (1 + 0.05 * rng.standard_normal(5))
        x_post, it_post, conv_post = solve_osem_anlm(
            A, b, np.ones(20), max_iterations=15, anlm_mode="post", h=0.3
        )
        x_ref, it_ref, conv_ref = solve_osem(A, b, np.ones(20), max_iterations=15)
        x_filtered = anlm_filter_1d(x_ref, h=0.3)
        np.testing.assert_allclose(x_post, x_filtered, rtol=1e-12)
        assert it_post == it_ref
        assert conv_post == conv_ref

    def test_subset_mode_differs_from_plain_osem(self):
        A = _response_matrix(6, 30, seed=8)
        rng = np.random.default_rng(9)
        b = (A @ np.ones(30)) * (1 + 0.05 * rng.standard_normal(6))
        x_plain, _, _ = solve_osem(A, b, np.ones(30), max_iterations=40)
        x_anlm, _, _ = solve_osem_anlm(
            A, b, np.ones(30), max_iterations=40, h=0.2
        )
        assert not np.allclose(x_anlm, x_plain, atol=1e-10)

    def test_subsets_variants(self):
        A = _response_matrix(6, 18, seed=10)
        b = A @ np.ones(18)
        for n_subsets in (1, 2, 3):
            x, _, _ = solve_osem_anlm(
                A, b, np.ones(18), max_iterations=20, n_subsets=n_subsets
            )
            assert len(x) == 18
            assert np.all(np.isfinite(x))
            assert np.all(x >= 0)

    def test_tiny_tolerance_runs_all_iterations(self):
        A = _response_matrix(4, 12, seed=11)
        b = A @ np.ones(12)
        x, iterations, converged = solve_osem_anlm(
            A, b, np.ones(12), max_iterations=7, tolerance=0.0
        )
        assert iterations == 7
        assert converged is False

    def test_deterministic(self):
        A = _response_matrix(5, 15, seed=12)
        b = A @ np.ones(15)
        x1, it1, c1 = solve_osem_anlm(A, b, np.ones(15), max_iterations=10)
        x2, it2, c2 = solve_osem_anlm(A, b, np.ones(15), max_iterations=10)
        np.testing.assert_array_equal(x1, x2)
        assert (it1, c1) == (it2, c2)

    def test_single_detector(self):
        A = _response_matrix(1, 8, seed=13)
        b = A @ np.ones(8)
        x, iterations, converged = solve_osem_anlm(A, b, np.ones(8), max_iterations=5)
        assert np.all(np.isfinite(x))
        assert iterations >= 1
        assert converged in (True, False)

    def test_invalid_n_subsets_zero(self):
        A = _response_matrix(4, 8, seed=14)
        with pytest.raises(ValueError, match="n_subsets must be >= 1"):
            solve_osem_anlm(A, A @ np.ones(8), np.ones(8), n_subsets=0)

    def test_invalid_n_subsets_too_large(self):
        A = _response_matrix(4, 8, seed=15)
        with pytest.raises(ValueError, match="must not exceed"):
            solve_osem_anlm(A, A @ np.ones(8), np.ones(8), n_subsets=5)

    def test_invalid_anlm_mode(self):
        A = _response_matrix(4, 8, seed=16)
        with pytest.raises(ValueError, match="anlm_mode"):
            solve_osem_anlm(A, A @ np.ones(8), np.ones(8), anlm_mode="bogus")

    def test_invalid_h(self):
        A = _response_matrix(4, 8, seed=17)
        with pytest.raises(ValueError, match="h must be"):
            solve_osem_anlm(A, A @ np.ones(8), np.ones(8), h=-0.5)

    def test_invalid_window_parameters(self):
        A = _response_matrix(4, 8, seed=18)
        b = A @ np.ones(8)
        with pytest.raises(ValueError, match="search_window"):
            solve_osem_anlm(A, b, np.ones(8), search_window=0)
        with pytest.raises(ValueError, match="similarity_window"):
            solve_osem_anlm(A, b, np.ones(8), similarity_window=0)
        with pytest.raises(ValueError, match="alpha"):
            solve_osem_anlm(A, b, np.ones(8), alpha=0.0)


# ============================================================================
# Detector wrapper
# ============================================================================


class TestUnfoldOsemAnlm:
    def test_basic(self, detector, all_readings):
        result = detector.unfold_osem_anlm(all_readings, save_result=False)
        assert "spectrum" in result
        assert "energy" in result
        assert "doserates" in result
        assert result["method"] == "OSEM-ANLM"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert np.all(np.isfinite(result["spectrum"]))

    def test_extra_output(self, detector, all_readings):
        result = detector.unfold_osem_anlm(
            all_readings,
            n_subsets=3,
            h=0.4,
            search_window=9,
            similarity_window=5,
            alpha=0.8,
            anlm_mode="post",
            log_space=False,
            save_result=False,
        )
        assert result["n_subsets"] == 3
        assert result["h"] == 0.4
        assert result["search_window"] == 9
        assert result["similarity_window"] == 5
        assert result["alpha"] == 0.8
        assert result["anlm_mode"] == "post"
        assert result["log_space"] is False

    def test_single_detector(self, detector, readings):
        result = detector.unfold_osem_anlm(readings, save_result=False)
        assert "spectrum" in result
        assert result["method"] == "OSEM-ANLM"

    def test_save_result(self, detector, readings):
        detector.clear_results()
        detector.unfold_osem_anlm(readings, save_result=True)
        assert detector.current_result is not None
        assert detector.current_result["method"] == "OSEM-ANLM"

    def test_montecarlo_errors(self, detector, all_readings):
        result = detector.unfold_osem_anlm(
            all_readings,
            calculate_errors=True,
            n_montecarlo=10,
            random_state=7,
            save_result=False,
        )
        assert "spectrum_uncert_mean" in result

    def test_max_neutron_energy(self, detector, all_readings):
        result = detector.unfold_osem_anlm(
            all_readings, max_neutron_energy=1.0, save_result=False
        )
        spectrum = np.asarray(result["spectrum"])
        energy = np.asarray(result["energy"])
        assert np.all(spectrum[energy > 1.0] == 0)
        assert np.all(spectrum[energy <= 1.0] >= 0)

    def test_exported_symbols(self):
        from bssunfold import Detector
        from bssunfold.core import (
            anlm_filter_1d,
            estimate_noise_1d,
            solve_osem_anlm,
            unfold_osem_anlm,
        )

        assert callable(solve_osem_anlm)
        assert callable(unfold_osem_anlm)
        assert callable(anlm_filter_1d)
        assert callable(estimate_noise_1d)
        assert hasattr(Detector, "unfold_osem_anlm")
