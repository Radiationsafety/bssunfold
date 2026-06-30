"""Tests for unfold_reconst — Turchin's statistical regularization (STREG1).

Covers the low-level helpers, the solve_reconst solver (all 4 parameter
modes + edge cases), the Detector.unfold_reconst wrapper, and exports.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ============================================================================
# Low-level helpers
# ============================================================================

class TestBuildOmoMatrix:
    def test_shape_and_values(self):
        from bssunfold.core.unfold_reconst import _build_omo_matrix
        OMO = _build_omo_matrix(10, 1e-3)
        assert OMO.shape == (5, 10)
        assert np.all(np.isfinite(OMO))

    def test_n3(self):
        from bssunfold.core.unfold_reconst import _build_omo_matrix
        OMO = _build_omo_matrix(3, 0.0)
        assert OMO.shape == (5, 3)

    def test_pp_effect(self):
        from bssunfold.core.unfold_reconst import _build_omo_matrix
        OMO_no = _build_omo_matrix(10, 0.0)
        OMO_pp = _build_omo_matrix(10, 1.0)
        expected = OMO_no[2, :] + 1.0  # PP * (XX[i+1] - XX[i]) = 1.0
        assert_allclose(OMO_pp[2, :], expected)


class TestInvertMatrix:
    def test_invert_2x2(self):
        from bssunfold.core.unfold_reconst import _invert_matrix
        D = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float)
        D_orig = D.copy()
        _invert_matrix(D)
        expected = np.linalg.inv(D_orig)
        assert_allclose(D, expected, atol=1e-12)

    def test_invert_5x5(self):
        from bssunfold.core.unfold_reconst import _invert_matrix
        np.random.seed(42)
        D = np.random.randn(5, 5)
        D = D @ D.T + np.eye(5) * 2.0
        D_orig = D.copy()
        _invert_matrix(D)
        assert_allclose(D @ D_orig, np.eye(5), atol=1e-10)

    def test_singular_raises(self):
        from bssunfold.core.unfold_reconst import _invert_matrix
        D = np.zeros((3, 3))
        with pytest.raises(np.linalg.LinAlgError, match="D\\(1,1\\)=0"):
            _invert_matrix(D)


class TestBuildSystemMatrix:
    def test_basic_structure(self):
        from bssunfold.core.unfold_reconst import _build_system_matrix
        n = 5
        B = np.eye(n)
        OMO = np.ones((5, n)) * 0.01
        D = _build_system_matrix(B, OMO, n, alpha=0.1, beta=2.0)
        assert D.shape == (n, n)
        # D[i,i] = 2.0 + OMO[2,i]*0.1 = 2.0 + 0.001 = 2.001
        # D[i,i+1] = OMO[1,i+1]*0.1 = 0.001
        # D[i,i+2] = OMO[0,i+2]*0.1 = 0.001
        assert_allclose(np.diag(D), 2.001, atol=1e-12)
        assert_allclose(np.diag(D, 1), 0.001, atol=1e-12)
        assert_allclose(np.diag(D, 2), 0.001, atol=1e-12)

    def test_symmetric(self):
        from bssunfold.core.unfold_reconst import _build_system_matrix, _build_omo_matrix
        n = 8
        np.random.seed(7)
        B = np.random.randn(n, n)
        B = B @ B.T + np.eye(n)
        OMO = _build_omo_matrix(n, 0.5)
        D = _build_system_matrix(B, OMO, n, alpha=0.3, beta=1.5)
        assert_allclose(D, D.T, atol=1e-12)


class TestReg1:
    def test_ich_gt0_returns_inverse(self):
        from bssunfold.core.unfold_reconst import _reg1, _build_system_matrix
        n = 5
        np.random.seed(1)
        B = np.eye(n)
        OMO = np.ones((5, n)) * 0.01
        A_vec = np.random.randn(n)
        D_inv, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha=0.1, beta=1.0, ich=2)
        assert FI is not None
        assert SIGMA is not None
        D_check = _build_system_matrix(B, OMO, n, alpha=0.1, beta=1.0)
        assert_allclose(D_inv @ D_check, np.eye(n), atol=1e-10)

    def test_ich_eq0_returns_none(self):
        from bssunfold.core.unfold_reconst import _reg1
        n = 5
        B = np.eye(n)
        OMO = np.zeros((5, n))
        A_vec = np.ones(n)
        D, FI, SIGMA = _reg1(B, OMO, A_vec, n, alpha=0.1, beta=1.0, ich=0)
        assert FI is None
        assert SIGMA is None
        assert D.shape == (n, n)

    def test_matches_numpy_solve(self):
        from bssunfold.core.unfold_reconst import _reg1, _build_system_matrix
        n = 4
        np.random.seed(99)
        B = np.random.randn(n, n)
        B = B @ B.T + np.eye(n) * 0.5
        OMO = np.random.randn(5, n) * 0.1
        A_vec = np.random.randn(n)
        alpha, beta = 0.2, 0.8
        D_inv, FI, _ = _reg1(B, OMO, A_vec, n, alpha, beta, ich=2)
        D_sys = _build_system_matrix(B, OMO, n, alpha, beta)
        expected = np.linalg.solve(D_sys, A_vec * beta)
        assert_allclose(FI, expected, atol=1e-10)


# ============================================================================
# solve_reconst — all 4 parameter modes
# ============================================================================

class TestSolveReconstModes:
    """Test the 4 alpha/beta selection modes."""

    @pytest.fixture
    def problem(self):
        np.random.seed(42)
        M, N = 6, 10
        A = np.random.rand(M, N) * 0.1
        true_spec = np.exp(-np.linspace(0, 3, N)) * 10
        b = A @ true_spec + np.random.randn(M) * 0.05
        sigma = np.full(M, 0.05)
        return A, b, sigma, N

    def test_mode_fixed_alpha_fixed_beta(self, problem):
        """α > 0 fixed, β > 0 fixed."""
        from bssunfold.core.unfold_reconst import solve_reconst
        A, b, sigma, N = problem
        spec = solve_reconst(A, b, alpha=0.1, beta=1.0, sigma_b=sigma)
        assert spec.shape == (N,)
        assert np.all(spec >= 0)
        assert np.all(np.isfinite(spec))

    def test_mode_auto_alpha_fixed_beta(self, problem):
        """α < 0 auto, β > 0 fixed."""
        from bssunfold.core.unfold_reconst import solve_reconst
        A, b, sigma, N = problem
        spec = solve_reconst(A, b, alpha=-1.0, beta=1.0, sigma_b=sigma)
        assert spec.shape == (N,)
        assert np.all(spec >= 0)

    def test_mode_fixed_alpha_auto_beta(self, problem):
        """α >= 0 fixed, β == 0 auto."""
        from bssunfold.core.unfold_reconst import solve_reconst
        A, b, sigma, N = problem
        spec = solve_reconst(A, b, alpha=0.1, beta=0.0, sigma_b=sigma)
        assert spec.shape == (N,)
        assert np.all(spec >= 0)

    def test_mode_auto_both(self, problem):
        """α < 0 auto, β == 0 auto."""
        from bssunfold.core.unfold_reconst import solve_reconst
        A, b, sigma, N = problem
        spec = solve_reconst(A, b, alpha=-1.0, beta=0.0, sigma_b=sigma)
        assert spec.shape == (N,)
        assert np.all(spec >= 0)


class TestSolveReconstEdgeCases:
    def test_default_sigma(self):
        """sigma_b=None → sqrt(b) used."""
        from bssunfold.core.unfold_reconst import solve_reconst
        np.random.seed(0)
        A = np.random.rand(4, 8) * 0.1
        b = np.ones(4) * 100.0
        spec = solve_reconst(A, b, alpha=0.01, beta=1.0)
        assert np.all(spec >= 0)
        assert np.all(np.isfinite(spec))

    def test_small_problem(self):
        """M < N (underdetermined) — the typical case."""
        from bssunfold.core.unfold_reconst import solve_reconst
        np.random.seed(1)
        A = np.random.rand(3, 9) * 0.05
        b = np.ones(3) * 50.0
        spec = solve_reconst(A, b, alpha=0.1, beta=1.0)
        assert spec.shape == (9,)
        assert np.all(spec >= 0)

    def test_zero_readings(self):
        """All zero readings produce zero spectrum."""
        from bssunfold.core.unfold_reconst import solve_reconst
        A = np.random.rand(4, 6) * 0.1
        b = np.zeros(4)
        spec = solve_reconst(A, b, alpha=0.1, beta=1.0)
        assert np.all(spec >= 0)
        residual = np.linalg.norm(A @ spec - b)
        assert residual < 1e-10

    def test_negative_readings(self):
        """Negative readings should be handled gracefully."""
        from bssunfold.core.unfold_reconst import solve_reconst
        np.random.seed(2)
        A = np.random.rand(4, 6) * 0.1
        b = np.array([-10.0, 100.0, 50.0, -5.0])
        spec = solve_reconst(A, b, alpha=0.1, beta=1.0)
        assert np.all(np.isfinite(spec))

    def test_noisy_vs_clean(self):
        """Clean data should give lower residual."""
        from bssunfold.core.unfold_reconst import solve_reconst
        np.random.seed(123)
        M, N = 5, 8
        A = np.random.rand(M, N) * 0.1
        true_spec = np.linspace(1, 3, N)
        b_clean = A @ true_spec
        b_noisy = b_clean + np.random.randn(M) * 0.2
        sigma = np.full(M, 0.2)
        s_clean = solve_reconst(A, b_clean, alpha=0.5, beta=10.0, sigma_b=sigma)
        s_noisy = solve_reconst(A, b_noisy, alpha=0.5, beta=10.0, sigma_b=sigma)
        r_clean = np.linalg.norm(A @ s_clean - b_clean)
        r_noisy = np.linalg.norm(A @ s_noisy - b_noisy)
        assert r_clean < r_noisy + 0.1


# ============================================================================
# Detector.unfold_reconst wrapper
# ============================================================================

@pytest.fixture
def detector():
    from bssunfold import Detector
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


class TestUnfoldReconstBasic:
    def test_basic_unfolding(self, detector, readings):
        """Standard unfold with auto alpha/beta."""
        result = detector.unfold_reconst(readings)
        assert 'spectrum' in result
        assert 'energy' in result
        assert 'doserates' in result
        assert result['method'] == 'Reconst'
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)

    def test_fixed_params(self, detector, readings):
        """Fixed alpha/beta."""
        result = detector.unfold_reconst(readings, alpha=0.1, beta=1.0)
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_with_pp(self, detector, readings):
        """Different PP parameter."""
        result = detector.unfold_reconst(readings, pp=1.0)
        assert 'spectrum' in result

    def test_save_result(self, detector, readings):
        """Save to history."""
        detector.unfold_reconst(readings, save_result=True)
        assert len(detector.results_history) >= 1

    def test_no_save(self, detector, readings):
        """Do not save to history."""
        detector.results_history.clear()
        detector.unfold_reconst(readings, save_result=False)
        assert len(detector.results_history) == 0


class TestUnfoldReconstWithErrors:
    @pytest.mark.slow
    def test_monte_carlo_errors(self, detector, readings):
        """Monte-Carlo uncertainty estimation."""
        result = detector.unfold_reconst(
            readings,
            alpha=0.1, beta=1.0,
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result

    @pytest.mark.slow
    def test_noise_level(self, detector, readings):
        """Different noise level for MC."""
        result = detector.unfold_reconst(
            readings,
            alpha=0.1, beta=1.0,
            calculate_errors=True,
            n_montecarlo=3,
            noise_level=0.05,
            random_state=7,
        )
        assert 'spectrum_uncert_std' in result

    def test_with_initial_spectrum(self, detector, readings):
        """initial_spectrum argument is accepted (ignored)."""
        initial = np.ones(detector.n_energy_bins)
        result = detector.unfold_reconst(
            readings, initial_spectrum=initial, alpha=0.1, beta=1.0
        )
        assert 'spectrum' in result


class TestUnfoldReconstMultipleReadings:
    def test_two_detectors(self, detector):
        readings = {
            detector.detector_names[0]: 100.0,
            detector.detector_names[1]: 50.0,
        }
        result = detector.unfold_reconst(readings, alpha=0.1, beta=1.0)
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)

    def test_all_detectors(self, detector):
        readings = {name: 100.0 for name in detector.detector_names[:5]}
        result = detector.unfold_reconst(readings, alpha=0.1, beta=1.0)
        assert 'spectrum' in result


# ============================================================================
# Module exports
# ============================================================================

class TestExports:
    def test_solve_reconst_exported(self):
        from bssunfold.core import solve_reconst
        assert callable(solve_reconst)

    def test_unfold_reconst_exported(self):
        from bssunfold import Detector
        assert hasattr(Detector, 'unfold_reconst')

    def test_unfold_reconst_core_exported(self):
        from bssunfold.core import unfold_reconst
        assert callable(unfold_reconst)

    def test_detector_method_runs(self, detector, readings):
        """End-to-end: Detector.unfold_reconst returns properly structured dict."""
        result = detector.unfold_reconst(readings, alpha=0.1, beta=1.0)
        for key in ('spectrum', 'energy', 'doserates',
                     'effective_readings', 'residual', 'residual_norm', 'method'):
            assert key in result, f'Missing key: {key}'
        assert result['method'] == 'Reconst'
        assert isinstance(result['residual_norm'], float)
        assert isinstance(result['effective_readings'], dict)


# ============================================================================
# _compute_omega and _compute_delta (regression)
# ============================================================================

class TestComputeOmega:
    def test_returns_finite(self):
        from bssunfold.core.unfold_reconst import _compute_omega, _build_omo_matrix
        n = 6
        np.random.seed(5)
        OMO = _build_omo_matrix(n, 0.01)
        D_inv = np.eye(n) * 2.0
        FI = np.random.randn(n)
        omega = _compute_omega(OMO, D_inv, FI, n, alpha=0.5)
        assert np.isfinite(omega)
        assert isinstance(omega, float)


class TestComputeDelta:
    def test_returns_finite(self):
        from bssunfold.core.unfold_reconst import _compute_delta
        n, m = 6, 4
        np.random.seed(6)
        B = np.random.randn(n, n)
        B = B @ B.T + np.eye(n)
        D_inv = np.eye(n) * 0.5
        FI = np.random.randn(n)
        A_vec = np.random.randn(n)
        F = np.random.randn(m)
        S = np.ones(m)
        delta = _compute_delta(B, D_inv, FI, A_vec, F, S, n, m, beta=0.5)
        assert np.isfinite(delta)
        assert isinstance(delta, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
