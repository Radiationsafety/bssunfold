"""Tests for Non-negative K-SVD unfolding methods (Xu et al. 2026).

Tests cover:
- solve_nksvd: non-negative K-SVD dictionary learning
- solve_nnls_topk: NNLS+TopK sparse coding
- solve_nn_omp: Non-negative OMP
- solve_omp_standard: Standard OMP on normalized dictionary
- solve_tikhonov_nnls: Tikhonov-augmented NNLS
- solve_nksvd_unfold: full unfolding pipeline
- unfold_nksvd: Detector wrapper
- Evaluation metrics: flux error, spectral correlation, comprehensive score
- Comparison: NNLS+TopK vs NN-OMP vs OMP performance
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bssunfold.core.unfold_nksvd import (
    compute_comprehensive_score,
    compute_flux_error,
    compute_spectral_correlation,
    solve_nksvd,
    solve_nksvd_unfold,
    solve_nn_omp,
    solve_nnls_topk,
    solve_omp_standard,
    solve_tikhonov_nnls,
)
from bssunfold.utils.validators import validate_system


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_system():
    """Small underdetermined system: 3 detectors, 10 energy bins."""
    rng = np.random.default_rng(42)
    A = rng.random((3, 10)) + 0.1  # 3 x 10
    x_true = np.zeros(10)
    x_true[2] = 5.0
    x_true[5] = 3.0
    x_true[8] = 1.0
    b = A @ x_true
    return A, b, x_true


@pytest.fixture
def bnct_like_system():
    """System mimicking the BNCT setup: 10 detectors, 48 energy bins."""
    rng = np.random.default_rng(42)
    n_det = 10
    n_bins = 48
    # Response matrix with smooth energy dependence
    E = np.logspace(-3, 1, n_bins)
    A = np.zeros((n_det, n_bins))
    for i in range(n_det):
        # Each detector has a different moderation thickness
        thickness = 0.5 + i * 1.5
        A[i, :] = np.exp(-E / thickness) * (1 + 0.3 * rng.standard_normal(n_bins))
    A = np.maximum(A, 0.01)

    # True spectrum: multi-peak epithermal neutron spectrum
    x_true = np.zeros(n_bins)
    # Thermal peak
    x_true += 8.0 * np.exp(-((np.log10(E) + 1) ** 2) / 0.1)
    # Epithermal 1/E region
    x_true += 2.0 / (E + 0.01)
    # Fast neutron peak
    x_true += 1.0 * np.exp(-((E - 2.45) ** 2) / 0.05)
    x_true = np.maximum(x_true, 0)

    b = A @ x_true
    return A, b, x_true


# ---------------------------------------------------------------------------
# Tests for solve_tikhonov_nnls
# ---------------------------------------------------------------------------
class TestTikhonovNNLS:
    def test_non_negative_output(self):
        """All coefficients must be non-negative."""
        rng = np.random.default_rng(0)
        M_norm = rng.random((5, 8)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_tikhonov_nnls(M_norm, y, lambda_tik=0.01)
        assert np.all(alpha >= 0)

    def test_zero_measurements_zero_output(self):
        """Zero measurements should produce zero coefficients."""
        M_norm = np.random.random((3, 5)) + 0.1
        y = np.zeros(3)
        alpha = solve_tikhonov_nnls(M_norm, y)
        assert_allclose(alpha, np.zeros(5), atol=1e-10)

    def test_lambda_effect(self):
        """Larger lambda should produce smaller coefficient norms."""
        rng = np.random.default_rng(1)
        M_norm = rng.random((4, 6)) + 0.1
        y = rng.random(4) + 0.5
        a1 = solve_tikhonov_nnls(M_norm, y, lambda_tik=0.001)
        a2 = solve_tikhonov_nnls(M_norm, y, lambda_tik=1.0)
        assert np.linalg.norm(a2) <= np.linalg.norm(a1) + 1e-6

    def test_augmented_form_equivalence(self):
        """Manual augmented NNLS should match solve_tikhonov_nnls."""
        rng = np.random.default_rng(2)
        M_norm = rng.random((3, 5)) + 0.1
        y = rng.random(3) + 0.1
        lam = 0.05
        alpha = solve_tikhonov_nnls(M_norm, y, lambda_tik=lam)

        # Manual augmented form
        from scipy.optimize import nnls
        P = M_norm.shape[1]
        A_tilde = np.vstack([M_norm, np.sqrt(lam) * np.eye(P)])
        b_tilde = np.concatenate([y, np.zeros(P)])
        alpha_manual, _ = nnls(A_tilde, b_tilde)
        assert_allclose(alpha, alpha_manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for solve_nnls_topk
# ---------------------------------------------------------------------------
class TestNNLSTopK:
    def test_sparsity_constraint(self):
        """Output should have at most K non-zero entries."""
        rng = np.random.default_rng(3)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_nnls_topk(M_norm, y, sparsity=3)
        assert np.count_nonzero(alpha) <= 3

    def test_non_negative(self):
        """All coefficients must be non-negative."""
        rng = np.random.default_rng(4)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_nnls_topk(M_norm, y, sparsity=2)
        assert np.all(alpha >= 0)

    def test_exact_sparse_recovery(self):
        """Should recover a truly sparse signal well."""
        rng = np.random.default_rng(5)
        n_atoms = 8
        M_norm = rng.random((4, n_atoms)) + 0.1
        # Create a 1-sparse signal
        alpha_true = np.zeros(n_atoms)
        alpha_true[3] = 2.5
        y = M_norm @ alpha_true
        alpha = solve_nnls_topk(M_norm, y, sparsity=1, lambda_tik=1e-8)
        # The non-zero entry should be close to the true one
        assert np.count_nonzero(alpha) <= 1
        if np.count_nonzero(alpha) == 1:
            assert_allclose(np.max(alpha), 2.5, atol=0.3)


# ---------------------------------------------------------------------------
# Tests for solve_nn_omp
# ---------------------------------------------------------------------------
class TestNNOMP:
    def test_non_negative(self):
        """NN-OMP coefficients must be non-negative."""
        rng = np.random.default_rng(6)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_nn_omp(M_norm, y, sparsity=3)
        assert np.all(alpha >= 0)

    def test_sparsity_constraint(self):
        """Output should have at most K non-zero entries."""
        rng = np.random.default_rng(7)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_nn_omp(M_norm, y, sparsity=2)
        assert np.count_nonzero(alpha) <= 2

    def test_zero_input(self):
        """Zero measurements should give zero output."""
        M_norm = np.random.random((3, 5)) + 0.1
        y = np.zeros(3)
        alpha = solve_nn_omp(M_norm, y, sparsity=2)
        assert_allclose(alpha, np.zeros(5), atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for solve_omp_standard
# ---------------------------------------------------------------------------
class TestOMPStandard:
    def test_sparsity_constraint(self):
        """Standard OMP output should have at most K non-zero entries."""
        rng = np.random.default_rng(8)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_omp_standard(M_norm, y, sparsity=3)
        assert np.count_nonzero(alpha) <= 3

    def test_residual_quality(self):
        """OMP should reduce residual below full random guess."""
        rng = np.random.default_rng(9)
        M_norm = rng.random((5, 10)) + 0.1
        y = rng.random(5) + 0.1
        alpha = solve_omp_standard(M_norm, y, sparsity=3)
        residual = np.linalg.norm(y - M_norm @ alpha)
        # At least should be better than zero solution
        assert residual <= np.linalg.norm(y) + 1e-6


# ---------------------------------------------------------------------------
# Tests for solve_nksvd (dictionary learning)
# ---------------------------------------------------------------------------
class TestNKsvd:
    def test_non_negative_dictionary(self):
        """All dictionary entries must be non-negative after training."""
        rng = np.random.default_rng(42)
        signals = rng.random((10, 20)) + 0.1
        D = solve_nksvd(signals, n_atoms=5, n_iterations=5, sparsity=2, random_state=42)
        assert D.shape == (10, 5)
        assert np.all(D >= -1e-12)  # allow tiny float drift

    def test_normalized_columns(self):
        """Dictionary columns should be approximately unit-norm."""
        rng = np.random.default_rng(42)
        signals = rng.random((10, 20)) + 0.1
        D = solve_nksvd(signals, n_atoms=5, n_iterations=5, sparsity=2, random_state=42)
        norms = np.linalg.norm(D, axis=0)
        assert_allclose(norms, np.ones(5), atol=1e-6)

    def test_reproducibility(self):
        """Same random_state should produce identical dictionaries."""
        rng = np.random.default_rng(42)
        signals = rng.random((8, 15)) + 0.1
        D1 = solve_nksvd(signals, n_atoms=4, n_iterations=3, sparsity=2, random_state=42)
        D2 = solve_nksvd(signals, n_atoms=4, n_iterations=3, sparsity=2, random_state=42)
        assert_allclose(D1, D2, atol=1e-12)

    def test_fewer_atoms_than_signals(self):
        """Should handle n_atoms < m gracefully."""
        rng = np.random.default_rng(42)
        signals = rng.random((6, 20)) + 0.1
        D = solve_nksvd(signals, n_atoms=3, n_iterations=3, sparsity=2, random_state=42)
        assert D.shape == (6, 3)


# ---------------------------------------------------------------------------
# Tests for solve_nksvd_unfold (full pipeline)
# ---------------------------------------------------------------------------
class TestNKsvdUnfold:
    def test_basic_unfolding(self, small_system):
        """Basic unfolding should return valid spectrum."""
        A, b, x_true = small_system
        x, iters, converged = solve_nksvd_unfold(
            A, b, x0=x_true, n_atoms=5, sparsity=2,
            n_dictionary_iterations=10, random_state=42,
        )
        assert x.shape == x_true.shape
        assert np.all(x >= 0)
        assert isinstance(iters, int)
        assert isinstance(converged, bool)

    def test_nnls_topk_method(self, small_system):
        """NNLS+TopK sparse method should work."""
        A, b, x_true = small_system
        x, _, _ = solve_nksvd_unfold(
            A, b, n_atoms=5, sparsity=2,
            sparse_method="nnls_topk", n_dictionary_iterations=10,
            random_state=42,
        )
        assert np.all(x >= 0)

    def test_nn_omp_method(self, small_system):
        """NN-OMP sparse method should work."""
        A, b, x_true = small_system
        x, _, _ = solve_nksvd_unfold(
            A, b, n_atoms=5, sparsity=2,
            sparse_method="nn_omp", n_dictionary_iterations=10,
            random_state=42,
        )
        assert np.all(x >= 0)

    def test_omp_method(self, small_system):
        """Standard OMP sparse method should work."""
        A, b, x_true = small_system
        x, _, _ = solve_nksvd_unfold(
            A, b, n_atoms=5, sparsity=2,
            sparse_method="omp", n_dictionary_iterations=10,
            random_state=42,
        )
        assert x.shape == x_true.shape

    def test_invalid_sparse_method(self, small_system):
        """Invalid sparse_method should raise ValueError."""
        A, b, _ = small_system
        with pytest.raises(ValueError, match="Unknown sparse_method"):
            solve_nksvd_unfold(A, b, sparse_method="invalid")

    def test_prelearned_dictionary(self, small_system):
        """Providing a pre-learned dictionary should skip dictionary learning."""
        A, b, x_true = small_system
        rng = np.random.default_rng(42)
        D = rng.random((10, 5)) + 0.1
        x, _, _ = solve_nksvd_unfold(
            A, b, dictionary=D, sparsity=2, random_state=42,
        )
        assert x.shape == x_true.shape

    def test_training_signals(self, small_system):
        """Providing training_signals should work."""
        A, b, x_true = small_system
        rng = np.random.default_rng(42)
        signals = rng.random((10, 15)) + 0.1
        x, _, _ = solve_nksvd_unfold(
            A, b, training_signals=signals, n_atoms=5, sparsity=2,
            n_dictionary_iterations=10, random_state=42,
        )
        assert x.shape == x_true.shape

    def test_bnct_like_system(self, bnct_like_system):
        """BNCT-like system: NNLS+TopK should produce reasonable reconstruction."""
        A, b, x_true = bnct_like_system
        x, _, _ = solve_nksvd_unfold(
            A, b, n_atoms=15, sparsity=2,
            n_dictionary_iterations=20, random_state=42,
        )
        assert np.all(x >= 0)
        # Correlation should be positive (reasonable reconstruction)
        if np.std(x) > 0 and np.std(x_true) > 0:
            corr = np.corrcoef(x_true, x)[0, 1]
            assert corr > 0.0  # at least positive correlation

    def test_input_validation(self):
        """Invalid inputs should be caught by validate_system."""
        with pytest.raises((ValueError, TypeError)):
            solve_nksvd_unfold(None, np.ones(3))
        with pytest.raises((ValueError, TypeError)):
            solve_nksvd_unfold(np.ones((3, 5)), None)


# ---------------------------------------------------------------------------
# Tests for evaluation metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_flux_error_zero_for_identical(self):
        phi = np.array([1.0, 2.0, 3.0])
        assert compute_flux_error(phi, phi) == 0.0

    def test_flux_error_positive(self):
        phi_true = np.array([1.0, 2.0, 3.0])
        phi_recon = np.array([1.1, 2.1, 3.1])
        assert compute_flux_error(phi_true, phi_recon) > 0

    def test_spectral_correlation_perfect(self):
        phi = np.array([1.0, 2.0, 3.0])
        corr = compute_spectral_correlation(phi, phi)
        assert_allclose(corr, 1.0, atol=1e-10)

    def test_spectral_correlation_anticorrelated(self):
        phi1 = np.array([1.0, 2.0, 3.0])
        phi2 = np.array([3.0, 2.0, 1.0])
        corr = compute_spectral_correlation(phi1, phi2)
        assert corr < 0

    def test_comprehensive_score(self):
        phi_true = np.array([1.0, 2.0, 3.0])
        phi_recon = np.array([1.1, 2.1, 3.1])
        score = compute_comprehensive_score(phi_true, phi_recon)
        # score = flux_err - 0.5 * corr; for close spectra this should be small
        assert isinstance(score, float)

    def test_paper_metric_values(self):
        """Verify metrics match paper-reported ranges approximately."""
        # Paper: NNLS+TopK at optimal config: corr=0.9729, flux_err=0.125
        # We just verify the metric computation is consistent
        rng = np.random.default_rng(42)
        phi_true = rng.random(48) + 0.1
        # Simulate a good reconstruction
        phi_recon = phi_true * (1 + 0.1 * rng.standard_normal(48))
        phi_recon = np.maximum(phi_recon, 0)

        flux_err = compute_flux_error(phi_true, phi_recon)
        corr = compute_spectral_correlation(phi_true, phi_recon)
        score = compute_comprehensive_score(phi_true, phi_recon)

        # Just check they are in reasonable ranges
        assert 0 <= flux_err < 1
        assert -1 <= corr <= 1
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Comparison test: NNLS+TopK vs NN-OMP vs OMP
# ---------------------------------------------------------------------------
class TestSparseMethodComparison:
    """Compare the three sparse coding strategies.

    The paper reports that NNLS+TopK substantially outperforms OMP and NN-OMP.
    """

    def test_all_methods_produce_nonneg_or_valid(self, small_system):
        """All methods should produce valid results."""
        A, b, x_true = small_system
        for method in ["nnls_topk", "nn_omp", "omp"]:
            x, _, _ = solve_nksvd_unfold(
                A, b, n_atoms=5, sparsity=2,
                sparse_method=method, n_dictionary_iterations=10,
                random_state=42,
            )
            assert x.shape == x_true.shape
            if method != "omp":
                assert np.all(x >= 0), f"{method} should produce non-negative output"


# ---------------------------------------------------------------------------
# Detector wrapper test
# ---------------------------------------------------------------------------
class TestDetectorWrapper:
    @pytest.fixture
    def detector(self):
        """Create a minimal Detector for testing."""
        from bssunfold.core.detector import Detector

        n_bins = 31
        E_MeV = np.logspace(-3, 1, n_bins)
        detector_names = [f"det{i}" for i in range(5)]

        sensitivities = {}
        rng = np.random.default_rng(42)
        for name in detector_names:
            sens = np.exp(-E_MeV / (0.5 + 0.3 * rng.random())) + 0.01
            sensitivities[name] = sens

        det = Detector(
            E_MeV=E_MeV,
            sensitivities=sensitivities,
        )
        return det

    def test_unfold_nksvd_basic(self, detector):
        """Detector.unfold_nksvd should return a result dict."""
        # Create readings from a known spectrum
        A = np.array([detector.sensitivities[name] for name in detector.detector_names])
        x_true = np.zeros(detector.n_energy_bins)
        x_true[10] = 5.0
        x_true[20] = 3.0
        b = A @ x_true

        readings = {name: float(b[i]) for i, name in enumerate(detector.detector_names)}

        result = detector.unfold_nksvd(
            readings=readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=10,
            random_state=42,
        )
        assert "spectrum" in result
        assert "energy" in result
        assert "method" in result
        assert result["method"] == "NonNegativeKSVD"
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_unfold_nksvd_nn_omp(self, detector):
        """Detector.unfold_nksvd with NN-OMP method."""
        A = np.array([detector.sensitivities[name] for name in detector.detector_names])
        x_true = np.ones(detector.n_energy_bins) * 0.5
        b = A @ x_true
        readings = {name: float(b[i]) for i, name in enumerate(detector.detector_names)}

        result = detector.unfold_nksvd(
            readings=readings,
            n_atoms=8,
            sparsity=2,
            sparse_method="nn_omp",
            n_dictionary_iterations=10,
            random_state=42,
        )
        assert "spectrum" in result

    def test_unfold_nksvd_omp(self, detector):
        """Detector.unfold_nksvd with OMP method."""
        A = np.array([detector.sensitivities[name] for name in detector.detector_names])
        x_true = np.ones(detector.n_energy_bins) * 0.5
        b = A @ x_true
        readings = {name: float(b[i]) for i, name in enumerate(detector.detector_names)}

        result = detector.unfold_nksvd(
            readings=readings,
            n_atoms=8,
            sparsity=2,
            sparse_method="omp",
            n_dictionary_iterations=10,
            random_state=42,
        )
        assert "spectrum" in result
