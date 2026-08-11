"""Tests for the Compressive Sensing (CS) unfolding method.

Covers the OMP sparse coding, K-SVD dictionary learning, SL0 reconstruction,
the main solve_cs solver, and the unfold_cs Detector wrapper.
"""

import numpy as np
import pandas as pd
import pytest

from bssunfold.core.unfold_cs import (
    solve_omp,
    solve_ksvd,
    solve_sl0,
    solve_cs,
    unfold_cs,
)


@pytest.fixture
def detector():
    """Create a Detector instance with default GSF response functions."""
    from bssunfold import Detector

    return Detector()


@pytest.fixture
def small_detector():
    """Create a small Detector with synthetic data for fast tests."""
    from bssunfold import Detector

    df = pd.DataFrame(
        {
            "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    return Detector(df)


# ---------------------------------------------------------------------------
# OMP
# ---------------------------------------------------------------------------
class TestOMP:
    def test_exact_sparse_recovery(self):
        """OMP should recover an exactly sparse signal."""
        rng = np.random.default_rng(0)
        D = rng.normal(size=(20, 40))
        # Normalize columns
        D = D / np.linalg.norm(D, axis=0)
        alpha_true = np.zeros(40)
        alpha_true[[3, 17, 25]] = [1.0, -0.5, 0.8]
        y = D @ alpha_true
        alpha = solve_omp(D, y, sparsity=3)
        assert np.allclose(D @ alpha, y, atol=1e-6)
        # Support should match
        support = np.where(np.abs(alpha) > 1e-6)[0]
        assert set(support) == {3, 17, 25}

    def test_sparsity_limit(self):
        """OMP should not exceed the requested sparsity."""
        rng = np.random.default_rng(1)
        D = rng.normal(size=(10, 30))
        D = D / np.linalg.norm(D, axis=0)
        y = rng.normal(size=10)
        alpha = solve_omp(D, y, sparsity=4)
        assert np.sum(np.abs(alpha) > 1e-8) <= 4

    def test_zero_signal(self):
        """OMP with a zero signal returns a zero coefficient vector."""
        D = np.eye(5)
        alpha = solve_omp(D, np.zeros(5), sparsity=2)
        assert np.allclose(alpha, 0)

    def test_output_shape(self):
        """OMP output shape matches the number of dictionary atoms."""
        D = np.eye(8)
        alpha = solve_omp(D, np.ones(8), sparsity=3)
        assert alpha.shape == (8,)


# ---------------------------------------------------------------------------
# K-SVD
# ---------------------------------------------------------------------------
class TestKSVD:
    def test_dictionary_shape(self):
        """K-SVD returns a dictionary of the requested shape."""
        rng = np.random.default_rng(2)
        signals = rng.normal(size=(10, 30))
        D = solve_ksvd(signals, n_atoms=12, n_iterations=3, sparsity=3)
        assert D.shape == (10, 12)

    def test_columns_normalized(self):
        """K-SVD dictionary columns should be unit norm."""
        rng = np.random.default_rng(3)
        signals = rng.normal(size=(8, 20))
        D = solve_ksvd(signals, n_atoms=6, n_iterations=2, sparsity=2)
        norms = np.linalg.norm(D, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_reconstruction_improves(self):
        """K-SVD dictionary should represent training signals well."""
        rng = np.random.default_rng(4)
        # Build signals that are sparse in a known basis
        basis = np.eye(10)
        signals = np.zeros((10, 20))
        for j in range(20):
            idx = rng.choice(10, size=2, replace=False)
            signals[:, j] = basis[:, idx[0]] * rng.uniform(0.5, 1.0) + basis[
                :, idx[1]
            ] * rng.uniform(0.5, 1.0)
        D = solve_ksvd(signals, n_atoms=10, n_iterations=10, sparsity=2)
        # Each signal should be reasonably well represented (K-SVD is a
        # heuristic; the reconstruction error should be bounded).
        for j in range(20):
            alpha = solve_omp(D, signals[:, j], sparsity=2)
            assert np.linalg.norm(D @ alpha - signals[:, j]) < 1.0

    def test_reproducible_with_seed(self):
        """K-SVD with the same seed gives the same dictionary."""
        rng = np.random.default_rng(5)
        signals = rng.normal(size=(6, 15))
        D1 = solve_ksvd(signals, n_atoms=4, n_iterations=2, random_state=42)
        D2 = solve_ksvd(signals, n_atoms=4, n_iterations=2, random_state=42)
        assert np.allclose(D1, D2)


# ---------------------------------------------------------------------------
# SL0
# ---------------------------------------------------------------------------
class TestSL0:
    def test_sparse_recovery_underdetermined(self):
        """SL0 should produce a sparser solution than the minimum-norm one."""
        rng = np.random.default_rng(6)
        n = 50
        m = 10
        # Use a sensing matrix with orthonormal rows (better RIP constant),
        # which is favorable for sparse recovery.
        A = rng.normal(size=(m, n))
        Q, _ = np.linalg.qr(A.T)
        A = Q[:, :m].T
        x_true = np.zeros(n)
        x_true[[5, 20, 40]] = [1.0, 0.7, 0.3]
        b = A @ x_true
        x = solve_sl0(A, b, sigma_min=0.001, max_iterations=200)
        # The solution must satisfy the measurement equation
        assert np.linalg.norm(A @ x - b) < 1e-4
        # SL0 should produce a solution sparser than the minimum-norm solution
        x_min = np.linalg.pinv(A) @ b
        nnz_sl0 = np.sum(np.abs(x) > 1e-3)
        nnz_min = np.sum(np.abs(x_min) > 1e-3)
        assert nnz_sl0 < nnz_min
        # The largest coefficients should overlap with the true support
        support = set(np.argsort(np.abs(x))[-3:])
        assert len(support & {5, 20, 40}) >= 2

    def test_output_shape(self):
        """SL0 output shape matches the number of unknowns."""
        A = np.eye(6)
        b = np.ones(6)
        x = solve_sl0(A, b, max_iterations=50)
        assert x.shape == (6,)

    def test_zero_measurement(self):
        """SL0 with zero measurements returns a near-zero solution."""
        A = np.eye(5)
        x = solve_sl0(A, np.zeros(5), max_iterations=50)
        assert np.allclose(x, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# solve_cs
# ---------------------------------------------------------------------------
class TestSolveCS:
    def test_basic_solve(self):
        """solve_cs returns a non-negative spectrum of correct length."""
        rng = np.random.default_rng(7)
        n = 30
        m = 5
        A = rng.normal(size=(m, n))
        x_true = np.zeros(n)
        x_true[[2, 10, 21]] = [1.0, 0.5, 0.8]
        b = A @ x_true
        x, iterations, converged = solve_cs(
            A,
            b,
            n_atoms=40,
            sparsity=3,
            max_iterations=100,
            random_state=0,
        )
        assert x.shape == (n,)
        assert np.all(x >= 0)
        assert isinstance(iterations, int)
        assert isinstance(converged, bool)

    def test_reconstruction_accuracy(self):
        """solve_cs should reproduce the measurements reasonably well."""
        rng = np.random.default_rng(8)
        n = 40
        m = 7
        A = rng.normal(size=(m, n))
        x_true = np.zeros(n)
        x_true[[1, 15, 30]] = [1.0, 0.6, 0.4]
        b = A @ x_true
        x, _, _ = solve_cs(
            A,
            b,
            n_atoms=50,
            sparsity=3,
            max_iterations=150,
            random_state=1,
        )
        # Relative residual should be bounded (CS reconstruction is approximate)
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert residual < 0.8

    def test_with_initial_spectrum(self):
        """solve_cs accepts an initial spectrum guess."""
        rng = np.random.default_rng(9)
        n = 20
        m = 4
        A = rng.normal(size=(m, n))
        b = A @ np.ones(n)
        x0 = np.ones(n)
        x, _, _ = solve_cs(
            A, b, x0=x0, n_atoms=25, max_iterations=50, random_state=2
        )
        assert x.shape == (n,)

    def test_with_prelearned_dictionary(self):
        """solve_cs accepts a pre-learned dictionary."""
        rng = np.random.default_rng(10)
        n = 20
        m = 4
        A = rng.normal(size=(m, n))
        b = A @ np.ones(n)
        D = np.eye(n)
        x, _, _ = solve_cs(
            A, b, dictionary=D, max_iterations=50, random_state=3
        )
        assert x.shape == (n,)

    def test_invalid_dictionary_shape(self):
        """solve_cs raises ValueError for a mismatched dictionary."""
        A = np.eye(5)
        b = np.ones(5)
        with pytest.raises(ValueError):
            solve_cs(A, b, dictionary=np.eye(7), max_iterations=10)


# ---------------------------------------------------------------------------
# unfold_cs wrapper
# ---------------------------------------------------------------------------
class TestUnfoldCS:
    def test_basic_unfolding(self, detector, small_detector):
        """Test basic CS unfolding on a small detector."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = small_detector.unfold_cs(
            readings,
            n_atoms=10,
            sparsity=2,
            max_iterations=50,
            random_state=0,
        )
        assert "spectrum" in result
        assert "energy" in result
        assert result["method"] == "CompressiveSensing"
        assert len(result["spectrum"]) == small_detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert "doserates" in result
        assert "effective_readings" in result

    def test_with_uncertainty(self, small_detector):
        """Test uncertainty calculation via Monte-Carlo."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = small_detector.unfold_cs(
            readings,
            n_atoms=10,
            sparsity=2,
            max_iterations=50,
            calculate_errors=True,
            n_montecarlo=5,
            noise_level=0.05,
            random_state=0,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result

    def test_with_initial_spectrum(self, small_detector):
        """Test CS unfolding with an initial spectrum guess."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        x0 = np.ones(small_detector.n_energy_bins)
        result = small_detector.unfold_cs(
            readings,
            initial_spectrum=x0,
            n_atoms=10,
            sparsity=2,
            max_iterations=50,
            random_state=0,
        )
        assert len(result["spectrum"]) == small_detector.n_energy_bins

    def test_save_result(self, small_detector):
        """Test that save_result stores the result in history."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        small_detector.unfold_cs(
            readings,
            n_atoms=10,
            sparsity=2,
            max_iterations=50,
            save_result=True,
            random_state=0,
        )
        assert len(small_detector.results_history) >= 1

    def test_module_level_unfold_cs(self, small_detector):
        """Test the module-level unfold_cs function directly."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = unfold_cs(
            detector_names=small_detector.detector_names,
            n_energy_bins=small_detector.n_energy_bins,
            E_MeV=small_detector.E_MeV,
            sensitivities=small_detector.sensitivities,
            cc_icrp116=small_detector._get_interpolated_cc(),
            save_result_callback=small_detector._save_result,
            readings=readings,
            n_atoms=10,
            sparsity=2,
            max_iterations=50,
            random_state=0,
        )
        assert "spectrum" in result
        assert result["method"] == "CompressiveSensing"
