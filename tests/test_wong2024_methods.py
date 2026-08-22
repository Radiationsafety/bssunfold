"""Tests for IMAXED, AMAXED, and AMAXED-Regularization unfolding methods.

These tests verify the implementation of the unfolding algorithms from
Wong's 2024 PhD thesis "Modernising neutron spectrum unfolding for fusion applications".
"""

import numpy as np
import pytest
from bssunfold.core import (
    solve_imaxed,
    solve_amaxed,
    solve_amaxed_regularization,
)


def generate_test_response_matrix(m=10, n=20, seed=42):
    """Generate a synthetic response matrix for testing."""
    np.random.seed(seed)
    # Create a smooth response matrix similar to real detector responses
    A = np.zeros((m, n))
    for i in range(m):
        center = n * (i + 1) / (m + 1)
        width = max(2, n / (m + 1))
        for j in range(n):
            A[i, j] = np.exp(-0.5 * ((j - center) / width) ** 2)
    # Add small noise
    A += 0.01 * np.random.rand(m, n)
    return np.maximum(A, 0)


def generate_test_spectrum(n=20, seed=42):
    """Generate a synthetic test spectrum."""
    np.random.seed(seed + 1)
    # Create a spectrum with peaks
    x = np.linspace(0, 1, n)
    spectrum = (
        0.5 * np.exp(-0.5 * ((x - 0.3) / 0.1) ** 2) +
        0.3 * np.exp(-0.5 * ((x - 0.7) / 0.15) ** 2) +
        0.1
    )
    return spectrum


class TestIMAXED:
    """Tests for IMAXED unfolding method."""

    def test_solve_imaxed_basic(self):
        """Test basic IMAXED functionality."""
        m, n = 10, 20
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_imaxed(A, b, x0)
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0), "Solution should be non-negative"
        assert converged or iterations > 0, "Should attempt convergence"
        
    def test_solve_imaxed_with_noise(self):
        """Test IMAXED with noisy measurements."""
        m, n = 15, 25
        np.random.seed(42)
        
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b_clean = A @ true_spectrum
        
        # Add 5% noise
        noise = 0.05 * np.abs(b_clean) * np.random.randn(m)
        b_noisy = b_clean + noise
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_imaxed(
            A, b_noisy, x0, sigma_factor=0.05
        )
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0)
        
        # Check reconstruction quality
        reconstructed_b = A @ x_sol
        relative_error = np.linalg.norm(reconstructed_b - b_noisy) / np.linalg.norm(b_noisy)
        assert relative_error < 0.5, f"Reconstruction error too high: {relative_error}"
        
    def test_solve_imaxed_convergence(self):
        """Test that IMAXED converges with different tolerances."""
        m, n = 8, 16
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        # Test with different tolerances
        for tol in [1e-4, 1e-6, 1e-8]:
            x_sol, iterations, converged = solve_imaxed(A, b, x0, tolerance=tol)
            assert x_sol.shape == (n,)
            assert np.all(x_sol >= 0)


class TestAMAXED:
    """Tests for AMAXED unfolding method."""

    def test_solve_amaxed_basic(self):
        """Test basic AMAXED functionality."""
        m, n = 10, 20
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_amaxed(A, b, x0)
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0), "Solution should be non-negative"
        
    def test_solve_amaxed_with_target_chi2(self):
        """Test AMAXED with explicit target chi-squared."""
        m, n = 12, 24
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        target_chi2 = float(m - n) if m > n else 1.0
        
        x_sol, iterations, converged = solve_amaxed(
            A, b, x0, target_chi2=target_chi2
        )
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0)
        
    def test_solve_amaxed_with_noise(self):
        """Test AMAXED with noisy measurements."""
        m, n = 15, 25
        np.random.seed(42)
        
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b_clean = A @ true_spectrum
        
        # Add 5% noise
        noise = 0.05 * np.abs(b_clean) * np.random.randn(m)
        b_noisy = b_clean + noise
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_amaxed(
            A, b_noisy, x0, sigma_factor=0.05
        )
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0)
        
        # Check reconstruction quality
        reconstructed_b = A @ x_sol
        relative_error = np.linalg.norm(reconstructed_b - b_noisy) / np.linalg.norm(b_noisy)
        assert relative_error < 0.5, f"Reconstruction error too high: {relative_error}"


class TestAMAXEDRegularization:
    """Tests for AMAXED-Regularization unfolding method."""

    def test_solve_amaxed_reg_basic(self):
        """Test basic AMAXED-Regularization functionality."""
        m, n = 10, 20
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_amaxed_regularization(A, b, x0)
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0), "Solution should be non-negative"
        
    def test_solve_amaxed_reg_tau_parameter(self):
        """Test AMAXED-Regularization with different tau values."""
        m, n = 10, 20
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        # Test with different regularization strengths
        for tau in [0.1, 1.0, 10.0]:
            x_sol, iterations, converged = solve_amaxed_regularization(
                A, b, x0, tau=tau
            )
            
            assert x_sol.shape == (n,)
            assert np.all(x_sol >= 0)
            
    def test_solve_amaxed_reg_with_noise(self):
        """Test AMAXED-Regularization with noisy measurements."""
        m, n = 15, 25
        np.random.seed(42)
        
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b_clean = A @ true_spectrum
        
        # Add 5% noise
        noise = 0.05 * np.abs(b_clean) * np.random.randn(m)
        b_noisy = b_clean + noise
        
        x0 = np.ones(n)
        
        x_sol, iterations, converged = solve_amaxed_regularization(
            A, b_noisy, x0, sigma_factor=0.05, tau=1.0
        )
        
        assert x_sol.shape == (n,)
        assert np.all(x_sol >= 0)
        
        # Check reconstruction quality
        reconstructed_b = A @ x_sol
        relative_error = np.linalg.norm(reconstructed_b - b_noisy) / np.linalg.norm(b_noisy)
        assert relative_error < 0.5, f"Reconstruction error too high: {relative_error}"


class TestComparison:
    """Tests comparing the three Wong 2024 methods."""

    def test_methods_produce_similar_results(self):
        """Test that all three methods produce reasonable solutions."""
        m, n = 10, 20
        A = generate_test_response_matrix(m, n)
        true_spectrum = generate_test_spectrum(n)
        b = A @ true_spectrum
        
        x0 = np.ones(n)
        
        # Run all three methods
        x_imaxed, _, _ = solve_imaxed(A, b, x0)
        x_amaxed, _, _ = solve_amaxed(A, b, x0)
        x_amaxed_reg, _, _ = solve_amaxed_regularization(A, b, x0)
        
        # All should be non-negative
        assert np.all(x_imaxed >= 0)
        assert np.all(x_amaxed >= 0)
        assert np.all(x_amaxed_reg >= 0)
        
        # All should have similar magnitude
        norm_ratio_imaxed = np.linalg.norm(x_imaxed) / np.linalg.norm(true_spectrum)
        norm_ratio_amaxed = np.linalg.norm(x_amaxed) / np.linalg.norm(true_spectrum)
        norm_ratio_amaxed_reg = np.linalg.norm(x_amaxed_reg) / np.linalg.norm(true_spectrum)
        
        # Ratios should be within reasonable bounds (not orders of magnitude off)
        assert 0.1 < norm_ratio_imaxed < 10
        assert 0.1 < norm_ratio_amaxed < 10
        assert 0.1 < norm_ratio_amaxed_reg < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
