"""Tests for the spectral basis abstraction."""

import pytest
import numpy as np


# ============================================================================
# Basis class unit tests
# ============================================================================


class TestBinBasis:
    def test_build_matrix_identity(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        Phi = basis.build_matrix(10, np.logspace(-1, 3, 10))
        assert Phi.shape == (10, 10)
        np.testing.assert_array_equal(Phi, np.eye(10))

    def test_to_coeffs_identity(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        x = np.array([1.0, 2.0, 3.0])
        E = np.array([0.1, 1.0, 10.0])
        c = basis.to_coeffs(x, E)
        np.testing.assert_array_almost_equal(c, x)

    def test_to_spectrum_identity(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        c = np.array([1.0, 2.0, 3.0])
        E = np.array([0.1, 1.0, 10.0])
        x = basis.to_spectrum(c, E)
        np.testing.assert_array_almost_equal(x, c)

    def test_roundtrip(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        x = np.random.rand(20)
        E = np.logspace(-1, 3, 20)
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        np.testing.assert_array_almost_equal(x_rec, x)

    def test_n_coeffs_raises_when_not_set(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        with pytest.raises(ValueError, match="n_coeffs is not set"):
            _ = basis.n_coeffs

    def test_n_coeffs_when_set(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis(n_coeffs=15)
        assert basis.n_coeffs == 15


class TestLegendreBasis:
    def test_build_matrix_shape(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=10)
        E = np.logspace(-1, 3, 50)
        Phi = basis.build_matrix(50, E)
        assert Phi.shape == (50, 10)

    def test_first_column_is_ones(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        E = np.logspace(-1, 3, 30)
        Phi = basis.build_matrix(30, E)
        np.testing.assert_array_almost_equal(Phi[:, 0], 1.0)

    def test_n_coeffs_property(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=12)
        assert basis.n_coeffs == 12

    def test_roundtrip(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=15)
        E = np.logspace(-1, 3, 51)
        # Use a smooth spectrum (exponential decay) for better reconstruction
        x = np.exp(-np.linspace(0, 4, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        # Smooth spectra reconstruct well with 15 polynomials
        assert np.allclose(x, x_rec, atol=0.05)

    def test_energy_axis_mapping(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        # Different energy ranges map to same normalized coords
        # (this is correct — basis adapts to any range)
        E1 = np.logspace(0, 3, 20)
        E2 = np.logspace(-2, 5, 20)
        Phi1 = basis.build_matrix(20, E1)
        Phi2 = basis.build_matrix(20, E2)
        assert Phi1.shape == Phi2.shape
        # Both map to [-1,1] so Phi should be identical (correct behavior)
        np.testing.assert_array_almost_equal(Phi1, Phi2)

    def test_uniform_energy_grid(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        # Uniform energy (all same value) — should fall back to linspace
        E = np.ones(20) * 1.0
        Phi = basis.build_matrix(20, E)
        assert Phi.shape == (20, 5)


class TestFourierBasis:
    def test_build_matrix_shape(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=10)
        E = np.logspace(-1, 3, 50)
        Phi = basis.build_matrix(50, E)
        assert Phi.shape == (50, 10)

    def test_first_column_is_ones(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=5)
        E = np.logspace(-1, 3, 30)
        Phi = basis.build_matrix(30, E)
        np.testing.assert_array_almost_equal(Phi[:, 0], 1.0)

    def test_n_terms_1(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=1)
        E = np.logspace(-1, 3, 20)
        Phi = basis.build_matrix(20, E)
        assert Phi.shape == (20, 1)
        np.testing.assert_array_almost_equal(Phi[:, 0], 1.0)

    def test_n_terms_raises_when_zero(self):
        from bssunfold.core.basis import FourierBasis
        with pytest.raises(ValueError, match="n_terms must be >= 1"):
            FourierBasis(n_terms=0)

    def test_n_coeffs_property(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=8)
        assert basis.n_coeffs == 8

    def test_roundtrip(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=20)
        E = np.logspace(-1, 3, 51)
        # Use a periodic-like spectrum for Fourier (sinusoidal shape)
        x = 0.5 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        # Periodic-like spectra reconstruct well with Fourier
        assert np.allclose(x, x_rec, atol=0.05)

    def test_energy_axis_mapping(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=5)
        # Different energy ranges map to same normalized coords
        E1 = np.logspace(0, 3, 20)
        E2 = np.logspace(-2, 5, 20)
        Phi1 = basis.build_matrix(20, E1)
        Phi2 = basis.build_matrix(20, E2)
        assert Phi1.shape == Phi2.shape
        # Both map to [0,1] so Phi should be identical (correct behavior)
        np.testing.assert_array_almost_equal(Phi1, Phi2)


# ============================================================================
# Basis integration tests with unfolding methods
# ============================================================================


@pytest.fixture
def detector():
    from bssunfold import Detector
    return Detector()


@pytest.fixture
def readings(detector):
    return {name: 100.0 for name in detector.detector_names[:3]}


class TestBasisWithCvxpy:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=10)
        result = detector.unfold_cvxpy(
            readings, basis=basis, regularization=1e-3, save_result=False
        )
        assert 'spectrum' in result
        assert len(result['spectrum']) == detector.n_energy_bins
        assert np.all(result['spectrum'] >= 0)
        assert result.get('basis') == 'LegendreBasis'
        assert result.get('n_coeffs') == 10

    def test_fourier_basis(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=10)
        result = detector.unfold_cvxpy(
            readings, basis=basis, regularization=1e-3, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)
        assert result.get('basis') == 'FourierBasis'

    def test_no_basis_default(self, detector, readings):
        result = detector.unfold_cvxpy(
            readings, regularization=1e-3, save_result=False
        )
        assert 'spectrum' in result
        assert 'basis' not in result


class TestBasisWithLandweber:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_landweber(
            readings, basis=basis, max_iterations=100, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithDoroshenko:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_doroshenko(
            readings, basis=basis, max_iterations=100, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithKaczmarz:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_kaczmarz(
            readings, basis=basis, max_iterations=100, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithQpsolvers:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_qpsolvers(
            readings, basis=basis, regularization=1e-3, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithTsvd:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_tsvd(
            readings, basis=basis, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithScipyDirect:
    def test_legendre_basis(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=8)
        result = detector.unfold_scipy_direct_method(
            readings, basis=basis, save_result=False
        )
        assert 'spectrum' in result
        assert np.all(result['spectrum'] >= 0)


# ============================================================================
# Module exports
# ============================================================================


class TestBasisExports:
    def test_basis_classes_exported(self):
        from bssunfold.core import SpectralBasis, BinBasis, LegendreBasis, FourierBasis
        assert callable(BinBasis)
        assert callable(LegendreBasis)
        assert callable(FourierBasis)

    def test_legendre_exported(self):
        from bssunfold.core import LegendreBasis
        assert callable(LegendreBasis)
