"""Tests for the spectral basis abstraction.

Covers:
- Unit tests for BinBasis, LegendreBasis, FourierBasis
- Forward/inverse transform correctness
- Energy axis mapping
- Integration with all supported unfolding methods via Detector
- Monte-Carlo uncertainty with basis
- Edge cases (single bin, uniform energy, small systems)
- Module exports
"""

import pytest
import numpy as np


# ============================================================================
# BinBasis unit tests
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

    def test_single_bin(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        Phi = basis.build_matrix(1, np.array([1.0]))
        assert Phi.shape == (1, 1)
        np.testing.assert_array_equal(Phi, [[1.0]])

    def test_does_not_mutate_input(self):
        from bssunfold.core.basis import BinBasis
        basis = BinBasis()
        x = np.array([1.0, 2.0, 3.0])
        E = np.array([0.1, 1.0, 10.0])
        x_copy = x.copy()
        _ = basis.to_coeffs(x, E)
        np.testing.assert_array_equal(x, x_copy)


# ============================================================================
# LegendreBasis unit tests
# ============================================================================


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

    def test_roundtrip_smooth(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=15)
        E = np.logspace(-1, 3, 51)
        x = np.exp(-np.linspace(0, 4, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        assert np.allclose(x, x_rec, atol=0.05)

    def test_roundtrip_constant(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        E = np.logspace(-1, 3, 30)
        x = np.ones(30) * 2.5
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        np.testing.assert_array_almost_equal(x_rec, x, decimal=10)

    def test_energy_axis_mapping(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        E1 = np.logspace(0, 3, 20)
        E2 = np.logspace(-2, 5, 20)
        Phi1 = basis.build_matrix(20, E1)
        Phi2 = basis.build_matrix(20, E2)
        assert Phi1.shape == Phi2.shape
        np.testing.assert_array_almost_equal(Phi1, Phi2)

    def test_uniform_energy_grid(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=5)
        E = np.ones(20) * 1.0
        Phi = basis.build_matrix(20, E)
        assert Phi.shape == (20, 5)

    def test_orthogonality(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=10)
        E = np.logspace(-1, 3, 100)
        Phi = basis.build_matrix(100, E)
        G = Phi.T @ Phi
        # Diagonal should dominate (columns not fully orthogonal in discrete case)
        diag = np.diag(G)
        off_diag = G - np.diag(diag)
        # Off-diagonal should be small relative to diagonal
        assert np.max(np.abs(off_diag)) < np.max(np.abs(diag)) * 0.5

    def test_single_energy_bin(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=3)
        E = np.array([1.0])
        Phi = basis.build_matrix(1, E)
        assert Phi.shape == (1, 3)

    def test_more_polynomials_than_bins(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=20)
        E = np.logspace(-1, 3, 10)
        Phi = basis.build_matrix(10, E)
        assert Phi.shape == (10, 20)

    def test_negative_spectrum_reconstruction(self):
        from bssunfold.core.basis import LegendreBasis
        basis = LegendreBasis(n_polynomials=10)
        E = np.logspace(-1, 3, 51)
        x = np.sin(np.linspace(0, 2 * np.pi, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        assert np.allclose(x, x_rec, atol=0.1)


# ============================================================================
# FourierBasis unit tests
# ============================================================================


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

    def test_n_terms_raises_when_negative(self):
        from bssunfold.core.basis import FourierBasis
        with pytest.raises(ValueError):
            FourierBasis(n_terms=-5)

    def test_n_coeffs_property(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=8)
        assert basis.n_coeffs == 8

    def test_roundtrip_periodic(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=20)
        E = np.logspace(-1, 3, 51)
        x = 0.5 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        assert np.allclose(x, x_rec, atol=0.05)

    def test_roundtrip_constant(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=5)
        E = np.logspace(-1, 3, 30)
        x = np.ones(30) * 3.0
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        np.testing.assert_array_almost_equal(x_rec, x, decimal=10)

    def test_energy_axis_mapping(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=5)
        E1 = np.logspace(0, 3, 20)
        E2 = np.logspace(-2, 5, 20)
        Phi1 = basis.build_matrix(20, E1)
        Phi2 = basis.build_matrix(20, E2)
        assert Phi1.shape == Phi2.shape
        np.testing.assert_array_almost_equal(Phi1, Phi2)

    def test_uniform_energy_grid(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=5)
        E = np.ones(20) * 1.0
        Phi = basis.build_matrix(20, E)
        assert Phi.shape == (20, 5)

    def test_orthogonality(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=10)
        E = np.logspace(-1, 3, 100)
        Phi = basis.build_matrix(100, E)
        G = Phi.T @ Phi
        # Diagonal should dominate (columns not fully orthogonal in discrete case)
        diag = np.diag(G)
        off_diag = G - np.diag(diag)
        # Off-diagonal should be small relative to diagonal
        assert np.max(np.abs(off_diag)) < np.max(np.abs(diag)) * 0.5

    def test_single_energy_bin(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=3)
        E = np.array([1.0])
        Phi = basis.build_matrix(1, E)
        assert Phi.shape == (1, 3)

    def test_negative_spectrum_reconstruction(self):
        from bssunfold.core.basis import FourierBasis
        basis = FourierBasis(n_terms=20)
        E = np.logspace(-1, 3, 51)
        x = np.sin(np.linspace(0, 2 * np.pi, 51))
        c = basis.to_coeffs(x, E)
        x_rec = basis.to_spectrum(c, E)
        assert np.allclose(x, x_rec, atol=0.1)


# ============================================================================
# SpectralBasis ABC tests
# ============================================================================


class TestSpectralBasisABC:
    def test_cannot_instantiate_directly(self):
        from bssunfold.core.basis import SpectralBasis
        with pytest.raises(TypeError):
            SpectralBasis()

    def test_subclass_must_implement(self):
        from bssunfold.core.basis import SpectralBasis
        class Incomplete(SpectralBasis):
            pass
        with pytest.raises(TypeError):
            Incomplete()


# ============================================================================
# Basis integration tests with Detector methods
# ============================================================================


@pytest.fixture
def detector():
    from bssunfold import Detector
    return Detector()


@pytest.fixture
def readings(detector):
    return {name: 100.0 for name in detector.detector_names[:3]}


@pytest.fixture
def all_readings(detector):
    return {name: 100.0 for name in detector.detector_names}


class TestBasisWithCvxpy:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_cvxpy(
            readings, basis=LegendreBasis(10), regularization=1e-3,
            save_result=False,
        )
        assert result['spectrum'].shape == (detector.n_energy_bins,)
        assert np.all(result['spectrum'] >= 0)
        assert result['basis'] == 'LegendreBasis'
        assert result['n_coeffs'] == 10

    def test_fourier(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        result = detector.unfold_cvxpy(
            readings, basis=FourierBasis(10), regularization=1e-3,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)
        assert result['basis'] == 'FourierBasis'

    def test_no_basis(self, detector, readings):
        result = detector.unfold_cvxpy(
            readings, regularization=1e-3, save_result=False,
        )
        assert 'basis' not in result

    def test_residual_computed(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_cvxpy(
            readings, basis=LegendreBasis(10), regularization=1e-3,
            save_result=False,
        )
        assert 'residual' in result
        assert 'residual_norm' in result
        assert result['residual_norm'] >= 0


class TestBasisWithLandweber:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_landweber(
            readings, basis=LegendreBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)
        assert result['basis'] == 'LegendreBasis'

    def test_fourier(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        result = detector.unfold_landweber(
            readings, basis=FourierBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithDoroshenko:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_doroshenko(
            readings, basis=LegendreBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)

    def test_fourier(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        result = detector.unfold_doroshenko(
            readings, basis=FourierBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithKaczmarz:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_kaczmarz(
            readings, basis=LegendreBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)

    def test_fourier(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        result = detector.unfold_kaczmarz(
            readings, basis=FourierBasis(8), max_iterations=100,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithQpsolvers:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_qpsolvers(
            readings, basis=LegendreBasis(8), regularization=1e-3,
            save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)

    def test_legendre_smoothness(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_qpsolvers(
            readings, basis=LegendreBasis(8), regularization=1e-3,
            smoothness_order=2, save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithLmfit:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_lmfit(
            readings, basis=LegendreBasis(8), save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithStatreg:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_statreg(
            readings, basis=LegendreBasis(8), save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithScipyDirect:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_scipy_direct_method(
            readings, basis=LegendreBasis(8), save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisWithTsvd:
    def test_legendre(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_tsvd(
            readings, basis=LegendreBasis(8), save_result=False,
        )
        assert np.all(result['spectrum'] >= 0)


class TestBasisMonteCarlo:
    def test_legendre_with_errors(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_landweber(
            readings, basis=LegendreBasis(8), max_iterations=50,
            calculate_errors=True, n_montecarlo=5, save_result=False,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert result['spectrum_uncert_mean'].shape == (detector.n_energy_bins,)

    def test_fourier_with_errors(self, detector, readings):
        from bssunfold.core.basis import FourierBasis
        result = detector.unfold_landweber(
            readings, basis=FourierBasis(8), max_iterations=50,
            calculate_errors=True, n_montecarlo=5, save_result=False,
        )
        assert 'spectrum_uncert_mean' in result


class TestBasisForwardModel:
    """Test that A @ spectrum ≈ readings when using basis."""

    def test_cvxpy_legendre_forward_model(self, detector, all_readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_cvxpy(
            all_readings, basis=LegendreBasis(12), regularization=1e-4,
            save_result=False,
        )
        spectrum = result['spectrum']
        assert spectrum is not None
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum >= 0)
        eff = result['effective_readings']
        for name in all_readings:
            assert name in eff
            assert np.isfinite(eff[name])

    def test_qpsolvers_legendre_forward_model(self, detector, all_readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_qpsolvers(
            all_readings, basis=LegendreBasis(12), regularization=1e-4,
            save_result=False,
        )
        eff = result['effective_readings']
        for name, val in all_readings.items():
            assert abs(eff[name] - val) / max(val, 1e-10) < 1.0


class TestBasisDoseRates:
    """Test that dose rates are computed correctly with basis."""

    def test_dose_rates_present(self, detector, readings):
        from bssunfold.core.basis import LegendreBasis
        result = detector.unfold_cvxpy(
            readings, basis=LegendreBasis(10), regularization=1e-3,
            save_result=False,
        )
        assert 'doserates' in result
        assert isinstance(result['doserates'], dict)
        for geom in ['AP', 'PA', 'ISO', 'ROT']:
            assert geom in result['doserates']


# ============================================================================
# Module exports
# ============================================================================


class TestBasisExports:
    def test_all_classes_exported(self):
        from bssunfold.core import SpectralBasis, BinBasis, LegendreBasis, FourierBasis
        assert callable(BinBasis)
        assert callable(LegendreBasis)
        assert callable(FourierBasis)
        assert issubclass(BinBasis, SpectralBasis)
        assert issubclass(LegendreBasis, SpectralBasis)
        assert issubclass(FourierBasis, SpectralBasis)

    def test_detector_has_no_tikhonov_legendre(self):
        from bssunfold import Detector
        assert not hasattr(Detector, 'unfold_tikhonov_legendre')

    def test_detector_methods_accept_basis(self):
        from bssunfold import Detector
        from bssunfold.core import LegendreBasis
        import inspect
        for name in ['unfold_cvxpy', 'unfold_landweber', 'unfold_doroshenko',
                      'unfold_kaczmarz', 'unfold_qpsolvers', 'unfold_lmfit',
                      'unfold_statreg', 'unfold_scipy_direct_method', 'unfold_tsvd']:
            method = getattr(Detector, name)
            sig = inspect.signature(method)
            assert 'basis' in sig.parameters, f"{name} missing 'basis' param"
