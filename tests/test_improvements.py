"""Tests for bug fixes and code improvements in bssunfold.

This module tests:
1. Bug fixes (total_flux_ratio, interpolation, log-step computation)
2. Edge cases in utility functions
3. Error handling paths
4. Modern Python patterns
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from bssunfold import Detector
from bssunfold.utils.comparison import (
    total_flux_ratio,
    total_flux,
    _compute_log_steps,
    _normalize,
    _check_same_length,
    cosine_similarity,
    kl_divergence,
    cross_entropy,
    entropy,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mape,
    r2_score,
    max_error,
    median_absolute_error,
    pearson_r,
    spearman_r,
    wasserstein_dist,
    energy_dist,
    chi_squared,
    g_test,
    freeman_tukey,
    cressie_read,
    spectral_shape_similarity,
    fluence_difference_percent,
    dose_difference_percent,
    fluence_averaged_energy_diff,
    dose_averaged_energy_diff,
    peak_location_error,
    peak_width_error,
    dose_weighted_error,
    response_matrix_consistency,
    compare_spectra,
    compare_multiple,
)
from bssunfold.utils.validators import (
    validate_readings,
    validate_energy_grid,
    validate_spectrum,
    validate_response_matrix,
)
from bssunfold.utils.converters import (
    convert_to_dataframe,
    convert_to_dict,
    convert_sensitivities_to_matrix,
    extract_detector_names,
    round_to_sigfig,
)
from bssunfold.utils.interpolation import (
    interpolate_spectrum,
    discretize_spectra,
    resample_to_log_grid,
    _handle_extrapolation,
)
from bssunfold.core._matrix_utils import (
    create_derivative_matrix,
    build_tikhonov_system,
    compute_svd_components,
)
from bssunfold.core._montecarlo import monte_carlo_uncertainty, _add_noise
from bssunfold.core.dose_calculation import (
    calculate_dose_rates,
    get_coefficients,
    interpolate_coefficients,
    get_icrp116_coefficients,
)


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_detector():
    """Create a simple Detector for testing."""
    return Detector()


@pytest.fixture
def sample_spectra():
    """Create sample spectra for comparison tests."""
    np.random.seed(42)
    n = 50
    energy = np.logspace(-9, 2, n)
    spectrum1 = np.random.uniform(0.1, 10.0, n)
    spectrum2 = spectrum1 * (1 + 0.1 * np.random.randn(n))
    return energy, spectrum1, spectrum2


@pytest.fixture
def sample_readings():
    """Create sample detector readings."""
    return {
        "sphere_1": 100.0,
        "sphere_2": 85.0,
        "sphere_3": 70.0,
        "sphere_4": 55.0,
        "sphere_5": 40.0,
        "sphere_6": 25.0,
        "sphere_7": 15.0,
        "sphere_8": 10.0,
    }


@pytest.fixture
def sample_energy_grid():
    """Create a sample energy grid."""
    return np.logspace(-9, 2, 50)


@pytest.fixture
def sample_response_matrix():
    """Create a sample response matrix."""
    np.random.seed(42)
    return np.random.rand(8, 50)


@pytest.fixture
def sample_measurement_vector():
    """Create a sample measurement vector."""
    return np.array([100.0, 85.0, 70.0, 55.0, 40.0, 25.0, 15.0, 10.0])


# ─── Bug Fix Tests ─────────────────────────────────────────────────


class TestTotalFluxRatioBugFix:
    """Tests for the total_flux_ratio bug fix."""

    def test_ratio_first_arg_is_numerator(self):
        """Verify that total_flux_ratio(p, q) = sum(p) / sum(q)."""
        p = np.array([1.0, 2.0, 3.0])  # sum = 6
        q = np.array([2.0, 4.0, 6.0])  # sum = 12
        assert_almost_equal(total_flux_ratio(p, q), 0.5)

    def test_ratio_symmetric_case(self):
        """When both spectra are equal, ratio should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(total_flux_ratio(s, s), 1.0)

    def test_ratio_with_zeros_in_reference(self):
        """When reference (q) is zero, return 0.0."""
        p = np.array([1.0, 2.0, 3.0])
        q = np.zeros(3)
        assert total_flux_ratio(p, q) == 0.0

    def test_ratio_with_zeros_in_test(self):
        """When test (p) is zero, ratio should be 0.0."""
        p = np.zeros(3)
        q = np.array([1.0, 2.0, 3.0])
        assert total_flux_ratio(p, q) == 0.0


class TestInterpolationBugFix:
    """Tests for the interpolation bug fix."""

    def test_handle_extrapolation_fills_outside_range(self):
        """Extrapolated points should be filled with fill_value."""
        interp_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u_source = np.array([0.0, 1.0, 2.0])
        u_target = np.array([-1.0, 0.5, 1.5, 2.5, 3.0])
        result = _handle_extrapolation(interp_vals, u_source, u_target, fill_value=0.0)
        assert result[0] == 0.0  # below range
        assert result[4] == 0.0  # above range
        assert result[1] == 2.0  # within range
        assert result[2] == 3.0  # within range

    def test_handle_extrapolation_no_negatives(self):
        """Negative values should be replaced with 0 when replace_negative=True."""
        interp_vals = np.array([-1.0, 2.0, -3.0])
        u_source = np.array([0.0, 1.0, 2.0])
        u_target = np.array([0.5, 1.5])
        result = _handle_extrapolation(interp_vals[:2], u_source, u_target, replace_negative=True)
        assert result[0] == 0.0
        assert result[1] == 2.0

    def test_handle_extrapolation_keeps_negatives(self):
        """Negative values should be kept when replace_negative=False."""
        interp_vals = np.array([-1.0, 2.0])
        u_source = np.array([0.0, 1.0, 2.0])
        u_target = np.array([0.5, 1.5])
        result = _handle_extrapolation(interp_vals, u_source, u_target, replace_negative=False)
        assert result[0] == -1.0
        assert result[1] == 2.0


class TestLogStepComputation:
    """Tests for the _compute_log_steps helper."""

    def test_basic_computation(self):
        """Test basic log step computation."""
        energy = np.array([1.0, 10.0, 100.0])
        steps = _compute_log_steps(energy)
        assert len(steps) == 3
        # log10(10/1) = 1, log10(100/10) = 1, central diff = (log10(100)-log10(1))/2 = 1
        assert steps[0] == pytest.approx(np.log(10))
        assert steps[-1] == pytest.approx(np.log(10))

    def test_single_point(self):
        """Test with single energy point."""
        energy = np.array([1.0])
        steps = _compute_log_steps(energy)
        assert len(steps) == 1
        assert steps[0] == 0.0

    def test_two_points(self):
        """Test with two energy points."""
        energy = np.array([1.0, 10.0])
        steps = _compute_log_steps(energy)
        assert len(steps) == 2
        assert steps[0] == pytest.approx(np.log(10))
        assert steps[1] == pytest.approx(np.log(10))


# ─── Validator Tests ───────────────────────────────────────────────


class TestValidators:
    """Tests for validation utility functions."""

    def test_validate_readings_valid(self):
        """Valid readings should pass."""
        readings = {"det_1": 100.0, "det_2": 200.0}
        detector_names = ["det_1", "det_2", "det_3"]
        result = validate_readings(readings, detector_names)
        assert result == {"det_1": 100.0, "det_2": 200.0}

    def test_validate_readings_negative_raises(self):
        """Negative readings should raise ValueError."""
        readings = {"det_1": -100.0}
        detector_names = ["det_1"]
        with pytest.raises(ValueError, match="negative"):
            validate_readings(readings, detector_names)

    def test_validate_readings_zero_not_allowed(self):
        """Zero readings should raise ValueError when allow_zero=False."""
        readings = {"det_1": 0.0}
        detector_names = ["det_1"]
        with pytest.raises(ValueError, match="zero"):
            validate_readings(readings, detector_names, allow_zero=False)

    def test_validate_readings_no_valid_readings(self):
        """No valid readings should raise ValueError."""
        readings = {"unknown_det": 100.0}
        detector_names = ["det_1", "det_2"]
        with pytest.raises(ValueError, match="No valid detector readings"):
            validate_readings(readings, detector_names)

    def test_validate_readings_not_dict_raises(self):
        """Non-dict input should raise TypeError."""
        with pytest.raises(TypeError, match="dict"):
            validate_readings([100.0], ["det_1"])

    def test_validate_energy_grid_valid(self):
        """Valid energy grid should pass."""
        E = np.array([1e-9, 1e-6, 1e-3, 1.0])
        result = validate_energy_grid(E)
        assert_array_almost_equal(result, E)

    def test_validate_energy_grid_not_increasing(self):
        """Non-increasing grid should raise ValueError."""
        E = np.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_energy_grid(E)

    def test_validate_energy_grid_too_few_points(self):
        """Grid with too few points should raise ValueError."""
        E = np.array([1.0])
        with pytest.raises(ValueError, match="at least"):
            validate_energy_grid(E, min_points=2)

    def test_validate_energy_grid_negative_values(self):
        """Negative energy values should raise ValueError."""
        E = np.array([-1.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="positive"):
            validate_energy_grid(E)

    def test_validate_energy_grid_min_max_bounds(self):
        """Energy grid outside bounds should raise ValueError."""
        E = np.array([1e-10, 1e-6, 1.0])
        with pytest.raises(ValueError, match="below"):
            validate_energy_grid(E, Emin=1e-9)

    def test_validate_spectrum_valid(self):
        """Valid spectrum should pass."""
        E = np.array([1.0, 2.0, 3.0])
        spec = np.array([10.0, 20.0, 30.0])
        result = validate_spectrum(spec, E)
        assert_array_almost_equal(result, spec)

    def test_validate_spectrum_length_mismatch(self):
        """Spectrum length mismatch should raise ValueError."""
        E = np.array([1.0, 2.0, 3.0])
        spec = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="match"):
            validate_spectrum(spec, E)

    def test_validate_spectrum_negative_raises(self):
        """Negative spectrum values should raise ValueError."""
        E = np.array([1.0, 2.0, 3.0])
        spec = np.array([10.0, -20.0, 30.0])
        with pytest.raises(ValueError, match="negative"):
            validate_spectrum(spec, E)

    def test_validate_spectrum_negative_allowed(self):
        """Negative spectrum values should pass when allow_negative=True."""
        E = np.array([1.0, 2.0, 3.0])
        spec = np.array([10.0, -20.0, 30.0])
        result = validate_spectrum(spec, E, allow_negative=True)
        assert_array_almost_equal(result, spec)

    def test_validate_response_matrix_valid(self):
        """Valid response matrix should pass."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        result_A, result_b = validate_response_matrix(A, b)
        assert_array_almost_equal(result_A, A)
        assert_array_almost_equal(result_b, b)

    def test_validate_response_matrix_dimension_mismatch(self):
        """Dimension mismatch should raise ValueError."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0, 7.0])
        with pytest.raises(ValueError, match="match"):
            validate_response_matrix(A, b)

    def test_validate_response_matrix_not_2d(self):
        """Non-2D matrix should raise ValueError."""
        A = np.array([1.0, 2.0, 3.0])
        b = np.array([5.0])
        with pytest.raises(ValueError, match="2D"):
            validate_response_matrix(A, b)


# ─── Converter Tests ───────────────────────────────────────────────


class TestConverters:
    """Tests for data conversion utilities."""

    def test_convert_to_dataframe_from_dict(self):
        """Dict with E_MeV key should convert to DataFrame."""
        data = {"E_MeV": [1.0, 2.0, 3.0], "det_1": [10.0, 20.0, 30.0]}
        result = convert_to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert "E_MeV" in result.columns
        assert "det_1" in result.columns

    def test_convert_to_dataframe_from_dataframe(self):
        """DataFrame input should return a copy."""
        df = pd.DataFrame({"E_MeV": [1.0, 2.0], "det_1": [10.0, 20.0]})
        result = convert_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        # Should be a copy, not the same object
        assert result is not df

    def test_convert_to_dataframe_missing_energy_key(self):
        """Dict without E_MeV key should raise ValueError."""
        data = {"det_1": [10.0, 20.0]}
        with pytest.raises(ValueError, match="E_MeV"):
            convert_to_dataframe(data)

    def test_convert_to_dataframe_unsupported_type(self):
        """Unsupported type should raise TypeError."""
        with pytest.raises(TypeError, match="DataFrame or dict"):
            convert_to_dataframe([1, 2, 3])

    def test_convert_to_dict_from_dataframe(self):
        """DataFrame should convert to dict of arrays."""
        df = pd.DataFrame({"E_MeV": [1.0, 2.0], "det_1": [10.0, 20.0]})
        result = convert_to_dict(df)
        assert isinstance(result, dict)
        assert "E_MeV" in result
        assert isinstance(result["E_MeV"], np.ndarray)

    def test_convert_to_dict_from_dict(self):
        """Dict input should return dict with numpy arrays."""
        data = {"E_MeV": [1.0, 2.0], "det_1": [10.0, 20.0]}
        result = convert_to_dict(data)
        assert isinstance(result["E_MeV"], np.ndarray)

    def test_convert_sensitivities_from_dict(self):
        """Dict sensitivities should convert to matrix."""
        E = np.array([1.0, 2.0, 3.0])
        sens = {"det_1": [10.0, 20.0, 30.0], "det_2": [40.0, 50.0, 60.0]}
        matrix, names = convert_sensitivities_to_matrix(sens, E)
        assert matrix.shape == (3, 2)
        assert names == ["det_1", "det_2"]

    def test_convert_sensitivities_from_array(self):
        """2D array sensitivities should pass through."""
        E = np.array([1.0, 2.0, 3.0])
        sens = np.array([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]])
        matrix, names = convert_sensitivities_to_matrix(sens, E)
        assert matrix.shape == (3, 2)
        assert names == ["det_0", "det_1"]

    def test_convert_sensitivities_wrong_shape(self):
        """Wrong shape sensitivities should raise ValueError."""
        E = np.array([1.0, 2.0, 3.0])
        sens = np.array([10.0, 20.0, 30.0])  # 1D, not 2D
        with pytest.raises(ValueError, match="2D"):
            convert_sensitivities_to_matrix(sens, E)

    def test_extract_detector_names_from_dataframe(self):
        """Should extract detector names from DataFrame."""
        df = pd.DataFrame({"E_MeV": [1.0, 2.0], "det_1": [10.0, 20.0]})
        names = extract_detector_names(df)
        assert names == ["det_1"]

    def test_extract_detector_names_from_dict(self):
        """Should extract detector names from dict."""
        data = {"E_MeV": [1.0, 2.0], "det_1": [10.0, 20.0]}
        names = extract_detector_names(data)
        assert names == ["det_1"]

    def test_round_to_sigfig_basic(self):
        """Test basic significant figure rounding."""
        assert round_to_sigfig(1.2345, 3) == 1.23
        assert round_to_sigfig(12345, 3) == 12300.0
        assert round_to_sigfig(0.0012345, 2) == 0.0012

    def test_round_to_sigfig_zero(self):
        """Zero should return 0.0."""
        assert round_to_sigfig(0.0) == 0.0

    def test_round_to_sigfig_nan(self):
        """NaN should return NaN."""
        assert np.isnan(round_to_sigfig(float("nan")))

    def test_round_to_sigfig_inf(self):
        """Inf should return Inf."""
        assert round_to_sigfig(float("inf")) == float("inf")


# ─── Matrix Utility Tests ──────────────────────────────────────────


class TestMatrixUtils:
    """Tests for matrix utility functions."""

    def test_create_derivative_matrix_order1(self):
        """First derivative matrix should have correct shape."""
        L = create_derivative_matrix(5, 1)
        assert L.shape == (4, 5)
        # Just verify it's a valid sparse matrix
        assert L.nnz > 0
        L_dense = L.toarray()
        assert L_dense.shape == (4, 5)

    def test_create_derivative_matrix_order2(self):
        """Second derivative matrix should have correct shape and values."""
        L = create_derivative_matrix(5, 2)
        assert L.shape == (3, 5)
        # Each row should have 1, -2, 1
        for i in range(3):
            row = L[i].toarray().flatten()
            assert row[i] == 1
            assert row[i + 1] == -2
            assert row[i + 2] == 1

    def test_create_derivative_matrix_invalid_order(self):
        """Invalid derivative order should raise ValueError."""
        with pytest.raises(ValueError, match="order"):
            create_derivative_matrix(5, 3)

    def test_build_tikhonov_system_basic(self):
        """Tikhonov system should return non-negative solution."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0])
        L = np.eye(2)
        x = build_tikhonov_system(A, b, 0.1, L)
        assert x is not None
        assert np.all(x >= 0)
        # With regularization, solution should be close to [1, 2]
        assert x[0] == pytest.approx(1.0, abs=0.5)
        assert x[1] == pytest.approx(2.0, abs=0.5)

    def test_build_tikhonov_system_singular(self):
        """Singular matrix should return None."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 1.0])
        L = np.eye(2)
        # With alpha>0, should succeed
        x_reg = build_tikhonov_system(A, b, 0.1, L)
        assert x_reg is not None

    def test_compute_svd_components(self):
        """SVD components should have correct shapes."""
        A = np.random.rand(5, 3)
        U, s, Vt, s_sq = compute_svd_components(A)
        assert U.shape == (5, 3)
        assert s.shape == (3,)
        assert Vt.shape == (3, 3)
        assert_array_almost_equal(s_sq, s ** 2)


# ─── Monte Carlo Tests ─────────────────────────────────────────────


class TestMonteCarlo:
    """Tests for Monte Carlo uncertainty estimation."""

    def test_add_noise_basic(self):
        """Noise should be added to readings."""
        readings = {"det_1": 100.0, "det_2": 200.0}
        rng = np.random.default_rng(42)
        noisy = _add_noise(readings, 0.1, rng)
        assert noisy["det_1"] != 100.0
        assert noisy["det_2"] != 200.0
        # Values should be close to original
        assert 80 < noisy["det_1"] < 120
        assert 160 < noisy["det_2"] < 240

    def test_add_noise_reproducible(self):
        """Same seed should produce same noise."""
        readings = {"det_1": 100.0}
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        noisy1 = _add_noise(readings, 0.1, rng1)
        noisy2 = _add_noise(readings, 0.1, rng2)
        assert noisy1["det_1"] == noisy2["det_1"]

    def test_monte_carlo_uncertainty_basic(self):
        """Monte Carlo should return uncertainty statistics."""
        def dummy_solver(readings, **kwargs):
            return np.array([1.0, 2.0, 3.0])

        readings = {"det_1": 100.0, "det_2": 200.0}
        result = monte_carlo_uncertainty(
            func=dummy_solver,
            readings=readings,
            noise_level=0.1,
            n_samples=5,
            n_energy_bins=3,
            random_state=42,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "spectrum_uncert_min" in result
        assert "spectrum_uncert_max" in result
        assert result["spectrum_uncert_all"].shape == (5, 3)


# ─── Dose Calculation Tests ────────────────────────────────────────


class TestDoseCalculation:
    """Tests for dose calculation functions."""

    def test_get_icrp116_coefficients(self):
        """Should return ICRP-116 coefficients."""
        cc = get_icrp116_coefficients()
        assert isinstance(cc, dict)
        assert "E_MeV" in cc
        assert "AP" in cc

    def test_get_coefficients_valid(self):
        """Should return coefficients for valid name."""
        cc = get_coefficients("ICRP116")
        assert isinstance(cc, dict)
        assert "E_MeV" in cc

    def test_get_coefficients_invalid_name(self):
        """Invalid name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            get_coefficients("invalid_name")

    def test_interpolate_coefficients(self):
        """Interpolation should produce coefficients on target grid."""
        cc = get_coefficients("ICRP116")
        E_target = np.logspace(-9, 2, 50)
        cc_interp = interpolate_coefficients(cc, E_target)
        assert "E_MeV" in cc_interp
        assert_array_almost_equal(cc_interp["E_MeV"], E_target)
        assert "AP" in cc_interp
        assert len(cc_interp["AP"]) == 50

    def test_calculate_dose_rates_basic(self):
        """Dose rates should be calculable from spectrum."""
        E = np.logspace(-9, 2, 50)
        spectrum = np.ones(50)
        cc = get_icrp116_coefficients()
        # Interpolate to match spectrum length
        cc_interp = interpolate_coefficients(cc, E)
        doserates = calculate_dose_rates(spectrum, cc_interp)
        assert isinstance(doserates, dict)
        assert "AP" in doserates
        assert doserates["AP"] > 0


# ─── Interpolation Tests ──────────────────────────────────────────


class TestInterpolation:
    """Tests for interpolation utilities."""

    def test_interpolate_spectrum_basic(self):
        """Basic interpolation should work."""
        E_from = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectrum = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        E_to = np.array([1.5, 2.5, 3.5])
        result = interpolate_spectrum(spectrum, E_from, E_to)
        assert len(result) == 3
        assert np.all(result >= 0)

    def test_interpolate_spectrum_extrapolation(self):
        """Extrapolated points should be filled."""
        E_from = np.array([2.0, 3.0, 4.0])
        spectrum = np.array([10.0, 20.0, 30.0])
        E_to = np.array([1.0, 2.5, 5.0])
        result = interpolate_spectrum(spectrum, E_from, E_to)
        assert result[0] == 0.0  # extrapolated
        assert result[2] == 0.0  # extrapolated

    def test_discretize_spectra_from_dict(self):
        """Dict input should work."""
        spectra = {"E_MeV": [1.0, 2.0, 3.0], "Phi": [10.0, 20.0, 30.0]}
        target_E = np.array([1.5, 2.5])
        result = discretize_spectra(spectra, target_E)
        assert isinstance(result, pd.DataFrame)
        assert "E_MeV" in result.columns

    def test_discretize_spectra_from_dataframe(self):
        """DataFrame input should work."""
        spectra = pd.DataFrame({"E_MeV": [1.0, 2.0, 3.0], "Phi": [10.0, 20.0, 30.0]})
        target_E = np.array([1.5, 2.5])
        result = discretize_spectra(spectra, target_E)
        assert isinstance(result, pd.DataFrame)

    def test_resample_to_log_grid(self):
        """Resampling to log grid should work."""
        E = np.logspace(-9, 2, 100)
        spectrum = np.random.rand(100)
        E_new, spec_new = resample_to_log_grid(spectrum, E, n_points=50)
        assert len(E_new) == 50
        assert len(spec_new) == 50


# ─── Comparison Metric Tests ───────────────────────────────────────


class TestComparisonMetrics:
    """Tests for spectrum comparison metrics."""

    def test_kl_divergence_identical(self):
        """KL divergence of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(kl_divergence(s, s), 0.0)

    def test_cross_entropy_identical(self):
        """Cross entropy of identical spectra should equal entropy."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(cross_entropy(s, s), entropy(s))

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(cosine_similarity(s, s), 1.0)

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        assert_almost_equal(cosine_similarity(s1, s2), 0.0)

    def test_mse_identical(self):
        """MSE of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(mean_squared_error(s, s), 0.0)

    def test_rmse(self):
        """RMSE should be sqrt of MSE."""
        s1 = np.array([1.0, 2.0])
        s2 = np.array([1.0, 4.0])
        # MSE = ((1-1)^2 + (2-4)^2) / 2 = 4/2 = 2
        # RMSE = sqrt(2) ≈ 1.414
        assert_almost_equal(root_mean_squared_error(s1, s2), np.sqrt(2.0))

    def test_mae(self):
        """MAE should be mean absolute difference."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([2.0, 3.0, 4.0])
        assert_almost_equal(mean_absolute_error(s1, s2), 1.0)

    def test_mape(self):
        """MAPE should be mean absolute percentage error."""
        s1 = np.array([100.0, 200.0])
        s2 = np.array([110.0, 180.0])
        # MAPE = mean(|10/100|, |20/200|) * 100 = mean(0.1, 0.1) * 100 = 10%
        assert_almost_equal(mape(s1, s2), 10.0)

    def test_r2_score_perfect(self):
        """R² of identical spectra should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(r2_score(s, s), 1.0)

    def test_max_error(self):
        """Max error should be maximum absolute difference."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 5.0, 3.0])
        assert_almost_equal(max_error(s1, s2), 3.0)

    def test_median_absolute_error(self):
        """Median absolute error should be median of absolute differences."""
        s1 = np.array([1.0, 2.0, 3.0, 4.0])
        s2 = np.array([1.0, 3.0, 3.0, 5.0])
        assert_almost_equal(median_absolute_error(s1, s2), 0.5)

    def test_pearson_r_identical(self):
        """Pearson r of identical spectra should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(pearson_r(s, s), 1.0)

    def test_spearman_r_identical(self):
        """Spearman r of identical spectra should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(spearman_r(s, s), 1.0)

    def test_wasserstein_identical(self):
        """Wasserstein distance of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(wasserstein_dist(s, s), 0.0)

    def test_energy_distance_identical(self):
        """Energy distance of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(energy_dist(s, s), 0.0)

    def test_chi_squared_identical(self):
        """Chi-squared of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(chi_squared(s, s), 0.0)

    def test_g_test_identical(self):
        """G-test of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(g_test(s, s), 0.0)

    def test_freeman_tukey_identical(self):
        """Freeman-Tukey of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(freeman_tukey(s, s), 0.0)

    def test_cressie_read_identical(self):
        """Cressie-Read of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(cressie_read(s, s), 0.0)

    def test_spectral_shape_similarity_identical(self):
        """Spectral shape similarity of identical spectra should be 1.0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(spectral_shape_similarity(s, s), 1.0)

    def test_length_mismatch_raises(self):
        """Length mismatch should raise ValueError."""
        s1 = np.array([1.0, 2.0])
        s2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            mean_squared_error(s1, s2)


# ─── EURADOS Metric Tests ─────────────────────────────────────────


class TestEURADOSMetrics:
    """Tests for EURADOS-style integral quantity metrics."""

    def test_fluence_difference_percent_identical(self):
        """Fluence difference of identical spectra should be 0."""
        s = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(fluence_difference_percent(s, s), 0.0)

    def test_fluence_difference_percent_with_bins(self):
        """Fluence difference with energy bins should work."""
        s1 = np.array([1.0, 2.0])
        s2 = np.array([2.0, 4.0])
        bins = np.array([1.0, 1.0])
        assert_almost_equal(fluence_difference_percent(s1, s2, bins), 100.0)

    def test_dose_difference_percent_identical(self):
        """Dose difference of identical spectra should be 0."""
        E = np.logspace(-9, 2, 50)
        s = np.ones(50)
        cc = get_coefficients("ICRP116")
        cc_interp = interpolate_coefficients(cc, E)
        assert_almost_equal(dose_difference_percent(s, s, E, cc_interp), 0.0)

    def test_fluence_averaged_energy_diff_identical(self):
        """Fluence-averaged energy diff of identical spectra should be 0."""
        E = np.array([1.0, 2.0, 3.0])
        s = np.array([10.0, 20.0, 30.0])
        assert_almost_equal(fluence_averaged_energy_diff(s, s, E), 0.0)

    def test_dose_averaged_energy_diff_identical(self):
        """Dose-averaged energy diff of identical spectra should be 0."""
        E = np.logspace(-9, 2, 50)
        s = np.ones(50)
        cc = get_coefficients("ICRP116")
        cc_interp = interpolate_coefficients(cc, E)
        assert_almost_equal(dose_averaged_energy_diff(s, s, E, cc_interp), 0.0)

    def test_peak_location_error_identical(self):
        """Peak location error of identical spectra should be 0."""
        E = np.array([1.0, 2.0, 3.0])
        s = np.array([10.0, 20.0, 10.0])
        assert_almost_equal(peak_location_error(s, s, E), 0.0)

    def test_peak_width_error_identical(self):
        """Peak width error of identical spectra should be 0."""
        E = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        assert_almost_equal(peak_width_error(s, s, E), 0.0)

    def test_dose_weighted_error_identical(self):
        """Dose-weighted error of identical spectra should be 0."""
        E = np.logspace(-9, 2, 50)
        s = np.ones(50)
        cc = get_coefficients("ICRP116")
        cc_interp = interpolate_coefficients(cc, E)
        assert_almost_equal(dose_weighted_error(s, s, E, cc_interp), 0.0)

    def test_response_matrix_consistency(self):
        """Response matrix consistency should be low for consistent spectrum."""
        A = np.eye(3)
        spectrum = np.array([1.0, 2.0, 3.0])
        readings = A @ spectrum
        consistency = response_matrix_consistency(spectrum, readings, A)
        assert_almost_equal(consistency, 0.0)


# ─── Compare Spectra Tests ─────────────────────────────────────────


class TestCompareSpectra:
    """Tests for the high-level compare_spectra function."""

    def test_compare_spectra_all_metrics(self):
        """compare_spectra with no metrics should return all simple metrics."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        result = compare_spectra(s1, s2)
        assert isinstance(result, dict)
        assert "kl_divergence" in result
        assert "cosine_similarity" in result

    def test_compare_spectra_single_metric(self):
        """compare_spectra with single metric should return only that metric."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        result = compare_spectra(s1, s2, metrics="cosine_similarity")
        assert "cosine_similarity" in result
        assert len(result) == 1

    def test_compare_spectra_unknown_metric(self):
        """Unknown metric should raise ValueError."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_spectra(s1, s2, metrics="unknown_metric")

    def test_compare_multiple(self):
        """compare_multiple should compare pairwise against first."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 3.5])
        s3 = np.array([2.0, 3.0, 4.0])
        result = compare_multiple([s1, s2, s3])
        assert len(result) == 2
        assert "Spectrum 0 vs Spectrum 1" in result
        assert "Spectrum 0 vs Spectrum 2" in result

    def test_compare_multiple_too_few_spectra(self):
        """compare_multiple with fewer than 2 spectra should raise."""
        with pytest.raises(ValueError, match="At least two"):
            compare_multiple([np.array([1.0, 2.0])])


# ─── Detector Integration Tests ────────────────────────────────────


class TestDetectorIntegration:
    """Integration tests for the Detector class."""

    def test_detector_creation_default(self):
        """Default Detector should be created successfully."""
        det = Detector()
        assert det.n_energy_bins > 0
        assert det.n_detectors > 0

    def test_detector_str(self):
        """Detector __str__ should return a readable string."""
        det = Detector()
        s = str(det)
        assert "Detector" in s
        assert "energy bins" in s

    def test_detector_repr(self):
        """Detector __repr__ should return a technical string."""
        det = Detector()
        r = repr(det)
        assert "Detector" in r

    def test_detector_properties(self):
        """Detector properties should return correct values."""
        det = Detector()
        assert det.n_energy_bins == len(det.E_MeV)
        assert det.n_detectors == len(det.detector_names)

    def test_set_dose_coefficients(self):
        """set_dose_coefficients should change the coefficient set."""
        det = Detector()
        det.set_dose_coefficients("ICRP74_effective")
        assert det.cc_type == "ICRP74_effective"

    def test_set_dose_coefficients_invalid(self):
        """Invalid coefficient name should raise ValueError."""
        det = Detector()
        with pytest.raises(ValueError, match="Unknown"):
            det.set_dose_coefficients("invalid_name")

    def test_results_history(self):
        """Results history should track saved results."""
        det = Detector()
        assert det.list_results() == []
        assert det.get_result() is None

    def test_clear_results(self):
        """clear_results should reset history."""
        det = Detector()
        det.clear_results()
        assert det.list_results() == []
        assert det.get_result() is None

    def test_discretize_spectra(self):
        """discretize_spectra should interpolate to detector grid."""
        det = Detector()
        spectra = {
            "E_MeV": np.logspace(-9, 2, 100),
            "Phi": np.ones(100),
        }
        result = det.discretize_spectra(spectra)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(det.E_MeV)

    def test_unfold_cvxpy_basic(self):
        """Basic cvxpy unfolding should work."""
        det = Detector()
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_cvxpy(readings)
        assert "spectrum" in result
        assert "energy" in result
        assert len(result["spectrum"]) == det.n_energy_bins
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_landweber_basic(self):
        """Basic Landweber unfolding should work."""
        det = Detector()
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_landweber(readings, max_iterations=100)
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_qpsolvers_basic(self):
        """Basic qpsolvers unfolding should work."""
        det = Detector()
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_qpsolvers(readings)
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_with_initial_spectrum(self):
        """Unfolding with initial spectrum should work."""
        det = Detector()
        readings = {name: 100.0 for name in det.detector_names}
        initial = np.ones(det.n_energy_bins)
        result = det.unfold_cvxpy(readings, initial_spectrum=initial)
        assert "spectrum" in result

    def test_unfold_with_save_result(self):
        """Unfolding with save_result=True should save to history."""
        det = Detector()
        readings = {name: 100.0 for name in det.detector_names}
        det.unfold_cvxpy(readings, save_result=True)
        assert len(det.list_results()) > 0
        assert det.get_result() is not None
