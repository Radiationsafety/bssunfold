"""Comprehensive coverage tests targeting untested code paths.

Tests for error injection, performance, application of all options
in methods, edge cases, and fallback implementations.
"""

import sys
import builtins
import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch, MagicMock
from numpy.testing import assert_almost_equal


# ============================================================================
# _matrix_utils.py (22% -> 100%)
# ============================================================================

class TestMatrixUtils:
    def test_create_derivative_matrix_order1(self):
        from bssunfold.core._matrix_utils import create_derivative_matrix
        n = 5
        L = create_derivative_matrix(n, 1)
        assert L.shape == (n - 1, n)
        assert L.nnz > 0

    def test_create_derivative_matrix_order2(self):
        from bssunfold.core._matrix_utils import create_derivative_matrix
        n = 5
        L = create_derivative_matrix(n, 2)
        assert L.shape == (n - 2, n)
        assert L[0, 0] == 1
        assert L[0, 1] == -2
        assert L[0, 2] == 1

    def test_create_derivative_matrix_invalid_order(self):
        from bssunfold.core._matrix_utils import create_derivative_matrix
        with pytest.raises(ValueError, match="Unsupported derivative order"):
            create_derivative_matrix(5, 3)

    def test_create_derivative_matrix_small_n(self):
        from bssunfold.core._matrix_utils import create_derivative_matrix
        L1 = create_derivative_matrix(2, 1)
        assert L1.shape == (1, 2)
        L2 = create_derivative_matrix(3, 2)
        assert L2.shape == (1, 3)

    def test_build_tikhonov_system_normal(self):
        from bssunfold.core._matrix_utils import build_tikhonov_system
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        L = np.eye(10)
        x = build_tikhonov_system(A, b, 0.01, L)
        assert x is not None
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_build_tikhonov_system_singular(self):
        from bssunfold.core._matrix_utils import build_tikhonov_system
        A = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.array([1.0, 2.0])
        L = np.eye(2)
        result = build_tikhonov_system(A, b, 0.0, L)
        assert result is None

    def test_compute_svd_components(self):
        from bssunfold.core._matrix_utils import compute_svd_components
        np.random.seed(42)
        A = np.random.rand(5, 10)
        U, s, Vt, s_sq = compute_svd_components(A)
        assert U.shape == (5, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 10)
        assert np.allclose(s_sq, s ** 2)


# ============================================================================
# regularization.py (7% -> 95%)
# ============================================================================

class TestRegularization:
    @pytest.fixture
    def ab(self):
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        return A, b

    def _without_pytikhonov(self):
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if 'pytikhonov' == name:
                raise ImportError
            return original_import(name, *args, **kwargs)
        return patch('builtins.__import__', side_effect=mock_import)

    def _restore_pytikhonov(self, saved):
        pass

    def test_estimate_noise_variance(self, ab):
        from bssunfold.core.regularization import _estimate_noise_variance
        A, b = ab
        var = _estimate_noise_variance(A, b)
        assert isinstance(var, float)
        assert var >= 0

    def test_lcurve_fallback(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import lcurve_selection
            lam = lcurve_selection(A, b)
            assert isinstance(lam, float)
            assert lam > 0

    def test_lcurve_fallback_few_alphas(self, ab):
        from bssunfold.core.regularization import _lcurve_fallback
        A_small = np.ones((1, 1))
        b_small = np.ones(1)
        lam = _lcurve_fallback(A_small, b_small, n_alphas=2, alpha_range=(1e-9, 1e2))
        assert lam == 1.0

    def test_gcv_fallback(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import gcv_selection
            lam = gcv_selection(A, b)
            assert isinstance(lam, float)
            assert lam > 0

    def test_gcv_fallback_all_inf(self, ab):
        from bssunfold.core.regularization import _gcv_fallback
        A_inf = np.ones((5, 10))
        b_inf = np.ones(5)
        lam = _gcv_fallback(A_inf, b_inf, n_alphas=5)
        assert lam > 0

    def test_dp_fallback(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import discrepancy_principle_selection
            lam = discrepancy_principle_selection(A, b, noise_var=0.01)
            assert isinstance(lam, float)
            assert lam > 0

    def test_dp_fallback_noise_var_none(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import discrepancy_principle_selection
            lam = discrepancy_principle_selection(A, b)
            assert isinstance(lam, float)

    def test_cosine_similarity_selection(self, ab):
        from bssunfold.core.regularization import cosine_similarity_selection
        A, b = ab
        initial = np.ones(10)
        lam = cosine_similarity_selection(A, b, initial)
        assert isinstance(lam, float)
        assert lam > 0

    def test_cosine_similarity_selection_zero_norm(self, ab):
        from bssunfold.core.regularization import cosine_similarity_selection
        A, b = ab
        initial = np.zeros(10)
        with pytest.raises(ValueError, match="Initial spectrum has zero norm"):
            cosine_similarity_selection(A, b, initial)

    def test_select_reg_parameter_lcurve(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import select_regularization_parameter
            lam = select_regularization_parameter(A, b, method='lcurve')
            assert isinstance(lam, float)

    def test_select_reg_parameter_gcv(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import select_regularization_parameter
            lam = select_regularization_parameter(A, b, method='gcv')
            assert isinstance(lam, float)

    def test_select_reg_parameter_dp(self, ab):
        A, b = ab
        with self._without_pytikhonov():
            from bssunfold.core.regularization import select_regularization_parameter
            lam = select_regularization_parameter(A, b, method='dp', noise_var=0.01)
            assert isinstance(lam, float)

    def test_select_reg_parameter_cosine(self, ab):
        from bssunfold.core.regularization import select_regularization_parameter
        A, b = ab
        initial = np.ones(10)
        lam = select_regularization_parameter(A, b, method='cosine', initial_spectrum=initial)
        assert isinstance(lam, float)

    def test_select_reg_parameter_unknown(self, ab):
        from bssunfold.core.regularization import select_regularization_parameter
        A, b = ab
        with pytest.raises(ValueError, match="Unknown regularization selection method"):
            select_regularization_parameter(A, b, method='unknown')

    def test_compare_regularization_methods(self, ab):
        from bssunfold.core.regularization import compare_regularization_methods
        A, b = ab
        result = compare_regularization_methods(A, b)
        assert 'lcurve' in result
        assert 'dp' in result
        assert 'gcv' in result
        assert 'selected' in result

    def test_compare_regularization_methods_with_plot(self, ab, tmp_path):
        from bssunfold.core.regularization import compare_regularization_methods
        A, b = ab
        plot_path = str(tmp_path / "reg_compare.png")
        result = compare_regularization_methods(A, b, noise_var=0.01, plot=True, plot_path=plot_path)
        assert result is not None

    def test_randomization_experiment(self, ab):
        from bssunfold.core.regularization import randomization_experiment
        A, b = ab
        result = randomization_experiment(A, b, noise_var=0.01, n_samples=3, rseed=0)
        assert 'lcurve' in result
        assert 'dp' in result
        assert 'gcv' in result
        assert 'lcurve_full' in result

    def test_randomization_experiment_custom_methods(self, ab):
        from bssunfold.core.regularization import randomization_experiment
        A, b = ab
        result = randomization_experiment(A, b, noise_var=0.01, n_samples=2, methods=['lcurve', 'gcv'])
        assert 'lcurve' in result
        assert 'gcv' in result
        assert 'dp' not in result

    def test_randomization_experiment_unknown_method(self, ab):
        from bssunfold.core.regularization import randomization_experiment
        A, b = ab
        with warnings.catch_warnings(record=True) as w:
            result = randomization_experiment(A, b, noise_var=0.01, n_samples=2, methods=['unknown_method'])
            assert len(w) >= 1
            assert any("Unknown method" in str(warn.message) for warn in w)


# ============================================================================
# converters.py (17% -> 100%)
# ============================================================================

class TestConverters:
    def test_convert_to_dataframe_with_df(self):
        from bssunfold.utils.converters import convert_to_dataframe
        df = pd.DataFrame({'E_MeV': [0.1, 0.2], 'det1': [1.0, 2.0]})
        result = convert_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert 'E_MeV' in result.columns

    def test_convert_to_dataframe_with_dict(self):
        from bssunfold.utils.converters import convert_to_dataframe
        data = {'E_MeV': [0.1, 0.2], 'det1': [1.0, 2.0]}
        result = convert_to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert 'E_MeV' in result.columns

    def test_convert_to_dataframe_dict_missing_key(self):
        from bssunfold.utils.converters import convert_to_dataframe
        data = {'wrong_key': [1.0, 2.0]}
        with pytest.raises(ValueError, match="must contain 'E_MeV' key"):
            convert_to_dataframe(data)

    def test_convert_to_dataframe_invalid_type(self):
        from bssunfold.utils.converters import convert_to_dataframe
        with pytest.raises(TypeError, match="data must be DataFrame or dict"):
            convert_to_dataframe("invalid")

    def test_convert_to_dict_with_dict(self):
        from bssunfold.utils.converters import convert_to_dict
        data = {'E_MeV': [0.1, 0.2], 'Phi': [1.0, 2.0]}
        result = convert_to_dict(data)
        assert isinstance(result, dict)
        assert isinstance(result['E_MeV'], np.ndarray)

    def test_convert_to_dict_with_df(self):
        from bssunfold.utils.converters import convert_to_dict
        df = pd.DataFrame({'E_MeV': [0.1, 0.2], 'Phi': [1.0, 2.0]})
        result = convert_to_dict(df)
        assert isinstance(result, dict)
        assert isinstance(result['E_MeV'], np.ndarray)

    def test_convert_to_dict_invalid_type(self):
        from bssunfold.utils.converters import convert_to_dict
        with pytest.raises(TypeError, match="data must be DataFrame or dict"):
            convert_to_dict("invalid")

    def test_convert_sensitivities_to_matrix_with_dict(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2, 0.3])
        sens = {'det1': np.array([1.0, 2.0, 3.0]), 'det2': np.array([0.1, 0.2, 0.3])}
        matrix, names = convert_sensitivities_to_matrix(sens, E)
        assert matrix.shape == (3, 2)
        assert names == ['det1', 'det2']

    def test_convert_sensitivities_to_matrix_with_array(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2, 0.3])
        sens = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        matrix, names = convert_sensitivities_to_matrix(sens, E)
        assert matrix.shape == (3, 2)

    def test_convert_sensitivities_array_mismatch(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2])
        sens = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="Number of rows"):
            convert_sensitivities_to_matrix(sens, E)

    def test_convert_sensitivities_array_not_2d(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2, 0.3])
        sens = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="sensitivities array must be 2D"):
            convert_sensitivities_to_matrix(sens, E)

    def test_convert_sensitivities_dict_mismatch(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2, 0.3])
        sens = {'det1': np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="has length"):
            convert_sensitivities_to_matrix(sens, E)

    def test_convert_sensitivities_invalid_type(self):
        from bssunfold.utils.converters import convert_sensitivities_to_matrix
        E = np.array([0.1, 0.2])
        with pytest.raises(TypeError, match="sensitivities must be dict or np.ndarray"):
            convert_sensitivities_to_matrix("invalid", E)

    def test_extract_detector_names_dataframe(self):
        from bssunfold.utils.converters import extract_detector_names
        df = pd.DataFrame({'E_MeV': [0.1, 0.2], 'det1': [1.0, 2.0], 'det2': [3.0, 4.0]})
        names = extract_detector_names(df)
        assert names == ['det1', 'det2']

    def test_extract_detector_names_dict(self):
        from bssunfold.utils.converters import extract_detector_names
        data = {'E_MeV': [0.1, 0.2], 'det1': [1.0, 2.0]}
        names = extract_detector_names(data)
        assert names == ['det1']

    def test_extract_detector_names_invalid(self):
        from bssunfold.utils.converters import extract_detector_names
        with pytest.raises(TypeError, match="data must be DataFrame or dict"):
            extract_detector_names("invalid")


# ============================================================================
# validators.py (34% -> 100%)
# ============================================================================

class TestValidators:
    def test_validate_readings_not_dict(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(TypeError, match="readings must be a dict"):
            validate_readings("invalid", ["det1"])

    def test_validate_readings_negative(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="is negative"):
            validate_readings({"det1": -1.0}, ["det1", "det2"])

    def test_validate_readings_zero_not_allowed(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="is zero, which is not allowed"):
            validate_readings({"det1": 0.0}, ["det1"], allow_zero=False)

    def test_validate_readings_no_valid(self):
        from bssunfold.utils.validators import validate_readings
        with pytest.raises(ValueError, match="No valid detector readings"):
            validate_readings({"invalid": 1.0}, ["det1"])

    def test_validate_energy_grid_basic(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.1, 0.2, 0.3])
        result = validate_energy_grid(E)
        assert np.allclose(result, E)

    def test_validate_energy_grid_with_bounds(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.1, 0.2, 0.3])
        result = validate_energy_grid(E, Emin=0.05, Emax=0.5)
        assert np.allclose(result, E)

    def test_validate_energy_grid_below_emin(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.01, 0.2, 0.3])
        with pytest.raises(ValueError, match="below allowed minimum"):
            validate_energy_grid(E, Emin=0.1)

    def test_validate_energy_grid_above_emax(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.1, 0.2, 3.0])
        with pytest.raises(ValueError, match="above allowed maximum"):
            validate_energy_grid(E, Emax=1.0)

    def test_validate_energy_grid_not_1d(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="must be a 1D array"):
            validate_energy_grid(E)

    def test_validate_energy_grid_too_few(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.1])
        with pytest.raises(ValueError, match="at least"):
            validate_energy_grid(E)

    def test_validate_energy_grid_non_positive(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([-0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="All energy values must be positive"):
            validate_energy_grid(E)

    def test_validate_energy_grid_not_increasing(self):
        from bssunfold.utils.validators import validate_energy_grid
        E = np.array([0.3, 0.1, 0.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_energy_grid(E)

    def test_validate_spectrum_basic(self):
        from bssunfold.utils.validators import validate_spectrum
        spectrum = np.array([1.0, 2.0, 3.0])
        E = np.array([0.1, 0.2, 0.3])
        result = validate_spectrum(spectrum, E)
        assert np.allclose(result, spectrum)

    def test_validate_spectrum_allow_negative(self):
        from bssunfold.utils.validators import validate_spectrum
        spectrum = np.array([1.0, -2.0, 3.0])
        E = np.array([0.1, 0.2, 0.3])
        result = validate_spectrum(spectrum, E, allow_negative=True)
        assert np.allclose(result, spectrum)

    def test_validate_spectrum_not_1d(self):
        from bssunfold.utils.validators import validate_spectrum
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0]])
        E = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="must be 1D array"):
            validate_spectrum(spectrum, E)

    def test_validate_spectrum_wrong_length(self):
        from bssunfold.utils.validators import validate_spectrum
        spectrum = np.array([1.0, 2.0])
        E = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="must match"):
            validate_spectrum(spectrum, E)

    def test_validate_spectrum_negative(self):
        from bssunfold.utils.validators import validate_spectrum
        spectrum = np.array([1.0, -2.0, 3.0])
        E = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="contains.*negative values"):
            validate_spectrum(spectrum, E)

    def test_validate_response_matrix_basic(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        A_out, b_out = validate_response_matrix(A, b)
        assert A_out.shape == (5, 10)
        assert b_out.shape == (5,)

    def test_validate_response_matrix_check_rank(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        A_out, b_out = validate_response_matrix(A, b, check_rank=True)
        assert A_out.shape == (5, 10)

    def test_validate_response_matrix_A_not_2d(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="A must be 2D array"):
            validate_response_matrix(A, b)

    def test_validate_response_matrix_b_not_1d(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.random.rand(5, 10)
        b = np.random.rand(5, 2)
        with pytest.raises(ValueError, match="b must be 1D array"):
            validate_response_matrix(A, b)

    def test_validate_response_matrix_shape_mismatch(self):
        from bssunfold.utils.validators import validate_response_matrix
        A = np.random.rand(5, 10)
        b = np.random.rand(3)
        with pytest.raises(ValueError, match="Number of rows in A"):
            validate_response_matrix(A, b)


# ============================================================================
# interpolation.py (54% -> 100%)
# ============================================================================

class TestInterpolation:
    def test_interpolate_spectrum_basic(self):
        from bssunfold.utils.interpolation import interpolate_spectrum
        E_from = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
        E_to = np.array([0.15, 0.3, 0.8, 1.5])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = interpolate_spectrum(spectrum, E_from, E_to)
        assert len(result) == len(E_to)
        assert np.all(result >= 0)

    def test_interpolate_spectrum_no_replace_negative(self):
        from bssunfold.utils.interpolation import interpolate_spectrum
        E_from = np.array([0.1, 0.2, 0.5])
        E_to = np.array([0.3])
        spectrum = np.array([1.0, 2.0, 3.0])
        result = interpolate_spectrum(spectrum, E_from, E_to, replace_negative=False)
        assert len(result) == 1

    def test_discretize_spectra_with_dict(self):
        from bssunfold.utils.interpolation import discretize_spectra
        spectra = {'E_MeV': [0.1, 0.2, 0.5], 'Phi': [1.0, 2.0, 3.0]}
        target = np.array([0.15, 0.3, 0.8])
        result = discretize_spectra(spectra, target)
        assert 'E_MeV' in result.columns
        assert 'Phi' in result.columns
        assert len(result) == len(target)

    def test_discretize_spectra_with_df(self):
        from bssunfold.utils.interpolation import discretize_spectra
        df = pd.DataFrame({'E_MeV': [0.1, 0.2, 0.5], 'Phi': [1.0, 2.0, 3.0]})
        target = np.array([0.15, 0.3, 0.8])
        result = discretize_spectra(df, target)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(target)

    def test_discretize_spectra_invalid_type(self):
        from bssunfold.utils.interpolation import discretize_spectra
        target = np.array([0.1, 0.2])
        with pytest.raises(TypeError, match="spectra must be DataFrame or dict"):
            discretize_spectra("invalid", target)

    def test_discretize_spectra_no_energy_column(self):
        from bssunfold.utils.interpolation import discretize_spectra
        df = pd.DataFrame({'energy': [0.1, 0.2], 'Phi': [1.0, 2.0]})
        target = np.array([0.15, 0.25])
        result = discretize_spectra(df, target, energy_column='wrong')
        assert len(result) == len(target)

    def test_resample_to_log_grid_basic(self):
        from bssunfold.utils.interpolation import resample_to_log_grid
        E = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_E, new_spec = resample_to_log_grid(spectrum, E, n_points=10)
        assert len(new_E) == 10
        assert len(new_spec) == 10
        assert np.all(new_E > 0)

    def test_resample_to_log_grid_default(self):
        from bssunfold.utils.interpolation import resample_to_log_grid
        E = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_E, new_spec = resample_to_log_grid(spectrum, E)
        assert len(new_E) == len(E)
        assert np.allclose(new_E[0], np.min(E))
        assert np.allclose(new_E[-1], np.max(E))


# ============================================================================
# plotting.py (25% -> 95%)
# ============================================================================

class TestPlotting:
    @pytest.fixture(autouse=True)
    def matplotlib_noninteractive(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_plot_spectrum_basic(self):
        from bssunfold.utils.plotting import plot_spectrum
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        fig, ax = plot_spectrum(E, spec, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_spectrum_with_ax(self):
        from bssunfold.utils.plotting import plot_spectrum
        import matplotlib.pyplot as plt
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        _, ax = plt.subplots()
        fig, ax = plot_spectrum(E, spec, ax=ax, show=False)
        plt.close(fig)

    def test_plot_spectrum_with_save(self, tmp_path):
        from bssunfold.utils.plotting import plot_spectrum
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        save_path = str(tmp_path / "spectrum.png")
        fig, ax = plot_spectrum(E, spec, save_to=save_path, show=False)
        assert tmp_path.joinpath("spectrum.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_spectrum_log_y(self):
        from bssunfold.utils.plotting import plot_spectrum
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        fig, ax = plot_spectrum(E, spec, log_y=True, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_spectrum_with_label(self):
        from bssunfold.utils.plotting import plot_spectrum
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        fig, ax = plot_spectrum(E, spec, label="test", show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_response_functions_basic(self):
        from bssunfold.utils.plotting import plot_response_functions
        E = np.array([0.1, 0.2, 0.5, 1.0])
        sens = {'det1': np.array([1.0, 2.0, 3.0, 4.0])}
        fig, ax = plot_response_functions(E, sens, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_response_functions_with_ax(self):
        from bssunfold.utils.plotting import plot_response_functions
        import matplotlib.pyplot as plt
        E = np.array([0.1, 0.2, 0.5, 1.0])
        sens = {'det1': np.array([1.0, 2.0, 3.0, 4.0])}
        _, ax = plt.subplots()
        fig, ax = plot_response_functions(E, sens, ax=ax, show=False)
        plt.close(fig)

    def test_plot_response_functions_with_save(self, tmp_path):
        from bssunfold.utils.plotting import plot_response_functions
        E = np.array([0.1, 0.2, 0.5, 1.0])
        sens = {'det1': np.array([1.0, 2.0, 3.0, 4.0])}
        save_path = str(tmp_path / "rf.png")
        fig, ax = plot_response_functions(E, sens, save_to=save_path, show=False)
        assert tmp_path.joinpath("rf.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_fill_between(self):
        from bssunfold.utils.plotting import plot_with_uncertainty
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        uncert_min = np.array([0.5, 1.0, 2.0, 3.0])
        uncert_max = np.array([1.5, 3.0, 4.0, 5.0])
        fig, ax = plot_with_uncertainty(E, spec, uncert_min=uncert_min, uncert_max=uncert_max, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_errorbar(self):
        from bssunfold.utils.plotting import plot_with_uncertainty
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        uncert_std = np.array([0.2, 0.3, 0.4, 0.5])
        fig, ax = plot_with_uncertainty(E, spec, uncert_std=uncert_std, plot_style='errorbar', show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_errorbar_many_points(self):
        from bssunfold.utils.plotting import plot_with_uncertainty
        E = np.logspace(-9, 2, 100)
        spec = np.ones(100) * 0.5
        uncert_std = np.ones(100) * 0.1
        fig, ax = plot_with_uncertainty(E, spec, uncert_std=uncert_std, plot_style='errorbar', show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_reference(self):
        from bssunfold.utils.plotting import plot_with_uncertainty
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        ref = {'E_MeV': np.array([0.1, 0.2, 0.5, 1.0]), 'Phi': np.array([1.0, 2.0, 3.0, 4.0])}
        fig, ax = plot_with_uncertainty(E, spec, reference_spectrum=ref, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_save(self, tmp_path):
        from bssunfold.utils.plotting import plot_with_uncertainty
        E = np.array([0.1, 0.2, 0.5, 1.0])
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        save_path = str(tmp_path / "uncert.png")
        fig, ax = plot_with_uncertainty(E, spec, save_to=save_path, show=False)
        assert tmp_path.joinpath("uncert.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_residuals_basic(self):
        from bssunfold.utils.plotting import plot_residuals
        measured = np.array([1.0, 2.0, 3.0])
        calculated = np.array([0.9, 2.1, 2.8])
        fig, ax = plot_residuals(measured, calculated, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_residuals_with_names(self):
        from bssunfold.utils.plotting import plot_residuals
        measured = np.array([1.0, 2.0, 3.0])
        calculated = np.array([0.9, 2.1, 2.8])
        names = ['det1', 'det2', 'det3']
        fig, ax = plot_residuals(measured, calculated, detector_names=names, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_residuals_with_save(self, tmp_path):
        from bssunfold.utils.plotting import plot_residuals
        measured = np.array([1.0, 2.0, 3.0])
        calculated = np.array([0.9, 2.1, 2.8])
        save_path = str(tmp_path / "residuals.png")
        fig, ax = plot_residuals(measured, calculated, save_to=save_path, show=False)
        assert tmp_path.joinpath("residuals.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_residuals_with_ax(self):
        from bssunfold.utils.plotting import plot_residuals
        import matplotlib.pyplot as plt
        measured = np.array([1.0, 2.0, 3.0])
        calculated = np.array([0.9, 2.1, 2.8])
        _, ax = plt.subplots()
        fig, ax = plot_residuals(measured, calculated, ax=ax, show=False)
        plt.close(fig)


# ============================================================================
# unfolding_methods.py (72% -> 95%)
# ============================================================================

class TestUnfoldingMethodsEdgeCases:
    def test_solve_landweber_zero_norm_A(self):
        from bssunfold.core.unfolding_methods import solve_landweber
        A = np.zeros((3, 5))
        b = np.zeros(3)
        x0 = np.ones(5)
        with pytest.warns(UserWarning, match="zero norm"):
            x, iterations, converged = solve_landweber(A, b, x0)
        assert np.allclose(x, x0)

    def test_solve_mlem_zero_b(self):
        from bssunfold.core.unfolding_methods import solve_mlem
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.zeros(5)
        x0 = np.ones(10) * 0.5
        x, iterations, converged = solve_mlem(A, b, x0, max_iterations=50)
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_kaczmarz_with_zero_rows(self):
        from bssunfold.core.unfolding_methods import solve_kaczmarz
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([1.0, 0.0])
        x0 = np.zeros(2)
        x, iterations, converged = solve_kaczmarz(A, b, x0, max_iterations=100)
        assert x.shape == (2,)

    def test_solve_kaczmarz_omega_warning(self):
        from bssunfold.core.unfolding_methods import solve_kaczmarz
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.zeros(10)
        with pytest.warns(UserWarning, match="omega"):
            x, iters, conv = solve_kaczmarz(A, b, x0, omega=3.0, max_iterations=10)
        assert x.shape == (10,)

    def test_solve_doroshenko_denominator_zero(self):
        from bssunfold.core.unfolding_methods import solve_doroshenko
        A = np.array([[0.0, 1.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0])
        x0 = np.array([0.0, 0.0])
        x, iterations, converged = solve_doroshenko(A, b, x0, max_iterations=100)
        assert x.shape == (2,)

    def test_solve_cvxpy_default_solver(self):
        from bssunfold.core.unfolding_methods import solve_cvxpy
        np.random.seed(42)
        A = np.random.rand(3, 5)
        b = np.random.rand(3)
        x = solve_cvxpy(A, b, alpha=1e-4, norm=2, solver='default')
        assert x.shape == (5,)

    def test_solve_qpsolvers_L1_norm(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x = solve_qpsolvers(A, b, alpha=1e-4, norm=1, solver='osqp')
        assert x is not None
        assert x.shape == (10,)

    def test_solve_qpsolvers_smoothness_order_1(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x = solve_qpsolvers(A, b, alpha=1e-4, norm=2, solver='osqp',
                            smoothness_order=1, smoothness_weight=1.0)
        assert x is not None
        assert x.shape == (10,)

    def test_solve_qpsolvers_smoothness_order_2(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x = solve_qpsolvers(A, b, alpha=1e-4, norm=2, solver='osqp',
                            smoothness_order=2, smoothness_weight=1.0)
        assert x is not None
        assert x.shape == (10,)

    def test_solve_qpsolvers_L1_smoothness(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x = solve_qpsolvers(A, b, alpha=1e-4, norm=1, solver='osqp',
                            smoothness_order=1, smoothness_weight=1.0)
        assert x is not None
        assert x.shape == (10,)

    def test_solve_qpsolvers_unavailable_solver(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        from qpsolvers import available_solvers
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x = solve_qpsolvers(A, b, alpha=1e-4, norm=2, solver='nonexistent_solver')
        if 'osqp' in available_solvers:
            assert x is not None
        else:
            assert x is None

    def test_solve_lmfit_leastsq(self):
        from bssunfold.core.unfolding_methods import solve_lmfit
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)
        x, success, message, nfev = solve_lmfit(
            A, b, x0, method='leastsq', model_name='elastic',
            regularization=1e-4, regularization2=1e-4, l1_weight=0.5
        )
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_lmfit_lasso_leastsq(self):
        from bssunfold.core.unfolding_methods import solve_lmfit
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)
        x, success, message, nfev = solve_lmfit(
            A, b, x0, method='leastsq', model_name='lasso', regularization=1e-4
        )
        assert x.shape == (10,)

    def test_solve_lmfit_ridge_leastsq(self):
        from bssunfold.core.unfolding_methods import solve_lmfit
        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.random.rand(10)
        b = A @ x_true
        x0 = np.ones(10)
        x, success, message, nfev = solve_lmfit(
            A, b, x0, method='leastsq', model_name='ridge', regularization=1e-4
        )
        assert x.shape == (10,)

    def test_solve_lmfit_invalid_model(self):
        from bssunfold.core.unfolding_methods import solve_lmfit
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        x0 = np.ones(10)
        with pytest.raises(ValueError, match="Unknown model_name"):
            solve_lmfit(A, b, x0, model_name='invalid')

    def test_solve_cvxpy_exception(self):
        with patch('cvxpy.Problem.solve', side_effect=Exception("Solver error")):
            from bssunfold.core.unfolding_methods import solve_cvxpy
            A = np.random.rand(5, 10)
            b = np.random.rand(5)
            with pytest.warns(UserWarning, match="CVXPY solving failed"):
                x = solve_cvxpy(A, b, alpha=1e-4)
                assert x.shape == (10,)
                assert np.all(x == 0)

    def test_solve_qpsolvers_L1_no_solution(self):
        with patch('qpsolvers.solve_qp', return_value=None):
            from bssunfold.core.unfolding_methods import solve_qpsolvers
            A = np.random.rand(5, 10)
            b = np.random.rand(5)
            x = solve_qpsolvers(A, b, alpha=1e-4, norm=1, solver='osqp')
            assert x is None

    def test_solve_qpsolvers_invalid_norm(self):
        from bssunfold.core.unfolding_methods import solve_qpsolvers
        A = np.random.rand(5, 10)
        b = np.random.rand(5)
        with pytest.raises(ValueError, match="Unsupported norm type"):
            solve_qpsolvers(A, b, alpha=1e-4, norm=3, solver='osqp')

    def test_solve_qpsolvers_none_result(self):
        with patch('qpsolvers.solve_qp', return_value=None):
            from bssunfold.core.unfolding_methods import solve_qpsolvers
            A = np.random.rand(5, 10)
            b = np.random.rand(5)
            with pytest.warns(UserWarning, match="did not find a solution"):
                x = solve_qpsolvers(A, b, alpha=1e-4, norm=2, solver='osqp')
                assert x is None


# ============================================================================
# unfold_qpsolvers.py (38% -> 95%)
# ============================================================================

class TestUnfoldQpsolversCoverage:
    @pytest.fixture
    def detector(self):
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        return {detector.detector_names[0]: 100.0}

    def test_qpsolvers_cosine_reg(self, detector, readings):
        initial = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_qpsolvers(
            readings, regularization=1e-3,
            regularization_method='cosine',
            initial_spectrum=initial,
        )
        assert 'spectrum' in result

    def test_qpsolvers_cosine_no_initial(self, detector, readings):
        with pytest.raises(ValueError, match="initial_spectrum must be provided"):
            detector.unfold_qpsolvers(
                readings, regularization_method='cosine',
            )

    def test_qpsolvers_lcurve_reg(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings, regularization_method='lcurve',
        )
        assert 'spectrum' in result

    def test_qpsolvers_gcv_reg(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings, regularization_method='gcv',
        )
        assert 'spectrum' in result

    def test_qpsolvers_dp_reg(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings, regularization_method='dp', noise_var=0.01,
        )
        assert 'spectrum' in result

    def test_qpsolvers_cosine_wrong_norm(self, detector, readings):
        initial = np.ones(detector.n_energy_bins) * 0.5
        with pytest.warns(UserWarning, match="assumes L2"):
            result = detector.unfold_qpsolvers(
                readings, norm=1, regularization_method='cosine',
                initial_spectrum=initial,
            )
        assert 'spectrum' in result

    def test_qpsolvers_auto_with_norm1(self, detector, readings):
        with pytest.warns(UserWarning, match="assume L2"):
            result = detector.unfold_qpsolvers(
                readings, norm=1, regularization_method='lcurve',
            )
        assert 'spectrum' in result

    def test_qpsolvers_smoothness_1(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings, smoothness_order=1, smoothness_weight=0.5,
        )
        assert 'spectrum' in result

    def test_qpsolvers_smoothness_2(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings, smoothness_order=2, smoothness_weight=0.5,
        )
        assert 'spectrum' in result

    def test_qpsolvers_with_errors(self, detector, readings):
        result = detector.unfold_qpsolvers(
            readings,
            regularization=1e-3,
            calculate_errors=True,
            n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_qpsolvers_save_result_false(self, detector, readings):
        result = detector.unfold_qpsolvers(readings, save_result=False)
        assert 'spectrum' in result

    def test_qpsolvers_cosine_wrong_initial_length(self, detector, readings):
        initial = np.ones(5)
        with pytest.raises(ValueError, match="must match number of energy bins"):
            detector.unfold_qpsolvers(
                readings, regularization_method='cosine', initial_spectrum=initial,
            )

    def test_qpsolvers_reg_selection_failure(self, detector, readings):
        with patch('bssunfold.core.unfold_qpsolvers.select_regularization_parameter',
                    side_effect=Exception("test error")):
            with pytest.raises(ValueError, match="Regularization selection failed"):
                detector.unfold_qpsolvers(readings, regularization_method='lcurve')

    def test_qpsolvers_solution_none(self, detector, readings):
        with patch('bssunfold.core.unfold_qpsolvers.solve_qpsolvers', return_value=None):
            with pytest.warns(UserWarning, match="Solution not found"):
                result = detector.unfold_qpsolvers(readings)
            assert 'spectrum' in result
            assert np.allclose(result['spectrum'], 0)


# ============================================================================
# unfold_cvxpy.py (66% -> 95%)
# ============================================================================

class TestUnfoldCvxpyCoverage:
    @pytest.fixture
    def detector(self):
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        return {detector.detector_names[0]: 100.0}

    def test_cvxpy_cosine_reg(self, detector, readings):
        initial = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_cvxpy(
            readings, regularization_method='cosine', initial_spectrum=initial,
        )
        assert 'spectrum' in result

    def test_cvxpy_cosine_no_initial(self, detector, readings):
        with pytest.raises(ValueError, match="initial_spectrum must be provided"):
            detector.unfold_cvxpy(readings, regularization_method='cosine')

    def test_cvxpy_lcurve_reg(self, detector, readings):
        result = detector.unfold_cvxpy(readings, regularization_method='lcurve')
        assert 'spectrum' in result

    def test_cvxpy_gcv_reg(self, detector, readings):
        result = detector.unfold_cvxpy(readings, regularization_method='gcv')
        assert 'spectrum' in result

    def test_cvxpy_dp_reg(self, detector, readings):
        result = detector.unfold_cvxpy(readings, regularization_method='dp', noise_var=0.01)
        assert 'spectrum' in result

    def test_cvxpy_cosine_wrong_initial_length(self, detector, readings):
        initial = np.ones(5)
        with pytest.raises(ValueError, match="must match number of energy bins"):
            detector.unfold_cvxpy(
                readings, regularization_method='cosine', initial_spectrum=initial,
            )

    def test_cvxpy_with_errors_and_random_state(self, detector, readings):
        result = detector.unfold_cvxpy(
            readings, regularization=1e-3, calculate_errors=True,
            n_montecarlo=5, random_state=42,
        )
        assert 'spectrum_uncert_mean' in result

    def test_cvxpy_norm_1(self, detector, readings):
        result = detector.unfold_cvxpy(readings, norm=1, regularization=1e-3)
        assert 'spectrum' in result

    def test_cvxpy_with_initial_spectrum(self, detector, readings):
        initial = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_cvxpy(readings, initial_spectrum=initial)
        assert 'spectrum' in result


# ============================================================================
# detector.py (82% -> 95%)
# ============================================================================

class TestDetectorCoverage:
    @pytest.fixture
    def detector(self):
        from bssunfold import Detector
        return Detector()

    @pytest.fixture
    def readings(self, detector):
        return {detector.detector_names[0]: 100.0}

    def test_cosine_similarity_zero_norm(self, detector):
        result = detector._cosine_similarity(np.zeros(5), np.ones(5))
        assert result == 0.0
        result = detector._cosine_similarity(np.ones(5), np.zeros(5))
        assert result == 0.0

    def test_cosine_similarity_normal(self, detector):
        result = detector._cosine_similarity(np.ones(5), np.ones(5))
        assert_almost_equal(result, 1.0)

    def test_add_noise_with_random_state(self, detector):
        readings = {'det1': 100.0, 'det2': 200.0}
        noisy1 = detector._add_noise(readings, noise_level=0.1, random_state=42)
        noisy2 = detector._add_noise(readings, noise_level=0.1, random_state=42)
        assert noisy1 == noisy2
        assert set(noisy1.keys()) == set(readings.keys())

    def test_normalize_initial_with_dict(self, detector):
        from bssunfold import Detector
        det = Detector()
        spectrum_dict = {'E_MeV': det.E_MeV.tolist(), 'Phi': np.ones(det.n_energy_bins).tolist()}
        result = det._normalize_initial_spectrum(spectrum_dict)
        assert result is not None
        assert len(result) == det.n_energy_bins
        assert np.all(result >= 0)

    def test_normalize_initial_with_dataframe(self, detector):
        spectra = pd.DataFrame({
            'E_MeV': detector.E_MeV,
            'Phi': np.ones(detector.n_energy_bins),
        })
        result = detector._normalize_initial_spectrum(spectra)
        assert result is not None
        assert len(result) == detector.n_energy_bins

    def test_normalize_initial_no_spectrum_column(self, detector):
        spectra = pd.DataFrame({
            'E_MeV': detector.E_MeV,
        })
        with pytest.raises(ValueError, match="No spectrum column found"):
            detector._normalize_initial_spectrum(spectra)

    def test_normalize_initial_wrong_length(self, detector):
        initial = np.ones(5)
        with pytest.raises(ValueError, match="must match"):
            detector._normalize_initial_spectrum(initial)

    def test_normalize_initial_invalid_type(self, detector):
        with pytest.raises(TypeError, match="initial_spectrum must be None"):
            detector._normalize_initial_spectrum(42)

    def test_save_figure_png(self, detector, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        save_path = str(tmp_path / "test.png")
        detector._save_figure(fig, save_to=save_path)
        assert tmp_path.joinpath("test.png").exists()
        plt.close(fig)

    def test_save_figure_jpg(self, detector, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        save_path = str(tmp_path / "test.jpg")
        detector._save_figure(fig, save_to=save_path)
        assert tmp_path.joinpath("test.jpg").exists()
        plt.close(fig)

    def test_save_figure_eps(self, detector, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        save_path = str(tmp_path / "test.eps")
        detector._save_figure(fig, save_to=save_path)
        assert tmp_path.joinpath("test.eps").exists()
        plt.close(fig)

    def test_save_figure_unsupported_format(self, detector, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        save_path = str(tmp_path / "test.gif")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detector._save_figure(fig, save_to=save_path)
        plt.close(fig)

    def test_save_figure_none_path(self, detector):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        detector._save_figure(fig, save_to=None)
        plt.close(fig)

    def test_get_effective_readings_needs_interpolation(self, detector):
        different_E = np.array([0.05, 0.15, 0.3, 0.6, 1.2])
        spectra = pd.DataFrame({
            'E_MeV': different_E,
            'Phi': np.ones(len(different_E)),
        })
        result = detector.get_effective_readings_for_spectra(spectra)
        assert isinstance(result, dict)

    def test_get_effective_readings_invalid_type(self, detector):
        with pytest.raises(TypeError, match="Input spectra must be DataFrame or dict"):
            detector.get_effective_readings_for_spectra("invalid")

    def test_get_effective_readings_via_dict(self, detector):
        spectra_dict = {'E_MeV': detector.E_MeV.tolist(), 'Phi': np.ones(detector.n_energy_bins).tolist()}
        result = detector.get_effective_readings_for_spectra(spectra_dict)
        assert isinstance(result, dict)

    def test_get_effective_readings_no_phi_col(self, detector):
        spectra_df = pd.DataFrame({
            'E_MeV': detector.E_MeV,
            'values': np.ones(detector.n_energy_bins),
        })
        result = detector.get_effective_readings_for_spectra(spectra_df)
        assert isinstance(result, dict)

    def test_import_optional_ok(self):
        from bssunfold import Detector
        result = Detector._import_optional('numpy', 'testing')
        assert result is not None

    def test_import_optional_fail(self):
        from bssunfold import Detector
        with pytest.raises(ImportError, match="nonexistent_module is required"):
            Detector._import_optional('nonexistent_module', 'testing')

    def test_compare_regularization_methods_on_detector(self, detector, readings):
        result = detector.compare_regularization_methods(readings)
        assert 'selected' in result
        assert 'lcurve' in result

    def test_randomization_experiment_on_detector(self, detector, readings):
        result = detector.randomization_experiment(readings, n_samples=3)
        assert 'lcurve' in result

    def test_plot_response_functions_no_show(self, detector):
        import matplotlib
        matplotlib.use('Agg')
        detector.plot_response_functions(show=False)

    def test_plot_with_uncertainty_no_uncert(self, detector, readings):
        import matplotlib
        matplotlib.use('Agg')
        result = detector.unfold_cvxpy(readings, regularization=1e-3, save_result=False)
        fig, ax = detector.plot_with_uncertainty(result, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_uncertainty_reference(self, detector, readings):
        import matplotlib
        matplotlib.use('Agg')
        result = detector.unfold_cvxpy(readings, regularization=1e-3, save_result=False)
        ref = {'E_MeV': detector.E_MeV, 'Phi': np.ones(detector.n_energy_bins) * 0.5}
        fig, ax = detector.plot_with_uncertainty(result, reference_spectrum=ref, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_response_functions_save_no_ext(self, detector, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detector.plot_response_functions(save_to=str(tmp_path / "test"), show=False)

    def test_build_system_no_readings(self, detector):
        from bssunfold import Detector
        small_df = pd.DataFrame({
            'E_MeV': [1e-9, 1e-8],
            'det1': [0.1, 0.2],
            'det2': [0.3, 0.4],
        })
        det = Detector(small_df)
        A, b, selected = det._build_system({'det1': 1.0})
        assert len(selected) == 1
        assert A.shape[0] == 1

    def test_validate_readings_on_detector(self, detector):
        validated = detector._validate_readings({detector.detector_names[0]: 50.0})
        assert validated[detector.detector_names[0]] == 50.0

    def test_unfold_cvxpy_save_result_false(self, detector, readings):
        result = detector.unfold_cvxpy(readings, save_result=False)
        assert 'spectrum' in result

    def test_list_results(self, detector, readings):
        detector.unfold_cvxpy(readings, save_result=True, regularization=1e-3)
        results = detector.list_results()
        assert len(results) > 0
        assert isinstance(results, list)

    def test_get_result_nonexistent(self, detector):
        result = detector.get_result("nonexistent_key")
        assert result is None

    def test_get_result_none(self, detector):
        result = detector.get_result()
        assert result is None or isinstance(result, dict)

    def test_discretize_spectra_no_energy_column(self, detector):
        df = pd.DataFrame({'energy': [1e-9, 1e-8], 'Phi': [1.0, 0.5]})
        result = detector.discretize_spectra(df)
        assert isinstance(result, pd.DataFrame)

    def test_unfold_mlem_basic(self, detector, readings):
        result = detector.unfold_mlem(readings, max_iterations=50)
        assert 'spectrum' in result

    def test_unfold_mlem_with_errors(self, detector, readings):
        result = detector.unfold_mlem(
            readings, max_iterations=50, calculate_errors=True, n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_unfold_landweber_with_errors(self, detector, readings):
        result = detector.unfold_landweber(
            readings, max_iterations=50, calculate_errors=True, n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_unfold_kaczmarz_with_errors(self, detector, readings):
        result = detector.unfold_kaczmarz(
            readings, max_iterations=50, calculate_errors=True, n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_unfold_lmfit_with_errors(self, detector, readings):
        result = detector.unfold_lmfit(
            readings, model_name='ridge', regularization=1e-4,
            calculate_errors=True, n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_unfold_mlem_odl_with_errors(self, detector, readings):
        result = detector.unfold_mlem_odl(
            readings, max_iterations=10, calculate_errors=True, n_montecarlo=5,
        )
        assert 'spectrum_uncert_mean' in result

    def test_unfold_combined_without_save(self, detector, readings):
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3, 'save_result': False}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline)
        assert 'spectrum' in result

    def test_unfold_combined_with_verbose_false(self, detector, readings):
        pipeline = [
            {'method': 'cvxpy', 'params': {'regularization': 1e-3}},
        ]
        result = detector.unfold_combined(readings, pipeline=pipeline, verbose=False)
        assert 'spectrum' in result

    def test_unfold_combined_none_final_result(self):
        from bssunfold import Detector
        df = pd.DataFrame({'E_MeV': [1e-9, 1e-8], 'det1': [0.1, 0.2]})
        det = Detector(df)
        result = det.unfold_combined({'det1': 1.0}, pipeline=[])
        assert result is None

    def test_unfold_combined_method_error(self, detector, readings):
        pipeline = [
            {'method': 'doesnotexist', 'params': {}},
        ]
        with pytest.raises(ValueError, match="Method.*not found"):
            detector.unfold_combined(readings, pipeline=pipeline)

    def test_combined_invalid_method_in_pipeline(self, detector, readings):
        pipeline = [{'method': 'invalid', 'params': {}}]
        with pytest.raises(ValueError):
            detector.unfold_combined(readings, pipeline=pipeline)


# ============================================================================
# platform_check.py (79% -> 95%)
# ============================================================================

class TestPlatformCheck:
    def test_check_jax_availability_false(self):
        from bssunfold.platform_check import check_jax_availability
        with patch('builtins.__import__', side_effect=ImportError):
            result = check_jax_availability()
            assert result is False

    def test_check_jax_availability_true(self):
        from bssunfold.platform_check import check_jax_availability
        result = check_jax_availability()
        assert isinstance(result, bool)

    def test_check_proxsuite_availability(self):
        from bssunfold.platform_check import check_proxsuite_availability
        result = check_proxsuite_availability()
        assert isinstance(result, bool)

    def test_check_proxsuite_availability_false(self):
        from bssunfold.platform_check import check_proxsuite_availability
        with patch('builtins.__import__', side_effect=ImportError):
            result = check_proxsuite_availability()
            assert result is False

    def test_check_qpsolvers_extra_availability_true(self):
        from bssunfold.platform_check import check_qpsolvers_extra_availability
        with patch('bssunfold.platform_check.QPSOLVERS_EXTRA_AVAILABLE', False):
            with patch('bssunfold.platform_check.available_solvers', {'osqp', 'ecos'}, create=True):
                result = check_qpsolvers_extra_availability()
                assert result is True

    def test_check_qpsolvers_extra_no_extra(self):
        from bssunfold.platform_check import check_qpsolvers_extra_availability
        with patch('bssunfold.platform_check.QPSOLVERS_EXTRA_AVAILABLE', False):
            with patch('qpsolvers.available_solvers', set(), create=True):
                result = check_qpsolvers_extra_availability()
                assert result is False

    def test_get_available_solvers(self):
        from bssunfold.platform_check import get_available_solvers
        solvers = get_available_solvers()
        assert isinstance(solvers, dict)
        assert 'ecos' in solvers

    def test_is_windows(self):
        from bssunfold.platform_check import is_windows
        assert isinstance(is_windows, bool)

    def test_is_unix(self):
        from bssunfold.platform_check import is_unix
        assert isinstance(is_unix, bool)

    def test_get_recommended_solver_proxqp(self):
        from bssunfold.platform_check import get_recommended_solver
        with patch('bssunfold.platform_check.PROXSUITE_AVAILABLE', True):
            with patch('bssunfold.platform_check.is_windows', False):
                solver = get_recommended_solver()
                assert solver == 'proxqp'

    def test_get_recommended_solver_osqp(self):
        from bssunfold.platform_check import get_recommended_solver
        with patch('bssunfold.platform_check.check_proxsuite_availability'):
            with patch('bssunfold.platform_check.PROXSUITE_AVAILABLE', False):
                solver = get_recommended_solver()
                assert solver == 'osqp'

    def test_get_recommended_solver_windows(self):
        from bssunfold.platform_check import get_recommended_solver
        with patch('bssunfold.platform_check.PROXSUITE_AVAILABLE', True):
            with patch('bssunfold.platform_check.is_windows', True):
                solver = get_recommended_solver()
                assert solver == 'osqp'


# ============================================================================
# logging_config.py (71% -> 100%)
# ============================================================================

class TestLoggingConfig:
    def test_setup_logging_custom_format(self):
        from bssunfold.logging_config import setup_logging
        import logging
        logger = setup_logging(level=logging.DEBUG, format_string="%(message)s")
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_handler(self):
        from bssunfold.logging_config import setup_logging
        import logging
        logger = setup_logging(level=logging.INFO, use_handler=True)
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_get_logger_with_name(self):
        from bssunfold.logging_config import get_logger
        logger = get_logger("test_module")
        assert logger.name == "bssunfold.test_module"

    def test_get_logger_none(self):
        from bssunfold.logging_config import get_logger
        logger = get_logger()
        assert logger is not None
        assert "bssunfold" in logger.name


# ============================================================================
# dose_calculation.py (84% -> 100%)
# ============================================================================

class TestDoseCalculation:
    def test_calculate_dose_rates_empty_cc(self):
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.ones(50)
        result = calculate_dose_rates(spectrum, cc_icrp116={})
        assert isinstance(result, dict)
        assert result['AP'] == 0.0

    def test_calculate_dose_rates_mismatched_length(self):
        from bssunfold.core.dose_calculation import calculate_dose_rates
        spectrum = np.ones(10)
        cc = {'AP': np.ones(50)}
        result = calculate_dose_rates(spectrum, cc_icrp116=cc)
        assert 'AP' in result

    def test_get_icrp116_coefficients(self):
        from bssunfold.core.dose_calculation import get_icrp116_coefficients, ICRP116_COEFFICIENTS
        result = get_icrp116_coefficients()
        assert isinstance(result, dict)

    def test_get_icrp116_coefficients_fallback(self):
        import bssunfold.core.dose_calculation as dc
        dc.ICRP116_COEFFICIENTS = None
        with patch('bssunfold.constants.ICRP116_COEFF_EFFECTIVE_DOSE', {}):
            result = dc.get_icrp116_coefficients()
            assert result == {}


# ============================================================================
# _montecarlo.py and _base_unfolder.py direct tests
# ============================================================================

class TestMonteCarloDirect:
    def test_monte_carlo_uncertainty_direct(self):
        from bssunfold.core._montecarlo import monte_carlo_uncertainty
        def dummy_func(readings, **kwargs):
            return np.ones(10) * readings.get('det1', 1.0)
        readings = {'det1': 100.0, 'det2': 200.0}
        result = monte_carlo_uncertainty(
            func=dummy_func, readings=readings,
            noise_level=0.1, n_samples=5, n_energy_bins=10, random_state=42,
        )
        assert 'spectrum_uncert_mean' in result
        assert 'spectrum_uncert_std' in result
        assert 'spectrum_uncert_min' in result
        assert 'spectrum_uncert_max' in result
        assert 'spectrum_uncert_median' in result
        assert 'spectrum_uncert_percentile_5' in result
        assert 'spectrum_uncert_percentile_95' in result
        assert 'spectrum_uncert_all' in result
        assert result['spectrum_uncert_all'].shape == (5, 10)

    def test_run_unfolding_direct(self):
        from bssunfold.core._base_unfolder import run_unfolding
        np.random.seed(42)
        A = np.random.rand(5, 10)

        def dummy_solve(A_mat, b_vec, **kwargs):
            return np.ones(A_mat.shape[1])

        detector_names = ['det1', 'det2', 'det3', 'det4', 'det5']
        sensitivities = {f'det{i}': A[i] for i in range(5)}
        E_MeV = np.logspace(-9, 2, 10)
        cc = {'AP': np.ones(10), 'PA': np.ones(10)}

        saved_results = {}

        def save_cb(result):
            saved_results['last'] = result
            return 'key_1'

        result = run_unfolding(
            detector_names=detector_names,
            n_energy_bins=10,
            E_MeV=E_MeV,
            sensitivities=sensitivities,
            cc_icrp116=cc,
            save_result_callback=save_cb,
            readings={'det1': 100.0, 'det2': 200.0},
            initial_spectrum=np.ones(10),
            default_initial=np.ones(10) * 0.5,
            solve_func=dummy_solve,
            solve_kwargs={},
            method_name='test',
            calculate_errors=True,
            noise_level=0.1,
            n_montecarlo=3,
            random_state=42,
            save_result=True,
        )
        assert 'spectrum' in result
        assert 'method' in result
        assert result['method'] == 'test'
        assert 'spectrum_uncert_mean' in result


# ============================================================================
# _process_input edge cases in detector.py
# ============================================================================

class TestDetectorInitEdgeCases:
    def test_init_with_dict_no_e_mev(self):
        from bssunfold import Detector
        data = {'det1': [1.0, 2.0]}
        with pytest.raises(ValueError, match="must contain 'E_MeV' key"):
            Detector(data)

    def test_init_with_sensitivities_not_array(self):
        from bssunfold import Detector
        E = np.array([0.1, 0.2])
        with pytest.raises(TypeError, match="must be dict or np.ndarray"):
            Detector(E_MeV=E, sensitivities="invalid")

    def test_init_invalid_combination(self):
        from bssunfold import Detector
        with pytest.raises(ValueError, match="Invalid input combination"):
            Detector(response_functions="invalid")

    def test_init_array_mismatch(self):
        from bssunfold import Detector
        E = np.array([0.1, 0.2, 0.3])
        sens = np.array([[1.0, 0.1], [2.0, 0.2]])
        with pytest.raises(ValueError, match="Number of rows in sensitivities"):
            Detector(E_MeV=E, sensitivities=sens)

    def test_init_less_than_2_energy_bins(self):
        from bssunfold import Detector
        E = np.array([0.1])
        sens = {'det1': [1.0]}
        with pytest.raises(IndexError):
            Detector(E_MeV=E, sensitivities=sens)

    def test_init_E_MeV_2d_raises(self):
        from bssunfold import Detector
        E = np.array([[0.1, 0.2], [0.3, 0.4]])
        sens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        with pytest.raises(ValueError):
            Detector(E_MeV=E, sensitivities=sens)

    def test_process_input_sensitivities_not_dict_or_array(self):
        from bssunfold.core.detector import Detector as Det
        with pytest.raises(TypeError, match="must be dict or np.ndarray"):
            Det()._process_input(None, np.array([0.1, 0.2]), 42)

    def test_init_sensitivities_not_2d_array(self):
        from bssunfold import Detector
        E = np.array([0.1, 0.2, 0.3, 0.4])
        sens = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="must be 2D array"):
            Detector(E_MeV=E, sensitivities=sens)

    def test_normalize_initial_spectrum_none(self):
        from bssunfold import Detector
        d = Detector()
        assert d._normalize_initial_spectrum(None) is None

    def test_normalize_initial_spectrum_ndarray(self):
        from bssunfold import Detector
        d = Detector()
        result = d._normalize_initial_spectrum(np.ones(d.n_energy_bins))
        assert result is not None
        assert len(result) == d.n_energy_bins

    def test_normalize_initial_spectrum_wrong_length(self):
        from bssunfold import Detector
        d = Detector()
        with pytest.raises(ValueError, match="must match"):
            d._normalize_initial_spectrum(np.ones(5))

    def test_normalize_initial_spectrum_dict_no_phi(self):
        from bssunfold import Detector
        d = Detector()
        s = {'E_MeV': d.E_MeV, 'flux': np.ones(d.n_energy_bins)}
        result = d._normalize_initial_spectrum(s)
        assert result is not None

    def test_normalize_initial_spectrum_invalid_type(self):
        from bssunfold import Detector
        d = Detector()
        with pytest.raises(TypeError, match="must be None"):
            d._normalize_initial_spectrum(123)

    def test_convert_rf_no_energy_column(self):
        from bssunfold import Detector
        d = Detector()
        rf_df = pd.DataFrame({
            'col0': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col1': [0.5, 1.0, 1.5, 2.0, 2.5],
            'col2': [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        rf_matrix, energies, names, _ = d._convert_rf_to_matrix_variable_step(rf_df)
        assert rf_matrix.shape == (5, 2)
        assert 'col1' in names

    def test_get_effective_readings_interp_no_phi(self):
        from bssunfold import Detector
        det = Detector()
        spectra_df = pd.DataFrame({
            'E_MeV': det.E_MeV,
            'flux': np.ones(det.n_energy_bins),
        })
        result = det.get_effective_readings_for_spectra(spectra_df)
        assert isinstance(result, dict)

