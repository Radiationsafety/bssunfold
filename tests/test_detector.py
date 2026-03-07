import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.bssunfold import Detector


@pytest.fixture
def sample_response_df():
    """Фикстура с примером DataFrame функций отклика."""
    # Создаем тестовый DataFrame с функциями отклика
    E_MeV = np.logspace(-9, 3, 60)
    data = {"E_MeV": E_MeV}
    for i, sphere in enumerate(["0in", "2in", "3in", "5in", "8in"]):
        data[sphere] = np.random.randn(len(E_MeV)) * 0.1 + 0.5
    return pd.DataFrame(data)


@pytest.fixture
def detector(sample_response_df):
    """Фикстура с инициализированным детектором."""
    return Detector(sample_response_df)


@pytest.fixture
def sample_readings():
    """Фикстура с примерами показаний детектора."""
    return {
        "0in": 0.00037707623092440032,
        "2in": 0.0099964357249166195,
        "3in": 0.053668754395163297,
        "5in": 0.18417232269591507,
        "8in": 0.22007281510471705,
    }


class TestDetectorInitialization:
    """Тесты инициализации класса Detector."""

    def test_init_with_valid_data(self, sample_response_df):
        """Тест инициализации с корректными данными."""
        detector = Detector(sample_response_df)
        assert detector is not None
        assert detector.n_detectors == 5  # 5 сфер в тестовых данных
        assert detector.n_energy_bins == 60  # 60 энергетический бин

    def test_str_repr_methods(self, detector):
        """Тест строковых представлений."""
        str_repr = str(detector)
        repr_repr = repr(detector)

        assert "Detector" in str_repr
        assert "energy bins" in str_repr
        assert "detectors" in str_repr

        assert "Detector" in repr_repr
        assert "E_MeV" in repr_repr
        assert "sensitivities" in repr_repr


class TestDetectorProperties:
    """Тесты свойств класса Detector."""

    def test_sensitivities_property(self, detector):
        """Тест свойства sensitivities."""
        sensitivities = detector.sensitivities
        assert isinstance(sensitivities, dict)
        assert len(sensitivities) == detector.n_detectors
        assert all(isinstance(v, np.ndarray) for v in sensitivities.values())

    def test_n_detectors_property(self, detector):
        """Тест свойства n_detectors."""
        assert detector.n_detectors == 5
        assert detector.n_detectors == len(detector.detector_names)

    def test_n_energy_bins_property(self, detector):
        """Тест свойства n_energy_bins."""
        assert detector.n_energy_bins == 60
        assert detector.n_energy_bins == len(detector.E_MeV)


class TestUnfoldingMethods:
    """Тесты методов развертки спектра."""

    def test_unfold_cvxpy_basic(self, detector, sample_readings):
        """Тест базового использования unfold_cvxpy."""
        result = detector.unfold_cvxpy(sample_readings, regularization=1e-4)

        assert isinstance(result, dict)
        assert "energy" in result
        assert "spectrum" in result
        assert "residual_norm" in result
        assert "method" in result
        assert result["method"] == "cvxpy"

        # Проверяем типы данных
        assert isinstance(result["energy"], np.ndarray)
        assert isinstance(result["spectrum"], np.ndarray)
        assert isinstance(result["residual_norm"], float)

        # Проверяем размерности
        assert len(result["energy"]) == detector.n_energy_bins
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_unfold_cvxpy_with_solver(self, detector, sample_readings):
        """Тест unfold_cvxpy с указанием солвера."""
        result = detector.unfold_cvxpy(sample_readings, solver="ECOS")
        assert result["method"] == "cvxpy"

    def test_unfold_cvxpy_with_errors(self, detector, sample_readings):
        """Тест unfold_cvxpy с расчетом ошибок."""
        result = detector.unfold_cvxpy(
            sample_readings,
            calculate_errors=True,
            regularization=1e-2,  # Используем регуляризацию для стабильности
        )

        # Проверяем наличие полей с ошибками
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "spectrum_uncert_min" in result
        assert "spectrum_uncert_max" in result

    def test_unfold_landweber_basic(self, detector, sample_readings):
        """Тест базового использования unfold_landweber."""
        result = detector.unfold_landweber(sample_readings, max_iterations=100)

        assert isinstance(result, dict)
        assert "energy" in result
        assert "spectrum" in result
        assert "iterations" in result
        assert "converged" in result
        assert "method" in result
        assert result["method"] == "Landweber"

        assert isinstance(result["iterations"], int)
        assert isinstance(result["converged"], bool)

    def test_unfold_landweber_with_initial_spectrum(
        self, detector, sample_readings
    ):
        """Тест unfold_landweber с начальным спектром."""
        initial_spectrum = np.ones(detector.n_energy_bins) * 0.1
        result = detector.unfold_landweber(
            sample_readings,
            initial_spectrum=initial_spectrum,
            max_iterations=50,
        )
        assert result["method"] == "Landweber"

    def test_unfold_landweber_with_errors(self, detector, sample_readings):
        """Тест unfold_landweber с расчетом ошибок."""
        result = detector.unfold_landweber(
            sample_readings,
            max_iterations=50,
            calculate_errors=True,
            n_montecarlo=10,  # Используем мало итераций для скорости
        )

        # Проверяем наличие полей с ошибками
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "montecarlo_samples" in result
        assert result["montecarlo_samples"] == 10

    def test_unfold_landweber_invalid_initial_spectrum(
        self, detector, sample_readings
    ):
        """Тест unfold_landweber с некорректным начальным спектром."""
        initial_spectrum = np.ones(10)  # Неправильный размер
        with pytest.raises(
            ValueError, match="must match number of energy bins"
        ):
            detector.unfold_landweber(
                sample_readings, initial_spectrum=initial_spectrum
            )

    def test_clear_results(self, detector, sample_readings):
        """Тест очистки результатов."""
        # Добавляем результаты
        detector.unfold_cvxpy(sample_readings)
        detector.unfold_landweber(sample_readings)

        # Очищаем
        detector.clear_results()

        # Проверяем
        assert len(detector.results_history) == 0
        assert detector.current_result is None


class TestDoseRateCalculation:
    """Тесты расчета мощности дозы."""

    def test_calculate_doserates(self, detector):
        """Тест расчета мощности дозы."""
        # Создаем тестовый спектр
        test_spectrum = np.ones(detector.n_energy_bins) * 0.1

        # коэффициенты ICRP-116
        with patch.object(
            detector,
            "_load_icrp116_coefficients",
            return_value={
                "E_MeV": detector.E_MeV,
                "AP": np.ones_like(detector.E_MeV) * 1e-9,
                "PA": np.ones_like(detector.E_MeV) * 2e-9,
                "ISO": np.ones_like(detector.E_MeV) * 3e-9,
            },
        ):
            doserates = detector._calculate_doserates(test_spectrum)

            assert isinstance(doserates, dict)
            assert "AP" in doserates
            assert "PA" in doserates
            assert "ISO" in doserates

            # Проверяем, что значения - числа с плавающей точкой
            assert all(isinstance(v, float) for v in doserates.values())

    def test_add_noise(self, detector):
        """Тест добавления шума к показаниям."""
        readings = {"det1": 100.0, "det2": 200.0, "det3": 300.0}

        # Фиксируем seed для воспроизводимости
        np.random.seed(42)
        noisy = detector._add_noise(readings, noise_level=0.1)

        assert isinstance(noisy, dict)
        assert set(noisy.keys()) == set(readings.keys())

        # Проверяем, что значения изменились, но не слишком сильно
        for key in readings:
            assert abs(noisy[key] - readings[key]) < readings[key] * 0.5


class TestDetectorInitializationVariants:
    """Тесты различных вариантов инициализации Detector."""

    def test_init_default(self):
        """Инициализация без аргументов (должна использовать RF_GSF)."""
        detector = Detector()
        assert detector.n_detectors > 0
        assert detector.n_energy_bins > 0
        assert detector.E_MeV is not None
        assert len(detector.E_MeV) == detector.n_energy_bins

    def test_init_with_dict(self):
        """Инициализация со словарём."""
        data = {
            "E_MeV": [0.1, 0.2, 0.3],
            "det1": [1.0, 2.0, 3.0],
            "det2": [0.5, 1.5, 2.5],
        }
        detector = Detector(data)
        assert detector.n_detectors == 2
        assert detector.n_energy_bins == 3
        assert "det1" in detector.detector_names
        assert "det2" in detector.detector_names

    def test_init_with_dataframe(self):
        """Инициализация с DataFrame."""
        df = pd.DataFrame({
            "E_MeV": [1e-9, 1e-8, 1e-7],
            "sphere1": [0.1, 0.2, 0.3],
            "sphere2": [0.4, 0.5, 0.6],
        })
        detector = Detector(df)
        assert detector.n_detectors == 2
        assert detector.n_energy_bins == 3

    def test_init_with_arrays(self):
        """Инициализация с E_MeV и sensitivities (dict)."""
        E = np.array([0.1, 0.2, 0.3, 0.4])
        sens = {"detX": [1.0, 2.0, 3.0, 4.0], "detY": [0.1, 0.2, 0.3, 0.4]}
        detector = Detector(E_MeV=E, sensitivities=sens)
        assert detector.n_detectors == 2
        assert detector.n_energy_bins == 4
        assert "detX" in detector.detector_names

    def test_init_with_2d_array(self):
        """Инициализация с E_MeV и sensitivities как 2D массив."""
        E = np.array([0.1, 0.2, 0.3])
        sens = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        detector = Detector(E_MeV=E, sensitivities=sens)
        assert detector.n_detectors == 2
        assert detector.n_energy_bins == 3
        assert "det_0" in detector.detector_names
        assert "det_1" in detector.detector_names

    def test_init_with_response_functions_rf_gsf(self):
        """Инициализация с response_functions=RF_GSF (старый способ)."""
        from bssunfold.constants import RF_GSF
        detector = Detector(response_functions=RF_GSF)
        assert detector.n_detectors > 0
        assert detector.n_energy_bins > 0

    def test_init_error_no_e_mev_in_dict(self):
        """Ошибка при отсутствии ключа 'E_MeV' в словаре."""
        data = {"det1": [1.0, 2.0]}
        with pytest.raises(ValueError, match="must contain 'E_MeV' key"):
            Detector(data)

    def test_init_error_mismatched_lengths(self):
        """Ошибка при несовпадении длин массивов чувствительности."""
        E = np.array([0.1, 0.2, 0.3])
        sens = {"det1": [1.0, 2.0]}  # длина 2 вместо 3
        with pytest.raises(ValueError, match="must match E_MeV length"):
            Detector(E_MeV=E, sensitivities=sens)

    def test_init_error_sensitivities_not_dict_or_array(self):
        """Ошибка при неверном типе sensitivities."""
        E = np.array([0.1, 0.2])
        with pytest.raises(TypeError, match="must be dict or np.ndarray"):
            Detector(E_MeV=E, sensitivities="invalid")

    def test_init_error_less_than_two_energy_bins(self):
        """Ошибка при менее чем 2 энергетических бинах."""
        data = {"E_MeV": [0.1], "det1": [1.0]}
        with pytest.raises(IndexError):
            Detector(data)

    def test_init_error_e_mev_not_1d(self):
        """Ошибка при E_MeV не 1D массиве."""
        data = {"E_MeV": [[0.1, 0.2]], "det1": [1.0, 2.0]}
        # При преобразовании в DataFrame возникает ошибка о несовпадении длин
        with pytest.raises(
            ValueError, match="All arrays must be of the same length"
        ):
            Detector(data)


class TestDetectorUtilities:
    """Тесты вспомогательных методов Detector."""

    def test_get_response_matrix(self, detector, sample_readings):
        """Тест получения матрицы отклика для заданных показаний."""
        A = detector.get_response_matrix(sample_readings)
        assert isinstance(A, np.ndarray)
        assert A.shape == (len(sample_readings), detector.n_energy_bins)

    def test_validate_readings_positive(self, detector):
        """Валидация положительных показаний."""
        readings = {"0in": 1.0, "2in": 2.0}
        validated = detector._validate_readings(readings)
        assert validated == readings

    def test_validate_readings_negative(self, detector):
        """Валидация отрицательных показаний вызывает ошибку."""
        readings = {"0in": -1.0}
        with pytest.raises(ValueError, match="is negative"):
            detector._validate_readings(readings)

    def test_validate_readings_no_detectors(self, detector):
        """Валидация пустого словаря показаний вызывает ошибку."""
        with pytest.raises(ValueError, match="No detector readings provided"):
            detector._validate_readings({})

    def test_build_system(self, detector, sample_readings):
        """Построение системы уравнений."""
        A, b, selected = detector._build_system(sample_readings)
        assert isinstance(A, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert isinstance(selected, list)
        assert A.shape == (len(selected), detector.n_energy_bins)
        assert b.shape == (len(selected),)
        assert set(selected) == set(sample_readings.keys())

    def test_standardize_output(self, detector):
        """Стандартизация выходного словаря."""
        spectrum = np.ones(detector.n_energy_bins) * 0.5
        A = np.random.rand(3, detector.n_energy_bins)
        b = np.random.rand(3)
        selected = ["det1", "det2", "det3"]
        output = detector._standardize_output(spectrum, A, b, selected, "test")
        assert "energy" in output
        assert "spectrum" in output
        assert "effective_readings" in output
        assert "residual" in output
        assert "residual_norm" in output
        assert "method" in output
        assert "doserates" in output
        assert output["method"] == "test"

    def test_save_and_get_result(self, detector, sample_readings):
        """Сохранение и получение результата."""
        result = detector.unfold_cvxpy(sample_readings, save_result=False)
        key = detector._save_result(result)
        assert key in detector.results_history
        retrieved = detector.get_result(key)
        assert retrieved is not None
        assert retrieved["method"] == "cvxpy"
        # Получение текущего результата
        current = detector.get_result()
        assert current is not None
        # Список результатов
        keys = detector.list_results()
        assert key in keys
        # Очистка
        detector.clear_results()
        assert len(detector.results_history) == 0
        assert detector.current_result is None

    def test_plot_response_functions(self, detector):
        """Тест построения графиков (мок)."""
        with patch("matplotlib.pyplot.show") as _:
            detector.plot_response_functions()
            # Проверяем, что show был вызван (или не вызывается,
            # но важно отсутствие ошибок)
            # mock_show.assert_called()  # может не вызываться,
            # если show не используется
            pass

    def test_discretize_spectra_dataframe(self, detector):
        """Дискретизация спектра из DataFrame."""
        df = pd.DataFrame({
            "E_MeV": [1e-9, 1e-8, 1e-7],
            "Phi": [1.0, 0.5, 0.2],
        })
        discretized = detector.discretize_spectra(df)
        assert isinstance(discretized, pd.DataFrame)
        assert "E_MeV" in discretized.columns
        assert "Phi" in discretized.columns
        assert len(discretized) == detector.n_energy_bins

    def test_discretize_spectra_dict(self, detector):
        """Дискретизация спектра из словаря."""
        spectra = {
            "E_MeV": [1e-9, 1e-8, 1e-7],
            "Phi": [1.0, 0.5, 0.2],
        }
        discretized = detector.discretize_spectra(spectra)
        assert isinstance(discretized, pd.DataFrame)
        assert "E_MeV" in discretized.columns
        assert "Phi" in discretized.columns

    def test_get_effective_readings_for_spectra(self, detector):
        """Расчёт эффективных показаний для заданного спектра."""
        spectra = pd.DataFrame({
            "E_MeV": detector.E_MeV,
            "Phi": np.ones(detector.n_energy_bins) * 0.5,
        })
        readings = detector.get_effective_readings_for_spectra(spectra)
        assert isinstance(readings, dict)
        for name in detector.detector_names:
            assert name in readings
            assert readings[name] >= 0

    def test_get_effective_readings_for_spectra_dict(self, detector):
        """Расчёт эффективных показаний для спектра в виде словаря."""
        spectra = {
            "E_MeV": detector.E_MeV.tolist(),
            "Phi": np.ones(detector.n_energy_bins).tolist(),
        }
        readings = detector.get_effective_readings_for_spectra(spectra)
        assert isinstance(readings, dict)
        for name in detector.detector_names:
            assert name in readings


class TestMLEMUnfolding:
    """Тесты метода MLEM развертки."""

    def test_unfold_mlem_odl_basic(self, detector, sample_readings):
        """Базовый тест unfold_mlem_odl."""
        result = detector.unfold_mlem_odl(sample_readings, max_iterations=10)
        assert isinstance(result, dict)
        assert "energy" in result
        assert "spectrum" in result
        assert "method" in result
        assert result["method"] == "MLEM (ODL)"
        assert len(result["energy"]) == detector.n_energy_bins
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_unfold_mlem_odl_with_initial_spectrum(
        self, detector, sample_readings
    ):
        """MLEM с начальным спектром."""
        initial = np.ones(detector.n_energy_bins) * 0.1
        result = detector.unfold_mlem_odl(
            sample_readings,
            initial_spectrum=initial,
            max_iterations=5,
        )
        assert result["method"] == "MLEM (ODL)"

    def test_unfold_mlem_odl_with_errors(self, detector, sample_readings):
        """MLEM с расчетом ошибок (Monte-Carlo)."""
        result = detector.unfold_mlem_odl(
            sample_readings,
            max_iterations=5,
            calculate_errors=True,
            n_montecarlo=5,
            noise_level=0.01,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "montecarlo_samples" in result
        assert result["montecarlo_samples"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
