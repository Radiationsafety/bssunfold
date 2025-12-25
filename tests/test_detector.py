import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from bssunfold import Detector


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
            regularization=1e-2  # Используем большую регуляризацию для стабильности
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
    
    def test_unfold_landweber_with_initial_spectrum(self, detector, sample_readings):
        """Тест unfold_landweber с начальным спектром."""
        initial_spectrum = np.ones(detector.n_energy_bins) * 0.1
        result = detector.unfold_landweber(
            sample_readings, 
            initial_spectrum=initial_spectrum,
            max_iterations=50
        )
        assert result["method"] == "Landweber"
    
    def test_unfold_landweber_with_errors(self, detector, sample_readings):
        """Тест unfold_landweber с расчетом ошибок."""
        result = detector.unfold_landweber(
            sample_readings,
            max_iterations=50,
            calculate_errors=True,
            n_montecarlo=10  # Используем мало итераций для скорости
        )
        
        # Проверяем наличие полей с ошибками
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result
        assert "montecarlo_samples" in result
        assert result["montecarlo_samples"] == 10
    
    def test_unfold_landweber_invalid_initial_spectrum(self, detector, sample_readings):
        """Тест unfold_landweber с некорректным начальным спектром."""
        initial_spectrum = np.ones(10)  # Неправильный размер
        with pytest.raises(ValueError, match="must match number of energy bins"):
            detector.unfold_landweber(
                sample_readings, 
                initial_spectrum=initial_spectrum
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
        
        # Мокаем коэффициенты ICRP-116
        with patch.object(detector, '_load_icrp116_coefficients', return_value={
            'E_MeV': detector.E_MeV,
            'AP': np.ones_like(detector.E_MeV) * 1e-9,
            'PA': np.ones_like(detector.E_MeV) * 2e-9,
            'ISO': np.ones_like(detector.E_MeV) * 3e-9
        }):
            doserates = detector._calculate_doserates(test_spectrum)
            
            assert isinstance(doserates, dict)
            assert 'AP' in doserates
            assert 'PA' in doserates
            assert 'ISO' in doserates
            
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])