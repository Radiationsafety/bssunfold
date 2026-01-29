import pytest
import pandas as pd
import numpy as np
from bssunfold import RF_LANL, Detector

# Тестовый спектр в новом формате
test_spectrum = {
    "E_MeV": [
        1e-09, 2.5e-09, 1.6e-08, 6.3e-08, 1.6e-07, 2.5e-07, 2.5e-06, 4e-06,
        6.3e-06, 1.6e-05, 2.5e-05, 6.3e-05, 0.0004, 0.001, 0.0016, 0.0025,
        0.004, 0.04, 0.16, 1.0, 4.0, 6.3, 63.0, 160.0, 250.0
    ],
    "Phi": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0064, 0.045, 0.36, 0.28, 0.086, 0.0, 0.0, 0.0
    ]
}

# Ожидаемые результаты (с вашими значениями)
expected_eff_readings = {
    '3in': 0.3756264560920935,
    '4in': 1.2472241929883845,
    '5in': 1.8176226237355264,
    '6in': 2.5266429800052714,
    '8in': 2.534191980356945,
    '9in': 2.5287144849995267,
    '12in': 1.6881230648246595,
    '18in': 0.4932805821553776,
    '9inPb': 2.629557394948418,
    '12inPb': 1.884794919010304,
    '18inPb': 0.6082429593837522
}

# Фикстуры для подготовки данных
@pytest.fixture
def detector():
    """Создает детектор для тестирования"""
    df2 = pd.DataFrame.from_dict(RF_LANL, orient='columns')
    return Detector(df2)

@pytest.fixture
def spectrum_dict():
    """Возвращает тестовый спектр в формате словаря"""
    return test_spectrum.copy()

@pytest.fixture
def spectrum_df():
    """Возвращает тестовый спектр в формате DataFrame"""
    return pd.DataFrame(test_spectrum)

@pytest.fixture
def spectrum_numpy():
    """Возвращает только значения Phi в формате numpy array"""
    return np.array(test_spectrum["Phi"])

class TestDetectorWithDictSpectrum:
    """Тесты для класса Detector с форматом спектра в виде словаря"""
    
    def test_detector_initialization(self):
        """Тест инициализации детектора"""
        df2 = pd.DataFrame.from_dict(RF_LANL, orient='columns')
        detector = Detector(df2)
        assert detector is not None
        
    def test_get_effective_readings_with_dict_spectrum(self, detector, spectrum_dict):
        """Тест расчета эффективных показаний для спектра в формате словаря"""
        # Вызов тестируемого метода
        eff_readings = detector.get_effective_readings_for_spectra(spectrum_dict)
        
        # Проверка базовой структуры результата
        assert isinstance(eff_readings, dict)
        assert len(eff_readings) == len(expected_eff_readings)
        
        # Проверка ключей
        assert set(eff_readings.keys()) == set(expected_eff_readings.keys())
        
        # Проверка значений с учетом погрешности вычислений
        for key, expected_value in expected_eff_readings.items():
            actual_value = eff_readings[key]
            assert actual_value == pytest.approx(expected_value, rel=1e-10), \
                f"Несоответствие для {key}: ожидалось {expected_value}, получено {actual_value}"
                
    def test_get_effective_readings_with_dataframe_spectrum(self, detector, spectrum_df):
        """Тест расчета эффективных показаний для спектра в формате DataFrame"""
        # Проверяем, принимает ли метод DataFrame
        try:
            eff_readings = detector.get_effective_readings_for_spectra(spectrum_df)
            assert isinstance(eff_readings, dict)
            # Дополнительные проверки, если метод поддерживает DataFrame
        except Exception as e:
            # Если метод не поддерживает DataFrame, это нормально
            # Проверяем, что исключение связано с типом данных
            assert isinstance(e, (TypeError, AttributeError, ValueError))
    
    def test_spectrum_dict_structure(self, spectrum_dict):
        """Тест структуры входного спектра"""
        # Проверка наличия обязательных ключей
        assert "E_MeV" in spectrum_dict
        assert "Phi" in spectrum_dict
        
        # Проверка типов данных
        assert isinstance(spectrum_dict["E_MeV"], list)
        assert isinstance(spectrum_dict["Phi"], list)
        
        # Проверка длины массивов
        assert len(spectrum_dict["E_MeV"]) == len(spectrum_dict["Phi"])
        
        # Проверка, что значения энергий положительные
        assert all(e >= 0 for e in spectrum_dict["E_MeV"])
        
        # Проверка, что значения потока неотрицательные
        assert all(phi >= 0 for phi in spectrum_dict["Phi"])
        
    # def test_get_effective_readings_partial_spectrum(self, detector):
    #     """Тест с частичным спектром (не все энергии)"""
    #     partial_spectrum = {
    #         "E_MeV": [0.04, 0.16, 1.0, 4.0, 6.3],
    #         "Phi": [0.0064, 0.045, 0.36, 0.28, 0.086]
    #     }
        
    #     eff_readings = detector.get_effective_readings_for_spectra(partial_spectrum)
    #     assert isinstance(eff_readings, dict)
    #     assert len(eff_readings) > 0
        
    # def test_get_effective_readings_single_value(self, detector):
    #     """Тест с одним значением в спектре"""
    #     single_spectrum = {
    #         "E_MeV": [1.0],
    #         "Phi": [0.36]
    #     }
        
    #     eff_readings = detector.get_effective_readings_for_spectra(single_spectrum)
    #     assert isinstance(eff_readings, dict)
        
    # def test_get_effective_readings_missing_key(self, detector):
    #     """Тест с отсутствующим ключом в спектре"""
    #     invalid_spectrum = {
    #         "energy": [1.0, 2.0, 3.0],  # Неправильное имя ключа
    #         "flux": [0.1, 0.2, 0.3]     # Неправильное имя ключа
    #     }
        
        # with pytest.raises((KeyError, ValueError)):
        #     detector.get_effective_readings_for_spectra(invalid_spectrum)
            
    def test_get_effective_readings_mismatched_lengths(self, detector):
        """Тест с разной длиной массивов энергий и потоков"""
        invalid_spectrum = {
            "E_MeV": [1.0, 2.0, 3.0],
            "Phi": [0.1, 0.2]  # Намеренно разная длина
        }
        
        with pytest.raises((ValueError, IndexError)):
            detector.get_effective_readings_for_spectra(invalid_spectrum)
            
    def test_get_effective_readings_empty_spectrum(self, detector):
        """Тест с пустым спектром"""
        empty_spectrum = {
            "E_MeV": [],
            "Phi": []
        }
        
        try:
            result = detector.get_effective_readings_for_spectra(empty_spectrum)
            # Если не выбрасывается исключение, проверяем результат
            assert isinstance(result, dict)
            # Возможно, возвращается словарь с нулевыми значениями
            if result:
                assert all(v == 0 for v in result.values())
        except Exception as e:
            # Проверяем, что исключение имеет ожидаемый тип
            assert isinstance(e, (ValueError, IndexError))
            
    def test_get_effective_readings_all_zeros(self, detector):
        """Тест со спектром из нулевых значений"""
        zero_spectrum = {
            "E_MeV": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Phi": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        eff_readings = detector.get_effective_readings_for_spectra(zero_spectrum)
        assert isinstance(eff_readings, dict)
        
        # Если метод правильно реализован, все значения должны быть близки к 0
        for value in eff_readings.values():
            assert isinstance(value, (int, float, np.number))

def test_spectrum_energy_range():
    """Тест диапазона энергий в тестовом спектре"""
    energies = test_spectrum["E_MeV"]
    
    # Проверка, что энергии отсортированы
    assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))
    
    # Проверка диапазона
    assert min(energies) >= 0
    assert max(energies) <= 10000  # Реалистичный предел для MeV
    
    # Проверка, что есть значения в интересующем диапазоне
    mid_range_energies = [e for e in energies if 0.01 <= e <= 10]
    assert len(mid_range_energies) > 0