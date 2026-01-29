import pytest
import pandas as pd
from src.bssunfold import Detector, RF_GSF  # Импортируем из вашего модуля

# Тестовые данные
EXPECTED_RESULT = {
    "AP": 341.7759518059329,
    "PA": 201.6519529052496,
    "LLAT": 154.24834282828056,
    "RLAT": 134.02570156563294,
    "ROT": 216.31597257105258,
    "ISO": 174.28640630314544,
}


@pytest.fixture
def detector():
    """Фикстура для создания экземпляра Detector"""
    df = pd.DataFrame.from_dict(RF_GSF, orient="columns")
    return Detector(df)


def test_unfold_mlem_odl_returns_correct_doserates(detector):
    """Тест проверяет, что unfold_mlem_odl возвращает правильные значения doserates"""
    # Входные данные
    readings = {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }

    # Выполнение метода
    result_mlem = detector.unfold_mlem_odl(
        readings,
        max_iterations=900,
    )

    # Проверка результата
    assert "doserates" in result_mlem, (
        "Результат должен содержать ключ 'doserates'"
    )

    doserates = result_mlem["doserates"]

    # Проверка ключей
    assert set(doserates.keys()) == set(EXPECTED_RESULT.keys()), (
        f"Ключи doserates не совпадают: {set(doserates.keys())} != {set(EXPECTED_RESULT.keys())}"
    )

    # Проверка значений с определенной точностью
    for key, expected_value in EXPECTED_RESULT.items():
        actual_value = doserates[key]
        assert pytest.approx(actual_value, rel=1e-10) == expected_value, (
            f"Значение для {key}: {actual_value} != {expected_value}"
        )


def test_unfold_mlem_odl_with_different_max_iterations(detector):
    """Тест проверяет работу метода с разным количеством итераций"""
    readings = {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }

    # Проверка с меньшим количеством итераций
    result_short = detector.unfold_mlem_odl(readings, max_iterations=100)
    result_long = detector.unfold_mlem_odl(readings, max_iterations=900)

    assert "doserates" in result_short
    assert "doserates" in result_long

    # Значения могут отличаться при разном количестве итераций,
    # но должны быть того же порядка
    for key in EXPECTED_RESULT.keys():
        assert result_short["doserates"][key] > 0, (
            f"Значение {key} должно быть положительным"
        )
        assert isinstance(result_short["doserates"][key], (int, float))


def test_unfold_mlem_odl_structure(detector):
    """Тест проверяет структуру возвращаемого результата"""
    readings = {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }

    result = detector.unfold_mlem_odl(readings, max_iterations=900)

    # Проверяем наличие ожидаемых ключей в результате
    assert isinstance(result, dict), "Результат должен быть словарем"
    assert "doserates" in result, "Результат должен содержать ключ 'doserates'"

    doserates = result["doserates"]
    assert isinstance(doserates, dict), "'doserates' должен быть словарем"

    # Проверяем все ожидаемые ключи
    expected_keys = ["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]
    for key in expected_keys:
        assert key in doserates, f"Ключ {key} должен присутствовать в doserates"
        assert isinstance(doserates[key], (int, float)), (
            f"Значение для {key} должно быть числом"
        )


# Дополнительные тесты для проверки обработки ошибок
def test_unfold_mlem_odl_with_empty_readings(detector):
    """Тест с пустыми входными данными"""
    with pytest.raises(Exception):
        detector.unfold_mlem_odl({}, max_iterations=900)


def test_unfold_mlem_odl_with_invalid_readings(detector):
    """Тест с некорректными входными данными"""
    with pytest.raises(Exception):
        detector.unfold_mlem_odl({"invalid_key": 1.0}, max_iterations=900)


# Опционально: тест для проверки точности с разными допусками
@pytest.mark.parametrize("tolerance", [1e-1, 1e-3, 1e-5])
def test_unfold_mlem_odl_precision(detector, tolerance):
    """Параметризованный тест для проверки с разной точностью"""
    readings = {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }

    result_mlem = detector.unfold_mlem_odl(readings, max_iterations=900)

    for key, expected_value in EXPECTED_RESULT.items():
        actual_value = result_mlem["doserates"][key]
        assert pytest.approx(actual_value, rel=tolerance) == expected_value, (
            f"Несоответствие при точности {tolerance} для {key}"
        )
