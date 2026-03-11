"""
Тестирование метода 'cosine' для выбора регуляризации по максимальному косинусному сходству с initial spectrum.
Проверяем работу unfold_cvxpy и unfold_qpsolvers.
"""
import sys
import numpy as np
import pandas as pd
from bssunfold import Detector, RF_GSF

def test_cosine():
    # Создаем детектор с дефолтными функциями отклика GSF
    df = pd.DataFrame.from_dict(RF_GSF, orient='columns')
    detector = Detector(df)
    
    # Используем readings из базового примера
    readings = {
        "0in": 0.00037707623092440032,
        "2in": 0.0099964357249166195,
        "3in": 0.053668754395163297,
        "5in": 0.18417232269591507,
        "6in": 0.21968230880012038,
        "8in": 0.22007281510471705,
        "10in": 0.17214800127917887,
        "12in": 0.12033147452298382,
        "15in": 0.066744761746482972,
        "18in": 0.03411946184195816,
    }
    
    # Создаем простой initial spectrum как массив numpy той же длины, что и энергетическая сетка детектора
    n_bins = detector.n_energy_bins
    # Простой спектр: пик в средних энергиях
    energies = detector.E_MeV
    # Гауссов пик в логарифмической шкале
    logE = np.log10(energies)
    peak = (logE - np.mean(logE)) / (np.std(logE) * 0.5)
    initial_spectrum_arr = np.exp(-peak**2)
    initial_spectrum_arr = initial_spectrum_arr / np.sum(initial_spectrum_arr)  # нормализуем
    
    print(f"Детектор: {detector}")
    print(f"Количество энергетических бинов: {n_bins}")
    print(f"Длина initial_spectrum: {len(initial_spectrum_arr)}")
    
    # Тестируем с массивом numpy
    print("\nТестирование метода 'cosine' для unfold_cvxpy (массив numpy)...")
    result_cvxpy = detector.unfold_cvxpy(
        readings,
        regularization_method='cosine',
        initial_spectrum=initial_spectrum_arr,
        norm=2,
        calculate_errors=False,
    )
    
    # Проверяем наличие ожидаемых полей
    expected_keys = {'spectrum', 'energy', 'effective_readings', 'doserates',
                     'regularization_method', 'selected_regularization'}
    missing = expected_keys - set(result_cvxpy.keys())
    if missing:
        print(f"Ошибка: в результате отсутствуют ключи: {missing}")
        sys.exit(1)
    
    # Проверяем, что regularization_method == 'cosine'
    assert result_cvxpy['regularization_method'] == 'cosine', \
        f"regularization_method должен быть 'cosine', получен {result_cvxpy['regularization_method']}"
    
    # Проверяем, что selected_regularization находится в диапазоне alphas (1e-9 до 1e2)
    alpha = result_cvxpy['selected_regularization']
    assert 1e-9 <= alpha <= 1e2, \
        f"selected_regularization {alpha} вне диапазона [1e-9, 1e2]"
    
    print(f"  OK: unfold_cvxpy завершен, выбран alpha = {alpha}")
    print(f"  Спектр shape: {result_cvxpy['spectrum'].shape}")
    
    # Тестируем unfold_qpsolvers
    print("\nТестирование метода 'cosine' для unfold_qpsolvers (массив numpy)...")
    result_qp = detector.unfold_qpsolvers(
        readings,
        regularization_method='cosine',
        initial_spectrum=initial_spectrum_arr,
        norm=2,
        calculate_errors=False,
    )
    
    missing = expected_keys - set(result_qp.keys())
    if missing:
        print(f"Ошибка: в результате отсутствуют ключи: {missing}")
        sys.exit(1)
    
    assert result_qp['regularization_method'] == 'cosine', \
        f"regularization_method должен быть 'cosine', получен {result_qp['regularization_method']}"
    
    alpha_qp = result_qp['selected_regularization']
    assert 1e-9 <= alpha_qp <= 1e2, \
        f"selected_regularization {alpha_qp} вне диапазона [1e-9, 1e2]"
    
    print(f"  OK: unfold_qpsolvers завершен, выбран alpha = {alpha_qp}")
    print(f"  Спектр shape: {result_qp['spectrum'].shape}")
    
    # Дополнительная проверка: косинусное сходство между развернутым спектром и initial_spectrum
    # должно быть положительным (но не обязательно максимальным, так как мы не проверяем весь цикл)
    # Вычислим косинусное сходство с помощью внутреннего метода
    cos_cvxpy = detector._cosine_similarity(result_cvxpy['spectrum'], initial_spectrum_arr)
    cos_qp = detector._cosine_similarity(result_qp['spectrum'], initial_spectrum_arr)
    print(f"\nКосинусное сходство (cvxpy): {cos_cvxpy:.6f}")
    print(f"Косинусное сходство (qpsolvers): {cos_qp:.6f}")
    
    # Проверяем, что спектры неотрицательны
    assert np.all(result_cvxpy['spectrum'] >= -1e-10), "Спектр cvxpy содержит отрицательные значения"
    assert np.all(result_qp['spectrum'] >= -1e-10), "Спектр qpsolvers содержит отрицательные значения"
    
    # Тестируем также с dict форматом (словарь с E_MeV и Phi)
    # Создадим словарь с правильными длинами
    print("\nТестирование метода 'cosine' с dict форматом...")
    # Используем ту же энергетическую сетку детектора и спектр
    dict_spectrum = {
        "E_MeV": detector.E_MeV.tolist(),
        "Phi": initial_spectrum_arr.tolist()
    }
    result_cvxpy_dict = detector.unfold_cvxpy(
        readings,
        regularization_method='cosine',
        initial_spectrum=dict_spectrum,
        norm=2,
        calculate_errors=False,
    )
    assert result_cvxpy_dict['regularization_method'] == 'cosine'
    print(f"  OK: dict формат работает, alpha = {result_cvxpy_dict['selected_regularization']:.3e}")
    
    print("\nВсе тесты пройдены успешно!")
    return True

if __name__ == "__main__":
    test_cosine()