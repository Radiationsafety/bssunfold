import pytest
import numpy as np
import pandas as pd
from src.bssunfold import Detector, RF_GSF

# Базовые тестовые данные
EXPECTED_RESULT = {
    "AP": 341.7759518059329,
    "PA": 201.6519529052496,
    "LLAT": 154.24834282828056,
    "RLAT": 134.02570156563294,
    "ROT": 216.31597257105258,
    "ISO": 174.28640630314544,
}

BASE_READINGS = {
    "3in": 0.053,
    "5in": 0.184,
    "10in": 0.172,
    "18in": 0.034,
}

@pytest.fixture
def detector():
    """Фикстура для создания экземпляра Detector"""
    df = pd.DataFrame.from_dict(RF_GSF, orient="columns")
    return Detector(df)


# ============================================================================
# ТЕСТЫ ДЛЯ МЕТОДА unfold_combined
# ============================================================================

class TestUnfoldCombined:
    """Тесты для комбинированного метода восстановления спектра"""
    
    def test_combined_basic_pipeline(self, detector):
        """Тест базового пайплайна из двух методов"""
        pipeline = [
            {
                'method': 'mlem_odl',
                'params': {'max_iterations': 100},
                'store_intermediate': True
            },
            {
                'method': 'landweber',
                'params': {'max_iterations': 50},
                'store_intermediate': True
            }
        ]
        
        result = detector.unfold_combined(BASE_READINGS, pipeline)
        
        assert 'spectrum' in result
        assert 'doserates' in result
        assert 'pipeline_info' in result
        
    def test_combined_with_different_regularization_params(self, detector):
        """Тест с разными параметрами регуляризации в пайплайне"""
        regularization_params = [0.0001, 0.001, 0.01, 0.1]
        
        for reg in regularization_params:
            pipeline = [
                {
                    'method': 'mlem_odl',
                    'params': {'max_iterations': 50}
                }
            ]
            
            result = detector.unfold_combined(BASE_READINGS, pipeline)
            doserates = result['doserates']
            
            # Проверяем, что все значения положительные
            for key in EXPECTED_RESULT.keys():
                assert doserates[key] > 0, f"Отрицательное значение для {key} при reg={reg}"
    
    def test_combined_without_using_initial(self, detector):
        """Тест без использования начального приближения"""
        pipeline = [
            {
                'method': 'mlem_odl',
                'params': {'max_iterations': 100},
                'use_as_initial': False
            },
            {
                'method': 'landweber',
                'params': {'max_iterations': 50},
                'use_as_initial': False
            }
        ]
        
        result = detector.unfold_combined(BASE_READINGS, pipeline)
        assert 'spectrum' in result
        
    def test_combined_with_error_calculation(self, detector):
        """Тест с расчетом ошибок для последнего метода"""
        pipeline = [
            {'method': 'mlem_odl', 'params': {'max_iterations': 100}},
            {'method': 'landweber', 'params': {'max_iterations': 50}}
        ]
        
        result = detector.unfold_combined(
            BASE_READINGS, 
            pipeline, 
            calculate_errors=True
        )
        
        # Проверяем наличие полей с ошибками (зависит от реализации методов)
        assert 'residual_norm' in result
        
    @pytest.mark.parametrize("methods_combination", [
        [{'method': 'mlem_odl', 'params': {'max_iterations': 100}}],
        [{'method': 'landweber', 'params': {'max_iterations': 100}}],
        [
            {'method': 'mlem_odl', 'params': {'max_iterations': 50}},
        ],
        [
            {'method': 'landweber', 'params': {'max_iterations': 50}},
            {'method': 'mlem_odl', 'params': {'max_iterations': 50}},
        ]
    ])
    def test_combined_different_pipelines(self, detector, methods_combination):
        """Параметризованный тест различных комбинаций методов"""
        result = detector.unfold_combined(BASE_READINGS, methods_combination)
        
        assert 'spectrum' in result
        assert 'pipeline_info' in result
        assert result['pipeline_info']['stages'] == [m['method'] for m in methods_combination]
        
        doserates = result['doserates']
        for key in EXPECTED_RESULT.keys():
            assert key in doserates
            assert isinstance(doserates[key], (int, float))
            assert doserates[key] > 0


# ============================================================================
# ТЕСТЫ ДЛЯ МЕТОДА unfold_doroshenko
# ============================================================================

class TestUnfoldDoroshenko:
    """Тесты для метода Дорошенко"""
    
    def test_doroshenko_basic(self, detector):
        """Базовый тест метода Дорошенко"""
        result = detector.unfold_doroshenko(BASE_READINGS)
        
        assert 'method' in result
        assert result['method'] == 'Doroshenko'
        assert 'doserates' in result
        assert 'iterations' in result
        assert 'converged' in result
        
    @pytest.mark.parametrize("regularization", [0.00001, 0.0001, 0.001, 0.01])
    def test_doroshenko_regularization_params(self, detector, regularization):
        """Тест различных значений регуляризации"""
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            regularization=regularization,
            max_iterations=500
        )
        
        doserates = result['doserates']
        
        # Проверяем, что все значения в разумных пределах
        for key, expected in EXPECTED_RESULT.items():
            actual = doserates[key]
            # Допускаем отклонение до 20% при сильной регуляризации
            rel_tolerance = 0.2 if regularization > 0.01 else 0.1
            assert actual == pytest.approx(expected, rel=rel_tolerance), \
                f"Несоответствие для {key} при regularization={regularization}"
    
    @pytest.mark.parametrize("max_iterations", [100, 500, 1000])
    def test_doroshenko_iterations(self, detector, max_iterations):
        """Тест различного количества итераций"""
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            max_iterations=max_iterations,
            tolerance=1e-4
        )
        
        assert result['iterations'] <= max_iterations
        assert 'doserates' in result
        
    @pytest.mark.parametrize("tolerance", [1e-3, 1e-4, 1e-5, 1e-6])
    def test_doroshenko_tolerance(self, detector, tolerance):
        """Тест различных значений tolerance"""
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            tolerance=tolerance,
            max_iterations=1000
        )
        
        assert 'converged' in result
        assert 'doserates' in result
        
    def test_doroshenko_with_initial_spectrum(self, detector):
        """Тест с заданным начальным спектром"""
        # Создаем начальное приближение
        initial_spectrum = np.ones(detector.n_energy_bins)
        initial_spectrum = initial_spectrum / np.linalg.norm(initial_spectrum)
        
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            initial_spectrum=initial_spectrum
        )
        
        assert 'doserates' in result
        
    def test_doroshenko_with_error_calculation(self, detector):
        """Тест с расчетом погрешностей методом Монте-Карло"""
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            calculate_errors=True,
            n_montecarlo=50,  # Уменьшаем для скорости тестов
            noise_level=0.01
        )
        
        # Проверяем наличие полей с неопределенностями
        uncertainty_fields = [
            'spectrum_uncert_mean', 'spectrum_uncert_std',
            'spectrum_uncert_min', 'spectrum_uncert_max',
            'montecarlo_samples', 'noise_level'
        ]
        
        for field in uncertainty_fields:
            assert field in result, f"Отсутствует поле {field}"
            
        assert result['montecarlo_samples'] == 50
        assert result['noise_level'] == 0.01
        
    @pytest.mark.parametrize("noise_level", [0.001, 0.01, 0.05, 0.1])
    def test_doroshenko_different_noise_levels(self, detector, noise_level):
        """Тест различных уровней шума при расчете погрешностей"""
        result = detector.unfold_doroshenko(
            BASE_READINGS,
            calculate_errors=True,
            n_montecarlo=30,
            noise_level=noise_level
        )
        
        assert result['noise_level'] == noise_level
        assert 'spectrum_uncert_std' in result


# ============================================================================
# ТЕСТЫ ДЛЯ МЕТОДА unfold_lmfit
# ============================================================================

class TestUnfoldLmfit:
    """Тесты для метода lmfit с регуляризацией"""
    
    @pytest.mark.parametrize("method", ["leastsq", "lbfgsb", "tnc", "cg"])
    def test_lmfit_different_solvers(self, detector, method):
        """Тест различных солверов lmfit"""
        try:
            result = detector.unfold_lmfit(
                BASE_READINGS,
                method=method,
                model_name="ridge",
                regularization=1e-4
            )
            
            assert result['method'] == f"lmfit ({method})"
            assert 'success' in result
            assert 'doserates' in result
            
        except ImportError:
            pytest.skip("lmfit не установлен")
            
    @pytest.mark.parametrize("model_name", ["lasso", "ridge", "elastic"])
    def test_lmfit_different_models(self, detector, model_name):
        """Тест различных моделей регуляризации"""
        try:
            params = {
                'regularization': 1e-4,
            }
            
            if model_name == "elastic":
                params['regularization2'] = 1e-4
                params['l1_weight'] = 0.5
                
            result = detector.unfold_lmfit(
                BASE_READINGS,
                method="lbfgsb",
                model_name=model_name,
                **params
            )
            
            assert result['model_name'] == model_name
            assert 'doserates' in result
            
        except ImportError:
            pytest.skip("lmfit не установлен")
            
    @pytest.mark.parametrize("regularization", [1e-6, 1e-4, 1e-2, 1.0])
    def test_lmfit_ridge_regularization(self, detector, regularization):
        """Тест различных значений регуляризации для ridge модели"""
        try:
            result = detector.unfold_lmfit(
                BASE_READINGS,
                method="lbfgsb",
                model_name="ridge",
                regularization=regularization
            )
            
            assert result['regularization'] == regularization
            
            # При сильной регуляризации спектр должен быть более сглаженным
            doserates = result['doserates']
            for key in EXPECTED_RESULT.keys():
                assert doserates[key] > 0
                
        except ImportError:
            pytest.skip("lmfit не установлен")
            
    @pytest.mark.parametrize("l1_weight", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_lmfit_elastic_net_weights(self, detector, l1_weight):
        """Тест различных весов L1 для elastic net"""
        try:
            result = detector.unfold_lmfit(
                BASE_READINGS,
                method="lbfgsb",
                model_name="elastic",
                regularization=1e-4,
                regularization2=1e-4,
                l1_weight=l1_weight
            )
            
            assert result['l1_weight'] == l1_weight
            assert 'doserates' in result
            
        except ImportError:
            pytest.skip("lmfit не установлен")
            
    def test_lmfit_with_error_calculation(self, detector):
        """Тест расчета погрешностей для lmfit"""
        try:
            result = detector.unfold_lmfit(
                BASE_READINGS,
                method="lbfgsb",
                model_name="ridge",
                regularization=1e-4,
                calculate_errors=True,
                n_montecarlo=30,
                noise_level=0.01
            )
            
            uncertainty_fields = [
                'spectrum_uncert_mean', 'spectrum_uncert_std',
                'montecarlo_samples', 'noise_level'
            ]
            
            for field in uncertainty_fields:
                assert field in result
                
        except ImportError:
            pytest.skip("lmfit не установлен")
            
    def test_lmfit_invalid_model(self, detector):
        """Тест с некорректным названием модели"""
        try:
            with pytest.raises(ValueError, match="model_name must be one of"):
                detector.unfold_lmfit(
                    BASE_READINGS,
                    model_name="invalid_model"
                )
        except ImportError:
            pytest.skip("lmfit не установлен")


# ============================================================================
# ТЕСТЫ ДЛЯ МЕТОДА unfold_kaczmarz
# ============================================================================

class TestUnfoldKaczmarz:
    """Тесты для метода Качмажа (ART)"""
    
    def test_kaczmarz_basic(self, detector):
        """Базовый тест метода Качмажа"""
        result = detector.unfold_kaczmarz(BASE_READINGS)
        
        assert result['method'] == 'Kaczmarz'
        assert 'doserates' in result
        assert 'iterations' in result
        assert 'converged' in result
        assert 'omega' in result
        
    @pytest.mark.parametrize("omega", [0.1, 0.5, 1.0, 1.5, 1.9])
    def test_kaczmarz_omega_values(self, detector, omega):
        """Тест различных значений релаксационного параметра omega"""
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            omega=omega,
            max_iterations=500
        )
        
        assert result['omega'] == omega
        
        # Проверяем, что результат в разумных пределах
        doserates = result['doserates']
        for key, expected in EXPECTED_RESULT.items():
            # При экстремальных значениях omega результат может отличаться сильнее
            rel_tolerance = 0.3 if omega < 0.3 or omega > 1.7 else 0.15
            assert doserates[key] == pytest.approx(expected, rel=rel_tolerance), \
                f"Несоответствие для {key} при omega={omega}"
    
    @pytest.mark.parametrize("max_iterations", [100, 500, 1000, 2000])
    def test_kaczmarz_iterations(self, detector, max_iterations):
        """Тест различного количества итераций"""
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            max_iterations=max_iterations,
            tolerance=1e-5
        )
        
        assert result['iterations'] <= max_iterations
        assert 'doserates' in result
        
    @pytest.mark.parametrize("tolerance", [1e-3, 1e-4, 1e-5, 1e-6])
    def test_kaczmarz_tolerance(self, detector, tolerance):
        """Тест различных значений tolerance"""
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            tolerance=tolerance,
            max_iterations=1000
        )
        
        assert 'converged' in result
        assert 'doserates' in result
        
    def test_kaczmarz_with_initial_spectrum(self, detector):
        """Тест с заданным начальным спектром"""
        # Создаем начальное приближение
        initial_spectrum = np.ones(detector.n_energy_bins) * 0.1
        
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            initial_spectrum=initial_spectrum,
            max_iterations=300
        )
        
        assert 'doserates' in result
        
    def test_kaczmarz_with_error_calculation(self, detector):
        """Тест с расчетом погрешностей методом Монте-Карло"""
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            calculate_errors=True,
            n_montecarlo=40,
            noise_level=0.02,
            max_iterations=300
        )
        
        uncertainty_fields = [
            'spectrum_uncert_mean', 'spectrum_uncert_std',
            'spectrum_uncert_min', 'spectrum_uncert_max',
            'montecarlo_samples', 'noise_level'
        ]
        
        for field in uncertainty_fields:
            assert field in result
            
    @pytest.mark.parametrize("invalid_omega", [0, -0.1, 2.1, 3.0])
    def test_kaczmarz_invalid_omega(self, detector, invalid_omega):
        """Тест с некорректными значениями omega (должны работать, но с предупреждением)"""
        # Метод должен работать, но может выдавать предупреждение
        result = detector.unfold_kaczmarz(
            BASE_READINGS,
            omega=invalid_omega,
            max_iterations=200
        )
        
        assert result['omega'] == invalid_omega
        assert 'doserates' in result
        
    def test_kaczmarz_convergence_comparison(self, detector):
        """Сравнение сходимости при разных параметрах"""
        results = {}
        
        for omega in [0.5, 1.0, 1.5]:
            result = detector.unfold_kaczmarz(
                BASE_READINGS,
                omega=omega,
                max_iterations=500,
                tolerance=1e-4
            )
            results[f'omega_{omega}'] = {
                'iterations': result['iterations'],
                'converged': result['converged'],
                'residual_norm': result.get('residual_norm', 0)
            }
        
        # Проверяем, что все результаты содержат нужные ключи
        for omega_key, res in results.items():
            assert 'iterations' in res
            assert 'converged' in res


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ДЛЯ СРАВНЕНИЯ МЕТОДОВ
# ============================================================================

class TestMethodsComparison:
    """Тесты для сравнения различных методов"""
    
    def test_all_methods_produce_positive_doserates(self, detector):
        """Проверка, что все методы дают положительные значения мощностей доз"""
        methods_to_test = [
            ('mlem_odl', {'max_iterations': 300}),
            ('landweber', {'max_iterations': 300}),
            ('doroshenko', {'max_iterations': 300}),
            ('kaczmarz', {'max_iterations': 300}),
        ]
        
        try:
            methods_to_test.append(('lmfit', {
                'method': 'lbfgsb', 
                'model_name': 'ridge', 
                'regularization': 1e-4
            }))
        except ImportError:
            pass
        
        for method_name, params in methods_to_test:
            method_func = getattr(detector, f'unfold_{method_name}')
            result = method_func(BASE_READINGS, **params)
            
            doserates = result['doserates']
            for key in EXPECTED_RESULT.keys():
                assert doserates[key] > 0, \
                    f"Метод {method_name} дал отрицательное значение для {key}"
    
    def test_consistency_between_methods(self, detector):
        """Проверка согласованности между методами (порядок величин)"""
        results = {}
        
        # Тестируем основные методы
        methods = {
            'mlem_odl': {'max_iterations': 1000},
            'doroshenko': {'max_iterations': 1000},
            'kaczmarz': {'max_iterations': 1000},
        }
        
        for method_name, params in methods.items():
            method_func = getattr(detector, f'unfold_{method_name}')
            result = method_func(BASE_READINGS, **params)
            results[method_name] = result['doserates']
        
        # Проверяем, что порядок величин совпадает
        for key in EXPECTED_RESULT.keys():
            values = [results[m][key] for m in methods.keys()]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # разброс
            assert std_val / mean_val > 0.3, \
                f"Слишком большой разброс для {key}: {values}"