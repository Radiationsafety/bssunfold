# Рефакторинг пакета bssunfold

## Обзор изменений

Этот документ описывает изменения, внесенные в ходе рефакторинга пакета bssunfold для улучшения модульности, поддерживаемости и кроссплатформенной совместимости.

## Цели рефакторинга

1. **Разделение ответственности** - код разделен на логические модули по принципу Single Responsibility Principle
2. **Обратная совместимость** - все публичные API сохранены, старый код продолжает работать без изменений
3. **Кроссплатформенность** - удалены жесткие зависимости от jaxlib и proxsuite, которые недоступны на Windows
4. **Улучшенная архитектура** - код организован по принципу Open/Closed для легкого добавления новых функций

## Новая структура пакета

```
src/bssunfold/
├── __init__.py              # Публичный API (обратная совместимость)
├── constants.py             # Константы (ICRP116, RF_GSF, RF_PTB, RF_LANL)
├── logging_config.py        # Настройка логирования
├── platform_check.py        # Проверка платформы и условные импорты
├── core/
│   ├── __init__.py
│   ├── detector.py          # Класс Detector (основная логика)
│   ├── unfolding_methods.py # Методы развертки (cvxpy, landweber, mlem, qpsolvers)
│   ├── regularization.py    # Выбор параметра регуляризации
│   └── dose_calculation.py  # Расчет дозовых ставок
├── utils/
│   ├── __init__.py
│   ├── validators.py        # Валидация входных данных
│   ├── converters.py        # Конвертация форматов данных
│   ├── interpolation.py     # Интерполяция спектров
│   └── plotting.py          # Функции визуализации
└── legacy/
    └── bridge.py            # Мост для обратной совместимости (если нужен)
```

## Изменения в зависимостях

### Удаленные зависимости
- `jax` - удален из обязательных зависимостей
- `jaxlib` - удален из обязательных зависимостей (недоступен на Windows)
- `proxsuite` - перемещен в опциональные зависимости

### Опциональные зависимости

Для установки с полным набором солверов (только Unix/Linux/Mac):
```bash
pip install bssunfold[all-solvers]
```

Для Windows рекомендуется:
```bash
pip install bssunfold[windows]
```

### Рекомендуемые солверы по платформе

| Платформа | Рекомендуемый солвер | Альтернативы |
|-----------|---------------------|--------------|
| Windows   | `osqp`              | `ecos`, `piqp` |
| Linux     | `proxqp`            | `osqp`, `ecos` |
| macOS     | `proxqp`            | `osqp`, `ecos` |

## Обратная совместимость

Все публичные функции и классы доступны через старые пути импорта:

```python
# Старый стиль (продолжает работать)
from bssunfold import Detector
from bssunfold import ICRP116_COEFF_EFFECTIVE_DOSE, RF_GSF

detector = Detector()
result = detector.unfold_cvxpy(readings)
```

## Новые возможности

### 1. Проверка доступности солверов

```python
from bssunfold import get_available_solvers, get_recommended_solver

# Получить все доступные солверы
solvers = get_available_solvers()
print(solvers)  # {'ecos': True, 'osqp': True, 'proxqp': False, ...}

# Получить рекомендуемый солвер для текущей платформы
solver = get_recommended_solver()
print(solver)  # 'osqp' на Windows, 'proxqp' на Linux
```

### 2. Настройка логирования

```python
from bssunfold import setup_logging
import logging

# Настроить логирование
setup_logging(level=logging.INFO)
```

### 3. Модульные утилиты

```python
from bssunfold.utils import (
    validate_readings,
    validate_energy_grid,
    interpolate_spectrum,
    convert_to_dataframe,
)

# Валидация данных
valid_readings = validate_readings(readings, detector.detector_names)

# Интерполяция спектра
new_spectrum = interpolate_spectrum(spectrum, E_old, E_new)
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Тесты рефакторинга
pytest tests/test_refactored.py -v

# Тесты с покрытием
pytest tests/ --cov=bssunfold --cov-report=html
```

### Проверка обратной совместимости

Тесты в `tests/test_refactored.py` проверяют:
1. Импорты старых API работают
2. Результаты методов совпадают с ожидаемыми
3. На Windows не импортируются jaxlib/proxsuite

## Миграция старого кода

### Что нужно изменить

**Ничего!** Старый код продолжает работать без изменений.

### Рекомендуемые обновления

Для использования новых возможностей:

```python
# Было
from bssunfold import Detector
detector = Detector()

# Стало (можно использовать новые утилиты)
from bssunfold import Detector, get_recommended_solver
from bssunfold.utils import validate_readings

detector = Detector()
readings = validate_readings(raw_readings, detector.detector_names)
solver = get_recommended_solver()
result = detector.unfold_cvxpy(readings, solver=solver)
```

## Обработка платформенных различий

### Windows

На Windows пакет автоматически:
- Показывает предупреждение о недоступности proxqp
- Использует fallback солверы (osqp, ecos, piqp)
- Не пытается импортировать jaxlib/proxsuite

### Linux/Mac

На Unix-системах:
- Доступны все солверы включая proxqp
- Рекомендуется использовать proxqp для лучшей производительности

## Известные ограничения

1. **pytikhonov** - требуется для автоматического выбора параметра регуляризации. Если не установлен, используются fallback реализации.

2. **proxsuite** - недоступен на Windows. Используйте `solver='osqp'` или `solver='piqp'` как альтернативу.

3. **jax/jaxlib** - опциональны. Используются только через qpsolvers.

## Будущие улучшения

1. Добавить больше unittest для покрытия всех методов
2. Интеграция с CI/CD для тестирования на разных платформах
3. Добавить type hints для всех публичных функций
4. Улучшить документацию в формате Google/NumPy docstrings

## Авторы рефакторинга

Рефакторинг выполнен с сохранением 100% обратной совместимости и улучшением архитектуры пакета.
