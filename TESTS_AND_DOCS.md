# Тесты и документация bssunfold

## ✅ Исправления

### 1. Тесты

**Созданные файлы:**
- `tests/test_all.py` — полный набор тестов (60+ тестов)
- `tests/test_refactored_fixed.py` — тесты рефакторинга
- `tests/test_new_methods_fixed.py` — тесты новых методов

**Исправленные проблемы:**
- ❌ Неправильные импорты (`src.bssunfold` → `bssunfold`)
- ❌ Устаревшие пути к модулям
- ❌ Отсутствие тестов для новых unfold_* методов
- ✅ Все тесты используют правильные импорты
- ✅ Добавлены тесты для всех 9 unfold_* методов
- ✅ Добавлены тесты Monte-Carlo uncertainty
- ✅ Добавлены тесты графиков

### 2. Документация Sphinx

**Исправления:**
- ✅ Удалён мусор из `docs/detector.rst` (вывод команд pytest)
- ✅ Исправлено подчёркивание заголовка "Unfold Methods"
- ✅ Создана папка `docs/_static/`
- ✅ Добавлена документация для всех unfold_* методов

## 🚀 Запуск тестов

```bash
# Все тесты
pytest tests/test_all.py -v

# С покрытием
pytest tests/test_all.py -v --cov=src/bssunfold --cov-report=html

# Только новые методы
pytest tests/test_new_methods_fixed.py -v

# Только рефакторинг
pytest tests/test_refactored_fixed.py -v

# Только конкретный тест
pytest tests/test_all.py::TestDetector::test_unfold_cvxpy -v
```

## 📚 Сборка документации

```bash
cd docs
make html

# Открыть документацию
# Linux/Mac: xdg-open _build/html/index.html
# Windows: start _build\html\index.html
```

## 📁 Структура тестов

```
tests/
├── test_all.py                    # Полный набор тестов ⭐
├── test_refactored_fixed.py       # Тесты рефакторинга
├── test_new_methods_fixed.py      # Тесты новых методов
├── test_detector.py               # Тесты Detector класса
├── test_methods2.py               # Дополнительные тесты
├── test_mlem.py                   # Тесты MLEM
└── test_readings.py               # Тесты показаний
```

## 🧪 Покрытие тестов

### Тестируемые методы:

| Метод | Тесты |
|-------|-------|
| `unfold_cvxpy` | ✅ Базовый, с uncertainty |
| `unfold_landweber` | ✅ Базовый, с uncertainty |
| `unfold_mlem` | ✅ Базовый |
| `unfold_qpsolvers` | ✅ Базовый |
| `unfold_doroshenko` | ✅ Базовый, с uncertainty |
| `unfold_kaczmarz` | ✅ Базовый, с omega |
| `unfold_lmfit` | ✅ elastic, lasso, ridge |
| `unfold_mlem_odl` | ✅ Базовый |
| `unfold_combined` | ✅ 2-stage, 3-stage, invalid |

### Тестируемые функции:

| Функция | Тесты |
|---------|-------|
| `solve_doroshenko` | ✅ Прямой вызов |
| `solve_kaczmarz` | ✅ Прямой вызов |
| `solve_lmfit` | ✅ Прямой вызов |

### Тестируемые утилиты:

| Утилита | Тесты |
|---------|-------|
| `plot_response_functions` | ✅ Сохранение в файл |
| `plot_with_uncertainty` | ✅ С uncertainty bands |
| `get_effective_readings_for_spectra` | ✅ Из DataFrame, dict |
| `discretize_spectra` | ✅ Интерполяция |
| `results_history` | ✅ save/get/list/clear |

## 🔧 Конфигурация pytest

В `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

## 🔍 Покрытие кода

Для проверки покрытия:

```bash
pytest tests/test_all.py -v --cov=src/bssunfold --cov-report=term-missing
```

Рекомендуемое покрытие: >80%

## 📊 Документация

Документация включает:

1. **Detector Class** — основной класс
2. **Unfold Methods** — 9 методов восстановления
3. **Core Functions** — 7 функций решения
4. **Regularization Selection** — 5 методов выбора регуляризации

Файлы документации:
- `docs/index.rst` — главная страница
- `docs/detector.rst` — API документация
- `docs/examples.rst` — примеры использования

## ⚠️ Известные ограничения

1. **ODL требуется для `unfold_mlem_odl`** — тесты пропускаются, если ODL не установлен
2. **Графики требуют matplotlib** — используется backend 'Agg' для тестов
3. **Monte-Carlo тесты медленные** — используется n_montecarlo=10 для скорости

## 🎯 Рекомендации

1. **Запускать тесты перед коммитом:**
   ```bash
   pytest tests/test_all.py -v
   ```

2. **Проверять покрытие:**
   ```bash
   pytest tests/test_all.py --cov=src/bssunfold --cov-report=html
   ```

3. **Собирать документацию:**
   ```bash
   cd docs && make html
   ```

4. **Проверять линтером:**
   ```bash
   ruff check src/bssunfold tests
   ```
