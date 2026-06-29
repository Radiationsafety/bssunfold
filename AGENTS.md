# AGENTS.md — bssunfold

## Commands

```bash
uv sync --group dev          # install all deps including dev
uv run pytest tests/         # run all 391 tests
uv run pytest -v --tb=short  # verbose, short traceback
uv run pytest tests/test_coverage.py  # primary coverage test file
uv run pytest --cov=src/bssunfold --cov-report=term-missing --cov-fail-under=95
uv run ruff check src/ tests/
uv run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

Run a single test: `uv run pytest tests/test_coverage.py::TestClass::test_name -v`

## Package layout

`src/bssunfold/` — all source code, installed as editable (`pip install -e .`).

- `core/detector.py` — main `Detector` class (245 stmts). The public API entry point.
- `core/regularization.py` — L-curve, GCV, discrepancy principle, cosine similarity (pytikhonov wrappers + fallbacks)
- `core/unfolding_methods.py` — all `solve_*` functions, ~222 stmts
- `core/unfold_qpsolvers.py`, `unfold_cvxpy.py`, `unfold_landweber.py`, `unfold_mlem.py`, `unfold_mlem_odl.py`, `unfold_doroshenko.py`, `unfold_kaczmarz.py`, `unfold_lmfit.py`, `unfold_combined.py` — one file per unfolding algorithm
- `core/_matrix_utils.py` — SVD, derivative matrix, tikhonov system building
- `core/_base_unfolder.py`, `core/_montecarlo.py` — internal base class and Monte Carlo uncertainty
- `utils/converters.py`, `interpolation.py`, `plotting.py`, `validators.py` — utility functions
- `platform_check.py` — OS detection, solver availability checks
- `constants.py` — ICRP116 dose coefficients, default response function data
- `logging_config.py` — logger setup

## Testing

### Test files (9 files, 391 tests)

| File | Focus |
|------|-------|
| `tests/test_coverage.py` | Primary coverage file: edge cases, error injection, fallbacks, ~205 tests |
| `tests/test_detector.py` | Detector class basics |
| `tests/test_all.py` | Broad test set |
| `tests/test_methods2.py` | Additional method tests |
| `tests/test_mlem.py` | MLEM-specific tests |
| `tests/test_readings.py` | Readings/effective readings tests |
| `tests/test_refactored_fixed.py` | Post-refactoring tests |
| `tests/test_new_methods_fixed.py` | New unfold_* method tests |

### Coverage quirks

- **pytikhonov IS installed** in CI/dev. To test fallback paths (except ImportError in `regularization.py`), patch `builtins.__import__` to raise `ImportError` only for `'pytikhonov'`. Simply popping from `sys.modules` does NOT work (reimport succeeds). Use:
  ```python
  import builtins
  orig = builtins.__import__
  def mock(name, *a, **kw):
      if name == 'pytikhonov': raise ImportError
      return orig(name, *a, **kw)
  with patch('builtins.__import__', side_effect=mock):
      ...
  ```
- **proxsuite IS installed on Unix**, so `check_proxsuite_availability()` returns `True`. To test the osqp path in `get_recommended_solver()`, also mock `check_proxsuite_availability`.
- **qpsolvers._solve_qp** is imported inside function bodies (`from qpsolvers import solve_qp`). Patch at `'qpsolvers.solve_qp'`, NOT at the module attribute.
- **cvxpy Problem** eagerly evaluates matrix expressions at construction. Patch before building the Problem.
- **Monte Carlo tests** use `n_montecarlo=10` for speed.
- **Plot tests** use matplotlib `'Agg'` backend.
- **ODL-dependent tests** (MLEM) are skipped if ODL not installed.

### Mocks for `platform_check` tests

Patch module-level attributes on the installed package path (e.g., `bssunfold.platform_check.PROXSUITE_AVAILABLE`), not the local file path.

## Python 3.15

- **Python 3.15** (3.15.0b3, Jun 2026): fully compatible — all 1123 tests pass (with all optional deps: odl, lmfit, qpsolvers, pytikhonov), ruff/flake8 clean.
- No binary wheels exist yet, so building from source is required. System -dev packages or extracted equivalents are needed for compilation of numpy, scipy, pandas, cvxpy, etc.
- The `requires-python = ">=3.11"` constraint remains unchanged.
- Classifier `"Programming Language :: Python :: 3.15"` has been added.

## Architecture

- `bssunfold.Detector` is the only public class. Configure with response functions DataFrame or CSV.
- All `unfold_*` methods return a dict with `'spectrum'`, `'doserates'`, `'readings'`, optional `'spectrum_uncert_mean'`.
- Regularization parameter selection delegates to `pytikhonov` when available, with pure-numpy fallbacks.
- Response functions follow a standard CSV format: column `E_MeV` + one column per detector sphere.
