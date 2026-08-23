# AGENTS.md — bssunfold

## Commands

```bash
uv sync --group dev          # install all deps including dev
uv run pytest tests/         # run all 1404 tests
uv run pytest -v --tb=short  # verbose, short traceback
uv run pytest tests/test_coverage.py  # primary coverage test file
uv run pytest --cov=src/bssunfold --cov-report=term-missing --cov-fail-under=95
uv run ruff check src/ tests/
uv run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
uv run bandit -r src/bssunfold                 # static security scan (src is clean)
uv run pip-audit --skip-editable -s osv -f json  # dependency vuln scan (needs network; -s osv dedupes)
uv run python tools/run_dynapyt.py             # DynaPyt dynamic analysis (BranchCoverage over a fast subset)
```

Run a single test: `uv run pytest tests/test_coverage.py::TestClass::test_name -v`

## Package layout

`src/bssunfold/` — all source code, installed as editable (`pip install -e .`).

- `core/detector.py` — main `Detector` class (245 stmts). The public API entry point.
- `core/regularization.py` — L-curve, GCV, discrepancy principle, cosine similarity (pytikhonov wrappers + fallbacks)
- `core/unfolding_methods.py` — all `solve_*` functions, ~222 stmts
- `core/unfold_qpsolvers.py`, `unfold_cvxpy.py`, `unfold_landweber.py`, `unfold_mlem.py`, `unfold_mlem_odl.py`, `unfold_doroshenko.py`, `unfold_kaczmarz.py`, `unfold_lmfit.py`, `unfold_smt.py`, `unfold_combined.py`, `unfold_ferdor.py`, `unfold_rebunki.py`, `unfold_nsduaz.py` — one file per unfolding algorithm (unfold_smt.py is a port of the Haskell/SBV `linearEqSolver`, backed by optional z3-solver)
- `core/unfold_interpret.py` — public `interpret_qp`/`unfold_interpret` entry points for pyoptexplain-based interpretation of the unfolding QP (solve + robustness, shadow prices, detector sensitivity, regularization sweep, scenarios). Backed by `core/_interpret_pyopt.py` (QP build/solve + perturbation analyses, leaf) and `core/_interpret_report.py` (result dataclass + markdown report + metrics, leaf). Optional dep: `bssunfold[interpret]`
- `core/unfold_cascade.py`, `core/unfold_composite.py` — multi-method pipeline: sequential coarse-to-fine cascade (with optional `multi_resolution` coarse pre-solve) and adaptive ensemble (stacked generalization). Both wired as `Detector.unfold_cascade` / `Detector.unfold_composite`.
- `core/_multires.py` — shared coarse-to-fine helpers (`build_coarse_detector`, `prolongate_spectrum`, `_coarsen_columns`, `_split_coarse`), extracted from `unfold_genetic.py`.
- `core/unfold_parametric.py` — FRUIT solver backends (`solve_parametric_cvxpy`/`qpsolvers`/`combined`) + public `unfold_parametric`; model/fit lives in `core/_fruit.py`
- `core/unfold_parametric2.py` — public BON95 `solve_parametric2`/`unfold_parametric2`; family logic lives in `core/_bon95.py`
- `core/_matrix_utils.py` — SVD, derivative matrix, tikhonov system building
- `core/_base_unfolder.py`, `core/_montecarlo.py` — internal base class and Monte Carlo uncertainty
- `core/_solver_backends.py` — shared solver-backend resolution (`_parse_solver_backend`, `_resolve_cvxpy_solvers`, `_resolve_qpsolver_name`)
- `core/_parametric_shared.py` — shared constants/fit helpers for the parametric family, incl. the canonical `_check_fit_quality` (leaf, no cyclic deps)
- `core/_bon95.py` — BON95 parametric family (extracted from `unfold_parametric2.py`)
- `core/_fruit.py` — FRUIT parametric model + NLS fit (extracted from `unfold_parametric.py`)
- `utils/converters.py`, `interpolation.py`, `plotting.py`, `validators.py` — utility functions
- `platform_check.py` — OS detection, solver availability checks
- `constants.py` — ICRP116 dose coefficients, default response function data
- `logging_config.py` — logger setup

## Testing

### Test files (43 files, ~1686 tests)

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
| `tests/test_smt.py` | SMT-based unfolding: exact solvers + solve_smt/unfold_smt (skipped if z3-solver not installed) |
| `tests/test_interpret.py` | pyoptexplain interpretation: build_interpretation_qp/solve_interpret/interpret_qp + Detector.unfold_interpret/interpret_result (skipped if pyoptexplain not installed) |
| `tests/test_security.py` | bandit static security scan (no HIGH findings) |
| `tests/test_krylov_tv.py` | CGLS (`unfold_cgls`), GKS (`unfold_gks`) and Tikhonov-TV (`unfold_tikhonov_tv`): solver edge cases + Detector wrappers + combined pipeline |
| `tests/test_ferdor.py` | FERDOR (`unfold_ferdor`/`solve_ferdor`): wrapper + core solver, discrepancy-principle smoothing, error handling |
| `tests/test_rebunki.py` | ReBUNKI/SPUNIT (`unfold_rebunki`/`solve_rebunki`): wrapper + core solver, positivity/validation |
| `tests/test_nsduaz.py` | NSDUAZ (`unfold_nsduaz`/`solve_nsduaz`): wrapper + core solver, catalogue selection, validation |
| `tests/test_genetic_improvements.py` | Genetic extensions: two-step coarse-to-fine, NSGA-II/Pareto selection, smoothers, TGASU crossover/mutation, `extra_starting` injection, `_coarsen_columns`/`_split_coarse` helpers |

### Analysis tools

- **DynaPyt** (`tools/run_dynapyt.py`) — dynamic analysis. Instruments a throwaway
  copy of the repo (`.dynapyt/` via `git archive`, real `src/` is never touched),
  runs BranchCoverage over a fast subset (`test_detector.py`, `test_readings.py`),
  writes the report to `.dynapyt_report/`. Use `--keep` to debug, `--analysis` to
  swap in other analyses. ⚠️ `CallGraph` and `TraceAll` are extremely slow on this
  codebase — avoid. Requires the `PYTHONPATH` trick in the script: the copy's root
  must shadow the editable install so `import src.bssunfold` hits instrumented code.
- **bandit** — static security scan (`uv run bandit -r src/bssunfold`). `src/` is
  clean (no eval/exec/subprocess/pickle/assert). Config in `[tool.bandit]`.
- **vulture** — dead-code scan (`uv run vulture src/bssunfold tests/ tools/`).
  Run it with tests/ included: a src-only scan flags public methods and helpers
  that are only exercised from tests. Known false positives to ignore: the
  `*args`/`**kwargs` fallback signatures in `_numba_jit.py` (must accept any
  args before raising ImportError) and the docplex `context.solver.log_output`
  API attribute. Verify candidate findings with `grep -rn "<name>" src/ tests/`.
- **pip-audit** — dependency vulnerability scan. Needs network (OSV). Runs in the
  `security-analysis.yml` CI job, not the unit suite. Current allowlist:
  `CVE-2026-61632` (pymdown-extensions path traversal, dev-only docs tool,
  jupyter-book pins `<11`). Other dev-tool transitives were fixed by upgrading
  cryptography, jupyter-server, mistune, pillow, setuptools in `uv.lock`.

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
- **pyoptexplain IS installed** in dev. To test the ImportError fallback in `unfold_interpret.py`, patch `builtins.__import__` to raise for `'pyoptexplain'` AND reset the lazy namespace cache (`bssunfold.core.unfold_interpret._pyopt._loaded = None`) before/after — the cache survives otherwise. NOTE: `bssunfold.core.unfold_interpret` attribute shadows the submodule with the `unfold_interpret` function; use `sys.modules["bssunfold.core.unfold_interpret"]` to reach the module object. `QuadraticMatrixScenarioRepresentation` (returned by `handle.quadratic_representation()` for scenario runs) has NO `.quadratic_representation()` method — `_make_analyzer` must accept an already-built representation.

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
- Interpretation (`unfold_interpret`/`interpret_result`) is backed by optional `pyoptexplain`; `Detector.unfold_interpret` returns the standard result dict plus `report` and `interpretation_metrics` keys.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).


# AGENTS.md

## Purpose
This repository is developed with AI-assisted workflows in OpenCode.
Agents working in this repo must prioritize:
- correctness
- maintainability
- explicit architecture
- testability
- safe incremental delivery
- business value per stage

Do not treat this repo as a one-shot generation task.
Work in small, reviewable increments.

---

## Working model

### Default execution model
For any non-trivial task:
1. Start with a plan.
2. Limit scope to the current stage only.
3. Implement the minimum viable increment.
4. Run relevant verification.
5. Produce a concise change report.

Do not attempt broad uncontrolled rewrites unless explicitly asked.

### Stage discipline
Work must be organized into stages.
Each stage should:
- produce demonstrable value
- be testable
- be reviewable
- have clear acceptance criteria
- have clear out-of-scope boundaries

### End-to-end preference
Prefer vertical slices over layer-by-layer delivery.
Do not build the whole backend first and postpone frontend/integration until later unless explicitly requested.
For product features, prefer:
- backend changes
- frontend changes
- DB changes
- integration
- tests
within the same stage.

---

## Planning rules

Before making large or risky changes, produce a plan that includes:
- objective
- scope
- out of scope
- impacted modules
- data model changes
- API changes
- frontend changes
- risks
- verification strategy
- rollout / migration notes if applicable

If the task is ambiguous, choose the simplest extensible design and state assumptions explicitly.

---

## Code quality rules

### General
- Prefer simple, explicit code over clever abstractions.
- Keep functions and modules cohesive.
- Avoid hidden side effects.
- Avoid premature generalization.
- Follow existing project conventions unless they are clearly harmful.

### Readability
- Write code for future maintainers.
- Use descriptive names.
- Keep files focused.
- Remove dead code when safe to do so.
- Do not leave unexplained hacks.

### Architecture
- Respect module boundaries.
- Do not introduce circular dependencies.
- Keep business logic out of UI glue code.
- Keep transport concerns separate from domain logic.
- Keep persistence concerns separate from application behavior where practical.

### Change safety
- Minimize blast radius.
- Avoid unnecessary rewrites.
- Preserve backward compatibility unless the task explicitly allows breaking changes.
- If a breaking change is introduced, document it clearly.

---

## Testing rules

Testing is required for meaningful changes.

### Minimum expectations
For each completed stage, add or update the appropriate level of tests:
- unit tests for isolated logic
- integration tests for contracts and boundaries
- end-to-end or smoke tests for critical user flows
- migration verification for DB changes
- type checks / lint / static analysis where applicable

### Required mindset
Do not say "done" if code was written but not verified.
If tests cannot be run, state exactly why.

### Test priority
Prioritize tests for:
- critical business flows
- auth / permissions
- money-related logic
- order / workflow transitions
- data integrity
- external integrations
- error handling

---

## Database and migrations

For any DB change:
- prefer explicit migrations
- make schema changes reversible when practical
- document data migration implications
- avoid destructive changes without explicit warning
- preserve production safety

If changing schema:
- update related models
- update seeds / fixtures if needed
- update integration tests
- mention rollout order if relevant

---

## API and contracts

For any API change:
- keep contracts explicit
- update request/response types
- update docs or examples if applicable
- preserve backward compatibility where possible
- document breaking changes clearly

When frontend and backend both change:
- keep payload contracts aligned
- do not leave frontend relying on stale mocks unless explicitly temporary
- state temporary mismatches clearly if they exist

---

## Frontend rules

Frontend changes must prioritize:
- clarity
- predictable state flow
- loading/error/empty states
- accessibility basics
- minimal but consistent UX

Do not build visual complexity before flow correctness.
Prefer working user paths over decorative completeness.

---

## Infrastructure and operations

Treat infra changes as production-sensitive.

### For infra-related tasks
- prefer configuration as code
- show intended changes explicitly
- mention rollout order
- mention rollback path
- do not run destructive commands unless explicitly authorized

### Secrets and safety
- never hardcode secrets
- do not expose credentials in logs or code
- prefer environment-based configuration
- preserve least privilege assumptions

---

## Expected final response format for implementation tasks

At the end of a task, provide:

1. Summary of changes
2. Files/modules affected
3. Verification performed
4. Known risks / tech debt / assumptions
5. Manual QA steps
6. Recommended next step

Do not claim verification that was not actually run.

---

## Review expectations

A review pass should look for:
- correctness issues
- edge cases
- missing tests
- risky migrations
- contract drift
- architecture drift
- unnecessary complexity
- maintainability issues

Review output should separate:
- critical issues
- important follow-ups
- optional improvements

---

## Delivery philosophy

The goal is not maximal code generation.
The goal is controlled delivery of working software.

Favor:
- small stages
- explicit assumptions
- visible progress
- stable contracts
- end-to-end validation
- maintainable code
over:
- giant rewrites
- implicit decisions
- hidden complexity
- unverified output

---

## If unsure

If unsure:
- inspect relevant code first
- propose a small safe plan
- choose the minimal extensible design
- document assumptions
- avoid pretending certainty