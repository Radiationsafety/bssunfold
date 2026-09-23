"""Comprehensive import + parameter + smoke tests for every public
``unfold_*`` method on the :class:`bssunfold.Detector` class.

The tests are organised in four layers:

1. **TestAllUnfoldMethodsImported** — every ``unfold_*`` name listed in
   ``bssunfold/core/__init__.py``'s ``__all__`` must be present on the
   ``Detector`` class.

2. **TestMethodSignatureContract** — each ``unfold_*`` accepts the
   canonical ``(readings, initial_spectrum=None, ...)`` first arguments,
   plus the standard Monte-Carlo / housekeeping kwargs
   (``calculate_errors``, ``noise_level``, ``n_montecarlo``,
   ``random_state``, ``save_result``).

3. **TestMethodSmoke** — for each method we exercise the API with a
   trivial synthetic reading set and check that:
   - the result dict has the keys documented in
     ``core/_base_unfolder.py::_standardize_output``;
   - the unfolded spectrum is finite and non-negative;
   - ``method`` name is a string;
   - the residual norm is a non-negative float.

4. **TestParameterAssignment** — for each method we set every documented
   kwarg to a value *different* from its default and verify that the
   call still succeeds (this catches typos in kwarg names and any
   "unexpected keyword" errors introduced by refactors).

Optional-dependency methods (``unfold_zfit``, ``unfold_maeo``,
``unfold_mystic_hybrid``, ``unfold_qubo``, ``unfold_cuqi``, …) are
skipped automatically when the underlying third-party package is not
importable.

These tests complement the existing per-method test files in
``tests/test_*.py`` — they are deliberately *not* a replacement for
those, but a single-gate check that the public API surface still
imports cleanly and accepts its documented kwargs.
"""
from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bssunfold import RF_GSF, Detector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Names of third-party backends used by some unfolders.  Each backend
#: maps to the names of unfold_* methods that depend on it.  A method
#: listed here will be skipped if the corresponding import fails.
#:
#: Note: ``unfold_odl_pdhg`` and ``unfold_odl_douglas_rachford`` are
#: named after the ODL formulation but are implemented in pure NumPy
#: (see the module docstring of ``unfold_odl_advanced.py``); they do
#: NOT require the ``odl`` package and are intentionally absent from
#: this list.
OPTIONAL_BACKENDS: dict[str, list[str]] = {
    "pymoo": ["unfold_maeo"],
    "zfit": ["unfold_zfit"],
    "mystic": ["unfold_mystic_hybrid", "unfold_mystic"],
    "pyqubo": ["unfold_qubo"],
    "cuqi": ["unfold_cuqi"],
    "lmfit": ["unfold_lmfit"],
    "numba": ["unfold_genetic", "unfold_fission_ga"],
    # Commercial engines — license required; skipped unless the engine
    # package is importable AND cvxpy reports the solver as installed.
    "gurobipy": ["unfold_gurobi"],
    "mosek": ["unfold_mosek"],
    "cplex": ["unfold_cplex"],
    "coptpy": ["unfold_copt"],
    "xpress": ["unfold_xpress"],
}

#: Commercial backends additionally need the cvxpy interface to report the
#: solver as installed (engine importable is not enough).
COMMERCIAL_METHODS: dict[str, str] = {
    "unfold_gurobi": "GUROBI",
    "unfold_mosek": "MOSEK",
    "unfold_cplex": "CPLEX",
    "unfold_copt": "COPT",
    "unfold_xpress": "XPRESS",
}


def _backend_missing(backend: str) -> bool:
    """Return True if the given optional dependency cannot be imported."""
    try:
        importlib.import_module(backend)
        return False
    except Exception:
        return True


def _skip_if_backend_missing(method_name: str) -> None:
    for backend, methods in OPTIONAL_BACKENDS.items():
        if method_name in methods and _backend_missing(backend):
            pytest.skip(f"optional backend '{backend}' not installed; "
                        f"required for {method_name}")
    if method_name in COMMERCIAL_METHODS:
        import cvxpy as cp
        if COMMERCIAL_METHODS[method_name] not in cp.installed_solvers():
            pytest.skip(f"commercial solver "
                        f"'{COMMERCIAL_METHODS[method_name]}' (license "
                        f"required) not installed; required for {method_name}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> Detector:
    """Default Detector with built-in GSF response functions."""
    return Detector(RF_GSF)


@pytest.fixture(scope="module")
def iaea_csv() -> pd.DataFrame:
    """IAEA Compendium Monte-Carlo spectra (61 rows × 21 columns)."""
    # Resolve the CSV shipped with the repo (matches the path used by the
    # example notebooks: ../tests/MonteCarlo_…_comparison.csv).
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = (
        repo_root
        / "tests"
        / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
    )
    return pd.read_csv(csv_path)


@pytest.fixture()
def iaea_readings(detector: Detector, iaea_csv: pd.DataFrame) -> dict[str, float]:
    """Readings derived from the IAEA t4-14-s.txt_1 spectrum."""
    return detector.get_effective_readings_for_spectra(
        iaea_csv[["E_MeV", "t4-14-s.txt_1"]]
    )


@pytest.fixture()
def simple_readings(detector: Detector) -> dict[str, float]:
    """Tiny synthetic reading set that exercises every sphere."""
    return {nm: float(i + 1) * 0.05 for i, nm in enumerate(detector.detector_names)}


# ---------------------------------------------------------------------------
# 1. Import / discovery
# ---------------------------------------------------------------------------

class TestAllUnfoldMethodsImported:
    """Every unfold_* exported from bssunfold.core must be on Detector."""

    def test_core_has_all_attribute(self) -> None:
        from bssunfold import core
        assert hasattr(core, "__all__"), "bssunfold.core.__all__ is missing"
        assert isinstance(core.__all__, list)
        assert len(core.__all__) > 50, "expected ~90 unfold_* methods"

    @pytest.mark.parametrize("method_name", [
        # The 23 methods that previously lacked dedicated examples — these
        # are the ones whose existence we most care about here.
        "unfold_gnowee", "unfold_maeo", "unfold_nnqp", "unfold_qpmad",
        "unfold_zfit", "unfold_iterative_refinement", "unfold_odl_pdhg",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
        "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
        "unfold_gurobi", "unfold_mosek", "unfold_cplex", "unfold_copt", "unfold_xpress"
    ])
    def test_method_is_detector_attribute(self, method_name: str) -> None:
        assert hasattr(Detector, method_name), \
            f"Detector.{method_name} is missing"
        assert callable(getattr(Detector, method_name)), \
            f"Detector.{method_name} is not callable"

    def test_core_init_exports_all_methods(self) -> None:
        from bssunfold import core
        # The 23 methods must all be in core.__all__
        names = {
            "unfold_gnowee", "unfold_maeo", "unfold_nnqp", "unfold_qpmad",
            "unfold_zfit", "unfold_iterative_refinement", "unfold_odl_pdhg",
            "unfold_amaxed_regularization", "unfold_crystal_ball",
            "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
            "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
            "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
            "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
            "unfold_tikhonov_legendre", "unfold_ensemble",
            "unfold_gurobi", "unfold_mosek", "unfold_cplex",
            "unfold_copt", "unfold_xpress",
        }
        missing = names - set(core.__all__)
        assert not missing, f"missing from core.__all__: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 2. Signature contract
# ---------------------------------------------------------------------------

#: Names of the kwargs that *every* unfold_* method must accept (the
#: shared "Monte-Carlo + housekeeping" tail defined by
#: ``core/_base_unfolder.py::run_unfolding``).
COMMON_KWARGS = [
    "initial_spectrum",
    "calculate_errors",
    "noise_level",
    "n_montecarlo",
    "save_result",
    "random_state",
    "max_neutron_energy",
]


class TestMethodSignatureContract:
    """Every unfold_* must accept readings + initial_spectrum + common kwargs."""

    @pytest.mark.parametrize("method_name", [
        "unfold_gnowee", "unfold_maeo", "unfold_nnqp", "unfold_qpmad",
        "unfold_zfit", "unfold_iterative_refinement", "unfold_odl_pdhg",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
        "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
        "unfold_gurobi", "unfold_mosek", "unfold_cplex", "unfold_copt", "unfold_xpress"
    ])
    def test_signature_starts_with_readings(self, method_name: str) -> None:
        sig = inspect.signature(getattr(Detector, method_name))
        params = list(sig.parameters.values())
        # Skip 'self'.
        assert params[0].name == "self"
        assert params[1].name == "readings", (
            f"{method_name}: second parameter should be 'readings', "
            f"got '{params[1].name}'"
        )

    @pytest.mark.parametrize("method_name", [
        "unfold_gnowee", "unfold_nnqp", "unfold_qpmad", "unfold_zfit",
        "unfold_iterative_refinement", "unfold_odl_pdhg",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
        "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
        "unfold_gurobi", "unfold_mosek", "unfold_cplex", "unfold_copt", "unfold_xpress"
    ])
    def test_signature_has_common_kwargs(self, method_name: str) -> None:
        sig = inspect.signature(getattr(Detector, method_name))
        param_names = set(sig.parameters.keys())
        missing = [k for k in COMMON_KWARGS if k not in param_names]
        # maeo is special-cased: it accepts **kwargs and forwards them,
        # so we don't require the explicit common kwargs.
        if method_name == "unfold_maeo":
            return
        assert not missing, \
            f"{method_name}: missing common kwargs {missing}"


# ---------------------------------------------------------------------------
# 3. Smoke tests — call each method with trivial inputs
# ---------------------------------------------------------------------------

#: Per-method minimal kwargs to keep the test fast.  These mirror the
#: kwargs used in the example notebooks.
SMOKE_KWARGS: dict[str, dict[str, Any]] = {
    "unfold_gnowee": dict(population=5, max_gens=2, max_fevals=20,
                          stall_limit=2, random_state=0),
    "unfold_maeo": dict(n_cycles=1, n_gen_per_cycle=2, pop_size=4, seed=0),
    "unfold_nnqp": dict(max_iterations=20),
    "unfold_qpmad": dict(max_iterations=20),
    "unfold_zfit": dict(max_iterations=5, use_mcmc=False),
    "unfold_iterative_refinement": dict(
        first_pass_kwargs={"max_iterations": 10},
        second_pass_kwargs={"max_iterations": 10}),
    "unfold_odl_pdhg": dict(max_iterations=5),
    "unfold_amaxed_regularization": dict(max_iterations=20),
    "unfold_crystal_ball": dict(),
    "unfold_directed_divergence": dict(max_iterations=10),
    "unfold_express": dict(max_iterations=2),
    "unfold_ferdor": dict(max_iterations=10),
    "unfold_imaxed": dict(max_iterations=20),
    "unfold_mystic_hybrid": dict(),
    "unfold_nsduaz": dict(max_iterations=20),
    "unfold_odl_douglas_rachford": dict(max_iterations=5),
    "unfold_qubo": dict(n_bits=4, num_reads=2, max_iterations=5),
    "unfold_rebunki": dict(max_iterations=20),
    "unfold_rfsp_jul": dict(max_iterations=10),
    "unfold_scipy_direct_method": dict(max_iterations=10),
    "unfold_staysl": dict(),
    "unfold_tikhonov_legendre": dict(),
    "unfold_ensemble": dict(),
    # commercial engines (license required); short timeouts keep CI fast
    "unfold_gurobi": dict(timeout=5.0),
    "unfold_mosek": dict(timeout=5.0),
    "unfold_cplex": dict(timeout=5.0),
    "unfold_copt": dict(timeout=5.0),
    "unfold_xpress": dict(timeout=5.0),
}

#: Required output keys per ``_standardize_output`` in _base_unfolder.py.
REQUIRED_OUTPUT_KEYS = [
    "energy", "spectrum", "spectrum_absolute", "effective_readings",
    "residual", "residual_norm", "method", "doserates",
]


class TestMethodSmoke:
    """End-to-end smoke test: each method returns a sensible result dict."""

    @pytest.mark.parametrize("method_name", list(SMOKE_KWARGS.keys()))
    def test_method_returns_valid_result(
        self, detector: Detector, simple_readings: dict[str, float],
        method_name: str,
    ) -> None:
        _skip_if_backend_missing(method_name)
        kwargs = SMOKE_KWARGS[method_name]
        fn = getattr(detector, method_name)
        result = fn(simple_readings, **kwargs)

        assert isinstance(result, dict), \
            f"{method_name}: result must be a dict, got {type(result)}"

        # Check required keys
        missing = [k for k in REQUIRED_OUTPUT_KEYS if k not in result]
        assert not missing, \
            f"{method_name}: result missing required keys {missing}, " \
            f"got {sorted(result.keys())}"

        # Spectrum sanity checks
        spec = result["spectrum"]
        assert isinstance(spec, np.ndarray), \
            f"{method_name}: 'spectrum' must be np.ndarray, got {type(spec)}"
        assert spec.ndim == 1, \
            f"{method_name}: 'spectrum' must be 1-D, got shape {spec.shape}"
        assert spec.shape == (detector.n_energy_bins,), \
            f"{method_name}: 'spectrum' shape {spec.shape} != " \
            f"({detector.n_energy_bins},)"
        assert np.all(np.isfinite(spec)), \
            f"{method_name}: 'spectrum' contains non-finite values"
        assert np.all(spec >= 0), \
            f"{method_name}: 'spectrum' contains negative values"

        # Method name string
        assert isinstance(result["method"], str) and result["method"], \
            f"{method_name}: 'method' must be a non-empty string"

        # Residual norm
        rn = result["residual_norm"]
        assert isinstance(rn, float) and math.isfinite(rn) and rn >= 0, \
            f"{method_name}: 'residual_norm' must be a finite non-negative " \
            f"float, got {rn!r}"

        # Energy grid
        E = result["energy"]
        assert isinstance(E, np.ndarray), \
            f"{method_name}: 'energy' must be np.ndarray"
        assert E.shape == (detector.n_energy_bins,), \
            f"{method_name}: 'energy' shape mismatch"

        # Effective readings & doserates dicts
        assert isinstance(result["effective_readings"], dict), \
            f"{method_name}: 'effective_readings' must be a dict"
        assert isinstance(result["doserates"], dict), \
            f"{method_name}: 'doserates' must be a dict"


# ---------------------------------------------------------------------------
# 4. Parameter assignment — explicit kwarg overwrite
# ---------------------------------------------------------------------------

class TestParameterAssignment:
    """For each method, overwrite every documented kwarg and verify the
    call still succeeds.  This catches typos and refactor regressions
    where a kwarg is silently renamed or removed.
    """

    #: For each method, a dict of <kwarg_name: non-default_value> pairs
    #: exercising every *non-shared* kwarg in the method signature.
    PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
        "unfold_gnowee": dict(
            population=10, max_gens=5, max_fevals=50, stall_limit=3,
            conv_tol=1e-5, opt_conv_tol=0.02, frac_elite=0.3, frac_levy=0.9,
            frac_mutation=0.3, alpha_levy=1.6, gamma_levy=1.1, n_levy=2,
            scaling_factor=5.0, init_sampling="random", regularization=0.02,
            norm=1, smoothness_order=1, smoothness_weight=0.5,
            entropy_weight=0.1, half_range=3.0, verbose=False,
        ),
        "unfold_maeo": dict(
            n_cycles=2, n_gen_per_cycle=3, pop_size=10,
            algorithms=["nsga3", "spea2"], lambda_smooth=0.02,
            convergence_assist_ratio=0.3, seed=42, verbose=False,
            save_result=False,
        ),
        "unfold_nnqp": dict(
            regularization=1e-3, smoothness_order=1, smoothness_weight=0.5,
            tol=1e-7, max_iterations=100, floor=1e-8,
        ),
        "unfold_qpmad": dict(
            regularization=1e-3, smoothness_order=1, smoothness_weight=0.5,
            floor=1e-8, lb=None, ub=None, backend="python", tol=1e-10,
            max_iterations=100,
        ),
        "unfold_zfit": dict(
            max_iterations=20, use_mcmc=False, n_samples=500,
            regularization=0.05, smoothness_weight=0.005,
        ),
        "unfold_iterative_refinement": dict(
            first_pass_kwargs={"max_iterations": 5},
            second_pass_kwargs={"max_iterations": 5},
            alpha=0.5, max_alpha_search=5,
        ),
        "unfold_odl_pdhg": dict(
            max_iterations=10, tau=0.5, sigma=0.5, use_tv=False,
            tv_weight=0.05, nonnegativity=True,
        ),
        "unfold_amaxed_regularization": dict(
            sigma_factor=0.2, tau=0.5, max_iterations=50, tolerance=1e-9,
            line_search_tol=1e-7,
        ),
        "unfold_crystal_ball": dict(
            regularization=0.1,
        ),
        "unfold_directed_divergence": dict(
            max_iterations=50, tol_chi2=1.5, tol_rel=1e-5,
            relative_uncertainty=0.03, smoothness_order=1,
            smoothness_weight=0.1,
        ),
        "unfold_express": dict(
            n_groups=4, interval_boundaries=None, max_iterations=2,
            tol_iteration=0.03, relative_uncertainty=0.03,
        ),
        "unfold_ferdor": dict(
            max_iterations=50, tolerance=0.002, smoothing=0.002,
            chi_squared_target=1.2, relative_uncertainty=0.08,
        ),
        "unfold_imaxed": dict(
            sigma_factor=0.2, max_iterations=50, tolerance=1e-9,
            line_search_tol=1e-7,
        ),
        "unfold_mystic_hybrid": dict(
            regularization=1e-3, norm=1, global_solver="diffev2",
            local_solver="fmin_powell", global_maxiter=5, global_maxfun=10,
            local_maxiter=5, local_maxfun=10, npop=10,
            regularization_method="manual", noise_var=0.01,
            smoothness_order=1, smoothness_weight=0.5,
        ),
        "unfold_nsduaz": dict(
            catalogue=None, use_catalogue=False, reference_name=None,
            smoothing=0.05, max_iterations=100, tolerance=0.005,
        ),
        "unfold_odl_douglas_rachford": dict(
            max_iterations=10, use_tv=False, tv_weight=0.05,
            nonnegativity=True,
        ),
        "unfold_qubo": dict(
            n_bits=4, max_value=None, regularization=0.005,
            max_iterations=50, annealing_time=500, num_reads=5,
        ),
        "unfold_rebunki": dict(
            smoothing=0.05, max_iterations=100, tolerance=0.005,
        ),
        "unfold_rfsp_jul": dict(
            max_iterations=50, tolerance=1e-5, weights=None,
        ),
        "unfold_scipy_direct_method": dict(
            tolerance=1e-9, max_iterations=50, method="cg",
        ),
        "unfold_staysl": dict(
            relative_uncertainty=0.08, prior_uncertainty=0.5,
        ),
        "unfold_tikhonov_legendre": dict(
            delta=0.02, n_polynomials=10,
        ),
        "unfold_ensemble": dict(
            methods=None, weights=None, combination="weighted_average",
            trim_fraction=0.1,
        ),
        # norm=1 requires nonneg=True (library guard), so exercise the
        # non-default combo norm=2 + nonneg=False here for all engines.
        "unfold_gurobi": dict(
            regularization=1e-3, norm=2, timeout=5.0, smoothness_order=1,
            smoothness_weight=0.5, nonneg=False,
            regularization_method="manual", noise_var=0.01,
        ),
        "unfold_mosek": dict(
            regularization=1e-3, norm=2, timeout=5.0, smoothness_order=1,
            smoothness_weight=0.5, nonneg=False,
            regularization_method="manual", noise_var=0.01,
        ),
        "unfold_cplex": dict(
            regularization=1e-3, norm=2, timeout=5.0, smoothness_order=1,
            smoothness_weight=0.5, nonneg=False,
            regularization_method="manual", noise_var=0.01,
        ),
        "unfold_copt": dict(
            regularization=1e-3, norm=2, timeout=5.0, smoothness_order=1,
            smoothness_weight=0.5, nonneg=False,
            regularization_method="manual", noise_var=0.01,
        ),
        "unfold_xpress": dict(
            regularization=1e-3, norm=2, timeout=5.0, smoothness_order=1,
            smoothness_weight=0.5, nonneg=False,
            regularization_method="manual", noise_var=0.01,
        ),
    }

    @pytest.mark.parametrize("method_name", list(PARAM_OVERRIDES.keys()))
    def test_kwargs_are_accepted(
        self, detector: Detector, simple_readings: dict[str, float],
        method_name: str,
    ) -> None:
        _skip_if_backend_missing(method_name)
        kwargs = {**SMOKE_KWARGS[method_name],
                  **self.PARAM_OVERRIDES[method_name]}
        # For iterative_refinement, max_iterations in SMOKE_KWARGS lives
        # inside first_pass_kwargs/second_pass_kwargs, not at the top
        # level — so we don't merge with SMOKE_KWARGS in that case.
        if method_name == "unfold_iterative_refinement":
            kwargs = self.PARAM_OVERRIDES[method_name]
        fn = getattr(detector, method_name)
        # Should not raise TypeError: unexpected keyword argument.
        result = fn(simple_readings, **kwargs)
        assert "spectrum" in result, f"{method_name}: no 'spectrum' key"
        assert np.all(np.isfinite(result["spectrum"])), \
            f"{method_name}: non-finite spectrum with overrides"

    @pytest.mark.parametrize("method_name", [
        "unfold_gnowee", "unfold_nnqp", "unfold_qpmad", "unfold_zfit",
        "unfold_iterative_refinement", "unfold_odl_pdhg",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
        "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
        "unfold_gurobi", "unfold_mosek", "unfold_cplex", "unfold_copt", "unfold_xpress"
    ])
    def test_unexpected_kwarg_raises(
        self, detector: Detector, simple_readings: dict[str, float],
        method_name: str,
    ) -> None:
        _skip_if_backend_missing(method_name)
        fn = getattr(detector, method_name)
        # maeo has **kwargs so it would NOT raise on unexpected names;
        # we skip it for this test.
        if method_name == "unfold_maeo":
            pytest.skip("unfold_maeo accepts **kwargs; not applicable")
        with pytest.raises(TypeError):
            fn(simple_readings, this_kwarg_does_not_exist=True)

    @pytest.mark.parametrize("method_name", [
        "unfold_gnowee", "unfold_nnqp", "unfold_qpmad", "unfold_zfit",
        "unfold_iterative_refinement", "unfold_odl_pdhg",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_mystic_hybrid", "unfold_nsduaz",
        "unfold_odl_douglas_rachford", "unfold_qubo", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
        "unfold_gurobi", "unfold_mosek", "unfold_cplex", "unfold_copt", "unfold_xpress"
    ])
    def test_initial_spectrum_kwarg(
        self, detector: Detector, simple_readings: dict[str, float],
        method_name: str,
    ) -> None:
        """Passing an explicit initial_spectrum must be accepted and
        must not change the spectrum length or dtype."""
        _skip_if_backend_missing(method_name)
        n = detector.n_energy_bins
        x0 = np.full(n, 0.5)
        kwargs = SMOKE_KWARGS.get(method_name, {})
        # iterative_refinement uses first_pass_kwargs / second_pass_kwargs,
        # not top-level max_iterations.
        if method_name == "unfold_iterative_refinement":
            kwargs = dict(
                first_pass_kwargs={"max_iterations": 5},
                second_pass_kwargs={"max_iterations": 5},
            )
        fn = getattr(detector, method_name)
        result = fn(simple_readings, initial_spectrum=x0, **kwargs)
        assert result["spectrum"].shape == (n,)


# ---------------------------------------------------------------------------
# 5. IAEA Compendium end-to-end check
# ---------------------------------------------------------------------------

class TestIAEACompendiumEndToEnd:
    """Repeat one full pass of the example notebooks inside pytest, on
    the IAEA t4-14-s.txt_1 spectrum.  This is the slowest test class;
    it is skipped unless the IAEA CSV is present.
    """

    SPECTRA_COLUMNS = [
        "ISO_ref_Cf252", "ISO_ref_AmBe", "t4-14-s.txt_1",
        "t4-17-s.txt_1", "t4-19-s.txt_1",
    ]

    def test_iaea_csv_exists(self, iaea_csv: pd.DataFrame) -> None:
        assert len(iaea_csv) > 0
        assert "E_MeV" in iaea_csv.columns
        # All 5 expected spectra columns must be present.
        missing = [c for c in self.SPECTRA_COLUMNS if c not in iaea_csv.columns]
        assert not missing, f"IAEA CSV missing columns: {missing}"

    @pytest.mark.parametrize("method_name", [
        # Run a representative subset on the IAEA benchmark.
        "unfold_nnqp", "unfold_qpmad", "unfold_iterative_refinement",
        "unfold_amaxed_regularization", "unfold_crystal_ball",
        "unfold_directed_divergence", "unfold_express", "unfold_ferdor",
        "unfold_imaxed", "unfold_nsduaz", "unfold_rebunki",
        "unfold_rfsp_jul", "unfold_scipy_direct_method", "unfold_staysl",
        "unfold_tikhonov_legendre", "unfold_ensemble",
    ])
    def test_method_runs_on_iaea(
        self, detector: Detector, iaea_csv: pd.DataFrame,
        method_name: str,
    ) -> None:
        _skip_if_backend_missing(method_name)
        readings = detector.get_effective_readings_for_spectra(
            iaea_csv[["E_MeV", "t4-14-s.txt_1"]]
        )
        kwargs = SMOKE_KWARGS[method_name]
        if method_name == "unfold_iterative_refinement":
            kwargs = dict(
                first_pass_kwargs={"max_iterations": 10},
                second_pass_kwargs={"max_iterations": 10},
            )
        fn = getattr(detector, method_name)
        result = fn(readings, **kwargs)
        assert np.all(np.isfinite(result["spectrum"]))
        assert np.all(result["spectrum"] >= 0)

        # Compute quality metrics vs ground truth.
        from bssunfold.utils.comparison import compare_spectra
        phi_true = np.interp(detector.E_MeV, iaea_csv["E_MeV"].values,
                             iaea_csv["t4-14-s.txt_1"].values)
        m = compare_spectra(
            phi_true, result["spectrum"],
            metrics=["relative_flux_error", "pearson_r",
                      "root_mean_squared_error", "comprehensive_score"],
        )
        # Sanity ranges — these are *very* permissive bounds; their purpose
        # is only to catch catastrophic regressions (NaN, all-zero, …).
        assert math.isfinite(m["relative_flux_error"])
        assert math.isfinite(m["pearson_r"])
        assert -1.0 <= m["pearson_r"] <= 1.0
        assert m["root_mean_squared_error"] >= 0
        assert math.isfinite(m["comprehensive_score"])


# ---------------------------------------------------------------------------
# 6. Notebook smoke-test: ensure every new IAEA notebook executes
# ---------------------------------------------------------------------------

class TestNotebooksExecute:
    """Execute each new IAEA-benchmark notebook with nbconvert (headless).

    Skipped when ``jupyter nbconvert`` is unavailable or when an
    optional backend required by a notebook is not installed.
    """

    NOTEBOOKS: dict[str, str] = {
        "56-gnowee-iaea.ipynb": "unfold_gnowee",
        "57-maeo-iaea.ipynb": "unfold_maeo",
        "58-nnqp-iaea.ipynb": "unfold_nnqp",
        "59-qpmad-iaea.ipynb": "unfold_qpmad",
        "60-zfit-iaea.ipynb": "unfold_zfit",
        "61-iterative-refinement-iaea.ipynb": "unfold_iterative_refinement",
        "62-odl-pdhg-iaea.ipynb": "unfold_odl_pdhg",
        "63-amaxed-regularization-iaea.ipynb": "unfold_amaxed_regularization",
        "64-crystal-ball-iaea.ipynb": "unfold_crystal_ball",
        "65-directed-divergence-iaea.ipynb": "unfold_directed_divergence",
        "66-express-iaea.ipynb": "unfold_express",
        "67-ferdor-iaea.ipynb": "unfold_ferdor",
        "68-imaxed-iaea.ipynb": "unfold_imaxed",
        "69-mystic-hybrid-iaea.ipynb": "unfold_mystic_hybrid",
        "70-nsduaz-iaea.ipynb": "unfold_nsduaz",
        "71-odl-douglas-rachford-iaea.ipynb": "unfold_odl_douglas_rachford",
        "72-qubo-iaea.ipynb": "unfold_qubo",
        "73-rebunki-iaea.ipynb": "unfold_rebunki",
        "74-rfsp-jul-iaea.ipynb": "unfold_rfsp_jul",
        "75-scipy-direct-iaea.ipynb": "unfold_scipy_direct_method",
        "76-staysl-iaea.ipynb": "unfold_staysl",
        "77-tikhonov-legendre-iaea.ipynb": "unfold_tikhonov_legendre",
        "78-ensemble-iaea.ipynb": "unfold_ensemble",
        "79-louhi-iaea.ipynb": "unfold_louhi",
        "80-osem-anlm-iaea.ipynb": "unfold_osem_anlm",
        "81-cuqi-quality-analysis-iaea.ipynb": "unfold_cuqi",
    }

    @pytest.fixture(autouse=True)
    def _check_nbconvert_available(self) -> None:
        import shutil
        if shutil.which("jupyter") is None:
            pytest.skip("jupyter not available on PATH")

    @pytest.mark.parametrize("nb_filename, method_name", list(NOTEBOOKS.items()))
    def test_notebook_executes(self, nb_filename: str, method_name: str) -> None:
        _skip_if_backend_missing(method_name)
        import subprocess
        import tempfile
        from pathlib import Path
        nb_path = Path(__file__).resolve().parent.parent / "examples" / nb_filename
        if not nb_path.exists():
            pytest.skip(f"notebook not found: {nb_path}")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            out_path = f.name
        try:
            proc = subprocess.run(
                [
                    "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=600",
                    str(nb_path), "--output", out_path,
                ],
                capture_output=True, text=True, timeout=900,
            )
            assert proc.returncode == 0, (
                f"notebook {nb_filename} failed (rc={proc.returncode}).\n"
                f"stderr (tail):\n{proc.stderr[-2000:]}"
            )
        finally:
            try:
                Path(out_path).unlink()
            except OSError:
                pass
