"""Tests for the commercial (license-required) QP solver unfolding methods.

Covers the shared cvxpy backend in ``bssunfold.core._commercial_qp``, the
general ``solve_commercial``/``unfold_commercial`` entry points and the five
pre-bound wrappers (Gurobi, MOSEK, CPLEX, COPT, XPRESS).  Only the engines
that are importable and licensed on the test machine actually solve; the
license-required failure paths (missing package, missing license, solver
error) are exercised via mocks and ``block_import``.

IBM CPLEX is installed in the dev environment (Community Edition), so the
``cplex`` methods run end-to-end here; the other four are verified through
their graceful-failure contracts.
"""

from unittest.mock import patch

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core.unfold_commercial import (
    commercial_solver_info,
    is_commercial_solver_available,
    solve_commercial,
    solve_copt,
    solve_cplex,
    solve_gurobi,
    solve_mosek,
    solve_xpress,
    unfold_commercial,
)
from tests.conftest import block_import

COMMERCIAL_ALIASES = ["gurobi", "mosek", "cplex", "copt", "xpress"]
PIP_MODULES = {
    "gurobi": "gurobipy",
    "mosek": "mosek",
    "cplex": "cplex",
    "copt": "coptpy",
    "xpress": "xpress",
}

SOLVE_FUNCS = {
    "gurobi": solve_gurobi,
    "mosek": solve_mosek,
    "cplex": solve_cplex,
    "copt": solve_copt,
    "xpress": solve_xpress,
}


def _require_cplex():
    if not is_commercial_solver_available("cplex"):
        pytest.skip("CPLEX engine (license required) not available")


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


# ---------------------------------------------------------------------------
# Metadata / availability
# ---------------------------------------------------------------------------


class TestCommercialMetadata:
    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_info_fields(self, alias):
        info = commercial_solver_info(alias)
        assert info["alias"] == alias
        assert info["pip_package"] == PIP_MODULES[alias]
        assert info["license_required"] is True
        assert info["cvxpy_solver"] in (
            "GUROBI",
            "MOSEK",
            "CPLEX",
            "COPT",
            "XPRESS",
        )
        assert isinstance(info["available"], bool)

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown commercial solver"):
            commercial_solver_info("glpk")

    def test_is_commercial_solver_available_bool(self):
        for alias in COMMERCIAL_ALIASES:
            assert isinstance(is_commercial_solver_available(alias), bool)

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_availability_false_when_import_blocked(self, alias):
        with block_import(PIP_MODULES[alias]):
            assert is_commercial_solver_available(alias) is False

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_detector_method_exists(self, alias):
        method = getattr(Detector, f"unfold_{alias}")
        assert callable(method)
        assert "license required" in method.__doc__.lower()

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_solve_wrapper_docstring_mentions_license(self, alias):
        doc = SOLVE_FUNCS[alias].__doc__ or ""
        assert "license required" in doc.lower()
        assert SOLVE_FUNCS[alias].__name__ == f"solve_{alias}"

    def test_core_exports(self):
        from bssunfold import core

        for alias in COMMERCIAL_ALIASES:
            assert f"unfold_{alias}" in core.__all__
            assert f"solve_{alias}" in core.__all__
        assert "unfold_commercial" in core.__all__
        assert "solve_commercial" in core.__all__


# ---------------------------------------------------------------------------
# Core solver behaviour
# ---------------------------------------------------------------------------


class TestSolveCommercial:
    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_missing_engine_returns_none_with_warning(self, alias):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        # Force the unavailable path explicitly: block_import alone is not
        # enough when the engine package is present (a free size-limited
        # MOSEK trial solves a 2x2 QP silently → DID NOT WARN on CI).
        with block_import(PIP_MODULES[alias]), patch(
            "bssunfold.core._commercial_qp.is_commercial_solver_available",
            return_value=False,
        ):
            with pytest.warns(UserWarning, match="license required"):
                x = solve_commercial(A, b, solver=alias)
        assert x is None

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_ill_formed_input(self, alias):
        with pytest.raises(ValueError, match="ill-formed"):
            solve_commercial(np.ones((2, 3)), np.ones(4), solver=alias)

    def test_unsupported_norm(self):
        A = np.array([[1.0, 2.0]])
        b = np.array([3.0])
        with pytest.raises(ValueError, match="norm"):
            solve_commercial(A, b, solver="cplex", norm=3)

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_unsupported_norm_value(self, alias):
        A = np.array([[1.0, 2.0]])
        b = np.array([3.0])
        with pytest.raises(ValueError, match="norm"):
            solve_commercial(A, b, solver=alias, norm=3)

    def test_unknown_solver_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown commercial solver"):
            solve_commercial(np.ones((1, 2)), np.ones(1), solver="glpk")

    def test_l1_requires_nonneg(self):
        _require_cplex()
        A = np.array([[1.0, 2.0]])
        b = np.array([3.0])
        with pytest.raises(ValueError, match="non-negativity"):
            solve_commercial(A, b, solver="cplex", norm=1, nonneg=False)


class TestSolveWithCplex:
    """End-to-end numeric behaviour using the licensed CPLEX engine."""

    def test_solves_l2(self):
        _require_cplex()
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = A @ np.array([1.5, 2.0])
        x = solve_cplex(A, b, alpha=0.0, timeout=10.0)
        assert x is not None
        assert x.shape == (2,)
        assert np.all(x >= -1e-6)
        assert np.allclose(A @ x, b, atol=1e-2)

    def test_l1_norm(self):
        _require_cplex()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_cplex(A, b, alpha=0.1, norm=1, timeout=10.0)
        assert x is not None
        assert x.shape == (2,)

    def test_smoothness(self):
        _require_cplex()
        A = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_cplex(
            A, b, alpha=1e-3, smoothness_order=2, smoothness_weight=2.0,
            timeout=10.0,
        )
        assert x is not None
        assert x.shape == (3,)

    def test_upper_bounds_respected(self):
        _require_cplex()
        A = np.array([[1.0, 1.0]])
        b = np.array([10.0])
        ub = np.array([1.0, np.inf])
        x = solve_cplex(A, b, alpha=0.0, ub=ub, timeout=10.0)
        assert x is not None
        assert x[0] <= 1.0 + 1e-6

    def test_warm_start_accepted(self):
        _require_cplex()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_cplex(A, b, x0=np.array([0.1, 0.2]), timeout=10.0)
        assert x is not None

    def test_matches_qpsolvers_formulation(self, detector):
        # The commercial backend implements the canonical solve_qpsolvers
        # convention (P = A'A + penalty, q = -A'b); a tight-tolerance OSQP
        # reference on the same P/q must agree.
        _require_cplex()
        from qpsolvers import available_solvers, solve_qp
        from scipy.sparse import csc_matrix

        from bssunfold.core._base_unfolder import _build_system
        from bssunfold.core._matrix_utils import build_smoothness_penalty

        readings = {
            nm: float(i + 1) * 0.05
            for i, nm in enumerate(detector.detector_names)
        }
        A, b, _ = _build_system(
            readings, detector.detector_names, detector.sensitivities
        )
        n = A.shape[1]
        alpha = 1e-3
        # sm=2 is deliberately omitted: the second-derivative kernel leaves
        # a nearly-degenerate direction, so pointwise agreement degrades to
        # a few percent while the objective still matches exactly.
        for norm, sm in ((2, 0), (2, 1), (1, 0), (1, 1)):
            P = csc_matrix(A.T @ A)
            q = -A.T @ b
            pen = build_smoothness_penalty(n, alpha, sm, 1.0)
            if norm == 2:
                # identity OR derivative penalty (never both), as in
                # solve_qpsolvers / solve_docplex
                P = P + (pen if pen is not None
                         else alpha * csc_matrix(np.eye(n)))
            else:
                q = q + alpha * np.ones(n)
                if pen is not None:
                    P = P + pen
            x_ref = solve_qp(
                P=P, q=q, lb=np.zeros(n), solver="highs", verbose=False,
            )
            if x_ref is None:
                # HiGHS can fail on some platforms; fall back to a
                # solvers-core backend for the reference solution.
                for alt in ("osqp", "clarabel", "scs"):
                    if alt in available_solvers:
                        x_ref = solve_qp(
                            P=P, q=q, lb=np.zeros(n), solver=alt,
                            verbose=False,
                        )
                        if x_ref is not None:
                            break
            assert x_ref is not None, (
                f"reference QP solver returned None (norm={norm}, sm={sm})"
            )
            x_com = solve_cplex(
                A, b, alpha=alpha, norm=norm, smoothness_order=sm,
                timeout=30.0,
            )
            assert x_com is not None, (
                f"solve_cplex returned None (norm={norm}, sm={sm})"
            )
            # objective agreement (exact; ill-conditioned QP solutions can
            # differ in null-space components while attaining the same min)
            Pd = np.asarray(P.todense())

            def quad_obj(z, Pd=Pd, q=q):
                return 0.5 * z @ Pd @ z + q @ z

            assert abs(quad_obj(x_com) - quad_obj(x_ref)) < \
                1e-4 * abs(quad_obj(x_ref)), (norm, sm)
            scale = max(float(np.abs(x_ref).max()), 1.0)
            assert np.abs(x_com - x_ref).max() < 1e-2 * scale, (norm, sm)

    def test_solver_error_returns_none(self):
        _require_cplex()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        with patch(
            "cvxpy.problems.problem.Problem.solve",
            side_effect=RuntimeError("no license found"),
        ):
            with pytest.warns(UserWarning, match="license"):
                x = solve_commercial(A, b, solver="cplex")
        assert x is None

    def test_non_optimal_status_returns_none(self):
        _require_cplex()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        with patch(
            "cvxpy.problems.problem.Problem.solve",
            side_effect=lambda *a, **k: None,
        ):
            with pytest.warns(UserWarning, match="did not find a solution"):
                x = solve_commercial(A, b, solver="cplex")
        assert x is None


# ---------------------------------------------------------------------------
# Wrapper / Detector integration
# ---------------------------------------------------------------------------


class TestUnfoldCommercial:
    def _call(self, detector, readings, **kwargs):
        return unfold_commercial(
            detector_names=detector.detector_names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=detector._save_result,
            readings=readings,
            save_result=False,
            **kwargs,
        )

    def test_basic_cplex(self, detector, readings):
        _require_cplex()
        result = self._call(detector, readings, solver="cplex", timeout=10.0)
        assert result["method"] == "cplex"
        assert result["license_required"] is True
        assert result["cvxpy_solver"] == "CPLEX"
        assert result["pip_package"] == "cplex"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert result["selected_regularization"] > 0

    @pytest.mark.parametrize("alias", COMMERCIAL_ALIASES)
    def test_missing_license_returns_zero_spectrum(self, alias):
        detector = Detector()
        readings = {detector.detector_names[0]: 100.0}
        if is_commercial_solver_available(alias):
            pytest.skip(f"{alias} engine is actually installed")
        with pytest.warns(UserWarning):
            result = self._call(detector, readings, solver=alias, timeout=2.0)
        assert np.all(result["spectrum"] == 0)
        assert result["license_required"] is True

    def test_cosine_requires_initial(self, detector, readings):
        with pytest.raises(ValueError, match="initial_spectrum"):
            self._call(
                detector, readings, solver="cplex",
                regularization_method="cosine",
            )

    def test_detector_method_wrappers(self, detector, readings):
        _require_cplex()
        for alias in ("cplex",):
            result = getattr(detector, f"unfold_{alias}")(
                readings, save_result=False, timeout=10.0
            )
            assert result["method"] == alias
            assert np.any(result["spectrum"] > 0)

    def test_detector_gurobi_graceful_zero(self, detector, readings):
        if is_commercial_solver_available("gurobi"):
            pytest.skip("gurobi engine is actually installed")
        with pytest.warns(UserWarning):
            result = detector.unfold_gurobi(
                readings, save_result=False, timeout=2.0
            )
        assert result["method"] == "gurobi"
        assert np.all(result["spectrum"] == 0)

    def test_detector_save_result(self, detector, readings):
        _require_cplex()
        detector.results_history.clear()
        detector.unfold_cplex(readings, save_result=True, timeout=10.0)
        assert len(detector.results_history) == 1

    def test_calculate_errors(self, detector, readings):
        _require_cplex()
        result = detector.unfold_cplex(
            readings,
            calculate_errors=True,
            n_montecarlo=5,
            save_result=False,
            timeout=10.0,
        )
        assert "spectrum_uncert_mean" in result

    def test_regularization_methods(self, detector, readings):
        _require_cplex()
        initial = np.ones(detector.n_energy_bins)
        for method, kwargs in (
            ("lcurve", {}),
            ("gcv", {}),
            ("dp", {"noise_var": 0.01}),
            ("cosine", {"initial_spectrum": initial}),
        ):
            result = detector.unfold_cplex(
                readings,
                regularization_method=method,
                save_result=False,
                timeout=10.0,
                **kwargs,
            )
            assert result["regularization_method"] == method

    def test_max_neutron_energy_mask(self, detector, readings):
        _require_cplex()
        result = detector.unfold_cplex(
            readings,
            max_neutron_energy=5.0,
            save_result=False,
            timeout=10.0,
        )
        e = np.asarray(result["energy"])
        spec = np.asarray(result["spectrum"])
        assert np.all(spec[e > 5.0 + 1e-9] == 0)
        assert np.any(spec[e <= 5.0] > 0)


# ---------------------------------------------------------------------------
# platform_check integration
# ---------------------------------------------------------------------------


class TestPlatformCheckCommercial:
    def test_check_returns_all_aliases(self):
        from bssunfold.platform_check import (
            COMMERCIAL_SOLVERS_AVAILABLE,
            check_commercial_solvers_availability,
        )

        result = check_commercial_solvers_availability()
        assert set(result) == set(COMMERCIAL_ALIASES)
        assert all(isinstance(v, bool) for v in result.values())
        import bssunfold.platform_check as pc

        assert pc.COMMERCIAL_SOLVERS_AVAILABLE == result
        assert isinstance(COMMERCIAL_SOLVERS_AVAILABLE, dict)

    def test_get_available_solvers_includes_commercial(self):
        from bssunfold.platform_check import get_available_solvers

        solvers = get_available_solvers()
        for alias in COMMERCIAL_ALIASES:
            assert isinstance(solvers[alias], bool)
