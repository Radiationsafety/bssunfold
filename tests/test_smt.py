"""Tests for the SMT-based unfolding method.

The ``z3-solver`` package is an optional backend installed in the dev group.
These tests cover the four exact solvers ported from the linearEqSolver
library (integer/rational, single/all solutions) plus the ``solve_smt`` core
solver and the ``unfold_smt`` wrapper exposed both on the ``Detector`` class
and as a module-level function.
"""

import builtins
from unittest.mock import patch

import numpy as np
import pytest
import z3

pytest.importorskip("z3")

from bssunfold import Detector  # noqa: E402


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


def _system(detector, readings):
    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)
    return A, b


def test_solve_integer_linear_eqs_repo_example():
    """The exact integer example from the linearEqSolver README."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs

    solution = solve_integer_linear_eqs(
        [[2, 3, 4], [6, -3, 9], [2, 0, 1]], [20, -6, 8]
    )
    assert solution == [5, 6, -2]


def test_solve_integer_linear_eqs_no_solution():
    """Inconsistent integer systems return None."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs

    assert solve_integer_linear_eqs([[1], [1]], [2, 3]) is None


def test_solve_integer_linear_eqs_ill_formed():
    """Ill-formed input raises ValueError."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs

    with pytest.raises(ValueError, match="ill-formed"):
        solve_integer_linear_eqs(np.ones((2, 3)), np.ones(4))


def test_solve_integer_linear_eqs_all_underspecified():
    """Underspecified systems return the requested number of solutions."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs_all

    solutions = solve_integer_linear_eqs_all(
        [[2, 3, 4], [6, -3, 9]], [20, -6], max_solutions=3
    )
    assert len(solutions) == 3
    assert len({tuple(s) for s in solutions}) == 3
    for x in solutions:
        assert 2 * x[0] + 3 * x[1] + 4 * x[2] == 20
        assert 6 * x[0] - 3 * x[1] + 9 * x[2] == -6


def test_solve_integer_linear_eqs_all_unique():
    """Uniquely determined systems return a single solution."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs_all

    assert solve_integer_linear_eqs_all(
        [[1, 0], [0, 1]], [1, 2], max_solutions=5
    ) == [[1, 2]]


def test_solve_integer_linear_eqs_all_zero_max():
    """max_solutions=0 returns an empty list."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs_all

    assert solve_integer_linear_eqs_all([[1]], [1], max_solutions=0) == []


def test_solve_rational_linear_eqs_repo_example():
    """The exact rational example from the linearEqSolver README."""
    from bssunfold.core.unfold_smt import solve_rational_linear_eqs

    solution = solve_rational_linear_eqs([[2.4, 3.6], [7.2, -5]], [12, -8.5])
    assert solution is not None
    x, y = solution
    assert x == pytest.approx(245 / 316)
    assert y == pytest.approx(445 / 158)


def test_solve_rational_linear_eqs_no_solution():
    """Inconsistent rational systems return None."""
    from bssunfold.core.unfold_smt import solve_rational_linear_eqs

    assert solve_rational_linear_eqs([[1.0], [1.0]], [2.0, 3.0]) is None


def test_solve_rational_linear_eqs_all_underspecified():
    """Rational all-solutions variant returns distinct feasible solutions."""
    from bssunfold.core.unfold_smt import solve_rational_linear_eqs_all

    solutions = solve_rational_linear_eqs_all(
        [[2.4, 3.6]], [12], max_solutions=3
    )
    assert len(solutions) == 3
    assert len({tuple(s) for s in solutions}) == 3
    for x, y in solutions:
        assert 2.4 * x + 3.6 * y == pytest.approx(12)


def test_build_constraints_ill_formed():
    """buildConstraints port raises on ill-formed input."""
    from bssunfold.core.unfold_smt import _build_constraints, _to_real_value

    xs = [z3.Real("x0"), z3.Real("x1")]
    with pytest.raises(ValueError, match="ill-formed"):
        _build_constraints([], [], [], z3, _to_real_value)
    with pytest.raises(ValueError, match="ill-formed"):
        _build_constraints(xs, [[1, 2, 3]], [1], z3, _to_real_value)
    with pytest.raises(ValueError, match="ill-formed"):
        _build_constraints(xs, [[1, 0], [0, 1]], [1], z3, _to_real_value)


def test_build_constraints_nonneg():
    """buildConstraints adds non-negativity constraints when requested."""
    from bssunfold.core.unfold_smt import _build_constraints, _to_real_value

    xs = [z3.Real("x0")]
    constraints = _build_constraints(
        xs, [[1]], [1], z3, _to_real_value, nonneg=True
    )
    assert len(constraints) == 2


class _FakeOptimize:
    """Minimal stand-in for z3.Optimize with scripted check results."""

    def __init__(self, results):
        self._results = list(results)
        self._calls = 0

    def set(self, *args, **kwargs):
        pass

    def add(self, *args):
        pass

    def minimize(self, *args):
        return object()

    def lower(self, *args):
        return 0

    def check(self):
        idx = min(self._calls, len(self._results) - 1)
        self._calls += 1
        return self._results[idx]


def test_solve_smt_random_state():
    """random_state is forwarded to the z3 optimizer."""
    from bssunfold.core.unfold_smt import solve_smt

    spectrum = solve_smt(
        np.array([[1.0, 2.0]]), np.array([10.0]), random_state=42
    )
    assert spectrum == pytest.approx([0.0, 5.0])


def test_solve_smt_parameter_error_warns():
    """A failing optimizer parameter falls back to a zero vector."""
    from bssunfold.core.unfold_smt import solve_smt

    with patch.object(z3, "set_param", side_effect=RuntimeError("boom")):
        with pytest.warns(UserWarning, match="SMT solver failed"):
            spectrum = solve_smt(
                np.array([[1.0, 2.0]]), np.array([10.0]), random_state=1
            )
    assert np.all(spectrum == 0)


def test_solve_smt_no_solution_warns():
    """An unsat first check warns and returns a zero vector."""
    from bssunfold.core.unfold_smt import solve_smt

    with patch.object(z3, "Optimize", return_value=_FakeOptimize([z3.unknown])):
        with pytest.warns(UserWarning, match="could not find a solution"):
            spectrum = solve_smt(np.array([[1.0, 2.0]]), np.array([10.0]))
    assert np.all(spectrum == 0)


def test_solve_smt_refine_failure_warns():
    """An unsat refinement check warns and returns a zero vector.

    ``solve_smt`` first attempts the L2 objective (KKT) and then the L1
    objective; the L2 attempt consumes the first ``sat`` from the shared
    mock, the L1 fallback performs the sat + unsat refinement sequence.
    """
    from bssunfold.core.unfold_smt import solve_smt

    with patch.object(
        z3, "Optimize", return_value=_FakeOptimize([z3.sat, z3.sat, z3.unknown])
    ):
        with pytest.warns(UserWarning, match="could not refine"):
            spectrum = solve_smt(np.array([[1.0, 2.0]]), np.array([10.0]))
    assert np.all(spectrum == 0)


def test_solve_smt_returns_array(detector, readings):
    """The core solve_smt solver returns a raw spectrum array."""
    from bssunfold.core.unfold_smt import solve_smt

    A, b = _system(detector, readings)
    spectrum = solve_smt(A, b)

    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (detector.n_energy_bins,)
    assert np.all(spectrum >= 0)
    assert np.linalg.norm(A @ spectrum - b) < 1.0


def test_solve_smt_minimizes_fluence():
    """The lexicographic objective picks the minimal-fluence solution."""
    from bssunfold.core.unfold_smt import solve_smt

    spectrum = solve_smt(np.array([[1.0, 2.0]]), np.array([10.0]))
    assert spectrum == pytest.approx([0.0, 5.0])


def test_solve_smt_nonneg_disabled():
    """nonneg=False allows negative components."""
    from bssunfold.core.unfold_smt import solve_smt

    spectrum = solve_smt(np.array([[1.0, 2.0]]), np.array([10.0]), nonneg=False)
    assert spectrum.shape == (2,)
    assert np.linalg.norm(np.array([1.0, 2.0]) @ spectrum - 10.0) < 1e-6


def test_solve_smt_ill_formed():
    """Ill-formed input raises ValueError."""
    from bssunfold.core.unfold_smt import solve_smt

    with pytest.raises(ValueError, match="ill-formed"):
        solve_smt(np.ones((2, 4)), np.ones(3))
    with pytest.raises(ValueError, match="ill-formed"):
        solve_smt(np.ones(3), np.ones(3))


def test_solve_smt_failure_returns_zero(detector, readings):
    """Solver exceptions are caught and return a zero spectrum with warning."""
    from bssunfold.core.unfold_smt import solve_smt

    A, b = _system(detector, readings)

    with patch(
        "bssunfold.core.unfold_smt._to_real_value",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.warns(UserWarning, match="failed"):
            spectrum = solve_smt(A, b)
    assert np.all(spectrum == 0)


def test_solve_smt_import_error():
    """Missing z3 raises a helpful ImportError."""
    from bssunfold.core.unfold_smt import solve_smt

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "z3" or name.startswith("z3."):
            raise ImportError("z3 not installed")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="z3-solver is required"):
            solve_smt(np.eye(3), np.ones(3))


def test_exact_solvers_import_error():
    """Missing z3 raises a helpful ImportError in the ported solvers."""
    from bssunfold.core.unfold_smt import solve_integer_linear_eqs

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "z3" or name.startswith("z3."):
            raise ImportError("z3 not installed")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="z3-solver is required"):
            solve_integer_linear_eqs([[1]], [1])


def test_unfold_smt_basic(detector, readings):
    """Basic unfold_smt call returns a standardized result."""
    result = detector.unfold_smt(readings, save_result=False)

    assert isinstance(result, dict)
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert result["method"] == "SMT"

    assert isinstance(result["energy"], np.ndarray)
    assert isinstance(result["spectrum"], np.ndarray)
    assert isinstance(result["residual_norm"], float)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)

    assert result["nonneg"] is True
    assert result["timeout_ms"] == 10000


def test_unfold_smt_nonneg_false(detector, readings):
    """nonneg=False is honored in the unfold wrapper."""
    result = detector.unfold_smt(readings, nonneg=False, save_result=False)
    assert result["nonneg"] is False
    assert len(result["spectrum"]) == detector.n_energy_bins


def test_unfold_smt_initial_spectrum_wrong_length(detector, readings):
    """Mismatched initial spectrum length raises ValueError."""
    with pytest.raises(ValueError, match="must match number of energy bins"):
        detector.unfold_smt(
            readings, initial_spectrum=np.ones(5), save_result=False
        )


def test_unfold_smt_with_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_smt(
        readings,
        calculate_errors=True,
        n_montecarlo=3,
        save_result=False,
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert "spectrum_uncert_min" in result
    assert "spectrum_uncert_max" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_smt_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_smt(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "SMT"


def test_unfold_smt_exported(detector):
    """unfold_smt is a Detector method."""
    assert hasattr(Detector, "unfold_smt")


def test_core_exports():
    """SMT solvers are exported from bssunfold.core."""
    from bssunfold.core import (
        solve_integer_linear_eqs,
        solve_integer_linear_eqs_all,
        solve_rational_linear_eqs,
        solve_rational_linear_eqs_all,
        solve_smt,
        unfold_smt,
    )

    assert solve_smt is not None
    assert unfold_smt is not None
    assert solve_integer_linear_eqs is not None
    assert solve_integer_linear_eqs_all is not None
    assert solve_rational_linear_eqs is not None
    assert solve_rational_linear_eqs_all is not None


def test_unfold_combined_smt(detector, readings):
    """'smt' can be used in a combined unfolding pipeline."""
    from bssunfold.core.unfold_combined import unfold_combined

    result = unfold_combined(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        pipeline=[
            {
                "method": "smt",
                "params": {"save_result": False},
            }
        ],
        verbose=False,
    )
    assert "spectrum" in result
    assert np.all(result["spectrum"] >= 0)


# Infeasible-under-nonnegativity system where the L1 and L2 objectives
# select different solutions (L2 gives the least-squares optimum).
_AL2 = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
)
_BL2 = np.array([2.0, -1.0, 0.5, 0.5])


def test_solve_smt_l2_matches_nnls():
    """The default L2 objective yields the exact least-squares residual."""
    from scipy.optimize import nnls

    from bssunfold.core.unfold_smt import solve_smt

    x_ref, _ = nnls(_AL2, _BL2)
    ref_resid = np.linalg.norm(_AL2 @ x_ref - _BL2)

    spectrum = solve_smt(_AL2, _BL2)
    resid = np.linalg.norm(_AL2 @ spectrum - _BL2)
    assert resid == pytest.approx(ref_resid, rel=1e-6)


def test_solve_smt_l2_beats_l1():
    """L2 objective gives a smaller L2 residual than the L1 objective."""
    from bssunfold.core.unfold_smt import solve_smt

    x_l2 = solve_smt(_AL2, _BL2, objective="l2")
    x_l1 = solve_smt(_AL2, _BL2, objective="l1")
    resid_l2 = np.linalg.norm(_AL2 @ x_l2 - _BL2)
    resid_l1 = np.linalg.norm(_AL2 @ x_l1 - _BL2)
    assert resid_l2 < resid_l1


def test_solve_smt_l2_nonneg_false_free_ls():
    """nonneg=False L2 path matches the unconstrained least-squares optimum."""
    from bssunfold.core.unfold_smt import solve_smt

    Au = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    bu = np.array([3.0, 1.0, 2.5, 0.7])
    x_ref, *_ = np.linalg.lstsq(Au, bu, rcond=None)
    ref_resid = np.linalg.norm(Au @ x_ref - bu)

    spectrum = solve_smt(Au, bu, nonneg=False)
    resid = np.linalg.norm(Au @ spectrum - bu)
    assert resid == pytest.approx(ref_resid, rel=1e-6)


def test_solve_smt_objective_l1_backcompat():
    """objective='l1' reproduces the historical lexicographic solution."""
    from bssunfold.core.unfold_smt import solve_smt

    spectrum = solve_smt(
        np.array([[1.0, 2.0]]), np.array([10.0]), objective="l1"
    )
    assert spectrum == pytest.approx([0.0, 5.0])


def test_solve_smt_unknown_objective_warns():
    """An unknown objective falls back to 'l2' with a warning."""
    from bssunfold.core.unfold_smt import solve_smt

    with pytest.warns(UserWarning, match="using 'l2'"):
        spectrum = solve_smt(
            np.array([[1.0, 2.0]]), np.array([10.0]), objective="sparse"
        )
    assert spectrum.shape == (2,)
    assert np.linalg.norm(np.array([1.0, 2.0]) @ spectrum - 10.0) < 1e-6


def test_unfold_smt_objective_metadata(detector, readings):
    """The objective is reported in the result metadata."""
    result_l2 = detector.unfold_smt(readings, save_result=False)
    assert result_l2["objective"] == "l2"

    result_l1 = detector.unfold_smt(
        readings, objective="l1", save_result=False
    )
    assert result_l1["objective"] == "l1"
