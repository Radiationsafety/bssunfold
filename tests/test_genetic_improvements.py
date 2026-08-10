"""Tests for the genetic/meta-heuristic unfolding improvements.

Covers the TGASU-style extensions: the two-step coarse-to-fine scheme,
the NSGA-II multi-objective solver, post-processing smoothers, the
arithmetic/iterative GA operators, and the ``extra_starting`` injection.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core.unfold_genetic import (
    _coarsen_columns,
    _make_starting_solutions,
    _split_coarse,
)


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {
        "3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034,
    }


@pytest.fixture
def selected(detector, readings):
    return [name for name in detector.detector_names if name in readings]


@pytest.fixture
def A(detector, selected):
    return np.array([detector.sensitivities[name] for name in selected])


@pytest.fixture
def b(readings, selected):
    return np.array([readings[name] for name in selected], dtype=float)


def test_unfold_genetic_two_step(detector, readings):
    """The two-step coarse-to-fine scheme runs and honors selection."""
    result = detector.unfold_genetic(
        readings, solver="pso", epoch=40, pop_size=24, two_step=True,
        save_result=False,
    )
    assert result["two_step"] is True
    assert np.all(result["spectrum"] >= 0)
    assert len(result["spectrum"]) == detector.n_energy_bins


def test_unfold_genetic_two_step_ga_tgasu(detector, readings):
    """TGASU-style two-step with arithmetic crossover + iterative mutation."""
    result = detector.unfold_genetic(
        readings, solver="ga", epoch=40, pop_size=24, two_step=True,
        crossover="arithmetic", mutation="iterative", save_result=False,
    )
    assert result["two_step"] is True
    assert result["crossover"] == "arithmetic"
    assert result["mutation"] == "iterative"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_nsga2(detector, readings):
    """The NSGA-II multi-objective solver runs with Pareto selection."""
    for select in ("knee", "min_residual", "max_entropy"):
        result = detector.unfold_genetic(
            readings, solver="nsga2", pareto_select=select, epoch=30,
            pop_size=24, save_result=False,
        )
        assert result["solver"] == "nsga2"
        assert result["pareto_select"] == select
        assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_smoothers(detector, readings):
    """Each post-processing smoother runs and keeps the spectrum positive."""
    for smoother in ("gaussian", "gaussian_mbc", "second_difference"):
        result = detector.unfold_genetic(
            readings, solver="pso", smoother=smoother, sigma_smooth=2.0,
            epoch=30, pop_size=20, save_result=False,
        )
        assert result["smoother"] == smoother
        assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_smoother_alias(detector, readings):
    """The 'mbc' alias normalizes to 'gaussian_mbc'."""
    result = detector.unfold_genetic(
        readings, solver="pso", smoother="mbc", epoch=30, pop_size=20,
        save_result=False,
    )
    assert result["smoother"] == "gaussian_mbc"


def test_unfold_genetic_tgasu_operators_flat(detector, readings):
    """Arithmetic crossover and iterative mutation on a flat one-stage run."""
    result = detector.unfold_genetic(
        readings, solver="ga", crossover="arithmetic", mutation="iterative",
        epoch=30, pop_size=20, save_result=False,
    )
    assert result["crossover"] == "arithmetic"
    assert result["mutation"] == "iterative"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_invalid_crossover(detector, readings):
    """Unknown crossover operators are rejected."""
    with pytest.raises(ValueError, match="crossover"):
        detector.unfold_genetic(
            readings, solver="ga", crossover="bogus", save_result=False
        )


def test_unfold_genetic_invalid_mutation(detector, readings):
    """Unknown mutation operators are rejected."""
    with pytest.raises(ValueError, match="mutation"):
        detector.unfold_genetic(
            readings, solver="ga", mutation="bogus", save_result=False
        )


def test_unfold_genetic_invalid_pareto_select(detector, readings):
    """Unknown pareto_select values are rejected."""
    with pytest.raises(ValueError, match="pareto_select"):
        detector.unfold_genetic(
            readings, solver="nsga2", pareto_select="bogus",
            save_result=False,
        )


def test_two_step_two_solvers_agree_with_metadata(detector, readings):
    """two_step + nsga2 combo runs and records both selections."""
    result = detector.unfold_genetic(
        readings, solver="nsga2", two_step=True, n_coarse=15, epoch=30,
        pop_size=24, save_result=False,
    )
    assert result["two_step"] is True
    assert result["solver"] == "nsga2"
    assert np.all(result["spectrum"] >= 0)


def test_coarsen_columns_shape(A):
    """_coarsen_columns groups columns into the requested bin count."""
    coarse = _coarsen_columns(A, 8)
    assert coarse.shape == (A.shape[0], 8)


def test_coarsen_columns_reduces_resolution(A, b):
    """The coarse response still approximates the fine one."""
    from bssunfold.core.unfold_genetic import _coarsen_columns, _split_coarse

    coarse = _coarsen_columns(A, 8)
    n = A.shape[1]
    x = np.ones(n)
    fine = A @ x
    coarse_y = coarse @ np.ones(8)
    assert np.allclose(coarse_y, fine, rtol=0.15)


def test_split_coarse_preserves_fluence(A, b):
    """_split_coarse distributes coarse-bin totals across the fine grid."""
    n = A.shape[1]
    coarse_totals = np.ones(8)
    split = _split_coarse(coarse_totals, n)
    assert split.shape == (n,)
    assert np.allclose(split.sum(), coarse_totals.sum())


def test_split_coarse_zero_handling(A):
    """_split_coarse handles zero coarse bins without errors."""
    n = A.shape[1]
    split = _split_coarse(np.zeros(8), n)
    assert split.shape == (n,)
    assert np.all(split >= 0)


def test_make_starting_solutions_injects_extra(detector):
    """extra individuals are injected without shifting the search box."""
    n = detector.n_energy_bins
    seed = np.full(n, 1.0)
    lb = np.full(n, -2.0 * np.log(10.0))
    ub = np.full(n, 2.0 * np.log(10.0))
    extra = np.linspace(1.0, 2.0, n)

    base = _make_starting_solutions(seed, lb, ub, 10)
    with_extra = _make_starting_solutions(
        seed, lb, ub, 10, extra=extra
    )
    assert len(with_extra) == len(base)
    assert np.allclose(with_extra[0], np.log(seed))
    assert np.allclose(with_extra[1], np.log(extra))


def test_make_starting_solutions_extra_masked_inside_bounds(detector):
    """Injected extras outside the log-space box are clipped."""
    n = detector.n_energy_bins
    seed = np.full(n, 1.0)
    lb = np.full(n, -2.0 * np.log(10.0))
    ub = np.full(n, 2.0 * np.log(10.0))
    extra = np.full(n, 1e9)

    with_extra = _make_starting_solutions(seed, lb, ub, 10, extra=extra)
    assert np.all(with_extra >= lb)
    assert np.all(with_extra <= ub)


def test_two_step_converges_to_good_fit(detector, readings, A, b):
    """Two-step residual is bounded by the unregularized measurement norm."""
    r_two = detector.unfold_genetic(
        readings, solver="pso", epoch=40, pop_size=24, two_step=True,
        save_result=False,
    )
    assert r_two["residual_norm"] < np.linalg.norm(b)
    assert r_two["residual_norm"] >= 0
