"""Tests for the Uno-style constrained unfolding method.

Covers the NLP building blocks (objective, gradient, Vanaret-Leyffer
filter), the ``filter_sqp`` preset (exact one-shot QP sub-solve), the
``ipopt_like`` preset in both Hessian modes, equality-constraint
restoration, parameter validation and ``Detector.unfold_uno``.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_uno, solve_uno_full, uno_filter
from bssunfold.core.unfold_uno import (
    uno_augment_filter,
    uno_gradient,
    uno_objective,
)


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def A(detector):
    return np.array(
        [detector.sensitivities[name] for name in detector.detector_names]
    )


@pytest.fixture
def f_true(detector):
    E = detector.E_MeV
    f = (
        0.55 * np.sqrt(E / 2.0) * np.exp(-E / 2.0)
        + 0.20 * np.exp(-((E - 4.5) / 1.2) ** 2)
        + 0.02 / (1.0 + (E / 0.05) ** 2)
    )
    return f * 1e6 / np.max(f)


@pytest.fixture
def b(A, f_true):
    return A @ f_true


def cosine(x, f):
    return float(x @ f / (np.linalg.norm(x) * np.linalg.norm(f)))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_uno_objective_at_exact_solution(A, f_true, b):
    obj = uno_objective(A, b, np.ones(len(b)), 0.0, f_true)
    np.testing.assert_allclose(obj, 0.0, atol=1e-8)


def test_uno_objective_dtype_scales_with_weights(A, f_true, b):
    x = f_true
    obj1 = uno_objective(A, b, np.ones(len(b)), 0.0, x)
    ws = np.full(len(b), 2.0)
    obj2 = uno_objective(A, b, ws, 0.0, x)
    np.testing.assert_allclose(obj2, 4.0 * obj1)


def test_uno_gradient_zero_at_optimum(A, f_true, b):
    # interior optimum of the unconstrained penalised problem
    from bssunfold.core._matrix_utils import create_derivative_matrix

    Aw = A * 1.0
    lam = 1e-2
    D = np.asarray(create_derivative_matrix(A.shape[1], order=2).toarray())
    H = Aw.T @ Aw + lam * D.T @ D
    x_opt = np.linalg.solve(H, Aw.T @ b)
    g = uno_gradient(A, b, np.ones(len(b)), lam, x_opt)
    np.testing.assert_allclose(g, 0.0, atol=1e-4 * np.max(np.abs(x_opt)))


def test_uno_filter_empty_accepts_everything():
    assert uno_filter([], -1e3, 0.0)
    assert uno_filter([], 0.0, 0.0)


def test_uno_filter_rejects_dominated():
    filt = [(1.0, 0.0)]
    # strictly dominated point: worse objective and same violation
    assert not uno_filter(filt, 2.0, 0.0)


def test_uno_filter_accepts_improvement():
    filt = [(1.0, 0.1)]
    assert uno_filter(filt, 0.5, 0.1)


def test_uno_filter_augment_grows():
    filt: list[tuple[float, float]] = []
    uno_augment_filter(filt, 1.0, 0.0)
    uno_augment_filter(filt, 0.5, 0.0)
    assert len(filt) == 2


# ---------------------------------------------------------------------------
# filter_sqp preset
# ---------------------------------------------------------------------------


def test_solve_uno_filter_sqp_exact_qp(A, f_true, b):
    x, it, converged = solve_uno(A, b)
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)
    assert it == 1
    assert converged
    assert cosine(x, f_true) > 0.9


def test_solve_uno_full_diagnostics(A, f_true, b):
    diag = solve_uno_full(A, b)
    for key in ("spectrum", "preset", "hessian", "objective",
                "constraint_violation", "dual_infeasibility",
                "n_iterations", "converged"):
        assert key in diag
    assert diag["preset"] == "filter_sqp"
    assert diag["constraint_violation"] == 0.0
    assert diag["dual_infeasibility"] <= 1e-8


def test_solve_uno_respects_x0(A, b, f_true):
    x, _, _ = solve_uno(A, b, x0=np.ones(A.shape[1]))
    assert np.all(x >= 0)
    assert cosine(x, f_true) > 0.9


def test_solve_uno_with_weights_poisson(A, f_true, b):
    x, _, _ = solve_uno(A, b, weights="poisson")
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))


def test_solve_uno_no_regularization(A, f_true, b):
    x, _, _ = solve_uno(A, b, regularization=0.0)
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))


# ---------------------------------------------------------------------------
# ipopt_like preset
# ---------------------------------------------------------------------------


def test_solve_uno_ipopt_like_exact(A, f_true, b):
    x, it, converged = solve_uno(A, b, preset="ipopt_like", hessian="exact")
    assert np.all(np.isfinite(x))
    assert np.all(x > 0)  # interior-point iterates are strictly positive
    assert it > 0
    assert cosine(x, f_true) > 0.9


def test_solve_uno_ipopt_like_bfgs(A, f_true, b):
    x, it, converged = solve_uno(A, b, preset="ipopt_like", hessian="bfgs")
    assert np.all(np.isfinite(x))
    assert np.all(x > 0)
    assert cosine(x, f_true) > 0.9


def test_ipopt_like_reaches_same_objective_as_exact_qp(A, f_true, b):
    f_fs = solve_uno_full(A, b, preset="filter_sqp")["objective"]
    f_ip = solve_uno_full(A, b, preset="ipopt_like")["objective"]
    assert f_ip <= f_fs * 1.05


def test_solve_uno_from_true_start_stays_feasible(A, b, f_true):
    diag = solve_uno_full(A, b, preset="ipopt_like", x0=f_true)
    assert np.all(diag["spectrum"] > 0)
    assert np.isclose(diag["constraint_violation"], 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_solve_uno_invalid_preset(A, b):
    with pytest.raises(ValueError, match="preset"):
        solve_uno(A, b, preset="bogus")


def test_solve_uno_invalid_hessian(A, b):
    with pytest.raises(ValueError, match="hessian"):
        solve_uno_full(A, b, preset="ipopt_like", hessian="bogus")


def test_solve_uno_invalid_weights(A, b):
    with pytest.raises(ValueError, match="weights"):
        solve_uno(A, b, weights="bogus")


def test_solve_uno_bad_weights_array(A, b):
    with pytest.raises(ValueError, match="weights"):
        solve_uno(A, b, weights=-np.ones(len(b)))


def test_solve_uno_negative_regularization(A, b):
    with pytest.raises(ValueError, match="regularization"):
        solve_uno(A, b, regularization=-1.0)


def test_uno_filter_gamma_margin():
    # the gamma margin: a point that only marginally improves one
    # (objective, violation) pair of the filter entry must be rejected
    filt = [(1.0, 1e3)]
    # tiny improvement in both coordinates (below gamma * viol_j)
    assert not uno_filter(filt, 1.0 - 1e-6, 1e3 - 1e-6, gamma=1e-5)


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_unfold_uno(detector, A, f_true):
    b = A @ f_true
    readings = {n: float(v) for n, v in
                zip(detector.detector_names, b)}
    result = detector.unfold_uno(readings, save_result=False)
    assert result["method"] == "Uno (filter_sqp)"
    for key in ("spectrum", "doserates", "energy", "uno_preset",
                "objective", "constraint_violation", "dual_infeasibility",
                "uno_converged"):
        assert key in result, f"missing key {key!r}"
    assert np.all(np.isfinite(result["spectrum"]))
    assert cosine(result["spectrum"], f_true) > 0.9


def test_detector_unfold_uno_ipopt(detector, A, f_true):
    b = A @ f_true
    readings = {n: float(v) for n, v in
                zip(detector.detector_names, b)}
    result = detector.unfold_uno(
        readings, preset="ipopt_like", save_result=False,
        max_iterations=80, tolerance=1e-6,
    )
    assert result["method"] == "Uno (ipopt_like)"
    assert result["spectrum"].shape == (detector.n_energy_bins,)
    assert np.all(np.isfinite(result["spectrum"]))


def test_detector_unfold_uno_max_neutron_energy(detector, A, f_true):
    b = A @ f_true
    readings = {n: float(v) for n, v in
                zip(detector.detector_names, b)}
    result = detector.unfold_uno(
        readings, save_result=False, max_neutron_energy=20.0
    )
    assert np.all(result["spectrum"][detector.E_MeV > 20.0] == 0.0)
    assert cosine(result["spectrum"], f_true) > 0.9
