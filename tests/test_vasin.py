"""Tests for the Vasin-Serazhitnikova regularization (Vasin & Serazhitnikova, 2010).

Covers the Holder seminorm, the epigraph solver ``solve_vasin`` (validation,
optimality of the returned minimizer, single-bin and zero-penalty edge cases),
the paper's discontinuous Model-2 benchmark including the contrast against
plain Tikhonov regularization, and the ``Detector.unfold_vasin`` integration.
"""

import numpy as np
import pytest
from scipy.optimize import nnls

from bssunfold import Detector
from bssunfold.core import holder_seminorm, solve_vasin, unfold_vasin

# ---------------------------------------------------------------------------
# Reference implementation of the functional (independent of the solver)
# ---------------------------------------------------------------------------


def phi_functional(x, A, b, alpha, mu, dist, quad_weights, weights=None):
    """Evaluate ``||W (A x - b)||_inf + alpha * (1/2 x'Qx + |x|_inf + [x]_mu)``.

    Mirrors eq. (8) of the paper with the same distance normalization the
    solver uses, but written without cvxpy so that the tests check the
    solver against an independent expression.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        w = np.ones(len(b))
    else:
        w = np.asarray(weights, dtype=float)
    discrepancy = float(np.max(np.abs(w * (A @ x - b))))
    D = np.abs(np.asarray(dist, dtype=float))
    diameter = float(D.max())
    if diameter > 0:
        D = D / diameter
    off = D > 0
    inv = np.zeros_like(D)
    inv[off] = D[off] ** (-mu)
    seminorm = float(np.max(np.abs(x[:, None] - x[None, :]) * inv))
    quad = 0.5 * float(np.sum(np.asarray(quad_weights, dtype=float) * x**2))
    return discrepancy + alpha * (quad + float(np.max(np.abs(x))) + seminorm)


def model2_problem(n=129):
    """The paper's test equation (14) with H = 2 and the Model-2 box."""
    grid = np.linspace(0.0, 2.0, n)
    h = grid[1] - grid[0]
    kernel = 2.0 / (2.0**2 + (grid[:, None] - grid[None, :]) ** 2) ** 2
    A = kernel * h
    u = np.zeros(n)
    u[43:86] = 2.0
    return grid, h, A, u, A @ u


def rise_width(x):
    """Number of grid nodes between the 20 % and 80 % levels of the flank."""
    level = float(np.max(x))
    low = np.flatnonzero(x >= 0.2 * level)
    high = np.flatnonzero(x >= 0.8 * level)
    return int(high[0] - low[0])


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
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


@pytest.fixture(scope="module")
def model2_solution():
    """One solve of the paper's Model-2 benchmark, reused by several tests."""
    grid, h, A_mat, u_true, b_vec = model2_problem()
    dist = np.abs(grid[:, None] - grid[None, :])
    x, attempts, converged = solve_vasin(
        A_mat,
        b_vec,
        alpha=1e-4,
        mu=1e-3,
        data_norm="linf",
        weights=None,
        dist=dist,
        quad_weights=np.full(len(grid), h),
    )
    return {
        "grid": grid,
        "h": h,
        "A": A_mat,
        "dist": dist,
        "q": np.full(len(grid), h),
        "truth": u_true,
        "b": b_vec,
        "x": x,
        "attempts": attempts,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Holder seminorm
# ---------------------------------------------------------------------------


def test_seminorm_of_constant_function_is_zero():
    x = np.full(8, 3.5)
    grid = np.linspace(0.0, 1.0, 8)
    assert holder_seminorm(x, grid, mu=0.5) == pytest.approx(0.0)


def test_seminorm_uniform_grid_is_normalized_slope():
    """With unit-diameter distances the mu=1 seminorm is the max slope."""
    x = np.array([0.0, 1.0, 2.0])
    assert holder_seminorm(x, np.array([0.0, 1.0, 2.0]), mu=1.0) == pytest.approx(2.0)


def test_seminorm_small_mu_degenerates_to_range():
    """mu -> 0 makes every distance cost 1, so the seminorm is the range.

    This is the documented mechanism behind the jump preservation: a step
    and a ramp of the same height are then equally expensive.
    """
    step = np.array([0.0, 0.0, 1.0, 1.0])
    ramp = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    grid = np.linspace(0.0, 1.0, 4)
    assert holder_seminorm(step, grid, mu=1e-8) == pytest.approx(1.0, rel=1e-4)
    assert holder_seminorm(ramp, grid, mu=1e-8) == pytest.approx(1.0, rel=1e-4)


def test_seminorm_grows_with_mu_for_a_localized_jump():
    step = np.array([0.0, 0.0, 5.0, 5.0])
    grid = np.linspace(0.0, 1.0, 4)
    low = holder_seminorm(step, grid, mu=0.1)
    high = holder_seminorm(step, grid, mu=1.0)
    assert high > low


def test_seminorm_accepts_distance_matrix():
    """Distances may be handed over as a matrix; normalization still applies."""
    x = np.array([0.0, 1.0, 1.0])
    dist = np.abs(np.array([0.0, 1.0, 2.0])[:, None] - np.array([0.0, 1.0, 2.0]))
    # the diameter-normalized grid puts x_1 at half a cell from x_0
    assert holder_seminorm(x, dist, mu=1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# solve_vasin: validation and edge cases
# ---------------------------------------------------------------------------


def test_solve_bad_data_norm(A, b):
    with pytest.raises(ValueError, match="data_norm must be one of"):
        solve_vasin(A, b, data_norm="l1")


@pytest.mark.parametrize("mu", [0.0, -0.5, 1.5])
def test_solve_bad_mu(A, b, mu):
    with pytest.raises(ValueError, match="mu must be in"):
        solve_vasin(A, b, mu=mu)


def test_solve_bad_weights_length(A, b):
    with pytest.raises(ValueError, match="weights must have length"):
        solve_vasin(A, b, weights=np.ones(2))


def test_solve_bad_weights_name(A, b):
    with pytest.raises(ValueError, match="weights must be"):
        solve_vasin(A, b, weights="absolute")


def test_solve_bad_dist_shape(A, b):
    n = A.shape[1]
    with pytest.raises(ValueError, match="dist must have shape"):
        solve_vasin(A, b, dist=np.zeros((n - 1, n - 1)))


def test_solve_bad_quad_weights(A, b):
    with pytest.raises(ValueError, match="quad_weights must have length"):
        solve_vasin(A, b, quad_weights=np.ones(3))


def test_solve_returns_valid_spectrum(A, b):
    x, attempts, converged = solve_vasin(A, b)
    assert x.shape == (A.shape[1],)
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)
    assert attempts >= 1
    assert converged is True


def test_solve_is_deterministic(A, b):
    x1, _, _ = solve_vasin(A, b, mu=0.3)
    x2, _, _ = solve_vasin(A, b, mu=0.3)
    np.testing.assert_allclose(x1, x2)


def test_solve_single_bin_grid():
    """A one-bin grid leaves the pair constraints empty; ell >= 0 closes it."""
    x, attempts, converged = solve_vasin(
        np.ones((2, 1)), np.array([2.0, 4.0]), alpha=1e-2
    )
    assert converged is True
    assert x.shape == (1,)
    assert np.isfinite(x[0]) and x[0] > 0


def test_solve_zero_alpha_runs(A, b):
    """alpha = 0 removes the stabilizer; the discrepancy fit still works."""
    x, _, converged = solve_vasin(A, b, alpha=0.0)
    assert converged is True
    assert np.linalg.norm(A @ x - b) <= np.linalg.norm(b)


def test_solve_l2_norm_runs(A, b):
    x, _, converged = solve_vasin(A, b, data_norm="l2")
    assert converged is True
    assert np.all(x >= 0)


def test_solve_relative_vs_unit_weights_differ(A, b):
    x_rel, _, _ = solve_vasin(A, b, weights="relative")
    x_unit, _, _ = solve_vasin(A, b, weights=None)
    assert not np.allclose(x_rel, x_unit)


def test_solve_large_alpha_shrinks_solution(A, b):
    x_small, _, _ = solve_vasin(A, b, alpha=1e-4)
    x_large, _, _ = solve_vasin(A, b, alpha=1e4)
    assert np.linalg.norm(x_large) < np.linalg.norm(x_small)


def test_solve_unknown_solver_falls_back(A, b):
    x, attempts, converged = solve_vasin(A, b, solver="NOT_A_SOLVER")
    assert attempts > 1
    assert converged is True
    assert np.all(x >= 0)


# ---------------------------------------------------------------------------
# Optimality of the epigraph solve
# ---------------------------------------------------------------------------


def test_solve_minimizes_the_functional():
    """The returned spectrum beats the truth and random perturbations."""
    problem = model2_problem(n=40)
    _, h, A_mat, u_true, b_vec = problem
    grid = problem[0]
    dist = np.abs(grid[:, None] - grid[None, :])
    q = np.full(len(grid), h)
    alpha, mu = 1e-3, 1e-2
    x, _, converged = solve_vasin(
        A_mat, b_vec, alpha=alpha, mu=mu, weights=None, dist=dist, quad_weights=q
    )
    assert converged
    ref = phi_functional(x, A_mat, b_vec, alpha, mu, dist, q)
    assert ref <= phi_functional(u_true, A_mat, b_vec, alpha, mu, dist, q) + 1e-9
    rng = np.random.default_rng(7)
    for direction in rng.normal(size=(12, len(grid))):
        for step in (0.02, 0.2, 2.0):
            candidate = np.maximum(x + step * direction, 0.0)
            value = phi_functional(
                candidate, A_mat, b_vec, alpha, mu, dist, q
            )
            assert value >= ref - 1e-6


def test_solver_agrees_with_direct_epigraph_reference():
    """Small problem: compare against an independently built CVXPY program."""
    import cvxpy as cp

    rng = np.random.default_rng(3)
    m, n = 6, 10
    A_mat = np.abs(rng.normal(size=(m, n))) + 0.1
    truth = np.abs(rng.normal(size=n)) + 0.2
    b_vec = A_mat @ truth
    grid = np.linspace(0.0, 1.0, n)
    dist = np.abs(grid[:, None] - grid[None, :])
    alpha, mu = 1e-2, 0.2

    x_solver, _, converged = solve_vasin(
        A_mat, b_vec, alpha=alpha, mu=mu, weights=None, dist=dist
    )
    assert converged

    x = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    s = cp.Variable()
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    operator = np.zeros((len(pairs), n))
    for row, (i, j) in enumerate(pairs):
        operator[row, i] = 1.0
        operator[row, j] = -1.0
    scale = np.array([(dist[i, j] / dist.max()) ** (-mu) for i, j in pairs])
    ell = cp.Variable()
    residual = A_mat @ x - b_vec
    signed = cp.multiply(operator @ x, scale)
    constraints = [
        t >= residual,
        t >= -residual,
        s >= cp.max(cp.abs(x)),
        ell >= signed,
        ell >= -signed,
        ell >= 0,
    ]
    problem = cp.Problem(
        cp.Minimize(t + alpha * (0.5 * cp.sum(cp.square(x)) + s + ell)),
        constraints,
    )
    problem.solve(solver=cp.CLARABEL)
    assert problem.status in ("optimal", "optimal_inaccurate")
    ref_value = problem.value
    achieved = phi_functional(x_solver, A_mat, b_vec, alpha, mu, dist, np.ones(n))
    assert achieved <= ref_value + 1e-4 * max(1.0, abs(ref_value))
    assert np.linalg.norm(x_solver - x.value) / np.linalg.norm(x.value) < 0.05


# ---------------------------------------------------------------------------
# The paper's Model 2: a discontinuous solution
# ---------------------------------------------------------------------------


def test_model2_converges(model2_solution):
    assert model2_solution["converged"] is True
    assert model2_solution["attempts"] >= 1


def test_model2_recovers_the_box(model2_solution):
    x = model2_solution["x"]
    truth = model2_solution["truth"]
    error = float(np.linalg.norm(x - truth) / np.linalg.norm(truth))
    assert error < 0.2


def test_model2_keeps_a_flat_plateau_and_empty_wings(model2_solution):
    x = model2_solution["x"]
    plateau = x[46:83]
    assert float(np.std(plateau)) < 0.02 * float(np.max(x))
    assert float(np.mean(plateau)) == pytest.approx(2.0, rel=0.05)
    assert float(np.max(x[:40])) < 0.05
    assert float(np.max(x[90:])) < 0.05


def test_model2_sharper_edge_than_plain_tikhonov(model2_solution):
    """The point of the article: the Holder stabilizer does not smear jumps.

    Plain Tikhonov (quadratic penalty, same positivity constraint, tuned to
    its best smoothing level) needs many more grid nodes to traverse the
    same discontinuity.
    """
    A_mat = model2_solution["A"]
    b_vec = model2_solution["b"]
    u_true = model2_solution["truth"]
    n = len(model2_solution["grid"])
    x_tikh, _ = nnls(
        np.vstack([A_mat, np.sqrt(1e-8) * np.eye(n)]),
        np.concatenate([b_vec, np.zeros(n)]),
    )
    assert rise_width(model2_solution["x"]) <= 4
    assert rise_width(x_tikh) >= 8
    assert (
        np.linalg.norm(model2_solution["x"] - u_true)
        < np.linalg.norm(x_tikh - u_true)
    )


def test_model2_lipschitz_limit_blunts_the_jump(model2_solution):
    """mu = 1 prices a one-cell jump at 1/h, so the box is no longer sharp."""
    x, _, converged = solve_vasin(
        model2_solution["A"],
        model2_solution["b"],
        alpha=1e-4,
        mu=1.0,
        weights=None,
        dist=model2_solution["dist"],
        quad_weights=model2_solution["q"],
    )
    assert converged
    assert rise_width(x) > rise_width(model2_solution["x"])


def test_model2_functional_is_minimal(model2_solution):
    """No simple competitor beats the solver on the article's functional."""
    competitors = {
        "truth": model2_solution["truth"],
        "zero": np.zeros_like(model2_solution["truth"]),
        "flat": np.full(
            len(model2_solution["truth"]), float(np.mean(model2_solution["b"]))
        ),
    }
    achieved = phi_functional(
        model2_solution["x"],
        model2_solution["A"],
        model2_solution["b"],
        1e-4,
        1e-3,
        model2_solution["dist"],
        model2_solution["q"],
    )
    for name, candidate in competitors.items():
        value = phi_functional(
            candidate,
            model2_solution["A"],
            model2_solution["b"],
            1e-4,
            1e-3,
            model2_solution["dist"],
            model2_solution["q"],
        )
        assert achieved <= value, f"{name} has a lower Phi than the solver output"


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_unfold_vasin_basic(detector, readings):
    result = detector.unfold_vasin(readings, save_result=False)
    assert isinstance(result, dict)
    assert result["method"] == "Vasin"
    for key in ("energy", "spectrum", "residual_norm", "effective_readings"):
        assert key in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert np.all(np.isfinite(result["spectrum"]))
    assert result["alpha"] == pytest.approx(1e-2)
    assert result["mu"] == pytest.approx(0.1)
    assert result["data_norm"] == "linf"
    assert result["weights"] == "relative"
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)


def test_unfold_vasin_all_spheres(detector):
    result = detector.unfold_vasin(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_vasin_matches_core_wrapper(detector, readings):
    res_det = detector.unfold_vasin(readings, mu=0.2, save_result=False)
    res_fn = unfold_vasin(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        ln_steps=detector.ln_steps,
        mu=0.2,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "Vasin"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_unfold_vasin_ignores_initial_spectrum(detector, readings):
    plain = detector.unfold_vasin(readings, save_result=False)
    shifted = detector.unfold_vasin(
        readings,
        initial_spectrum=np.full(detector.n_energy_bins, 0.3),
        save_result=False,
    )
    assert np.allclose(plain["spectrum"], shifted["spectrum"])


def test_unfold_vasin_mu_changes_the_shape(detector, readings):
    smooth = detector.unfold_vasin(readings, mu=1.0, save_result=False)
    sharp = detector.unfold_vasin(readings, mu=0.02, save_result=False)
    assert not np.allclose(smooth["spectrum"], sharp["spectrum"])


def test_unfold_vasin_data_norm_l2(detector, readings):
    result = detector.unfold_vasin(readings, data_norm="l2", save_result=False)
    assert result["data_norm"] == "l2"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_vasin_explicit_weights(detector, readings):
    weights = np.ones(len(readings))
    result = detector.unfold_vasin(readings, weights=weights, save_result=False)
    assert result["weights"] == "custom"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_vasin_max_neutron_energy(detector, readings):
    result = detector.unfold_vasin(
        readings, save_result=False, max_neutron_energy=10.0
    )
    assert result["energy"].shape == (len(detector.E_MeV),)
    assert np.all(result["spectrum"][result["energy"] > 10.0] == 0.0)


def test_unfold_vasin_deterministic(detector, readings):
    r1 = detector.unfold_vasin(readings, save_result=False)
    r2 = detector.unfold_vasin(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_vasin_save_result(detector, readings):
    detector.unfold_vasin(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "Vasin"


def test_unfold_vasin_calculate_errors(detector, readings):
    result = detector.unfold_vasin(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_detector_exposes_method():
    assert hasattr(Detector, "unfold_vasin")


def test_core_exports():
    import bssunfold.core as core

    assert "solve_vasin" in core.__all__
    assert "unfold_vasin" in core.__all__
    assert "holder_seminorm" in core.__all__
    assert core.holder_seminorm is holder_seminorm
    assert core.solve_vasin is solve_vasin
