"""Tests for the NNQP-based and qpmad-based unfolding methods.

Covers:

* the coordinate-descent NNQP solver on synthetic QPs;
* the active-set QP solver on synthetic problems with box bounds;
* ``solve_nnqp`` and ``solve_qpmad`` (the core solver interfaces);
* ``unfold_nnqp`` and ``unfold_qpmad`` exposed both as module-level
  functions and as ``Detector.unfold_nnqp`` / ``Detector.unfold_qpmad``;
* hyper-parameter validation, reproducibility, ``max_neutron_energy``
  truncation, and Monte-Carlo uncertainty estimation.
"""

from __future__ import annotations

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_nnqp, solve_qpmad, unfold_nnqp, unfold_qpmad
from bssunfold.core.unfold_nnqp import _nnqp as nnqp_solver
from bssunfold.core.unfold_qpmad import _solve_qp_goldfarb_idnani as qp_solver


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def detector() -> Detector:
    return Detector()


@pytest.fixture
def readings(detector: Detector) -> dict[str, float]:
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic under-determined BSS problem (5 detectors, 12 energy bins)."""
    rng = np.random.default_rng(7)
    n, m = 12, 5
    A = rng.uniform(0.1, 1.0, size=(m, n))
    x_true = np.zeros(n)
    x_true[3] = 1.0
    x_true[4] = 0.5
    x_true[5] = 0.3
    b = A @ x_true
    return A, b, x_true


@pytest.fixture
def positive_definite_qp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small positive-definite QP ``0.5 x^T Q x + f^T x`` with optimum x*=0.

    Used to verify the inner NNQP / QP solvers on a problem with a known
    optimum.
    """
    rng = np.random.default_rng(42)
    p = 6
    M = rng.standard_normal((p, p))
    Q = M @ M.T + np.eye(p)  # SPD
    f = rng.standard_normal(p)
    return Q, f, np.zeros(p)


# --------------------------------------------------------------------------- #
# Inner NNQP solver                                                            #
# --------------------------------------------------------------------------- #
def test_nnqp_solver_finds_origin(positive_definite_qp):
    """NNQP solves min 0.5 x^T Q x + f^T x s.t. x >= 0; if f >= 0, x* = 0."""
    Q, f, x_true = positive_definite_qp
    # Make the gradient non-negative so the optimum is at the origin.
    f_pos = np.abs(f)
    x, iters, conv = nnqp_solver(Q, f_pos, tol=1e-8, max_iterations=5000,
                                  random_state=0)
    assert np.all(x >= -1e-9)
    assert np.linalg.norm(x) < 1e-3
    assert isinstance(iters, int) and iters > 0


def test_nnqp_solver_finds_interior_optimum(positive_definite_qp):
    """When f has mixed signs, NNQP should approach the cvxpy-style solution."""
    Q, f, x_true = positive_definite_qp
    x, _, _ = nnqp_solver(Q, f, tol=1e-9, max_iterations=5000,
                          random_state=0)
    assert np.all(x >= -1e-9)
    # KKT condition: x_i > 0 => (Q x + f)_i = 0
    grad = Q @ x + f
    active = x > 1e-6
    if np.any(active):
        assert np.allclose(grad[active], 0.0, atol=1e-3)


def test_nnqp_solver_reproducible(positive_definite_qp):
    Q, f, _ = positive_definite_qp
    x1, _, _ = nnqp_solver(Q, f, tol=1e-9, max_iterations=2000, random_state=42)
    x2, _, _ = nnqp_solver(Q, f, tol=1e-9, max_iterations=2000, random_state=42)
    assert np.allclose(x1, x2)


def test_nnqp_solver_validates_input():
    with pytest.raises(ValueError, match="square"):
        nnqp_solver(np.zeros((3, 4)), np.zeros(3))
    with pytest.raises(ValueError, match="length"):
        nnqp_solver(np.eye(3), np.zeros(4))
    with pytest.raises(ValueError, match="x0"):
        nnqp_solver(np.eye(3), np.zeros(3), x0=np.zeros(4))


# --------------------------------------------------------------------------- #
# Inner qpmad-style active-set QP solver                                       #
# --------------------------------------------------------------------------- #
def test_qp_solver_unconstrained(positive_definite_qp):
    """With no bounds, the QP solver returns the unconstrained minimum."""
    Q, f, _ = positive_definite_qp
    x, status = qp_solver(Q, f, tol=1e-9)
    assert status == "OK"
    x_exact = np.linalg.solve(Q, -f)
    assert np.allclose(x, x_exact, atol=1e-6)


def test_qp_solver_nonnegative(positive_definite_qp):
    """With lb=0, the solution stays non-negative."""
    Q, f, _ = positive_definite_qp
    n = len(f)
    x, status = qp_solver(Q, f, lb=np.zeros(n), ub=np.full(n, np.inf),
                          tol=1e-9, max_iterations=2000)
    assert status == "OK"
    assert np.all(x >= -1e-6)


def test_qp_solver_box_bounds(positive_definite_qp):
    """User-provided box bounds are respected."""
    Q, f, _ = positive_definite_qp
    n = len(f)
    lb = np.full(n, -0.5)
    ub = np.full(n, 0.5)
    x, status = qp_solver(Q, f, lb=lb, ub=ub, tol=1e-9, max_iterations=5000)
    assert status == "OK"
    assert np.all(x >= -0.5 - 1e-6)
    assert np.all(x <= 0.5 + 1e-6)


def test_qp_solver_reproducible(positive_definite_qp):
    Q, f, _ = positive_definite_qp
    n = len(f)
    x1, s1 = qp_solver(Q, f, lb=np.zeros(n), ub=np.full(n, np.inf))
    x2, s2 = qp_solver(Q, f, lb=np.zeros(n), ub=np.full(n, np.inf))
    assert s1 == s2
    assert np.allclose(x1, x2)


def test_qp_solver_validates_input():
    with pytest.raises(ValueError, match="square"):
        qp_solver(np.zeros((3, 4)), np.zeros(3))


# --------------------------------------------------------------------------- #
# solve_nnqp                                                                   #
# --------------------------------------------------------------------------- #
def test_solve_nnqp_returns_tuple(synthetic_problem):
    A, b, _ = synthetic_problem
    spec, iters, conv = solve_nnqp(A, b, regularization=1e-4, random_state=0)
    assert isinstance(spec, np.ndarray)
    assert spec.shape == (A.shape[1],)
    assert np.all(spec >= 0)
    assert isinstance(iters, int)
    assert isinstance(conv, bool)


def test_solve_nnqp_drives_residual_down(synthetic_problem):
    A, b, _ = synthetic_problem
    spec, _, _ = solve_nnqp(A, b, regularization=1e-4, smoothness_order=0,
                             tol=1e-7, max_iterations=10_000, random_state=0)
    rel_err = np.linalg.norm(A @ spec - b) / np.linalg.norm(b)
    # Under-determined problem; NNQP should drive the residual close to zero.
    assert rel_err < 0.01, f"rel_err={rel_err}"


def test_solve_nnqp_invalid_smoothness_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="smoothness"):
        solve_nnqp(A, b, smoothness_order=5)


def test_solve_nnqp_smoothness_reduces_oscillation(synthetic_problem):
    """smoothness_order=2 yields a smoother spectrum than smoothness_order=0."""
    A, b, _ = synthetic_problem
    s0, _, _ = solve_nnqp(A, b, smoothness_order=0, regularization=1e-4,
                           random_state=0, max_iterations=2000)
    s2, _, _ = solve_nnqp(A, b, smoothness_order=2, regularization=1e-2,
                           random_state=0, max_iterations=2000)
    def d2(v):
        return np.sum(np.diff(v, 2) ** 2)
    assert d2(s2) <= d2(s0) + 1e-9


# --------------------------------------------------------------------------- #
# solve_qpmad                                                                  #
# --------------------------------------------------------------------------- #
def test_solve_qpmad_returns_tuple(synthetic_problem):
    A, b, _ = synthetic_problem
    spec, code, conv = solve_qpmad(A, b, regularization=1e-4)
    assert isinstance(spec, np.ndarray)
    assert spec.shape == (A.shape[1],)
    assert np.all(spec >= 0)
    assert isinstance(code, int)
    assert isinstance(conv, bool)


def test_solve_qpmad_drives_residual_down(synthetic_problem):
    A, b, _ = synthetic_problem
    spec, _, _ = solve_qpmad(A, b, regularization=1e-4, smoothness_order=0,
                              tol=1e-9, max_iterations=10_000)
    rel_err = np.linalg.norm(A @ spec - b) / np.linalg.norm(b)
    assert rel_err < 0.01, f"rel_err={rel_err}"


def test_solve_qpmad_invalid_smoothness_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="smoothness"):
        solve_qpmad(A, b, smoothness_order=5)


def test_solve_qpmad_invalid_backend_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="backend"):
        solve_qpmad(A, b, backend="invalid")


def test_solve_qpmad_box_bounds(synthetic_problem):
    A, b, _ = synthetic_problem
    n = A.shape[1]
    lb = np.full(n, 0.0)
    ub = np.full(n, 0.3)
    spec, _, _ = solve_qpmad(A, b, lb=lb, ub=ub, regularization=1e-3,
                              max_iterations=5000)
    assert np.all(spec >= lb - 1e-6)
    assert np.all(spec <= ub + 1e-6)


def test_solve_qpmad_qpmad_backend_falls_back(synthetic_problem):
    """If qpmad C++ bindings are missing, the method falls back gracefully."""
    A, b, _ = synthetic_problem
    spec, code, conv = solve_qpmad(A, b, backend="qpmad")
    assert np.all(spec >= 0)
    assert np.all(np.isfinite(spec))


# --------------------------------------------------------------------------- #
# Detector.unfold_nnqp                                                         #
# --------------------------------------------------------------------------- #
def test_unfold_nnqp_basic(detector, readings):
    result = detector.unfold_nnqp(readings, regularization=1e-3,
                                   max_iterations=2000, random_state=0)
    assert isinstance(result, dict)
    assert result["method"] == "NNQP"
    assert "spectrum" in result
    assert "energy" in result
    assert "residual_norm" in result
    assert isinstance(result["spectrum"], np.ndarray)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert isinstance(result["residual_norm"], float)
    assert result["regularization"] == pytest.approx(1e-3)
    assert result["smoothness_order"] == 0
    assert result["floor"] == pytest.approx(1e-6)


def test_unfold_nnqp_invalid_smoothness_raises(detector, readings):
    with pytest.raises(ValueError, match="smoothness"):
        detector.unfold_nnqp(readings, smoothness_order=5)


def test_unfold_nnqp_with_initial_spectrum(detector, readings):
    init = np.ones(detector.n_energy_bins) * 0.5
    result = detector.unfold_nnqp(readings, initial_spectrum=init,
                                   max_iterations=500, random_state=0)
    assert result["method"] == "NNQP"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_nnqp_max_neutron_energy_truncates(detector, readings):
    """max_neutron_energy zeroes the spectrum above the cutoff after expansion."""
    n = detector.n_energy_bins
    e_max = float(detector.E_MeV[n // 2])
    result = detector.unfold_nnqp(readings, max_neutron_energy=e_max,
                                   max_iterations=500, random_state=0)
    energy = result["energy"]
    spectrum = result["spectrum"]
    assert len(energy) == n
    above = energy > e_max + 1e-9
    assert np.all(spectrum[above] == 0.0)


def test_unfold_nnqp_empty_readings_raises(detector):
    with pytest.raises(ValueError, match="readings"):
        detector.unfold_nnqp({})


# --------------------------------------------------------------------------- #
# Detector.unfold_qpmad                                                        #
# --------------------------------------------------------------------------- #
def test_unfold_qpmad_basic(detector, readings):
    result = detector.unfold_qpmad(readings, regularization=1e-3,
                                    max_iterations=2000)
    assert isinstance(result, dict)
    assert result["method"] == "qpmad"
    assert "spectrum" in result
    assert "energy" in result
    assert "residual_norm" in result
    assert isinstance(result["spectrum"], np.ndarray)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert isinstance(result["residual_norm"], float)
    assert result["regularization"] == pytest.approx(1e-3)
    assert result["backend"] == "python"


def test_unfold_qpmad_invalid_smoothness_raises(detector, readings):
    with pytest.raises(ValueError, match="smoothness"):
        detector.unfold_qpmad(readings, smoothness_order=5)


def test_unfold_qpmad_invalid_backend_raises(detector, readings):
    with pytest.raises(ValueError, match="backend"):
        detector.unfold_qpmad(readings, backend="invalid")


def test_unfold_qpmad_max_neutron_energy_truncates(detector, readings):
    n = detector.n_energy_bins
    e_max = float(detector.E_MeV[n // 2])
    result = detector.unfold_qpmad(readings, max_neutron_energy=e_max,
                                    max_iterations=1000)
    energy = result["energy"]
    spectrum = result["spectrum"]
    assert len(energy) == n
    above = energy > e_max + 1e-9
    assert np.all(spectrum[above] == 0.0)


def test_unfold_qpmad_empty_readings_raises(detector):
    with pytest.raises(ValueError, match="readings"):
        detector.unfold_qpmad({})


def test_unfold_qpmad_with_box_bounds(detector, readings):
    n = detector.n_energy_bins
    lb = np.zeros(n)
    ub = np.full(n, 0.5)
    result = detector.unfold_qpmad(readings, lb=lb, ub=ub,
                                    regularization=1e-3,
                                    max_iterations=5000)
    spec = result["spectrum"]
    assert np.all(spec >= lb - 1e-6)
    assert np.all(spec <= ub + 1e-6)


# --------------------------------------------------------------------------- #
# Module-level wrappers                                                        #
# --------------------------------------------------------------------------- #
def test_module_level_unfold_nnqp(detector, readings):
    res = unfold_nnqp(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        regularization=1e-3, max_iterations=500, random_state=0,
    )
    assert res["method"] == "NNQP"
    assert len(res["spectrum"]) == detector.n_energy_bins


def test_module_level_unfold_qpmad(detector, readings):
    res = unfold_qpmad(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        regularization=1e-3, max_iterations=500,
    )
    assert res["method"] == "qpmad"
    assert len(res["spectrum"]) == detector.n_energy_bins
