"""Tests for the AMG/stationary-preconditioned Krylov unfolding method.

Covers the classical stationary-iteration preconditioners (Jacobi,
Gauss-Seidel, SOR, SSOR -- the ``Rlinsolve`` family), the algebraic
multigrid preconditioner (optional ``pyamg``), the projected-restart
non-negativity loop, validation of the solver parameters and the
``Detector.unfold_amg`` integration.
"""

import warnings

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_amg
from bssunfold.core.unfold_amg import (
    AMG_AVAILABLE,
    build_preconditioner,
)
from bssunfold.core.unfold_amg import (
    unfold_amg as unfold_amg_module,
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
def readings(detector, A, f_true):
    exact = A @ f_true
    noisy = np.random.default_rng(42).poisson(exact * 100).astype(float) / 100.0
    return {name: float(v) for name, v in zip(detector.detector_names, noisy)}


def cosine(x, f):
    return float(x @ f / (np.linalg.norm(x) * np.linalg.norm(f)))


# ---------------------------------------------------------------------------
# Preconditioner construction
# ---------------------------------------------------------------------------


def test_build_preconditioner_jacobi_matches_diagonal():
    rng = np.random.default_rng(0)
    A = rng.random((6, 10))
    M = build_preconditioner(A, kind="jacobi")
    r = rng.random(10)
    N = A.T @ A
    np.testing.assert_allclose(M @ r, r / np.diag(N), rtol=1e-12)


def test_build_preconditioner_stationary_matches_dense_solve():
    rng = np.random.default_rng(1)
    A = rng.random((8, 12))
    N = A.T @ A + 1e-6 * np.eye(12)
    D = np.diag(np.diag(N))  # diagonal *matrix* (broadcast-safe)
    L = np.tril(N, -1)
    U = np.triu(N, 1)
    r = rng.random(12)

    M_gs = build_preconditioner(A, kind="gs", damping=1e-6)
    np.testing.assert_allclose(
        M_gs @ r, np.linalg.solve(D + L, r), rtol=1e-6
    )

    omega = 1.4
    M_sor = build_preconditioner(A, kind="sor", omega=omega, damping=1e-6)
    np.testing.assert_allclose(
        M_sor @ r,
        omega * np.linalg.solve(D + omega * L, r),
        rtol=1e-6,
    )

    M_ssor = build_preconditioner(A, kind="ssor", omega=omega, damping=1e-6)
    expected = (
        omega
        * (2.0 - omega)
        * np.linalg.solve(
            D + omega * U, D @ np.linalg.solve(D + omega * L, r)
        )
    )
    np.testing.assert_allclose(M_ssor @ r, expected, rtol=1e-10)


def test_build_preconditioner_none_is_identity():
    A = np.ones((3, 5))
    M = build_preconditioner(A, kind="none")
    r = np.arange(5, dtype=float)
    np.testing.assert_allclose(M @ r, r)


def test_build_preconditioner_amg_available_or_fallback():
    rng = np.random.default_rng(2)
    A = rng.random((6, 10))
    r = rng.random(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = build_preconditioner(A, kind="amg", damping=1e-6)
    y = M @ r
    assert np.all(np.isfinite(y))
    if not AMG_AVAILABLE:
        # fallback must be the Jacobi preconditioner
        np.testing.assert_allclose(
            y, r / np.diag(A.T @ A + 1e-6 * np.eye(10)), rtol=1e-12
        )


def test_build_preconditioner_fallback_without_pyamg():
    """With the AMG flag off, 'amg' degrades to Jacobi with a warning."""
    import sys

    rng = np.random.default_rng(3)
    A = rng.random((6, 10))
    r = rng.random(10)
    # NOTE: ``bssunfold.core.unfold_amg`` is shadowed by the unfold_amg
    # *function* re-exported from the package ``__init__``; use
    # sys.modules to reach the module object (same trick as
    # bssunfold.core.unfold_interpret, see AGENTS.md).
    mod = sys.modules["bssunfold.core.unfold_amg"]

    with monkeypatch_module_flag(mod):
        with pytest.warns(RuntimeWarning, match="pyamg is not installed"):
            M = mod.build_preconditioner(A, kind="amg", damping=1e-6)
        np.testing.assert_allclose(
            M @ r, r / np.diag(A.T @ A + 1e-6 * np.eye(10)), rtol=1e-12
        )


def monkeypatch_module_flag(mod):
    """Context manager forcing ``mod.AMG_AVAILABLE = False`` temporarily."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        original = mod.AMG_AVAILABLE
        mod.AMG_AVAILABLE = False
        try:
            yield
        finally:
            mod.AMG_AVAILABLE = original

    return ctx()


def test_build_preconditioner_invalid_kind():
    with pytest.raises(ValueError, match="preconditioner"):
        build_preconditioner(np.ones((3, 4)), kind="bogus")


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pre,method",
    [
        ("amg", "cg"),
        ("jacobi", "cg"),
        ("ssor", "cg"),
        ("gs", "gmres"),
        ("sor", "bicgstab"),
        ("none", "cg"),
        ("none", "gmres"),
    ],
)
def test_solve_amg_all_combinations(A, f_true, pre, method):
    b = A @ f_true
    x, iterations, converged = solve_amg(
        A, b, preconditioner=pre, method=method
    )
    assert x.shape == (A.shape[1],)
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))
    assert iterations >= 0
    assert isinstance(converged, bool)
    assert cosine(x, f_true) > 0.85


def test_solve_amg_undamped_pure_normal_equations(A, f_true):
    """regularization=0.0 reproduces the pure least-squares solution."""
    b = A @ f_true
    x, _, _ = solve_amg(
        A, b, preconditioner="none", regularization=0.0,
        nonnegativity=False, outer_iterations=1,
    )
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert cosine(x, f_true) > 0.9
    assert np.linalg.norm(x - x_ref) < 1e-4 * max(
        1.0, np.linalg.norm(x_ref)
    )


def test_solve_amg_projected_restarts_reduce_residual(A, f_true):
    b = A @ f_true + 0.05
    x1, _, _ = solve_amg(A, b, outer_iterations=1, preconditioner="jacobi")
    x5, _, _ = solve_amg(A, b, outer_iterations=8, preconditioner="jacobi")
    res1 = np.linalg.norm(A.T @ (A @ x1) - A.T @ b)
    res5 = np.linalg.norm(A.T @ (A @ x5) - A.T @ b)
    assert res5 <= res1 + 1e-12


def test_solve_amg_no_nonnegativity(A, f_true):
    b = A @ f_true
    x, _, _ = solve_amg(A, b, nonnegativity=False, preconditioner="none")
    # without clamping the solution may contain negative bins
    assert x.shape == (A.shape[1],)


def test_solve_amg_invalid_method(A):
    with pytest.raises(ValueError, match="method"):
        solve_amg(A, np.ones(A.shape[0]), method="bogus")


def test_solve_amg_invalid_preconditioner(A):
    with pytest.raises(ValueError, match="preconditioner"):
        solve_amg(A, np.ones(A.shape[0]), preconditioner="bogus")


def test_solve_amg_invalid_regularization(A):
    with pytest.raises(ValueError, match="regularization"):
        solve_amg(A, np.ones(A.shape[0]), regularization=-1.0)


def test_solve_amg_invalid_outer_iterations(A):
    with pytest.raises(ValueError, match="outer_iterations"):
        solve_amg(A, np.ones(A.shape[0]), outer_iterations=0)


def test_solve_amg_cg_with_nonsymmetric_preconditioner_warns(A):
    with pytest.warns(RuntimeWarning, match="incompatible with method='cg'"):
        solve_amg(A, np.ones(A.shape[0]), method="cg", preconditioner="gs")


def test_solve_amg_omega_out_of_range_warns(A):
    with pytest.warns(RuntimeWarning, match="omega"):
        solve_amg(A, np.ones(A.shape[0]), omega=2.5, method="gmres",
                  preconditioner="sor")


def test_solve_amg_with_initial_guess(A, f_true):
    b = A @ f_true
    x0 = np.full(A.shape[1], 1e-3)
    x, _, _ = solve_amg(A, b, x0=x0, preconditioner="jacobi")
    assert np.all(x >= 0)


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_unfold_amg(detector, readings):
    result = detector.unfold_amg(readings)
    assert result["method"] == "AMG-Krylov"
    for key in ("spectrum", "doserates", "effective_readings",
                "residual", "krylov_method", "preconditioner"):
        assert key in result
    assert result["krylov_method"] == "cg"
    assert result["preconditioner"] == "amg"
    assert np.all(result["spectrum"] >= 0)


def test_detector_unfold_amg_stationary(detector, readings):
    result = detector.unfold_amg(
        readings, preconditioner="ssor", omega=1.5, method="gmres"
    )
    assert result["preconditioner"] == "ssor"
    assert result["omega"] == 1.5
    assert np.all(np.isfinite(result["spectrum"]))


def test_detector_unfold_amg_save_result(detector, readings):
    result = detector.unfold_amg(readings, save_result=True)
    assert "saved_key" in result
    assert result["saved_key"] in detector.results_history


def test_detector_unfold_amg_max_energy(detector, readings):
    result = detector.unfold_amg(readings, max_neutron_energy=5.0)
    E = result["energy"]
    above = E > 5.0
    assert np.all(result["spectrum"][above] == 0)
    assert np.any(result["spectrum"][~above] > 0)


def test_detector_unfold_amg_errors(detector, readings):
    result = detector.unfold_amg(
        readings, calculate_errors=True, n_montecarlo=5, random_state=0
    )
    assert "spectrum_uncert_mean" in result


def test_module_wrapper_matches_detector(detector, readings):
    r1 = detector.unfold_amg(dict(readings))
    r2 = unfold_amg_module(
        detector_names=detector.detector_names,
        n_energy_bins=detector.E_MeV.shape[0],
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
    )
    np.testing.assert_allclose(r1["spectrum"], r2["spectrum"], rtol=1e-10)


def test_unfold_amg_exported_from_core():
    from bssunfold.core import solve_amg as s
    from bssunfold.core import unfold_amg as u

    assert callable(s) and callable(u)
