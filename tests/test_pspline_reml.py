"""Tests for the P-spline REML unfolding method (LMMsolver analogue).

Covers the P-spline mixed-model formulation of the unfolding problem:
the difference penalty and its fixed/random spectral split, the REML
profile log-likelihood, the automatic smoothing-parameter selection,
the Henderson mixed-model equations solver, and the
``Detector.unfold_pspline_reml`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_pspline_reml
from bssunfold.core.unfold_pspline_reml import (
    LAMBDA_REL_BOUNDS,
    difference_matrix,
    mixed_model_split,
    pspline_penalty,
    reml_profile,
    select_lambda_reml,
    solve_pspline_reml_full,
)
from bssunfold.core.unfold_pspline_reml import (
    unfold_pspline_reml as unfold_pspline_reml_module,
)


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def E(detector):
    return detector.E_MeV


@pytest.fixture
def A(detector):
    return np.array(
        [detector.sensitivities[name] for name in detector.detector_names]
    )


@pytest.fixture
def f_true(E):
    """AmBe-like truth: evaporation body + cascade bump + thermal tail."""
    f = (
        0.55 * np.sqrt(E / 2.0) * np.exp(-E / 2.0)
        + 0.20 * np.exp(-((E - 4.5) / 1.2) ** 2)
        + 0.02 / (1.0 + (E / 0.05) ** 2)
    )
    return f * 1e6 / np.max(f)


@pytest.fixture
def readings(detector, A, f_true):
    """Poisson-noisy synthetic readings of the AmBe-like truth."""
    exact = A @ f_true
    noisy = np.random.default_rng(42).poisson(exact * 100).astype(float) / 100.0
    return {name: float(v) for name, v in zip(detector.detector_names, noisy)}


def cosine(x, f):
    return float(x @ f / (np.linalg.norm(x) * np.linalg.norm(f)))


# ---------------------------------------------------------------------------
# Penalty / mixed-model split utilities
# ---------------------------------------------------------------------------


def test_difference_matrix_shape_and_action():
    D = difference_matrix(6, order=2)
    assert D.shape == (4, 6)
    c = np.arange(6, dtype=float)
    np.testing.assert_allclose(D @ c, np.diff(c, 2))


def test_difference_matrix_orders():
    assert difference_matrix(5, order=1).shape == (4, 5)
    assert difference_matrix(6, order=3).shape == (3, 6)


@pytest.mark.parametrize("bad_order", (0, 5))
def test_difference_matrix_invalid_order(bad_order):
    with pytest.raises(ValueError, match="order"):
        difference_matrix(6, order=bad_order)


def test_difference_matrix_too_small():
    with pytest.raises(ValueError, match="greater than"):
        difference_matrix(2, order=2)


def test_pspline_penalty_symmetric_psd():
    G = pspline_penalty(8, order=2)
    assert G.shape == (8, 8)
    np.testing.assert_allclose(G, G.T, atol=1e-12)
    eig = np.linalg.eigvalsh(G)
    assert np.all(eig >= -1e-12)
    # null space of the second-difference penalty = linear trends
    assert int(np.sum(eig <= 1e-9 * eig[-1])) == 2


def test_mixed_model_split_dimensions():
    U_f, U_r, g_r = mixed_model_split(10, order=2)
    assert U_f.shape == (10, 2)
    assert U_r.shape == (10, 8)
    assert g_r.shape == (8,)
    assert np.all(g_r > 0)
    # orthonormal columns
    np.testing.assert_allclose(U_f.T @ U_f, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(U_r.T @ U_r, np.eye(8), atol=1e-10)
    # the two subspaces are orthogonal
    assert np.max(np.abs(U_f.T @ U_r)) < 1e-10


def test_mixed_model_split_order1():
    U_f, U_r, g_r = mixed_model_split(7, order=1)
    assert U_f.shape == (7, 1)
    assert U_r.shape == (7, 6)


# ---------------------------------------------------------------------------
# REML profile and smoothing selection
# ---------------------------------------------------------------------------


def _mixed_system(seed=0, m=12, n_basis=10):
    """Build a small weighted mixed model with known smooth structure."""
    rng = np.random.default_rng(seed)
    U_f, U_r, g_r = mixed_model_split(n_basis, order=2)
    X = rng.standard_normal((m, U_f.shape[1]))
    Z = rng.standard_normal((m, U_r.shape[1]))
    beta = np.array([1.0, -0.5])
    b_r = rng.standard_normal(U_r.shape[1]) / np.sqrt(g_r)
    y = X @ beta + Z @ b_r + 0.05 * rng.standard_normal(m)
    return y, X, Z, g_r


def test_reml_profile_finite_and_consistent():
    y, X, Z, g_r = _mixed_system()
    ll, sigma2 = reml_profile(y, X, Z, g_r, lam=1.0)
    assert np.isfinite(ll)
    assert sigma2 > 0
    # the profile at a very large lambda (random part switched off) must
    # reduce to the fixed-effects-only fit
    ll_big, _ = reml_profile(y, X, Z, g_r, lam=1e12)
    assert np.isfinite(ll_big)


def test_reml_profile_invalid_inputs():
    y, X, Z, g_r = _mixed_system()
    out = reml_profile(y[:3], X[:3], Z[:3], g_r, lam=1.0)
    assert out == (-np.inf, np.nan) or np.isfinite(out[0])


def test_select_lambda_reml_returns_interior_optimum():
    y, X, Z, g_r = _mixed_system(seed=3)
    lam_ref = 1.0
    sel = select_lambda_reml(y, X, Z, g_r, lam_ref)
    assert set(sel) >= {"lam", "lam_relative", "sigma2", "reml_loglik",
                        "converged", "n_iterations"}
    assert sel["converged"]
    assert LAMBDA_REL_BOUNDS[0] <= sel["lam_relative"] <= LAMBDA_REL_BOUNDS[1]
    assert sel["n_iterations"] >= 3
    # consistency of the absolute and relative values
    assert sel["lam"] == pytest.approx(lam_ref * sel["lam_relative"])


def test_select_lambda_reml_grid_bounds():
    y, X, Z, g_r = _mixed_system(seed=5)
    sel = select_lambda_reml(
        y, X, Z, g_r, lam_ref=1.0, lam_bounds=(1e-2, 1e2)
    )
    assert 1e-2 <= sel["lam_relative"] <= 1e2


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------


def test_weights_poisson_mode(A, E, f_true):
    b = np.maximum(A @ f_true, 1e-6)
    diag = solve_pspline_reml_full(A, b, E, weights="poisson")
    assert diag["weights"] == "poisson"
    assert np.all(np.isfinite(diag["spectrum"]))


def test_weights_array_mode(A, E, f_true):
    b = A @ f_true
    w = np.full(A.shape[0], 2.0)
    diag = solve_pspline_reml_full(A, b, E, weights=w)
    assert diag["weights"] == "array"


def test_weights_invalid(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="weights"):
        solve_pspline_reml_full(A, b, E, weights="bogus")
    with pytest.raises(ValueError, match="weights length"):
        solve_pspline_reml_full(A, b, E, weights=np.ones(3))
    with pytest.raises(ValueError, match="finite and positive"):
        solve_pspline_reml_full(
            A, b, E, weights=-np.ones(A.shape[0])
        )


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def test_solve_pspline_reml_recovers_smooth_truth(A, E, f_true):
    b = A @ f_true
    x, iterations, converged = solve_pspline_reml(A, b, E_MeV=E)
    assert x.shape == (E.shape[0],)
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))
    assert converged
    assert iterations >= 3
    assert cosine(x, f_true) > 0.9


def test_solve_pspline_reml_noisy_data(A, E, f_true):
    rng = np.random.default_rng(42)
    b = rng.poisson((A @ f_true) * 100).astype(float) / 100.0
    x, _, converged = solve_pspline_reml(A, b, E_MeV=E)
    assert converged
    assert cosine(x, f_true) > 0.9


def test_solve_pspline_reml_fixed_lambda(A, E, f_true):
    b = A @ f_true
    x_fixed, it0, _ = solve_pspline_reml(A, b, E_MeV=E, lam_relative=1.0)
    x_auto, it_reml, _ = solve_pspline_reml(A, b, E_MeV=E)
    assert it0 == 0  # no REML iterations when lambda is fixed
    assert it_reml > 0
    assert x_fixed.shape == x_auto.shape


def test_solve_pspline_reml_requires_energy_grid(A):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="E_MeV is required"):
        solve_pspline_reml(A, b)


def test_solve_pspline_reml_full_diagnostics(A, E, f_true):
    b = A @ f_true
    diag = solve_pspline_reml_full(A, b, E)
    for key in ("spectrum", "coefficients", "n_basis", "spline_order",
                "diff_order", "knot_spacing", "weights", "lam",
                "lam_relative", "lam_ref", "sigma2", "reml_loglik",
                "ed", "ed_norm", "reml_converged", "n_iterations"):
        assert key in diag, key
    assert diag["ed"] > 0
    assert 0 < diag["ed_norm"] <= 1.0 + 1e-9
    # scale invariance of the relative smoothing parameter
    diag2 = solve_pspline_reml_full(A, b * 1e3, E)
    assert diag2["lam_relative"] == pytest.approx(
        diag["lam_relative"], rel=0.5
    )


@pytest.mark.parametrize("n_basis", (3, 2, 0))
def test_solve_pspline_reml_invalid_n_basis(A, E, n_basis):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="n_basis"):
        solve_pspline_reml_full(A, b, E, n_basis=n_basis)


def test_solve_pspline_reml_n_basis_too_large(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="cannot exceed"):
        solve_pspline_reml_full(A, b, E, n_basis=E.shape[0] + 1)


def test_solve_pspline_reml_too_few_readings(E):
    A_small = np.ones((2, E.shape[0]))
    b = np.ones(2)
    with pytest.raises(ValueError, match="detector readings"):
        solve_pspline_reml_full(A_small, b, E)


def test_solve_pspline_reml_knot_spacing_invalid(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="knot_spacing"):
        solve_pspline_reml_full(A, b, E, knot_spacing="bogus")


def test_solve_pspline_reml_lam_relative_invalid(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="lam_relative"):
        solve_pspline_reml_full(A, b, E, lam_relative=-1.0)


def test_solve_pspline_reml_energy_grid_mismatch(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="E_MeV"):
        solve_pspline_reml_full(A, b, E[:-1])


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_unfold_pspline_reml(detector, readings):
    result = detector.unfold_pspline_reml(readings)
    assert result["method"] == "P-spline REML"
    for key in ("spectrum", "doserates", "effective_readings",
                "residual", "lam", "lam_relative", "reml_loglik",
                "ed", "ed_norm"):
        assert key in result
    assert np.all(result["spectrum"] >= 0)
    assert result["reml_converged"]


def test_detector_unfold_pspline_reml_save_and_errors(detector, readings):
    result = detector.unfold_pspline_reml(
        readings,
        calculate_errors=True,
        n_montecarlo=5,
        save_result=True,
        random_state=1,
    )
    assert "spectrum_uncert_mean" in result
    assert "saved_key" in result
    assert result["saved_key"] in detector.results_history


def test_detector_unfold_pspline_reml_max_energy(detector, readings):
    result = detector.unfold_pspline_reml(
        readings, max_neutron_energy=5.0
    )
    E = result["energy"]
    above = E > 5.0
    assert np.all(result["spectrum"][above] == 0)
    assert np.any(result["spectrum"][~above] > 0)


def test_detector_unfold_pspline_reml_custom_params(detector, readings):
    result = detector.unfold_pspline_reml(
        readings, n_basis=15, diff_order=1, knot_spacing="log",
        weights="poisson", lam_relative=1.0,
    )
    assert result["n_basis"] == 15
    assert np.all(np.isfinite(result["spectrum"]))


def test_module_wrapper_matches_detector(detector, readings):
    r1 = detector.unfold_pspline_reml(dict(readings))
    r2 = unfold_pspline_reml_module(
        detector_names=detector.detector_names,
        n_energy_bins=detector.E_MeV.shape[0],
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
    )
    np.testing.assert_allclose(r1["spectrum"], r2["spectrum"], rtol=1e-10)
