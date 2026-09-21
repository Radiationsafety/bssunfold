"""Tests for the LOUHI unfolding method (Routti & Sandberg 1980).

Covers the generalized smoothing operator, the Hildreth quadratic
programming core of ``solve_louhi``, the automatic smoothing-weight
regression, the statistical error propagation (``louhi_covariance``),
parameter validation and the ``Detector.unfold_louhi`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import (
    louhi_covariance,
    louhi_smoothing_matrix,
    solve_louhi,
    unfold_louhi,
)
from bssunfold.core.unfold_louhi import _hildreth_qp


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


@pytest.fixture
def f_true(detector):
    E = detector.E_MeV
    f = (
        0.55 * np.sqrt(E / 2.0) * np.exp(-E / 2.0)
        + 0.20 * np.exp(-((E - 4.5) / 1.2) ** 2)
        + 0.02 / (1.0 + (E / 0.05) ** 2)
    )
    return f * 1e6 / np.max(f)


# ---------------------------------------------------------------------------
# Smoothing operator
# ---------------------------------------------------------------------------


def test_smoothing_matrix_orders():
    for order, rows in ((0, 60), (1, 59), (2, 58)):
        L = louhi_smoothing_matrix(60, order)
        assert L.shape == (rows, 60)


def test_smoothing_matrix_identity_order():
    L = louhi_smoothing_matrix(8, 0)
    np.testing.assert_allclose(L, np.eye(8))


def test_smoothing_matrix_first_difference():
    L = louhi_smoothing_matrix(4, 1)
    expected = np.array(
        [[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, -1.0, 1.0]]
    )
    np.testing.assert_allclose(L, expected)


def test_smoothing_matrix_second_difference():
    L = louhi_smoothing_matrix(4, 2)
    expected = np.array([[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]])
    np.testing.assert_allclose(L, expected)


def test_smoothing_matrix_small_n():
    for n in (1, 2):
        for order in (0, 1, 2):
            L = louhi_smoothing_matrix(n, order)
            assert L.shape[1] == n
            assert L.shape[0] >= 1


def test_smoothing_matrix_bad_order():
    with pytest.raises(ValueError, match="smooth_order must be one of"):
        louhi_smoothing_matrix(10, 3)


def test_smoothing_matrix_bad_n():
    with pytest.raises(ValueError, match="n must be positive"):
        louhi_smoothing_matrix(0, 1)


# ---------------------------------------------------------------------------
# Hildreth quadratic programming core
# ---------------------------------------------------------------------------


def test_hildreth_matches_unconstrained_when_interior():
    """Without active constraints the QP equals the unconstrained solution."""
    rng = np.random.default_rng(42)
    n = 12
    M = rng.normal(size=(n, n))
    H = 3.0 * np.eye(n) + 0.1 * (M + M.T)  # well-conditioned SPD
    g = 2.0 * np.ones(n)
    x_qp, _, converged = _hildreth_qp(H, g, np.zeros(n), max_iterations=1000)
    x_unc = np.linalg.solve(H, g)
    assert np.all(x_unc > 0)  # interior: constraints inactive
    assert converged
    np.testing.assert_allclose(x_qp, x_unc, rtol=1e-4)


def test_hildreth_projects_negatives_to_zero():
    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    g = np.array([-1.0, 3.0])
    x, _, _ = _hildreth_qp(H, g, np.zeros(2))
    assert x[0] == 0.0
    assert x[1] == pytest.approx(1.5)


def test_hildreth_max_iterations_honored():
    H = np.eye(3)
    g = np.ones(3)
    x, sweeps, _ = _hildreth_qp(H, g, np.zeros(3), max_iterations=1)
    assert sweeps == 1
    assert x.shape == (3,)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def test_solve_louhi_core(A, b):
    spectrum, iterations, converged = solve_louhi(A, b, x0=np.ones(A.shape[1]))
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int)
    assert isinstance(converged, bool)
    resid = A @ spectrum - b
    assert np.linalg.norm(resid) < np.linalg.norm(b)


def test_solve_louhi_recovers_smooth_truth(detector, A, f_true):
    """With an informed a-priori spectrum LOUHI recovers the truth.

    The generalized smoothing anchors the underdetermined null space to
    the default spectrum, so an a-priori with the correct shape (here:
    half scale plus a constant offset) is reproduced with a high cosine
    similarity.
    """
    b = A @ f_true
    x0 = 0.5 * f_true + 0.1
    x, _, _ = solve_louhi(A, b, x0=x0, smoothness=1.0, smooth_order=1)
    cos = float(x @ f_true / (np.linalg.norm(x) * np.linalg.norm(f_true)))
    assert cos > 0.9


def test_solve_louhi_orders_run(A, b):
    for order in (0, 1, 2):
        spectrum, _, _ = solve_louhi(
            A, b, x0=np.ones(A.shape[1]), smooth_order=order, smoothness=0.5
        )
        assert spectrum.shape == (A.shape[1],)
        assert np.all(spectrum >= 0)


def test_solve_louhi_zero_smoothness(A, b):
    """smoothness=0 reduces to pure weighted least squares + projection."""
    spectrum, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]), smoothness=0.0)
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)


def test_solve_louhi_negative_smoothness(A, b):
    with pytest.raises(ValueError, match="smoothness must be non-negative"):
        solve_louhi(A, b, x0=np.ones(A.shape[1]), smoothness=-1.0)


def test_solve_louhi_max_iterations(A, b):
    spectrum, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]), max_iterations=5)
    assert spectrum.shape == (A.shape[1],)


def test_solve_louhi_zero_readings(A, b):
    with pytest.raises(ValueError, match="positive measurement"):
        solve_louhi(A, np.zeros_like(b), x0=np.ones(A.shape[1]))


def test_solve_louhi_negative_readings(A, b):
    with pytest.raises(ValueError, match="positive measurement"):
        solve_louhi(A, -np.ones_like(b), x0=np.ones(A.shape[1]))


def test_solve_louhi_empty_measurements():
    with pytest.raises(ValueError, match="non-empty"):
        solve_louhi(np.empty((0, 4)), np.empty(0), x0=np.ones(4))


def test_solve_louhi_shape_mismatch(A, b):
    with pytest.raises(ValueError, match="does not match"):
        solve_louhi(A, b[:-1], x0=np.ones(A.shape[1]))


def test_solve_louhi_bad_x0_length(A, b):
    with pytest.raises(ValueError, match="does not match"):
        solve_louhi(A, b, x0=np.ones(3))


def test_solve_louhi_explicit_sigma(A, b):
    spectrum, _, _ = solve_louhi(
        A, b, x0=np.ones(A.shape[1]), sigma=np.full(len(b), 0.05)
    )
    assert spectrum.shape == (A.shape[1],)


def test_solve_louhi_bad_sigma_shape(A, b):
    with pytest.raises(ValueError, match="sigma must have shape"):
        solve_louhi(A, b, x0=np.ones(A.shape[1]), sigma=np.ones(2))


def test_solve_louhi_sigma_clamped(A, b):
    """A zero sigma entry does not produce NaNs."""
    sigma = np.full(len(b), 0.05)
    sigma[0] = 0.0
    spectrum, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]), sigma=sigma)
    assert np.all(np.isfinite(spectrum))


def test_solve_louhi_deterministic(A, b):
    x1, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]))
    x2, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]))
    np.testing.assert_allclose(x1, x2)


# ---------------------------------------------------------------------------
# Automatic smoothing weight (nonlinear regression mode)
# ---------------------------------------------------------------------------


def test_solve_louhi_auto_smooth_reaches_target(A, b):
    sigma = 0.1 * np.maximum(b, 1e-12)
    for target in (4.0, 10.0):
        spectrum, _, _ = solve_louhi(
            A, b, x0=np.ones(A.shape[1]), auto_smooth=True, chi2_target=target
        )
        chi2 = float(np.sum(((b - A @ spectrum) / sigma) ** 2))
        assert chi2 == pytest.approx(target, rel=0.15)


def test_solve_louhi_auto_smooth_default_target(A, b):
    sigma = 0.1 * np.maximum(b, 1e-12)
    spectrum, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]), auto_smooth=True)
    chi2 = float(np.sum(((b - A @ spectrum) / sigma) ** 2))
    assert chi2 == pytest.approx(len(b), rel=0.15)


def test_solve_louhi_auto_smooth_unreachable_target(A, b):
    """A target below the achievable misfit still returns a valid spectrum."""
    spectrum, _, _ = solve_louhi(
        A, b, x0=np.ones(A.shape[1]), auto_smooth=True, chi2_target=1e-9
    )
    assert spectrum.shape == (A.shape[1],)
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0)


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_louhi_covariance_finite(A, b):
    sigma = 0.1 * np.maximum(b, 1e-12)
    x, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]))
    sd = louhi_covariance(A, sigma, 1.0, 1, np.ones(A.shape[1]), x)
    assert sd.shape == (A.shape[1],)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_louhi_covariance_zero_bins_are_zero(A, b):
    sigma = 0.1 * np.maximum(b, 1e-12)
    x, _, _ = solve_louhi(A, b, x0=np.ones(A.shape[1]))
    sd = louhi_covariance(A, sigma, 1.0, 1, np.ones(A.shape[1]), x)
    assert np.all(sd[x <= 0] == 0.0)


def test_louhi_covariance_singular_hessian():
    """A singular free-set Hessian falls back to the pseudo-inverse."""
    A = np.ones((4, 3))
    sigma = np.full(4, 0.1)
    x = np.ones(3)
    sd = louhi_covariance(A, sigma, 0.0, 0, np.ones(3), x)
    assert np.all(np.isfinite(sd))


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_unfold_louhi_basic(detector, readings):
    result = detector.unfold_louhi(readings, save_result=False)
    assert isinstance(result, dict)
    assert result["method"] == "LOUHI"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)
    assert result["smoothness"] == pytest.approx(1.0)
    assert result["smooth_order"] == 1
    assert result["auto_smooth"] is False


def test_unfold_louhi_all_spheres(detector):
    result = detector.unfold_louhi(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_louhi_aliases(detector, readings):
    res_det = detector.unfold_louhi(readings, save_result=False)
    res_fn = unfold_louhi(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "LOUHI"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_unfold_louhi_initial_spectrum(detector, readings):
    init = np.full(detector.n_energy_bins, 0.5)
    result = detector.unfold_louhi(readings, initial_spectrum=init, save_result=False)
    assert np.all(result["spectrum"] >= 0)


def test_unfold_louhi_auto_smooth(detector, readings):
    result = detector.unfold_louhi(readings, auto_smooth=True, save_result=False)
    assert result["auto_smooth"] is True
    assert np.all(result["spectrum"] >= 0)


def test_unfold_louhi_max_neutron_energy(detector, readings):
    result = detector.unfold_louhi(
        readings, save_result=False, max_neutron_energy=10.0
    )
    # the full energy grid is retained; spectrum is zeroed above cutoff
    assert result["energy"].shape == (len(detector.E_MeV),)
    assert np.all(result["spectrum"][result["energy"] > 10.0] == 0.0)


def test_unfold_louhi_deterministic(detector, readings):
    r1 = detector.unfold_louhi(readings, save_result=False)
    r2 = detector.unfold_louhi(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_louhi_save_result(detector, readings):
    detector.unfold_louhi(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "LOUHI"


def test_unfold_louhi_calculate_errors(detector, readings):
    result = detector.unfold_louhi(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_louhi_variance_reduction(detector, readings):
    result = detector.unfold_louhi(
        readings,
        calculate_errors=True,
        n_montecarlo=5,
        variance_reduction="antithetic",
        save_result=False,
    )
    assert "spectrum_uncert_mean" in result


def test_core_exports():
    from bssunfold.core import louhi_covariance, louhi_smoothing_matrix

    assert louhi_covariance is not None
    assert louhi_smoothing_matrix is not None
