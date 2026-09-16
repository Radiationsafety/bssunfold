"""Tests for the GEE unfolding method (R ``gee`` port).

Covers the working-correlation construction, the Liang-Zeger moment
estimators of ``alpha``/``phi``, the core ``gee_fit``/``solve_gee``
solver (all families x correlation structures), the robust/naive
sandwich covariances, parameter validation and the
``Detector.unfold_gee`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import estimate_alpha, solve_gee, working_correlation
from bssunfold.core.unfold_gee import gee_fit


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
# Working correlations
# ---------------------------------------------------------------------------


def test_working_correlation_independence():
    R = working_correlation(0.5, 4, "independence")
    np.testing.assert_allclose(R, np.eye(4))


def test_working_correlation_exchangeable():
    R = working_correlation(0.3, 4, "exchangeable")
    expected = 0.3 * np.ones((4, 4)) + (1.0 - 0.3) * np.eye(4)
    np.testing.assert_allclose(R, expected)


def test_working_correlation_ar1():
    R = working_correlation(0.5, 4, "ar1")
    expected = 0.5 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4)))
    np.testing.assert_allclose(R, expected)


def test_working_correlation_positive_definite():
    for a in (-0.15, 0.0, 0.3, 0.9):
        R = working_correlation(a, 6, "exchangeable")
        eig = np.linalg.eigvalsh(R)
        assert np.all(eig > 0)


def test_working_correlation_invalid_alpha():
    with pytest.raises(ValueError, match="exchangeable alpha"):
        working_correlation(-0.4, 6, "exchangeable")
    with pytest.raises(ValueError, match="ar1"):
        working_correlation(1.5, 6, "ar1")


def test_working_correlation_invalid_m():
    with pytest.raises(ValueError, match="m"):
        working_correlation(0.2, 0, "exchangeable")


def test_workfing_correlation_unknown_kind_inside_estimate():
    r = np.array([1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="kind"):
        estimate_alpha(r, "bogus")


def test_estimate_alpha_independence():
    a, phi = estimate_alpha(np.array([1.0, 2.0, 3.0]), "independence")
    assert a == 0.0
    assert np.isclose(phi, np.mean([1, 4, 9]))


def test_estimate_alpha_exchangeable_perfectly_correlated():
    r = np.array([1.0, 1.0, 1.0, 1.0])
    a, _ = estimate_alpha(r, "exchangeable")
    # alpha = 1 must be clipped into the strict SPD region (cap 0.95)
    np.testing.assert_allclose(a, 0.95, atol=1e-12)


def test_estimate_alpha_ar1_lag1():
    r = np.array([2.0, 2.0, 2.0, 2.0])
    a, phi = estimate_alpha(r, "ar1")
    np.testing.assert_allclose(a, 0.95, atol=1e-12)
    np.testing.assert_allclose(phi, 4.0)


def test_estimate_alpha_clipped_to_spd_region():
    # Alternating signs give alpha = -1 for ar1; the estimator must clip
    # into the SPD region of the exchangeable/AR-1 family.
    r = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    a, _ = estimate_alpha(r, "ar1")
    assert -1.0 < a < 1.0
    assert np.isfinite(a)


def test_estimate_alpha_too_short():
    a, phi = estimate_alpha(np.array([1.0]), "exchangeable")
    assert a == 0.0 and phi == 1.0


def test_estimate_alpha_degenerate():
    a, _ = estimate_alpha(np.array([0.0, 0.0, 0.0]), "ar1")
    assert a == 0.0


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma"])
@pytest.mark.parametrize("corstr", ["independence", "exchangeable", "ar1"])
def test_solve_gee_all_combinations(A, f_true, b, family, corstr):
    x, iterations, converged = solve_gee(A, b, family=family, corstr=corstr)
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)
    assert iterations > 0
    assert converged
    assert cosine(x, f_true) > 0.9


def test_solve_gee_returns_diagnostics(A, b):
    diag = gee_fit(A, b, corstr="exchangeable")
    for key in (
        "spectrum", "cov_robust", "cov_naive", "robust_se", "naive_se",
        "alpha", "phi", "residuals", "pearson_residuals", "pearson_chi2",
        "df", "iterations", "converged", "family", "corstr",
    ):
        assert key in diag


def test_gee_fit_covariances_psd(A, b):
    diag = gee_fit(A, b, corstr="exchangeable")
    for key in ("cov_robust", "cov_naive"):
        cov = diag[key]
        nrm = max(float(np.max(np.abs(cov))), 1.0)
        eig = np.linalg.eigvalsh(cov / nrm)
        # finite-precision covariance matrices are allowed a small
        # numerical negative jut
        assert eig[0] > -1e-6, f"{key} not PSD"
        np.testing.assert_allclose(
            diag[key], diag[key].T, atol=nrm * 1e-12
        )


def test_gee_fit_se_consistency(A, b):
    diag = gee_fit(A, b)
    np.testing.assert_allclose(diag["robust_se"], np.sqrt(
        np.maximum(np.diag(diag["cov_robust"]), 0.0)
    ))
    np.testing.assert_allclose(diag["naive_se"], np.sqrt(
        np.maximum(np.diag(diag["cov_naive"]), 0.0))
    )
    assert np.all(diag["robust_se"] >= 0)
    assert np.all(diag["naive_se"] >= 0)


def test_gee_fit_equivalence_with_tikhonov_gls(A, b):
    # independence + zero regularisation = penalised GLS with identity
    # working correlation; the projection-clipped least-squares solution.
    diag = gee_fit(
        A, b, corstr="independence", regularization=1e-6,
        max_iterations=100,
    )
    x = diag["spectrum"]
    H = A.T @ A
    x_ref = np.maximum(
        np.linalg.lstsq(H, A.T @ b, rcond=None)[0], 0.0
    )
    assert diag["converged"]
    assert cosine(x, x_ref) > 0.95


def test_gee_fit_initial_spectrum(A, b, f_true):
    x0 = f_true.copy()
    diag = gee_fit(A, b, x0=x0, max_iterations=1)
    assert diag["iterations"] == 1
    assert np.all(diag["spectrum"] >= 0)
    assert np.all(np.isfinite(diag["spectrum"]))


def test_gee_fit_converges_unconstrained(A, b):
    diag = gee_fit(
        A, b, corstr="independence", regularization=1e-3,
        tolerance=1e-8,
    )
    assert diag["converged"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_solve_gee_invalid_family(A, b):
    with pytest.raises(ValueError, match="family"):
        solve_gee(A, b, family="bogus")


def test_solve_gee_invalid_corstr(A, b):
    with pytest.raises(ValueError, match="corstr"):
        solve_gee(A, b, corstr="bogus")


def test_solve_gee_negative_regularization(A, b):
    with pytest.raises(ValueError, match="regularization"):
        solve_gee(A, b, regularization=-1.0)


def test_gee_fit_bad_system(A):
    with pytest.raises(ValueError):
        gee_fit(A, b=np.ones(A.shape[0] + 1))


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_unfold_gee(detector, A, f_true):
    b = A @ f_true
    readings = {n: float(v) for n, v in
                zip(detector.detector_names, b)}
    result = detector.unfold_gee(readings, save_result=False)
    assert result["method"] == "GEE"
    for key in ("spectrum", "doserates", "energy", "alpha", "phi",
                "robust_se", "naive_se", "spectrum_uncert_robust",
                "gee_converged", "family", "corstr"):
        assert key in result, f"missing key {key!r}"
    assert np.all(np.isfinite(result["spectrum"]))
    assert cosine(result["spectrum"], f_true) > 0.9


def test_detector_unfold_gee_max_neutron_energy(detector, A, f_true):
    b = A @ f_true
    readings = {n: float(v) for n, v in
                zip(detector.detector_names, b)}
    result = detector.unfold_gee(
        readings, save_result=False, max_neutron_energy=20.0
    )
    assert result["spectrum"].shape == (detector.n_energy_bins,)
    # bins above the cutoff are forced to zero
    assert np.all(result["spectrum"][detector.E_MeV > 20.0] == 0.0)
    assert cosine(result["spectrum"], f_true) > 0.9


def test_readings_keys_subset(detector, A, f_true):
    b = A @ f_true
    full = {n: float(v) for n, v in zip(detector.detector_names, b)}
    subset = dict(list(full.items())[:4])
    result = detector.unfold_gee(subset, save_result=False)
    assert np.all(np.isfinite(result["spectrum"]))
