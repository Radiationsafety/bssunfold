"""Tests for the B-spline MLEM unfolding method (MLEM-BS).

Covers the B-spline MLEM algorithm of Mazankova et al., "Experimental
Measurement of Neutron Flux and Its Mathematical Data Processing"
(CNDGS'2026, https://doi.org/10.47459/cndcgs.2026.61): the B-spline basis
parameterisation of the flux (x = B b, effective matrix RB = R B), the
regularized MLEM iteration with the second-derivative penalty
``P(b) = ||D^(2) b||_2^2`` and the sieve restriction to non-negative
coefficients (Eq. 4-5 of the paper with the Szkutnik sieve), the ``K_S``
goodness-of-fit statistic used for automatic parameter selection (Eq. 6),
the Poisson bootstrap confidence intervals (Eqs. 7-9) and the
``Detector.unfold_mlem_bs`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_mlem, solve_mlem_bs, solve_mlem_bs_full, unfold_mlem
from bssunfold.core.unfold_mlem_bs import (
    AUTO_BETA_RELATIVE_GRID,
    build_bspline_basis,
    ks_statistic,
    second_difference_matrix,
    unfold_mlem_bs,
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


@pytest.fixture
def readings_clean(A, f_true):
    exact = A @ f_true
    return {
        name: float(v) for name, v in zip(detector_names(A), exact)
    }


def detector_names(A):
    return [f"sphere{i}" for i in range(len(A))]


def _second_diff_norm(x):
    return float(np.linalg.norm(np.diff(x, 2)))


# ---------------------------------------------------------------------------
# Basis / penalty / statistic utilities
# ---------------------------------------------------------------------------


def test_build_bspline_basis_shape_and_partition(E):
    """B is (n_bins, n_basis), non-negative, rows sum to 1 (partition of unity)."""
    B = build_bspline_basis(E, 40, spline_order=4)
    assert B.shape == (len(E), 40)
    assert np.all(B >= 0)
    # rows with full support sum to unity (edges may be truncated by clamping)
    inner = B.sum(axis=1)
    assert np.allclose(inner[2:-2], 1.0, atol=1e-10)


def test_build_bspline_basis_log_knots(E):
    """knot_spacing='log' places knots on a logarithmic grid."""
    B = build_bspline_basis(E, 30, spline_order=4, knot_spacing="log")
    assert B.shape == (len(E), 30)
    assert np.all(np.isfinite(B))


def test_build_bspline_basis_auto_resolves_to_log(E):
    """The grid spans ~9 decades, so 'auto' must pick log knots."""
    from bssunfold.core.unfold_mlem_bs import _resolve_knot_spacing

    assert _resolve_knot_spacing("auto", E) == "log"


def test_build_bspline_basis_validation(E):
    with pytest.raises(ValueError):
        build_bspline_basis(E, 2, spline_order=4)  # n_basis < order
    with pytest.raises(ValueError):
        build_bspline_basis(E, 0)


def test_second_difference_matrix():
    D2 = second_difference_matrix(6)
    assert D2.shape == (4, 6)
    x = np.array([1.0, 2, 4, 7, 11, 16])  # second differences == 1
    assert np.allclose(D2 @ x, 1.0)
    # ||D2 b||^2 gradient consistency: grad = 2 D2^T D2 b
    b = np.random.default_rng(0).normal(size=6)
    eps = 1e-7
    num = (np.sum((D2 @ (b + eps * np.eye(6)[0])) ** 2)
           - np.sum((D2 @ (b - eps * np.eye(6)[0])) ** 2)) / (2 * eps)
    ana = (2 * D2.T @ D2 @ b)[0]
    assert num == pytest.approx(ana, rel=1e-5)


def test_ks_statistic_zero_for_poisson_consistent_fit():
    """K_S ~ 0 when the model matches the data-generating Poisson mean."""
    rng = np.random.default_rng(1)
    model = rng.uniform(50, 200, size=100)
    counts = rng.poisson(model).astype(float)
    assert ks_statistic(counts, model) < 0.1
    # a wrong model gives a clearly positive statistic
    assert ks_statistic(counts, model * 0.5) > 0.1


def test_ks_statistic_formula():
    b = np.array([9.0, 11.0, 12.0])
    m = np.array([10.0, 10.0, 10.0])
    expected = abs(np.sum((b - m) ** 2) / np.sum(m) - 1.0)
    assert ks_statistic(b, m) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def test_solve_mlem_bs_tuple_api(A, E, f_true):
    exact = A @ f_true
    spec, iters, _conv = solve_mlem_bs(A, exact, x0=None, E_MeV=E, max_iterations=300)
    assert spec.shape == (len(E),)
    assert 0 < iters <= 300
    assert float(spec.min()) >= 0.0
    assert np.all(np.isfinite(spec))


def test_solve_mlem_bs_full_diagnostics(A, E, f_true):
    exact = A @ f_true
    result = solve_mlem_bs_full(A, exact, x0=None, E_MeV=E, max_iterations=300)
    spec, iters, diag = result["spectrum"], result["iterations"], result
    assert spec.shape == (len(E),)
    assert diag["method"] == "MLEM-BS"
    for key in (
        "ks_history",
        "ks_final",
        "chi2_pearson",
        "n_basis",
        "interior_knots",
        "beta_effective",
        "coefficients",
    ):
        assert key in diag
    assert len(diag["ks_history"]) == iters + 1
    assert diag["coefficients"].shape == (diag["n_basis"],)
    assert np.all(diag["coefficients"] >= 0)  # sieve: non-negative b
    assert len(diag["interior_knots"]) == diag["n_basis"] - 4  # order p = 4


def test_solve_mlem_bs_reconstructs_readings(A, E, f_true):
    exact = A @ f_true
    spec, *_ = solve_mlem_bs(A, exact, x0=None, E_MeV=E, max_iterations=500)
    resid = np.linalg.norm(A @ spec - exact) / np.linalg.norm(exact)
    assert resid < 0.05


def test_solve_mlem_bs_length_mismatch(A, E):
    with pytest.raises(ValueError):
        solve_mlem_bs(A, np.ones(A.shape[0]), x0=None, E_MeV=E[:-1])


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_mlem_bs_smoothing_vs_mlem(detector, readings):
    """The B-spline sieve must suppress noise: much smoother than plain MLEM."""
    res_mlem = detector.unfold_mlem(readings, max_iterations=300)
    res_bs = detector.unfold_mlem_bs(readings, max_iterations=500)
    tv_mlem = _second_diff_norm(res_mlem["spectrum"])
    tv_bs = _second_diff_norm(res_bs["spectrum"])
    assert tv_bs < tv_mlem / 5.0
    # without losing the data: residual stays within a few times the MLEM one
    assert np.linalg.norm(res_bs["residual"]) <= 3.0 * np.linalg.norm(
        res_mlem["residual"]
    )
    assert res_bs["method"] == "MLEM-BS"


def test_penalty_strength_controls_smoothing(detector, readings):
    res_plain = detector.unfold_mlem_bs(readings, max_iterations=500)
    res_pen = detector.unfold_mlem_bs(
        readings, max_iterations=800, beta_relative=1e-2
    )
    assert res_pen["beta_effective"] > 0
    assert _second_diff_norm(res_pen["spectrum"]) <= _second_diff_norm(
        res_plain["spectrum"]
    )


def test_auto_parameter_selection_ks_minimum(detector, readings):
    """auto_params=True selects (N_s, beta, iterations) by minimizing K_S (Eq. 6)."""
    res = detector.unfold_mlem_bs(readings, max_iterations=600, auto_params=True)
    auto = res["auto_selection"]
    chosen = auto["chosen"]
    assert len(auto["candidates"]) > 0
    assert chosen["ks"] == min(c["ks"] for c in auto["candidates"])
    assert chosen["beta_relative"] in AUTO_BETA_RELATIVE_GRID
    assert np.all(np.isfinite(res["spectrum"]))


def test_bootstrap_ci(detector, readings, E):
    """Poisson bootstrap percentile CI (Eqs. 7-9)."""
    res = detector.unfold_mlem_bs(
        readings, max_iterations=500, bootstrap_ci=True,
        n_bootstrap=40, random_state=7,
    )
    lo, hi = res["ci_low"], res["ci_high"]
    assert res["ci_level"] == 0.95
    assert lo.shape == hi.shape == (len(E),)
    assert np.all(lo <= hi)
    fast = (E > 0.5) & (E < 12.0)
    assert lo[fast].max() > 0


def test_bootstrap_ci_coverage_overdetermined():
    """95% CI covers the truth on an overdetermined system (m >> N_s).

    This is the regime the bootstrap of Eqs. 7-9 is designed for (the
    paper's experiment has thousands of measured bins vs N_s = 200).
    """
    rng = np.random.default_rng(0)
    m_det, n_bins = 40, 100
    E_b = np.geomspace(1e-2, 15.0, n_bins)
    centers = np.geomspace(3e-2, 12.0, m_det)
    A_b = np.array(
        [np.exp(-0.5 * ((np.log(E_b) - np.log(c)) / 0.55) ** 2) for c in centers]
    ) * rng.uniform(0.8, 1.2, size=(m_det, 1))
    f_b = (
        0.55 * np.sqrt(E_b / 2.0) * np.exp(-E_b / 2.0)
        + 0.2 * np.exp(-((E_b - 4.5) / 1.2) ** 2)
    )
    f_b *= 100.0 / f_b.max()
    b_b = rng.poisson(A_b @ f_b).astype(float)
    names = [f"v{i}" for i in range(m_det)]
    res = unfold_mlem_bs(
        names, n_bins, E_b, {n: A_b[i] for i, n in enumerate(names)},
        {n: np.zeros(n_bins) for n in names},
        lambda r: None,
        {n: float(v) for n, v in zip(names, b_b)},
        n_basis=25, max_iterations=1500, bootstrap_ci=True, n_bootstrap=60,
        random_state=11,
    )
    fast = E_b > 0.3
    coverage = np.mean(
        (f_b[fast] >= res["ci_low"][fast]) & (f_b[fast] <= res["ci_high"][fast])
    )
    assert coverage >= 0.75


def test_max_neutron_energy_truncation(detector, readings, E):
    res = detector.unfold_mlem_bs(
        readings, max_iterations=300, max_neutron_energy=10.0,
        bootstrap_ci=True, n_bootstrap=10, random_state=3,
    )
    assert res["spectrum"].shape == (len(E),)
    assert res["spectrum"][E > 10.0].max(initial=0.0) == 0.0
    assert res["ci_low"].shape == (len(E),)


def test_clean_data_small_residual(detector, A, f_true):
    readings = {
        name: float(v)
        for name, v in zip(detector.detector_names, A @ f_true)
    }
    res = detector.unfold_mlem_bs(readings, max_iterations=800)
    rel = np.linalg.norm(res["residual"]) / np.linalg.norm(A @ f_true)
    assert rel < 0.05


def test_validation_errors(detector, readings, A, E):
    with pytest.raises(ValueError):
        detector.unfold_mlem_bs(readings, beta=1.0, beta_relative=1.0)
    with pytest.raises(ValueError):
        detector.unfold_mlem_bs(readings, knot_spacing="cubic")
    with pytest.raises(ValueError):
        solve_mlem_bs(A, np.ones(A.shape[0]), x0=None, E_MeV=E[:-1])


def test_linear_grid_paper_style(detector, E):
    """Linear binning (the paper's setup) works via uniform knots."""
    E_lin = np.linspace(0.5, 12.0, 120)
    f_lin = (
        0.55 * np.sqrt(E_lin / 2.0) * np.exp(-E_lin / 2.0)
        + 0.2 * np.exp(-((E_lin - 4.5) / 1.2) ** 2)
    )
    f_lin *= 1e4 / f_lin.max()
    sens = [detector.sensitivities[n] for n in detector.detector_names]
    A_lin = np.array([np.interp(E_lin, E, s_row) for s_row in sens])
    spec, _, _ = solve_mlem_bs(
        A_lin, A_lin @ f_lin * 10, x0=None, E_MeV=E_lin, n_basis=30
    )
    assert spec.shape == (120,)
    assert np.all(np.isfinite(spec))


def test_backward_compatible_api(A, detector, readings, E):
    """Existing MLEM entry points are unaffected by the new module."""
    spec, *_ = solve_mlem(
        A, np.array(list(readings.values())), x0=np.ones(len(E)) * 0.5
    )
    assert spec.shape == (len(E),)
    res = unfold_mlem(
        detector.detector_names, detector.n_energy_bins, E,
        {k: detector.sensitivities[k] for k in detector.detector_names},
        detector.cc_icrp116, lambda r: None, readings,
    )
    assert res["spectrum"].shape == (len(E),)
