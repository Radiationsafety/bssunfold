"""Tests for the SSR unfolding method (R sisireg 1.2.1 port).

Covers the Sign-Simplicity-Regression building blocks ported from the
R package ``sisireg`` (sign criteria, rolling medians, QSOR solvers,
minimum statistic, prediction), the SSR-regularised unfolding solver
(MLEM data step + SSR parsimony sweep), and the
``Detector.unfold_ssr`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_ssr
from bssunfold.core.unfold_ssr import (
    max_run_quantile,
    number_of_extrema,
    partial_sum_max,
    partial_sum_quantile,
    partial_sum_valid,
    rolling_median,
    run_valid,
    solve_ssr_full,
    ssr,
    ssr_min_statistic,
    ssr_min_statistic_ne,
    ssr_ne,
    ssr_predict,
)
from bssunfold.core.unfold_ssr import unfold_ssr as unfold_ssr_module


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
# Sign criteria (R maxRunR / fnR / psmaxR / psvalid / runvalid)
# ---------------------------------------------------------------------------


def test_max_run_quantile_known_values():
    # k = int(3.3 + 1.44 ln n), truncating conversion
    assert max_run_quantile(10) == 6
    assert max_run_quantile(60) == 9
    assert max_run_quantile(100) == 9
    assert max_run_quantile(1000) == 13


def test_max_run_quantile_invalid():
    with pytest.raises(ValueError, match="positive"):
        max_run_quantile(0)


def test_partial_sum_quantile_scalar_and_array():
    assert partial_sum_quantile(60, 9) == pytest.approx(9.0)
    # below the crossover the truncation at k dominates
    assert partial_sum_quantile(100, 4) == pytest.approx(4.0)
    out = partial_sum_quantile(100, np.array([2.0, 10.0]))
    assert out.shape == (2,)
    np.testing.assert_allclose(
        out,
        np.minimum(np.sqrt(1 + 2.33 * np.log(100) * np.array([2.0, 10.0])),
                   [2.0, 10.0]),
    )


def test_rolling_median_matches_windows():
    v = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    out = rolling_median(v, 3)
    assert out.shape == (6,)
    np.testing.assert_allclose(out, [3.0, 1.0, 4.0, 5.0, 5.0, 6.0])
    # window of one returns the data itself
    np.testing.assert_allclose(rolling_median(v, 1), v)


def test_rolling_median_invalid_window():
    with pytest.raises(ValueError, match="window k"):
        rolling_median(np.ones(4), 5)
    with pytest.raises(ValueError, match="window k"):
        rolling_median(np.ones(4), 0)


def test_start_values_layout():
    # k=5: k2=2, k1=1 -> head/tail medians of 2 points, middle rolling
    dat = np.arange(10, dtype=float)
    from bssunfold.core.unfold_ssr import _ssr_start_values

    s = _ssr_start_values(dat, 5)
    assert s[0] == pytest.approx(np.median(dat[:2]))
    assert s[-1] == pytest.approx(np.median(dat[-2:]))
    np.testing.assert_allclose(s[2:8], [2, 3, 4, 5, 6, 7])
    # even window: the middle segment wins the one-point head overlap
    s4 = _ssr_start_values(dat, 4)
    np.testing.assert_allclose(s4[1:8], rolling_median(dat, 4))


def test_partial_sum_max_known():
    dat = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    mu = np.zeros(7)
    # signs: +1 -1 -1 +1 -1 -1 -1; windows of 3: max |sum| = 3
    assert partial_sum_max(dat, mu, 3) == 3
    assert partial_sum_max(dat, mu, 1) == 1
    assert partial_sum_max(dat, mu, 7) == 3


def test_number_of_extrema():
    assert number_of_extrema([0.0, 1.0, 0.0, 1.0, 0.0]) == 3
    assert number_of_extrema([0.0, 1.0, 2.0, 3.0]) == 0
    assert number_of_extrema([3.0, 2.0, 1.0, 0.0]) == 0
    assert number_of_extrema([0.0, 1.0, 2.0, 1.0, 0.0]) == 1
    assert number_of_extrema([1.0]) == 0


def test_partial_sum_valid_small_noise():
    rng = np.random.default_rng(7)
    n = 100
    dat = np.sin(np.linspace(0, 6, n)) * 5 + 10 + 0.1 * rng.standard_normal(n)
    # a well-fitting model: residuals are small zero-mean noise
    mu = np.sin(np.linspace(0, 6, n)) * 5 + 10
    assert partial_sum_valid(dat, mu)
    # a flat model on a strongly sloped series: huge systematic runs
    dat2 = np.linspace(0, 50, 100)
    assert not partial_sum_valid(dat2, np.zeros(100))


def test_run_valid():
    rng = np.random.default_rng(11)
    n = 60
    dat = np.full(n, 5.0) + 0.1 * rng.standard_normal(n)
    assert run_valid(dat, np.full(n, 5.0))
    # 15 consecutive residuals of the same sign violate the max run
    dat2 = np.zeros(60)
    mu2 = np.zeros(60)
    mu2[20:35] = 10.0  # 15 consecutive positive residuals
    assert not run_valid(dat2, mu2)


# ---------------------------------------------------------------------------
# QSOR solvers (R ssr / ssr_ne)
# ---------------------------------------------------------------------------


def test_ssr_reduces_noise_equidistant():
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 1, 60)
    truth = np.sin(2 * np.pi * xs) * 5 + 10
    y = truth + 0.8 * rng.standard_normal(60)
    mu = ssr(y, simanz=5000)
    assert mu.shape == y.shape
    assert np.all(np.isfinite(mu))
    assert np.linalg.norm(mu - truth) < np.linalg.norm(y - truth)
    # the parsimony principle keeps the extrema count small
    assert number_of_extrema(mu) <= number_of_extrema(y)


def test_ssr_respects_boundary_values():
    rng = np.random.default_rng(2)
    y = 10 + rng.standard_normal(60)
    mu = ssr(y, y1=1.0, yn=2.0, simanz=2000)
    assert mu[0] == pytest.approx(1.0)
    assert mu[-1] == pytest.approx(2.0)


def test_ssr_l2_variant_finite():
    rng = np.random.default_rng(3)
    y = 10 + rng.standard_normal(60)
    mu = ssr(y, funk=2, simanz=1000)
    assert np.all(np.isfinite(mu))
    mu_l1 = ssr(y, funk=1, simanz=1000)
    assert np.all(np.isfinite(mu_l1))


def test_ssr_does_not_modify_input():
    rng = np.random.default_rng(4)
    y = 10 + rng.standard_normal(60)
    y_orig = y.copy()
    ssr(y, simanz=100)
    np.testing.assert_allclose(y, y_orig)


def test_ssr_invalid_inputs():
    with pytest.raises(ValueError, match="data points"):
        ssr(np.ones(7))
    with pytest.raises(ValueError, match="funk"):
        ssr(np.ones(60), funk=3)
    with pytest.raises(ValueError, match="simanz"):
        ssr(np.ones(60), simanz=0)


def test_ssr_ne_sorts_and_reduces_noise():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 1, 60))
    truth = np.sin(2 * np.pi * x) * 5 + 10
    y = truth + 0.8 * rng.standard_normal(60)
    perm = rng.permutation(60)
    xs, mu = ssr_ne(x, y, simanz=5000)
    # consistently shuffled (x, y) pairs are sorted internally with the
    # same outcome
    xs2, mu2 = ssr_ne(x[perm], y[perm], simanz=5000)
    np.testing.assert_allclose(xs, xs2)
    np.testing.assert_allclose(mu, mu2)
    assert np.linalg.norm(mu - truth) < np.linalg.norm(y - truth)


def test_ssr_ne_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ssr_ne(np.linspace(0, 1, 60), np.ones(59))


# ---------------------------------------------------------------------------
# Minimum statistic and prediction
# ---------------------------------------------------------------------------


def test_ssr_min_statistic_returns_adequate_model():
    rng = np.random.default_rng(6)
    y = np.sin(np.linspace(0, 4, 60)) * 3 + 10 + 0.5 * rng.standard_normal(60)
    mu, fn = ssr_min_statistic(y, simanz=2000)
    assert np.all(np.isfinite(mu))
    assert 0 <= fn <= max_run_quantile(60) + 1


def test_ssr_min_statistic_ne():
    rng = np.random.default_rng(8)
    x = np.sort(rng.uniform(0, 1, 60))
    y = np.sin(2 * np.pi * x) * 3 + 10 + 0.5 * rng.standard_normal(60)
    xs, mu, fn = ssr_min_statistic_ne(x, y, simanz=2000)
    np.testing.assert_allclose(xs, x)
    assert np.all(np.isfinite(mu))
    assert 0 <= fn <= max_run_quantile(60) + 1


def test_ssr_predict_interpolation_and_extrapolation():
    x = np.linspace(0, 1, 11)
    mu = 2.0 * x  # linear model
    xx = np.array([-0.5, 0.25, 0.5, 1.5])
    out = ssr_predict(x, mu, xx)
    # exact on nodes and inside, linear extrapolation with slope 2
    np.testing.assert_allclose(out, 2.0 * xx, atol=1e-12)
    # unsorted model arguments give the same prediction
    order = [3, 0, 7, 1, 9, 2, 10, 4, 6, 5, 8]
    out2 = ssr_predict(x[order], mu[order], xx)
    np.testing.assert_allclose(out2, out)


def test_ssr_predict_invalid():
    with pytest.raises(ValueError, match="same length"):
        ssr_predict(np.linspace(0, 1, 10), np.ones(9), np.zeros(3))
    with pytest.raises(ValueError, match="2 model points"):
        ssr_predict(np.array([1.0]), np.array([1.0]), np.zeros(3))


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def test_solve_ssr_recovers_smooth_truth(A, E, f_true):
    b = A @ f_true
    x, iterations, converged = solve_ssr(A, b, E_MeV=E)
    assert x.shape == (E.shape[0],)
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))
    assert iterations > 0
    assert isinstance(converged, bool)
    assert cosine(x, f_true) > 0.9


def test_solve_ssr_noisy_data(A, E, f_true):
    rng = np.random.default_rng(42)
    b = rng.poisson((A @ f_true) * 100).astype(float) / 100.0
    x, _, _ = solve_ssr(A, b, E_MeV=E)
    assert cosine(x, f_true) > 0.9


def test_solve_ssr_uses_initial_spectrum(A, E, f_true):
    b = A @ f_true
    x, _, _ = solve_ssr(A, b, x0=f_true.copy(), E_MeV=E)
    assert np.all(x >= 0)
    assert cosine(x, f_true) > 0.9


def test_solve_ssr_zero_initial_falls_back_to_flat(A, E, f_true):
    b = A @ f_true
    x_zero, _, _ = solve_ssr(A, b, x0=np.zeros(E.shape[0]), E_MeV=E)
    x_none, _, _ = solve_ssr(A, b, E_MeV=E)
    np.testing.assert_allclose(x_zero, x_none)


def test_solve_ssr_without_energy_grid(A, E, f_true):
    b = A @ f_true
    x, _, _ = solve_ssr(A, b)
    assert x.shape == (E.shape[0],)
    assert np.all(np.isfinite(x))


def test_solve_ssr_permuted_energy_grid_equivalent(A, E, f_true):
    # permuting the bin order (grid AND response columns consistently)
    # describes the same physical problem
    rng = np.random.default_rng(0)
    b = A @ f_true
    perm = rng.permutation(E.shape[0])
    E_perm = E[perm]
    A_perm = A[:, perm]
    x_fwd, _, _ = solve_ssr(A, b, E_MeV=E)
    x_perm, _, _ = solve_ssr(A_perm, b, E_MeV=E_perm)
    # map the permuted result back to ascending energies
    np.testing.assert_allclose(
        x_perm[np.argsort(E_perm)], x_fwd, rtol=1e-8
    )


def test_solve_ssr_fixed_fn(A, E, f_true):
    b = A @ f_true
    diag = solve_ssr_full(A, b, E_MeV=E, fn=4, max_iterations=50)
    assert diag["fn"] == 4
    assert diag["fn_start"] == 4
    assert len(diag["fn_ladder"]) == 1


@pytest.mark.parametrize("bad_fn", (0, -1, "bogus", 2.5))
def test_solve_ssr_invalid_fn(A, E, bad_fn):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="fn must be"):
        solve_ssr_full(A, b, E_MeV=E, fn=bad_fn)


def test_solve_ssr_full_diagnostics(A, E, f_true):
    b = A @ f_true
    diag = solve_ssr_full(A, b, E_MeV=E, max_iterations=60)
    for key in (
        "spectrum",
        "n_iterations",
        "converged",
        "fn",
        "fn_start",
        "k_run",
        "n_extrema",
        "ps_valid_data",
        "run_valid_data",
        "max_run_data",
        "ssr_sweeps",
        "fn_ladder",
    ):
        assert key in diag, key
    n = E.shape[0]
    assert diag["k_run"] == max_run_quantile(n)
    assert diag["fn"] <= diag["fn_start"] + 1
    assert diag["n_extrema"] <= 2 * diag["k_run"]
    assert diag["max_run_data"] >= 0
    assert diag["ssr_sweeps"] > 0
    assert isinstance(diag["fn_ladder"], list) and diag["fn_ladder"]
    for entry in diag["fn_ladder"]:
        assert set(entry) >= {
            "fn",
            "ps_valid_data",
            "run_valid_data",
            "n_extrema",
            "n_iterations",
            "converged",
        }


def test_solve_ssr_full_ladder_descends_or_stops(A, E, f_true):
    b = A @ f_true
    diag = solve_ssr_full(A, b, E_MeV=E, max_iterations=60)
    fns = [entry["fn"] for entry in diag["fn_ladder"]]
    # the ladder starts at fn_start and descends by one
    assert fns[0] == diag["fn_start"]
    assert all(fns[i] - fns[i + 1] == 1 for i in range(len(fns) - 2))
    # the reported fn is the last ladder entry (or the fallback)
    assert diag["fn"] in (fns[-1], diag["fn_start"] + 1)


def test_solve_ssr_too_few_readings(E):
    A_small = np.ones((2, E.shape[0]))
    b = np.ones(2)
    with pytest.raises(ValueError, match="detector readings"):
        solve_ssr_full(A_small, b, E_MeV=E)


def test_solve_ssr_too_few_energy_bins(A):
    A_small = A[:, :6]
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="energy"):
        solve_ssr_full(A_small, b, E_MeV=np.arange(6.0))


def test_solve_ssr_energy_grid_mismatch(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="E_MeV"):
        solve_ssr_full(A, b, E_MeV=E[:-1])


def test_solve_ssr_invalid_max_iterations(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="max_iterations"):
        solve_ssr_full(A, b, E_MeV=E, max_iterations=0)


def test_solve_ssr_x0_length_mismatch(A, E):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError, match="x0"):
        solve_ssr_full(A, b, x0=np.ones(E.shape[0] + 1), E_MeV=E)


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_unfold_ssr(detector, readings):
    result = detector.unfold_ssr(readings)
    assert result["method"] == "SSR"
    for key in (
        "spectrum",
        "doserates",
        "effective_readings",
        "residual",
        "fn",
        "fn_start",
        "k_run",
        "n_extrema",
        "ps_valid_data",
        "run_valid_data",
        "ssr_converged",
    ):
        assert key in result, key
    assert np.all(result["spectrum"] >= 0)
    assert np.all(np.isfinite(result["spectrum"]))


def test_detector_unfold_ssr_save_and_errors(detector, readings):
    result = detector.unfold_ssr(
        readings,
        calculate_errors=True,
        n_montecarlo=5,
        save_result=True,
        random_state=1,
    )
    assert "spectrum_uncert_mean" in result
    assert "saved_key" in result
    assert result["saved_key"] in detector.results_history


def test_detector_unfold_ssr_max_energy(detector, readings):
    result = detector.unfold_ssr(readings, max_neutron_energy=5.0)
    E = result["energy"]
    above = E > 5.0
    assert np.all(result["spectrum"][above] == 0)
    assert np.any(result["spectrum"][~above] > 0)


def test_detector_unfold_ssr_custom_params(detector, readings):
    result = detector.unfold_ssr(
        readings,
        fn=3,
        max_iterations=60,
        smooth_every=2,
        inner_sweeps=2,
        fn_ladder_cap=1,
    )
    assert result["fn"] == 3
    assert np.all(np.isfinite(result["spectrum"]))


def test_module_wrapper_matches_detector(detector, readings):
    r1 = detector.unfold_ssr(dict(readings), max_iterations=80)
    r2 = unfold_ssr_module(
        detector_names=detector.detector_names,
        n_energy_bins=detector.E_MeV.shape[0],
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        max_iterations=80,
    )
    np.testing.assert_allclose(r1["spectrum"], r2["spectrum"], rtol=1e-10)
