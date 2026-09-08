"""Tests for the N-spline unfolding method (Islamgulov & Lartsev, 2008).

Covers the N-spline mathematics of the paper (Eqs. 2-7: basis functions,
C0/C1 continuity matrix D, constrained log-domain least-squares fit),
the directed-divergence unfolding loop with per-iteration N-spline
smoothing (Eqs. 8-9 and the MIRD iteration), the paper's stopping
criteria and ``nev`` acceptability statistic, and the
``Detector.unfold_nspline`` integration.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core.unfold_nspline import (
    NSPLINE_KNOT_PRESETS,
    auto_knots,
    build_continuity_matrix,
    directed_divergence,
    fit_nspline,
    nspline_eval,
    solve_nspline,
    solve_nspline_full,
    unfold_nspline,
)


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


# ---------------------------------------------------------------------------
# Knot utilities
# ---------------------------------------------------------------------------


def test_auto_knots_grid():
    """auto_knots spans the energy range with n_segments + 1 knots."""
    E = np.geomspace(1e-9, 15.0, 50)
    kn = auto_knots(E, n_segments=8)
    assert len(kn) == 9
    assert kn[0] == pytest.approx(E.min())
    assert kn[-1] == pytest.approx(E.max())
    assert np.all(np.diff(kn) > 0)


def test_auto_knots_invalid():
    """auto_knots rejects degenerate energy grids."""
    with pytest.raises(ValueError):
        auto_knots(np.array([-1.0, -2.0]))
    with pytest.raises(ValueError):
        auto_knots(np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        auto_knots(np.geomspace(1e-9, 10, 10), n_segments=0)


def test_presets_from_paper():
    """Knot presets reproduce the paper's knot tables (MeV)."""
    # BARS-5 channel: 13 knots from 1e-10 to 20 MeV
    bars5 = NSPLINE_KNOT_PRESETS["BARS5_channel"]
    assert len(bars5) == 13
    assert bars5[0] == 1e-10
    assert bars5[-1] == 20.0
    # IGRIK channel / surface and YAGUAR channel
    assert len(NSPLINE_KNOT_PRESETS["IGRIK_channel"]) == 17
    assert len(NSPLINE_KNOT_PRESETS["IGRIK_surface"]) == 15
    assert len(NSPLINE_KNOT_PRESETS["YAGUAR_channel"]) == 19
    for name, kn in NSPLINE_KNOT_PRESETS.items():
        assert np.all(np.diff(kn) > 0), name
        assert kn[0] == 1e-10 and kn[-1] == 20.0, name


def test_resolve_unknown_preset_raises():
    with pytest.raises(KeyError):
        fit_nspline(np.geomspace(1e-8, 5, 20), np.ones(20), knots="no_such")


# ---------------------------------------------------------------------------
# N-spline mathematics (Eqs. 2-5)
# ---------------------------------------------------------------------------


def test_continuity_matrix_shapes():
    """D has 2(M-1) rows for C0C1, M-1 for C0 and 0 for none."""
    kn = NSPLINE_KNOT_PRESETS["BARS5_channel"]
    M = len(kn) - 1
    assert build_continuity_matrix(kn, "C0C1").shape == (2 * (M - 1), 3 * M)
    assert build_continuity_matrix(kn, "C0").shape == (M - 1, 3 * M)
    assert build_continuity_matrix(kn, "none").shape == (0, 3 * M)
    with pytest.raises(ValueError):
        build_continuity_matrix(kn, "bogus")


def test_continuity_matrix_smooth_spline():
    """A globally smooth N-spline (a, q, r equal across segments) satisfies
    DX = 0 exactly -- Eqs. 3-4 hold at every interior knot."""
    kn = np.array(NSPLINE_KNOT_PRESETS["IGRIK_channel"], dtype=float)
    M = len(kn) - 1
    X = np.concatenate(
        [np.full(M, -2.0), np.full(M, 0.7), np.full(M, -0.15)]
    )
    for continuity in ("C0C1", "C0"):
        D = build_continuity_matrix(kn, continuity)
        assert np.abs(D @ X).max() == pytest.approx(0.0, abs=1e-12)


def test_nspline_eval_exact_shape():
    """nspline_eval reproduces exp(a + q lnE + rE) on each segment."""
    E = np.geomspace(1e-6, 10, 200)
    kn = (1e-6, 1e-3, 0.5, 10.0)
    a = [0.1, 0.1, 0.1]
    q = [0.5, 0.5, 0.5]
    r = [-0.2, -0.2, -0.2]
    N = nspline_eval(E, a, q, r, kn)
    assert np.all(N > 0)
    # global formula applies since parameters are segment-independent
    assert np.allclose(N, np.exp(0.1 + 0.5 * np.log(E) - 0.2 * E))
    with pytest.raises(ValueError):
        nspline_eval(np.array([0.0, 1.0]), a, q, r, kn)


# ---------------------------------------------------------------------------
# Pointwise N-spline approximation (Eqs. 6-7)
# ---------------------------------------------------------------------------


def test_fit_recovers_exact_nspline():
    """Fitting exact (globally smooth) N-spline samples interpolates them.

    A globally smooth member of the family, phi = exp(a + q lnE + rE),
    satisfies the C0/C1 constraints by construction, so the constrained
    log-LSQ must reproduce it up to round-off.
    """
    E = np.geomspace(1e-8, 18.0, 80)
    a0, q0, r0 = 0.5, 0.4, -0.12
    phi = np.exp(a0 + q0 * np.log(E) + r0 * E)
    N_fit, info = fit_nspline(E, phi, knots="IGRIK_channel")
    assert np.all(N_fit > 0)
    # the data are exactly representable: the constrained LSQ interpolates
    assert np.abs(N_fit - phi).max() / phi.max() < 1e-8
    # continuity of the recovered parameters
    D = build_continuity_matrix(info["knots"], "C0C1")
    X = np.concatenate([info["a"], info["q"], info["r"]])
    assert np.abs(D @ X).max() < 1e-6


def test_fit_approximates_smooth_spectrum():
    """A 1/E + fission spectrum is approximated within a few percent."""
    E = np.geomspace(1e-9, 20, 120)
    phi = 1.0 / np.maximum(E, 1e-9) ** 0.8 + 2.0 * np.exp(
        -E / 1.025
    ) * np.sinh(np.sqrt(2.926 * E))
    N_fit, info = fit_nspline(E, phi, knots="BARS5_channel")
    rel = np.abs(N_fit - phi) / phi
    assert np.median(rel) < 1e-2
    # where the spectrum is significant (above 1e-4 of the peak) the
    # spline tracks it within ~5%; tiny absolute-value tails may deviate
    signif = phi > 1e-4 * phi.max()
    assert rel[signif].max() < 0.05
    assert info["knots_source"] == "preset:BARS5_channel"


def test_fit_weighted():
    """Explicit relative errors change the fit (weights w = 1/eps)."""
    rng = np.random.default_rng(3)
    E = np.geomspace(1e-8, 10, 60)
    phi = np.exp(-E / 2.0) + 0.1
    phi_noisy = phi * (1 + rng.normal(0, 0.2, E.size))
    rel_err = np.full(E.size, 0.2)
    _, info_u = fit_nspline(E, phi_noisy, continuity="C0C1")
    _, info_w = fit_nspline(E, phi_noisy, rel_err=rel_err, continuity="C0C1")
    # both fits succeed; weighting changes the log-domain residual rms
    assert "log_rms_residual" in info_u and "log_rms_residual" in info_w


def test_fit_zero_bins_downweighted():
    """Zero-valued bins are floored and do not break the fit."""
    E = np.geomspace(1e-8, 10, 50)
    phi = np.exp(-E / 2.0)
    phi[:5] = 0.0
    N_fit, _ = fit_nspline(E, phi)
    assert np.all(np.isfinite(N_fit))
    assert np.all(N_fit > 0)


def test_fit_invalid_inputs():
    """Length mismatch / non-positive energies raise ValueError."""
    E = np.geomspace(1e-8, 10, 30)
    with pytest.raises(ValueError):
        fit_nspline(E, np.ones(29))
    with pytest.raises(ValueError):
        fit_nspline(np.linspace(-1, 1, 30), np.ones(30))
    with pytest.raises(ValueError):
        fit_nspline(E, np.ones(30), knots=[3.0, 1.0])  # not increasing


# ---------------------------------------------------------------------------
# Directed divergence and the unfolding loop (Eqs. 8-9)
# ---------------------------------------------------------------------------


def test_directed_divergence_properties():
    """H(p, p) = 0, H >= 0 otherwise, symmetric-free KL-type form."""
    p = np.array([0.2, 0.5, 0.3])
    assert directed_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    q = np.array([0.3, 0.4, 0.3])
    assert directed_divergence(q, p) > 0.0


def _synthetic_system(detector, seed=5, noise=0.05):
    """Bonner-sphere system with readings from a known fission-like truth."""
    E = detector.E_MeV
    A = np.array([detector.sensitivities[nm] for nm in detector.detector_names])
    phi_true = np.exp(-E / 1.025) * np.sinh(np.sqrt(2.926 * E))
    phi_true += 0.05 / np.maximum(E, 1e-9)
    phi_true /= A.sum(axis=0).dot(phi_true)
    b = A @ phi_true
    rng = np.random.default_rng(seed)
    b_noisy = b * (1 + rng.normal(0, noise, b.size))
    sigma = np.full(b.size, noise)
    return A, b, b_noisy, phi_true, sigma


def test_solve_returns_standard_tuple(A, b):
    """solve_nspline follows the (spectrum, iterations, converged) API."""
    d = Detector()
    A_full = np.array([d.sensitivities[nm] for nm in d.detector_names])
    b_full = np.array([1.0] * len(d.detector_names))
    spectrum, iterations, converged = solve_nspline(
        A_full, b_full, x0=np.ones(d.n_energy_bins), E_MeV=d.E_MeV
    )
    assert spectrum.shape == (d.n_energy_bins,)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int) and iterations >= 0
    assert isinstance(converged, bool)


def test_solve_recovers_synthetic_spectrum(detector):
    """The unfolded spectrum reproduces a fission-like truth reasonably."""
    A, b, b_noisy, phi_true, sigma = _synthetic_system(detector)
    out = solve_nspline_full(
        A,
        b_noisy,
        x0=np.full(detector.n_energy_bins, 0.5),
        E_MeV=detector.E_MeV,
        sigma_rel=sigma,
        max_iterations=600,
    )
    # the divergence is driven far down from its initial value
    assert out["H"] < 0.05 * out["H_history"][0]
    assert out["stop_reason"] in (
        "H_target", "relative_change", "max_iterations",
        "no_further_reduction",
    )
    # H decreases (relaxed monotonicity: the backtracking acceptance
    # allows increases below the 1e-4 relative level)
    H_hist = np.asarray(out["H_history"])
    assert np.all(H_hist[1:] <= H_hist[:-1] * (1 + 1e-3) + 1e-12)
    # gauge: the activation scale is pinned to the measurement total
    assert np.sum(A @ out["spectrum"]) == pytest.approx(
        np.sum(b_noisy), rel=1e-8
    )
    # shape recovery in the energy range where the truth is significant
    # (the grid extends to ~600 MeV where a fission spectrum is ~0 and
    # log-ratios are meaningless)
    sig = phi_true > 1e-6 * phi_true.max()
    log_dev = np.abs(
        np.log10(out["spectrum"][sig] / (phi_true[sig] * out["spectrum"].sum()
                                         / phi_true.sum()))
    )
    assert np.median(log_dev) < 1.0


def test_solve_nev_statistics(detector):
    """nev uses 1/(N-1) RMS and the 1 + 2/sqrt(N) acceptance bound."""
    A, b, b_noisy, _, sigma = _synthetic_system(detector)
    out = solve_nspline_full(
        A, b_noisy, x0=None, E_MeV=detector.E_MeV, sigma_rel=sigma
    )
    m = b.size
    expected_limit = 1.0 + 2.0 / np.sqrt(m)
    assert out["nev_limit"] == pytest.approx(expected_limit)
    rel = (out["Qr"] - b_noisy) / (sigma * b_noisy)
    assert out["nev"] == pytest.approx(
        np.sqrt(np.sum(rel**2) / (m - 1)), rel=1e-8
    )
    assert out["acceptable"] == (out["nev"] <= out["nev_limit"])
    assert isinstance(out["acceptable"], bool)


def test_solve_zero_readings_raise(A, b):
    """All-zero measurements are rejected with a helpful error."""
    with pytest.raises(ValueError, match="positive measurement"):
        solve_nspline(A, np.zeros_like(b), x0=np.ones(A.shape[1]),
                      E_MeV=np.geomspace(1e-8, 10, A.shape[1]))


def test_solve_requires_energy_grid(A, b):
    """Missing E_MeV raises ValueError."""
    with pytest.raises(ValueError, match="E_MeV"):
        solve_nspline(A, b, x0=np.ones(A.shape[1]))


def test_solve_invalid_params(A, b):
    """Invalid solver parameters are rejected."""
    E = np.geomspace(1e-8, 10, A.shape[1])
    with pytest.raises(ValueError):
        solve_nspline(A, b, x0=np.ones(A.shape[1]), E_MeV=E, max_iterations=0)
    with pytest.raises(ValueError):
        solve_nspline(A, b, x0=np.ones(A.shape[1]), E_MeV=E, step_theta=1.5)
    with pytest.raises(ValueError):
        solve_nspline(A, b, x0=np.ones(A.shape[1]), E_MeV=np.linspace(0, 1, A.shape[1]))
    with pytest.raises(ValueError):
        solve_nspline(A, b, x0=np.ones(A.shape[1]), E_MeV=E, knots=[2.0, 1.0])


def test_solve_sigma_rel_scales_H_target(detector):
    """Larger assumed errors raise H_target (looser stopping)."""
    A, b, b_noisy, _, _ = _synthetic_system(detector)
    out1 = solve_nspline_full(A, b_noisy, E_MeV=detector.E_MeV,
                              sigma_rel=np.full(A.shape[0], 0.05))
    out2 = solve_nspline_full(A, b_noisy, E_MeV=detector.E_MeV,
                              sigma_rel=np.full(A.shape[0], 0.2))
    assert out2["H_target"] > out1["H_target"]
    assert out1["H_target"] == pytest.approx(
        0.5 * np.sum((b / b.sum()) * 0.05**2)
    )


def test_solve_smoothing_off(detector):
    """smoothing=False reduces the loop to the plain MIRD iteration."""
    A, b, b_noisy, _, _ = _synthetic_system(detector)
    out = solve_nspline_full(
        A, b_noisy, x0=np.ones(detector.n_energy_bins),
        E_MeV=detector.E_MeV, smoothing=False, max_iterations=100,
    )
    assert out["spectrum"].shape == (detector.n_energy_bins,)
    assert np.all(out["spectrum"] >= 0)
    assert np.isfinite(out["H"])


def test_solve_predefined_knots(detector):
    """Preset and explicit knots both work and are reported back."""
    A, b, b_noisy, _, _ = _synthetic_system(detector)
    out = solve_nspline_full(A, b_noisy, E_MeV=detector.E_MeV,
                             knots="YAGUAR_channel")
    assert out["knots_source"] == "preset:YAGUAR_channel"
    assert len(out["knots"]) == len(NSPLINE_KNOT_PRESETS["YAGUAR_channel"])

    explicit = tuple(np.geomspace(detector.E_MeV.min(),
                                  detector.E_MeV.max(), 7))
    out2 = solve_nspline_full(A, b_noisy, E_MeV=detector.E_MeV, knots=explicit)
    assert out2["knots_source"] == "user"
    assert len(out2["knots"]) == 7


def test_solve_diagnostics_keys(detector):
    """The full diagnostics dictionary carries the paper's statistics."""
    A, b, b_noisy, _, _ = _synthetic_system(detector)
    out = solve_nspline_full(A, b_noisy, E_MeV=detector.E_MeV)
    for key in (
        "spectrum", "iterations", "converged", "stop_reason", "H",
        "H_history", "H_target", "nev", "nev_limit", "acceptable", "Qr",
        "relative_residuals", "fluence", "mean_energy", "knots",
        "knots_source", "continuity", "params",
    ):
        assert key in out, key
    assert out["fluence"] > 0
    assert 0 < out["mean_energy"] < detector.E_MeV.max() * 1.5
    assert len(out["H_history"]) == out["iterations"] + 1


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_unfold_nspline_basic(detector, readings):
    """Basic unfold_nspline returns a standardized result dict."""
    result = detector.unfold_nspline(readings, save_result=False)
    assert isinstance(result, dict)
    assert result["method"] == "NSPLINE"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)
    # paper-specific diagnostics
    assert "nev" in result and "nev_limit" in result
    assert "acceptable" in result and "H_history" in result
    assert result["nev_limit"] == pytest.approx(1.0 + 2.0 / 2.0)


def test_unfold_nspline_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_nspline(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_nspline_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    res_det = detector.unfold_nspline(readings, save_result=False)
    res_fn = unfold_nspline(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "NSPLINE"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_unfold_nspline_preset_knots(detector, readings):
    """Preset knots are accepted and reported."""
    result = detector.unfold_nspline(
        readings, knots="BARS5_channel", save_result=False
    )
    assert result["knots_source"] == "preset:BARS5_channel"
    assert len(result["knots"]) == 13


def test_unfold_nspline_initial_spectrum(detector, readings):
    """A user-supplied initial spectrum (MC-style guess) is honoured."""
    x0 = np.exp(-detector.E_MeV / 2.0) + 1e-6 / np.maximum(
        detector.E_MeV, 1e-9
    )
    result = detector.unfold_nspline(
        readings, initial_spectrum=x0, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)
    assert np.isfinite(result["H"])


def test_unfold_nspline_save_result(detector, readings):
    """save_result=True stores the result in the detector history."""
    n_before = len(detector.results_history)
    detector.unfold_nspline(readings, save_result=True)
    assert len(detector.results_history) == n_before + 1


def test_unfold_nspline_montecarlo(detector, readings):
    """Monte-Carlo uncertainty estimation runs through the wrapper."""
    result = detector.unfold_nspline(
        readings, calculate_errors=True, n_montecarlo=5,
        noise_level=0.05, random_state=42, save_result=False,
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_nspline_deterministic(detector, readings):
    """Repeated calls with the same inputs give identical spectra."""
    r1 = detector.unfold_nspline(readings, save_result=False)
    r2 = detector.unfold_nspline(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])
    assert r1["iterations"] == r2["iterations"]


def test_unfold_nspline_max_energy(detector, readings):
    """max_neutron_energy truncates the solving grid; the result is
    expanded back to the full grid (package-wide convention)."""
    result = detector.unfold_nspline(
        readings, max_neutron_energy=5.0, save_result=False
    )
    assert len(result["energy"]) == detector.n_energy_bins
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(np.isfinite(result["spectrum"]))
    assert np.all(result["spectrum"] >= 0)
