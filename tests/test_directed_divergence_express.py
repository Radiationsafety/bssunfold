"""Tests for directed-divergence and Express unfolding methods.

These tests cover the ``unfold_directed_divergence`` and ``unfold_express``
wrappers (Detector methods and module functions) and the core
``solve_directed_divergence`` and ``solve_express`` solvers.
"""

import numpy as np
import pytest

from bssunfold import Detector


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


# ------------------------------------------------------------------ #
#  Directed divergence — core solver
# ------------------------------------------------------------------ #

def test_solve_directed_divergence_core(A, b):
    """The core solver returns a non-negative spectrum."""
    from bssunfold.core.unfold_directed_divergence import solve_directed_divergence

    spectrum, iterations, converged = solve_directed_divergence(
        A, b, x0=np.ones(A.shape[1])
    )
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int)
    assert isinstance(converged, bool)
    resid = A @ spectrum - b
    assert np.linalg.norm(resid) < np.linalg.norm(b)


def test_solve_directed_divergence_zero_readings(A, b):
    """Zero readings are valid input (no raise); negative readings raise."""
    from bssunfold.core.unfold_directed_divergence import solve_directed_divergence

    spectrum, _, _ = solve_directed_divergence(
        A, np.zeros_like(b), np.ones(A.shape[1])
    )
    assert np.all(np.isfinite(spectrum))


def test_solve_directed_divergence_negative_readings(A, b):
    """Negative readings are rejected with a helpful error."""
    from bssunfold.core.unfold_directed_divergence import solve_directed_divergence

    bad = b.copy()
    bad[0] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        solve_directed_divergence(A, bad, np.ones(A.shape[1]))


def test_solve_directed_divergence_smoothness(A, b):
    """The smoothness option produces finite non-negative results."""
    from bssunfold.core.unfold_directed_divergence import solve_directed_divergence

    spectrum, _, _ = solve_directed_divergence(
        A, b, np.ones(A.shape[1]),
        max_iterations=5,
        smoothness_order=2,
        smoothness_weight=0.1,
    )
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0)


# ------------------------------------------------------------------ #
#  Directed divergence — Detector wrapper
# ------------------------------------------------------------------ #

def test_unfold_directed_divergence_basic(detector, readings):
    """Basic unfold_directed_divergence returns a standardized result dict."""
    result = detector.unfold_directed_divergence(readings, save_result=False)

    assert isinstance(result, dict)
    assert result["method"] == "Directed divergence"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)


def test_unfold_directed_divergence_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_directed_divergence(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_directed_divergence_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    from bssunfold.core.unfold_directed_divergence import unfold_directed_divergence

    res_det = detector.unfold_directed_divergence(readings, save_result=False)
    res_fn = unfold_directed_divergence(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "Directed divergence"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_unfold_directed_divergence_deterministic(detector, readings):
    """Same inputs reproduce the same spectrum."""
    r1 = detector.unfold_directed_divergence(readings, save_result=False)
    r2 = detector.unfold_directed_divergence(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_directed_divergence_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_directed_divergence(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "Directed divergence"


def test_unfold_directed_divergence_calculate_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_directed_divergence(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_directed_divergence_max_neutron_energy(detector, readings):
    """max_neutron_energy is accepted and _expand_result restores full length."""
    result = detector.unfold_directed_divergence(
        readings, max_neutron_energy=1.0, save_result=False
    )
    assert len(result["spectrum"]) == detector.n_energy_bins


def test_unfold_directed_divergence_smoothness_option(detector, readings):
    """Smoothness_order and smoothness_weight are accepted."""
    result = detector.unfold_directed_divergence(
        readings,
        smoothness_order=2,
        smoothness_weight=0.05,
        save_result=False,
    )
    assert np.all(result["spectrum"] >= 0)


# ------------------------------------------------------------------ #
#  Express — core solver
# ------------------------------------------------------------------ #

def test_solve_express_core(A, b, detector):
    """The Express solver fits a piecewise-exponential model."""
    from bssunfold.core.unfold_express import solve_express

    spectrum, iterations, converged = solve_express(
        A, b, detector.E_MeV, n_groups=5, max_iterations=5
    )
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum > 0)
    assert iterations > 0
    assert isinstance(converged, bool)


def test_solve_express_explicit_boundaries(A, b, detector):
    """Explicit interval_boundaries are accepted."""
    from bssunfold.core.unfold_express import solve_express

    boundaries = np.linspace(float(detector.E_MeV[0]), float(detector.E_MeV[-1]), 5)
    spectrum, _, _ = solve_express(
        A, b, detector.E_MeV, interval_boundaries=boundaries
    )
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum > 0)


def test_solve_express_negative_readings(A, b, detector):
    """Negative readings are rejected with a helpful error."""
    from bssunfold.core.unfold_express import solve_express

    bad = b.copy()
    bad[0] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        solve_express(A, bad, detector.E_MeV)


# ------------------------------------------------------------------ #
#  Express — Detector wrapper
# ------------------------------------------------------------------ #

def test_unfold_express_basic(detector, readings):
    """Basic unfold_express returns a standardized result dict."""
    result = detector.unfold_express(readings, save_result=False)

    assert isinstance(result, dict)
    assert result["method"] == "Express"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] > 0)


def test_unfold_express_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_express(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] > 0)


def test_unfold_express_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    from bssunfold.core.unfold_express import unfold_express

    res_det = detector.unfold_express(readings, save_result=False)
    res_fn = unfold_express(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "Express"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_unfold_express_deterministic(detector, readings):
    """Same inputs reproduce the same spectrum."""
    r1 = detector.unfold_express(readings, save_result=False)
    r2 = detector.unfold_express(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_express_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_express(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "Express"


def test_unfold_express_calculate_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_express(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_express_max_neutron_energy(detector, readings):
    """max_neutron_energy is accepted and _expand_result restores full length."""
    result = detector.unfold_express(
        readings, max_neutron_energy=1.0, save_result=False
    )
    assert len(result["spectrum"]) == detector.n_energy_bins


def test_unfold_express_n_groups(detector, readings):
    """Custom n_groups is accepted."""
    result = detector.unfold_express(
        readings, n_groups=4, save_result=False
    )
    assert np.all(result["spectrum"] > 0)


# ------------------------------------------------------------------ #
#  Core exports
# ------------------------------------------------------------------ #

def test_core_exports():
    """All new solvers and wrappers are exported from bssunfold.core."""
    from bssunfold.core import (
        solve_directed_divergence,
        solve_express,
        unfold_directed_divergence,
        unfold_express,
    )

    assert solve_directed_divergence is not None
    assert unfold_directed_divergence is not None
    assert solve_express is not None
    assert unfold_express is not None
