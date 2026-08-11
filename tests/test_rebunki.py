"""Tests for the ReBUNKI iterative unfolding method.

ReBUNKI (Realistic BONner sphere UNfolding in KInteractive mode) is a
few-iteration spectral-stripping technique with limited data-augmentation.
These tests cover the ``unfold_rebunki`` wrapper (Detector method and module
function) and the core ``solve_rebunki`` solver.
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


def test_unfold_rebunki_basic(detector, readings):
    """Basic unfold_rebunki returns a standardized result dict."""
    result = detector.unfold_rebunki(readings, save_result=False)

    assert isinstance(result, dict)
    assert result["method"] == "ReBUNKI"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)


def test_unfold_rebunki_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_rebunki(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_rebunki_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    from bssunfold.core.unfold_rebunki import unfold_rebunki

    res_det = detector.unfold_rebunki(readings, save_result=False)
    res_fn = unfold_rebunki(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "ReBUNKI"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_solve_rebunki_core(A, b):
    """The core solver returns a spectrum close to the measurements."""
    from bssunfold.core.unfold_rebunki import solve_rebunki

    spectrum, iterations, converged = solve_rebunki(
        A, b, x0=np.ones(A.shape[1])
    )
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int)
    assert isinstance(converged, bool)
    resid = A @ spectrum - b
    assert np.linalg.norm(resid) < np.linalg.norm(b)


def test_solve_rebunki_zero_readings(A, b):
    """All-zero readings are rejected with a helpful error."""
    from bssunfold.core.unfold_rebunki import solve_rebunki

    with pytest.raises(ValueError, match="positive measurements"):
        solve_rebunki(A, np.zeros_like(b), x0=np.ones(A.shape[1]))


def test_solve_rebunki_iterations(A, b):
    """The iteration count is honored by the core solver."""
    from bssunfold.core.unfold_rebunki import solve_rebunki

    spectrum, _, _ = solve_rebunki(
        A, b, x0=np.ones(A.shape[1]), max_iterations=5
    )
    assert spectrum.shape == (A.shape[1],)


def test_unfold_rebunki_deterministic(detector, readings):
    """Same inputs reproduce the same spectrum."""
    r1 = detector.unfold_rebunki(readings, save_result=False)
    r2 = detector.unfold_rebunki(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_rebunki_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_rebunki(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "ReBUNKI"


def test_unfold_rebunki_calculate_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_rebunki(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_solve_rebunki_negative_readings(A, b):
    """Negative readings are rejected with a helpful error."""
    from bssunfold.core.unfold_rebunki import solve_rebunki

    bad = b.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        solve_rebunki(A, bad, x0=np.ones(A.shape[1]))


def test_core_exports():
    """solve_rebunki and unfold_rebunki are exported from bssunfold.core."""
    from bssunfold.core import solve_rebunki, unfold_rebunki

    assert solve_rebunki is not None
    assert unfold_rebunki is not None
