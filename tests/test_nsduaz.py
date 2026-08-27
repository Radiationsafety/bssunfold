"""Tests for the NSDUAZ unfolding method.

NSDUAZ (Neutron Spectra Determination Using Arbitrary number of zenith
angles) is a Bonner-sphere-spectrometer unfolding scheme that constructs a
library of trial spectra from nuclear-data reference fluxes (maxwellian,
fission, 1/E, etc.) and blends them via a Bayesian weighting. These tests
cover the ``unfold_nsduaz`` wrapper (Detector method and module function)
and the core ``solve_nsduaz`` solver.
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


def test_unfold_nsduaz_basic(detector, readings):
    """Basic unfold_nsduaz returns a standardized result dict."""
    result = detector.unfold_nsduaz(readings, save_result=False)

    assert isinstance(result, dict)
    assert result["method"] == "NSDUAZ"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert "catalogue" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)


def test_unfold_nsduaz_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_nsduaz(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_nsduaz_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    from bssunfold.core.unfold_nsduaz import unfold_nsduaz

    res_det = detector.unfold_nsduaz(readings, save_result=False)
    res_fn = unfold_nsduaz(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "NSDUAZ"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_solve_nsduaz_core(A, b):
    """The core solver returns a spectrum close to the measurements."""
    from bssunfold.core.unfold_nsduaz import solve_nsduaz

    spectrum, iterations, converged = solve_nsduaz(A, b, x0=np.ones(A.shape[1]))
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int)
    assert isinstance(converged, bool)
    resid = A @ spectrum - b
    assert np.linalg.norm(resid) < np.linalg.norm(b)


def test_solve_nsduaz_zero_readings(A, b):
    """All-zero readings are rejected with a helpful error."""
    from bssunfold.core.unfold_nsduaz import solve_nsduaz

    with pytest.raises(ValueError, match="positive measurements"):
        solve_nsduaz(A, np.zeros_like(b), x0=np.ones(A.shape[1]))


def test_solve_nsduaz_catalogue(A, b):
    """The core solver runs with a custom smoothing parameter."""
    from bssunfold.core.unfold_nsduaz import solve_nsduaz

    spectrum, _, _ = solve_nsduaz(A, b, x0=np.ones(A.shape[1]), smoothing=0.5)
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)


def test_unfold_nsduaz_deterministic(detector, readings):
    """Same inputs reproduce the same spectrum."""
    r1 = detector.unfold_nsduaz(readings, save_result=False)
    r2 = detector.unfold_nsduaz(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_nsduaz_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_nsduaz(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "NSDUAZ"


def test_unfold_nsduaz_calculate_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_nsduaz(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_solve_nsduaz_negative_readings(A, b):
    """Negative readings are rejected with a helpful error."""
    from bssunfold.core.unfold_nsduaz import solve_nsduaz

    bad = b.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        solve_nsduaz(A, bad, x0=np.ones(A.shape[1]))


def test_builtin_catalogue(detector):
    """The built-in catalogue has the expected shapes and positive fluxes."""
    from bssunfold.core.unfold_nsduaz import builtin_catalogue

    cat = builtin_catalogue(detector.E_MeV)
    assert isinstance(cat, dict)
    assert len(cat) > 0
    for label, spec in cat.items():
        assert spec.shape == (detector.n_energy_bins,)
        assert np.all(spec > 0)
        assert np.all(np.isfinite(spec))


def test_find_reference_index_fallback():
    """Names without a 20.32 cm sphere fall back to the largest response."""
    from bssunfold.core.unfold_nsduaz import _find_reference_index

    A = np.array([[1.0, 0.5], [2.0, 0.5]])
    idx = _find_reference_index(["3in", "18in"], A)
    assert idx == 1


def test_find_reference_index_20in():
    """The '20in' convention is recognised."""
    from bssunfold.core.unfold_nsduaz import _find_reference_index

    A = np.eye(3)
    idx = _find_reference_index(["3in", "20in", "18in"], A)
    assert idx == 1


def test_select_catalogue_initial(detector, readings):
    """select_catalogue_initial picks a usable initial spectrum."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    spec, label = select_catalogue_initial(
        readings,
        detector.detector_names,
        detector.sensitivities,
        E_MeV=detector.E_MeV,
    )
    assert spec.shape == (detector.n_energy_bins,)
    assert np.all(spec >= 0)
    assert isinstance(label, str) and label


def test_select_catalogue_initial_no_readings(detector):
    """No readings raises a helpful error."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    with pytest.raises(ValueError, match="No detector readings"):
        select_catalogue_initial(
            {}, detector.detector_names, detector.sensitivities
        )


def test_select_catalogue_initial_bad_reference(detector, readings):
    """A reference sphere missing from the readings is rejected."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    with pytest.raises(ValueError, match="reference_name"):
        select_catalogue_initial(
            readings,
            detector.detector_names,
            detector.sensitivities,
            reference_name="bogus",
        )


def test_select_catalogue_initial_custom_catalogue(detector, readings):
    """A user-supplied catalogue is used for the selection."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    custom = {
        "flat": np.ones(detector.n_energy_bins),
        "ramp": np.linspace(0.5, 1.5, detector.n_energy_bins),
    }
    spec, label = select_catalogue_initial(
        readings,
        detector.detector_names,
        detector.sensitivities,
        catalogue=custom,
    )
    assert spec.shape == (detector.n_energy_bins,)
    assert label in custom


def test_select_catalogue_initial_bad_shape(detector, readings):
    """A catalogue spectrum with the wrong length is rejected."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    with pytest.raises(ValueError, match="has shape"):
        select_catalogue_initial(
            readings,
            detector.detector_names,
            detector.sensitivities,
            catalogue={"bad": np.ones(5)},
        )


def test_select_catalogue_initial_empty(detector, readings):
    """An unusable catalogue raises a helpful error."""
    from bssunfold.core.unfold_nsduaz import select_catalogue_initial

    with pytest.raises(ValueError, match="Catalogue is empty"):
        select_catalogue_initial(
            readings,
            detector.detector_names,
            detector.sensitivities,
            catalogue={"zero": np.zeros(detector.n_energy_bins)},
        )


def test_unfold_nsduaz_custom_catalogue(detector, readings):
    """A user-supplied catalogue flows through the Detector wrapper."""
    custom = {"flat": np.ones(detector.n_energy_bins)}
    result = detector.unfold_nsduaz(
        readings, catalogue=custom, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_nsduaz_flat_mode(detector, readings):
    """use_catalogue=False selects a flat-spectrum initial guess."""
    result = detector.unfold_nsduaz(
        readings, use_catalogue=False, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_core_exports():
    """solve_nsduaz and unfold_nsduaz are exported from bssunfold.core."""
    from bssunfold.core import solve_nsduaz, unfold_nsduaz

    assert solve_nsduaz is not None
    assert unfold_nsduaz is not None
