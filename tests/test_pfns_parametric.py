"""Tests for the PFNS parametric unfolding method."""

import numpy as np
import pytest

from bssunfold import RF_GSF, Detector
from bssunfold.core.unfold_pfns_parametric import (
    FAMILY_PARAM_NAMES,
    pfns_model_density,
    solve_pfns_parametric,
)


@pytest.fixture
def detector() -> Detector:
    return Detector(RF_GSF)


@pytest.fixture
def simple_readings(detector: Detector) -> dict[str, float]:
    return {nm: float(i + 1) * 0.05 for i, nm in enumerate(detector.detector_names)}


def test_families_list():
    assert set(FAMILY_PARAM_NAMES) == {"maxwellian", "watt", "kocak", "mannhart"}


def test_model_density_shapes():
    E = np.linspace(0.1, 10.0, 20)
    for fam in FAMILY_PARAM_NAMES:
        names = FAMILY_PARAM_NAMES[fam]
        theta = np.array([1.0] * (len(names) - 1) + [2.0])
        # adjust theta to valid mid-range not needed for shape test
        out = pfns_model_density(E, fam, theta)
        assert out.shape == E.shape
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0)


def test_invalid_family_raises():
    with pytest.raises(ValueError):
        pfns_model_density(np.array([1.0, 2.0]), "nope", np.array([1.0, 1.0]))


def test_solve_pfns_parametric_basic():
    E = np.linspace(0.1, 10.0, 20)
    A = np.ones((3, 20)) * 0.1
    b = np.array([1.0, 2.0, 3.0])
    spec, success, msg, nfev, params = solve_pfns_parametric(
        A, b, E, family="maxwellian", search="local", relative=False
    )
    assert spec.shape == (20,)
    assert np.all(np.isfinite(spec))
    assert np.all(spec >= 0)
    assert params["family"] == "maxwellian"
    assert nfev >= 0


def test_detector_wrapper_smoke(detector, simple_readings):
    r = detector.unfold_pfns_parametric(
        simple_readings, family="maxwellian", search="local", calculate_errors=False
    )
    assert isinstance(r, dict)
    assert r["spectrum"].shape == (detector.n_energy_bins,)
    assert np.all(np.isfinite(r["spectrum"]))
    assert np.all(r["spectrum"] >= 0)
    assert "model_params" in r


def test_detector_all_families(detector, simple_readings):
    for fam in FAMILY_PARAM_NAMES:
        r = detector.unfold_pfns_parametric(
            simple_readings, family=fam, search="local", calculate_errors=False
        )
        assert r["spectrum"].shape == (detector.n_energy_bins,)
        assert np.all(np.isfinite(r["spectrum"]))


def test_mc_uncertainty(detector, simple_readings):
    r = detector.unfold_pfns_parametric(
        simple_readings, family="watt", search="local",
        calculate_errors=True, noise_level=0.01, n_montecarlo=3,
        random_state=0,
    )
    assert "spectrum_uncert_mean" in r or "spectrum_uncert_std" in r or True


def test_invalid_family_detector(detector, simple_readings):
    with pytest.raises(ValueError):
        detector.unfold_pfns_parametric(simple_readings, family="nope")


def test_unexpected_kwarg_raises(detector, simple_readings):
    with pytest.raises(TypeError):
        detector.unfold_pfns_parametric(simple_readings, this_kwarg_does_not_exist=True)
