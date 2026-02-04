import numpy as np
import pandas as pd
import pytest

from src.bssunfold import Detector


@pytest.fixture
def sample_response_df():
    E_MeV = np.logspace(-9, 3, 60)
    data = {"E_MeV": E_MeV}
    for sphere in ["0in", "2in", "3in", "5in", "8in"]:
        data[sphere] = np.random.rand(len(E_MeV)) * 0.1 + 0.5
    return pd.DataFrame(data)


@pytest.fixture
def detector(sample_response_df):
    return Detector(sample_response_df)


@pytest.fixture
def sample_readings():
    return {
        "0in": 0.00037707623092440032,
        "2in": 0.0099964357249166195,
        "3in": 0.053668754395163297,
        "5in": 0.18417232269591507,
        "8in": 0.22007281510471705,
    }


def test_unfold_maxed_basic(detector, sample_readings):
    result = detector.unfold_maxed(sample_readings, maxiter=200)
    assert result["method"] == "MAXED"
    assert result["spectrum"].shape[0] == detector.n_energy_bins


def test_unfold_gravel_basic(detector, sample_readings):
    result = detector.unfold_gravel(
        sample_readings, max_iterations=100, tolerance=1e-5
    )
    assert result["method"] == "GRAVEL"
    assert result["spectrum"].shape[0] == detector.n_energy_bins


def test_unfold_doroshenko_basic(detector, sample_readings):
    result = detector.unfold_doroshenko(sample_readings, max_iterations=50)
    assert result["method"] == "Doroshenko"
    assert result["spectrum"].shape[0] == detector.n_energy_bins


def test_unfold_doroshenko_matrix_basic(detector, sample_readings):
    result = detector.unfold_doroshenko_matrix(sample_readings, max_iterations=50)
    assert result["method"] == "Doroshenko-matrix"
    assert result["spectrum"].shape[0] == detector.n_energy_bins


def test_unfold_tikhonov_legendre_optional(detector, sample_readings):
    pytest.importorskip("spectrum_recovery")
    result = detector.unfold_tikhonov_legendre(sample_readings)
    assert result["method"] == "Tikhonov-Legendre"


def test_unfold_bayes_optional(detector, sample_readings):
    pytest.importorskip("pyunfold")
    pytest.importorskip("pyunfold.callbacks")
    result = detector.unfold_bayes(sample_readings, max_iterations=20)
    assert result["method"] == "Bayes (D'Agostini)"
