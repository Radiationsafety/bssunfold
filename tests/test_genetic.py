"""Tests for the genetic/meta-heuristic unfolding method.

The ``mealpy`` package is an optional backend installed in the dev group.
These tests cover the ``solve_genetic`` core solver and the
``unfold_genetic`` wrapper exposed both on the ``Detector`` class and as a
module-level function, including all supported meta-heuristic solvers,
regularization variants and error handling.
"""

import builtins
import numpy as np
import pytest
from unittest.mock import patch

from bssunfold import Detector

SOLVERS = ["pso", "ga", "de", "es", "ep", "abc", "gwo", "cmaes"]


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def initial(detector):
    return np.ones(detector.n_energy_bins) * 0.5


@pytest.fixture(params=SOLVERS)
def solver(request):
    return request.param


def test_unfold_genetic_basic(detector, readings):
    """Basic unfold_genetic call returns a standardized result."""
    result = detector.unfold_genetic(
        readings, epoch=40, pop_size=20, save_result=False
    )

    assert isinstance(result, dict)
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "method" in result
    assert result["method"] == "genetic_pso"

    assert isinstance(result["energy"], np.ndarray)
    assert isinstance(result["spectrum"], np.ndarray)
    assert isinstance(result["residual_norm"], float)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)

    assert result["solver"] == "pso"
    assert result["norm"] == 2
    assert result["regularization"] == pytest.approx(1e-4)
    assert result["epoch"] == 40
    assert result["pop_size"] == 20


def test_unfold_genetic_all_solvers(detector, readings, solver):
    """Every supported meta-heuristic solver runs and honors selection."""
    result = detector.unfold_genetic(
        readings, solver=solver, epoch=30, pop_size=20, save_result=False
    )
    assert result["method"] == f"genetic_{solver}"
    assert result["solver"] == solver
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_alias(detector, readings):
    """Long-form solver aliases resolve to canonical names."""
    result = detector.unfold_genetic(
        readings, solver="genetic_algorithm", epoch=30, pop_size=20,
        save_result=False,
    )
    assert result["method"] == "genetic_ga"
    assert result["solver"] == "ga"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_unknown_solver(detector, readings):
    """Unknown solver falls back to pso with a warning."""
    with pytest.warns(UserWarning, match="not supported"):
        result = detector.unfold_genetic(
            readings, solver="bogus", epoch=30, pop_size=20, save_result=False
        )
    assert result["method"] == "genetic_pso"
    assert result["solver"] == "pso"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_norm1(detector, readings):
    """L1 regularization norm is supported."""
    result = detector.unfold_genetic(
        readings, norm=1, regularization=1e-3, epoch=40, pop_size=20,
        save_result=False,
    )
    assert result["norm"] == 1
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_smoothness(detector, readings):
    """Smoothness constraints of order 1 and 2 are supported."""
    for order in (1, 2):
        result = detector.unfold_genetic(
            readings,
            smoothness_order=order,
            smoothness_weight=0.5,
            epoch=40,
            pop_size=20,
            save_result=False,
        )
        assert result["smoothness_order"] == order
        assert result["smoothness_weight"] == 0.5
        assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_entropy(detector, readings):
    """The Shannon-entropy objective can be enabled."""
    result = detector.unfold_genetic(
        readings,
        entropy_weight=1e-3,
        epoch=40,
        pop_size=20,
        save_result=False,
    )
    assert result["entropy_weight"] == pytest.approx(1e-3)
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_n_runs(detector, readings):
    """Multiple runs are averaged and reported in metadata."""
    result = detector.unfold_genetic(
        readings, n_runs=2, epoch=40, pop_size=20, save_result=False
    )
    assert result["n_runs"] == 2
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_early_stop(detector, readings):
    """Early stopping is accepted and stored in metadata."""
    result = detector.unfold_genetic(
        readings, early_stop=10, epoch=40, pop_size=20, save_result=False
    )
    assert result["early_stop"] == 10
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_initial_spectrum(detector, readings, initial):
    """A provided initial spectrum is accepted and used."""
    result = detector.unfold_genetic(
        readings,
        initial_spectrum=initial,
        epoch=40,
        pop_size=20,
        save_result=False,
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_genetic_wrong_initial_length(detector, readings):
    """Initial spectrum with a wrong length raises ValueError."""
    with pytest.raises(ValueError, match="must match number of energy bins"):
        detector.unfold_genetic(
            readings, initial_spectrum=np.ones(5), save_result=False
        )


def test_unfold_genetic_deterministic(detector, readings):
    """Same random_state reproduces the same spectrum."""
    r1 = detector.unfold_genetic(
        readings, epoch=30, pop_size=20, random_state=42, save_result=False
    )
    r2 = detector.unfold_genetic(
        readings, epoch=30, pop_size=20, random_state=42, save_result=False
    )
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_genetic_with_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_genetic(
        readings,
        regularization=1e-3,
        calculate_errors=True,
        n_montecarlo=3,
        epoch=30,
        pop_size=20,
        save_result=False,
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert "spectrum_uncert_min" in result
    assert "spectrum_uncert_max" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_genetic_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_genetic(readings, epoch=30, pop_size=20, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "genetic_pso"


def test_solve_genetic_bad_norm_returns_zero(detector, readings):
    """Unsupported norm is caught and returns a zero spectrum with warning."""
    from bssunfold.core.unfold_genetic import solve_genetic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    with pytest.warns(UserWarning, match="failed"):
        spectrum = solve_genetic(A, b, regularization=1e-3, norm=3)
    assert np.all(spectrum == 0)


def test_solve_genetic_bad_smoothness_returns_zero(detector, readings):
    """Unsupported smoothness order is caught and returns zero spectrum."""
    from bssunfold.core.unfold_genetic import solve_genetic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    with pytest.warns(UserWarning, match="failed"):
        spectrum = solve_genetic(A, b, smoothness_order=5)
    assert np.all(spectrum == 0)


def test_solve_genetic_import_error():
    """Missing mealpy raises a helpful ImportError."""
    from bssunfold.core.unfold_genetic import solve_genetic

    A = np.eye(3)
    b = np.ones(3)

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mealpy" or name.startswith("mealpy."):
            raise ImportError("mealpy not installed")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="mealpy is required"):
            solve_genetic(A, b)


def test_solve_genetic_failure_returns_zero(detector, readings):
    """Solver exceptions are caught and return a zero spectrum with warning."""
    from bssunfold.core.unfold_genetic import solve_genetic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    with patch(
        "bssunfold.core.unfold_genetic._solve_genetic_impl",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.warns(UserWarning, match="failed"):
            spectrum = solve_genetic(A, b)
    assert np.all(spectrum == 0)


def test_solve_genetic_returns_array(detector, readings):
    """The core solve_genetic solver returns a raw spectrum array."""
    from bssunfold.core.unfold_genetic import solve_genetic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    spectrum = solve_genetic(A, b, epoch=40, pop_size=20)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (detector.n_energy_bins,)
    assert np.all(spectrum >= 0)


def test_solve_genetic_zero_readings(detector, readings):
    """Zero readings are handled via the denominator fallback."""
    from bssunfold.core.unfold_genetic import solve_genetic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.zeros(len(selected))

    spectrum = solve_genetic(A, b, epoch=30, pop_size=20)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (detector.n_energy_bins,)
    assert np.all(spectrum >= 0)


def test_build_model_unknown_solver():
    """_build_model rejects unsupported solver keys defensively."""
    from bssunfold.core.unfold_genetic import _build_model, _import_mealpy

    mealpy = _import_mealpy()
    with pytest.raises(ValueError, match="Unsupported solver"):
        _build_model(mealpy, "bogus", epoch=10, pop_size=20)


def test_core_exports():
    """solve_genetic and unfold_genetic are exported from bssunfold.core."""
    from bssunfold.core import solve_genetic, unfold_genetic

    assert solve_genetic is not None
    assert unfold_genetic is not None


def test_unfold_combined_genetic(detector, readings):
    """'genetic' can be used in a combined unfolding pipeline."""
    from bssunfold.core.unfold_combined import unfold_combined

    result = unfold_combined(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        pipeline=[
            {
                "method": "genetic",
                "params": {
                    "epoch": 30,
                    "pop_size": 20,
                    "save_result": False,
                },
            }
        ],
        verbose=False,
    )
    assert "spectrum" in result
    assert np.all(result["spectrum"] >= 0)
