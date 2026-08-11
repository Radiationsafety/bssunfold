"""Tests for the mystic-based unfolding method.

The ``mystic`` package is an optional backend installed in the dev group.
These tests cover the ``solve_mystic`` core solver and the ``unfold_mystic``
wrapper exposed both on the ``Detector`` class and as a module-level function.
"""

import builtins
import numpy as np
import pytest
from unittest.mock import patch

from bssunfold import Detector


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def initial(detector):
    return np.ones(detector.n_energy_bins) * 0.5


def test_unfold_mystic_basic(detector, readings):
    """Basic unfold_mystic call returns a standardized result."""
    result = detector.unfold_mystic(
        readings, regularization=1e-3, save_result=False
    )

    assert isinstance(result, dict)
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "method" in result
    assert result["method"] == "mystic_fmin_powell"

    assert isinstance(result["energy"], np.ndarray)
    assert isinstance(result["spectrum"], np.ndarray)
    assert isinstance(result["residual_norm"], float)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)

    assert result["solver"] == "fmin_powell"
    assert result["norm"] == 2
    assert result["selected_regularization"] == pytest.approx(1e-3)


def test_unfold_mystic_with_solver(detector, readings):
    """Explicit mystic solver selection is honored."""
    result = detector.unfold_mystic(
        readings, solver="fmin", maxiter=300, save_result=False
    )
    assert result["method"] == "mystic_fmin"
    assert result["solver"] == "fmin"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_mystic_diffev(detector, readings):
    """Population-based diffev solver works with derived bounds."""
    result = detector.unfold_mystic(
        readings, solver="diffev", maxiter=50, maxfun=3000, save_result=False
    )
    assert result["method"] == "mystic_diffev"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_mystic_unknown_solver(detector, readings):
    """Unknown solver falls back to fmin_powell with a warning."""
    with pytest.warns(UserWarning, match="not supported"):
        result = detector.unfold_mystic(
            readings, solver="bogus", maxiter=200, save_result=False
        )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_mystic_norm1(detector, readings):
    """L1 regularization norm is supported."""
    result = detector.unfold_mystic(
        readings, norm=1, regularization=1e-3, maxiter=300, save_result=False
    )
    assert result["norm"] == 1
    assert np.all(result["spectrum"] >= 0)


def test_unfold_mystic_smoothness(detector, readings):
    """Smoothness constraints of order 1 and 2 are supported."""
    for order in (1, 2):
        result = detector.unfold_mystic(
            readings,
            smoothness_order=order,
            smoothness_weight=0.5,
            maxiter=300,
            save_result=False,
        )
        assert result["smoothness_order"] == order
        assert result["smoothness_weight"] == 0.5
        assert np.all(result["spectrum"] >= 0)


@pytest.mark.parametrize(
    "method",
    ["manual", "cosine", "lcurve", "gcv", "dp"],
)
def test_unfold_mystic_reg_methods(detector, readings, initial, method):
    """All supported regularization selection methods run."""
    kw = dict(
        regularization_method=method,
        maxiter=300,
        save_result=False,
    )
    if method == "cosine":
        kw["initial_spectrum"] = initial
    if method == "dp":
        kw["noise_var"] = 0.01
    result = detector.unfold_mystic(readings, **kw)
    assert "spectrum" in result
    assert result["regularization_method"] == method
    assert result["selected_regularization"] > 0


def test_unfold_mystic_cosine_no_initial(detector, readings):
    """Cosine selection without initial spectrum raises ValueError."""
    with pytest.raises(ValueError, match="initial_spectrum must be provided"):
        detector.unfold_mystic(
            readings, regularization_method="cosine", save_result=False
        )


def test_unfold_mystic_cosine_wrong_length(detector, readings):
    """Cosine selection with mismatched initial length raises ValueError."""
    with pytest.raises(ValueError, match="must match number of energy bins"):
        detector.unfold_mystic(
            readings,
            regularization_method="cosine",
            initial_spectrum=np.ones(5),
            save_result=False,
        )


def test_unfold_mystic_cosine_wrong_norm(detector, readings, initial):
    """Cosine selection warns when norm != 2."""
    with pytest.warns(UserWarning, match="assumes L2"):
        result = detector.unfold_mystic(
            readings,
            norm=1,
            regularization_method="cosine",
            initial_spectrum=initial,
            maxiter=300,
            save_result=False,
        )
    assert "spectrum" in result


def test_unfold_mystic_auto_with_norm1(detector, readings):
    """Automatic regularization selection warns when norm != 2."""
    with pytest.warns(UserWarning, match="assume L2"):
        result = detector.unfold_mystic(
            readings, norm=1, regularization_method="lcurve", save_result=False
        )
    assert "spectrum" in result


def test_solve_mystic_bad_norm_returns_zero(detector, readings):
    """Unsupported norm is caught and returns a zero spectrum with warning."""
    from bssunfold.core.unfold_mystic import solve_mystic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    with pytest.warns(UserWarning, match="failed"):
        spectrum = solve_mystic(A, b, alpha=1e-3, norm=3, maxiter=200)
    assert np.all(spectrum == 0)


def test_unfold_mystic_reg_selection_failure(detector, readings):
    """Regularization selection failure raises ValueError."""
    with patch(
        "bssunfold.core.unfold_mystic.select_regularization_parameter",
        side_effect=Exception("test error"),
    ):
        with pytest.raises(ValueError, match="Regularization selection failed"):
            detector.unfold_mystic(
                readings, regularization_method="lcurve", save_result=False
            )


def test_unfold_mystic_with_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_mystic(
        readings,
        regularization=1e-3,
        calculate_errors=True,
        n_montecarlo=3,
        maxiter=200,
        save_result=False,
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert "spectrum_uncert_min" in result
    assert "spectrum_uncert_max" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_unfold_mystic_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_mystic(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "mystic_fmin_powell"


def test_solve_mystic_returns_array(detector, readings):
    """The core solve_mystic solver returns a raw spectrum array."""
    from bssunfold.core.unfold_mystic import solve_mystic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    spectrum = solve_mystic(A, b, alpha=1e-3, maxiter=300)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (detector.n_energy_bins,)
    assert np.all(spectrum >= -1e-6)


def test_solve_mystic_import_error():
    """Missing mystic raises a helpful ImportError."""
    from bssunfold.core.unfold_mystic import solve_mystic

    A = np.eye(3)
    b = np.ones(3)

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mystic" or name.startswith("mystic."):
            raise ImportError("mystic not installed")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="mystic is required"):
            solve_mystic(A, b, alpha=1e-3)


def test_solve_mystic_failure_returns_zero(detector, readings):
    """Solver exceptions are caught and return a zero spectrum with warning."""
    from bssunfold.core.unfold_mystic import solve_mystic

    selected = [name for name in detector.detector_names if name in readings]
    A = np.array([detector.sensitivities[name] for name in selected])
    b = np.array([readings[name] for name in selected], dtype=float)

    with patch(
        "bssunfold.core.unfold_mystic._solver_function",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.warns(UserWarning, match="failed"):
            spectrum = solve_mystic(A, b, alpha=1e-3)
    assert np.all(spectrum == 0)


def test_core_exports():
    """solve_mystic and unfold_mystic are exported from bssunfold.core."""
    from bssunfold.core import solve_mystic, unfold_mystic

    assert solve_mystic is not None
    assert unfold_mystic is not None


def test_unfold_combined_mystic(detector, readings):
    """'mystic' can be used in a combined unfolding pipeline."""
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
                "method": "mystic",
                "params": {"maxiter": 200, "save_result": False},
            }
        ],
        verbose=False,
    )
    assert "spectrum" in result
    assert np.all(result["spectrum"] >= 0)
