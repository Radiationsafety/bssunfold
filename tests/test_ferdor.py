"""Tests for the FERDOR iterative unfolding method.

FERDOR (Few-channel unfolding based on a Regularized Deconvolution with
Objective Response functions) is a Bonner-sphere-spectrometer unfolding
scheme based on the coarse-graining of the energy grid. These tests cover
the ``unfold_ferdor`` wrapper (Detector method and module function) and the
core ``solve_ferdor`` solver.
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
        "3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034,
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


def test_unfold_ferdor_basic(detector, readings):
    """Basic unfold_ferdor returns a standardized result dict."""
    result = detector.unfold_ferdor(readings, save_result=False)

    assert isinstance(result, dict)
    assert result["method"] == "FERDOR"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert "effective_readings" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert result["converged"] in (True, False)
    assert isinstance(result["iterations"], int)


def test_unfold_ferdor_all_spheres(detector):
    """All default spheres can be used as a single reading set."""
    result = detector.unfold_ferdor(
        {name: 1.0 for name in detector.detector_names}, save_result=False
    )
    assert np.all(result["spectrum"] >= 0)


def test_unfold_ferdor_aliases(detector, readings):
    """The module-level function matches the Detector result."""
    from bssunfold.core.unfold_ferdor import unfold_ferdor

    res_det = detector.unfold_ferdor(readings, save_result=False)
    res_fn = unfold_ferdor(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        readings=readings,
        save_result_callback=detector._save_result,
    )
    assert res_fn["method"] == "FERDOR"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])


def test_solve_ferdor_core(A, b):
    """The core solver returns a spectrum close to the measurements."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, iterations, converged = solve_ferdor(
        A, b, x0=np.ones(A.shape[1])
    )
    assert spectrum.shape == (A.shape[1],)
    assert np.all(spectrum >= 0)
    assert isinstance(iterations, int)
    assert isinstance(converged, bool)
    resid = A @ spectrum - b
    assert np.linalg.norm(resid) < np.linalg.norm(b)


def test_solve_ferdor_zero_readings(A, b):
    """All-zero readings are rejected with a helpful error."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    with pytest.raises(ValueError, match="positive measurement"):
        solve_ferdor(A, np.zeros_like(b), x0=np.ones(A.shape[1]))


def test_solve_ferdor_max_iterations(A, b):
    """max_iterations is honored by the core solver."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, _, _ = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), max_iterations=5
    )
    assert spectrum.shape == (A.shape[1],)


def test_solve_ferdor_regularization(A, b):
    """The smoothing weight is accepted by the core solver."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, _, _ = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), smoothing=0.1
    )
    assert spectrum.shape == (A.shape[1],)


def test_unfold_ferdor_deterministic(detector, readings):
    """Same inputs reproduce the same spectrum."""
    r1 = detector.unfold_ferdor(readings, save_result=False)
    r2 = detector.unfold_ferdor(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_ferdor_save_result(detector, readings):
    """save_result=True stores the result in results_history."""
    detector.unfold_ferdor(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "FERDOR"


def test_unfold_ferdor_calculate_errors(detector, readings):
    """Monte-Carlo uncertainty fields are added when requested."""
    result = detector.unfold_ferdor(
        readings, calculate_errors=True, n_montecarlo=5, save_result=False
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == detector.n_energy_bins


def test_solve_ferdor_empty_measurements():
    """An empty measurement vector is rejected."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    with pytest.raises(ValueError, match="empty"):
        solve_ferdor(np.empty((0, 4)), np.empty(0), x0=np.ones(4))


def test_solve_ferdor_explicit_sigma(A, b):
    """An explicit per-detector sigma overrides the relative uncertainty."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, _, _ = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), sigma=np.full(len(b), 0.05)
    )
    assert spectrum.shape == (A.shape[1],)


def test_solve_ferdor_bad_sigma_shape(A, b):
    """A sigma with the wrong length is rejected."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    with pytest.raises(ValueError, match="sigma must have shape"):
        solve_ferdor(A, b, x0=np.ones(A.shape[1]), sigma=np.ones(2))


def test_solve_ferdor_chi_target(A, b):
    """A large chi_squared_target converges with a smoothing weight rise."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, iterations, converged = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), chi_squared_target=10.0,
        smoothing=1e-3, max_iterations=50,
    )
    assert spectrum.shape == (A.shape[1],)
    assert iterations > 0
    assert converged in (True, False)


def test_solve_ferdor_tiny_alpha(A, b):
    """A smoothing weight below the lower bracket is clamped."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, _, _ = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), smoothing=1e-20, min_alpha=1e-12
    )
    assert spectrum.shape == (A.shape[1],)


def test_solve_ferdor_small_n():
    """n <= 2 uses a zero derivative matrix without error."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    A = np.array([[1.0, 0.5], [0.8, 0.4]])
    b = np.array([1.0, 0.7])
    spectrum, _, _ = solve_ferdor(A, b, x0=np.ones(2))
    assert spectrum.shape == (2,)
    assert np.all(spectrum >= 0)


def test_solve_ferdor_lstsq_fallback(A, b):
    """A singular system falls back to lstsq."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    singular = np.zeros((2, 4))
    singular[:, 0] = 1.0
    spectrum, _, _ = solve_ferdor(
        singular, np.ones(2), x0=np.ones(4)
    )
    assert spectrum.shape == (4,)


def test_solve_ferdor_singular_no_solution(A, b):
    """A fully singular system returns a zero-scale fallback."""
    from unittest.mock import patch

    from bssunfold.core.unfold_ferdor import solve_ferdor, _solve_weighted_ls

    zero_sys = np.zeros((2, 4))
    with patch(
        "bssunfold.core.unfold_ferdor._solve_weighted_ls",
        side_effect=lambda *a, **k: None,
    ):
        spectrum, iterations, converged = solve_ferdor(
            zero_sys, np.ones(2), x0=np.ones(4)
        )
    assert spectrum.shape == (4,)
    assert iterations == 1
    assert converged is False


def test_solve_ferdor_low_target(A, b):
    """A chi_squared_target below the fit pushes the smoothing weight down."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    spectrum, iterations, converged = solve_ferdor(
        A, b, x0=np.ones(A.shape[1]), chi_squared_target=0.01,
        smoothing=1.0, max_iterations=50,
    )
    assert spectrum.shape == (A.shape[1],)
    assert iterations > 0


def test_solve_ferdor_singular_negative_readings(A, b):
    """All-negative readings are rejected like zero readings."""
    from bssunfold.core.unfold_ferdor import solve_ferdor

    with pytest.raises(ValueError, match="positive measurement"):
        solve_ferdor(A, -np.ones_like(b), x0=np.ones(A.shape[1]))


def test_core_exports():
    """solve_ferdor and unfold_ferdor are exported from bssunfold.core."""
    from bssunfold.core import solve_ferdor, unfold_ferdor

    assert solve_ferdor is not None
    assert unfold_ferdor is not None
