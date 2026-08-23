"""Tests for the SCIP-based unfolding method.

The ``pyscipopt`` package (SCIP Optimization Suite interface) is an optional
backend installed in the dev group. These tests cover the ``solve_scip`` core
solver and the ``unfold_scip`` wrapper exposed both on the ``Detector`` class
and as a module-level function.
"""

import builtins
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("pyscipopt")

from bssunfold import Detector  # noqa: E402


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def initial(detector):
    return np.ones(detector.n_energy_bins)


def _mock_no_module(name):
    """Build a builtins.__import__ patch that blocks ``name`` imports."""
    original_import = builtins.__import__

    def mock(_name, *args, **kwargs):
        if _name == name or _name.startswith(name + "."):
            raise ImportError(f"No module named '{name}'")
        return original_import(_name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=mock)


class TestSolveScip:
    def test_solves_l2(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = A @ np.array([1.5, 2.0])
        x = solve_scip(A, b, alpha=0.0, timeout=10.0)
        assert x is not None
        assert x.shape == (2,)
        assert np.all(x >= -1e-6)
        assert np.allclose(A @ x, b, atol=1e-2)

    def test_l1_norm(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_scip(A, b, alpha=0.1, norm=1, timeout=10.0)
        assert x is not None
        assert x.shape == (2,)

    def test_smoothness_order_1(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_scip(A, b, alpha=1e-3, smoothness_order=1, timeout=10.0)
        assert x is not None
        assert x.shape == (3,)

    def test_smoothness_order_2(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_scip(A, b, alpha=1e-3, smoothness_order=2, timeout=10.0)
        assert x is not None

    def test_nonneg_false(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        x = solve_scip(A, b, nonneg=False, timeout=10.0)
        assert x is not None

    def test_ill_formed_input(self):
        from bssunfold.core.unfold_scip import solve_scip

        with pytest.raises(ValueError, match="ill-formed"):
            solve_scip(np.ones((2, 3)), np.ones(4))

    def test_unsupported_norm(self):
        from bssunfold.core.unfold_scip import solve_scip

        A = np.array([[1.0, 2.0]])
        b = np.array([3.0])
        with pytest.raises(ValueError, match="norm"):
            solve_scip(A, b, norm=3)

    def test_deterministic_with_seed(self):
        from bssunfold.core.unfold_scip import solve_scip

        rng = np.random.default_rng(0)
        A = rng.random((5, 10)) + 0.1
        b = A @ np.abs(rng.normal(size=10)) + 0.01
        x1 = solve_scip(A, b, alpha=1e-3, random_state=42, timeout=10.0)
        x2 = solve_scip(A, b, alpha=1e-3, random_state=42, timeout=10.0)
        assert x1 is not None and x2 is not None
        assert np.allclose(x1, x2)

    def test_import_error_message(self):
        from bssunfold.core.unfold_scip import solve_scip

        with _mock_no_module("pyscipopt"):
            with pytest.raises(ImportError, match="pyscipopt"):
                solve_scip(np.ones((1, 2)), np.ones(1))


class TestUnfoldScip:
    def test_basic(self, detector, readings):
        result = detector.unfold_scip(readings, save_result=False, timeout=10.0)
        assert "spectrum" in result
        assert "energy" in result
        assert result["method"] == "SCIP"
        assert len(result["spectrum"]) == detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert "selected_regularization" in result
        assert result["nonneg"] is True

    def test_norm1(self, detector, readings):
        result = detector.unfold_scip(
            readings, norm=1, save_result=False, timeout=10.0
        )
        assert result["method"] == "SCIP"
        assert result["norm"] == 1

    def test_smoothness(self, detector, readings):
        result = detector.unfold_scip(
            readings,
            smoothness_order=2,
            smoothness_weight=2.0,
            save_result=False,
            timeout=10.0,
        )
        assert result["smoothness_order"] == 2
        assert result["smoothness_weight"] == 2.0

    def test_nonneg_false(self, detector, readings):
        result = detector.unfold_scip(
            readings, nonneg=False, save_result=False, timeout=10.0
        )
        assert result["nonneg"] is False

    def test_cosine_regularization(self, detector, readings, initial):
        result = detector.unfold_scip(
            readings,
            regularization_method="cosine",
            initial_spectrum=initial,
            save_result=False,
            timeout=10.0,
        )
        assert result["regularization_method"] == "cosine"
        assert result["selected_regularization"] > 0

    def test_cosine_requires_initial(self, detector, readings):
        with pytest.raises(ValueError, match="initial_spectrum"):
            detector.unfold_scip(
                readings, regularization_method="cosine", save_result=False
            )

    def test_lcurve_regularization(self, detector, readings):
        result = detector.unfold_scip(
            readings,
            regularization_method="lcurve",
            save_result=False,
            timeout=10.0,
        )
        assert result["regularization_method"] == "lcurve"

    def test_gcv_regularization(self, detector, readings):
        result = detector.unfold_scip(
            readings,
            regularization_method="gcv",
            save_result=False,
            timeout=10.0,
        )
        assert result["regularization_method"] == "gcv"

    def test_dp_regularization(self, detector, readings):
        result = detector.unfold_scip(
            readings,
            regularization_method="dp",
            noise_var=0.01,
            save_result=False,
            timeout=10.0,
        )
        assert result["regularization_method"] == "dp"

    def test_wrong_initial_spectrum_length(self, detector, readings):
        with pytest.raises(ValueError, match="must match"):
            detector.unfold_scip(
                readings, initial_spectrum=np.ones(5), save_result=False
            )

    def test_with_errors(self, detector, readings):
        result = detector.unfold_scip(
            readings,
            calculate_errors=True,
            n_montecarlo=5,
            noise_level=0.05,
            save_result=False,
            timeout=10.0,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result

    def test_save_result(self, detector, readings):
        detector.results_history.clear()
        detector.unfold_scip(readings, save_result=True, timeout=10.0)
        assert len(detector.results_history) == 1

    def test_solver_failure_returns_zero(self, detector, readings):
        with patch("bssunfold.core.unfold_scip.solve_scip", return_value=None):
            with pytest.warns(Warning):
                result = detector.unfold_scip(
                    readings, save_result=False, timeout=10.0
                )
        assert np.all(result["spectrum"] == 0)


def test_unfold_scip_is_detector_method():
    assert hasattr(Detector, "unfold_scip")


def test_core_exports():
    from bssunfold.core import solve_scip, unfold_scip

    assert solve_scip is not None
    assert unfold_scip is not None


def test_unfold_scip_module_function(detector, readings):
    from bssunfold.core.unfold_scip import unfold_scip

    result = unfold_scip(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        save_result=False,
        timeout=10.0,
    )
    assert "spectrum" in result


def test_unfold_combined_scip(detector, readings):
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
                "method": "scip",
                "params": {"save_result": False, "timeout": 10.0},
            }
        ],
        verbose=False,
    )
    assert "spectrum" in result
    assert np.all(result["spectrum"] >= 0)


def test_platform_check_scip():
    from bssunfold.platform_check import (
        SCIP_AVAILABLE,
        check_scip_availability,
    )

    assert isinstance(check_scip_availability(), bool)
    assert isinstance(SCIP_AVAILABLE, bool)

    with patch("builtins.__import__", side_effect=ImportError):
        assert check_scip_availability() is False
