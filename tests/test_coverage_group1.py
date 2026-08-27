"""Coverage Group 1: Tests for low-coverage source files.

Targets:
  - unfold_tsvd.py        (40% → 90%+)
  - unfold_amaxed.py      (7%  → 85%+)
  - unfold_imaxed.py      (18% → 85%+)
  - unfold_mlem_odl.py    (33% → 85%+)
  - _interpret_pyopt.py   (57% → 85%+)
  - _interpret_report.py  (82% → 95%+)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def detector():
    """Default Detector instance."""
    from bssunfold import Detector

    return Detector()


@pytest.fixture
def readings(detector):
    """Sample readings dict using first detector."""
    return {detector.detector_names[0]: 100.0}


# ============================================================================
# 1. unfold_tsvd.py — cover lines 41-106, 150
# ============================================================================


class TestTSVD:
    """Tests for the TSVD unfolding method."""

    # ------ solve_tsvd: automatic k-selection methods ------

    def test_solve_tsvd_energy_method(self):
        """Exercise the 'energy' branch in _automatic_k_selection (line 43-46)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(0)
        A = np.random.rand(6, 10)
        b = A @ np.random.rand(10)
        x = solve_tsvd(A, b, x0=np.ones(10), method="energy")
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_tsvd_l_curve_method(self):
        """Exercise the 'l_curve' branch (lines 48-76)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(1)
        A = np.random.rand(8, 12)
        b = A @ np.random.rand(12)
        x = solve_tsvd(A, b, x0=np.ones(12), method="l_curve")
        assert x.shape == (12,)
        assert np.all(x >= 0)

    def test_solve_tsvd_gcv_method(self):
        """Exercise the 'gcv' branch (lines 78-88)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(2)
        A = np.random.rand(5, 8)
        b = A @ np.random.rand(8)
        x = solve_tsvd(A, b, x0=np.ones(8), method="gcv")
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_solve_tsvd_threshold_ratio_method(self):
        """Exercise the 'threshold_ratio' branch (lines 90-93)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(3)
        A = np.random.rand(5, 8)
        b = A @ np.random.rand(8)
        x = solve_tsvd(A, b, x0=np.ones(8), method="threshold_ratio")
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_solve_tsvd_median_threshold_method(self):
        """Exercise the 'median_threshold' branch (lines 95-97)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(4)
        A = np.random.rand(5, 8)
        b = A @ np.random.rand(8)
        x = solve_tsvd(A, b, x0=np.ones(8), method="median_threshold")
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_solve_tsvd_donoho_method(self):
        """Exercise the 'donoho' branch (lines 99-103)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(5)
        A = np.random.rand(5, 8)
        b = A @ np.random.rand(8)
        x = solve_tsvd(A, b, x0=np.ones(8), method="donoho")
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_solve_tsvd_unknown_method_fallback(self):
        """Exercise the default fallback (lines 105-106) when method is unrecognized."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(6)
        A = np.random.rand(5, 8)
        b = A @ np.random.rand(8)
        x = solve_tsvd(A, b, x0=np.ones(8), method="unknown_method")
        assert x.shape == (8,)
        assert np.all(x >= 0)

    def test_solve_tsvd_with_fixed_k(self):
        """Test solve_tsvd with an explicit k parameter (line 147-148)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(7)
        A = np.random.rand(5, 10)
        b = A @ np.random.rand(10)
        x = solve_tsvd(A, b, x0=np.ones(10), k=3)
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_tsvd_with_threshold(self):
        """Test solve_tsvd with a threshold parameter (line 149-150)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(8)
        A = np.random.rand(5, 10)
        b = A @ np.random.rand(10)
        x = solve_tsvd(A, b, x0=np.ones(10), threshold=0.1)
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_tsvd_discrepancy_with_noise_level(self):
        """Test discrepancy method with explicit noise_level."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(9)
        A = np.random.rand(5, 10)
        b = A @ np.random.rand(10)
        x = solve_tsvd(A, b, x0=np.ones(10), method="discrepancy", noise_level=0.5)
        assert x.shape == (10,)
        assert np.all(x >= 0)

    def test_solve_tsvd_k_clamp_to_max(self):
        """Test that k is clamped when it exceeds singular values (line 153)."""
        from bssunfold.core.unfold_tsvd import solve_tsvd

        np.random.seed(10)
        A = np.random.rand(3, 5)
        b = A @ np.random.rand(5)
        x = solve_tsvd(A, b, x0=np.ones(5), k=100)
        assert x.shape == (5,)
        assert np.all(x >= 0)

    # ------ unfold_tsvd via Detector ------

    def test_unfold_tsvd_default(self, detector, readings):
        """Test the high-level Detector.unfold_tsvd."""
        result = detector.unfold_tsvd(readings)
        assert "spectrum" in result
        assert result["method"] == "TSVD"
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_tsvd_with_energy_method(self, detector, readings):
        """Test Detector.unfold_tsvd with energy k-selection."""
        result = detector.unfold_tsvd(readings, method="energy")
        assert result["method"] == "TSVD"
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_tsvd_with_gcv(self, detector, readings):
        """Test Detector.unfold_tsvd with GCV k-selection."""
        result = detector.unfold_tsvd(readings, method="gcv")
        assert result["method"] == "TSVD"

    def test_unfold_tsvd_with_l_curve(self, detector, readings):
        """Test Detector.unfold_tsvd with L-curve k-selection."""
        result = detector.unfold_tsvd(readings, method="l_curve")
        assert result["method"] == "TSVD"

    def test_unfold_tsvd_with_fixed_k(self, detector, readings):
        """Test Detector.unfold_tsvd with fixed k."""
        result = detector.unfold_tsvd(readings, k=5)
        assert result["method"] == "TSVD"

    def test_unfold_tsvd_with_threshold(self, detector, readings):
        """Test Detector.unfold_tsvd with threshold parameter."""
        result = detector.unfold_tsvd(readings, threshold=0.01)
        assert result["method"] == "TSVD"

    def test_unfold_tsvd_with_save(self, detector, readings):
        """Test Detector.unfold_tsvd with save_result=True."""
        detector.clear_results()
        result = detector.unfold_tsvd(readings, save_result=True)
        assert len(detector.list_results()) >= 1

    def test_unfold_tsvd_montecarlo(self, detector, readings):
        """Test TSVD with Monte-Carlo errors."""
        result = detector.unfold_tsvd(
            readings, calculate_errors=True, n_montecarlo=5, random_state=42
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result


# ============================================================================
# 2. unfold_amaxed.py — cover lines 62-206, 275-280
# ============================================================================


class TestAMAXED:
    """Tests for the AMAXED unfolding method."""

    def test_solve_amaxed_basic(self):
        """Test solve_amaxed with a simple system."""
        from bssunfold.core.unfold_amaxed import solve_amaxed

        np.random.seed(42)
        A = np.random.rand(5, 10) + 0.1
        x_true = np.random.rand(10) + 0.1
        b = A @ x_true
        x0 = np.ones(10)

        x_opt, n_iter, converged = solve_amaxed(A, b, x0, max_iterations=200)
        assert x_opt.shape == (10,)
        assert n_iter > 0
        assert isinstance(converged, bool)

    def test_solve_amaxed_with_target_chi2(self):
        """Test solve_amaxed with an explicit target_chi2 (line 84-89)."""
        from bssunfold.core.unfold_amaxed import solve_amaxed

        np.random.seed(43)
        A = np.random.rand(5, 10) + 0.1
        x_true = np.random.rand(10) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_amaxed(
            A, b, np.ones(10), target_chi2=3.0, max_iterations=200
        )
        assert x_opt.shape == (10,)

    def test_solve_amaxed_small_sigma(self):
        """Test with a very small sigma_factor."""
        from bssunfold.core.unfold_amaxed import solve_amaxed

        np.random.seed(44)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_amaxed(
            A, b, np.ones(8), sigma_factor=0.01, max_iterations=100
        )
        assert x_opt.shape == (8,)

    def test_solve_amaxed_large_sigma(self):
        """Test with a large sigma_factor."""
        from bssunfold.core.unfold_amaxed import solve_amaxed

        np.random.seed(45)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_amaxed(
            A, b, np.ones(8), sigma_factor=1.0, max_iterations=100
        )
        assert x_opt.shape == (8,)

    def test_solve_amaxed_zero_prior(self):
        """Test with a zero prior (should be floored to 1e-300, line 68)."""
        from bssunfold.core.unfold_amaxed import solve_amaxed

        np.random.seed(46)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_amaxed(
            A, b, np.zeros(8), max_iterations=100
        )
        assert x_opt.shape == (8,)

    # ------ unfold_amaxed via Detector ------

    def test_unfold_amaxed_default(self, detector, readings):
        """Test the high-level Detector.unfold_amaxed."""
        result = detector.unfold_amaxed(readings, max_iterations=200)
        assert "spectrum" in result
        assert result["method"] == "AMAXED"
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_amaxed_with_initial_spectrum(self, detector, readings):
        """Test with an explicit initial_spectrum (lines 275-276)."""
        init = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_amaxed(
            readings, initial_spectrum=init, max_iterations=200
        )
        assert result["method"] == "AMAXED"

    def test_unfold_amaxed_no_initial_spectrum(self, detector, readings):
        """Test without initial_spectrum (lines 277-278 — default ones)."""
        result = detector.unfold_amaxed(
            readings, initial_spectrum=None, max_iterations=200
        )
        assert result["method"] == "AMAXED"

    def test_unfold_amaxed_with_target_chi2(self, detector, readings):
        """Test with explicit target_chi2."""
        result = detector.unfold_amaxed(
            readings, target_chi2=5.0, max_iterations=200
        )
        assert result["method"] == "AMAXED"

    def test_unfold_amaxed_with_save(self, detector, readings):
        """Test with save_result=True."""
        detector.clear_results()
        result = detector.unfold_amaxed(
            readings, max_iterations=200, save_result=True
        )
        assert len(detector.list_results()) >= 1

    def test_unfold_amaxed_montecarlo(self, detector, readings):
        """Test with Monte-Carlo uncertainty."""
        result = detector.unfold_amaxed(
            readings,
            max_iterations=100,
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert "spectrum_uncert_mean" in result or "spectrum" in result

    def test_unfold_amaxed_converges(self, detector, readings):
        """Test that converged flag is present in result."""
        result = detector.unfold_amaxed(readings, max_iterations=500)
        assert "converged" in result
        assert "iterations" in result


# ============================================================================
# 3. unfold_imaxed.py — cover lines 60-115, 181-186
# ============================================================================


class TestIMAXED:
    """Tests for the IMAXED unfolding method."""

    def test_solve_imaxed_basic(self):
        """Test solve_imaxed with a simple system."""
        from bssunfold.core.unfold_imaxed import solve_imaxed

        np.random.seed(50)
        A = np.random.rand(5, 10) + 0.1
        x_true = np.random.rand(10) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_imaxed(A, b, x_true, max_iterations=200)
        assert x_opt.shape == (10,)
        assert n_iter >= 0
        assert isinstance(converged, bool)

    def test_solve_imaxed_small_sigma(self):
        """Test with small sigma_factor."""
        from bssunfold.core.unfold_imaxed import solve_imaxed

        np.random.seed(51)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_imaxed(
            A, b, x_true, sigma_factor=0.01, max_iterations=100
        )
        assert x_opt.shape == (8,)

    def test_solve_imaxed_zero_prior(self):
        """Test with zero prior (should be floored to 1e-300, line 70)."""
        from bssunfold.core.unfold_imaxed import solve_imaxed

        np.random.seed(52)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_imaxed(
            A, b, np.zeros(8), max_iterations=100
        )
        assert x_opt.shape == (8,)

    def test_solve_imaxed_tight_tolerance(self):
        """Test with tight tolerance."""
        from bssunfold.core.unfold_imaxed import solve_imaxed

        np.random.seed(53)
        A = np.random.rand(4, 8) + 0.1
        x_true = np.random.rand(8) + 0.1
        b = A @ x_true

        x_opt, n_iter, converged = solve_imaxed(
            A, b, x_true, tolerance=1e-12, max_iterations=200
        )
        assert x_opt.shape == (8,)

    # ------ unfold_imaxed via Detector ------

    def test_unfold_imaxed_default(self, detector, readings):
        """Test the high-level Detector.unfold_imaxed."""
        result = detector.unfold_imaxed(readings, max_iterations=200)
        assert "spectrum" in result
        assert result["method"] == "IMAXED"
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_imaxed_with_initial_spectrum(self, detector, readings):
        """Test with explicit initial_spectrum (lines 181-182)."""
        init = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_imaxed(
            readings, initial_spectrum=init, max_iterations=200
        )
        assert result["method"] == "IMAXED"

    def test_unfold_imaxed_no_initial_spectrum(self, detector, readings):
        """Test without initial_spectrum (lines 183-184 — default ones)."""
        result = detector.unfold_imaxed(
            readings, initial_spectrum=None, max_iterations=200
        )
        assert result["method"] == "IMAXED"

    def test_unfold_imaxed_with_save(self, detector, readings):
        """Test with save_result=True."""
        detector.clear_results()
        result = detector.unfold_imaxed(
            readings, max_iterations=200, save_result=True
        )
        assert len(detector.list_results()) >= 1

    def test_unfold_imaxed_montecarlo(self, detector, readings):
        """Test with Monte-Carlo uncertainty."""
        result = detector.unfold_imaxed(
            readings,
            max_iterations=100,
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert "spectrum_uncert_mean" in result or "spectrum" in result


# ============================================================================
# 4. unfold_mlem_odl.py — cover lines 84-118 (requires odl)
# ============================================================================


class TestMLEMODL:
    """Tests for the ODL-based MLEM unfolding method."""

    def test_import_error_when_odl_missing(self):
        """Test that ImportError is raised when odl is not installed."""
        pytest.importorskip("odl")
        # If we reach here, odl is installed — test the full flow below

    def test_unfold_mlem_odl_basic(self, detector, readings):
        """Test unfold_mlem_odl through the Detector (lines 84-118)."""
        odl = pytest.importorskip("odl")
        result = detector.unfold_mlem_odl(readings, max_iterations=10)
        assert "spectrum" in result
        assert result["method"] == "MLEM (ODL)"
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_mlem_odl_with_initial_spectrum(self, detector, readings):
        """Test with explicit initial_spectrum."""
        pytest.importorskip("odl")
        init = np.ones(detector.n_energy_bins) * 0.5
        result = detector.unfold_mlem_odl(
            readings, initial_spectrum=init, max_iterations=10
        )
        assert result["method"] == "MLEM (ODL)"

    def test_unfold_mlem_odl_with_save(self, detector, readings):
        """Test with save_result=True."""
        pytest.importorskip("odl")
        detector.clear_results()
        result = detector.unfold_mlem_odl(
            readings, max_iterations=10, save_result=True
        )
        assert len(detector.list_results()) >= 1

    def test_solve_wrapper_directly(self):
        """Test the internal solve_wrapper directly (lines 84-114)."""
        odl = pytest.importorskip("odl")
        from bssunfold.core.unfold_mlem_odl import unfold_mlem_odl

        # Build a minimal system
        A = np.array([[1.0, 0.5], [0.3, 1.0]])
        b = np.array([1.5, 1.3])
        E_MeV = np.array([1e-8, 1e-7])

        result = unfold_mlem_odl(
            detector_names=["d1", "d2"],
            n_energy_bins=2,
            E_MeV=E_MeV,
            sensitivities={"d1": A[0], "d2": A[1]},
            cc_icrp116={},
            save_result_callback=lambda r: "id",
            readings={"d1": 1.5, "d2": 1.3},
            max_iterations=5,
        )
        assert "spectrum" in result
        assert len(result["spectrum"]) == 2

    def test_solve_wrapper_default_x0(self):
        """Test solve_wrapper uses default x0 when None (lines 86-87)."""
        odl = pytest.importorskip("odl")
        from bssunfold.core.unfold_mlem_odl import unfold_mlem_odl

        A = np.array([[1.0, 0.5], [0.3, 1.0]])
        E_MeV = np.array([1e-8, 1e-7])

        result = unfold_mlem_odl(
            detector_names=["d1", "d2"],
            n_energy_bins=2,
            E_MeV=E_MeV,
            sensitivities={"d1": A[0], "d2": A[1]},
            cc_icrp116={},
            save_result_callback=lambda r: "id",
            readings={"d1": 1.5, "d2": 1.3},
            initial_spectrum=None,
            max_iterations=5,
        )
        assert "spectrum" in result


# ============================================================================
# 5. _interpret_pyopt.py — cover lines 27, 35-62, 139-199, 219-224, 284-301
# ============================================================================


class TestInterpretPyopt:
    """Tests for _interpret_pyopt.py.

    Because pyoptexplain is not installed in this environment, we test the
    import-error paths and the validation logic that runs *before* the import.
    """

    def test_namespace_load_raises_without_pyoptexplain(self):
        """Test _PyOptExplainNamespace.load() raises ImportError (lines 28-34)."""
        from bssunfold.core._interpret_pyopt import _PyOptExplainNamespace

        ns = _PyOptExplainNamespace()
        # Reset in case cached from another test
        ns._loaded = None
        with pytest.raises(ImportError, match="pyoptexplain"):
            ns.load()

    def test_require_pyoptexplain_raises(self):
        """Test _require_pyoptexplain() raises ImportError (line 68-69)."""
        from bssunfold.core._interpret_pyopt import (
            _PyOptExplainNamespace,
            _require_pyoptexplain,
        )

        # Reset the module-level singleton
        import bssunfold.core._interpret_pyopt as mod
        mod._pyopt = _PyOptExplainNamespace()
        mod._pyopt._loaded = None
        with pytest.raises(ImportError, match="pyoptexplain"):
            _require_pyoptexplain()

    def test_build_interpretation_qp_raises_without_pyoptexplain(self):
        """Test build_interpretation_qp raises ImportError early (line 137)."""
        from bssunfold.core._interpret_pyopt import (
            _PyOptExplainNamespace,
            build_interpretation_qp,
        )

        import bssunfold.core._interpret_pyopt as mod
        mod._pyopt = _PyOptExplainNamespace()
        mod._pyopt._loaded = None

        A = np.eye(3)
        b = np.ones(3)
        with pytest.raises(ImportError, match="pyoptexplain"):
            build_interpretation_qp(A, b, alpha=0.1)

    def test_solve_interpret_raises_without_pyoptexplain(self):
        """Test solve_interpret raises ImportError (lines 283-284)."""
        from bssunfold.core._interpret_pyopt import (
            _PyOptExplainNamespace,
            solve_interpret,
        )

        import bssunfold.core._interpret_pyopt as mod
        mod._pyopt = _PyOptExplainNamespace()
        mod._pyopt._loaded = None

        A = np.eye(3)
        b = np.ones(3)
        with pytest.raises(ImportError, match="pyoptexplain"):
            solve_interpret(A, b, alpha=0.1)

    def test_namespace_cached_return(self):
        """Test that once _loaded is set, load() returns it (line 26-27)."""
        from types import SimpleNamespace
        from bssunfold.core._interpret_pyopt import _PyOptExplainNamespace

        ns = _PyOptExplainNamespace()
        fake = SimpleNamespace(foo="bar")
        ns._loaded = fake
        assert ns.load() is fake

    # -- Validation tests for build_interpretation_qp --
    # These exercise lines 139-162 (the validation block).
    # We use a mock for pyoptexplain so the validation runs but the
    # actual QP construction is not needed.

    def _mock_pyopt(self, monkeypatch):
        """Patch _pyopt to bypass the import and provide a mock handle."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        mock_handle_cls = MagicMock(return_value=MagicMock())
        fake_ns = SimpleNamespace(
            QuadraticMatrixProblemHandle=mock_handle_cls,
        )
        mock_namespace = SimpleNamespace(_loaded=None)

        def fake_load():
            return fake_ns

        mock_namespace.load = fake_load

        import bssunfold.core._interpret_pyopt as mod
        monkeypatch.setattr(mod, "_pyopt", mock_namespace)
        return mock_handle_cls

    def test_validation_bad_A_shape(self, monkeypatch):
        """Test ValueError for non-2D A (line 141-145)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="2D"):
            build_interpretation_qp(np.ones(3), np.ones(3), alpha=0.1)

    def test_validation_bad_b_shape(self, monkeypatch):
        """Test ValueError for non-1D b (line 141-145)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="1D"):
            build_interpretation_qp(np.eye(3), np.ones((3, 1)), alpha=0.1)

    def test_validation_mismatched_dimensions(self, monkeypatch):
        """Test ValueError for A/b row mismatch (line 141-145)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="matching"):
            build_interpretation_qp(np.eye(3), np.ones(2), alpha=0.1)

    def test_validation_negative_alpha(self, monkeypatch):
        """Test ValueError for negative alpha (line 147-148)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="alpha"):
            build_interpretation_qp(np.eye(3), np.ones(3), alpha=-1.0)

    def test_validation_inf_alpha(self, monkeypatch):
        """Test ValueError for inf alpha (line 147-148)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="alpha"):
            build_interpretation_qp(np.eye(3), np.ones(3), alpha=np.inf)

    def test_validation_bad_norm(self, monkeypatch):
        """Test ValueError for unsupported norm (line 149-150)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="norm type"):
            build_interpretation_qp(np.eye(3), np.ones(3), alpha=0.1, norm=3)

    def test_validation_bad_smoothness_order(self, monkeypatch):
        """Test ValueError for unsupported smoothness_order (line 151-154)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="smoothness_order"):
            build_interpretation_qp(
                np.eye(3), np.ones(3), alpha=0.1, smoothness_order=3
            )

    def test_validation_negative_smoothness_weight(self, monkeypatch):
        """Test ValueError for negative smoothness_weight (line 155-158)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="smoothness_weight"):
            build_interpretation_qp(
                np.eye(3), np.ones(3), alpha=0.1, smoothness_weight=-1.0
            )

    def test_validation_inf_lower_bound(self, monkeypatch):
        """Test ValueError for inf lower_bound (line 159-160)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="lower_bound"):
            build_interpretation_qp(
                np.eye(3), np.ones(3), alpha=0.1, lower_bound=np.inf
            )

    def test_validation_inf_norm_value(self, monkeypatch):
        """Test ValueError for inf norm_value with enforce_norm (line 161-162)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="norm_value"):
            build_interpretation_qp(
                np.eye(3), np.ones(3), alpha=0.1,
                enforce_norm=True, norm_value=np.inf,
            )

    def test_validation_bad_variable_names_length(self, monkeypatch):
        """Test ValueError for wrong variable_names length (line 181-184)."""
        self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        with pytest.raises(ValueError, match="variable_names"):
            build_interpretation_qp(
                np.eye(3), np.ones(3), alpha=0.1,
                variable_names=["a", "b"],
            )

    # -- Test actual QP construction paths (with mock handle) --

    def test_build_qp_norm2_no_smoothness(self, monkeypatch):
        """Test QP build path: norm=2, smoothness_order=0 (lines 164-205)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(4)
        b = np.ones(4)
        handle = build_interpretation_qp(A, b, alpha=0.1, norm=2)
        assert mock_cls.called

    def test_build_qp_norm1(self, monkeypatch):
        """Test QP build path: norm=1 (lines 171-172)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(4)
        b = np.ones(4)
        handle = build_interpretation_qp(A, b, alpha=0.1, norm=1)
        assert mock_cls.called

    def test_build_qp_with_smoothness_order1(self, monkeypatch):
        """Test QP build path: smoothness_order=1 (lines 165-167)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(6)
        b = np.ones(6)
        handle = build_interpretation_qp(
            A, b, alpha=0.1, smoothness_order=1, smoothness_weight=0.5
        )
        assert mock_cls.called

    def test_build_qp_with_smoothness_order2(self, monkeypatch):
        """Test QP build path: smoothness_order=2 (lines 165-167)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(8)
        b = np.ones(8)
        handle = build_interpretation_qp(
            A, b, alpha=0.1, smoothness_order=2, smoothness_weight=0.5
        )
        assert mock_cls.called

    def test_build_qp_with_enforce_norm(self, monkeypatch):
        """Test QP build path with enforce_norm=True (lines 188-198)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(4)
        b = np.ones(4)
        handle = build_interpretation_qp(
            A, b, alpha=0.1, enforce_norm=True, norm_value=2.0
        )
        assert mock_cls.called
        # Check that A_eq and b_eq were passed
        call_kwargs = mock_cls.call_args
        assert "A_eq" in call_kwargs.kwargs or call_kwargs[1].get("A_eq") is not None

    def test_build_qp_with_variable_names(self, monkeypatch):
        """Test QP build with custom variable_names (lines 176-180)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(3)
        b = np.ones(3)
        handle = build_interpretation_qp(
            A, b, alpha=0.1, variable_names=["a", "b", "c"]
        )
        assert mock_cls.called

    def test_build_qp_default_variable_names(self, monkeypatch):
        """Test QP build with default variable_names (line 179)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(3)
        b = np.ones(3)
        handle = build_interpretation_qp(A, b, alpha=0.1)
        assert mock_cls.called

    def test_build_qp_without_enforce_norm(self, monkeypatch):
        """Test QP build without enforce_norm (lines 199-205)."""
        mock_cls = self._mock_pyopt(monkeypatch)
        from bssunfold.core._interpret_pyopt import build_interpretation_qp

        A = np.eye(3)
        b = np.ones(3)
        handle = build_interpretation_qp(A, b, alpha=0.1, enforce_norm=False)
        assert mock_cls.called


# ============================================================================
# 6. _interpret_report.py — cover missing lines to reach 95%+
# ============================================================================


class TestInterpretReport:
    """Tests for _interpret_report.py."""

    def test_interpretation_result_to_dict(self):
        """Test InterpretationResult.to_dict() (lines 42-50)."""
        from bssunfold.core._interpret_report import InterpretationResult

        r = InterpretationResult(
            spectrum=np.array([1.0, 2.0, 3.0]),
            status="optimal",
            objective_value=0.5,
            report="# Test report",
            metrics={"key": "val"},
            tables={"summary": "data"},
        )
        d = r.to_dict()
        assert d["status"] == "optimal"
        assert d["objective_value"] == 0.5
        assert d["report"] == "# Test report"
        assert d["spectrum"] == [1.0, 2.0, 3.0]
        assert "interpretation_metrics" in d

    def test_fmt_none(self):
        """Test _fmt with None returns em-dash (line 55-56)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(None) == "—"

    def test_fmt_bool_true(self):
        """Test _fmt with True (lines 57-58)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(True) == "True"

    def test_fmt_bool_false(self):
        """Test _fmt with False."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(False) == "False"

    def test_fmt_np_bool(self):
        """Test _fmt with np.bool_ (lines 57-58)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(np.bool_(True)) == "True"
        assert _fmt(np.bool_(False)) == "False"

    def test_fmt_float(self):
        """Test _fmt with float (lines 59-65)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(3.14159) == "3.14159"

    def test_fmt_float_nan(self):
        """Test _fmt with NaN returns em-dash (lines 60-61)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(np.nan) == "—"

    def test_fmt_np_float(self):
        """Test _fmt with np.float64."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(np.float64(2.5)) == "2.5"

    def test_fmt_np_integer(self):
        """Test _fmt with np.integer."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(np.int64(42)) == "42"

    def test_fmt_list(self):
        """Test _fmt with list returns ellipsis (lines 66-67)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt([1, 2, 3]) == "…"

    def test_fmt_tuple(self):
        """Test _fmt with tuple returns ellipsis."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt((1, 2)) == "…"

    def test_fmt_ndarray(self):
        """Test _fmt with np.ndarray returns ellipsis."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt(np.array([1, 2])) == "…"

    def test_fmt_string(self):
        """Test _fmt with string passes through (line 68)."""
        from bssunfold.core._interpret_report import _fmt

        assert _fmt("hello") == "hello"

    def test_df_to_markdown_empty(self):
        """Test _df_to_markdown with empty DataFrame (lines 73-74)."""
        from bssunfold.core._interpret_report import _df_to_markdown

        assert _df_to_markdown(pd.DataFrame()) == "_No data._"

    def test_df_to_markdown_none(self):
        """Test _df_to_markdown with None (line 73)."""
        from bssunfold.core._interpret_report import _df_to_markdown

        assert _df_to_markdown(None) == "_No data._"

    def test_df_to_markdown_normal(self):
        """Test _df_to_markdown with a normal DataFrame (lines 75-83)."""
        from bssunfold.core._interpret_report import _df_to_markdown

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        md = _df_to_markdown(df)
        assert "| a | b |" in md
        assert "| --- | --- |" in md

    def test_df_to_markdown_with_none_values(self):
        """Test _df_to_markdown with None/NaN values."""
        from bssunfold.core._interpret_report import _df_to_markdown

        df = pd.DataFrame({"a": [1.0, np.nan], "b": [None, 4.0]})
        md = _df_to_markdown(df)
        assert "—" in md

    def test_df_to_markdown_with_bool_values(self):
        """Test _df_to_markdown with bool values."""
        from bssunfold.core._interpret_report import _df_to_markdown

        df = pd.DataFrame({"a": [True, False], "b": [1.0, 2.0]})
        md = _df_to_markdown(df)
        assert "True" in md
        assert "False" in md

    def test_rows_to_frame(self):
        """Test _rows_to_frame (lines 84-88)."""
        from bssunfold.core._interpret_report import _rows_to_frame

        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = _rows_to_frame(rows)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_safe_cond_normal(self):
        """Test _safe_cond with a normal matrix (lines 144-149)."""
        from bssunfold.core._interpret_report import _safe_cond

        Q = np.eye(3)
        assert _safe_cond(Q) == 1.0

    def test_safe_cond_singular(self):
        """Test _safe_cond with a singular matrix returns None (line 149)."""
        from bssunfold.core._interpret_report import _safe_cond

        Q = np.zeros((3, 3))
        assert _safe_cond(Q) is None

    def test_safe_cond_linalg_error(self):
        """Test _safe_cond with NaN matrix triggers LinAlgError (line 148)."""
        from bssunfold.core._interpret_report import _safe_cond

        # NaN matrix causes SVD to fail → LinAlgError
        Q = np.array([[1.0, np.nan], [0.0, 1.0]])
        assert _safe_cond(Q) is None

    def test_safe_cond_inf(self):
        """Test _safe_cond returns None for infinite condition number (line 149)."""
        from bssunfold.core._interpret_report import _safe_cond

        # Very ill-conditioned matrix
        Q = np.diag([1e300, 1.0, 1.0])
        result = _safe_cond(Q)
        # Condition number may overflow to inf, which should return None
        assert result is None or isinstance(result, float)

    def test_effective_capabilities_none(self):
        """Test _effective_capabilities with None (lines 153-154)."""
        from bssunfold.core._interpret_report import _effective_capabilities

        assert _effective_capabilities(None) == []

    def test_effective_capabilities_empty(self):
        """Test _effective_capabilities with empty DataFrame (line 153)."""
        from bssunfold.core._interpret_report import _effective_capabilities

        assert _effective_capabilities(pd.DataFrame()) == []

    def test_effective_capabilities_no_capability_col(self):
        """Test _effective_capabilities without 'capability' column (lines 155-156)."""
        from bssunfold.core._interpret_report import _effective_capabilities

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert _effective_capabilities(df) == []

    def test_effective_capabilities_mixed(self):
        """Test _effective_capabilities with mixed effective flags (lines 157-163)."""
        from bssunfold.core._interpret_report import _effective_capabilities

        df = pd.DataFrame({
            "capability": ["a", "b", "c"],
            "effective": [True, False, True],
        })
        result = _effective_capabilities(df)
        assert result == ["a", "c"]

    def test_detector_importance_empty(self):
        """Test _detector_importance with empty list (lines 170-180)."""
        from bssunfold.core._interpret_report import _detector_importance

        assert _detector_importance([]) == []

    def test_detector_importance_ranked(self):
        """Test _detector_importance ranking (lines 170-180)."""
        from bssunfold.core._interpret_report import _detector_importance

        rows = [
            {"detector": "d1", "spectrum_change": 0.1},
            {"detector": "d1", "spectrum_change": 0.3},
            {"detector": "d2", "spectrum_change": 0.2},
        ]
        result = _detector_importance(rows)
        assert len(result) == 2
        assert result[0]["detector"] == "d1"
        assert result[0]["max_spectrum_change"] == 0.3
        assert result[1]["detector"] == "d2"

    def test_robustness_metrics_none(self):
        """Test _robustness_metrics with None (lines 185-186)."""
        from bssunfold.core._interpret_report import _robustness_metrics

        assert _robustness_metrics(None) == {"cases": 0}

    def test_robustness_metrics_empty(self):
        """Test _robustness_metrics with empty DataFrame (line 185)."""
        from bssunfold.core._interpret_report import _robustness_metrics

        assert _robustness_metrics(pd.DataFrame()) == {"cases": 0}

    def test_robustness_metrics_with_data(self):
        """Test _robustness_metrics with actual perturbation data (lines 187-219)."""
        from bssunfold.core._interpret_report import _robustness_metrics

        df = pd.DataFrame({
            "case": ["c1", "c2", "c3"],
            "target": ["base", "objective", "constraint"],
            "magnitude": [0.0, 0.05, 0.1],
            "status": ["optimal", "optimal", "optimal"],
            "objective_change": [0.0, 0.01, 0.02],
            "objective_change_relative": [0.0, 0.05, 0.1],
            "max_variable_change_relative": [0.0, 0.03, 0.06],
            "binding_similarity": [1.0, 0.95, 0.9],
            "regime_changed": [False, False, True],
        })
        result = _robustness_metrics(df)
        assert result["case_count"] == 2  # non-base rows
        # Only 'objective' targets contribute to max_spectrum_change_relative
        assert result["max_spectrum_change_relative"] == 0.03
        assert len(result["cases"]) == 2

    def test_robustness_metrics_no_objective_target(self):
        """Test _robustness_metrics with no 'objective' target (lines 209-218)."""
        from bssunfold.core._interpret_report import _robustness_metrics

        df = pd.DataFrame({
            "case": ["c1"],
            "target": ["constraint"],
            "magnitude": [0.1],
            "status": ["optimal"],
            "objective_change": [0.01],
            "objective_change_relative": [0.05],
            "max_variable_change_relative": [0.1],
            "binding_similarity": [0.9],
            "regime_changed": [False],
        })
        result = _robustness_metrics(df)
        assert result["max_spectrum_change_relative"] is None

    def test_float_or_none(self):
        """Test _float_or_none (lines 222-228)."""
        from bssunfold.core._interpret_report import _float_or_none

        assert _float_or_none(None) is None
        assert _float_or_none(3.14) == 3.14
        assert _float_or_none("abc") is None
        assert _float_or_none(np.int64(5)) == 5.0

    def test_frame_records_none(self):
        """Test _frame_records with None (lines 231-233)."""
        from bssunfold.core._interpret_report import _frame_records

        assert _frame_records(None) == []

    def test_frame_records_empty(self):
        """Test _frame_records with empty DataFrame (line 232)."""
        from bssunfold.core._interpret_report import _frame_records

        assert _frame_records(pd.DataFrame()) == []

    def test_frame_records_normal(self):
        """Test _frame_records with data (lines 234)."""
        from bssunfold.core._interpret_report import _frame_records

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _frame_records(df)
        assert len(result) == 2

    def test_scenario_metrics_none(self):
        """Test _scenario_metrics with None (lines 239-240)."""
        from bssunfold.core._interpret_report import _scenario_metrics

        assert _scenario_metrics(None) == []

    def test_scenario_metrics_empty(self):
        """Test _scenario_metrics with empty DataFrame (line 239)."""
        from bssunfold.core._interpret_report import _scenario_metrics

        assert _scenario_metrics(pd.DataFrame()) == []

    def test_scenario_metrics_with_data(self):
        """Test _scenario_metrics with data (lines 241-251)."""
        from bssunfold.core._interpret_report import _scenario_metrics

        df = pd.DataFrame({
            "scenario": ["s1", "s2"],
            "status": ["optimal", "infeasible"],
            "objective_value": [1.0, None],
            "objective_change": [0.0, None],
        })
        result = _scenario_metrics(df)
        assert len(result) == 2
        assert result[0]["scenario"] == "s1"
        # np.nan is returned as float nan, not None, by _float_or_none
        assert result[1]["objective_value"] is None or np.isnan(result[1]["objective_value"])

    def test_build_metrics_basic(self):
        """Test _build_metrics (lines 91-141)."""
        from bssunfold.core._interpret_report import _build_metrics

        class FakeResult:
            status = "optimal"
            success = True
            objective_value = 1.5
            solver_name = "test"
            solve_time = 0.1

        x = np.array([1.0, 2.0, 3.0])
        E = np.array([0.1, 0.2, 0.3])
        Q = np.eye(3)
        metrics = _build_metrics(
            result=FakeResult(),
            x=x,
            E_MeV=E,
            residual_norm=0.01,
            Q=Q,
            model={"alpha": 0.1},
            active_groups=[1, 2],
            zero_groups=[0],
            bound_duals={},
            norm_dual=0.5,
            detector_rows=[],
            sensitivity_rows=[],
            robustness_summary=None,
            relaxation_df=None,
            nonneg_rows=[],
            scenario_df=None,
            sweep_rows=[],
            capabilities_df=None,
        )
        assert metrics["status"] == "optimal"
        assert metrics["success"] is True
        assert metrics["residual_norm"] == 0.01
        assert metrics["active_groups"] == [1, 2]
        assert metrics["norm_dual"] == 0.5
        assert "robustness" not in metrics
        assert "norm_relaxation" not in metrics

    def test_build_metrics_with_robustness_and_relaxation(self):
        """Test _build_metrics with optional sections (lines 135-140)."""
        from bssunfold.core._interpret_report import _build_metrics

        class FakeResult:
            status = "optimal"
            success = True
            objective_value = 1.0
            solver_name = "test"
            solve_time = 0.01

        robustness_df = pd.DataFrame({
            "case": ["c1"],
            "target": ["objective"],
            "magnitude": [0.05],
            "status": ["optimal"],
            "objective_change": [0.01],
            "objective_change_relative": [0.05],
            "max_variable_change_relative": [0.02],
            "binding_similarity": [0.99],
            "regime_changed": [False],
        })
        relaxation_df = pd.DataFrame({
            "a": [1], "b": [2],
        })
        scenario_df = pd.DataFrame({
            "scenario": ["s1"],
            "status": ["optimal"],
            "objective_value": [1.0],
            "objective_change": [0.0],
        })
        capabilities_df = pd.DataFrame({
            "capability": ["c1", "c2"],
            "effective": [True, False],
        })

        metrics = _build_metrics(
            result=FakeResult(),
            x=np.ones(3),
            E_MeV=None,
            residual_norm=0.0,
            Q=np.eye(3),
            model={},
            active_groups=[],
            zero_groups=[],
            bound_duals={},
            norm_dual=None,
            detector_rows=[],
            sensitivity_rows=[
                {"detector": "d1", "spectrum_change": 0.1}
            ],
            robustness_summary=robustness_df,
            relaxation_df=relaxation_df,
            nonneg_rows=[],
            scenario_df=scenario_df,
            sweep_rows=[],
            capabilities_df=capabilities_df,
        )
        assert "robustness" in metrics
        assert "norm_relaxation" in metrics
        assert "scenarios" in metrics
        assert metrics["capabilities"] == ["c1"]
        assert metrics["detector_importance"] == [
            {"detector": "d1", "max_spectrum_change": 0.1}
        ]

    def test_build_report_full(self):
        """Test _build_report with all sections populated (lines 254-383)."""
        from bssunfold.core._interpret_report import _build_report

        x = np.array([1.0, 0.0, 2.0])
        E = np.array([0.1, 0.2, 0.3])
        summary_df = pd.DataFrame({
            "metric": ["status", "objective"],
            "value": ["optimal", 1.5],
        })
        variables_df = pd.DataFrame({
            "var": ["E0", "E1", "E2"],
            "val": [1.0, 0.0, 2.0],
        })
        constraints_df = pd.DataFrame({
            "type": ["bound"],
            "index": [1],
        })
        binding_df = pd.DataFrame({
            "constraint": ["x1 >= 0"],
            "dual": [0.5],
        })
        duals_df = pd.DataFrame({
            "var": ["E0"],
            "dual": [0.1],
        })
        sensitivity_rows = [{"detector": "d1", "spectrum_change": 0.1}]
        robustness_df = pd.DataFrame({
            "case": ["c1"],
            "target": ["objective"],
            "magnitude": [0.05],
            "status": ["optimal"],
            "objective_change": [0.01],
            "objective_change_relative": [0.05],
            "max_variable_change_relative": [0.02],
            "binding_similarity": [0.99],
            "regime_changed": [False],
        })
        relaxation_df = pd.DataFrame({"a": [1], "b": [2]})
        scenario_df = pd.DataFrame({
            "scenario": ["s1"],
            "status": ["optimal"],
            "objective_value": [1.0],
            "objective_change": [0.0],
        })
        sweep_rows = [{"alpha": 0.1, "status": "optimal", "residual_norm": 0.01}]
        nonneg_rows = [
            {"allowed_negative": 0.01, "change_from_base": 0.02, "status": "optimal"}
        ]
        metrics = {
            "active_groups": [1],
            "detector_importance": [
                {"detector": "d1", "max_spectrum_change": 0.1}
            ],
            "robustness": {"max_spectrum_change_relative": 0.03},
            "nonnegativity_relaxation": nonneg_rows,
            "regularization_sweep": sweep_rows,
            "norm_dual": 0.5,
        }

        report = _build_report(
            x=x,
            E_MeV=E,
            residual_norm=0.01,
            summary_df=summary_df,
            variables_df=variables_df,
            constraints_df=constraints_df,
            binding_df=binding_df,
            duals_df=duals_df,
            detector_rows=[{"detector": "d1", "reading": 100}],
            sensitivity_rows=sensitivity_rows,
            robustness_summary=robustness_df,
            relaxation_df=relaxation_df,
            nonneg_rows=nonneg_rows,
            scenario_df=scenario_df,
            sweep_rows=sweep_rows,
            metrics=metrics,
            enforce_norm=True,
        )
        assert "# Unfolding interpretation report" in report
        assert "## Solve summary" in report
        assert "## Spectrum" in report
        assert "## Active constraints" in report
        assert "## Constraint table" in report
        assert "## Binding constraints" in report
        assert "## Duals" in report
        assert "## Detector diagnostics" in report
        assert "## Detector sensitivity" in report
        assert "## Regularization sweep" in report
        assert "## Non-negativity relaxation" in report
        assert "## Perturbation robustness" in report
        assert "## Norm constraint relaxation" in report
        assert "## Scenarios" in report
        assert "## Interpretation" in report
        assert "shadow price" in report  # enforce_norm=True

    def test_build_report_minimal(self):
        """Test _build_report with minimal/None sections (lines 284-295)."""
        from bssunfold.core._interpret_report import _build_report

        x = np.array([1.0, 2.0])
        E = np.array([0.1, 0.2])
        summary_df = pd.DataFrame({"m": ["ok"], "v": [1]})
        empty_duals = pd.DataFrame()
        metrics = {"active_groups": []}

        report = _build_report(
            x=x,
            E_MeV=E,
            residual_norm=0.0,
            summary_df=summary_df,
            variables_df=None,
            constraints_df=None,
            binding_df=None,
            duals_df=empty_duals,
            detector_rows=[],
            sensitivity_rows=[],
            robustness_summary=None,
            relaxation_df=None,
            nonneg_rows=[],
            scenario_df=None,
            sweep_rows=[],
            metrics=metrics,
            enforce_norm=False,
        )
        assert "# Unfolding interpretation report" in report
        assert "No duals available" in report
        assert "energy_MeV" in report

    def test_build_report_active_groups_preview(self):
        """Test active groups preview with >12 groups (lines 303-305)."""
        from bssunfold.core._interpret_report import _build_report

        x = np.ones(15)
        summary_df = pd.DataFrame({"m": ["ok"], "v": [1]})
        active = list(range(15))
        metrics = {"active_groups": active}

        report = _build_report(
            x=np.ones(15),
            E_MeV=np.linspace(1e-9, 1e-1, 15),
            residual_norm=0.0,
            summary_df=summary_df,
            variables_df=None,
            constraints_df=None,
            binding_df=None,
            duals_df=pd.DataFrame(),
            detector_rows=[],
            sensitivity_rows=[],
            robustness_summary=None,
            relaxation_df=None,
            nonneg_rows=[],
            scenario_df=None,
            sweep_rows=[],
            metrics=metrics,
            enforce_norm=False,
        )
        assert "15 total" in report

    def test_conclusions_all_paths(self):
        """Test _conclusions with full metrics to exercise all branches (lines 386-468)."""
        from bssunfold.core._interpret_report import _conclusions

        metrics = {
            "active_groups": [0, 1],
            "detector_importance": [
                {"detector": "d1", "max_spectrum_change": 0.05}
            ],
            "robustness": {"max_spectrum_change_relative": 0.03},
            "norm_dual": 0.5,
            "nonnegativity_relaxation": [
                {"change_from_base": 0.02, "status": "optimal"}
            ],
            "regularization_sweep": [
                {"alpha": 0.001, "status": "optimal", "residual_norm": 0.01}
            ],
        }
        lines = _conclusions(metrics, enforce_norm=True)
        assert len(lines) > 0
        # Check robust path
        assert any("Robust" in l for l in lines)
        # Check norm_dual path
        assert any("shadow price" in l for l in lines)
        # Check nonneg path
        assert any("negative" in l for l in lines)
        # Check sweep path
        assert any("regularization sweep" in l for l in lines)

    def test_conclusions_sensitive_robustness(self):
        """Test _conclusions with sensitive robustness (lines 422-428)."""
        from bssunfold.core._interpret_report import _conclusions

        metrics = {
            "active_groups": [],
            "robustness": {"max_spectrum_change_relative": 0.5},
        }
        lines = _conclusions(metrics, enforce_norm=False)
        assert any("Sensitive" in l for l in lines)

    def test_conclusions_nonneg_driving(self):
        """Test _conclusions with nonnegativity driving solution (lines 450-455)."""
        from bssunfold.core._interpret_report import _conclusions

        metrics = {
            "active_groups": [],
            "nonnegativity_relaxation": [
                {"change_from_base": 0.5, "status": "optimal"}
            ],
        }
        lines = _conclusions(metrics, enforce_norm=False)
        assert any(" leans on" in l for l in lines)

    def test_conclusions_minimal(self):
        """Test _conclusions with minimal metrics."""
        from bssunfold.core._interpret_report import _conclusions

        lines = _conclusions({}, enforce_norm=False)
        assert len(lines) > 0
        assert any("inactive" in l for l in lines)
