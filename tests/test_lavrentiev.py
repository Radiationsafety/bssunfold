"""Tests for Lavrentiev regularization with shift unfolding.

Covers ``solve_lavrentiev`` (the core solver) and
``Detector.unfold_lavrentiev`` (the Detector wrapper), including
rectangular response matrices, shift behaviour, and IAEA end-to-end
validation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bssunfold.core.unfold_lavrentiev import solve_lavrentiev


def _make_problem(m=8, n=20, seed=42):
    """Build a rectangular response matrix and synthetic readings."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.1, 1.0, size=(m, n))
    x_true = rng.uniform(0.5, 2.0, size=n)
    b = A @ x_true
    return A, b, x_true


class TestSolveLavrentiev:
    def test_recovers_truth_moderate_alpha(self):
        A, b, x_true = _make_problem()
        x = solve_lavrentiev(A, b, alpha=1e-2)
        rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert rel_err < 0.5

    def test_rectangular_matrix(self):
        A, b, x_true = _make_problem(m=5, n=30)
        x = solve_lavrentiev(A, b, alpha=1e-4)
        assert x.shape == (30,)
        assert np.all(np.isfinite(x))

    def test_square_matrix(self):
        A, b, x_true = _make_problem(m=15, n=15)
        x = solve_lavrentiev(A, b, alpha=1e-4)
        assert_allclose(x, x_true, rtol=1e-3, atol=1e-3)

    def test_zero_shift_matches_tikhonov(self):
        A, b, _ = _make_problem()
        x_lav = solve_lavrentiev(A, b, x0=np.zeros(A.shape[1]), alpha=0.5)
        ATA = A.T @ A
        ATb = A.T @ b
        x_tik = np.linalg.solve(ATA + 0.5 * np.eye(A.shape[1]), ATb)
        assert_allclose(x_lav, x_tik)

    def test_shift_pulls_solution_toward_x0(self):
        A, b, _ = _make_problem()
        x0 = np.ones(A.shape[1])
        x = solve_lavrentiev(A, b, x0=x0, alpha=1.0)
        x_no_shift = solve_lavrentiev(A, b, x0=np.zeros(A.shape[1]), alpha=1.0)
        dist_shift = np.linalg.norm(x - x0)
        dist_no_shift = np.linalg.norm(x_no_shift - x0)
        assert dist_shift < dist_no_shift

    def test_large_alpha_converges_to_x0(self):
        A, b, _ = _make_problem()
        x0 = np.ones(A.shape[1]) * 3.0
        x = solve_lavrentiev(A, b, x0=x0, alpha=1e10)
        assert_allclose(x, x0, rtol=1e-6)

    def test_small_alpha_approaches_least_squares(self):
        A, b, _ = _make_problem()
        x_lav = solve_lavrentiev(A, b, alpha=1e-6)
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        rel_err = np.linalg.norm(x_lav - x_ls) / np.linalg.norm(x_ls)
        assert rel_err < 0.1

    def test_invalid_alpha_raises(self):
        A, b, _ = _make_problem()
        with pytest.raises(ValueError):
            solve_lavrentiev(A, b, alpha=0.0)
        with pytest.raises(ValueError):
            solve_lavrentiev(A, b, alpha=-1.0)

    def test_invalid_x0_length_raises(self):
        A, b, _ = _make_problem()
        with pytest.raises(ValueError):
            solve_lavrentiev(A, b, x0=np.ones(A.shape[1] + 1))

    def test_none_x0_defaults_to_zero(self):
        A, b, _ = _make_problem()
        x_none = solve_lavrentiev(A, b, x0=None, alpha=1.0)
        x_zero = solve_lavrentiev(A, b, x0=np.zeros(A.shape[1]), alpha=1.0)
        assert_allclose(x_none, x_zero)


class TestDetectorLavrentiev:
    def test_detector_workflow(self, detector):
        readings = {
            name: float(val)
            for name, val in zip(
                detector.detector_names,
                np.linspace(1.0, 5.0, len(detector.detector_names)),
            )
        }
        result = detector.unfold_lavrentiev(readings, alpha=1.0)
        assert "spectrum" in result
        assert "doserates" in result
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        assert np.all(np.isfinite(result["spectrum"]))
        assert "alpha" in result

    def test_detector_with_initial_spectrum(self, detector):
        n = detector.n_energy_bins
        x0 = np.ones(n) * 0.5
        readings = {
            name: float(val)
            for name, val in zip(
                detector.detector_names,
                np.linspace(1.0, 5.0, len(detector.detector_names)),
            )
        }
        result = detector.unfold_lavrentiev(
            readings, initial_spectrum=x0, alpha=1.0
        )
        assert result["spectrum"].shape == (n,)
        assert np.all(np.isfinite(result["spectrum"]))

    def test_detector_max_neutron_energy(self, detector):
        readings = {
            name: float(val)
            for name, val in zip(
                detector.detector_names,
                np.linspace(1.0, 5.0, len(detector.detector_names)),
            )
        }
        result = detector.unfold_lavrentiev(
            readings, alpha=1.0, max_neutron_energy=0.005
        )
        assert np.any(result["spectrum"] == 0)

    def test_detector_nonnegative_spectrum(self, detector):
        readings = {
            name: float(val)
            for name, val in zip(
                detector.detector_names,
                np.linspace(1.0, 5.0, len(detector.detector_names)),
            )
        }
        result = detector.unfold_lavrentiev(readings, alpha=1.0)
        assert np.all(result["spectrum"] >= 0)

    def test_detector_reproducibility(self, detector):
        readings = {
            name: float(val)
            for name, val in zip(
                detector.detector_names,
                np.linspace(1.0, 5.0, len(detector.detector_names)),
            )
        }
        r1 = detector.unfold_lavrentiev(readings, alpha=1.0)
        r2 = detector.unfold_lavrentiev(readings, alpha=1.0)
        assert_allclose(r1["spectrum"], r2["spectrum"])


class TestIAEAEndToEnd:
    @pytest.fixture
    def iaea_reference(self):
        from pathlib import Path

        import pandas as pd

        csv_path = (
            Path(__file__).parent
            / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
        )
        return pd.read_csv(csv_path)

    def test_lavrentiev_on_ambe(self, detector, iaea_reference):
        ref = iaea_reference
        spec = {"E_MeV": ref["E_MeV"].values, "Phi": ref["ISO_ref_AmBe"].values}
        readings = detector.get_effective_readings_for_spectra(spec)
        result = detector.unfold_lavrentiev(readings, alpha=0.01)
        assert np.all(np.isfinite(result["spectrum"]))
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        rel_resid = result["residual_norm"] / np.linalg.norm(
            list(readings.values())
        )
        assert rel_resid < 0.15

    def test_lavrentiev_on_cf252(self, detector, iaea_reference):
        ref = iaea_reference
        spec = {"E_MeV": ref["E_MeV"].values, "Phi": ref["ISO_ref_Cf252"].values}
        readings = detector.get_effective_readings_for_spectra(spec)
        result = detector.unfold_lavrentiev(readings, alpha=0.01)
        assert np.all(np.isfinite(result["spectrum"]))
        rel_resid = result["residual_norm"] / np.linalg.norm(
            list(readings.values())
        )
        assert rel_resid < 0.15
