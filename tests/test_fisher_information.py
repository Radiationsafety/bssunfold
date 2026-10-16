"""Tests for the minimum Fisher information (MFI) unfolding.

Covers the iterative weight-update solver ``solve_fisher_information``
(validation, alpha=0 least-squares equivalence, non-negativity, weight
capping at W_max, smoothing effect of the Fisher penalty, derivative
orders and grid-aware operators), and the ``Detector.
unfold_fisher_information`` integration including the standard result
keys and Monte-Carlo errors.

Reference: Miksuta-Michalik et al., Plasma Phys. Control. Fusion 68
(2026) 105001, eqs. (3)-(6).
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_fisher_information, unfold_fisher_information

# ---------------------------------------------------------------------------
# Synthetic problem helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic():
    """Small dense synthetic system with a peaked positive truth."""
    rng = np.random.default_rng(42)
    n, m = 25, 6
    A = rng.random((m, n)) + 0.1
    x_true = np.zeros(n)
    x_true[:8] = np.linspace(1.0, 0.2, 8)
    x_true[12:16] = 0.5
    b = A @ x_true
    return A, b, x_true


def curvature(x):
    return float(np.sum(np.diff(x) ** 2))


# ---------------------------------------------------------------------------
# solve_fisher_information — validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_alpha_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="alpha"):
            solve_fisher_information(A, b, alpha=-1.0)

    def test_bad_derivative_order_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="derivative_order"):
            solve_fisher_information(A, b, derivative_order=3)

    def test_nonpositive_w_max_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="w_max"):
            solve_fisher_information(A, b, w_max=0.0)

    def test_bad_max_iterations_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="max_iterations"):
            solve_fisher_information(A, b, max_iterations=0)

    def test_bad_tolerance_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="tolerance"):
            solve_fisher_information(A, b, tolerance=0.0)

    def test_bad_damping_raises(self, synthetic):
        A, b, _ = synthetic
        with pytest.raises(ValueError, match="damping"):
            solve_fisher_information(A, b, damping=0.0)
        with pytest.raises(ValueError, match="damping"):
            solve_fisher_information(A, b, damping=1.5)


# ---------------------------------------------------------------------------
# solve_fisher_information — mathematical properties
# ---------------------------------------------------------------------------

class TestSolverProperties:
    def test_alpha_zero_matches_least_squares(self, synthetic):
        """With no penalty the fixed point is the clipped LS solution."""
        A, b, _ = synthetic
        x, k, conv = solve_fisher_information(A, b, alpha=0.0)
        ls = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0.0)
        assert conv
        assert k <= 2
        np.testing.assert_allclose(x, ls, atol=1e-10)

    def test_returns_iteration_metadata(self, synthetic):
        A, b, _ = synthetic
        x, k, conv = solve_fisher_information(A, b, alpha=1e-3)
        assert isinstance(k, (int, np.integer))
        assert 1 <= k <= 50
        assert isinstance(conv, (bool, np.bool_))

    def test_result_nonnegative_finite(self, synthetic):
        A, b, _ = synthetic
        x, _, _ = solve_fisher_information(A, b, alpha=1e-2)
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0.0)
        assert x.shape == (A.shape[1],)

    def test_zero_initial_guess_does_not_diverge(self, synthetic):
        """A flat positive fallback replaces an all-zero x0 (eq. 6 spirit)."""
        A, b, _ = synthetic
        x, _, conv = solve_fisher_information(A, b, x0=np.zeros(A.shape[1]),
                                              alpha=1e-3)
        assert np.all(np.isfinite(x))
        assert conv

    def test_w_max_caps_weights_no_nan(self, synthetic):
        """Bins with zero flux use W_max; no inf/NaN may appear."""
        A, b, _ = synthetic
        x0 = np.zeros(A.shape[1])
        x0[:5] = 1.0  # most bins start at zero -> weight = w_max
        x, _, _ = solve_fisher_information(A, b, x0=x0, alpha=1e-2, w_max=1e6)
        assert np.all(np.isfinite(x))

    def test_fisher_penalty_smooths(self, synthetic):
        """Increasing alpha must not increase the solution's curvature."""
        A, b, _ = synthetic
        x_low, _, _ = solve_fisher_information(A, b, alpha=1e-4)
        x_high, _, _ = solve_fisher_information(A, b, alpha=10.0)
        assert curvature(x_high) <= curvature(x_low) + 1e-12

    def test_stationarity_of_fixed_point(self, synthetic):
        """At the converged fixed point one more weighted solve is a no-op:
        (AᵀA + α BᵀWB) x = Aᵀ b with W built from x itself."""
        A, b, _ = synthetic
        n = A.shape[1]
        x, k, conv = solve_fisher_information(A, b, alpha=1e-2,
                                              max_iterations=200,
                                              tolerance=1e-12)
        assert conv
        from bssunfold.core._matrix_utils import create_derivative_matrix
        B = np.asarray(create_derivative_matrix(n, 1).todense())
        absB = np.abs(B)
        row_flux = (absB @ x) / absB.sum(axis=1)
        w = 1.0 / np.maximum(row_flux, 1e-4 * np.max(x))
        resid = (A.T @ A + 1e-2 * (B.T @ (w[:, None] * B))) @ x - A.T @ b
        assert np.linalg.norm(resid) <= 1e-6 * np.linalg.norm(A.T @ b)

    def test_derivative_order_two_runs(self, synthetic):
        A, b, _ = synthetic
        x1, _, _ = solve_fisher_information(A, b, derivative_order=1)
        x2, _, _ = solve_fisher_information(A, b, derivative_order=2)
        assert x1.shape == x2.shape
        assert np.all(np.isfinite(x2))

    def test_grid_aware_operator_runs(self, synthetic):
        A, b, _ = synthetic
        n = A.shape[1]
        E = np.geomspace(0.025, 20.0, n)
        x_legacy, _, _ = solve_fisher_information(A, b, alpha=1e-2)
        x_lnE, _, _ = solve_fisher_information(A, b, alpha=1e-2, E_MeV=E)
        assert not np.allclose(x_legacy, x_lnE)
        assert np.all(np.isfinite(x_lnE))

    def test_damping_slows_but_converges(self, synthetic):
        A, b, _ = synthetic
        x_full, k_full, _ = solve_fisher_information(A, b, alpha=1e-2)
        x_damp, k_damp, conv = solve_fisher_information(
            A, b, alpha=1e-2, damping=0.5, max_iterations=500)
        assert conv
        assert k_damp >= k_full
        np.testing.assert_allclose(x_damp, x_full, rtol=1e-3, atol=1e-8)

    def test_max_iterations_respected(self, synthetic):
        A, b, _ = synthetic
        _, k, _ = solve_fisher_information(A, b, max_iterations=3,
                                           tolerance=1e-30)
        assert k == 3


# ---------------------------------------------------------------------------
# unfold_fisher_information / Detector integration
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    from bssunfold.constants import RF_GSF
    return Detector(RF_GSF)


@pytest.fixture
def readings(detector):
    return {name: float(i + 1) * 0.05
            for i, name in enumerate(detector.detector_names)}


class TestDetectorIntegration:
    REQUIRED_KEYS = ["energy", "spectrum", "spectrum_absolute",
                     "effective_readings", "residual", "residual_norm",
                     "method", "doserates"]

    def test_detector_method_returns_standard_dict(self, detector, readings):
        result = detector.unfold_fisher_information(readings)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"missing {key}"
        assert result["method"] == "Fisher information (MFI)"
        spec = result["spectrum"]
        assert spec.shape == (detector.n_energy_bins,)
        assert np.all(np.isfinite(spec)) and np.all(spec >= 0)

    def test_extra_metadata_keys(self, detector, readings):
        result = detector.unfold_fisher_information(
            readings, alpha=0.02, derivative_order=2, max_iterations=10)
        assert result["alpha"] == 0.02
        assert result["derivative_order"] == 2
        assert "iterations" in result and "converged" in result

    def test_initial_spectrum_accepted(self, detector, readings):
        x0 = np.full(detector.n_energy_bins, 0.5)
        result = detector.unfold_fisher_information(readings,
                                                    initial_spectrum=x0)
        assert result["spectrum"].shape == (detector.n_energy_bins,)

    def test_montecarlo_errors(self, detector, readings):
        result = detector.unfold_fisher_information(
            readings, calculate_errors=True, n_montecarlo=10,
            random_state=1)
        assert "spectrum_uncert_mean" in result

    def test_max_neutron_energy_mask(self, detector, readings):
        result = detector.unfold_fisher_information(
            readings, max_neutron_energy=10.0)
        assert result["spectrum"].shape == (detector.n_energy_bins,)

    def test_standalone_function_signature(self, detector, readings):
        n = detector.n_energy_bins
        result = unfold_fisher_information(
            detector_names=detector.detector_names,
            n_energy_bins=n,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=lambda r: "saved",
            readings=readings,
            ln_steps=detector.ln_steps,
            initial_spectrum=None,
            max_iterations=5,
        )
        assert np.all(np.isfinite(result["spectrum"]))
