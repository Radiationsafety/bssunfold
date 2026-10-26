"""Tests for discrepancy-principle stopping (iterative regularization).

Covers:
- regularization._discrepancy_target (explicit and estimated noise level).
- solve_landweber / solve_kaczmarz with criterion="discrepancy":
  early stopping at the noise level, warm starts, invalid parameters,
  numpy-fallback vs numba-kernel equivalence, backward compatibility of
  the default "tolerance" criterion.
- unfold_landweber / unfold_kaczmarz wrappers and Detector methods
  (stopping_criterion/tau metadata in the result).
"""

from unittest.mock import patch

import numpy as np
import pytest

import bssunfold.core._numba_jit as numba_mod
from bssunfold.core import solve_kaczmarz, solve_landweber
from bssunfold.core.regularization import _discrepancy_target


@pytest.fixture
def noisy_problem():
    """Overdetermined synthetic problem with ~1% relative noise.

    The noise is scaled so that ``||noise|| ~ noise_level * ||b_clean||``,
    matching the semantics of the discrepancy target
    ``delta = noise_level * ||b||``.
    """
    rng = np.random.default_rng(42)
    m, n = 30, 12
    A = rng.standard_normal((m, n)) * 0.5 + 0.5
    x_true = np.abs(rng.standard_normal(n)) + 0.1
    b_clean = A @ x_true
    noise_level = 0.01
    noise = noise_level * np.linalg.norm(b_clean) * rng.standard_normal(m) / np.sqrt(m)
    b = b_clean + noise
    return A, b, np.zeros(n), noise_level


class TestDiscrepancyTarget:
    """_discrepancy_target helper."""

    def test_explicit_noise_level(self, noisy_problem):
        A, b, _, noise_level = noisy_problem
        target = _discrepancy_target(A, b, noise_level=noise_level, tau=1.1)
        assert np.isclose(target, 1.1 * noise_level * np.linalg.norm(b))

    def test_tau_scaling(self, noisy_problem):
        A, b, _, noise_level = noisy_problem
        t1 = _discrepancy_target(A, b, noise_level=noise_level, tau=1.0)
        t2 = _discrepancy_target(A, b, noise_level=noise_level, tau=2.0)
        assert np.isclose(t2, 2.0 * t1)

    def test_estimated_noise_level(self, noisy_problem):
        A, b, _, _ = noisy_problem
        target = _discrepancy_target(A, b, noise_level=None)
        assert target > 0
        assert np.isfinite(target)

    def test_invalid_noise_level_raises(self, noisy_problem):
        A, b, _, _ = noisy_problem
        with pytest.raises(ValueError, match="noise_level must be positive"):
            _discrepancy_target(A, b, noise_level=0.0)

    def test_invalid_tau_raises(self, noisy_problem):
        A, b, _, noise_level = noisy_problem
        with pytest.raises(ValueError, match="tau must be positive"):
            _discrepancy_target(A, b, noise_level=noise_level, tau=-1.0)


class TestSolveLandweberDiscrepancy:
    """solve_landweber with criterion='discrepancy'."""

    def test_default_criterion_ignores_noise_level(self, noisy_problem):
        """Default 'tolerance' criterion does not stop at the noise level."""
        A, b, x0, noise_level = noisy_problem
        # The positivity-constrained iteration plateaus well above the
        # discrepancy target, so the default criterion runs to the cap...
        x, iterations, converged = solve_landweber(A, b, x0, max_iterations=100)
        assert not converged
        assert iterations == 100
        # ...while the discrepancy criterion stops much earlier.
        _, it_dp, c_dp = solve_landweber(
            A, b, x0, max_iterations=100, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        assert c_dp
        assert it_dp < 100

    def test_discrepancy_stops_at_noise_level(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        x, iterations, converged = solve_landweber(
            A, b, x0, max_iterations=10000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        target = 1.1 * noise_level * np.linalg.norm(b)
        assert converged
        assert iterations < 10000
        assert np.linalg.norm(A @ x - b) <= target

    def test_discrepancy_earlier_than_tolerance(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        _, it_tol, _ = solve_landweber(A, b, x0, max_iterations=10000)
        _, it_dp, _ = solve_landweber(
            A, b, x0, max_iterations=10000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        # Iterative regularization stops before the noise-fitting regime.
        assert it_dp < it_tol

    def test_estimated_noise_still_stops(self, noisy_problem):
        A, b, x0, _ = noisy_problem
        x, iterations, converged = solve_landweber(
            A, b, x0, max_iterations=10000, criterion="discrepancy"
        )
        assert converged
        assert iterations < 10000

    def test_warm_start_immediate_stop(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        x_star, _, _ = solve_landweber(
            A, b, x0, max_iterations=10000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        x, iterations, converged = solve_landweber(
            A, b, x_star, max_iterations=10, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        assert iterations == 0
        assert converged
        assert np.allclose(x, x_star)

    def test_invalid_criterion_raises(self, noisy_problem):
        A, b, x0, _ = noisy_problem
        with pytest.raises(ValueError, match="Unknown criterion"):
            solve_landweber(A, b, x0, criterion="magic")

    def test_zero_noise_level_raises(self, noisy_problem):
        A, b, x0, _ = noisy_problem
        with pytest.raises(ValueError, match="noise_level must be positive"):
            solve_landweber(A, b, x0, criterion="discrepancy", noise_level=0.0)

    def test_numpy_fallback_matches_default_path(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        kwargs = dict(
            max_iterations=5000, criterion="discrepancy", noise_level=noise_level
        )
        x_default, it_default, c_default = solve_landweber(A, b, x0, **kwargs)
        with patch.object(numba_mod, "NUMBA_AVAILABLE", False):
            x_fallback, it_fallback, c_fallback = solve_landweber(A, b, x0, **kwargs)
        assert it_default == it_fallback
        assert c_default == c_fallback
        assert np.allclose(x_default, x_fallback)

    def test_backward_compatible_default(self, noisy_problem):
        """Default-path results are unaffected by the numpy fallback change."""
        A, b, x0, _ = noisy_problem
        kwargs = dict(max_iterations=5000)
        x_default, it_default, c_default = solve_landweber(A, b, x0, **kwargs)
        with patch.object(numba_mod, "NUMBA_AVAILABLE", False):
            x_fallback, it_fallback, c_fallback = solve_landweber(A, b, x0, **kwargs)
        assert it_default == it_fallback
        assert np.allclose(x_default, x_fallback)


class TestSolveKaczmarzDiscrepancy:
    """solve_kaczmarz with criterion='discrepancy'."""

    def test_discrepancy_stops_at_noise_level(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        x, iterations, converged = solve_kaczmarz(
            A, b, x0, max_iterations=100000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        target = 1.1 * noise_level * np.linalg.norm(b)
        assert converged
        assert iterations < 100000
        assert np.linalg.norm(A @ x - b) <= target

    def test_discrepancy_stops_in_whole_sweeps(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        m = A.shape[0]
        _, iterations, _ = solve_kaczmarz(
            A, b, x0, max_iterations=100000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        assert iterations % m == 0

    def test_discrepancy_earlier_than_tolerance(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        _, it_tol, _ = solve_kaczmarz(A, b, x0, max_iterations=100000)
        _, it_dp, _ = solve_kaczmarz(
            A, b, x0, max_iterations=100000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        assert it_dp <= it_tol

    def test_warm_start_immediate_stop(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        x_star, _, _ = solve_kaczmarz(
            A, b, x0, max_iterations=100000, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        x, iterations, converged = solve_kaczmarz(
            A, b, x_star, max_iterations=5, criterion="discrepancy",
            noise_level=noise_level, tau=1.1,
        )
        assert iterations == 0
        assert converged
        assert np.allclose(x, x_star)

    def test_invalid_criterion_raises(self, noisy_problem):
        A, b, x0, _ = noisy_problem
        with pytest.raises(ValueError, match="Unknown criterion"):
            solve_kaczmarz(A, b, x0, criterion="magic")

    def test_numpy_fallback_matches_default_path(self, noisy_problem):
        A, b, x0, noise_level = noisy_problem
        kwargs = dict(
            max_iterations=100000, criterion="discrepancy", noise_level=noise_level
        )
        x_default, it_default, c_default = solve_kaczmarz(A, b, x0, **kwargs)
        with patch.object(numba_mod, "NUMBA_AVAILABLE", False):
            x_fallback, it_fallback, c_fallback = solve_kaczmarz(A, b, x0, **kwargs)
        assert it_default == it_fallback
        assert c_default == c_fallback
        assert np.allclose(x_default, x_fallback)


class TestUnfoldWrappersDiscrepancy:
    """unfold_* wrappers and Detector methods carry the criterion through."""

    @pytest.fixture
    def readings(self, detector):
        return {name: 100.0 for name in detector.detector_names}

    @pytest.mark.parametrize("method_name", ["unfold_landweber", "unfold_kaczmarz"])
    def test_detector_discrepancy_metadata(self, detector, readings, method_name):
        result = getattr(detector, method_name)(
            readings,
            max_iterations=200,
            criterion="discrepancy",
            tau=1.2,
            noise_level=0.01,
        )
        assert result["stopping_criterion"] == "discrepancy"
        assert result["tau"] == 1.2
        assert "iterations" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["conditioning"]["cond_2"] > 0

    @pytest.mark.parametrize("method_name", ["unfold_landweber", "unfold_kaczmarz"])
    def test_detector_default_metadata(self, detector, readings, method_name):
        result = getattr(detector, method_name)(readings, max_iterations=20)
        assert result["stopping_criterion"] == "tolerance"

    def test_detector_discrepancy_with_montecarlo(self, detector, readings):
        """Discrepancy criterion composes with Monte-Carlo uncertainty."""
        result = detector.unfold_landweber(
            readings,
            max_iterations=100,
            criterion="discrepancy",
            noise_level=0.01,
            calculate_errors=True,
            n_montecarlo=3,
        )
        assert result["stopping_criterion"] == "discrepancy"
        assert "spectrum_uncert_mean" in result
        assert np.all(result["spectrum"] >= 0)
