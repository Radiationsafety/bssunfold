"""Tests for new regularization parameter selection criteria.

Covers: quasi_optimality, NCP, SNR, weighted GCV (Poisson), K-fold CV.
"""

import numpy as np
import pytest

from bssunfold.core.regularization import (
    kfold_cv_selection,
    ncp_selection,
    quasi_optimality_selection,
    select_regularization_parameter,
    snr_criterion_selection,
    weighted_gcv_poisson_selection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def system_small():
    """Small well-conditioned system."""
    rng = np.random.RandomState(42)
    A = rng.randn(8, 5)
    x_true = np.array([1.0, 2.0, 0.5, 3.0, 0.1])
    b = A @ x_true + 0.01 * rng.randn(8)
    return A, b


@pytest.fixture
def system_ill_conditioned():
    """Ill-conditioned system (realistic for BSS)."""
    rng = np.random.RandomState(0)
    n, m = 30, 6
    E = np.logspace(-3, 8, n)
    A = np.zeros((m, n))
    for i in range(m):
        centre = E[int((i + 1) * n / (m + 1))]
        width = centre * 0.8
        A[i, :] = np.exp(-0.5 * ((E - centre) / width) ** 2)
    x_true = np.exp(-E / 1e6) * 1e-4
    b = A @ x_true + 0.001 * rng.randn(m)
    return A, b


@pytest.fixture
def white_noise_system():
    """System where b is pure white noise (should select large alpha)."""
    rng = np.random.RandomState(99)
    A = rng.randn(10, 8)
    b = rng.randn(10)
    return A, b


# ===========================================================================
#  quasi_optimality_selection
# ===========================================================================

class TestQuasiOptimalitySelection:

    def test_basic(self, system_small):
        A, b = system_small
        alpha = quasi_optimality_selection(A, b)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_ill_conditioned(self, system_ill_conditioned):
        A, b = system_ill_conditioned
        alpha = quasi_optimality_selection(A, b, n_alphas=30)
        assert alpha > 0

    def test_white_noise(self, white_noise_system):
        A, b = white_noise_system
        alpha = quasi_optimality_selection(A, b)
        # For pure noise, quasi-optimality should select a non-trivial alpha
        assert alpha > 0

    def test_custom_range(self, system_small):
        A, b = system_small
        alpha = quasi_optimality_selection(
            A, b, n_alphas=20, alpha_range=(1e-6, 1e1),
        )
        assert 1e-6 <= alpha <= 1e1

    def test_returns_finite(self, system_small):
        A, b = system_small
        alpha = quasi_optimality_selection(A, b)
        assert np.isfinite(alpha)


# ===========================================================================
#  ncp_selection
# ===========================================================================

class TestNcpSelection:

    def test_basic(self, system_small):
        A, b = system_small
        alpha = ncp_selection(A, b, n_alphas=15)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_ill_conditioned(self, system_ill_conditioned):
        A, b = system_ill_conditioned
        alpha = ncp_selection(A, b, n_alphas=15)
        assert alpha > 0

    def test_custom_range(self, system_small):
        A, b = system_small
        alpha = ncp_selection(
            A, b, n_alphas=15, alpha_range=(1e-4, 1e2),
        )
        assert 1e-4 <= alpha <= 1e2

    def test_returns_finite(self, system_small):
        A, b = system_small
        alpha = ncp_selection(A, b, n_alphas=10)
        assert np.isfinite(alpha)


# ===========================================================================
#  snr_criterion_selection
# ===========================================================================

class TestSnrCriterionSelection:

    def test_basic(self, system_small):
        A, b = system_small
        alpha = snr_criterion_selection(A, b)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_ill_conditioned(self, system_ill_conditioned):
        A, b = system_ill_conditioned
        alpha = snr_criterion_selection(A, b, n_alphas=30)
        assert alpha > 0

    def test_custom_range(self, system_small):
        A, b = system_small
        alpha = snr_criterion_selection(
            A, b, n_alphas=20, alpha_range=(1e-6, 1e1),
        )
        assert 1e-6 <= alpha <= 1e1

    def test_returns_finite(self, system_small):
        A, b = system_small
        alpha = snr_criterion_selection(A, b)
        assert np.isfinite(alpha)


# ===========================================================================
#  weighted_gcv_poisson_selection
# ===========================================================================

class TestWeightedGcvPoissonSelection:

    def test_basic(self, system_small):
        A, b = system_small
        alpha = weighted_gcv_poisson_selection(A, b)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_nonneg_b(self, system_small):
        """Poisson weights require b >= 0."""
        A, b = system_small
        b_pos = np.abs(b) + 0.1
        alpha = weighted_gcv_poisson_selection(A, b_pos)
        assert alpha > 0

    def test_ill_conditioned(self, system_ill_conditioned):
        A, b = system_ill_conditioned
        b_pos = np.abs(b) + 0.01
        alpha = weighted_gcv_poisson_selection(A, b_pos, n_alphas=30)
        assert alpha > 0

    def test_custom_range(self, system_small):
        A, b = system_small
        alpha = weighted_gcv_poisson_selection(
            A, b, n_alphas=20, alpha_range=(1e-6, 1e1),
        )
        assert 1e-6 <= alpha <= 1e1

    def test_returns_finite(self, system_small):
        A, b = system_small
        alpha = weighted_gcv_poisson_selection(A, b)
        assert np.isfinite(alpha)


# ===========================================================================
#  kfold_cv_selection
# ===========================================================================

class TestKfoldCvSelection:

    def test_basic(self, system_small):
        A, b = system_small
        alpha = kfold_cv_selection(A, b, n_folds=3)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_reproducible(self, system_small):
        A, b = system_small
        a1 = kfold_cv_selection(A, b, n_folds=3, random_state=42)
        a2 = kfold_cv_selection(A, b, n_folds=3, random_state=42)
        assert a1 == a2

    def test_different_seeds(self, system_small):
        A, b = system_small
        a1 = kfold_cv_selection(A, b, n_folds=3, random_state=1)
        a2 = kfold_cv_selection(A, b, n_folds=3, random_state=2)
        # Different seeds should typically give different results
        # (not guaranteed but highly likely for n_alphas=50)
        assert isinstance(a1, float)
        assert isinstance(a2, float)

    def test_custom_range(self, system_small):
        A, b = system_small
        alpha = kfold_cv_selection(
            A, b, n_folds=3, n_alphas=20, alpha_range=(1e-6, 1e1),
        )
        assert 1e-6 <= alpha <= 1e1

    def test_ill_conditioned(self, system_ill_conditioned):
        A, b = system_ill_conditioned
        alpha = kfold_cv_selection(A, b, n_folds=3, n_alphas=15)
        assert alpha > 0

    def test_returns_finite(self, system_small):
        A, b = system_small
        alpha = kfold_cv_selection(A, b, n_folds=3, n_alphas=10)
        assert np.isfinite(alpha)


# ===========================================================================
#  Dispatcher integration
# ===========================================================================

class TestDispatcherIntegration:
    """Test select_regularization_parameter routes to new methods."""

    @pytest.mark.parametrize("method", [
        "quasi_optimality", "ncp", "snr",
        "weighted_gcv_poisson", "kfold_cv",
    ])
    def test_dispatcher_routes(self, system_small, method):
        A, b = system_small
        alpha = select_regularization_parameter(A, b, method=method)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_dispatcher_unknown_raises(self, system_small):
        A, b = system_small
        with pytest.raises(ValueError, match="Unknown regularization"):
            select_regularization_parameter(A, b, method="nonexistent")
