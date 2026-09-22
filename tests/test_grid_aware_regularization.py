"""Tests for grid-aware regularization (derivatives w.r.t. ln E).

The grid-aware operators scale the finite differences by the per-bin
natural-logarithmic widths so that the smoothness penalty is (approximately)
invariant to the choice of the energy grid. Default (``E_MeV=None``) keeps
the legacy bin-index behaviour.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core._matrix_utils import (
    build_smoothness_penalty,
    create_derivative_matrix,
    make_regularization_operator,
)


class TestGridAwareDerivativeMatrix:
    """create_derivative_matrix with an E_MeV grid."""

    def test_uniform_grid_scales_like_analytic(self):
        """On a uniform dlnE grid the operator equals the legacy one / h."""
        n = 6
        h = 0.1 * np.log(10.0)
        E = np.exp(np.arange(n) * h)
        L_legacy = create_derivative_matrix(n, 1).toarray()
        L_grid = create_derivative_matrix(n, 1, E_MeV=E).toarray()
        assert np.allclose(L_grid, L_legacy / h)

    def test_first_derivative_values(self):
        """L @ phi approximates dphi/dlnE on a non-uniform grid."""
        # phi = a + b * ln(E) is exactly linear in ln E
        E = np.array([0.1, 0.3, 1.0, 4.0, 20.0])  # non-uniform
        a, b = 2.0, 3.0
        phi = a + b * np.log(E)
        L = create_derivative_matrix(5, 1, E_MeV=E).toarray()
        d = L @ phi
        assert np.allclose(d, b, rtol=1e-12)

    def test_second_derivative_uniform_reduction(self):
        """Non-uniform order-2 reduces to [1,-2,1]/h^2 for a uniform grid."""
        n = 6
        h = 0.2 * np.log(10.0)
        E = np.exp(np.arange(n) * h)
        L_legacy = create_derivative_matrix(n, 2).toarray()
        L_grid = create_derivative_matrix(n, 2, E_MeV=E).toarray()
        assert np.allclose(L_grid, L_legacy / h**2)

    def test_second_derivative_quadratic_exact(self):
        """phi = a + b*lnE + c*lnE^2 has exact second derivative 2c."""
        E = np.array([0.05, 0.4, 2.0, 9.0, 60.0, 400.0])
        a, b, c = 1.0, -2.0, 0.5
        u = np.log(E)
        phi = a + b * u + c * u**2
        L = create_derivative_matrix(6, 2, E_MeV=E).toarray()
        d2 = L @ phi
        assert np.allclose(d2, 2.0 * c, rtol=1e-12)

    def test_legacy_default_unchanged(self):
        """E_MeV=None reproduces the legacy plain differences."""
        n = 5
        L1 = create_derivative_matrix(n, 1).toarray()
        assert np.allclose(
            L1, np.eye(n - 1, n, k=1) - np.eye(n - 1, n, k=0)
        )
        L2 = create_derivative_matrix(n, 2).toarray()
        expected = np.eye(n - 2, n, k=2) - 2 * np.eye(n - 2, n, k=1) + np.eye(
            n - 2, n, k=0
        )
        assert np.allclose(L2, expected)

    def test_invalid_grid_rejected(self):
        with pytest.raises(ValueError, match="strictly"):
            create_derivative_matrix(3, 1, E_MeV=np.array([1.0, 0.5, 2.0]))
        with pytest.raises(ValueError, match="strictly"):
            create_derivative_matrix(3, 1, E_MeV=np.array([-1.0, 0.5, 2.0]))

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="length"):
            create_derivative_matrix(4, 1, E_MeV=np.array([1.0, 2.0]))

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="order"):
            create_derivative_matrix(4, 3)


class TestGridAwarePenalty:
    """build_smoothness_penalty / make_regularization_operator with E_MeV."""

    def test_penalty_invariance_across_grids(self):
        """Same smooth physical spectrum on two grids -> similar penalty.

        A lethargy-smooth spectrum sampled on a 0.1-decade and a 0.2-decade
        grid yields nearly the same weighted roughness, while the legacy
        bin-index penalty differs by the grid factor.
        """
        # same physical dlnE-range, different sampling
        E_fine = np.exp(np.arange(30) * 0.1 * np.log(10.0)) * 1e-6
        E_coarse = np.exp(np.arange(15) * 0.2 * np.log(10.0)) * 1e-6

        def phi(E):
            u = np.log(E / 1e-3)
            return np.exp(-((u / 4.0) ** 2))  # wide smooth bump in lnE

        def qform(P, x):
            return float(x @ (P @ x))

        pen_fine = qform(
            build_smoothness_penalty(30, 1.0, 2, E_MeV=E_fine), phi(E_fine)
        )
        pen_coarse = qform(
            build_smoothness_penalty(15, 1.0, 2, E_MeV=E_coarse), phi(E_coarse)
        )
        # grid-aware: nearly grid-independent (within ~30%), while the legacy
        # bin-index penalty is off by the h^-4 grid factor (~6x here)
        assert pen_fine > 0 and pen_coarse > 0
        assert pen_fine == pytest.approx(pen_coarse, rel=0.35)

        # legacy penalty scales with the grid (h^-4 for order 2 net of the
        # 2x term count): the coarse grid penalizes the same spectrum ~8x more
        leg_fine = qform(build_smoothness_penalty(30, 1.0, 2), phi(E_fine))
        leg_coarse = qform(build_smoothness_penalty(15, 1.0, 2), phi(E_coarse))
        assert leg_coarse / leg_fine > 5.0

    def test_operator_helpers_pass_through(self):
        E = np.exp(np.arange(5) * 0.2 * np.log(10.0))
        L = make_regularization_operator(5, 2, identity_for_zero=False, E_MeV=E)
        assert L is not None and L.shape == (3, 5)
        L1 = make_regularization_operator(5, 1, E_MeV=E)
        assert L1.shape == (4, 5)

    def test_penalty_none_for_order_zero(self):
        assert build_smoothness_penalty(5, 1.0, 0) is None
        assert build_smoothness_penalty(5, 1.0, 0, E_MeV=np.ones(5)) is None


class TestGridAwareSolvers:
    """grid_aware opt-in on the CGLS / GKS Detector methods."""

    @pytest.fixture
    def detector(self):
        return Detector()

    @pytest.fixture
    def readings(self):
        return {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

    def test_cgls_grid_aware_smoke(self, detector, readings):
        result = detector.unfold_cgls(
            readings,
            regularization=1e-3,
            smoothness_order=2,
            grid_aware=True,
            max_iterations=30,
        )
        assert result["grid_aware"] is True
        assert np.all(np.isfinite(result["spectrum"]))
        assert np.all(result["spectrum"] >= 0)

    def test_cgls_default_not_grid_aware(self, detector, readings):
        result = detector.unfold_cgls(
            readings,
            regularization=1e-3,
            smoothness_order=2,
            max_iterations=30,
        )
        assert result["grid_aware"] is False

    def test_cgls_grid_aware_changes_result(self, detector, readings):
        kw = dict(
            regularization=1e-2,
            smoothness_order=1,
            max_iterations=50,
            save_result=False,
        )
        r_legacy = detector.unfold_cgls(readings, **kw)
        r_grid = detector.unfold_cgls(readings, grid_aware=True, **kw)
        assert not np.allclose(r_legacy["spectrum"], r_grid["spectrum"])

    def test_gks_grid_aware_smoke(self, detector, readings):
        result = detector.unfold_gks(readings, smoothness_order=2, grid_aware=True)
        assert result["grid_aware"] is True
        assert np.all(np.isfinite(result["spectrum"]))

    def test_gks_default_not_grid_aware(self, detector, readings):
        result = detector.unfold_gks(readings, smoothness_order=2)
        assert result["grid_aware"] is False
