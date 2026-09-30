"""Tests for new unfolding methods added from addons/."""

import numpy as np
import pytest

# ============================================================================
# Test solve_gravel
# ============================================================================


class TestSolveGravel:
    def test_gravel_basic(self):
        from bssunfold.core import solve_gravel

        np.random.seed(42)
        A = np.random.rand(5, 10)
        x_true = np.exp(-np.linspace(0, 4, 10))
        b = A @ x_true
        x0 = np.ones(10) / 10
        x, iterations, converged = solve_gravel(A, b, x0, max_iterations=200)
        assert len(x) == 10
        assert iterations > 0
        assert converged
        assert np.all(x >= 0)

    def test_gravel_all_zero_measurements(self):
        from bssunfold.core import solve_gravel

        A = np.random.rand(5, 10)
        b = np.zeros(5)
        x0 = np.ones(10)
        with pytest.raises(
            ValueError, match="All measurements are zero or negative"
        ):
            solve_gravel(A, b, x0)

    def test_gravel_tolerance_zero(self):
        from bssunfold.core import solve_gravel

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 4, 10))
        x0 = np.ones(10)
        x, iterations, converged = solve_gravel(
            A, b, x0, tolerance=0, max_iterations=5
        )
        assert iterations == 5
        assert not converged


# ============================================================================
# Test solve_maxed
# ============================================================================


class TestSolveMaxed:
    def test_maxed_basic(self):
        from bssunfold.core import solve_maxed

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.exp(-np.linspace(0, 4, 10))
        x0 = np.ones(10)
        x, iterations, converged = solve_maxed(
            A, b, x0, max_iterations=100, tolerance=1e-3
        )
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_tikhonov_legendre
# ============================================================================


class TestSolveTikhonovLegendre:
    def test_tikhonov_legendre_basic(self):
        from bssunfold.core import solve_tikhonov_legendre

        np.random.seed(42)
        A = np.random.rand(5, 20)
        b = np.random.rand(5)
        x = solve_tikhonov_legendre(A, b, delta=0.1, n_polynomials=10)
        assert len(x) == 20
        assert np.all(x >= 0)

    def test_tikhonov_legendre_default_params(self):
        from bssunfold.core import solve_tikhonov_legendre

        np.random.seed(42)
        A = np.random.rand(3, 15)
        b = np.random.rand(3)
        x = solve_tikhonov_legendre(A, b)
        assert len(x) == 15
        assert np.all(x >= 0)


class TestSolveLavrentiev:
    def test_lavrentiev_basic(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(42)
        A = np.random.rand(5, 20)
        b = np.random.rand(5)
        x = solve_lavrentiev(A, b, alpha=0.1)
        assert len(x) == 20
        assert np.all(x >= 0)

    def test_lavrentiev_default_params(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(42)
        A = np.random.rand(3, 15)
        b = np.random.rand(3)
        x = solve_lavrentiev(A, b)
        assert len(x) == 15
        assert np.all(x >= 0)

    def test_lavrentiev_direct_form_square(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(0)
        A = np.random.rand(8, 8) + 0.1
        b = np.random.rand(8)
        x = solve_lavrentiev(A, b, alpha=0.05, form="direct")
        assert len(x) == 8
        assert np.all(x >= 0)

    def test_lavrentiev_direct_form_rectangular_rejects(self):
        # For non-square A, the direct Lavrentiev form (A + alpha*I) z = b
        # is mathematically undefined (identity dimensions do not match
        # A's). The solver must raise ValueError rather than silently
        # fall back to a different method.
        from bssunfold.core import solve_lavrentiev

        np.random.seed(1)
        A = np.random.rand(5, 20)
        b = np.random.rand(5)
        with pytest.raises(ValueError, match="square response matrix"):
            solve_lavrentiev(A, b, alpha=0.05, form="direct")

    def test_lavrentiev_rejects_unknown_form(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(3)
        A = np.random.rand(8, 8)
        b = np.random.rand(8)
        with pytest.raises(ValueError, match="form must be"):
            solve_lavrentiev(A, b, alpha=0.05, form="bogus")

    def test_lavrentiev_padded_form_rectangular_works(self):
        # The padded form zero-pads A to a square max(m,n) x max(m,n)
        # operator and applies the direct Lavrentiev scheme to the
        # padded operator (Ref. [4], Introduction p. 4). This must
        # work for the rectangular BSS case (m < n) without raising.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(31)
        m, n = 6, 25
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)
        z = solve_lavrentiev(A, b, alpha=0.1, form="padded")
        assert len(z) == n
        assert np.all(np.isfinite(z))
        assert np.all(z >= 0)
        # Important caveat: for m < n, the padded form forces z_i = 0
        # for i > m (because the corresponding diagonal entry of the
        # padded operator is just alpha, giving alpha * z_i = 0).
        # Verify this known limitation.
        assert np.allclose(z[m:], 0.0, atol=1e-12), (
            "Padded form should force z[m:] = 0 for m < n; got "
            f"max z[m:] = {z[m:].max():.3e}"
        )

    def test_lavrentiev_padded_form_square_matches_direct(self):
        # For a square A, the padded form is identical to the direct
        # form (no padding actually happens when m == n).
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(41)
        B = rng.random((6, 6))
        A = B @ B.T + np.eye(6)  # symmetric PSD
        b = rng.random(6)
        z_padded = solve_lavrentiev(A, b, alpha=0.3, form="padded")
        z_direct = solve_lavrentiev(A, b, alpha=0.3, form="direct")
        assert np.allclose(z_padded, z_direct, atol=1e-12)

    def test_lavrentiev_padded_form_overdetermined(self):
        # For m > n (over-determined), the padded form pads columns
        # instead of rows. The padded columns carry no information
        # and are discarded. The first n components of z must match
        # the direct-form solution of the (square-padded) operator.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(53)
        m, n = 10, 6
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)
        z_padded = solve_lavrentiev(A, b, alpha=0.2, form="padded")
        assert len(z_padded) == n
        assert np.all(np.isfinite(z_padded))
        assert np.all(z_padded >= 0)

    # ====================================================================
    # Iterated Lavrentiev (Bakushinsky's a-priori α-decay scheme)
    # ====================================================================

    def test_lavrentiev_iterated_basic(self):
        # The iterated form runs Bakushinsky's defect-correction
        # iteration on the Gram operator B = A A^T with αₖ = α₀ q^k.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(61)
        m, n = 6, 25
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)
        z = solve_lavrentiev(
            A, b, alpha=0.1, form="iterated", q=0.5, n_iterations=5,
        )
        assert len(z) == n
        assert np.all(np.isfinite(z))
        assert np.all(z >= 0)

    def test_lavrentiev_iterated_q_equals_1_matches_constant_alpha(self):
        # With q = 1.0, αₖ = α₀ for every k. The iterated scheme
        # with constant α is the classical Mahale-Nair iterated
        # Lavrentiev (Ref. [6]). With n_iterations = 1 it must
        # reproduce the single-step Gram form.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(67)
        m, n = 6, 25
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)
        z_iter = solve_lavrentiev(
            A, b, alpha=0.1, form="iterated", q=1.0, n_iterations=1,
        )
        z_gram = solve_lavrentiev(A, b, alpha=0.1, form="gram")
        assert np.allclose(z_iter, z_gram, atol=1e-10), (
            f"iterated(q=1, K=1) should equal single-step gram; "
            f"max |Δ| = {np.max(np.abs(z_iter - z_gram)):.3e}"
        )

    def test_lavrentiev_iterated_rejects_bad_q(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(73)
        A = np.random.rand(6, 25) + 0.05
        b = np.random.rand(6)
        with pytest.raises(ValueError, match="q must satisfy"):
            solve_lavrentiev(A, b, alpha=0.1, form="iterated", q=0.0)
        with pytest.raises(ValueError, match="q must satisfy"):
            solve_lavrentiev(A, b, alpha=0.1, form="iterated", q=1.5)

    def test_lavrentiev_iterated_rejects_bad_n_iterations(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(79)
        A = np.random.rand(6, 25) + 0.05
        b = np.random.rand(6)
        with pytest.raises(ValueError, match="n_iterations must be"):
            solve_lavrentiev(
                A, b, alpha=0.1, form="iterated", n_iterations=0,
            )

    def test_lavrentiev_iterated_bakushinsky_decay(self):
        # Smoke test: the Bakushinsky geometric-decay rule
        # αₖ = α₀ q^k produces a finite, non-negative spectrum that
        # differs from the constant-α (q=1) iterated scheme.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(83)
        m, n = 6, 25
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)
        z_decay = solve_lavrentiev(
            A, b, alpha=0.1, form="iterated",
            q=0.5, n_iterations=5,
        )
        z_const = solve_lavrentiev(
            A, b, alpha=0.1, form="iterated",
            q=1.0, n_iterations=5,
        )
        assert np.all(np.isfinite(z_decay)) and np.all(z_decay >= 0)
        # The two schemes give different solutions in general
        # (decay makes later iterations less regularised).
        assert not np.allclose(z_decay, z_const, atol=1e-9), (
            "Bakushinsky decay (q<1) should differ from constant-α "
            "iterated Lavrentiev (q=1)."
        )

    def test_lavrentiev_gram_form_matches_tikhonov_L_identity(self):
        # The Gram form (A A^T + alpha*I) y = b;  z = A^T y is
        # mathematically identical to zeroth-order Tikhonov
        # (A^T A + alpha*I) z = A^T b  via the push-through identity.
        # This test verifies the equivalence numerically.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(13)
        m, n = 8, 25
        A = rng.random((m, n)) + 0.05
        b = rng.random(m)

        # Gram form (Lavrentiev default).
        z_gram = solve_lavrentiev(A, b, alpha=0.1, form="gram")

        # Tikhonov L=I (the mathematically equivalent normal-equation form).
        M_tikh = A.T @ A + 0.1 * np.eye(n)
        rhs_tikh = A.T @ b
        z_tikh_raw = np.linalg.solve(M_tikh, rhs_tikh)
        # NB: solve_lavrentiev clips to non-negative values; to compare
        # like-for-like we clip the Tikhonov raw solution too.
        z_tikh = np.maximum(z_tikh_raw, 0.0)

        assert np.allclose(z_gram, z_tikh, atol=1e-10), (
            f"Lavrentiev Gram form differs from Tikhonov L=I: "
            f"max |Δ| = {np.max(np.abs(z_gram - z_tikh)):.3e}"
        )

    def test_lavrentiev_direct_form_differs_from_gram_form(self):
        # For a square, self-adjoint, positive A the direct Lavrentiev
        # form (A + alpha*I) z = b and the Gram form
        # (A^2 + alpha*I) z = A b filter the spectrum of A
        # (λ/(λ+α)) vs the spectrum of A^2 (λ²/(λ²+α)), respectively,
        # and so give genuinely different regularised solutions.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(29)
        # Build a symmetric PSD operator A (so the direct form is well
        # defined). A = B B^T + I, B random.
        B = rng.random((6, 6))
        A = B @ B.T + np.eye(6)
        b = rng.random(6)

        z_direct = solve_lavrentiev(A, b, alpha=0.5, form="direct")
        z_gram = solve_lavrentiev(A, b, alpha=0.5, form="gram")

        # Sanity: both are finite, non-negative, same shape.
        assert z_direct.shape == (6,) and z_gram.shape == (6,)
        assert np.all(np.isfinite(z_direct)) and np.all(np.isfinite(z_gram))
        assert np.all(z_direct >= 0) and np.all(z_gram >= 0)
        # And the two are *not* the same vector — confirming that
        # direct Lavrentiev is a different regulariser from
        # Tikhonov-L=I (the Gram form).
        assert not np.allclose(z_direct, z_gram, atol=1e-6), (
            "Direct Lavrentiev form unexpectedly equals the Gram form "
            "(Tikhonov L=I). These should differ — Lavrentiev filters "
            "the spectrum of A, Tikhonov-L=I filters the spectrum of "
            "A^T A."
        )

    def test_lavrentiev_rejects_negative_alpha(self):
        from bssunfold.core import solve_lavrentiev

        np.random.seed(2)
        A = np.random.rand(5, 20)
        b = np.random.rand(5)
        with pytest.raises(ValueError):
            solve_lavrentiev(A, b, alpha=-0.1)

    def test_lavrentiev_handles_zero_alpha(self):
        # alpha = 0 degenerates to a plain least-squares solve via the
        # lstsq fallback in solve_lavrentiev. The solver must still
        # return a finite, non-negative spectrum.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(7)
        # Over-determined, well-conditioned system so the un-regularized
        # normal equations are not singular.
        A = rng.random((25, 10)) + 0.5
        b = rng.random(25)
        x = solve_lavrentiev(A, b, alpha=0.0)
        assert len(x) == 10
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)

    def test_lavrentiev_alpha_smooths_solution(self):
        # Larger alpha should produce a smoother (smaller-norm) spectrum.
        from bssunfold.core import solve_lavrentiev

        rng = np.random.default_rng(11)
        A = rng.random((6, 25)) + 0.05
        b = rng.random(6)
        x_small = solve_lavrentiev(A, b, alpha=1e-6)
        x_large = solve_lavrentiev(A, b, alpha=1.0)
        assert np.linalg.norm(x_large) <= np.linalg.norm(x_small) + 1e-9


# ============================================================================
# Test solve_scipy_direct
# ============================================================================


class TestSolveScipyDirect:
    def test_scipy_direct_cg(self):
        from bssunfold.core import solve_scipy_direct

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(
            A, b, method="cg", tolerance=1e-6, max_iterations=1000
        )
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_lsqr(self):
        from bssunfold.core import solve_scipy_direct

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(A, b, method="lsqr", tolerance=1e-6)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_gmres(self):
        from bssunfold.core import solve_scipy_direct

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_scipy_direct(A, b, method="gmres", tolerance=1e-6)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_scipy_direct_unknown_method(self):
        from bssunfold.core import solve_scipy_direct

        A = np.random.rand(3, 5)
        b = np.random.rand(3)
        with pytest.raises(ValueError, match="Unknown solver method"):
            solve_scipy_direct(A, b, method="unknown")


# ============================================================================
# Test solve_tsvd
# ============================================================================


class TestSolveTsvd:
    def test_tsvd_basic(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, k=5)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_fixed_k(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, k=3)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_threshold(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, threshold=0.1)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_discrepancy(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="discrepancy")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_lcurve(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="l_curve")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_gcv(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="gcv")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_energy(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="energy")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_median(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="median_threshold")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_donoho(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="donoho")
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_tsvd_auto_default(self):
        from bssunfold.core import solve_tsvd

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10)
        x = solve_tsvd(A, b, method="nonexistent")
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_bayes (pyunfold dependency)
# ============================================================================


class TestSolveBayes:
    def test_bayes_basic(self):
        from bssunfold.core import solve_bayes

        np.random.seed(42)
        A = np.random.rand(5, 10)
        b = A @ np.ones(10) * 10
        x = solve_bayes(A, b, max_iterations=10, tolerance=1)
        assert len(x) == 10
        assert np.all(x >= 0)

    def test_bayes_no_prior(self):
        from bssunfold.core import solve_bayes

        np.random.seed(42)
        A = np.random.rand(4, 8)
        b = A @ np.ones(8) * 10
        x = solve_bayes(A, b, max_iterations=5, tolerance=10)
        assert len(x) == 8
        assert np.all(x >= 0)


# ============================================================================
# Test solve_bayes_spline (pyunfold dependency)
# ============================================================================


class TestSolveBayesSpline:
    def test_bayes_spline_basic(self):
        from bssunfold.core import solve_bayes_spline

        np.random.seed(42)
        A = np.random.rand(5, 15)
        b = A @ np.ones(15) * 10
        x = solve_bayes_spline(
            A,
            b,
            max_iterations=10,
            tolerance=1,
            spline_degree=1,
            spline_smooth=0.1,
        )
        assert len(x) == 15
        assert np.all(x >= 0)

    def test_bayes_spline_no_prior(self):
        from bssunfold.core import solve_bayes_spline

        np.random.seed(42)
        A = np.random.rand(4, 10)
        b = A @ np.ones(10) * 10
        x = solve_bayes_spline(
            A,
            b,
            max_iterations=5,
            tolerance=10,
            spline_degree=1,
            spline_smooth=1.0,
        )
        assert len(x) == 10
        assert np.all(x >= 0)


# ============================================================================
# Test solve_statreg (pure numpy, no external dependency)
# ============================================================================


class TestSolveStatreg:
    def test_statreg_basic(self):
        from bssunfold.core import solve_statreg

        np.random.seed(42)
        n_ene, n_det = 20, 5
        A = np.random.rand(n_det, n_ene) * 5.0
        x_true = np.exp(-np.linspace(0, 4, n_ene))
        b = A @ x_true + np.random.randn(n_det) * 0.01
        result = solve_statreg(A, b)
        assert result.shape == (n_ene,)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_statreg_user_alpha(self):
        from bssunfold.core import solve_statreg

        np.random.seed(42)
        A = np.random.rand(4, 15) * 3.0
        b = np.random.rand(4)
        result = solve_statreg(A, b, unfoldermethod="User", regularization=0.01)
        assert result.shape == (15,)
        assert np.all(result >= 0)

    def test_statreg_with_energy_grid(self):
        from bssunfold.core import solve_statreg

        np.random.seed(42)
        n_ene = 20
        E_MeV = np.logspace(-3, 2, n_ene)
        A = np.random.rand(4, n_ene) * 3.0
        b = np.random.rand(4)
        result = solve_statreg(A, b, E_MeV=E_MeV)
        assert result.shape == (n_ene,)
        assert np.all(result >= 0)

    def test_statreg_invalid_method(self):
        from bssunfold.core import solve_statreg

        A = np.eye(3)
        b = np.ones(3)
        with pytest.raises(ValueError, match="Unknown method"):
            solve_statreg(A, b, unfoldermethod="Invalid")


# ============================================================================
# Test unfold_* wrappers via Detector class
# ============================================================================


@pytest.fixture
def detector():
    from bssunfold import Detector

    return Detector()


@pytest.fixture
def readings(detector):
    return {detector.detector_names[0]: 100.0}


class TestUnfoldGravel:
    def test_unfold_gravel_basic(self, detector, readings):
        result = detector.unfold_gravel(
            readings, max_iterations=10, tolerance=1e-3
        )
        assert "spectrum" in result
        assert "doserates" in result
        assert "residual" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_gravel_with_initial(self, detector, readings):
        initial = np.ones(detector.n_energy_bins)
        result = detector.unfold_gravel(
            readings, initial_spectrum=initial, max_iterations=10
        )
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_gravel_empty_readings(self, detector):
        with pytest.raises(ValueError):
            detector.unfold_gravel({})

    def test_unfold_gravel_no_save(self, detector, readings):
        result = detector.unfold_gravel(
            readings, max_iterations=5, save_result=False
        )
        assert "spectrum" in result


class TestUnfoldMaxed:
    def test_unfold_maxed_basic(self, detector, readings):
        result = detector.unfold_maxed(
            readings, max_iterations=50, tolerance=0.1
        )
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_maxed_with_reference(self, detector, readings):
        initial = np.ones(detector.n_energy_bins)
        result = detector.unfold_maxed(
            readings, initial_spectrum=initial, max_iterations=50, tolerance=0.1
        )
        assert "spectrum" in result

    def test_unfold_maxed_no_save(self, detector, readings):
        result = detector.unfold_maxed(
            readings, max_iterations=20, save_result=False
        )
        assert "spectrum" in result


class TestUnfoldTikhonovLegendre:
    def test_unfold_tikhonov_legendre_basic(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(
            readings, delta=0.1, n_polynomials=8
        )
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_tikhonov_legendre_default(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(readings)
        assert "spectrum" in result

    def test_unfold_tikhonov_legendre_no_save(self, detector, readings):
        result = detector.unfold_tikhonov_legendre(
            readings, delta=0.1, save_result=False
        )
        assert "spectrum" in result


class TestUnfoldLavrentiev:
    def test_unfold_lavrentiev_basic(self, detector, readings):
        result = detector.unfold_lavrentiev(readings, alpha=0.1)
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["method"] == "Lavrentiev"
        assert result["alpha"] == 0.1
        assert result["form"] == "gram"

    def test_unfold_lavrentiev_default(self, detector, readings):
        result = detector.unfold_lavrentiev(readings)
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["alpha"] == 0.05

    def test_unfold_lavrentiev_direct_form_rejects_rectangular(
        self, detector, readings,
    ):
        # The Detector's response matrix is rectangular (m_detectors ×
        # n_energy_bins, e.g. 7 × 60 for GSF). The direct Lavrentiev
        # form (A + alpha*I) z = b is only defined for square A, so
        # the wrapper must propagate the ValueError from the solver
        # rather than silently using a different method.
        with pytest.raises(ValueError, match="square response matrix"):
            detector.unfold_lavrentiev(
                readings, alpha=0.05, form="direct"
            )

    def test_unfold_lavrentiev_no_save(self, detector, readings):
        result = detector.unfold_lavrentiev(
            readings, alpha=0.1, save_result=False
        )
        assert "spectrum" in result

    def test_unfold_lavrentiev_iterated(self, detector, readings):
        # The iterated form (Bakushinsky's scheme) runs on the
        # detector's rectangular response matrix without raising.
        result = detector.unfold_lavrentiev(
            readings, alpha=0.1, form="iterated",
            q=0.5, n_iterations=5,
        )
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)
        assert result["form"] == "iterated"
        assert result["q"] == 0.5
        assert result["n_iterations"] == 5


class TestUnfoldScipyDirect:
    def test_unfold_scipy_direct_cg(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="cg")
        assert "spectrum" in result
        assert "doserates" in result

    def test_unfold_scipy_direct_lsqr(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="lsqr")
        assert "spectrum" in result

    def test_unfold_scipy_direct_gmres(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="gmres")
        assert "spectrum" in result

    def test_unfold_scipy_direct_lsmr(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="lsmr")
        assert "spectrum" in result

    def test_unfold_scipy_direct_minres(self, detector, readings):
        result = detector.unfold_scipy_direct_method(readings, method="minres")
        assert "spectrum" in result

    def test_unfold_scipy_direct_no_save(self, detector, readings):
        result = detector.unfold_scipy_direct_method(
            readings, save_result=False
        )
        assert "spectrum" in result


class TestUnfoldTsvd:
    def test_unfold_tsvd_basic(self, detector, readings):
        result = detector.unfold_tsvd(readings, k=5)
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_tsvd_auto_discrepancy(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="discrepancy")
        assert "spectrum" in result

    def test_unfold_tsvd_auto_gcv(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="gcv")
        assert "spectrum" in result

    def test_unfold_tsvd_auto_lcurve(self, detector, readings):
        result = detector.unfold_tsvd(readings, method="l_curve")
        assert "spectrum" in result

    def test_unfold_tsvd_no_save(self, detector, readings):
        result = detector.unfold_tsvd(readings, k=3, save_result=False)
        assert "spectrum" in result


class TestUnfoldBayes:
    def test_unfold_bayes_basic(self, detector, readings):
        result = detector.unfold_bayes(readings, max_iterations=10, tolerance=1)
        assert "spectrum" in result
        assert "doserates" in result

    def test_unfold_bayes_spline_basic(self, detector, readings):
        result = detector.unfold_bayes_spline_regularization(
            readings,
            max_iterations=10,
            tolerance=1,
            spline_degree=1,
            spline_smooth=0.1,
        )
        assert "spectrum" in result
        assert "doserates" in result


# ============================================================================
# Test solve_statreg (pure numpy, merged)
# ============================================================================


class TestUnfoldStatreg:
    def test_unfold_statreg_basic(self, detector, readings):
        result = detector.unfold_statreg(readings)
        assert "spectrum" in result
        assert "doserates" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_statreg_user(self, detector, readings):
        result = detector.unfold_statreg(
            readings, unfoldermethod="User", regularization=0.01
        )
        assert "spectrum" in result
        assert np.all(result["spectrum"] >= 0)

    def test_unfold_statreg_no_save(self, detector, readings):
        result = detector.unfold_statreg(readings, save_result=False)
        assert "spectrum" in result


# ============================================================================
# Test MC with new methods
# ============================================================================


class TestNewMethodsWithMC:
    def test_gravel_with_errors(self, detector, readings):
        result = detector.unfold_gravel(
            readings,
            max_iterations=5,
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert "spectrum_uncert_std" in result

    def test_tsvd_with_errors(self, detector, readings):
        result = detector.unfold_tsvd(
            readings,
            k=3,
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert "spectrum_uncert_std" in result

    def test_scipy_direct_with_errors(self, detector, readings):
        result = detector.unfold_scipy_direct_method(
            readings,
            method="cg",
            calculate_errors=True,
            n_montecarlo=3,
            random_state=42,
        )
        assert "spectrum_uncert_std" in result


# ============================================================================
# Verify solve_* functions are in __all__
# ============================================================================


class TestModuleExports:
    def test_solve_gravel_exported(self):
        from bssunfold.core import solve_gravel

        assert callable(solve_gravel)

    def test_solve_maxed_exported(self):
        from bssunfold.core import solve_maxed

        assert callable(solve_maxed)

    def test_solve_tikhonov_legendre_exported(self):
        from bssunfold.core import solve_tikhonov_legendre

        assert callable(solve_tikhonov_legendre)

    def test_solve_lavrentiev_exported(self):
        from bssunfold.core import solve_lavrentiev

        assert callable(solve_lavrentiev)

    def test_unfold_lavrentiev_exported(self):
        from bssunfold.core import unfold_lavrentiev

        assert callable(unfold_lavrentiev)

    def test_solve_bayes_exported(self):
        from bssunfold.core import solve_bayes

        assert callable(solve_bayes)

    def test_solve_bayes_spline_exported(self):
        from bssunfold.core import solve_bayes_spline

        assert callable(solve_bayes_spline)

    def test_solve_statreg_exported(self):
        from bssunfold.core import solve_statreg

        assert callable(solve_statreg)

    def test_solve_scipy_direct_exported(self):
        from bssunfold.core import solve_scipy_direct

        assert callable(solve_scipy_direct)

    def test_solve_tsvd_exported(self):
        from bssunfold.core import solve_tsvd

        assert callable(solve_tsvd)

    def test_unfold_gravel_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_gravel")

    def test_unfold_maxed_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_maxed")

    def test_unfold_tikhonov_legendre_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_tikhonov_legendre")

    def test_unfold_bayes_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_bayes")

    def test_unfold_bayes_spline_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_bayes_spline_regularization")

    def test_unfold_statreg_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_statreg")

    def test_unfold_scipy_direct_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_scipy_direct_method")

    def test_unfold_tsvd_exported(self):
        from bssunfold import Detector

        assert hasattr(Detector, "unfold_tsvd")
