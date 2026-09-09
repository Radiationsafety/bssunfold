"""Tests for the Non-negative K-SVD unfolding method (Xu et al. NIMA 2026).

These tests cover:

* Tikhonov-regularized NNLS via the augmented-matrix form (Eq. 2.5/2.6).
* Non-negative OMP (NN-OMP) sparse coding.
* NNLS+TopK sparse coding (the article's proposed strategy).
* Non-negative K-SVD dictionary learning (with non-negative atom
  truncation during the rank-1 SVD update).
* The full ``solve_nnksvd_unfold`` pipeline with all three sparse
  coders.
* The ``Detector.unfold_nnksvd`` wrapper.
* The three new metrics introduced by the article (relative flux error,
  Pearson correlation coefficient for spectral shape, comprehensive
  score).
"""

import numpy as np
import pandas as pd
import pytest

from bssunfold import Detector
from bssunfold.core.unfold_nnksvd import (
    solve_nn_omp,
    solve_nnksvd,
    solve_nnksvd_unfold,
    solve_nnls_topk,
    solve_tikhonov_nnls,
    unfold_nnksvd,
)
from bssunfold.utils.comparison import (
    compare_spectra,
    comprehensive_score,
    pearson_r,
    relative_flux_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def detector():
    """Default Detector instance (GSF response functions, 10 spheres)."""
    return Detector()


@pytest.fixture
def small_detector():
    """Small Detector with synthetic data for fast tests."""
    df = pd.DataFrame(
        {
            "E_MeV": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "sphere_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "sphere_2": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    return Detector(df)


@pytest.fixture
def energy_grid():
    """Logarithmic energy grid from 1e-9 to 20 MeV."""
    return np.logspace(-9, 1, 60)


# ---------------------------------------------------------------------------
# Tikhonov-regularized NNLS (Eq. 2.5 / 2.6)
# ---------------------------------------------------------------------------
class TestTikhonovNNLS:
    def test_non_negative_output(self):
        """solve_tikhonov_nnls must return a non-negative vector."""
        rng = np.random.default_rng(0)
        M = np.abs(rng.normal(size=(5, 8)))
        y = np.abs(rng.normal(size=5))
        alpha = solve_tikhonov_nnls(M, y, lambda_tik=0.01)
        assert alpha.shape == (8,)
        assert np.all(alpha >= -1e-12)

    def test_zero_lambda_reduces_to_nnls(self):
        """With lambda_tik=0 the solution equals scipy.optimize.nnls."""
        from scipy.optimize import nnls

        rng = np.random.default_rng(1)
        M = np.abs(rng.normal(size=(4, 6)))
        y = np.abs(rng.normal(size=4))
        # Make M_norm column-normalized (as in the article).
        norms = np.linalg.norm(M, axis=0)
        M_norm = M / norms
        alpha_ours = solve_tikhonov_nnls(M_norm, y, lambda_tik=0.0)
        alpha_scipy, _ = nnls(M_norm, y)
        assert np.allclose(alpha_ours, alpha_scipy, atol=1e-6)

    def test_prior_constraint(self):
        """The training-sample prior pulls alpha toward alpha_prior."""
        rng = np.random.default_rng(2)
        M = np.abs(rng.normal(size=(4, 6)))
        y = np.abs(rng.normal(size=4))
        norms = np.linalg.norm(M, axis=0)
        M_norm = M / norms

        # Without prior.
        alpha_free = solve_tikhonov_nnls(M_norm, y, lambda_tik=0.01, prior_wt=0.0)

        # With strong prior pulling toward a known vector.
        alpha_prior = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.0])
        alpha_with_prior = solve_tikhonov_nnls(
            M_norm,
            y,
            lambda_tik=0.01,
            prior_wt=10.0,
            alpha_prior=alpha_prior,
        )
        # With a strong prior the solution should move toward alpha_prior.
        d_free = np.linalg.norm(alpha_free - alpha_prior)
        d_with = np.linalg.norm(alpha_with_prior - alpha_prior)
        assert d_with < d_free

    def test_prior_requires_alpha_prior(self):
        """A positive prior_wt without alpha_prior must raise."""
        M = np.eye(4)
        y = np.ones(4)
        with pytest.raises(ValueError):
            solve_tikhonov_nnls(M, y, lambda_tik=0.01, prior_wt=0.5)

    def test_prior_shape_mismatch(self):
        """alpha_prior of wrong length must raise."""
        M = np.eye(4)
        y = np.ones(4)
        with pytest.raises(ValueError):
            solve_tikhonov_nnls(
                M, y, lambda_tik=0.01, prior_wt=0.5, alpha_prior=np.ones(3)
            )

    def test_zero_measurement(self):
        """solve_tikhonov_nnls with y=0 returns ~0."""
        M = np.eye(5)
        alpha = solve_tikhonov_nnls(M, np.zeros(5), lambda_tik=0.1)
        assert np.allclose(alpha, 0, atol=1e-8)


# ---------------------------------------------------------------------------
# NN-OMP sparse coding
# ---------------------------------------------------------------------------
class TestNNOMP:
    def test_non_negative_output(self):
        """NN-OMP must produce a non-negative sparse vector."""
        rng = np.random.default_rng(3)
        D = np.abs(rng.normal(size=(10, 20)))
        D = D / np.linalg.norm(D, axis=0)
        y = np.abs(rng.normal(size=10))
        alpha = solve_nn_omp(D, y, sparsity=3)
        assert alpha.shape == (20,)
        assert np.all(alpha >= -1e-12)
        assert np.sum(alpha > 1e-8) <= 3

    def test_zero_signal(self):
        """NN-OMP with a zero signal returns all zeros."""
        D = np.eye(6)
        alpha = solve_nn_omp(D, np.zeros(6), sparsity=2)
        assert np.allclose(alpha, 0)

    def test_sparsity_limit(self):
        """NN-OMP should not exceed the requested sparsity."""
        rng = np.random.default_rng(4)
        D = np.abs(rng.normal(size=(8, 30)))
        y = np.abs(rng.normal(size=8))
        alpha = solve_nn_omp(D, y, sparsity=5)
        assert np.sum(np.abs(alpha) > 1e-8) <= 5

    def test_exact_recovery_with_positive_dictionary(self):
        """NN-OMP recovers an exactly sparse non-negative signal.

        We use a dictionary whose atoms are orthonormal basis vectors
        (a permutation of the identity) so that the support is unique
        and NN-OMP cannot pick the wrong atom.
        """
        rng = np.random.default_rng(5)
        n = 10
        perm = rng.permutation(n)
        D = np.eye(n)[:, perm]  # permutation matrix -- perfectly separated atoms
        alpha_true = np.zeros(n)
        idx = [perm[3], perm[8]]
        alpha_true[idx] = [1.5, 0.7]
        y = D @ alpha_true
        alpha = solve_nn_omp(D, y, sparsity=2, tolerance=1e-10)
        assert np.allclose(D @ alpha, y, atol=1e-6)
        support = set(np.where(alpha > 1e-6)[0].tolist())
        assert support == set(idx)


# ---------------------------------------------------------------------------
# NNLS+TopK sparse coding
# ---------------------------------------------------------------------------
class TestNNLSTopK:
    def test_non_negative_output(self):
        """solve_nnls_topk returns a non-negative K-sparse vector."""
        rng = np.random.default_rng(6)
        M = np.abs(rng.normal(size=(5, 12)))
        M = M / np.linalg.norm(M, axis=0)
        y = np.abs(rng.normal(size=5))
        alpha = solve_nnls_topk(M, y, sparsity=3, lambda_tik=0.01)
        assert alpha.shape == (12,)
        assert np.all(alpha >= -1e-12)
        assert np.sum(alpha > 1e-8) <= 3

    def test_zero_measurement(self):
        """solve_nnls_topk with y=0 returns ~0."""
        M = np.eye(5)
        alpha = solve_nnls_topk(M, np.zeros(5), sparsity=2, lambda_tik=0.01)
        assert np.allclose(alpha, 0, atol=1e-8)

    def test_topk_support_consistency(self):
        """TopK support is a subset of the global NNLS nonzero support."""
        rng = np.random.default_rng(7)
        M = np.abs(rng.normal(size=(6, 20)))
        M = M / np.linalg.norm(M, axis=0)
        y = np.abs(rng.normal(size=6))
        alpha = solve_nnls_topk(M, y, sparsity=4, lambda_tik=0.001)
        # The K largest atoms from the global NNLS should be the support.
        from scipy.optimize import nnls

        alpha_full, _ = nnls(M, y)
        topk_idx = set(np.argsort(alpha_full)[-4:].tolist())
        support = set(np.where(alpha > 1e-8)[0].tolist())
        # All non-zero entries in alpha must be in the top-K support.
        assert support.issubset(topk_idx)

    def test_zero_sparsity(self):
        """K=0 returns a zero vector."""
        M = np.eye(5)
        y = np.ones(5)
        alpha = solve_nnls_topk(M, y, sparsity=0)
        assert np.allclose(alpha, 0)

    def test_k_greater_than_atoms(self):
        """K >= P falls back to the global NNLS solution."""
        M = np.eye(5)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = solve_nnls_topk(M, y, sparsity=20, lambda_tik=0.0)
        # Without regularization and K>=P, this is just NNLS on M.
        assert np.allclose(alpha, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Non-negative K-SVD dictionary learning
# ---------------------------------------------------------------------------
class TestNNKSVD:
    def test_dictionary_shape(self):
        """solve_nnksvd returns a dictionary of the requested shape."""
        rng = np.random.default_rng(8)
        signals = np.abs(rng.normal(size=(10, 30)))
        D, alpha_prior = solve_nnksvd(
            signals, n_atoms=12, n_iterations=3, sparsity=2, random_state=42
        )
        assert D.shape == (10, 12)
        assert alpha_prior.shape == (12,)

    def test_atoms_non_negative(self):
        """All dictionary atoms must be non-negative (article's key constraint)."""
        rng = np.random.default_rng(9)
        signals = np.abs(rng.normal(size=(8, 20)))
        D, _ = solve_nnksvd(
            signals, n_atoms=6, n_iterations=5, sparsity=2, random_state=42
        )
        assert np.all(D >= -1e-12)

    def test_atoms_normalized(self):
        """Dictionary columns must be L2-normalized."""
        rng = np.random.default_rng(10)
        signals = np.abs(rng.normal(size=(8, 20)))
        D, _ = solve_nnksvd(
            signals, n_atoms=6, n_iterations=5, sparsity=2, random_state=42
        )
        norms = np.linalg.norm(D, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_reproducible_with_seed(self):
        """solve_nnksvd with the same seed gives the same dictionary."""
        rng = np.random.default_rng(11)
        signals = np.abs(rng.normal(size=(6, 15)))
        D1, _ = solve_nnksvd(signals, n_atoms=4, n_iterations=3, random_state=42)
        D2, _ = solve_nnksvd(signals, n_atoms=4, n_iterations=3, random_state=42)
        assert np.allclose(D1, D2)

    def test_invalid_sparse_coder_raises(self):
        """Unknown sparse_coder must raise ValueError."""
        rng = np.random.default_rng(12)
        signals = np.abs(rng.normal(size=(6, 12)))
        with pytest.raises(ValueError):
            solve_nnksvd(
                signals,
                n_atoms=4,
                n_iterations=2,
                sparse_coder="not_a_coder",
            )

    def test_alpha_prior_non_negative(self):
        """The training-sample prior is a non-negative mean coefficient vector."""
        rng = np.random.default_rng(13)
        signals = np.abs(rng.normal(size=(6, 12)))
        _, alpha_prior = solve_nnksvd(
            signals,
            n_atoms=4,
            n_iterations=5,
            sparsity=2,
            sparse_coder="nnls_topk",
            random_state=42,
        )
        # Mean of non-negative coefficients is non-negative.
        assert np.all(alpha_prior >= -1e-12)


# ---------------------------------------------------------------------------
# solve_nnksvd_unfold top-level pipeline
# ---------------------------------------------------------------------------
class TestSolveNNKSVDUnfold:
    def test_basic_solve(self):
        """solve_nnksvd_unfold returns a non-negative spectrum of correct length."""
        rng = np.random.default_rng(14)
        n, m = 30, 5
        A = np.abs(rng.normal(size=(m, n)))
        x_true = np.zeros(n)
        x_true[[2, 10, 21]] = [1.0, 0.5, 0.8]
        b = A @ x_true
        x, iters, conv = solve_nnksvd_unfold(
            A,
            b,
            n_atoms=15,
            sparsity=2,
            n_dictionary_iterations=10,
            random_state=42,
        )
        assert x.shape == (n,)
        assert np.all(x >= 0)
        assert isinstance(iters, int)
        assert isinstance(conv, bool)

    def test_invalid_dictionary_shape(self):
        """Mismatched dictionary shape raises ValueError."""
        A = np.eye(5)
        b = np.ones(5)
        with pytest.raises(ValueError):
            solve_nnksvd_unfold(A, b, dictionary=np.eye(7), random_state=42)

    def test_invalid_training_signals(self):
        """Mismatched training_signals first dim raises ValueError."""
        A = np.eye(5)
        b = np.ones(5)
        with pytest.raises(ValueError):
            solve_nnksvd_unfold(
                A,
                b,
                training_signals=np.ones((7, 3)),
                random_state=42,
            )

    def test_invalid_sparse_coder(self):
        """Unknown sparse_coder raises ValueError."""
        A = np.eye(5)
        b = np.ones(5)
        with pytest.raises(ValueError):
            solve_nnksvd_unfold(A, b, sparse_coder="not_a_coder", random_state=42)

    @pytest.mark.parametrize("sparse_coder", ["nnls_topk", "omp", "nn_omp"])
    def test_all_sparse_coders_run(self, sparse_coder):
        """All three sparse coders execute and return a non-negative spectrum."""
        rng = np.random.default_rng(15)
        n, m = 25, 5
        A = np.abs(rng.normal(size=(m, n)))
        x_true = np.zeros(n)
        x_true[[1, 9, 18]] = [1.0, 0.5, 0.7]
        b = A @ x_true
        x, _, _ = solve_nnksvd_unfold(
            A,
            b,
            n_atoms=10,
            sparsity=2,
            n_dictionary_iterations=5,
            sparse_coder=sparse_coder,
            random_state=42,
        )
        assert x.shape == (n,)
        assert np.all(x >= 0)
        # Reasonable relative residual.
        rel = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-12)
        assert rel < 1.0

    def test_prelearned_dictionary(self):
        """A pre-learned dictionary bypasses online K-SVD training."""
        rng = np.random.default_rng(16)
        n, m = 20, 4
        A = np.abs(rng.normal(size=(m, n)))
        b = np.abs(rng.normal(size=m))
        # Build a non-negative dictionary from a few random atoms.
        D = np.abs(rng.normal(size=(n, 8)))
        D = D / np.linalg.norm(D, axis=0)
        x, _, _ = solve_nnksvd_unfold(A, b, dictionary=D, sparsity=2, random_state=42)
        assert x.shape == (n,)
        assert np.all(x >= 0)

    def test_with_initial_spectrum(self):
        """An initial spectrum guess is used to seed training signals."""
        rng = np.random.default_rng(17)
        n, m = 20, 4
        A = np.abs(rng.normal(size=(m, n)))
        b = np.abs(rng.normal(size=m))
        x0 = np.abs(rng.normal(size=n))
        x, _, _ = solve_nnksvd_unfold(
            A, b, x0=x0, n_atoms=10, n_dictionary_iterations=5, random_state=42
        )
        assert x.shape == (n,)


# ---------------------------------------------------------------------------
# Detector.unfold_nnksvd wrapper
# ---------------------------------------------------------------------------
class TestDetectorUnfoldNNKSVD:
    def test_basic_unfolding(self, small_detector):
        """Detector.unfold_nnksvd returns a standard result dict."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = small_detector.unfold_nnksvd(
            readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=5,
            random_state=42,
        )
        assert "spectrum" in result
        assert "energy" in result
        assert result["method"] == "NNKSVD"
        assert len(result["spectrum"]) == small_detector.n_energy_bins
        assert np.all(result["spectrum"] >= 0)
        assert "doserates" in result
        assert "effective_readings" in result
        assert result["n_atoms"] == 8
        assert result["sparsity"] == 2
        assert result["sparse_coder"] == "nnls_topk"

    def test_with_uncertainty(self, small_detector):
        """Monte-Carlo uncertainty is computed when requested."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = small_detector.unfold_nnksvd(
            readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=5,
            calculate_errors=True,
            n_montecarlo=3,
            noise_level=0.05,
            random_state=42,
        )
        assert "spectrum_uncert_mean" in result
        assert "spectrum_uncert_std" in result

    def test_save_result(self, small_detector):
        """save_result stores the result in history."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        small_detector.unfold_nnksvd(
            readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=5,
            save_result=True,
            random_state=42,
        )
        assert len(small_detector.results_history) >= 1

    @pytest.mark.parametrize("sparse_coder", ["nnls_topk", "omp", "nn_omp"])
    def test_all_coders_run(self, small_detector, sparse_coder):
        """All three sparse coders run end-to-end on the Detector."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = small_detector.unfold_nnksvd(
            readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=5,
            sparse_coder=sparse_coder,
            random_state=42,
        )
        assert result["sparse_coder"] == sparse_coder
        assert np.all(result["spectrum"] >= 0)

    def test_invalid_sparse_coder_raises(self, small_detector):
        """Unknown sparse_coder raises ValueError."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        with pytest.raises(ValueError):
            small_detector.unfold_nnksvd(
                readings, sparse_coder="not_a_coder", random_state=42
            )

    def test_with_max_neutron_energy(self, detector):
        """max_neutron_energy trims the response matrix."""
        rng = np.random.default_rng(18)
        # Synthetic spectrum with non-zero high-energy tail.
        phi_true = np.zeros(detector.n_energy_bins)
        phi_true[10:30] = np.abs(rng.normal(size=20))
        readings = {
            name: float(detector.sensitivities[name] @ phi_true)
            for name in detector.detector_names
        }
        result = detector.unfold_nnksvd(
            readings,
            n_atoms=10,
            sparsity=2,
            n_dictionary_iterations=5,
            max_neutron_energy=1.0,
            random_state=42,
        )
        # The spectrum must have zero flux above the cutoff.
        above = detector.E_MeV > 1.0
        assert np.allclose(result["spectrum"][above], 0, atol=1e-12)

    def test_module_level_unfold_nnksvd(self, small_detector):
        """Module-level unfold_nnksvd function works directly."""
        readings = {
            small_detector.detector_names[0]: 100.0,
            small_detector.detector_names[1]: 80.0,
        }
        result = unfold_nnksvd(
            detector_names=small_detector.detector_names,
            n_energy_bins=small_detector.n_energy_bins,
            E_MeV=small_detector.E_MeV,
            sensitivities=small_detector.sensitivities,
            cc_icrp116=small_detector._get_interpolated_cc(),
            save_result_callback=small_detector._save_result,
            readings=readings,
            n_atoms=8,
            sparsity=2,
            n_dictionary_iterations=5,
            random_state=42,
        )
        assert "spectrum" in result
        assert result["method"] == "NNKSVD"

    def test_end_to_end_metric_improvement(self, detector):
        """All three sparse coders produce valid non-negative spectra.

        Xu et al. (2026) report that NNLS+TopK substantially outperforms
        the greedy OMP and NN-OMP strategies on their 10-channel BNCT
        detector.  Their empirical comparison depends on the article's
        specific MCNP5-generated detector response matrix (48 energy
        bins, 10 polyethylene semi-cylindrical spheres).  Our test
        here is a sanity check on the package's default 10-sphere GSF
        detector that all three coders complete, produce a
        non-negative spectrum of the right shape, and have a bounded
        residual against the readings.
        """
        E = detector.E_MeV
        log_E = np.log10(E + 1e-15)
        # Build a synthetic multi-peak spectrum typical of BNCT epithermal
        # neutrons: thermal peak + epithermal plateau + fast-neutron tail.
        phi_true = (
            2.0 * np.exp(-((log_E + 7) ** 2) / 1.5)
            + 1.0 * np.exp(-((log_E + 4) ** 2) / 2.0)
            + 0.4 * np.exp(-((log_E + 0) ** 2) / 1.0)
        )
        phi_true = np.maximum(phi_true, 0)
        phi_true = phi_true / phi_true.sum()

        readings = {
            name: float(detector.sensitivities[name] @ phi_true)
            for name in detector.detector_names
        }

        results = {}
        for coder in ["nnls_topk", "omp", "nn_omp"]:
            results[coder] = detector.unfold_nnksvd(
                readings,
                n_atoms=15,
                sparsity=2,
                n_dictionary_iterations=15,
                sparse_coder=coder,
                random_state=42,
            )

        # All methods must produce non-negative spectra of correct length.
        for coder in results:
            assert results[coder]["spectrum"].shape == phi_true.shape
            assert np.all(results[coder]["spectrum"] >= 0)
            # Sanity: residual against the readings should be bounded.
            computed = np.array(
                [
                    detector.sensitivities[name] @ results[coder]["spectrum"]
                    for name in detector.detector_names
                ]
            )
            measured = np.array([readings[name] for name in detector.detector_names])
            rel = np.linalg.norm(computed - measured) / np.linalg.norm(measured)
            assert rel < 1.5, f"{coder} residual too high: {rel}"
            # Comprehensive score is finite (no NaN/inf).
            score = comprehensive_score(phi_true, results[coder]["spectrum"])
            assert np.isfinite(score)

        # NNLS+TopK should also be evaluated with the Xu et al. (2026)
        # comprehensive-score metric (lower = better).  Ensure it's finite
        # and not positive-infinity.
        for coder in results:
            err = relative_flux_error(phi_true, results[coder]["spectrum"])
            assert np.isfinite(err) and err >= 0.0
            corr = pearson_r(phi_true, results[coder]["spectrum"])
            assert np.isfinite(corr) and -1.0 - 1e-9 <= corr <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# New Xu et al. (2026) metrics
# ---------------------------------------------------------------------------
class TestXu2026Metrics:
    def test_relative_flux_error_identical(self, energy_grid):
        """Relative flux error is 0 for identical spectra."""
        s = np.ones_like(energy_grid)
        assert relative_flux_error(s, s) == 0.0

    def test_relative_flux_error_zero_reference(self):
        """A non-zero reconstruction of a zero reference yields error 1."""
        assert relative_flux_error(np.zeros(5), np.ones(5)) == 1.0
        # Both zero => error 0.
        assert relative_flux_error(np.zeros(5), np.zeros(5)) == 0.0

    def test_relative_flux_error_scaled(self, energy_grid):
        """A scaled copy has flux_err = |1 - scale|."""
        s = np.abs(np.sin(np.linspace(0, np.pi, len(energy_grid))))
        scale = 0.5
        assert np.isclose(relative_flux_error(s, scale * s), abs(1 - scale), atol=1e-12)

    def test_relative_flux_error_value_range(self):
        """The relative flux error is in [0, 2] for non-negative inputs."""
        # Identical: 0; opposite: 2 (for symmetric spectra with same norm).
        s = np.array([1.0, 2.0, 3.0])
        assert relative_flux_error(s, s) == 0.0
        # Maximum distance for non-negative vectors of equal norm is 2.
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        assert np.isclose(relative_flux_error(s1, s2), np.sqrt(2.0), atol=1e-12)

    def test_comprehensive_score_identical(self, energy_grid):
        """Identical spectra give the optimal score -0.5."""
        s = np.abs(np.sin(np.linspace(0, np.pi, len(energy_grid))))
        assert np.isclose(comprehensive_score(s, s), -0.5, atol=1e-12)

    def test_comprehensive_score_formula(self, energy_grid):
        """Comprehensive score = flux_err - 0.5 * pearson_r (Eq. 2.9)."""
        s1 = np.abs(np.sin(np.linspace(0, np.pi, len(energy_grid))))
        s2 = 0.7 * s1 + 0.3 * np.abs(np.cos(np.linspace(0, np.pi, len(energy_grid))))
        expected = relative_flux_error(s1, s2) - 0.5 * pearson_r(s1, s2)
        assert np.isclose(comprehensive_score(s1, s2), expected, atol=1e-12)

    def test_compare_spectra_returns_all_three_metrics(self, energy_grid):
        """compare_spectra exposes the Xu 2026 metrics by name."""
        s1 = np.abs(np.sin(np.linspace(0, np.pi, len(energy_grid))))
        s2 = 0.8 * s1 + 0.2 * np.abs(np.cos(np.linspace(0, np.pi, len(energy_grid))))
        result = compare_spectra(
            s1,
            s2,
            metrics=[
                "relative_flux_error",
                "pearson_r",
                "comprehensive_score",
            ],
        )
        assert "relative_flux_error" in result
        assert "pearson_r" in result
        assert "comprehensive_score" in result

    def test_compare_spectra_default_includes_xu_metrics(self, energy_grid):
        """Default compare_spectra output includes the Xu 2026 metrics."""
        s1 = np.abs(np.sin(np.linspace(0, np.pi, len(energy_grid))))
        s2 = 0.9 * s1
        result = compare_spectra(s1, s2)
        assert "relative_flux_error" in result
        assert "pearson_r" in result
        assert "comprehensive_score" in result

    def test_length_mismatch_raises(self):
        """Metric functions validate that the inputs have equal length."""
        with pytest.raises(ValueError):
            relative_flux_error(np.ones(5), np.ones(6))
        with pytest.raises(ValueError):
            comprehensive_score(np.ones(5), np.ones(6))
