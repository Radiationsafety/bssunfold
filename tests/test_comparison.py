"""Tests for bssunfold.utils.comparison module and Detector.compare()."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from bssunfold import Detector, compare_spectra
from bssunfold.utils.comparison import (
    _normalize,
    _check_same_length,
    kl_divergence,
    cross_entropy,
    entropy,
    entropy_difference_percent,
    wasserstein_dist,
    energy_dist,
    kolmogorov_smirnov_stat,
    pearson_r,
    spearman_r,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mape,
    r2_score,
    max_error,
    median_absolute_error,
    cosine_similarity,
    mmd_rbf,
    chi_squared,
    g_test,
    freeman_tukey,
    cressie_read,
    anderson_darling,
    standardized_mean_difference,
    wilcoxon_test,
    mannwhitneyu_test,
    compare_multiple,
)
from bssunfold import utils as bss_utils


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def const_spectra():
    """Two identical constant spectra."""
    s1 = np.ones(50)
    s2 = np.ones(50)
    return s1, s2


@pytest.fixture
def diff_spectra():
    """Two distinct spectra (step vs ramp)."""
    s1 = np.concatenate([np.zeros(25), np.ones(25)])
    s2 = np.linspace(0, 1, 50)
    return s1, s2


@pytest.fixture
def sin_cos_spectra():
    """Spectra with known correlation structure."""
    x = np.linspace(0, 2 * np.pi, 100)
    s1 = np.sin(x) + 1
    s2 = np.cos(x) + 1
    return s1, s2


@pytest.fixture
def detector():
    """Default Detector instance."""
    return Detector()


@pytest.fixture
def sample_readings():
    """Sample detector readings for unfolding."""
    return {name: float(1.0 + i * 0.1) for i, name in enumerate(Detector().detector_names)}


# ─── Internal helpers ─────────────────────────────────────────────


class TestInternal:
    def test_normalize(self):
        p = np.array([1.0, 2.0, 3.0])
        pn = _normalize(p)
        assert_almost_equal(pn.sum(), 1.0)
        assert np.all(pn > 0)

    def test_normalize_zeros(self):
        p = np.zeros(5)
        pn = _normalize(p)
        assert_almost_equal(pn.sum(), 1.0)
        assert np.all(pn > 0)

    def test_normalize_negative(self):
        p = np.array([-1.0, 2.0])
        pn = _normalize(p)
        assert_almost_equal(pn.sum(), 1.0)
        assert np.all(pn > 0)

    def test_check_same_length_ok(self):
        _check_same_length(np.ones(3), np.ones(3))

    def test_check_same_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            _check_same_length(np.ones(3), np.ones(5))


# ─── Entropy metrics ──────────────────────────────────────────────


class TestEntropyMetrics:
    def test_kl_divergence_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(kl_divergence(s1, s2), 0.0, decimal=10)

    def test_kl_divergence_positive(self, diff_spectra):
        s1, s2 = diff_spectra
        assert kl_divergence(s1, s2) >= 0

    def test_kl_divergence_asymmetric(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != kl_qp

    def test_cross_entropy_identical(self, const_spectra):
        s1, s2 = const_spectra
        h1 = entropy(s1)
        ce = cross_entropy(s1, s2)
        assert_almost_equal(ce, h1, decimal=10)

    def test_entropy_identical(self, const_spectra):
        s1, _ = const_spectra
        e = entropy(s1)
        assert e >= 0

    def test_entropy_uniform(self):
        u = np.ones(10)
        e = entropy(u)
        assert_almost_equal(e, np.log(10), decimal=10)

    def test_entropy_difference_percent_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(entropy_difference_percent(s1, s2), 0.0, decimal=10)


# ─── Distribution distances ───────────────────────────────────────


class TestDistributionDistances:
    def test_wasserstein_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(wasserstein_dist(s1, s2), 0.0, decimal=10)

    def test_wasserstein_positive(self, diff_spectra):
        s1, s2 = diff_spectra
        assert wasserstein_dist(s1, s2) >= 0

    def test_energy_distance_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(energy_dist(s1, s2), 0.0, decimal=10)

    def test_ks_stat_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(kolmogorov_smirnov_stat(s1, s2), 0.0, decimal=10)

    def test_ks_stat_different(self, diff_spectra):
        s1, s2 = diff_spectra
        assert kolmogorov_smirnov_stat(s1, s2) > 0


# ─── Correlation ──────────────────────────────────────────────────


class TestCorrelation:
    def test_pearson_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(pearson_r(s1, s2), 0.0, decimal=10)

    def test_pearson_inverse(self):
        x = np.linspace(0, 1, 50)
        assert_almost_equal(pearson_r(x, -x), -1.0, decimal=10)

    def test_spearman_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(spearman_r(s1, s2), 0.0, decimal=10)

    def test_spearman_inverse(self):
        x = np.linspace(0, 1, 50)
        assert_almost_equal(spearman_r(x, -x), -1.0, decimal=5)

    def test_correlation_sin_cos(self, sin_cos_spectra):
        s1, s2 = sin_cos_spectra
        r = pearson_r(s1, s2)
        assert -1.0 <= r <= 1.0


# ─── Error metrics ────────────────────────────────────────────────


class TestErrorMetrics:
    def test_mse_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(mean_squared_error(s1, s2), 0.0)

    def test_mse_different(self, diff_spectra):
        s1, s2 = diff_spectra
        assert mean_squared_error(s1, s2) > 0

    def test_rmse(self):
        s1 = np.array([0, 2])
        s2 = np.array([0, 0])
        assert_almost_equal(root_mean_squared_error(s1, s2), np.sqrt(2.0))

    def test_mae_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(mean_absolute_error(s1, s2), 0.0)

    def test_mape_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(mape(s1, s2), 0.0)

    def test_mape_simple(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.1, 2.2, 3.3])
        expected = np.mean([0.1, 0.1, 0.1]) * 100
        assert_almost_equal(mape(s1, s2), expected, decimal=10)

    def test_mape_zero_reference(self):
        s1 = np.zeros(5)
        s2 = np.ones(5)
        assert mape(s1, s2) == 0.0

    def test_r2_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(r2_score(s1, s2), 0.0)

    def test_r2_worse_than_mean(self):
        s1 = np.array([1, 2, 3])
        s2 = np.array([10, 20, 30])
        assert r2_score(s1, s2) < 0

    def test_max_error_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(max_error(s1, s2), 0.0)

    def test_max_error_simple(self):
        s1 = np.array([0, 1])
        s2 = np.array([5, 1])
        assert_almost_equal(max_error(s1, s2), 5.0)

    def test_median_absolute_error_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(median_absolute_error(s1, s2), 0.0)

    def test_error_metrics_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            mean_squared_error(np.ones(3), np.ones(5))


# ─── Kernel / similarity ──────────────────────────────────────────


class TestKernelMetrics:
    def test_cosine_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(cosine_similarity(s1, s2), 1.0)

    def test_cosine_orthogonal(self):
        s1 = np.array([1, 0])
        s2 = np.array([0, 1])
        assert_almost_equal(cosine_similarity(s1, s2), 0.0)

    def test_cosine_opposite(self):
        s1 = np.array([1, 0])
        s2 = np.array([-1, 0])
        assert_almost_equal(cosine_similarity(s1, s2), -1.0)

    def test_cosine_zero_norm(self):
        s1 = np.zeros(5)
        s2 = np.ones(5)
        assert cosine_similarity(s1, s2) == 0.0

    def test_mmd_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(mmd_rbf(s1, s2), 0.0, decimal=10)

    def test_mmd_different(self, diff_spectra):
        s1, s2 = diff_spectra
        assert mmd_rbf(s1, s2) >= 0

    def test_mmd_with_gamma(self):
        s1 = np.random.default_rng(42).normal(0, 1, 100)
        s2 = np.random.default_rng(43).normal(0, 1, 100)
        val = mmd_rbf(s1, s2, gamma=1.0)
        assert not np.isnan(val)

    def test_cosine_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity(np.ones(3), np.ones(5))


# ─── Chi-squared family ───────────────────────────────────────────


class TestChiSquaredFamily:
    def test_chi_squared_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(chi_squared(s1, s2), 0.0, decimal=10)

    def test_chi_squared_positive(self, diff_spectra):
        s1, s2 = diff_spectra
        assert chi_squared(s1, s2) >= 0

    def test_g_test_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(g_test(s1, s2), 0.0, decimal=10)

    def test_freeman_tukey_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(freeman_tukey(s1, s2), 0.0, decimal=10)

    def test_cressie_read_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(cressie_read(s1, s2), 0.0, decimal=10)


# ─── Statistical tests ────────────────────────────────────────────


class TestStatisticalTests:
    def test_anderson_darling_identical(self, const_spectra):
        s1, s2 = const_spectra
        result = anderson_darling(s1, s2)
        assert result >= 0

    def test_wilcoxon_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(wilcoxon_test(s1, s2), 0.0, decimal=5)

    def test_mannwhitneyu_identical(self, const_spectra):
        s1, s2 = const_spectra
        result = mannwhitneyu_test(s1, s2)
        assert result >= 0

    def test_smd_identical(self, const_spectra):
        s1, s2 = const_spectra
        assert_almost_equal(standardized_mean_difference(s1, s2), 0.0)

    def test_smd_positive(self):
        s1 = np.array([1, 2, 3])
        s2 = np.array([5, 6, 7])
        val = standardized_mean_difference(s1, s2)
        assert val != 0


# ─── High-level compare_spectra ───────────────────────────────────


class TestCompareSpectra:
    def test_all_metrics(self, const_spectra):
        s1, s2 = const_spectra
        result = compare_spectra(s1, s2)
        assert isinstance(result, dict)
        assert "kl_divergence" in result
        assert "cosine_similarity" in result
        assert "mean_squared_error" in result
        assert len(result) > 20

    def test_single_metric(self, const_spectra):
        s1, s2 = const_spectra
        result = compare_spectra(s1, s2, metrics="mean_squared_error")
        assert result == {"mean_squared_error": 0.0}

    def test_multiple_metrics(self, const_spectra):
        s1, s2 = const_spectra
        result = compare_spectra(
            s1, s2, metrics=["mean_squared_error", "mean_absolute_error", "cosine_similarity"]
        )
        assert set(result.keys()) == {"mean_squared_error", "mean_absolute_error", "cosine_similarity"}
        assert_almost_equal(result["mean_squared_error"], 0.0)
        assert_almost_equal(result["cosine_similarity"], 1.0)

    def test_invalid_metric(self, const_spectra):
        s1, s2 = const_spectra
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_spectra(s1, s2, metrics="nonexistent")

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compare_spectra(np.ones(3), np.ones(5))


# ─── compare_multiple ─────────────────────────────────────────────


class TestCompareMultiple:
    def test_two_spectra(self, const_spectra):
        s1, s2 = const_spectra
        result = compare_multiple([s1, s2], metrics="mean_squared_error")
        key = list(result.keys())[0]
        assert "vs" in key
        assert result[key]["mean_squared_error"] == 0.0

    def test_three_spectra(self):
        s1 = np.ones(50)
        s2 = np.ones(50) * 2
        s3 = np.ones(50) * 3
        result = compare_multiple([s1, s2, s3], metrics="mean_squared_error")
        assert len(result) == 2

    def test_custom_labels(self):
        s1 = np.ones(50)
        s2 = np.ones(50) * 2
        result = compare_multiple(
            [s1, s2], metrics="mean_squared_error", labels=["Ref", "Test"]
        )
        assert "Ref vs Test" in result

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="At least two"):
            compare_multiple([np.ones(5)])

    def test_label_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels must match"):
            compare_multiple([np.ones(5), np.ones(5)], labels=["a"])


# ─── Detector.compare() ───────────────────────────────────────────


class TestDetectorCompare:
    def test_two_arrays(self, detector):
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        result = detector.compare(s1, s2, metrics="mean_squared_error")
        assert isinstance(result, dict)
        assert_almost_equal(result["mean_squared_error"], 1.0)

    def test_two_results(self, detector, sample_readings):
        r1 = detector.unfold_qpsolvers(sample_readings, save_result=False)
        r2 = detector.unfold_cvxpy(sample_readings, save_result=False)
        result = detector.compare(r1, r2, metrics="cosine_similarity")
        assert "cosine_similarity" in result
        assert 0 <= result["cosine_similarity"] <= 1

    def test_three_spectra_returns_dataframe(self, detector):
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        s3 = np.ones(detector.n_energy_bins) * 3
        result = detector.compare(s1, s2, s3, metrics="mean_squared_error")
        assert isinstance(result, pd.DataFrame)

    def test_with_labels(self, detector):
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        result = detector.compare(s1, s2, metrics="mean_squared_error", labels=["A", "B"])
        assert isinstance(result, dict)

    def test_length_mismatch_raises(self, detector):
        with pytest.raises(ValueError, match="bins"):
            detector.compare(np.ones(5), np.ones(10))

    def test_fewer_than_two_raises(self, detector):
        with pytest.raises(ValueError, match="At least two"):
            detector.compare(np.ones(detector.n_energy_bins))

    def test_dict_without_spectrum_key_raises(self, detector):
        with pytest.raises(ValueError, match="no 'spectrum'"):
            detector.compare({"wrong": "data"}, np.ones(detector.n_energy_bins))

    def test_invalid_type_raises(self, detector):
        with pytest.raises(TypeError, match="must be ndarray or dict"):
            detector.compare(42, 43)

    def test_plot_two_spectra(self, detector):
        import matplotlib
        matplotlib.use("Agg")
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        result, fig, ax1, ax2 = detector.compare(
            s1, s2, metrics="mean_squared_error", plot=True, return_fig=True
        )
        assert result is not None
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_save_to_file(self, detector, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        save_path = str(tmp_path / "compare_test.png")
        detector.compare(s1, s2, metrics="mean_squared_error", plot=True, save_to=save_path)
        assert tmp_path.joinpath("compare_test.png").exists()

    def test_plot_save_to_eps(self, detector, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        save_path = str(tmp_path / "compare_test.eps")
        detector.compare(s1, s2, metrics="mean_squared_error", plot=True, save_to=save_path)
        assert tmp_path.joinpath("compare_test.eps").exists()

    def test_all_metrics_via_detector(self, detector):
        s1 = np.ones(detector.n_energy_bins)
        s2 = np.ones(detector.n_energy_bins) * 2
        result = detector.compare(s1, s2)
        assert isinstance(result, dict)
        assert len(result) > 20


# ─── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_spectrum(self):
        s1 = np.zeros(50)
        s2 = np.ones(50)
        result = compare_spectra(s1, s2, metrics="cosine_similarity")
        assert result["cosine_similarity"] == 0.0

    def test_negative_spectrum(self):
        s1 = np.array([-1.0, -2.0, -3.0])
        s2 = np.array([1.0, 2.0, 3.0])
        result = compare_spectra(s1, s2, metrics="kl_divergence")
        assert result["kl_divergence"] >= 0

    def test_all_metrics_return_finite(self, diff_spectra):
        s1, s2 = diff_spectra
        result = compare_spectra(s1, s2)
        for key, val in result.items():
            assert np.isfinite(val), f"Metric {key} returned non-finite: {val}"

    def test_const_spectra_consistency(self):
        """All metrics should return same values for same constant spectra."""
        s1 = np.ones(50)
        s2 = np.ones(50) * 2
        r1 = compare_spectra(s1, s2)
        r2 = compare_spectra(s1, s2)
        for key in r1:
            v1, v2 = r1[key], r2[key]
            if np.isnan(v1):
                assert np.isnan(v2), f"Metric {key}: nan mismatch"
            else:
                assert_almost_equal(v1, v2, err_msg=f"Metric {key} inconsistent")


# ─── Import from top-level ────────────────────────────────────────


class TestImports:
    def test_import_from_top_level(self):
        from bssunfold import compare_spectra as cs
        assert cs is not None

    def test_import_from_utils(self):
        assert hasattr(bss_utils, "compare_spectra")
        assert hasattr(bss_utils, "kl_divergence")
        assert hasattr(bss_utils, "cosine_similarity")
