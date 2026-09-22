"""Tests for the uncertainty-weighted comparison metrics.

``weighted_chi2`` / ``reduced_chi2`` propagate per-bin 1-sigma
uncertainties (e.g. ``spectrum_uncert_std`` from the Monte-Carlo
propagation) into the comparison statistics; ``compare_spectra`` computes
them automatically when ``uncertainties`` is provided.
"""

import logging

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.utils.comparison import (
    compare_spectra,
    reduced_chi2,
    weighted_chi2,
)


class TestWeightedChi2:
    def test_exact_value(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.2, 2.0, 2.6])
        sigma = np.array([0.1, 0.1, 0.1])
        # (0.2/0.1)^2 + 0 + (0.4/0.1)^2
        assert weighted_chi2(a, b, sigma) == pytest.approx(20.0)

    def test_identical_spectra_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert weighted_chi2(a, a, np.ones(3)) == 0.0

    def test_tuple_uncertainties_quadrature(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        combined = weighted_chi2(a, b, np.array([np.sqrt(2.0)] * 2))
        tupled = weighted_chi2(a, b, (np.ones(2), np.ones(2)))
        assert tupled == pytest.approx(combined)

    def test_higher_sigma_lower_chi2(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        small = weighted_chi2(a, b, np.full(3, 0.1))
        large = weighted_chi2(a, b, np.full(3, 1.0))
        assert large < small

    def test_zero_sigma_rejected(self):
        a = np.ones(3)
        with pytest.raises(ValueError, match="strictly positive"):
            weighted_chi2(a, a + 1, np.array([1.0, 0.0, 1.0]))

    def test_negative_sigma_rejected(self):
        a = np.ones(3)
        with pytest.raises(ValueError, match="strictly positive"):
            weighted_chi2(a, a + 1, np.array([1.0, -1.0, 1.0]))

    def test_nan_sigma_rejected(self):
        a = np.ones(3)
        with pytest.raises(ValueError, match="finite"):
            weighted_chi2(a, a + 1, np.array([1.0, np.nan, 1.0]))

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            weighted_chi2(np.ones(3), np.ones(4), np.ones(3))

    def test_uncertainties_length_mismatch(self):
        with pytest.raises(ValueError, match="spectrum length"):
            weighted_chi2(np.ones(3), np.ones(3), np.ones(4))


class TestReducedChi2:
    def test_ddof_default(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.2, 2.0, 2.6])
        sigma = np.full(3, 0.1)
        assert reduced_chi2(a, b, sigma) == pytest.approx(
            weighted_chi2(a, b, sigma) / 2
        )

    def test_ddof_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        # chi2 = sum(1/1)^2 = 3, dof = 3 - 0 = 3 -> reduced = 1
        assert reduced_chi2(a, a + 1, np.ones(3), ddof=0) == pytest.approx(1.0)

    def test_ddof_too_large(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            reduced_chi2(np.ones(3), np.ones(3), np.ones(3), ddof=3)

    def test_good_fit_near_one(self):
        """Differences consistent with sigma give reduced chi2 near 1."""
        rng = np.random.default_rng(42)
        true = np.full(200, 10.0)
        sigma = np.full(200, 0.5)
        noisy = true + rng.normal(0, 0.5, 200)
        rc = reduced_chi2(true, noisy, sigma)
        assert rc == pytest.approx(1.0, abs=0.3)


class TestCompareSpectraIntegration:
    def test_auto_computed_with_uncertainties(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.2, 2.0, 2.6])
        res = compare_spectra(a, b, uncertainties=np.full(3, 0.1))
        assert res["weighted_chi2"] == pytest.approx(20.0)
        assert res["reduced_chi2"] == pytest.approx(10.0)

    def test_not_computed_without_uncertainties(self):
        a = np.array([1.0, 2.0, 3.0])
        res = compare_spectra(a, a + 1)
        assert "weighted_chi2" not in res
        assert "reduced_chi2" not in res

    def test_requested_without_uncertainties_warns(self, caplog):
        a = np.array([1.0, 2.0, 3.0])
        with caplog.at_level(logging.WARNING, logger="bssunfold.utils.comparison"):
            res = compare_spectra(a, a + 1, metrics=["weighted_chi2"])
        assert np.isnan(res["weighted_chi2"])
        assert any("uncertainties" in r.message for r in caplog.records)

    def test_requested_explicit(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.2, 2.0, 2.6])
        res = compare_spectra(
            a, b, metrics=["weighted_chi2", "reduced_chi2"],
            uncertainties=np.full(3, 0.1),
        )
        assert set(res) == {"weighted_chi2", "reduced_chi2"}

    def test_unknown_metric_still_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_spectra(np.ones(3), np.ones(3), metrics="bogus")

    def test_tuple_uncertainties_through_aggregator(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        # combined sigma = sqrt(1 + 1); chi2 = 2 * (1/sqrt(2))^2 = 1
        res = compare_spectra(a, b, uncertainties=(np.ones(2), np.ones(2)))
        assert res["weighted_chi2"] == pytest.approx(1.0)

    def test_e2e_with_detector_uncertainties(self):
        """Spectrum uncertainties from a Detector result feed the metrics.

        Bins clamped to zero by non-negativity have zero MC std and are
        excluded (zero sigma cannot be weighted).
        """
        det = Detector()
        readings = {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}
        result = det.unfold_landweber(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=20,
            random_state=1,
        )
        spec = result["spectrum"]
        mean = result["spectrum_uncert_mean"]
        std = result["spectrum_uncert_std"]
        mask = std > 0
        assert np.any(mask), "expected at least one bin with positive std"
        res = compare_spectra(spec[mask], mean[mask], uncertainties=std[mask])
        assert np.isfinite(res["weighted_chi2"])
        assert res["weighted_chi2"] >= 0
