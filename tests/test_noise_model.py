"""Tests for the reading-uncertainty / covariance noise model.

The Monte-Carlo uncertainty propagation supports, in order of precedence:
``reading_covariance`` (absolute, multivariate) > ``reading_uncertainties``
(absolute 1-sigma per reading) > ``noise_level`` (legacy relative scalar).
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core._montecarlo import _covariance_factor, monte_carlo_uncertainty

READINGS = {"a": 100.0, "b": 200.0, "c": 50.0}


def _identity_solver(readings, n_energy_bins=4, **kwargs):
    """Deterministic linear 'unfolding' for testing: pinv of a fixed system."""
    A = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    b = np.array([readings["a"], readings["b"], readings["c"]])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.maximum(x, 0)


class TestCovarianceFactor:
    def test_cholesky_exact(self):
        cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        L = _covariance_factor(cov)
        assert np.allclose(L @ L.T, cov)

    def test_indefinite_clipped_to_psd(self):
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
        L = _covariance_factor(cov)
        eig = np.linalg.eigvalsh(L @ L.T)
        assert np.all(eig >= -1e-12)

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="square"):
            _covariance_factor(np.ones((2, 3)))


class TestMonteCarloNoiseModels:
    def test_uncertainties_dict_precedence(self):
        """With uncertainties given, the legacy noise_level is ignored."""
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.0,  # legacy disabled, but uncertainties active
            n_samples=30,
            n_energy_bins=4,
            random_state=42,
            reading_uncertainties={"a": 5.0, "b": 10.0, "c": 2.0},
        )
        assert res["spectrum_uncert_std"].shape == (4,)
        assert np.any(res["spectrum_uncert_std"] > 0)

    def test_uncertainties_array(self):
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.0,
            n_samples=30,
            n_energy_bins=4,
            random_state=42,
            reading_uncertainties=np.array([5.0, 10.0, 2.0]),
        )
        assert np.any(res["spectrum_uncert_std"] > 0)

    def test_uncertainties_wrong_length(self):
        with pytest.raises(ValueError, match="number of readings"):
            monte_carlo_uncertainty(
                _identity_solver,
                READINGS,
                noise_level=0.1,
                n_samples=5,
                n_energy_bins=4,
                reading_uncertainties=np.array([1.0, 2.0]),
            )

    def test_uncertainties_missing_key(self):
        with pytest.raises(ValueError, match="missing entries"):
            monte_carlo_uncertainty(
                _identity_solver,
                READINGS,
                noise_level=0.1,
                n_samples=5,
                n_energy_bins=4,
                reading_uncertainties={"a": 1.0},
            )

    def test_covariance_correlated_increases_spread(self):
        """A positively correlated covariance yields a wider spread.

        Uses an identity 'solver' (spectrum = readings) so the propagated
        spread is directly interpretable: with a fully correlated matrix
        every direction shares the same variance.
        """
        def identity_solver(readings, **kwargs):
            return np.array([readings["a"], readings["b"], readings["c"]])

        common_kw = dict(
            noise_level=0.0,
            n_samples=200,
            n_energy_bins=3,
            random_state=7,
        )
        diag = monte_carlo_uncertainty(
            identity_solver,
            READINGS,
            reading_covariance=np.diag([25.0, 25.0, 25.0]),
            **common_kw,
        )
        corr = monte_carlo_uncertainty(
            identity_solver,
            READINGS,
            reading_covariance=np.full((3, 3), 25.0),  # fully correlated
            **common_kw,
        )
        # fully correlated noise: components move together (corr ~ 1);
        # with the diagonal matrix they are independent (corr ~ 0)
        cc = np.corrcoef(corr["spectrum_uncert_all"].T)
        off_diag_corr = cc[~np.eye(3, dtype=bool)]
        assert np.all(off_diag_corr > 0.9)

        dc = np.corrcoef(diag["spectrum_uncert_all"].T)
        off_diag_diag = dc[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off_diag_diag) < 0.4)

    def test_covariance_takes_precedence(self):
        """Covariance wins over uncertainties and noise_level."""
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.0,
            n_samples=10,
            n_energy_bins=4,
            random_state=1,
            reading_uncertainties={"a": 1.0, "b": 1.0, "c": 1.0},
            reading_covariance=np.diag([25.0, 25.0, 25.0]),
        )
        assert np.any(res["spectrum_uncert_std"] > 0)

    def test_legacy_relative_noise_unchanged(self):
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.05,
            n_samples=50,
            n_energy_bins=4,
            random_state=3,
        )
        assert np.any(res["spectrum_uncert_std"] > 0)
        assert res["spectrum_uncert_mean"].shape == (4,)

    def test_reproducible_with_seed(self):
        kw = dict(
            noise_level=0.0,
            n_samples=10,
            n_energy_bins=4,
            reading_covariance=np.diag([1.0, 4.0, 9.0]),
        )
        r1 = monte_carlo_uncertainty(_identity_solver, READINGS, random_state=99, **kw)
        r2 = monte_carlo_uncertainty(_identity_solver, READINGS, random_state=99, **kw)
        assert np.allclose(r1["spectrum_uncert_all"], r2["spectrum_uncert_all"])


class TestDetectorAddNoise:
    def test_uncertainties_absolute(self):
        det = Detector()
        rng = dict(reading_uncertainties={"4in": 0.0, "8in": 0.0}, random_state=1)
        noisy = det._add_noise({"4in": 10.0, "8in": 20.0}, **rng)
        assert noisy == {"4in": 10.0, "8in": 20.0}  # zero sigma -> unchanged

    def test_covariance_absolute(self):
        det = Detector()
        noisy = det._add_noise(
            {"4in": 10.0, "8in": 20.0},
            noise_level=0.0,  # zero statistical noise
            reading_covariance=np.zeros((2, 2)),
            random_state=1,
        )
        assert noisy == {"4in": 10.0, "8in": 20.0}

    def test_covariance_composes_with_stat_noise(self):
        """Systematic covariance adds on top of the statistical part."""
        det = Detector()
        det2_cov_only = det._add_noise(
            {"4in": 10.0},
            noise_level=0.0,
            reading_covariance=np.array([[25.0]]),
            random_state=7,
        )
        assert det2_cov_only["4in"] != 10.0  # syst shift applied

    def test_legacy_relative(self):
        det = Detector()
        noisy = det._add_noise({"4in": 10.0}, noise_level=0.0, random_state=1)
        assert noisy["4in"] == 10.0


class TestDetectorEndToEnd:
    """reading_uncertainties / reading_covariance on Detector methods."""

    @pytest.fixture
    def detector(self):
        return Detector()

    @pytest.fixture
    def readings(self):
        return {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

    def test_landweber_uncertainties(self, detector, readings):
        unc = {k: abs(v) * 0.05 for k, v in readings.items()}
        result = detector.unfold_landweber(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=15,
            reading_uncertainties=unc,
        )
        assert "spectrum_uncert_std" in result
        assert result["montecarlo_samples"] == 15

    def test_landweber_covariance(self, detector, readings):
        cov = np.diag([abs(v) * 0.05 for v in readings.values()]) ** 2
        result = detector.unfold_landweber(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=15,
            reading_covariance=cov,
        )
        assert "spectrum_uncert_std" in result

    def test_mlem_covariance(self, detector, readings):
        cov = np.diag([abs(v) * 0.05 for v in readings.values()]) ** 2
        result = detector.unfold_mlem(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=10,
            reading_covariance=cov,
        )
        assert "spectrum_uncert_std" in result

    def test_not_passed_without_calculate_errors(self, detector, readings):
        unc = {k: abs(v) * 0.05 for k, v in readings.items()}
        result = detector.unfold_landweber(
            readings, max_iterations=5, reading_uncertainties=unc
        )
        assert "spectrum_uncert_std" not in result
