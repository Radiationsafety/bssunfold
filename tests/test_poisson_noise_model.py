"""Tests for the Poisson counting-statistics noise model.

``noise_model='poisson'`` interprets the readings as counting data
(counts, or rates with ``measurement_time``) and resamples them as
Poisson-distributed counts. The systematic covariance (if given) is
composed additively on top of the statistical part
(``V_b = V_stat + V_syst``).
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core._montecarlo import monte_carlo_uncertainty

READINGS = {"a": 10000.0, "b": 40000.0}


def _identity_solver(readings, **kwargs):
    return np.array([readings["a"], readings["b"]])


class TestPoissonModel:
    def test_counts_variance_matches_sqrt(self):
        """std of resampled counts approximates sqrt(expected counts)."""
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.0,
            n_samples=4000,
            n_energy_bins=2,
            random_state=11,
            noise_model="poisson",
        )
        assert res["noise_model"] == "poisson"
        assert res["spectrum_uncert_std"] == pytest.approx(
            np.sqrt([10000.0, 40000.0]), rel=0.05
        )

    def test_rates_with_measurement_time(self):
        """Rates b (counts per T) have Var = b / T after resampling."""
        rates = {"a": 100.0, "b": 400.0}
        res = monte_carlo_uncertainty(
            _identity_solver,
            rates,
            noise_level=0.0,
            n_samples=4000,
            n_energy_bins=2,
            random_state=12,
            noise_model="poisson",
            measurement_time=100.0,
        )
        # Var(rate_i) = rate_i / T -> std = sqrt(rate/T)
        assert res["spectrum_uncert_std"] == pytest.approx(
            np.sqrt([1.0, 4.0]), rel=0.06
        )

    def test_measurement_time_must_be_positive(self):
        with pytest.raises(ValueError, match="measurement_time"):
            monte_carlo_uncertainty(
                _identity_solver,
                READINGS,
                noise_level=0.0,
                n_samples=5,
                n_energy_bins=2,
                noise_model="poisson",
                measurement_time=0.0,
            )

    def test_negative_readings_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            monte_carlo_uncertainty(
                _identity_solver,
                {"a": -1.0, "b": 2.0},
                noise_level=0.0,
                n_samples=5,
                n_energy_bins=2,
                noise_model="poisson",
            )

    def test_zero_counts_are_deterministic(self):
        res = monte_carlo_uncertainty(
            _identity_solver,
            {"a": 0.0, "b": 10000.0},
            noise_level=0.0,
            n_samples=50,
            n_energy_bins=2,
            random_state=1,
            noise_model="poisson",
        )
        assert np.all(res["spectrum_uncert_all"][:, 0] == 0.0)

    def test_invalid_noise_model(self):
        with pytest.raises(ValueError, match="noise_model"):
            monte_carlo_uncertainty(
                _identity_solver,
                READINGS,
                noise_level=0.1,
                n_samples=5,
                n_energy_bins=2,
                noise_model="bogus",
            )

    def test_composes_with_covariance(self):
        """Poisson stat part + Gaussian syst part add in quadrature."""
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.0,
            n_samples=4000,
            n_energy_bins=2,
            random_state=13,
            noise_model="poisson",
            reading_covariance=np.diag([100.0, 100.0]),
        )
        # Var_total = Var_poisson + Var_syst = counts + 100
        expected = np.sqrt([10100.0, 40100.0])
        assert res["spectrum_uncert_std"] == pytest.approx(expected, rel=0.05)

    def test_gaussian_model_key(self):
        res = monte_carlo_uncertainty(
            _identity_solver,
            READINGS,
            noise_level=0.1,
            n_samples=10,
            n_energy_bins=2,
            random_state=1,
        )
        assert res["noise_model"] == "gaussian"

    def test_reproducible(self):
        kw = dict(
            noise_level=0.0,
            n_samples=20,
            n_energy_bins=2,
            noise_model="poisson",
        )
        r1 = monte_carlo_uncertainty(_identity_solver, READINGS, random_state=5, **kw)
        r2 = monte_carlo_uncertainty(_identity_solver, READINGS, random_state=5, **kw)
        assert np.allclose(r1["spectrum_uncert_all"], r2["spectrum_uncert_all"])


class TestDetectorPoisson:
    @pytest.fixture
    def detector(self):
        return Detector()

    @pytest.fixture
    def readings(self):
        return {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

    def test_end_to_end(self, detector, readings):
        result = detector.unfold_landweber(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=30,
            noise_model="poisson",
            measurement_time=60.0,
        )
        assert result["noise_model"] == "poisson"
        assert "spectrum_uncert_std" in result
        assert np.all(np.isfinite(result["spectrum_uncert_std"]))

    def test_gaussian_default(self, detector, readings):
        result = detector.unfold_landweber(
            readings,
            max_iterations=10,
            calculate_errors=True,
            n_montecarlo=10,
        )
        assert result["noise_model"] == "gaussian"

    def test_add_noise_poisson(self, detector):
        noisy = detector._add_noise(
            {"4in": 10000.0},
            noise_model="poisson",
            random_state=1,
        )
        # Poisson(10000) deviates from the mean by ~sqrt(10000) = 100
        assert abs(noisy["4in"] - 10000.0) < 1000.0

    def test_add_noise_poisson_rates(self, detector):
        noisy = detector._add_noise(
            {"4in": 100.0},
            noise_model="poisson",
            measurement_time=100.0,
            random_state=1,
        )
        # rate std = sqrt(100/100) = 1.0
        assert abs(noisy["4in"] - 100.0) < 10.0

    def test_add_noise_poisson_composes_covariance(self, detector):
        noisy = detector._add_noise(
            {"4in": 0.0},
            noise_model="poisson",
            reading_covariance=np.array([[4.0]]),
            random_state=3,
        )
        assert abs(noisy["4in"]) < 20.0  # systematic shift on zero counts
