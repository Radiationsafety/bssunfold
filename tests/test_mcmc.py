"""Tests for the Bayesian MCMC unfolding method."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

from src.bssunfold import Detector, RF_GSF


@pytest.fixture
def detector():
    """Fixture to create a Detector instance with GSF response functions."""
    import pandas as pd
    df = pd.DataFrame.from_dict(RF_GSF, orient="columns")
    return Detector(df)


@pytest.fixture
def sample_readings():
    """Sample detector readings for testing."""
    return {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC and ArviZ are required for MCMC tests")
class TestUnfoldMCMC:
    """Test suite for the unfold_mcmc method."""

    def test_unfold_mcmc_returns_correct_structure(self, detector, sample_readings):
        """Test that unfold_mcmc returns a dictionary with correct structure."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        # Check that result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check required keys
        required_keys = [
            'energy',
            'spectrum',
            'spectrum_absolute',
            'spectrum_uncertainty',
            'spectrum_lower',
            'spectrum_upper',
            'effective_readings',
            'residual',
            'residual_norm',
            'method',
            'doserates',
            'mcmc_stats',
        ]

        for key in required_keys:
            assert key in result, f"Result should contain key '{key}'"

    def test_unfold_mcmc_spectrum_properties(self, detector, sample_readings):
        """Test properties of the unfolded spectrum."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        spectrum = result['spectrum']
        energy = result['energy']

        # Spectrum should be non-negative
        assert np.all(spectrum >= 0), "Spectrum should be non-negative"

        # Spectrum should have same length as energy grid
        assert len(spectrum) == len(energy), \
            f"Spectrum length {len(spectrum)} should match energy grid length {len(energy)}"

        # Spectrum should not be all zeros
        assert np.any(spectrum > 0), "Spectrum should not be all zeros"

    def test_unfold_mcmc_uncertainty_properties(self, detector, sample_readings):
        """Test properties of uncertainty estimates."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        std = result['spectrum_uncertainty']
        lower = result['spectrum_lower']
        upper = result['spectrum_upper']

        # Standard deviation should be non-negative
        assert np.all(std >= 0), "Standard deviation should be non-negative"

        # HPD bounds should be non-negative
        assert np.all(lower >= 0), "Lower HPD bound should be non-negative"
        assert np.all(upper >= 0), "Upper HPD bound should be non-negative"

        # Lower bound should be less than or equal to upper bound
        assert np.all(lower <= upper), \
            "Lower HPD bound should be <= upper HPD bound"

    def test_unfold_mcmc_doserates(self, detector, sample_readings):
        """Test dose rate calculation from MCMC results."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        doserates = result['doserates']

        # Check that doserates is a dictionary
        assert isinstance(doserates, dict), "Doserates should be a dictionary"

        # Check expected dose rate keys
        expected_keys = ["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]
        for key in expected_keys:
            assert key in doserates, f"Doserates should contain key '{key}'"
            assert isinstance(doserates[key], (int, float)), \
                f"Dose rate for {key} should be numeric"

    def test_unfold_mcmc_mcmc_stats(self, detector, sample_readings):
        """Test MCMC statistics in result."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        mcmc_stats = result['mcmc_stats']

        # Check required MCMC stats keys
        required_stats_keys = [
            'samples',
            'median',
            'hpd_lower',
            'hpd_upper',
            'rhat',
            'ess',
            'n_samples_total',
            'n_chains',
            'tune_samples',
            'target_accept',
            'use_hierarchical',
            'trace',
        ]

        for key in required_stats_keys:
            assert key in mcmc_stats, f"MCMC stats should contain key '{key}'"

        # Check samples shape
        samples = mcmc_stats['samples']
        assert samples.ndim == 2, "Samples should be 2D array"
        assert samples.shape[0] > 0, "Should have samples"
        assert samples.shape[1] == len(result['energy']), \
            "Samples should have same number of energy bins as result"

        # Check R-hat values (should be close to 1.0 for convergence)
        rhat = mcmc_stats['rhat']
        assert np.all(np.isfinite(rhat)), "R-hat values should be finite"

        # Check ESS values (should be positive)
        ess = mcmc_stats['ess']
        assert np.all(ess > 0), "ESS values should be positive"

    def test_unfold_mcmc_with_hierarchical_priors(self, detector, sample_readings):
        """Test MCMC unfolding with hierarchical priors."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            use_hierarchical=True,
            random_state=42,
        )

        # Check that hierarchical mode was used
        assert result['mcmc_stats']['use_hierarchical'] is True, \
            "Hierarchical mode should be enabled"

        # Spectrum should still be valid
        spectrum = result['spectrum']
        assert np.all(spectrum >= 0), "Spectrum should be non-negative"
        assert np.any(spectrum > 0), "Spectrum should not be all zeros"

    def test_unfold_mcmc_different_priors(self, detector, sample_readings):
        """Test MCMC unfolding with different prior parameters."""
        result1 = detector.unfold_mcmc(
            sample_readings,
            sigma_prior=0.05,
            lambda_prior=0.5,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        result2 = detector.unfold_mcmc(
            sample_readings,
            sigma_prior=0.2,
            lambda_prior=2.0,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        # Both should produce valid spectra
        assert np.all(result1['spectrum'] >= 0)
        assert np.all(result2['spectrum'] >= 0)

        # Prior parameters should be stored in result
        assert result1['sigma_prior'] == 0.05
        assert result1['lambda_prior'] == 0.5
        assert result2['sigma_prior'] == 0.2
        assert result2['lambda_prior'] == 2.0

    def test_unfold_mcmc_reproducibility(self, detector, sample_readings):
        """Test that MCMC results are reproducible with same random seed."""
        result1 = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=123,
        )

        result2 = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=123,
        )

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(
            result1['spectrum'],
            result2['spectrum'],
            decimal=10,
            err_msg="Results should be identical with same random seed"
        )

    def test_unfold_mcmc_residual_calculation(self, detector, sample_readings):
        """Test residual calculation in MCMC results."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        residual = result['residual']
        residual_norm = result['residual_norm']
        effective_readings = result['effective_readings']

        # Residual should be an array
        assert isinstance(residual, np.ndarray), "Residual should be numpy array"

        # Residual norm should be a scalar
        assert isinstance(residual_norm, (int, float)), \
            "Residual norm should be a scalar"

        # Residual norm should be non-negative
        assert residual_norm >= 0, "Residual norm should be non-negative"

        # Effective readings should be a dictionary with same keys as input
        assert isinstance(effective_readings, dict), \
            "Effective readings should be a dictionary"
        assert set(effective_readings.keys()) == set(sample_readings.keys()), \
            "Effective readings should have same keys as input readings"

    def test_unfold_mcmc_method_identifier(self, detector, sample_readings):
        """Test that method identifier is correctly set."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        assert result['method'] == 'mcmc', "Method should be identified as 'mcmc'"

    def test_unfold_mcmc_energy_grid(self, detector, sample_readings):
        """Test that energy grid is correctly returned."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=2,
            random_state=42,
        )

        energy = result['energy']

        # Energy should be a numpy array
        assert isinstance(energy, np.ndarray), "Energy should be numpy array"

        # Energy should be positive
        assert np.all(energy > 0), "Energy values should be positive"

        # Energy should be sorted in ascending order
        assert np.all(np.diff(energy) > 0), \
            "Energy values should be in ascending order"


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC and ArviZ are required for MCMC tests")
class TestUnfoldMCMCErrors:
    """Test error handling in unfold_mcmc method."""

    def test_unfold_mcmc_empty_readings(self, detector):
        """Test that empty readings raise an error."""
        with pytest.raises(Exception):
            detector.unfold_mcmc(
                {},
                n_samples=100,
                tune=50,
                chains=2,
                random_state=42,
            )

    def test_unfold_mcmc_invalid_detector_names(self, detector):
        """Test that invalid detector names raise an error."""
        with pytest.raises(Exception):
            detector.unfold_mcmc(
                {"invalid_detector": 1.0},
                n_samples=100,
                tune=50,
                chains=2,
                random_state=42,
            )

    def test_unfold_mcmc_negative_readings(self, detector):
        """Test handling of negative readings (should fail or be handled)."""
        # Negative readings are physically meaningless
        # The method should either reject them or handle them gracefully
        # Note: This test may take longer due to MCMC sampling
        # For now, we just verify that the method runs without crashing
        # Negative values in input may still produce valid output due to Bayesian model
        result = detector.unfold_mcmc(
            {"3in": -0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034},
            n_samples=50,  # Reduced samples for faster test
            tune=25,
            chains=1,
            random_state=42,
        )
        # Spectrum should still be non-negative due to HalfNormal prior
        assert np.all(result['spectrum'] >= 0)


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC and ArviZ are required for MCMC tests")
class TestUnfoldMCMCParameters:
    """Test different parameter configurations for unfold_mcmc."""

    def test_unfold_mcmc_single_chain(self, detector, sample_readings):
        """Test MCMC with single chain."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=1,
            random_state=42,
        )

        assert result['mcmc_stats']['n_chains'] == 1
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_mcmc_multiple_chains(self, detector, sample_readings):
        """Test MCMC with multiple chains."""
        result = detector.unfold_mcmc(
            sample_readings,
            n_samples=100,
            tune=50,
            chains=3,
            random_state=42,
        )

        assert result['mcmc_stats']['n_chains'] == 3
        assert np.all(result['spectrum'] >= 0)

    def test_unfold_mcmc_different_target_accept(self, detector, sample_readings):
        """Test MCMC with different target acceptance rates."""
        for target_accept in [0.7, 0.8, 0.9]:
            result = detector.unfold_mcmc(
                sample_readings,
                n_samples=100,
                tune=50,
                chains=2,
                target_accept=target_accept,
                random_state=42,
            )

            assert result['mcmc_stats']['target_accept'] == target_accept
            assert np.all(result['spectrum'] >= 0)

    def test_unfold_mcmc_sample_sizes(self, detector, sample_readings):
        """Test MCMC with different sample sizes."""
        for n_samples in [50, 100, 200]:
            result = detector.unfold_mcmc(
                sample_readings,
                n_samples=n_samples,
                tune=50,
                chains=2,
                random_state=42,
            )

            # Total samples should be approximately n_samples * chains
            total_samples = result['mcmc_stats']['n_samples_total']
            expected_min = n_samples * 2  # At least this many samples
            assert total_samples >= expected_min, \
                f"Should have at least {expected_min} samples, got {total_samples}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
