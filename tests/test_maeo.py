"""Test MAEO unfolding methods."""

import numpy as np
import pytest
from bssunfold import Detector


def test_maeo_basic():
    """Test basic MAEO functionality."""
    detector = Detector()
    
    # Create simple test readings
    np.random.seed(42)
    n_detectors = detector.n_detectors
    readings = {
        detector.detector_names[i]: float(100 + 10 * np.random.randn())
        for i in range(n_detectors)
    }
    
    # Run MAEO with minimal settings
    result = detector.unfold_maeo(
        readings,
        n_cycles=2,
        n_gen_per_cycle=2,
        pop_size=10,
        algorithms=["nsga3", "spea2"],
        seed=42,
    )
    
    # Check results
    assert "spectrum" in result
    assert "energy" in result
    assert "method" in result
    assert result["method"] == "MAEO"
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert "maeo_info" in result
    assert "best_algorithm" in result["maeo_info"]
    

def test_maeo_with_prior():
    """Test MAEO with prior spectrum (3 objectives)."""
    detector = Detector()
    
    # Create test readings
    readings = {
        detector.detector_names[i]: 100.0
        for i in range(detector.n_detectors)
    }
    
    # Create a simple prior spectrum
    prior = np.ones(detector.n_energy_bins) * 1e6
    
    result = detector.unfold_maeo(
        readings,
        n_cycles=2,
        n_gen_per_cycle=2,
        pop_size=10,
        prior_spectrum=prior,
        seed=42,
    )
    
    assert "spectrum" in result
    assert len(result["spectrum"]) == detector.n_energy_bins
    

def test_maeo_all_algorithms():
    """Test MAEO with all four default algorithms."""
    detector = Detector()
    
    readings = {
        detector.detector_names[i]: 50.0 + 10 * i
        for i in range(detector.n_detectors)
    }
    
    result = detector.unfold_maeo(
        readings,
        n_cycles=2,
        n_gen_per_cycle=2,
        pop_size=10,
        algorithms=["nsga3", "ctaea", "agemoea2", "spea2"],
        seed=42,
    )
    
    assert "maeo_info" in result
    assert result["maeo_info"]["algorithms_used"] == ["nsga3", "ctaea", "agemoea2", "spea2"]
    assert "hypervolume_history" in result["maeo_info"]
    

def test_maeo_convergence_assist():
    """Test MAEO convergence assist mechanism."""
    detector = Detector()
    
    readings = {
        detector.detector_names[i]: 100.0
        for i in range(detector.n_detectors)
    }
    
    # Test with high convergence assist ratio
    result_high = detector.unfold_maeo(
        readings,
        n_cycles=4,
        n_gen_per_cycle=2,
        pop_size=10,
        convergence_assist_ratio=0.5,  # 50% convergence phase
        seed=42,
    )
    
    # Test with low convergence assist ratio
    result_low = detector.unfold_maeo(
        readings,
        n_cycles=4,
        n_gen_per_cycle=2,
        pop_size=10,
        convergence_assist_ratio=0.1,  # 10% convergence phase
        seed=42,
    )
    
    assert "maeo_info" in result_high
    assert "maeo_info" in result_low
    

def test_maeo_initial_spectrum():
    """Test MAEO with initial spectrum warm-start."""
    detector = Detector()
    
    readings = {
        detector.detector_names[i]: 100.0
        for i in range(detector.n_detectors)
    }
    
    # Provide initial spectrum guess
    initial = np.ones(detector.n_energy_bins) * 1e5
    
    result = detector.unfold_maeo(
        readings,
        n_cycles=2,
        n_gen_per_cycle=2,
        pop_size=10,
        initial_spectrum=initial,
        seed=42,
    )
    
    assert "spectrum" in result
    assert np.all(result["spectrum"] >= 0)


def test_maeo_result_format():
    """Test that MAEO returns properly formatted results."""
    detector = Detector()
    
    readings = {
        detector.detector_names[i]: 100.0
        for i in range(detector.n_detectors)
    }
    
    result = detector.unfold_maeo(
        readings,
        n_cycles=2,
        n_gen_per_cycle=2,
        pop_size=10,
        seed=42,
    )
    
    # Check standard output fields
    required_fields = [
        "energy",
        "spectrum",
        "spectrum_absolute",
        "effective_readings",
        "residual",
        "residual_norm",
        "method",
        "doserates",
    ]
    
    for field in required_fields:
        assert field in result, f"Missing field: {field}"
    
    # Check MAEO-specific fields
    assert "maeo_info" in result
    assert "maeo_pareto_front" in result or result.get("maeo_pareto_front") is None
    
    # Check types
    assert isinstance(result["energy"], np.ndarray)
    assert isinstance(result["spectrum"], np.ndarray)
    assert isinstance(result["residual_norm"], float)
    assert result["method"] == "MAEO"


if __name__ == "__main__":
    # Run tests
    print("Running MAEO tests...")
    
    print("  test_maeo_basic... ", end="")
    test_maeo_basic()
    print("PASSED")
    
    print("  test_maeo_with_prior... ", end="")
    test_maeo_with_prior()
    print("PASSED")
    
    print("  test_maeo_all_algorithms... ", end="")
    test_maeo_all_algorithms()
    print("PASSED")
    
    print("  test_maeo_convergence_assist... ", end="")
    test_maeo_convergence_assist()
    print("PASSED")
    
    print("  test_maeo_initial_spectrum... ", end="")
    test_maeo_initial_spectrum()
    print("PASSED")
    
    print("  test_maeo_result_format... ", end="")
    test_maeo_result_format()
    print("PASSED")
    
    print("\nAll MAEO tests passed!")
