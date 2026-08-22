#!/usr/bin/env python3
"""Quick test of composite unfolding method on IAEA spectra."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from bssunfold import Detector, RF_PTB
from bssunfold.core import unfold_composite, DEFAULT_BIN_METHODS
from bssunfold.utils.comparison import cosine_similarity, r2_score

# Load test data
CSV_PATH = Path(__file__).parent / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
mc_spectra = pd.read_csv(CSV_PATH)

print("="*80)
print("COMPOSITE METHOD TEST")
print("="*80)

# Create detector
detector = Detector(pd.DataFrame(RF_PTB))
print(f"\nDetector: PTB ({detector.n_detectors} spheres, {detector.n_energy_bins} bins)")

# Test on first 5 spectra
spectrum_names = [c for c in mc_spectra.columns if c != "E_MeV"][:5]
ref_energy = mc_spectra["E_MeV"].values

print(f"\nTesting on {len(spectrum_names)} spectra...")
print(f"Available spectrum bins: {list(DEFAULT_BIN_METHODS.keys())}")

results = []

for spec_name in spectrum_names:
    spec_values = mc_spectra[spec_name].values.astype(float)
    ref_dict = {"E_MeV": ref_energy, "Phi": spec_values}
    
    # Discretize to detector grid
    interp_df = detector.discretize_spectra(ref_dict)
    ref_on_grid = interp_df["Phi"].values
    
    # Get readings
    readings = detector.get_effective_readings_for_spectra(ref_dict)
    
    print(f"\n--- Spectrum: {spec_name[:40]} ---")
    
    # Run composite method
    try:
        result = unfold_composite(detector, readings, n_methods=3, timeout_per_method=10.0)
        
        if result.get("spectrum") is not None:
            unfolded = result["spectrum"]
            
            # Compute metrics
            cos_sim = cosine_similarity(ref_on_grid, unfolded)
            r2 = r2_score(ref_on_grid, unfolded)
            
            print(f"  Status: {result.get('status', 'N/A')}")
            print(f"  Spectrum type: {result.get('spectrum_type', 'N/A')}")
            print(f"  Methods used: {result.get('successful_methods', [])}")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  R² score: {r2:.4f}")
            print(f"  Consistency: {result.get('consistency', 0):.4f}")
            
            results.append({
                "spectrum": spec_name,
                "status": result.get("status"),
                "type": result.get("spectrum_type"),
                "methods": ",".join(result.get("successful_methods", [])),
                "cosine": cos_sim,
                "r2": r2,
                "consistency": result.get("consistency", 0),
            })
        else:
            print(f"  ERROR: {result.get('message', 'Unknown error')}")
            results.append({
                "spectrum": spec_name,
                "status": "ERROR",
                "type": None,
                "methods": "",
                "cosine": 0,
                "r2": 0,
                "consistency": 0,
            })
            
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        results.append({
            "spectrum": spec_name,
            "status": f"EXCEPTION: {e}",
            "type": None,
            "methods": "",
            "cosine": 0,
            "r2": 0,
            "consistency": 0,
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

if len(results_df) > 0:
    print(f"\nAverage cosine similarity: {results_df['cosine'].mean():.4f}")
    print(f"Average R² score: {results_df['r2'].mean():.4f}")
    print(f"Success rate: {(results_df['status'] == 'OK').sum() / len(results_df) * 100:.1f}%")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
