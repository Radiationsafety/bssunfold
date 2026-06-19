#!/usr/bin/env python3
"""Evaluate qpsolvers and cvxpy with all available solvers across all detector types.

Runs all solver × detector combinations on the IAEA Compendium dataset (251 spectra),
computes dose rates, and saves results to CSV for notebook visualization.

Usage:
    python tests/dose_rate_qpsolvers_cvxpy.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bssunfold import (
    Detector,
    RF_GSF,
    RF_PTB,
    RF_LANL,
    RF_JINR,
    RF_FERMILAB,
    RF_EURADOS,
    RF_IHEP,
)
from bssunfold.core.dose_calculation import calculate_dose_rates

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────

IAEA_CSV = Path(__file__).parent / "IAEA_Compendium_dataset.csv"
OUTPUT_CSV = Path(__file__).parent / "qpsolvers_cvxpy_dose_rate_results.csv"

CSV_ENERGIES = np.array(
    [
        1e-9,
        2.15e-9,
        4.64e-9,
        1e-8,
        2.15e-8,
        4.64e-8,
        1e-7,
        2.15e-7,
        4.64e-7,
        1e-6,
        2.15e-6,
        4.64e-6,
        1e-5,
        2.15e-5,
        4.64e-5,
        0.0001,
        0.000215,
        0.000464,
        0.001,
        0.00215,
        0.00464,
        0.01,
        0.0125,
        0.0158,
        0.0199,
        0.0251,
        0.0316,
        0.0398,
        0.0501,
        0.063,
        0.0794,
        0.1,
        0.125,
        0.158,
        0.199,
        0.251,
        0.316,
        0.398,
        0.501,
        0.63,
        0.794,
        1.0,
        1.25,
        1.58,
        1.99,
        2.51,
        3.16,
        3.98,
        5.01,
        6.3,
        7.94,
        10.0,
        15.8,
        25.1,
        39.8,
        63.0,
        100.0,
        158.0,
        251.0,
        398.0,
    ]
)

DOSE_GEOMETRIES = ["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]

DETECTOR_CONFIGS = {
    "GSF": lambda: Detector(),
    "PTB": lambda: Detector(pd.DataFrame(RF_PTB)),
    "LANL": lambda: Detector(pd.DataFrame(RF_LANL)),
    "JINR": lambda: Detector(pd.DataFrame(RF_JINR)),
    "FERMILAB": lambda: Detector(pd.DataFrame(RF_FERMILAB)),
    "EURADOS": lambda: Detector(pd.DataFrame(RF_EURADOS)),
    "IHEP": lambda: Detector(pd.DataFrame(RF_IHEP)),
}

QPSOLVERS_LIST = [
    "clarabel",
    "ecos",
    "highs",
    "osqp",
    "piqp",
    "proxqp",
    "qpalm",
    "scs",
]

CVXPY_LIST = [
    "CLARABEL",
    "ECOS",
    "ECOS_BB",
    "HIGHS",
    "OSQP",
    "PIQP",
    "PROXQP",
    "QPALM",
    "SCIPY",
    "SCS",
]


# ── Helpers ───────────────────────────────────────────────────────


def compute_dose(spectrum):
    try:
        return calculate_dose_rates(spectrum)
    except Exception:
        return {g: 0.0 for g in DOSE_GEOMETRIES}


def try_unfold_qpsolvers(detector, readings, solver):
    try:
        result = detector.unfold_qpsolvers(
            readings,
            regularization_method="gcv",
            solver=solver,
            save_result=False,
        )
        if result and "spectrum" in result:
            return result["spectrum"], "OK"
    except Exception:
        pass
    return None, f"ERROR: qpsolvers/{solver}"


def try_unfold_cvxpy(detector, readings, solver):
    try:
        result = detector.unfold_cvxpy(
            readings,
            regularization_method="gcv",
            solver=solver,
            save_result=False,
        )
        if result and "spectrum" in result:
            return result["spectrum"], "OK"
    except Exception:
        pass
    return None, f"ERROR: cvxpy/{solver}"


# ── Main ──────────────────────────────────────────────────────────


def main():
    if not IAEA_CSV.exists():
        sys.exit(f"CSV not found: {IAEA_CSV}")

    print("=" * 70)
    print("  QPSolvers & CVXPY — Dose Rate Evaluation (7 detectors)")
    print("=" * 70)

    # Load data
    iaea_df = pd.read_csv(IAEA_CSV)
    energy_cols = [c for c in iaea_df.columns if c.startswith("Energy_bin")]
    n_spectra = len(iaea_df)
    print(f"Loaded {n_spectra} spectra from {IAEA_CSV.name}")

    # Pre-create detectors and compute readings
    print("\nPre-computing effective readings for each detector...")
    detector_readings = {}
    for det_name, det_fn in DETECTOR_CONFIGS.items():
        detector = det_fn()
        readings_map = {}
        for _, iaea_row in iaea_df.iterrows():
            place = iaea_row["Place"]
            csv_spectrum = np.array(
                [iaea_row[c] for c in energy_cols], dtype=float
            )
            ref_dict = {"E_MeV": CSV_ENERGIES, "Phi": csv_spectrum}
            readings = detector.get_effective_readings_for_spectra(ref_dict)
            readings_map[place] = readings
        detector_readings[det_name] = (detector, readings_map)
        print(
            f"  {det_name}: {detector.n_detectors} detectors, {len(readings_map)} readings computed"
        )

    # Also compute reference doses
    print("\nComputing reference doses...")
    ref_doses = {}
    for _, iaea_row in iaea_df.iterrows():
        place = iaea_row["Place"]
        csv_spectrum = np.array([iaea_row[c] for c in energy_cols], dtype=float)
        ref_dict = {"E_MeV": CSV_ENERGIES, "Phi": csv_spectrum}
        interp_df = detector_readings["GSF"][0].discretize_spectra(ref_dict)
        ref_on_grid = interp_df["Phi"].values
        ref_doses[place] = compute_dose(ref_on_grid)
    print(f"  Reference doses computed for {len(ref_doses)} spectra")

    # Run all combinations
    total = (
        (len(QPSOLVERS_LIST) + len(CVXPY_LIST))
        * len(DETECTOR_CONFIGS)
        * n_spectra
    )
    print(f"\nTotal unfold calls: {total}")
    print(
        f"  qpsolvers: {len(QPSOLVERS_LIST)} solvers × {len(DETECTOR_CONFIGS)} detectors × {n_spectra} spectra"
    )
    print(
        f"  cvxpy: {len(CVXPY_LIST)} solvers × {len(DETECTOR_CONFIGS)} detectors × {n_spectra} spectra\n"
    )

    rows = []
    done = 0
    t0 = time.time()

    # QPSolvers
    for solver in QPSOLVERS_LIST:
        for det_name, (detector, readings_map) in detector_readings.items():
            for place, readings in readings_map.items():
                dose_ref = ref_doses[place]
                spectrum, status = try_unfold_qpsolvers(
                    detector, readings, solver
                )
                if spectrum is not None:
                    dose_unfolded = compute_dose(spectrum)
                else:
                    dose_unfolded = {g: 0.0 for g in DOSE_GEOMETRIES}

                for geom in DOSE_GEOMETRIES:
                    rows.append(
                        {
                            "detector": det_name,
                            "place": place,
                            "solver_type": "qpsolvers",
                            "solver_name": solver,
                            "geometry": geom,
                            "dose_ref": dose_ref.get(geom, 0.0),
                            "dose_unfolded": dose_unfolded.get(geom, 0.0),
                            "status": status,
                        }
                    )

                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate / 60
                    print(
                        f"  [{done}/{total}] qpsolvers/{solver} + {det_name} | {place} | {rate:.1f} calls/s | ETA: {eta:.0f} min"
                    )

    # CVXPY
    for solver in CVXPY_LIST:
        for det_name, (detector, readings_map) in detector_readings.items():
            for place, readings in readings_map.items():
                dose_ref = ref_doses[place]
                spectrum, status = try_unfold_cvxpy(detector, readings, solver)
                if spectrum is not None:
                    dose_unfolded = compute_dose(spectrum)
                else:
                    dose_unfolded = {g: 0.0 for g in DOSE_GEOMETRIES}

                for geom in DOSE_GEOMETRIES:
                    rows.append(
                        {
                            "detector": det_name,
                            "place": place,
                            "solver_type": "cvxpy",
                            "solver_name": solver,
                            "geometry": geom,
                            "dose_ref": dose_ref.get(geom, 0.0),
                            "dose_unfolded": dose_unfolded.get(geom, 0.0),
                            "status": status,
                        }
                    )

                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate / 60
                    print(
                        f"  [{done}/{total}] cvxpy/{solver} + {det_name} | {place} | {rate:.1f} calls/s | ETA: {eta:.0f} min"
                    )

    elapsed = time.time() - t0
    print(
        f"\nDone: {len(rows)} rows ({done} unfold calls) in {elapsed / 60:.1f} min"
    )

    # Save
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "OK"]
    err = df[df["status"] != "OK"]
    print(f"OK: {len(ok)}  |  ERROR: {len(err)}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
