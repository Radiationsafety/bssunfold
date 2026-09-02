"""Tests for the bin-wise adaptive unfolding method (unfold_binned)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bssunfold import Detector

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def det() -> Detector:
    """Default Detector with GSF response functions."""
    return Detector()


@pytest.fixture
def dummy_lookup() -> dict:
    """Minimal bin lookup with 60 bins, 3 methods."""
    rng = np.random.default_rng(42)
    methods = ["method_a", "method_b", "method_c"]
    bin_to = {}
    for b in range(60):
        scores = sorted(
            [(m, float(rng.uniform(0, 0.01))) for m in methods],
            key=lambda x: x[1],
        )
        bin_to[b] = scores
    return {
        "bin_to_methods": bin_to,
        "unique_methods": methods,
        "n_bins": 60,
    }


@pytest.fixture
def dummy_readings(det: Detector) -> dict:
    """Readings for a Cf-252-like flat spectrum."""
    rng = np.random.default_rng(0)
    return {
        n: float(rng.uniform(0.1, 10.0))
        for n in det.detector_names
    }


# ── Test load/save bin lookup ─────────────────────────────────────────────

class TestLookupIO:
    def test_save_and_load(self, dummy_lookup):
        from bssunfold.core.unfold_binned import load_bin_lookup, save_bin_lookup

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_bin_lookup(dummy_lookup, path)
        loaded = load_bin_lookup(path)

        assert loaded["n_bins"] == 60
        assert set(loaded["unique_methods"]) == set(dummy_lookup["unique_methods"])
        assert len(loaded["bin_to_methods"]) == 60

        # Verify bin keys are int after load.
        for k in loaded["bin_to_methods"]:
            assert isinstance(k, int)

        Path(path).unlink()

    def test_load_nonexistent_raises(self):
        from bssunfold.core.unfold_binned import load_bin_lookup

        with pytest.raises(FileNotFoundError):
            load_bin_lookup("/nonexistent/path.json")

    def test_load_default_lookup(self):
        from bssunfold.core.unfold_binned import load_bin_lookup

        lookup = load_bin_lookup()
        assert lookup["n_bins"] == 60
        assert len(lookup["unique_methods"]) > 0
        assert len(lookup["bin_to_methods"]) == 60


# ── Test solve_binned ─────────────────────────────────────────────────────

class TestSolveBinned:
    def test_basic_assembly(self, dummy_lookup):
        from bssunfold.core.unfold_binned import solve_binned

        rng = np.random.default_rng(42)
        n_bins = 60
        m = 10

        A = rng.uniform(0.1, 1.0, (m, n_bins))
        x_ref = rng.uniform(1e-4, 1e-2, n_bins)
        b = A @ x_ref

        def mock_solver(A_mat, b_vec, x0=None, **kw):
            return x_ref + rng.normal(0, 1e-5, n_bins)

        methods = {
            "method_a": (mock_solver, {}),
            "method_b": (mock_solver, {}),
            "method_c": (mock_solver, {}),
        }

        spectrum, meta = solve_binned(A, b, dummy_lookup, methods)

        assert spectrum.shape == (n_bins,)
        assert np.all(np.isfinite(spectrum))
        assert np.sum(spectrum) > 0
        assert len(meta["successful_methods"]) == 3
        assert meta["method_map"].shape == (n_bins,)

    def test_fallback_when_no_method_succeeds(self, dummy_lookup):
        from bssunfold.core.unfold_binned import solve_binned

        rng = np.random.default_rng(42)
        n_bins = 60
        m = 10

        A = rng.uniform(0.1, 1.0, (m, n_bins))
        b = rng.uniform(0.1, 1.0, m)

        def failing_solver(A_mat, b_vec, x0=None, **kw):
            raise RuntimeError("always fails")

        methods = {"method_a": (failing_solver, {})}

        spectrum, meta = solve_binned(A, b, dummy_lookup, methods)

        assert spectrum.shape == (n_bins,)
        assert meta["successful_methods"] == []

    def test_timeout(self):
        from bssunfold.core.unfold_binned import solve_binned

        n_bins = 60
        m = 5
        rng = np.random.default_rng(42)
        A = rng.uniform(0.1, 1.0, (m, n_bins))
        b = rng.uniform(0.1, 1.0, m)

        import time

        def slow_solver(A_mat, b_vec, x0=None, **kw):
            time.sleep(10)
            return np.ones(n_bins)

        lookup = {
            "bin_to_methods": {i: [("slow", 0.0)] for i in range(n_bins)},
            "unique_methods": ["slow"],
            "n_bins": n_bins,
        }

        spectrum, meta = solve_binned(
            A, b, lookup, {"slow": (slow_solver, {})},
            timeout_per_method=0.5,
        )
        assert "slow" in meta["errors"] or meta["successful_methods"] == []

    def test_partial_success(self):
        from bssunfold.core.unfold_binned import solve_binned

        rng = np.random.default_rng(42)
        n_bins = 60
        m = 10
        A = rng.uniform(0.1, 1.0, (m, n_bins))
        x_ref = rng.uniform(1e-4, 1e-2, n_bins)
        b = A @ x_ref

        def good_solver(A_mat, b_vec, x0=None, **kw):
            return x_ref * (1 + 0.01 * rng.normal(size=n_bins))

        def bad_solver(A_mat, b_vec, x0=None, **kw):
            return np.zeros(n_bins)  # invalid output

        lookup = {
            "bin_to_methods": {
                i: [("good", 0.001), ("bad", 0.005)] for i in range(n_bins)
            },
            "unique_methods": ["good", "bad"],
            "n_bins": n_bins,
        }

        methods = {
            "good": (good_solver, {}),
            "bad": (bad_solver, {}),
        }

        spectrum, meta = solve_binned(A, b, lookup, methods)
        assert "good" in meta["successful_methods"]
        assert "bad" not in meta["successful_methods"]
        assert np.all(spectrum > 0)


# ── Test unfold_binned on Detector ────────────────────────────────────────

class TestUnfoldBinned:
    def test_detector_has_method(self, det: Detector):
        assert hasattr(det, "unfold_binned")
        assert callable(det.unfold_binned)

    def test_unfold_with_default_lookup(self, det: Detector, dummy_readings):
        """Test that unfold_binned runs with the shipped lookup."""
        result = det.unfold_binned(
            dummy_readings,
            timeout_per_method=5.0,
        )
        assert result["spectrum"] is not None
        assert result["spectrum"].shape == (det.n_energy_bins,)
        assert np.all(np.isfinite(result["spectrum"]))
        assert result["method"] == "binned"
        assert "method_map" in result
        assert "successful_methods" in result

    def test_unfold_with_custom_lookup(self, det: Detector, dummy_readings,
                                       dummy_lookup):
        """Test with a minimal custom lookup (only fast methods)."""
        # Use only one method to keep it fast.
        slim_lookup = {
            "bin_to_methods": {
                i: [("tsvd", 0.01)] for i in range(60)
            },
            "unique_methods": ["tsvd"],
            "n_bins": 60,
        }
        result = det.unfold_binned(
            dummy_readings,
            bin_lookup=slim_lookup,
            timeout_per_method=10.0,
        )
        assert result["spectrum"] is not None
        assert result["spectrum"].shape == (det.n_energy_bins,)
        assert "tsvd" in result["candidate_methods"]

    def test_result_has_standard_keys(self, det: Detector, dummy_readings):
        result = det.unfold_binned(dummy_readings, timeout_per_method=5.0)
        expected_keys = {
            "energy", "spectrum", "spectrum_absolute", "effective_readings",
            "residual", "residual_norm", "method", "doserates",
            "method_map", "successful_methods", "individual_spectra",
            "candidate_methods", "bin_lookup",
        }
        assert expected_keys.issubset(result.keys())

    def test_doserates_computed(self, det: Detector, dummy_readings):
        result = det.unfold_binned(dummy_readings, timeout_per_method=5.0)
        if result["spectrum"] is not None:
            assert result["doserates"] is not None
            assert isinstance(result["doserates"], dict)
            assert len(result["doserates"]) > 0


# ── Test build_bin_lookup ─────────────────────────────────────────────────

class TestBuildBinLookup:
    def test_build_from_synthetic_data(self):
        from bssunfold.core.unfold_binned import build_bin_lookup

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create synthetic reference CSV.
            n_bins = 10
            rng = np.random.default_rng(42)
            ref_df_data = {"key": [f"key_{i}" for i in range(5)]}
            for b in range(n_bins):
                ref_df_data[f"energy_{b+1}"] = rng.uniform(1e-4, 1e-2, 5)

            import pandas as pd
            ref_csv = tmpdir / "refs.csv"
            pd.DataFrame(ref_df_data).to_csv(ref_csv, index=False)

            # Create synthetic method NPZ files.
            spectra_dir = tmpdir / "spectra"
            spectra_dir.mkdir()

            for method in ["method_a", "method_b"]:
                data = {}
                for i in range(5):
                    data[f"key_{i}"] = (
                        np.array(ref_df_data["energy_1"])[i]
                        * np.ones(n_bins)
                        + rng.normal(0, 1e-5, n_bins)
                    )
                np.savez(spectra_dir / f"unfold_{method}.npz", **data)

            lookup = build_bin_lookup(spectra_dir, ref_csv, n_bins=n_bins, top_k=2)

            assert lookup["n_bins"] == n_bins
            assert len(lookup["unique_methods"]) <= 2
            assert len(lookup["bin_to_methods"]) == n_bins
