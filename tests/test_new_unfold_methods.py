"""Tests for the new unfolding methods merged from the feature branches.

Covers the ODL-style PDHG / Douglas-Rachford solvers (pure-NumPy
implementation), the IMAXED / AMAXED / AMAXED-Regularization detectors
wrappers, and the QUBO / zfit backends (which require optional packages
and are skipped when those are not installed).
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import (
    solve_odl_douglas_rachford,
    solve_odl_pdhg,
)


def _synthetic_problem(m=10, n=20, seed=42):
    rng = np.random.default_rng(seed)
    centers = np.linspace(0.1, 0.9, m)
    x = np.linspace(0, 1, n)
    A = np.zeros((m, n))
    for i, c in enumerate(centers):
        A[i] = np.exp(-0.5 * ((x - c) / 0.08) ** 2)
    A += 0.01 * rng.standard_normal((m, n))
    A = np.maximum(A, 0)
    true = 0.5 * np.exp(-0.5 * ((x - 0.3) / 0.1) ** 2) + 0.2
    b = A @ true + 0.01 * rng.standard_normal(m)
    return A, b, true, n


class TestODLSolvers:
    """PDHG and Douglas-Rachford are pure-NumPy: no optional dependencies."""

    def test_pdhg_reduces_residual(self):
        A, b, true, n = _synthetic_problem()
        x, it, conv = solve_odl_pdhg(A, b, max_iterations=400)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))
        assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 0.1

    def test_pdhg_nonnegativity(self):
        A, b, true, n = _synthetic_problem()
        x, _, _ = solve_odl_pdhg(A, b, max_iterations=400, nonnegativity=True)
        assert x.min() >= -1e-9

    def test_pdhg_no_tv_matches_least_squares(self):
        A, b, true, n = _synthetic_problem()
        x, _, _ = solve_odl_pdhg(
            A, b, max_iterations=2000, use_tv=False, nonnegativity=False
        )
        # Without TV/nonnegativity PDHG minimizes ||A x - b||^2, so the data
        # misfit must reach (near) the least-squares optimum. A is
        # rank-deficient, so compare the misfit magnitude, not the vector.
        res_pdhg = np.linalg.norm(A @ x - b)
        assert res_pdhg < 1e-4

    def test_douglas_rachford_runs(self):
        A, b, true, n = _synthetic_problem()
        x, it, conv = solve_odl_douglas_rachford(A, b, max_iterations=600)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))
        # Douglas-Rachford is a different splitting; require a finite, reduced
        # residual rather than a tight tolerance.
        assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 0.6

    def test_douglas_rachford_nonnegativity(self):
        A, b, true, n = _synthetic_problem()
        x, _, _ = solve_odl_douglas_rachford(
            A, b, max_iterations=400, nonnegativity=True
        )
        assert x.min() >= -1e-9


class TestDetectorWrappersNoDeps:
    """Wrappers that need no optional packages."""

    def _readings(self, d):
        rng = np.random.default_rng(0)
        return {
            d.detector_names[i]: float(50 + 5 * rng.standard_normal())
            for i in range(d.n_detectors)
        }

    def test_unfold_imaxed(self):
        d = Detector()
        r = d.unfold_imaxed(self._readings(d), max_iterations=50)
        assert "spectrum" in r
        assert len(r["spectrum"]) == d.n_energy_bins

    def test_unfold_amaxed(self):
        d = Detector()
        r = d.unfold_amaxed(self._readings(d), max_iterations=50)
        assert "spectrum" in r
        assert len(r["spectrum"]) == d.n_energy_bins

    def test_unfold_amaxed_regularization(self):
        d = Detector()
        r = d.unfold_amaxed_regularization(self._readings(d), max_iterations=50)
        assert "spectrum" in r
        assert len(r["spectrum"]) == d.n_energy_bins

    def test_unfold_odl_pdhg(self):
        d = Detector()
        r = d.unfold_odl_pdhg(self._readings(d), max_iterations=80)
        assert "spectrum" in r
        assert len(r["spectrum"]) == d.n_energy_bins

    def test_unfold_odl_douglas_rachford(self):
        d = Detector()
        r = d.unfold_odl_douglas_rachford(self._readings(d), max_iterations=80)
        assert "spectrum" in r
        assert len(r["spectrum"]) == d.n_energy_bins


@pytest.mark.qubo
class TestQUBOBackend:
    def test_unfold_qubo_runs(self):
        pytest.importorskip("pyqubo")
        pytest.importorskip("dwave.samplers")
        d = Detector()
        rng = np.random.default_rng(0)
        readings = {
            d.detector_names[i]: float(50 + 5 * rng.standard_normal())
            for i in range(d.n_detectors)
        }
        r = d.unfold_qubo(readings, max_iterations=50, num_reads=2)
        assert "spectrum" in r


@pytest.mark.zfit
class TestZfitBackend:
    def test_unfold_zfit_runs(self):
        pytest.importorskip("zfit")
        d = Detector()
        rng = np.random.default_rng(0)
        readings = {
            d.detector_names[i]: float(50 + 5 * rng.standard_normal())
            for i in range(d.n_detectors)
        }
        r = d.unfold_zfit(readings, max_iterations=50)
        assert "spectrum" in r
