from __future__ import annotations

import numpy as np

from bssunfold import Detector


class TestEnsemble:
    """Tests for unfold_ensemble module."""

    def _make_system(self, n_det=7, n_ene=50):
        np.random.seed(42)
        A = np.random.rand(n_det, n_ene) + 0.1
        x_true = np.random.rand(n_ene) + 0.1
        b = A @ x_true
        return A, b, x_true

    def test_solve_ensemble_weighted_average(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble

        A, b, _ = self._make_system()
        result, meta = solve_ensemble(A, b, combination="weighted_average")
        assert result.shape == (A.shape[1],)
        assert np.all(result >= 0)
        assert "method_names" in meta

    def test_solve_ensemble_median(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble

        A, b, _ = self._make_system()
        result, meta = solve_ensemble(A, b, combination="median")
        assert result.shape == (A.shape[1],)

    def test_solve_ensemble_trimmed_mean(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble

        A, b, _ = self._make_system()
        result, meta = solve_ensemble(
            A, b, combination="trimmed_mean", trim_fraction=0.25
        )
        assert result.shape == (A.shape[1],)

    def test_solve_ensemble_best_residual(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble

        A, b, _ = self._make_system()
        result, meta = solve_ensemble(A, b, combination="best_residual")
        assert result.shape == (A.shape[1],)

    def test_solve_ensemble_custom_methods(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble
        from bssunfold.core.unfold_landweber import solve_landweber
        from bssunfold.core.unfold_mlem import solve_mlem

        A, b, _ = self._make_system()
        methods = [
            (solve_mlem, {"max_iterations": 10, "tolerance": 1e-3}),
            (solve_landweber, {"max_iterations": 10, "tolerance": 1e-3}),
        ]
        result, meta = solve_ensemble(A, b, methods=methods)
        assert result.shape == (A.shape[1],)

    def test_solve_ensemble_custom_weights(self):
        from bssunfold.core.unfold_ensemble import solve_ensemble

        A, b, _ = self._make_system()
        result, meta = solve_ensemble(
            A, b, weights=np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        )
        assert result.shape == (A.shape[1],)

    def test_unfold_ensemble_via_detector(self):
        d = Detector()
        readings = {name: 1.0 for name in d.detector_names}
        result = d.unfold_ensemble(readings)
        assert "spectrum" in result
        assert len(result["spectrum"]) == d.n_energy_bins


class TestIterativeRefinement:
    """Tests for unfold_iterative_refinement module."""

    def _make_system(self, n_det=7, n_ene=50):
        np.random.seed(42)
        A = np.random.rand(n_det, n_ene) + 0.1
        x_true = np.random.rand(n_ene) + 0.1
        b = A @ x_true
        return A, b, x_true

    def test_solve_iterative_refinement_basic(self):
        from bssunfold.core.unfold_iterative_refinement import (
            solve_iterative_refinement,
        )

        A, b, _ = self._make_system()
        result, meta = solve_iterative_refinement(A, b)
        assert result.shape == (A.shape[1],)
        assert np.all(result >= 0)
        assert "alpha" in meta
        assert "first_pass_residual" in meta

    def test_solve_iterative_refinement_custom_methods(self):
        from bssunfold.core.unfold_cgls import solve_cgls
        from bssunfold.core.unfold_iterative_refinement import (
            solve_iterative_refinement,
        )
        from bssunfold.core.unfold_mlem import solve_mlem

        A, b, _ = self._make_system()
        result, meta = solve_iterative_refinement(
            A,
            b,
            first_pass_solver=solve_mlem,
            second_pass_solver=solve_cgls,
            first_pass_kwargs={"max_iterations": 10},
            second_pass_kwargs={"max_iterations": 10},
        )
        assert result.shape == (A.shape[1],)

    def test_solve_iterative_refinement_fixed_alpha(self):
        from bssunfold.core.unfold_iterative_refinement import (
            solve_iterative_refinement,
        )

        A, b, _ = self._make_system()
        result, meta = solve_iterative_refinement(A, b, alpha=0.5)
        assert result.shape == (A.shape[1],)
        assert meta["alpha"] == 0.5

    def test_unfold_iterative_refinement_via_detector(self):
        d = Detector()
        readings = {name: 1.0 for name in d.detector_names}
        result = d.unfold_iterative_refinement(readings)
        assert "spectrum" in result
        assert len(result["spectrum"]) == d.n_energy_bins
