"""Tests for the optimization-course methods (MIPT OPTIMIZATION-METHODS-COURSE).

Covers the methods added from the MIPT optimization course curriculum:

- projected gradient descent (lecture 9, homework 14)
- Frank-Wolfe conditional gradient (lecture 9)
- mirror descent (lecture 10, homework 16)
- ADMM (lecture 11, homework 18)
- L-BFGS-B quasi-Newton (lecture 7, homework 10)
- coordinate descent (lecture 15)
- subgradient methods (lecture 8, homework 12)
- extragradient for saddle problems (lecture 13, homework 20)
- 1D optimization: golden section / dichotomy / Brent (lecture 1, homework 1)
- variance-reduced Monte-Carlo uncertainty (lecture 14)
- Lagrange duality / KKT diagnostics (seminars 9-10)
"""

import numpy as np
import pytest
from scipy.optimize import nnls

from bssunfold import Detector
from bssunfold.core import (
    brent_minimize,
    dichotomy_minimize,
    golden_section_minimize,
    nnls_duality_gap,
    nnls_kkt_residuals,
    project_onto_set,
    select_regularization_1d,
    solve_admm,
    solve_coordinate_descent,
    solve_extragradient,
    solve_frank_wolfe,
    solve_lbfgsb,
    solve_mirror_descent,
    solve_pgd,
    solve_subgradient,
)
from bssunfold.core._montecarlo import monte_carlo_uncertainty


@pytest.fixture
def detector() -> Detector:
    """Default Detector instance with default response functions."""
    return Detector()


@pytest.fixture
def system():
    """Small synthetic well-posed unfolding system."""
    rng = np.random.default_rng(7)
    m, n = 8, 15
    A = np.abs(rng.standard_normal((m, n))) * np.exp(
        -np.abs(rng.standard_normal((m, n)))
    )
    x_true = np.exp(-np.linspace(0, 3, n)) * 100
    b = A @ x_true
    return A, b, x_true


def fit_ratio(A, b, x) -> float:
    """Data-fit objective relative to the trivial zero-spectrum value."""
    return 0.5 * float(np.sum((A @ x - b) ** 2)) / max(0.5 * float(b @ b), 1e-30)


# ============================================================================
# 1D optimization building blocks (lecture 1, homework 1)
# ============================================================================


class TestOneDimensionalMinimizers:
    def _make_unimodal(self):
        return lambda t: (t - 1.3) ** 2 * np.exp(t)

    @pytest.mark.parametrize(
        "minimize", [golden_section_minimize, dichotomy_minimize, brent_minimize]
    )
    def test_finds_minimum(self, minimize):
        f = self._make_unimodal()
        t, f_opt = minimize(f, -1.0, 5.0, tolerance=1e-10)
        assert abs(t - 1.3) < 1e-6
        assert f_opt <= f(1.3) + 1e-10

    def test_brent_faster_than_golden(self):
        f = self._make_unimodal()
        calls = {"golden": 0, "brent": 0}

        def f_golden(t):
            calls["golden"] += 1
            return f(t)

        def f_brent(t):
            calls["brent"] += 1
            return f(t)

        golden_section_minimize(f_golden, -1.0, 5.0, tolerance=1e-9)
        brent_minimize(f_brent, -1.0, 5.0, tolerance=1e-9)
        assert calls["brent"] < calls["golden"]

    def test_degenerate_bracket(self):
        f = self._make_unimodal()
        for minimize in (golden_section_minimize, dichotomy_minimize, brent_minimize):
            t, f_opt = minimize(f, 2.0, 2.0)
            assert t == 2.0
            assert f_opt == pytest.approx(f(2.0))

    @pytest.mark.parametrize(
        "minimize", [golden_section_minimize, dichotomy_minimize, brent_minimize]
    )
    def test_infinite_bounds_raise(self, minimize):
        with pytest.raises(ValueError, match="finite bounds"):
            minimize(lambda t: t, 0.0, np.inf)


# ============================================================================
# Projected gradient descent (lecture 9)
# ============================================================================


class TestProjectOntoSet:
    def test_nonnegative(self):
        x = np.array([-1.0, 0.5, 2.0])
        assert np.allclose(project_onto_set(x, "nonnegative"), [0, 0.5, 2])

    def test_box(self):
        x = np.array([-1.0, 0.5, 2.0])
        assert np.allclose(project_onto_set(x, "box", x_max=1.0), [0, 0.5, 1])

    def test_simplex_preserves_total(self):
        x = np.array([0.2, 1.7, -0.4])
        p = project_onto_set(x, "simplex", total_fluence=2.0)
        assert p.sum() == pytest.approx(2.0)
        assert np.all(p >= 0)

    def test_simplex_requires_fluence(self):
        with pytest.raises(ValueError, match="total_fluence"):
            project_onto_set(np.ones(3), "simplex")

    def test_unknown_constraint(self):
        with pytest.raises(ValueError, match="Unknown constraint"):
            project_onto_set(np.ones(3), "sphere")


class TestPGD:
    def test_basic_convergence(self, system):
        A, b, _ = system
        x, it, conv = solve_pgd(A, b, np.zeros(15), max_iterations=20000)
        assert conv
        assert np.all(x >= 0)
        # data fit within 1e-4 of the trivial value (NNLS itself fits
        # noiseless systems exactly; first-order methods stop earlier)
        assert fit_ratio(A, b, x) <= 1e-5

    def test_simplex_preserves_fluence(self, system):
        A, b, _ = system
        F = 100.0
        x, _, _ = solve_pgd(
            A, b, np.full(15, F / 15), constraint="simplex", total_fluence=F
        )
        assert x.sum() == pytest.approx(F, rel=1e-9)
        assert np.all(x >= 0)

    def test_box_respects_bounds(self, system):
        A, b, _ = system
        x, _, _ = solve_pgd(
            A, b, np.zeros(15), constraint="box", x_max=5.0, max_iterations=5000
        )
        assert np.all(x <= 5.0 + 1e-12)
        assert np.all(x >= 0)

    def test_backtracking(self, system):
        A, b, _ = system
        x, it, conv = solve_pgd(
            A, b, np.zeros(15), backtracking=True, max_iterations=20000
        )
        assert fit_ratio(A, b, x) <= 1e-5

    def test_invalid_constraint(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="Unknown constraint"):
            solve_pgd(A, b, np.zeros(15), constraint="ball")


# ============================================================================
# Mirror descent (lecture 10)
# ============================================================================


class TestMirrorDescent:
    def test_entropy_preserves_fluence(self, system):
        A, b, _ = system
        x0 = np.full(15, 10.0)
        x, _, _ = solve_mirror_descent(A, b, x0, mirror_map="entropy")
        assert x.sum() == pytest.approx(x0.sum(), rel=1e-8)
        assert np.all(x >= 0)

    def test_entropy_beats_uniform_start(self, system):
        A, b, x_true = system
        x0 = np.full(15, x_true.sum() / 15)
        x, _, _ = solve_mirror_descent(A, b, x0, mirror_map="entropy")
        # data fit improves on the uniform start
        assert 0.5 * np.sum((A @ x - b) ** 2) <= 0.5 * np.sum((A @ x0 - b) ** 2)

    @pytest.mark.parametrize("mirror_map", ["log", "l2", "pnorm"])
    def test_other_maps_stay_positive(self, system, mirror_map):
        A, b, _ = system
        x0 = np.full(15, 1.0)
        x, _, _ = solve_mirror_descent(A, b, x0, mirror_map=mirror_map)
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)

    def test_unknown_map_raises(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="mirror_map"):
            solve_mirror_descent(A, b, np.ones(15), mirror_map="cosine")

    def test_fixed_step(self, system):
        A, b, _ = system
        x0 = np.full(15, 10.0)
        x, it, _ = solve_mirror_descent(
            A, b, x0, mirror_map="entropy", step_size=1e-4, max_iterations=50
        )
        assert it == 50
        assert np.all(np.isfinite(x))


# ============================================================================
# Frank-Wolfe (lecture 9)
# ============================================================================


class TestFrankWolfe:
    def test_fluence_preserved_and_gap_converges(self, system):
        A, b, _ = system
        F = float(np.sum(b)) / max(np.mean(A), 1e-30)
        x, it, conv = solve_frank_wolfe(A, b, np.full(15, F / 15), total_fluence=F)
        assert x.sum() == pytest.approx(F, rel=1e-9)
        assert np.all(x >= 0)
        assert conv

    def test_away_steps_not_worse(self, system):
        A, b, _ = system
        F = 1000.0
        x0 = np.full(15, F / 15)

        def f(z):
            return 0.5 * np.sum((A @ z - b) ** 2)

        x_plain, _, _ = solve_frank_wolfe(
            A, b, x0, total_fluence=F, away_steps=False, max_iterations=300
        )
        x_away, _, _ = solve_frank_wolfe(
            A, b, x0, total_fluence=F, away_steps=True, max_iterations=300
        )
        assert f(x_away) <= f(x_plain) + 1e-9

    def test_negative_fluence_raises(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="total_fluence"):
            solve_frank_wolfe(A, b, np.ones(15), total_fluence=-1.0)


# ============================================================================
# ADMM (lecture 11)
# ============================================================================


class TestADMM:
    def test_degenerate_equals_nnls(self, system):
        A, b, _ = system
        x, it, conv = solve_admm(A, b, np.zeros(15), max_iterations=5)
        ref, _ = nnls(A, b)
        assert np.allclose(x, ref)
        assert conv

    def test_nonnegativity_exact(self, system):
        A, b, _ = system
        x, _, _ = solve_admm(A, b, np.zeros(15), l1_penalty=1.0, max_iterations=100)
        assert np.all(x >= 0)

    def test_tv_smooths(self, system):
        A, b, _ = system
        rng = np.random.default_rng(3)
        b_noisy = b * (1 + 0.02 * rng.standard_normal(b.size))
        x_plain, _, _ = solve_admm(
            A, b_noisy, np.zeros(15), max_iterations=300, adaptive_rho=True
        )
        x_tv, _, _ = solve_admm(
            A, b_noisy, np.zeros(15), tv_penalty=50.0, max_iterations=300
        )

        def tv(z):
            return np.sum(np.abs(np.diff(z)))

        assert tv(x_tv) < tv(x_plain)

    def test_l1_sparsifies(self, system):
        A, b, _ = system
        x0, _, _ = solve_admm(A, b, np.zeros(15), max_iterations=200)
        x_l1, _, _ = solve_admm(
            A, b, np.zeros(15), l1_penalty=0.05 * float(np.max(b)), max_iterations=200
        )
        assert np.sum(x_l1 == 0) >= np.sum(x0 == 0)

    def test_fixed_rho(self, system):
        A, b, _ = system
        x, _, _ = solve_admm(
            A, b, np.zeros(15), rho=1.0, adaptive_rho=False, max_iterations=200
        )
        assert np.all(np.isfinite(x)) and np.all(x >= 0)


# ============================================================================
# L-BFGS-B (lecture 7)
# ============================================================================


class TestLbfgsb:
    def test_matches_nnls_objective(self, system):
        A, b, _ = system
        x, it, conv = solve_lbfgsb(A, b, np.zeros(15))
        ref, _ = nnls(A, b)
        assert 0.5 * np.sum((A @ x - b) ** 2) <= 0.5 * np.sum(
            (A @ ref - b) ** 2
        ) + 1e-6 * max(0.5 * float(b @ b), 1.0)
        assert conv

    def test_box_bounds(self, system):
        A, b, _ = system
        x, _, _ = solve_lbfgsb(A, b, np.zeros(15), x_max=10.0)
        assert np.all(x <= 10.0 + 1e-9)
        assert np.all(x >= 0)

    def test_smoothness_reduces_curvature(self, system):
        A, b, _ = system
        rng = np.random.default_rng(5)
        b_noisy = b * (1 + 0.01 * rng.standard_normal(b.size))
        x0, _, _ = solve_lbfgsb(A, b_noisy, np.zeros(15))
        x_s, _, _ = solve_lbfgsb(A, b_noisy, np.zeros(15), smoothness=1e4)

        def d2(z):
            return np.sum(np.diff(z, 2) ** 2)

        assert d2(x_s) <= d2(x0) + 1e-9

    def test_history_parameter(self, system):
        A, b, _ = system
        x, _, _ = solve_lbfgsb(A, b, np.zeros(15), lbfgs_history=3)
        assert fit_ratio(A, b, x) <= 1e-8


# ============================================================================
# Coordinate descent (lecture 15)
# ============================================================================


class TestCoordinateDescent:
    def test_matches_nnls_objective(self, system):
        A, b, _ = system
        x, it, conv = solve_coordinate_descent(A, b, np.zeros(15))
        ref, _ = nnls(A, b)

        def f(z):
            return 0.5 * np.sum((A @ z - b) ** 2)

        assert f(x) <= f(ref) + 1e-6
        assert conv

    def test_random_selection(self, system):
        A, b, _ = system
        x, it, conv = solve_coordinate_descent(
            A, b, np.zeros(15), selection="random", random_state=0
        )
        ref, _ = nnls(A, b)

        def f(z):
            return 0.5 * np.sum((A @ z - b) ** 2)

        assert f(x) <= f(ref) + 1e-6
        assert conv

    def test_l1_sparsifies(self, system):
        A, b, _ = system
        x0, _, _ = solve_coordinate_descent(A, b, np.zeros(15))
        x_l1, _, _ = solve_coordinate_descent(
            A, b, np.zeros(15), l1_penalty=0.1 * float(np.max(b))
        )
        assert np.sum(x_l1 == 0) >= np.sum(x0 == 0)

    def test_ridge_penalty(self, system):
        A, b, _ = system
        x, _, _ = solve_coordinate_descent(A, b, np.zeros(15), l2_penalty=10.0)
        assert np.all(np.isfinite(x)) and np.all(x >= 0)

    def test_invalid_selection(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="selection"):
            solve_coordinate_descent(A, b, np.zeros(15), selection="greedy")


# ============================================================================
# Subgradient methods (lecture 8)
# ============================================================================


class TestSubgradient:
    def test_policies_run_and_stay_nonnegative(self, system):
        A, b, _ = system
        for policy in ("polyak", "diminishing", "fixed"):
            x, it, _ = solve_subgradient(
                A,
                b,
                np.zeros(15),
                step_policy=policy,
                max_iterations=200,
            )
            assert np.all(x >= 0)
            assert np.all(np.isfinite(x))

    def test_best_iterate_improves_data_fit(self, system):
        A, b, _ = system
        f0 = 0.5 * np.sum(b**2)
        x, _, _ = solve_subgradient(A, b, np.zeros(15), max_iterations=500)
        assert 0.5 * np.sum((A @ x - b) ** 2) < f0

    def test_invalid_policy(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="step_policy"):
            solve_subgradient(A, b, np.zeros(15), step_policy="nesterov")


# ============================================================================
# Extragradient (lecture 13)
# ============================================================================


class TestExtragradient:
    def test_basic(self, system):
        A, b, _ = system
        x, it, conv = solve_extragradient(
            A, b, np.zeros(15), max_iterations=20000, noise_level=0.0
        )
        assert np.all(x >= 0)
        assert fit_ratio(A, b, x) <= 1e-5

    def test_noise_ball_limits_residual(self, system):
        A, b, _ = system
        rng = np.random.default_rng(11)
        b_noisy = b * (1 + 0.05 * rng.standard_normal(b.size))
        delta = 0.05 * float(np.linalg.norm(b_noisy))
        x, _, _ = solve_extragradient(
            A, b_noisy, np.zeros(15), noise_level=0.05, max_iterations=5000
        )
        # the robust formulation trades data fit for bounded sensitivity;
        # the achieved residual must stay comparable to the noise ball
        assert np.linalg.norm(A @ x - b_noisy) <= 5.0 * delta + 1e-9

    def test_fixed_step(self, system):
        A, b, _ = system
        x, _, _ = solve_extragradient(
            A, b, np.zeros(15), step_size=1e-6, max_iterations=100
        )
        assert np.all(np.isfinite(x))


# ============================================================================
# Duality / KKT diagnostics (seminars 9-10)
# ============================================================================


class TestDualDiagnostics:
    def test_gap_at_nnls_solution_is_tiny(self, system):
        A, b, _ = system
        x, _ = nnls(A, b)
        cert = nnls_duality_gap(A, b, x)
        assert cert["duality_gap"] <= 1e-6 * max(1.0, cert["primal_value"])

    def test_gap_positive_away_from_optimum(self, system):
        A, b, _ = system
        cert = nnls_duality_gap(A, b, np.ones(15))
        assert cert["duality_gap"] > 1e-6

    def test_kkt_residuals(self, system):
        A, b, _ = system
        x, _ = nnls(A, b)
        res = nnls_kkt_residuals(A, b, x, tolerance=1e-12)
        assert res["stationarity"] <= 1e-6 * max(1.0, np.linalg.norm(b))
        assert res["feasibility"] == 0.0
        assert res["active_set"].dtype == bool


# ============================================================================
# 1D regularization-parameter search
# ============================================================================


class TestSelectRegularization1D:
    def test_gcv_brent(self, system):
        A, b, _ = system
        res = select_regularization_1d(A, b, criterion="gcv", search="brent")
        assert A.shape[1] in (8, 15)  # sanity on fixture shape usage
        assert 1e-8 <= res["best_lambda"] <= 1e4
        assert res["iterations"] >= 1
        assert np.all(res["spectrum"] >= 0)

    def test_discrepancy_golden(self, system):
        A, b, _ = system
        res = select_regularization_1d(
            A, b, criterion="discrepancy", search="golden", noise_level=0.01
        )
        assert 1e-8 <= res["best_lambda"] <= 1e4

    @pytest.mark.parametrize("search", ["brent", "golden", "dichotomy"])
    def test_all_searches(self, system, search):
        A, b, _ = system
        res = select_regularization_1d(A, b, criterion="predictive", search=search)
        assert res["search"] == search
        assert np.isfinite(res["criterion_value"])

    def test_custom_solver(self, system):
        A, b, _ = system

        from bssunfold.core import solve_lbfgsb

        def solver(A_, b_, regularization=0.0, **kw):
            x, _, _ = solve_lbfgsb(
                A_, b_, np.zeros(A_.shape[1]), regularization=regularization
            )
            return x

        res = select_regularization_1d(A, b, solve_func=solver, criterion="discrepancy")
        assert 1e-8 <= res["best_lambda"] <= 1e4

    def test_invalid_args(self, system):
        A, b, _ = system
        with pytest.raises(ValueError, match="criterion"):
            select_regularization_1d(A, b, criterion="aic")
        with pytest.raises(ValueError, match="search"):
            select_regularization_1d(A, b, search="pso")
        with pytest.raises(ValueError, match="lo"):
            select_regularization_1d(A, b, lo=4.0, hi=-8.0)


# ============================================================================
# Variance-reduced Monte-Carlo (lecture 14)
# ============================================================================


class _LinearUnfolder:
    """Deterministic linear 'unfolder' for MC statistics tests."""

    def __init__(self, A, x_base):
        self.A = A
        self.x_base = x_base

    def __call__(self, readings, **kwargs):
        b = np.array([readings[k] for k in sorted(readings)])
        return self.x_base + np.linalg.pinv(self.A) @ (b - self.A @ self.x_base)


class TestMonteCarloVarianceReduction:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="variance_reduction"):
            monte_carlo_uncertainty(
                lambda r: np.zeros(3),
                {"a": 1.0},
                noise_level=0.01,
                n_samples=4,
                n_energy_bins=3,
                variance_reduction="quasi",
            )

    def test_control_requires_matrix(self):
        with pytest.raises(ValueError, match="response_matrix"):
            monte_carlo_uncertainty(
                lambda r: np.zeros(3),
                {"a": 1.0},
                noise_level=0.01,
                n_samples=4,
                n_energy_bins=3,
                variance_reduction="control",
            )

    def test_control_reduces_variance(self):
        rng = np.random.default_rng(0)
        A = np.abs(rng.standard_normal((6, 12)))
        x_base = np.exp(-np.linspace(0, 3, 12)) * 10
        b = A @ x_base
        readings = {f"d{i}": float(v) for i, v in enumerate(b)}
        unfolder = _LinearUnfolder(A, x_base)

        common = dict(
            func=unfolder,
            readings=readings,
            noise_level=0.05,
            n_samples=40,
            n_energy_bins=12,
            random_state=42,
        )
        plain = monte_carlo_uncertainty(**common)
        ctrl = monte_carlo_uncertainty(
            **common, variance_reduction="control", response_matrix=A
        )
        assert np.mean(ctrl["spectrum_uncert_std"]) < np.mean(
            plain["spectrum_uncert_std"]
        )
        assert ctrl["variance_reduction_factor"] > 1.0

    def test_antithetic_pairs(self):
        A = np.abs(np.random.default_rng(1).standard_normal((4, 8)))
        x_base = np.ones(8)
        b = A @ x_base
        readings = {f"d{i}": float(v) for i, v in enumerate(b)}
        out = monte_carlo_uncertainty(
            func=_LinearUnfolder(A, x_base),
            readings=readings,
            noise_level=0.05,
            n_samples=20,
            n_energy_bins=8,
            random_state=0,
            variance_reduction="antithetic",
        )
        assert out["spectrum_uncert_all"].shape == (20, 8)
        # antithetic pairing cancels the linear part: mean close to base
        assert np.allclose(out["spectrum_uncert_mean"], x_base, atol=1e-6)

    def test_both_mode(self):
        A = np.abs(np.random.default_rng(2).standard_normal((5, 10)))
        x_base = np.ones(10)
        b = A @ x_base
        readings = {f"d{i}": float(v) for i, v in enumerate(b)}
        out = monte_carlo_uncertainty(
            func=_LinearUnfolder(A, x_base),
            readings=readings,
            noise_level=0.05,
            n_samples=16,
            n_energy_bins=10,
            random_state=0,
            variance_reduction="both",
            response_matrix=A,
        )
        assert "variance_reduction" in out
        assert np.allclose(out["spectrum_uncert_mean"], x_base, atol=1e-6)


# ============================================================================
# Detector integration
# ============================================================================


class TestDetectorIntegration:
    @pytest.fixture
    def readings(self, detector):
        n = len(detector.E_MeV)
        x_true = np.exp(-np.linspace(0, 3, n)) * 50
        return {
            name: float(np.dot(A, x_true))
            for name, A in detector.sensitivities.items()
            if name in detector.detector_names[:6]
        }

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("unfold_pgd", {"max_iterations": 200}),
            ("unfold_mirror_descent", {"max_iterations": 100}),
            ("unfold_frank_wolfe", {"max_iterations": 100}),
            ("unfold_admm", {"max_iterations": 50}),
            ("unfold_lbfgsb", {}),
            ("unfold_coordinate_descent", {"max_iterations": 200}),
            ("unfold_subgradient", {"max_iterations": 100}),
            ("unfold_extragradient", {"max_iterations": 100}),
        ],
    )
    def test_methods_smoke(self, detector, readings, method, kwargs):
        result = getattr(detector, method)(readings, **kwargs)
        assert result["spectrum"].shape == (len(detector.E_MeV),)
        assert np.all(result["spectrum"] >= 0)
        assert result["method"]
        assert "doserates" in result

    def test_pgd_duality_gap_attached(self, detector, readings):
        result = detector.unfold_pgd(readings)
        gap = result["duality_gap"]
        assert {"primal_value", "dual_value", "duality_gap", "relative_gap"} <= set(gap)
        assert gap["duality_gap"] >= 0

    def test_pgd_simplex_constraint(self, detector, readings):
        result = detector.unfold_pgd(
            readings, constraint="simplex", total_fluence=100.0, max_iterations=200
        )
        assert result["spectrum"].sum() == pytest.approx(100.0, rel=1e-6)

    def test_variance_reduction_passthrough(self, detector, readings):
        result = detector.unfold_pgd(
            readings,
            calculate_errors=True,
            n_montecarlo=6,
            variance_reduction="control",
            random_state=0,
        )
        assert result.get("variance_reduction") == "control"
        assert "spectrum_uncert_std" in result

    def test_max_neutron_energy(self, detector, readings):
        result = detector.unfold_pgd(readings, max_neutron_energy=5.0)
        # the full energy grid is retained; spectrum is zeroed above cutoff
        assert result["energy"].shape == (len(detector.E_MeV),)
        assert np.all(result["spectrum"][result["energy"] > 5.0] == 0.0)
