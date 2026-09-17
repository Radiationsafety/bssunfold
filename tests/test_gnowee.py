"""Tests for the Gnowee-based unfolding method.

Covers:

* the Lévy / TLF samplers in :mod:`bssunfold.core._gnowee`;
* the ``run_gnowee`` optimizer on a simple continuous benchmark;
* ``solve_gnowee`` (the core solver interface used by ``run_unfolding``);
* ``unfold_gnowee`` exposed both as a module-level function and as a
  ``Detector.unfold_gnowee`` method;
* hyper-parameter sanity checks and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_gnowee, unfold_gnowee
from bssunfold.core._gnowee import (
    GnoweeHeuristics,
    GnoweeSettings,
    levy,
    rejection_bounds,
    run_gnowee,
    simple_bounds,
    tlf,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def detector() -> Detector:
    return Detector()


@pytest.fixture
def readings(detector: Detector) -> dict[str, float]:
    return {detector.detector_names[0]: 100.0}


@pytest.fixture
def synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic under-determined BSS problem (5 detectors, 12 energy bins)."""
    rng = np.random.default_rng(7)
    n, m = 12, 5
    A = rng.uniform(0.1, 1.0, size=(m, n))
    x_true = np.zeros(n)
    x_true[3] = 1.0
    x_true[4] = 0.5
    x_true[5] = 0.3
    b = A @ x_true
    return A, b, x_true


# --------------------------------------------------------------------------- #
# Sampler tests                                                                #
# --------------------------------------------------------------------------- #
def test_levy_shape_1d():
    rng = np.random.default_rng(0)
    z = levy(5, alpha=1.5, gam=1.0, rng=rng)
    assert z.shape == (5,)
    assert np.all(np.isfinite(z))


def test_levy_shape_2d():
    rng = np.random.default_rng(1)
    z = levy(4, 3, alpha=1.5, rng=rng)  # levy(nc=4, nr=3) → shape (3, 4)
    assert z.shape == (3, 4)
    assert np.all(np.isfinite(z))


def test_levy_alpha_validation():
    with pytest.raises(ValueError, match="alpha"):
        levy(5, alpha=2.5)
    with pytest.raises(ValueError, match="alpha"):
        levy(5, alpha=0.1)


def test_levy_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        levy(5, gam=-1.0)


def test_tlf_range():
    z = tlf(5, 5, alpha=1.5, rng=np.random.default_rng(2))
    assert z.shape == (5, 5)
    assert np.all(z >= 0.0)
    assert np.all(z <= 1.0)


# --------------------------------------------------------------------------- #
# Boundary helpers                                                             #
# --------------------------------------------------------------------------- #
def test_simple_bounds_clips():
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    child = np.array([2.0, -3.0])
    out = simple_bounds(child, lb, ub)
    assert np.allclose(out, [1.0, -1.0])


def test_rejection_bounds_halves_until_feasible():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    parent = np.array([0.5, 0.5])
    # step that jumps far outside
    step = np.array([10.0, -10.0])
    child = parent + step
    out = rejection_bounds(parent, child, step.copy(), lb, ub)
    assert np.all(out >= lb - 1e-12)
    assert np.all(out <= ub + 1e-12)


# --------------------------------------------------------------------------- #
# Optimizer                                                                    #
# --------------------------------------------------------------------------- #
def test_run_gnowee_sphere_optimum():
    """Gnowee on the 5-D sphere gets within a few % of the optimum."""
    rng = np.random.default_rng(42)
    lb = np.full(5, -5.0)
    ub = np.full(5, 5.0)
    s = GnoweeSettings(
        population=20, max_gens=80, max_fevals=2000,
        stall_limit=150, verbose=False,
    )
    x, f, timeline = run_gnowee(
        lb, ub, lambda v: float(np.dot(v, v)), settings=s, rng=rng,
    )
    assert x.shape == (5,)
    assert f < 1.0
    assert len(timeline) >= 1


def test_run_gnowee_respects_bounds():
    """The optimizer never returns a solution outside the box."""
    rng = np.random.default_rng(123)
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    s = GnoweeSettings(population=10, max_gens=20, max_fevals=200)
    x, _, _ = run_gnowee(lb, ub, lambda v: float(np.sum(v**2)),
                         settings=s, rng=rng)
    assert np.all(x >= lb - 1e-12)
    assert np.all(x <= ub + 1e-12)


def test_run_gnowee_seed_solution_is_first_individual():
    """The seed solution is injected into the population when provided."""
    rng = np.random.default_rng(0)
    lb = np.full(3, -10.0)
    ub = np.full(3, 10.0)
    s = GnoweeSettings(population=8, max_gens=5, max_fevals=200)
    seed = np.array([0.1, 0.2, 0.3])
    x, f, _ = run_gnowee(lb, ub, lambda v: float(np.dot(v, v)),
                         settings=s, rng=rng, seed_solution=seed)
    # The seed is a great starting point — fitness should be small.
    assert f <= float(np.dot(seed, seed)) + 1e-6


# --------------------------------------------------------------------------- #
# solve_gnowee                                                                 #
# --------------------------------------------------------------------------- #
def test_solve_gnowee_returns_tuple(synthetic_problem):
    A, b, x_true = synthetic_problem
    spec, ne, conv, diag = solve_gnowee(
        A, b, x0=np.ones(A.shape[1]) * 0.1,
        population=15, max_gens=60, max_fevals=1500,
        stall_limit=120, smoothness_order=0, half_range=3.0,
        random_state=0,
    )
    assert isinstance(spec, np.ndarray)
    assert spec.shape == (A.shape[1],)
    assert np.all(spec >= 0)
    assert isinstance(ne, int) and ne >= 0
    assert isinstance(conv, bool)
    assert isinstance(diag, dict)
    assert "best_fitness" in diag
    assert "evaluations" in diag
    assert "generations" in diag


def test_solve_gnowee_drives_residual_down(synthetic_problem):
    """Gnowee achieves a sensible relative residual on a synthetic problem."""
    A, b, _ = synthetic_problem
    spec, _, _, _ = solve_gnowee(
        A, b, x0=np.ones(A.shape[1]) * 0.1,
        population=20, max_gens=120, max_fevals=4000,
        stall_limit=200, opt_conv_tol=1e-12,
        regularization=1e-4, smoothness_order=0, half_range=3.0,
        random_state=0,
    )
    rel_err = np.linalg.norm(A @ spec - b) / np.linalg.norm(b)
    # Under-determined problem (5 detectors, 12 bins) — 25% is the upper
    # bound of what deterministic methods (Landweber, MLEM) achieve here.
    assert rel_err < 0.25, f"rel_err={rel_err}"


def test_solve_gnowee_invalid_norm_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="norm"):
        solve_gnowee(A, b, norm=3)


def test_solve_gnowee_invalid_smoothness_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="smoothness"):
        solve_gnowee(A, b, smoothness_order=5)


def test_solve_gnowee_invalid_init_sampling_raises(synthetic_problem):
    A, b, _ = synthetic_problem
    with pytest.raises(ValueError, match="init_sampling"):
        solve_gnowee(A, b, init_sampling="nolh")


def test_solve_gnowee_reproducible(synthetic_problem):
    """Same random_state → identical solution."""
    A, b, _ = synthetic_problem
    kw = dict(population=10, max_gens=20, max_fevals=200,
              stall_limit=50, smoothness_order=0, random_state=42)
    s1, _, _, _ = solve_gnowee(A, b, x0=np.ones(A.shape[1]) * 0.1, **kw)
    s2, _, _, _ = solve_gnowee(A, b, x0=np.ones(A.shape[1]) * 0.1, **kw)
    assert np.allclose(s1, s2)


# --------------------------------------------------------------------------- #
# Detector.unfold_gnowee                                                       #
# --------------------------------------------------------------------------- #
def test_unfold_gnowee_basic(detector, readings):
    """Basic Detector.unfold_gnowee call returns a standardised result."""
    result = detector.unfold_gnowee(
        readings, population=12, max_gens=30, max_fevals=400,
        stall_limit=60, random_state=0, save_result=False,
    )
    assert isinstance(result, dict)
    assert result["method"] == "Gnowee"
    assert "energy" in result
    assert "spectrum" in result
    assert "residual_norm" in result
    assert isinstance(result["spectrum"], np.ndarray)
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(result["spectrum"] >= 0)
    assert isinstance(result["residual_norm"], float)
    # Metadata
    assert result["population"] == 12
    assert result["max_gens"] == 30
    assert result["regularization"] == pytest.approx(1e-2)
    assert result["norm"] == 2


def test_unfold_gnowee_reproducible(detector, readings):
    """Same random_state produces identical spectra."""
    r1 = detector.unfold_gnowee(
        readings, population=10, max_gens=15, max_fevals=200,
        stall_limit=40, random_state=7,
    )
    r2 = detector.unfold_gnowee(
        readings, population=10, max_gens=15, max_fevals=200,
        stall_limit=40, random_state=7,
    )
    assert np.allclose(r1["spectrum"], r2["spectrum"])


def test_unfold_gnowee_with_initial_spectrum(detector, readings):
    """A provided initial_spectrum is accepted and used as the population seed."""
    initial = np.ones(detector.n_energy_bins) * 0.5
    result = detector.unfold_gnowee(
        readings, initial_spectrum=initial,
        population=10, max_gens=15, max_fevals=200,
        stall_limit=40, random_state=0,
    )
    assert result["method"] == "Gnowee"
    assert np.all(result["spectrum"] >= 0)


def test_unfold_gnowee_invalid_norm_raises(detector, readings):
    with pytest.raises(ValueError, match="norm"):
        detector.unfold_gnowee(readings, norm=3)


def test_unfold_gnowee_max_neutron_energy_truncates(detector, readings):
    """max_neutron_energy zeroes the spectrum above the cutoff.

    ``_expand_result`` pads the result back to the full energy grid (so the
    user can plot/compare against the original Detector grid) but sets
    spectrum values above ``max_neutron_energy`` to zero.  We verify that
    invariant here.
    """
    e_max = float(detector.E_MeV[len(detector.E_MeV) // 2])
    result = detector.unfold_gnowee(
        readings, max_neutron_energy=e_max,
        population=8, max_gens=10, max_fevals=150,
        stall_limit=30, random_state=0,
    )
    energy = result["energy"]
    spectrum = result["spectrum"]
    assert len(energy) == detector.n_energy_bins
    # Everything strictly above e_max should be zero after expansion
    above = energy > e_max + 1e-9
    assert np.all(spectrum[above] == 0.0)


def test_unfold_gnowee_extra_diagnostics(detector, readings):
    """The result carries the Gnowee-specific diagnostic fields."""
    result = detector.unfold_gnowee(
        readings, population=8, max_gens=10, max_fevals=150,
        stall_limit=30, random_state=0,
    )
    for key in ("best_fitness", "evaluations", "generations",
                "alpha_levy", "gamma_levy", "frac_levy", "frac_elite",
                "frac_mutation", "scaling_factor", "init_sampling"):
        assert key in result, f"missing {key}"


# --------------------------------------------------------------------------- #
# Module-level unfold_gnowee                                                   #
# --------------------------------------------------------------------------- #
def test_unfold_gnowee_module_level(detector, readings):
    """The module-level function returns the same shape of result."""
    res = unfold_gnowee(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        population=8, max_gens=10, max_fevals=150,
        stall_limit=30, random_state=0,
    )
    assert res["method"] == "Gnowee"
    assert len(res["spectrum"]) == detector.n_energy_bins


def test_unfold_gnowee_empty_readings_raises(detector):
    with pytest.raises(ValueError, match="readings"):
        detector.unfold_gnowee({})


# --------------------------------------------------------------------------- #
# GnoweeHeuristics sanity                                                      #
# --------------------------------------------------------------------------- #
def test_gnowee_settings_validation():
    with pytest.raises(ValueError, match="frac_mutation"):
        GnoweeHeuristics(
            lb=np.zeros(3), ub=np.ones(3),
            objective=lambda v: 0.0,
            settings=GnoweeSettings(frac_mutation=-0.1),
        )
    with pytest.raises(ValueError, match="frac_elite"):
        GnoweeHeuristics(
            lb=np.zeros(3), ub=np.ones(3),
            objective=lambda v: 0.0,
            settings=GnoweeSettings(frac_elite=1.5),
        )
    with pytest.raises(ValueError, match="frac_levy"):
        GnoweeHeuristics(
            lb=np.zeros(3), ub=np.ones(3),
            objective=lambda v: 0.0,
            settings=GnoweeSettings(frac_levy=2.0),
        )


def test_gnowee_initialize_shapes():
    gh = GnoweeHeuristics(
        lb=np.zeros(4), ub=np.ones(4),
        objective=lambda v: float(np.sum(v**2)),
        settings=GnoweeSettings(init_sampling="lhc"),
        rng=np.random.default_rng(0),
    )
    samples = gh.initialize(15, "lhc")
    assert samples.shape == (15, 4)
    assert np.all(samples >= 0 - 1e-12)
    assert np.all(samples <= 1 + 1e-12)
    samples_r = gh.initialize(10, "random")
    assert samples_r.shape == (10, 4)


def test_gnowee_initialize_invalid_method():
    gh = GnoweeHeuristics(
        lb=np.zeros(3), ub=np.ones(3),
        objective=lambda v: 0.0,
    )
    with pytest.raises(ValueError, match="init_sampling"):
        gh.initialize(5, "nolh")


def test_gnowee_initialize_bounds_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        GnoweeHeuristics(
            lb=np.zeros(3), ub=np.zeros(4),
            objective=lambda v: 0.0,
        )
