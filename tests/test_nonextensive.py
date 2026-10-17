"""Tests for the non-extensive (Tsallis) parametric unfolding method.

Covers the three-component q-exponential model of ZhangSun et al.,
Applied Radiation and Isotopes (2026):

* the Tsallis q-exponential and its Boltzmann-Gibbs limit ``q -> 1``;
* the model layout (15 parameters, component energy windows, positivity);
* the core solver ``solve_nonextensive`` (synthetic recovery, bounds,
  ``relative`` weighting, global/local search, reproducibility, errors);
* the ``Detector.unfold_nonextensive`` integration;
* an IAEA Compendium end-to-end run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bssunfold import RF_GSF, Detector
from bssunfold.core import (
    E_FAST_MIN_EV,
    E_THERMAL_MAX_EV,
    NONEXTENSIVE_PARAM_BOUNDS,
    NONEXTENSIVE_PARAM_NAMES,
    nonextensive_model,
    solve_nonextensive,
    unfold_nonextensive,
)

_EV_PER_MEV = 1.0e6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector(RF_GSF)


@pytest.fixture()
def system(detector: Detector):
    names = detector.detector_names
    A = np.array([detector.sensitivities[n] for n in names])
    return A, detector.E_MeV, names


@pytest.fixture()
def simple_readings(detector: Detector) -> dict[str, float]:
    return {nm: float(i + 1) * 0.02 for i, nm in enumerate(detector.detector_names)}


@pytest.fixture(scope="module")
def iaea_readings(detector: Detector) -> dict[str, float]:
    repo_root = Path(__file__).resolve().parent.parent
    csv = pd.read_csv(
        repo_root
        / "tests"
        / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
    )
    return detector.get_effective_readings_for_spectra(
        csv[["E_MeV", "t4-14-s.txt_1"]]
    )


def _true_theta() -> np.ndarray:
    """A physically plausible parameter vector inside the bounds."""
    theta = np.zeros(len(NONEXTENSIVE_PARAM_NAMES))
    # per component: [log10C, beta, g, q, n]
    comps = {
        "th": [1.0, 0.5, 0.0, 1.2, 0.5],
        "epi": [0.5, 0.0, 0.1, 1.1, 0.3],
        "f": [0.2, 0.0, 0.5, 1.3, 0.2],
    }
    for i, name in enumerate(NONEXTENSIVE_PARAM_NAMES):
        comp = name.split("_", 1)[1]
        kind = name.split("_", 1)[0]
        theta[i] = comps[comp][["log10C", "beta", "g", "q", "n"].index(kind)]
    return theta


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------


class TestModel:
    def test_param_names_layout(self) -> None:
        assert len(NONEXTENSIVE_PARAM_NAMES) == 15
        assert NONEXTENSIVE_PARAM_NAMES[0] == "log10C_th"
        assert NONEXTENSIVE_PARAM_NAMES[-1] == "n_f"
        for name in NONEXTENSIVE_PARAM_NAMES:
            assert name in NONEXTENSIVE_PARAM_BOUNDS

    def test_nodes(self) -> None:
        assert E_THERMAL_MAX_EV == pytest.approx(0.5)
        assert E_FAST_MIN_EV == pytest.approx(1.0e5)

    def test_output_shape_and_positivity(self, detector: Detector) -> None:
        phi = nonextensive_model(detector.E_MeV, _true_theta())
        assert phi.shape == detector.E_MeV.shape
        assert np.all(np.isfinite(phi))
        assert np.all(phi >= 0.0)

    def test_wrong_theta_length_raises(self, detector: Detector) -> None:
        with pytest.raises(ValueError, match="theta must have 15"):
            nonextensive_model(detector.E_MeV, np.zeros(7))

    def test_q_one_matches_plain_exponential(self, detector: Detector) -> None:
        """At ``q = 1`` each component reduces to ``C exp(-u) E^-n``."""
        theta = _true_theta().copy()
        # force q = 1 for every component
        for i, name in enumerate(NONEXTENSIVE_PARAM_NAMES):
            if name.startswith("q_"):
                theta[i] = 1.0

        phi = nonextensive_model(detector.E_MeV, theta)

        E_eV = detector.E_MeV * _EV_PER_MEV
        expected = np.zeros_like(E_eV)
        bounds = {
            "th": E_eV < E_THERMAL_MAX_EV,
            "epi": (E_eV >= E_THERMAL_MAX_EV) & (E_eV < E_FAST_MIN_EV),
            "f": E_eV >= E_FAST_MIN_EV,
        }
        for comp in ("th", "epi", "f"):
            sl = [
                i for i, n in enumerate(NONEXTENSIVE_PARAM_NAMES)
                if n.endswith(f"_{comp}")
            ]
            log10C, beta, g, q, n = theta[sl]
            u = beta * E_eV ** (-0.5) + g * E_eV
            m = bounds[comp]
            expected[m] = (10.0**log10C) * np.exp(-u[m]) * E_eV[m] ** (-n)

        np.testing.assert_allclose(phi, expected, rtol=1e-9, atol=1e-30)

    def test_component_windows_are_independent(self, detector: Detector) -> None:
        base = _true_theta()
        shifted = base.copy()
        # bump only the thermal amplitude
        shifted[0] += 1.0
        phi_a = nonextensive_model(detector.E_MeV, base)
        phi_b = nonextensive_model(detector.E_MeV, shifted)

        E_eV = detector.E_MeV * _EV_PER_MEV
        above = E_eV >= E_THERMAL_MAX_EV
        # bins at/above the thermal node are untouched
        np.testing.assert_array_equal(phi_a[above], phi_b[above])
        assert np.any(phi_a[~above] != phi_b[~above])

    def test_default_vector_is_flat(self, detector: Detector) -> None:
        """The neutral start (q=1.1, beta=g=n=0, C=1) is a flat unit spectrum."""
        from bssunfold.core.unfold_nonextensive import _default_theta

        phi = nonextensive_model(detector.E_MeV, _default_theta())
        assert np.allclose(phi, 1.0)


# ---------------------------------------------------------------------------
# 2. Core solver
# ---------------------------------------------------------------------------


class TestSolve:
    def test_local_recovers_synthetic_readings(self, system) -> None:
        A, E, _ = system
        theta_true = _true_theta()
        b = A @ nonextensive_model(E, theta_true)

        spectrum, success, _msg, nfev, params = solve_nonextensive(
            A, b, E, search="local",
            initial_params=dict(zip(NONEXTENSIVE_PARAM_NAMES, theta_true)),
        )
        assert spectrum.shape == E.shape
        assert np.all(np.isfinite(spectrum)) and np.all(spectrum >= 0)
        assert params["max_relative_deviation"] < 1e-6
        assert nfev > 0

    def test_global_runs_and_is_finite(self, system) -> None:
        A, E, _ = system
        b = A @ nonextensive_model(E, _true_theta())
        spectrum, _success, _msg, nfev, params = solve_nonextensive(
            A, b, E, search="global", ga_maxiter=5, seed=0,
        )
        assert np.all(np.isfinite(spectrum))
        assert np.isfinite(params["objective"])
        assert nfev > 0

    def test_invalid_search_raises(self, system) -> None:
        A, E, _ = system
        b = A @ nonextensive_model(E, _true_theta())
        with pytest.raises(ValueError, match="search must be"):
            solve_nonextensive(A, b, E, search="bogus")

    def test_params_within_bounds(self, system) -> None:
        A, E, _ = system
        b = A @ nonextensive_model(E, _true_theta())
        _sp, _s, _m, _n, params = solve_nonextensive(
            A, b, E, search="global", ga_maxiter=3, seed=1,
        )
        for name in NONEXTENSIVE_PARAM_NAMES:
            lo, hi = NONEXTENSIVE_PARAM_BOUNDS[name]
            assert lo - 1e-9 <= params[name] <= hi + 1e-9

    def test_unweighted_option(self, system) -> None:
        A, E, _ = system
        b = A @ nonextensive_model(E, _true_theta())
        _sp, _s, _m, _n, params = solve_nonextensive(
            A, b, E, search="local", relative=False,
            initial_params=dict(zip(NONEXTENSIVE_PARAM_NAMES, _true_theta())),
        )
        assert np.isfinite(params["objective"])

    def test_reproducible_with_seed(self, system) -> None:
        A, E, _ = system
        b = A @ nonextensive_model(E, _true_theta())
        sp1 = solve_nonextensive(A, b, E, search="global", ga_maxiter=3, seed=42)[0]
        sp2 = solve_nonextensive(A, b, E, search="global", ga_maxiter=3, seed=42)[0]
        np.testing.assert_array_equal(sp1, sp2)


# ---------------------------------------------------------------------------
# 3. Workflow wrapper
# ---------------------------------------------------------------------------


class TestUnfoldWrapper:
    def test_standard_keys(self, detector, simple_readings) -> None:
        names = detector.detector_names
        result = unfold_nonextensive(
            detector_names=names,
            n_energy_bins=detector.n_energy_bins,
            E_MeV=detector.E_MeV,
            sensitivities=detector.sensitivities,
            cc_icrp116=detector._get_interpolated_cc(),
            save_result_callback=detector._save_result,
            readings=simple_readings,
            search="local",
        )
        for key in (
            "spectrum", "doserates", "effective_readings",
            "residual_norm", "method", "model_params", "energy_nodes_eV",
        ):
            assert key in result, f"missing key: {key}"
        assert result["method"] == "Tsallis non-extensive"
        assert result["energy_nodes_eV"] == (E_THERMAL_MAX_EV, E_FAST_MIN_EV)
        assert np.all(np.isfinite(result["spectrum"]))


# ---------------------------------------------------------------------------
# 4. Detector integration
# ---------------------------------------------------------------------------


class TestDetectorIntegration:
    def test_basic(self, detector, simple_readings) -> None:
        result = detector.unfold_nonextensive(simple_readings, search="local")
        assert result["spectrum"].shape == detector.E_MeV.shape
        assert np.all(np.isfinite(result["spectrum"]))
        assert np.all(result["spectrum"] >= 0)
        assert "model_params" in result

    def test_energy_masking(self, detector, simple_readings) -> None:
        full = detector.unfold_nonextensive(simple_readings, search="local")
        masked = detector.unfold_nonextensive(
            simple_readings, search="local", max_neutron_energy=1.0
        )
        # masked run is expanded back to the full grid
        assert masked["spectrum"].shape == full["spectrum"].shape
        E_eV = detector.E_MeV * _EV_PER_MEV
        assert np.all(masked["spectrum"][E_eV > 1.0e6] == 0.0)

    def test_initial_params_accepted(self, detector, simple_readings) -> None:
        theta = _true_theta()
        init = dict(zip(NONEXTENSIVE_PARAM_NAMES, theta))
        result = detector.unfold_nonextensive(
            simple_readings, search="local", initial_params=init
        )
        assert np.all(np.isfinite(result["spectrum"]))

    def test_monte_carlo_errors(self, detector, simple_readings) -> None:
        result = detector.unfold_nonextensive(
            simple_readings, search="local",
            calculate_errors=True, n_montecarlo=5, random_state=0,
        )
        assert np.all(np.isfinite(result["spectrum"]))


# ---------------------------------------------------------------------------
# 5. IAEA end-to-end
# ---------------------------------------------------------------------------


class TestIAEA:
    def test_end_to_end(self, detector, iaea_readings) -> None:
        result = detector.unfold_nonextensive(
            iaea_readings, search="global", ga_maxiter=5, random_state=0,
        )
        spectrum = result["spectrum"]
        assert spectrum.shape == detector.E_MeV.shape
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum >= 0)
        assert np.isfinite(result["residual_norm"])
        mp = result["model_params"]
        assert np.isfinite(mp["objective"])
        assert len(mp["relative_deviations"]) == len(
            [n for n in detector.detector_names if n in iaea_readings]
        )
