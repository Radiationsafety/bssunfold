"""Tests for the Hybrid8-style start spectrum (``hybrid8_start``).

Covers the ``hybrid8_start`` generator (shape, normalization, validation),
its wiring into ``unfold_sandii`` / ``Detector.unfold_sandii`` via
``starting_spectrum="hybrid8"``, and an IAEA end-to-end smoke run.
"""

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import HYBRID8_ANCHORS, hybrid8_start


@pytest.fixture
def grid():
    return np.logspace(-9, 3, 121)


class TestHybrid8Start:
    def test_positive_and_finite(self, grid):
        spec = hybrid8_start(grid)
        assert spec.shape == grid.shape
        assert np.all(np.isfinite(spec))
        assert np.all(spec > 0)

    def test_normalization(self, grid):
        spec = hybrid8_start(grid, total_fluence=3.7)
        assert np.isclose(np.sum(spec * grid), 3.7)

    def test_scale_linearity(self, grid):
        np.testing.assert_allclose(
            hybrid8_start(grid, total_fluence=2.0),
            2.0 * hybrid8_start(grid, total_fluence=1.0),
        )

    def test_four_regions(self, grid):
        ephi = hybrid8_start(grid) * grid
        # evaporation peak near 2 MeV beats the epithermal valley near 30 MeV
        at_2 = np.interp(np.log(2.0), np.log(grid), np.log(ephi))
        at_30 = np.interp(np.log(30.0), np.log(grid), np.log(ephi))
        at_100 = np.interp(np.log(100.0), np.log(grid), np.log(ephi))
        at_1e4 = np.interp(np.log(1e-4), np.log(grid), np.log(ephi))
        assert at_2 > at_30  # epithermal dip between the peaks
        assert at_100 > at_30  # cascade peak
        assert at_100 > at_1e4  # thermal region below the evaporation peak
        # local maximum of E*Phi in the evaporation region
        mask = (grid > 0.2) & (grid < 20.0)
        assert grid[mask][np.argmax(ephi[mask])] == pytest.approx(2.0, rel=0.3)
        # local maximum in the cascade region
        mask = (grid > 30.0) & (grid < 1000.0)
        assert grid[mask][np.argmax(ephi[mask])] == pytest.approx(100.0, rel=0.3)

    def test_monotone_decrease_above_cascade(self, grid):
        ephi = hybrid8_start(grid) * grid
        mask = grid > 100.0
        assert np.all(np.diff(ephi[mask]) < 0)

    def test_out_of_range_clamped(self):
        # At the anchor energies the (unnormalized) shape equals the anchors;
        # outside the anchor range the log-log interpolation clamps to the
        # end values.
        e_lo, v_lo = HYBRID8_ANCHORS[0]
        e_hi, v_hi = HYBRID8_ANCHORS[-1]
        E = np.array([e_lo / 1e3, e_lo, e_hi, e_hi * 1e3])
        shape = np.array([v_lo, v_lo, v_hi, v_hi])
        spec = hybrid8_start(E)
        # clamped ends equal the anchor values up to the normalization scale
        np.testing.assert_allclose(spec * E / shape, 1.0 / np.sum(shape))

    def test_validation(self, grid):
        with pytest.raises(ValueError, match="total_fluence"):
            hybrid8_start(grid, total_fluence=0.0)
        with pytest.raises(ValueError, match="total_fluence"):
            hybrid8_start(grid, total_fluence=-1.0)
        with pytest.raises(ValueError, match="E_MeV"):
            hybrid8_start(np.array([]))
        with pytest.raises(ValueError, match="E_MeV"):
            hybrid8_start(np.array([1.0, -2.0]))
        with pytest.raises(ValueError, match="E_MeV"):
            hybrid8_start(np.array([[1.0, 2.0]]))

    def test_anchors_sorted_and_positive(self):
        energies = np.array([e for e, _ in HYBRID8_ANCHORS])
        values = np.array([v for _, v in HYBRID8_ANCHORS])
        assert np.all(np.diff(energies) > 0)
        assert np.all(energies > 0)
        assert np.all(values > 0)


class TestSandiiWiring:
    def test_hybrid8_default_matches_generator(self):
        from bssunfold.core import unfold_sandii

        E = np.logspace(-8, 2, 40)
        n = len(E)
        A = np.ones((3, n)) * np.exp(-((E[None, :] - 1.0) ** 2))
        sensitivities = {f"d{i}": A[i] for i in range(3)}
        cc = {f"d{i}": np.ones(n) for i in range(3)}
        readings = {f"d{i}": 1.0 for i in range(3)}
        result = unfold_sandii(
            detector_names=list(sensitivities),
            n_energy_bins=n,
            E_MeV=E,
            sensitivities=sensitivities,
            cc_icrp116=cc,
            save_result_callback=lambda r: "test",
            readings=readings,
            starting_spectrum="hybrid8",
            max_iterations=5,
        )
        assert result["starting_spectrum"] == "hybrid8"
        assert result["spectrum"].shape == (n,)
        assert np.all(np.isfinite(result["spectrum"]))

    def test_unknown_name_raises(self):
        from bssunfold.core import unfold_sandii

        E = np.logspace(-8, 2, 10)
        with pytest.raises(ValueError, match="starting_spectrum"):
            unfold_sandii(
                detector_names=["d0"],
                n_energy_bins=len(E),
                E_MeV=E,
                sensitivities={"d0": np.ones(len(E))},
                cc_icrp116={"d0": np.ones(len(E))},
                save_result_callback=lambda r: "test",
                readings={"d0": 1.0},
                starting_spectrum="bogus",
            )


@pytest.fixture
def detector():
    from bssunfold.constants import RF_GSF

    return Detector(RF_GSF)


@pytest.fixture
def readings(detector):
    return {
        name: float(i + 1) * 0.05
        for i, name in enumerate(detector.detector_names)
    }


class TestDetectorSandiiHybrid8:
    def test_end_to_end(self, detector, readings):
        result = detector.unfold_sandii(
            readings, starting_spectrum="hybrid8", max_iterations=50
        )
        spec = result["spectrum"]
        assert spec.shape == (detector.n_energy_bins,)
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0)
        assert result["starting_spectrum"] == "hybrid8"

    def test_differs_from_flat_start(self, detector, readings):
        flat = detector.unfold_sandii(readings, max_iterations=3)
        hybrid = detector.unfold_sandii(
            readings, starting_spectrum="hybrid8", max_iterations=3
        )
        assert not np.allclose(flat["spectrum"], hybrid["spectrum"])

    def test_initial_spectrum_takes_precedence(self, detector, readings):
        x0 = np.full(detector.n_energy_bins, 0.5)
        a = detector.unfold_sandii(
            readings, initial_spectrum=x0, starting_spectrum="hybrid8",
            max_iterations=3,
        )
        b = detector.unfold_sandii(readings, initial_spectrum=x0, max_iterations=3)
        np.testing.assert_allclose(a["spectrum"], b["spectrum"])

    def test_flat_default_unchanged(self, detector, readings):
        r1 = detector.unfold_sandii(readings, max_iterations=5)
        r2 = detector.unfold_sandii(readings, starting_spectrum=None, max_iterations=5)
        np.testing.assert_allclose(r1["spectrum"], r2["spectrum"])

    def test_monte_carlo(self, detector, readings):
        result = detector.unfold_sandii(
            readings,
            starting_spectrum="hybrid8",
            calculate_errors=True,
            n_montecarlo=10,
            random_state=1,
        )
        assert "spectrum_uncert_mean" in result
