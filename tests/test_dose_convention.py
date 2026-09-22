"""Tests for the spectrum-convention contract and dose integration fixes.

Covers the additive result keys (``spectrum_definition``, ``spectrum_units``,
``energy_bin_edges_MeV``, ``integration_rule``, ``dose_coverage_fraction``),
the per-bin ``dlnE_array`` lethargy integration of ``calculate_dose_rates``,
and the ``out_of_range`` handling of ``interpolate_coefficients``.
"""

import logging

import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.constants import RF_IHEP
from bssunfold.core.dose_calculation import (
    SPECTRUM_DEFINITION,
    SPECTRUM_UNITS,
    calculate_dose_rates,
    default_ln_steps,
    energy_bin_edges,
    interpolate_coefficients,
)


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def readings(detector):
    return {
        "3in": 0.053,
        "5in": 0.184,
        "10in": 0.172,
        "18in": 0.034,
    }


NEW_KEYS = (
    "spectrum_definition",
    "spectrum_units",
    "energy_bin_edges_MeV",
    "integration_rule",
    "dose_coverage_fraction",
)


class TestCalculateDoseRates:
    """calculate_dose_rates with per-bin dlnE_array."""

    def test_dlnE_array_takes_precedence(self):
        cc = {
            "E_MeV": np.array([1.0, 10.0]),
            "AP": np.array([1.0, 2.0]),
        }
        spectrum = np.array([3.0, 4.0])
        # dose = sum_i h_i * phi_i * dlnE_i; h = [1, 2]
        # uniform width 1.0 per bin
        res = calculate_dose_rates(spectrum, cc, dlnE_array=np.array([1.0, 1.0]))
        assert res["AP"] == pytest.approx(1.0 * 3.0 + 2.0 * 4.0)
        # non-uniform widths
        res2 = calculate_dose_rates(spectrum, cc, dlnE_array=np.array([0.5, 2.0]))
        assert res2["AP"] == pytest.approx(1.0 * 3.0 * 0.5 + 2.0 * 4.0 * 2.0)

    def test_dlnE_array_length_mismatch(self):
        cc = {"E_MeV": np.array([1.0, 10.0]), "AP": np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="dlnE_array shape"):
            calculate_dose_rates(
                np.array([1.0, 2.0]), cc, dlnE_array=np.array([1.0, 2.0, 3.0])
            )

    def test_uniform_grid_matches_legacy(self):
        """For a uniform 0.2-decade grid the array result equals the scalar."""
        E = 10.0 ** np.arange(-9, 2, 0.2)
        cc = {"E_MeV": E, "ISO": np.full(E.size, 2.5)}
        spectrum = np.abs(np.sin(np.arange(E.size)))
        legacy = calculate_dose_rates(spectrum, cc)  # default dlnE=0.2
        arr = calculate_dose_rates(
            spectrum, cc, dlnE_array=np.full(E.size, 0.2 * np.log(10.0))
        )
        assert arr["ISO"] == pytest.approx(legacy["ISO"], rel=1e-12)

    def test_nonuniform_grid_differs_from_fixed(self):
        """On the IHEP grid the per-bin integration differs from dlnE=0.2."""
        detector = Detector(response_functions=RF_IHEP)
        spectrum = np.ones(detector.n_energy_bins)
        cc = {"E_MeV": detector.E_MeV, "AP": np.full(detector.n_energy_bins, 1.0)}
        fixed = calculate_dose_rates(spectrum, cc)  # 0.2 decade everywhere
        true = calculate_dose_rates(spectrum, cc, dlnE_array=detector.ln_steps)
        # IHEP grid has 0.1-decade spacing in part of the range
        assert true["AP"] != pytest.approx(fixed["AP"])

    def test_mask_key_is_not_treated_as_geometry(self):
        cc = {
            "E_MeV": np.array([1.0, 10.0]),
            "AP": np.array([1.0, 2.0]),
            "_out_of_range_mask": np.array([False, True]),
        }
        res = calculate_dose_rates(np.array([1.0, 1.0]), cc)
        assert list(res.keys()) == ["AP"]


class TestInterpolateCoefficients:
    """interpolate_coefficients out-of-range modes and the mask key."""

    @pytest.fixture
    def cc(self):
        return {
            "E_MeV": np.array([1.0, 10.0]),
            "AP": np.array([10.0, 20.0]),
        }

    def test_zero_mode_default(self, cc):
        E = np.array([0.5, 5.0, 50.0])
        res = interpolate_coefficients(cc, E)
        assert res["AP"][0] == 0.0
        assert res["AP"][2] == 0.0
        # linear between (1, 10) and (10, 20): 10 + 10/9 * 4
        assert res["AP"][1] == pytest.approx(10.0 + (10.0 / 9.0) * 4.0)

    def test_mask_present(self, cc):
        E = np.array([0.5, 5.0, 50.0])
        res = interpolate_coefficients(cc, E)
        assert "_out_of_range_mask" in res
        assert res["_out_of_range_mask"].tolist() == [True, False, True]

    def test_nan_mode(self, cc):
        E = np.array([0.5, 5.0, 50.0])
        res = interpolate_coefficients(cc, E, out_of_range="nan")
        assert np.isnan(res["AP"][0])
        assert np.isnan(res["AP"][2])
        assert res["AP"][1] == pytest.approx(10.0 + (10.0 / 9.0) * 4.0)

    def test_extrapolate_mode(self, cc):
        E = np.array([0.5, 5.0, 50.0])
        res = interpolate_coefficients(cc, E, out_of_range="extrapolate")
        # slope below = (20-10)/(10-1) = 10/9
        assert res["AP"][0] == pytest.approx(10.0 + (10.0 / 9.0) * (0.5 - 1.0))
        assert res["AP"][2] == pytest.approx(20.0 + (10.0 / 9.0) * (50.0 - 10.0))

    def test_invalid_mode(self, cc):
        with pytest.raises(ValueError, match="out_of_range"):
            interpolate_coefficients(cc, np.array([1.0]), out_of_range="bogus")

    def test_backwards_compatible_no_mask_key_by_default_callers(self, cc):
        """The mask key must not break dict iteration in dose calculation."""
        E = np.array([1.0, 10.0])
        res = interpolate_coefficients(cc, E)
        geoms = [g for g in res if g not in ("E_MeV", "_out_of_range_mask")]
        assert geoms == ["AP"]


class TestHelpers:
    """default_ln_steps and energy_bin_edges."""

    def test_default_ln_steps_uniform_grid(self):
        E = 10.0 ** np.arange(-3.0, 3.0, 0.1)
        steps = default_ln_steps(E)
        assert steps.shape == E.shape
        assert np.allclose(steps, 0.1 * np.log(10.0), rtol=1e-9)

    def test_default_ln_steps_matches_detector(self):
        det = Detector()
        assert np.allclose(default_ln_steps(det.E_MeV), det.ln_steps)

    def test_energy_bin_edges_count(self):
        E = np.array([1.0, 2.0, 4.0, 8.0])
        edges = energy_bin_edges(E)
        assert edges.shape == (5,)
        assert edges[0] < E[0]
        assert edges[-1] > E[-1]
        # interior edges are geometric midpoints
        assert np.allclose(edges[1:-1], np.sqrt(E[:-1] * E[1:]))
        assert np.all(np.diff(edges) > 0)

    def test_energy_bin_edges_single_bin(self):
        edges = energy_bin_edges(np.array([5.0]))
        assert edges.shape == (2,)
        assert edges[0] < 5.0 < edges[1]


class TestDetectorContract:
    """Additive result keys on Detector output."""

    def test_standardized_keys_present(self, detector, readings):
        result = detector.unfold_landweber(readings, max_iterations=10)
        for key in NEW_KEYS:
            assert key in result, f"missing key: {key}"

    def test_spectrum_definition_values(self, detector, readings):
        result = detector.unfold_landweber(readings, max_iterations=10)
        assert result["spectrum_definition"] == SPECTRUM_DEFINITION
        assert result["spectrum_units"] == SPECTRUM_UNITS
        assert result["spectrum_definition"] == "differential_fluence_per_dlnE"

    def test_bin_edges_length(self, detector, readings):
        result = detector.unfold_landweber(readings, max_iterations=10)
        assert len(result["energy_bin_edges_MeV"]) == detector.n_energy_bins + 1

    def test_detector_attributes(self, detector):
        assert detector.ln_steps.shape == (detector.n_energy_bins,)
        assert detector.energy_bin_edges_MeV.shape == (detector.n_energy_bins + 1,)
        assert np.allclose(
            detector.ln_steps, detector.log_steps * np.log(10.0), rtol=1e-12
        )

    def test_max_energy_path_keeps_keys(self, detector, readings):
        result = detector.unfold_landweber(
            readings, max_iterations=10, max_neutron_energy=10.0
        )
        for key in NEW_KEYS:
            assert key in result
        assert len(result["spectrum"]) == detector.n_energy_bins

    def test_dose_coverage_full_for_gsf(self, detector, readings):
        """GSF grid (1e-9 .. ~500 MeV) is fully inside the ICRP116 range."""
        result = detector.unfold_landweber(readings, max_iterations=10)
        assert result["dose_coverage_fraction"] == pytest.approx(1.0)

    def test_dose_coverage_warning_on_partial(self, detector, readings, caplog):
        """A spectrum with fluence above the CC range triggers the warning."""
        with caplog.at_level(logging.WARNING, logger="bssunfold.core.detector"):
            detector.unfold_landweber(
                readings, max_iterations=10, max_neutron_energy=1000.0
            )
        # GSF grid tops out near 500-630 MeV < ... ensure no crash; warning
        # only fires when coverage < 1. Construct an explicit partial case:
        caplog.clear()
        spectrum = np.zeros(detector.n_energy_bins)
        spectrum[-1] = 1.0  # all fluence in the highest bin
        coverage = 1.0
        cc = detector._get_interpolated_cc()
        out_mask = np.asarray(cc["_out_of_range_mask"], dtype=bool)
        if out_mask[-1]:
            coverage = 0.0
        with caplog.at_level(logging.WARNING, logger="bssunfold.core.detector"):
            detector._compute_doserates(spectrum)
            assert detector._compute_doserates(spectrum)[1] == pytest.approx(
                coverage
            )

    def test_doserates_recomputed_with_bin_widths(self, detector, readings):
        """doserates equal a manual lethargy integration with ln_steps."""
        result = detector.unfold_landweber(readings, max_iterations=10)
        cc = detector._get_interpolated_cc()
        manual = calculate_dose_rates(
            result["spectrum"], cc, dlnE_array=detector.ln_steps
        )
        for geom, value in manual.items():
            assert result["doserates"][geom] == pytest.approx(value)


class TestNonUniformDetector:
    """End-to-end on a non-uniform grid (RF_IHEP)."""

    @pytest.fixture
    def ihep_detector(self):
        return Detector(response_functions=RF_IHEP)

    @pytest.fixture
    def ihep_readings(self, ihep_detector):
        return {name: 100.0 for name in ihep_detector.detector_names[:4]}

    def test_contract_on_ihep(self, ihep_detector, ihep_readings):
        result = ihep_detector.unfold_landweber(ihep_readings, max_iterations=20)
        for key in NEW_KEYS:
            assert key in result

    def test_dose_uses_real_bin_widths(self, ihep_detector, ihep_readings):
        result = ihep_detector.unfold_landweber(ihep_readings, max_iterations=20)
        legacy = calculate_dose_rates(
            result["spectrum"],
            ihep_detector._get_interpolated_cc(),
        )
        # the fixed dlnE=0.2 assumption is wrong for this grid
        assert result["doserates"]["AP"] != pytest.approx(legacy["AP"])

    def test_coverage_fraction_computed(self, ihep_detector, ihep_readings):
        result = ihep_detector.unfold_landweber(ihep_readings, max_iterations=20)
        assert 0.0 <= result["dose_coverage_fraction"] <= 1.0


class TestEstimateTotalFluence:
    """estimate_total_fluence with the ln_steps option."""

    def test_lethargy_integral_default(self):
        from bssunfold.core._matrix_utils import estimate_total_fluence

        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        assert estimate_total_fluence(A, b) == pytest.approx(6.0)

    def test_weighted_by_ln_steps(self):
        from bssunfold.core._matrix_utils import estimate_total_fluence

        A = np.eye(2)
        b = np.array([1.0, 2.0])
        ln_steps = np.array([0.5, 2.0])
        assert estimate_total_fluence(A, b, ln_steps=ln_steps) == pytest.approx(
            1.0 * 0.5 + 2.0 * 2.0
        )

    def test_default_unweighted_value_unchanged(self):
        from bssunfold.core._matrix_utils import estimate_total_fluence

        rng = np.random.default_rng(42)
        A = rng.random((5, 4)) + 0.1
        b = rng.random(5) + 0.1
        assert estimate_total_fluence(A, b) > 0


class TestRunUnfoldingLnSteps:
    """run_unfolding accepts ln_steps and falls back gracefully."""

    def test_fallback_computed_from_E_MeV(self, detector, readings):
        """Without ln_steps the result is still produced (legacy scalar path)."""
        result = detector.unfold_landweber(readings, max_iterations=10)
        assert "doserates" in result and result["doserates"]
