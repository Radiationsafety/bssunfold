"""Tests for dose conversion coefficient selection and interpolation."""

import numpy as np
import pytest

from bssunfold import (
    Detector,
    ICRP116_COEFF_EFFECTIVE_DOSE,
    ICRP74_COEFF_EFFECTIVE_DOSE,
    NRB99_2009_COEFF_EFFECTIVE_DOSE,
    ICRP74_COEFF_OPERATIONAL_QUANTITIES,
    get_coefficients,
    interpolate_coefficients,
)
from bssunfold.core.dose_calculation import (
    DOSE_COEFFICIENTS_REGISTRY,
    calculate_dose_rates,
)


# ---------------------------------------------------------------------------
# Constants structure tests
# ---------------------------------------------------------------------------

class TestCoefficientStructures:
    """Verify that all CC datasets have the expected structure."""

    def test_icrp116_has_emev(self):
        assert "E_MeV" in ICRP116_COEFF_EFFECTIVE_DOSE

    def test_icrp116_energy_length(self):
        assert len(ICRP116_COEFF_EFFECTIVE_DOSE["E_MeV"]) == 60

    def test_icrp116_geometries(self):
        expected = {"E_MeV", "AP", "PA", "LLAT", "RLAT", "ROT", "ISO"}
        assert set(ICRP116_COEFF_EFFECTIVE_DOSE.keys()) == expected

    def test_icrp74_effective_has_emev(self):
        assert "E_MeV" in ICRP74_COEFF_EFFECTIVE_DOSE

    def test_icrp74_effective_energy_length(self):
        assert len(ICRP74_COEFF_EFFECTIVE_DOSE["E_MeV"]) == 60

    def test_icrp74_effective_geometries(self):
        expected = {"E_MeV", "AP", "PA", "RLAT", "ROT", "ISO"}
        assert set(ICRP74_COEFF_EFFECTIVE_DOSE.keys()) == expected

    def test_nrb99_has_emev(self):
        assert "E_MeV" in NRB99_2009_COEFF_EFFECTIVE_DOSE

    def test_nrb99_energy_length(self):
        assert len(NRB99_2009_COEFF_EFFECTIVE_DOSE["E_MeV"]) == 24

    def test_nrb99_geometries(self):
        expected = {"E_MeV", "AP", "ISO"}
        assert set(NRB99_2009_COEFF_EFFECTIVE_DOSE.keys()) == expected

    def test_icrp74_operational_has_emev(self):
        assert "E_MeV" in ICRP74_COEFF_OPERATIONAL_QUANTITIES

    def test_icrp74_operational_energy_length(self):
        assert len(ICRP74_COEFF_OPERATIONAL_QUANTITIES["E_MeV"]) == 60

    def test_icrp74_operational_quantities(self):
        expected = {"E_MeV", "ADE", "PDE0", "PDE45", "PDE60", "PDE75"}
        assert set(ICRP74_COEFF_OPERATIONAL_QUANTITIES.keys()) == expected

    def test_all_energies_positive(self):
        for name, cc in DOSE_COEFFICIENTS_REGISTRY.items():
            e = np.array(cc["E_MeV"])
            assert np.all(e > 0), f"{name} has non-positive energies"

    def test_all_energies_monotonic(self):
        for name, cc in DOSE_COEFFICIENTS_REGISTRY.items():
            e = np.array(cc["E_MeV"])
            assert np.all(np.diff(e) > 0), f"{name} energies not monotonic"

    def test_all_values_non_negative(self):
        for name, cc in DOSE_COEFFICIENTS_REGISTRY.items():
            for key, vals in cc.items():
                if key != "E_MeV":
                    arr = np.array(vals)
                    assert np.all(arr >= 0), (
                        f"{name}.{key} has negative values"
                    )


# ---------------------------------------------------------------------------
# get_coefficients() tests
# ---------------------------------------------------------------------------

class TestGetCoefficients:
    """Test the get_coefficients() function."""

    def test_icrp116(self):
        cc = get_coefficients("ICRP116")
        assert cc is ICRP116_COEFF_EFFECTIVE_DOSE

    def test_icrp74_effective(self):
        cc = get_coefficients("ICRP74_effective")
        assert cc is ICRP74_COEFF_EFFECTIVE_DOSE

    def test_nrb99(self):
        cc = get_coefficients("NRB99_2009_effective")
        assert cc is NRB99_2009_COEFF_EFFECTIVE_DOSE

    def test_icrp74_operational(self):
        cc = get_coefficients("ICRP74_operational")
        assert cc is ICRP74_COEFF_OPERATIONAL_QUANTITIES

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dose coefficient"):
            get_coefficients("NONEXISTENT")

    def test_available_in_registry(self):
        assert set(DOSE_COEFFICIENTS_REGISTRY.keys()) == {
            "ICRP116",
            "ICRP74_effective",
            "NRB99_2009_effective",
            "ICRP74_operational",
        }


# ---------------------------------------------------------------------------
# interpolate_coefficients() tests
# ---------------------------------------------------------------------------

class TestInterpolateCoefficients:
    """Test the interpolate_coefficients() function."""

    def test_matching_grid_no_change(self):
        cc = get_coefficients("ICRP116")
        E = np.array(cc["E_MeV"])
        cc_int = interpolate_coefficients(cc, E)
        np.testing.assert_array_almost_equal(cc_int["E_MeV"], E)
        for key in cc:
            if key != "E_MeV":
                np.testing.assert_array_almost_equal(
                    cc_int[key], cc[key], decimal=10
                )

    def test_different_grid_interpolates(self):
        cc = get_coefficients("ICRP116")
        E_target = np.logspace(-9, 3, 100)
        cc_int = interpolate_coefficients(cc, E_target)
        assert len(cc_int["E_MeV"]) == 100
        assert "AP" in cc_int
        assert "PA" in cc_int

    def test_fill_value_outside_range(self):
        cc = get_coefficients("NRB99_2009_effective")
        # NRB99 range: 25 eV to 20 MeV
        E_target = np.array([1e-10, 1e-9, 1e1, 1e2])  # below and above
        cc_int = interpolate_coefficients(cc, E_target, fill_value=0.0)
        # Below range should be 0
        assert cc_int["AP"][0] == 0.0
        # Above range should be 0
        assert cc_int["AP"][-1] == 0.0

    def test_interpolation_reasonable_values(self):
        cc = get_coefficients("ICRP74_effective")
        E_target = np.logspace(-9, 3, 60)
        cc_int = interpolate_coefficients(cc, E_target)
        # At 1 MeV, AP should be around 299 (from CSV)
        idx = np.argmin(np.abs(E_target - 1.0))
        assert 250 < cc_int["AP"][idx] < 350

    def test_preserves_all_keys(self):
        cc = get_coefficients("ICRP74_operational")
        E_target = np.logspace(-9, 3, 60)
        cc_int = interpolate_coefficients(cc, E_target)
        assert set(cc_int.keys()) == set(cc.keys())

    def test_custom_fill_value(self):
        cc = get_coefficients("NRB99_2009_effective")
        E_target = np.array([1e-10])
        cc_int = interpolate_coefficients(cc, E_target, fill_value=-1.0)
        assert cc_int["AP"][0] == -1.0


# ---------------------------------------------------------------------------
# Detector cc_type tests
# ---------------------------------------------------------------------------

class TestDetectorCCType:
    """Test Detector cc_type parameter and set_dose_coefficients()."""

    def test_default_cc_type(self):
        det = Detector()
        assert det.cc_type == "ICRP116"

    def test_custom_cc_type(self):
        det = Detector(cc_type="ICRP74_effective")
        assert det.cc_type == "ICRP74_effective"
        assert det.cc_icrp116 is ICRP74_COEFF_EFFECTIVE_DOSE

    def test_set_dose_coefficients(self):
        det = Detector()
        det.set_dose_coefficients("NRB99_2009_effective")
        assert det.cc_type == "NRB99_2009_effective"
        assert det.cc_icrp116 is NRB99_2009_COEFF_EFFECTIVE_DOSE

    def test_set_dose_coefficients_invalid(self):
        det = Detector()
        with pytest.raises(ValueError, match="Unknown dose coefficient"):
            det.set_dose_coefficients("INVALID")

    def test_get_interpolated_cc_shape(self):
        det = Detector()
        cc_int = det._get_interpolated_cc()
        assert len(cc_int["E_MeV"]) == det.n_energy_bins
        for key in cc_int:
            if key != "E_MeV":
                assert len(cc_int[key]) == det.n_energy_bins

    def test_get_interpolated_cc_nrb99(self):
        det = Detector()
        det.set_dose_coefficients("NRB99_2009_effective")
        cc_int = det._get_interpolated_cc()
        # NRB99 only covers 25 eV - 20 MeV, so extreme energies should be 0
        assert cc_int["AP"][0] == 0.0  # 1e-9 MeV is below range
        assert cc_int["AP"][-1] == 0.0  # high energy above range


# ---------------------------------------------------------------------------
# calculate_dose_rates with different CC
# ---------------------------------------------------------------------------

class TestCalculateDoseRates:
    """Test calculate_dose_rates with different coefficient datasets."""

    def test_icrp116_default(self):
        spectrum = np.ones(60)
        result = calculate_dose_rates(spectrum)
        assert "AP" in result
        assert "PA" in result

    def test_icrp74_effective(self):
        cc = get_coefficients("ICRP74_effective")
        spectrum = np.ones(60)
        result = calculate_dose_rates(spectrum, cc)
        assert "AP" in result
        assert "PA" in result

    def test_nrb99(self):
        cc = get_coefficients("NRB99_2009_effective")
        spectrum = np.ones(24)
        result = calculate_dose_rates(spectrum, cc)
        assert "AP" in result
        assert "ISO" in result

    def test_operational(self):
        cc = get_coefficients("ICRP74_operational")
        spectrum = np.ones(60)
        result = calculate_dose_rates(spectrum, cc)
        assert "ADE" in result
        assert "PDE0" in result

    def test_different_cc_different_results(self):
        cc116 = get_coefficients("ICRP116")
        cc74 = get_coefficients("ICRP74_effective")
        spectrum = np.ones(60)
        r116 = calculate_dose_rates(spectrum, cc116)
        r74 = calculate_dose_rates(spectrum, cc74)
        # Values should differ between ICRP-116 and ICRP-74
        assert r116["AP"] != r74["AP"]


# ---------------------------------------------------------------------------
# Detector unfold integration test
# ---------------------------------------------------------------------------

class TestDetectorUnfoldIntegration:
    """Integration test: unfold with different CC types."""

    def test_unfold_with_icrp74(self):
        det = Detector(cc_type="ICRP74_effective")
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_cvxpy(readings, regularization=1e-3)
        assert "doserates" in result
        assert "AP" in result["doserates"]

    def test_unfold_with_nrb99(self):
        det = Detector(cc_type="NRB99_2009_effective")
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_cvxpy(readings, regularization=1e-3)
        assert "doserates" in result
        assert "AP" in result["doserates"]

    def test_change_cc_after_creation(self):
        det = Detector()
        det.set_dose_coefficients("ICRP74_effective")
        readings = {name: 100.0 for name in det.detector_names}
        result = det.unfold_cvxpy(readings, regularization=1e-3)
        assert "doserates" in result
