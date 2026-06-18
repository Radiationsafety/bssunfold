"""Tests for built-in response function datasets (RF_GSF, RF_PTB, RF_LANL, RF_JINR, RF_FERMILAB, RF_EURADOS, RF_IHEP)."""

import pytest
from bssunfold.constants import (
    RF_GSF,
    RF_PTB,
    RF_LANL,
    RF_JINR,
    RF_FERMILAB,
    RF_EURADOS,
    RF_IHEP,
)
from bssunfold import Detector


ALL_RFS = {
    "RF_GSF": RF_GSF,
    "RF_PTB": RF_PTB,
    "RF_LANL": RF_LANL,
    "RF_JINR": RF_JINR,
    "RF_FERMILAB": RF_FERMILAB,
    "RF_EURADOS": RF_EURADOS,
    "RF_IHEP": RF_IHEP,
}


class TestRFStructuralIntegrity:
    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_has_emev_key(self, name):
        rf = ALL_RFS[name]
        assert "E_MeV" in rf, f"{name} missing 'E_MeV' key"

    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_all_keys_same_length(self, name):
        rf = ALL_RFS[name]
        lengths = {k: len(v) for k, v in rf.items()}
        assert len(set(lengths.values())) == 1, f"{name} has inconsistent key lengths: {lengths}"

    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_positive_energy_values(self, name):
        rf = ALL_RFS[name]
        for v in rf["E_MeV"]:
            assert v > 0, f"{name} has non-positive energy value: {v}"

    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_monotonic_energy(self, name):
        rf = ALL_RFS[name]
        e = rf["E_MeV"]
        for i in range(1, len(e)):
            assert e[i] > e[i - 1], f"{name} energy not strictly monotonic at index {i}: {e[i-1]} >= {e[i]}"

    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_non_negative_detector_response(self, name):
        rf = ALL_RFS[name]
        for key, vals in rf.items():
            if key == "E_MeV":
                continue
            for i, v in enumerate(vals):
                assert v >= 0, f"{name}[{key}][{i}] is negative: {v}"


class TestRFDetectorCounts:
    def test_gsf_has_ten_detectors(self):
        assert len(RF_GSF) == 11  # E_MeV + 10 detectors

    def test_ptb_has_fifteen_detectors(self):
        assert len(RF_PTB) == 16  # E_MeV + 15 detectors

    def test_lanl_has_eleven_detectors(self):
        assert len(RF_LANL) == 12  # E_MeV + 11 detectors

    def test_jinr_has_nine_detectors(self):
        assert len(RF_JINR) == 10  # E_MeV + 9 detectors

    def test_fermilab_has_eight_detectors(self):
        assert len(RF_FERMILAB) == 9  # E_MeV + 8 detectors

    def test_eurados_has_thirteen_detectors(self):
        assert len(RF_EURADOS) == 14  # E_MeV + 13 detectors

    def test_ihep_has_twelve_detectors(self):
        assert len(RF_IHEP) == 13  # E_MeV + 12 detectors


class TestRFEnergyRanges:
    def test_standard_range_gsf(self):
        assert RF_GSF["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_GSF["E_MeV"][-1] == pytest.approx(630.957, rel=1e-3)

    def test_standard_range_ptb(self):
        assert RF_PTB["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_PTB["E_MeV"][-1] == pytest.approx(630.957, rel=1e-3)

    def test_standard_range_lanl(self):
        assert RF_LANL["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_LANL["E_MeV"][-1] == pytest.approx(630.957, rel=1e-3)

    def test_jinr_range(self):
        assert RF_JINR["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_JINR["E_MeV"][-1] == pytest.approx(630.957, rel=1e-3)

    def test_fermilab_range(self):
        assert RF_FERMILAB["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_FERMILAB["E_MeV"][-1] == pytest.approx(630.957, rel=1e-3)

    def test_eurados_limited_range(self):
        assert RF_EURADOS["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_EURADOS["E_MeV"][-1] == pytest.approx(20.0)

    def test_ihep_extended_range(self):
        assert RF_IHEP["E_MeV"][0] == pytest.approx(1e-9)
        assert RF_IHEP["E_MeV"][-1] == pytest.approx(2000.0)


class TestRFSpecialDetectors:
    def test_jinr_has_cad0in(self):
        assert "Cd0in" in RF_JINR

    def test_jinr_has_10inpPb(self):
        assert "10inPb" in RF_JINR

    def test_eurados_has_cd2in(self):
        assert "Cd2in" in RF_EURADOS

    def test_eurados_has_halfinch_detectors(self):
        for d in ["3.5in", "4.5in"]:
            assert d in RF_EURADOS

    def test_ihep_has_15in(self):
        assert "15in" in RF_IHEP


class TestRFDetectorCreation:
    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_detector_from_rf(self, name):
        rf = ALL_RFS[name]
        det = Detector(rf)
        assert det is not None

    @pytest.mark.parametrize("name", ALL_RFS.keys())
    def test_detector_sphere_count(self, name):
        rf = ALL_RFS[name]
        det = Detector(rf)
        expected_spheres = len(rf) - 1
        assert det.n_detectors == expected_spheres


class TestRFImport:
    def test_import_from_package(self):
        import bssunfold
        assert hasattr(bssunfold, "RF_GSF")
        assert hasattr(bssunfold, "RF_PTB")
        assert hasattr(bssunfold, "RF_LANL")
        assert hasattr(bssunfold, "RF_JINR")
        assert hasattr(bssunfold, "RF_FERMILAB")
        assert hasattr(bssunfold, "RF_EURADOS")
        assert hasattr(bssunfold, "RF_IHEP")
