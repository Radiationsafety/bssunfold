"""Tests for the Ogorodnikov (2024) ports.

Covers the two unfolding methods ported from:

    I. N. Ogorodnikov, "Inverse problems of spectroscopy and spectrometry
    in applied research", Traektoriya Issledovaniy no. 2 (10), pp. 42-83
    (2024):

* ``unfold_fission_ga`` -- the multisphere spectrometer algorithm
  (``BonnerFinder()``): Fission model + genetic algorithm + nonlinear
  least squares, with the article's validation criteria;
* ``unfold_tikhonov_sobolev_dp`` -- the Tikhonov regularization with
  the generalized discrepancy principle (``alfaFinder()``), including
  the standalone regularization-parameter selection routine.

The tests follow the article's quasi-real experiment scheme (sections
3.3 / 4.5): a model spectrum is folded with the packaged GSF response
functions of the :class:`~bssunfold.Detector`, a random noise signal is
added, and the recovered solution is compared against the truth.  The
IAEA Compendium spectra shipped with the test suite are used as
additional package-data cases.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from bssunfold.core._matrix_utils import compute_log_steps
from bssunfold.core.unfold_fission_ga import (
    ED_EPITHERMAL,
    FISSION_PARAM_BOUNDS,
    FISSION_PARAM_NAMES,
    T0_THERMAL,
    _validate_fit,
    fission_model,
    solve_fission_ga,
)
from bssunfold.core.unfold_tikhonov_sobolev_dp import (
    STATUS_OK,
    STATUS_RHO_NEGATIVE,
    STATUS_RHO_POSITIVE,
    alpha_finder_generalized_discrepancy,
    generalized_discrepancy,
    solve_tikhonov_sobolev_dp,
)

IAEA_CSV_PATH = (
    Path(__file__).parent
    / "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"
)


# ─── Helpers ──────────────────────────────────────────────────────


def _response_matrix(detector):
    """Response matrix (n_detectors x n_energy) of a detector."""
    return np.array(
        [detector.sensitivities[name] for name in detector.detector_names],
        dtype=float,
    )


def _ln_steps(detector):
    """Natural-log bin widths on the detector energy grid."""
    return compute_log_steps(detector.E_MeV, detector.n_energy_bins) * np.log(10)


def _true_fission_params():
    """Reference Fission-model parameters (all inside article bounds)."""
    return {
        "a1": 0.35,
        "a2": 0.25,
        "a3": 0.40,
        "b": 0.15,
        "beta": 0.35,
        "alpha": 0.6,
        "TF": 1.4,
    }


def _quasi_real_readings(detector, spectrum_bins, noise=0.0, seed=None):
    """Detector readings of a spectrum with uniform noise (eq. 4.22)."""
    A = _response_matrix(detector)
    exact = A @ spectrum_bins
    rng = np.random.default_rng(seed)
    if noise <= 0:
        return {name: float(v) for name, v in zip(detector.detector_names, exact)}
    noisy = exact + noise * np.abs(exact) * rng.uniform(-1.0, 1.0, size=exact.shape)
    return {name: float(v) for name, v in zip(detector.detector_names, noisy)}


@pytest.fixture
def fission_truth(detector):
    """True Fission-model per-bin fluence spectrum and its parameters."""
    ln_steps = _ln_steps(detector)
    params = _true_fission_params()
    shape = fission_model(detector.E_MeV, **params)
    phi_scale = 1e4
    spectrum_bins = shape * ln_steps * phi_scale
    return spectrum_bins, params, phi_scale, ln_steps


# ─── Fission model ────────────────────────────────────────────────


class TestFissionModel:
    def test_shape_and_positivity(self, detector):
        result = fission_model(detector.E_MeV, **_true_fission_params())
        assert result.shape == detector.E_MeV.shape
        assert np.all(result >= 0)

    def test_constants(self):
        assert T0_THERMAL == 2.53e-8
        assert ED_EPITHERMAL == 7.07e-8

    def test_param_names_and_bounds(self):
        assert FISSION_PARAM_NAMES == ("a1", "a2", "a3", "b", "beta", "alpha", "TF")
        assert FISSION_PARAM_BOUNDS["b"] == (-0.5, 0.5)
        assert FISSION_PARAM_BOUNDS["TF"] == (1.0, 2.0)
        assert FISSION_PARAM_BOUNDS["alpha"] == (0.0, 1.0)

    def test_thermal_dominates_low_energy(self, detector):
        E = detector.E_MeV
        thermal = fission_model(
            E, a1=1.0, a2=0.0, a3=0.0, b=0.0, beta=0.5, alpha=0.5, TF=1.5
        )
        fast = fission_model(
            E, a1=0.0, a2=0.0, a3=1.0, b=0.0, beta=0.5, alpha=0.5, TF=1.5
        )
        low = E < 1e-7
        high = E > 0.1
        assert thermal[low].sum() > 0
        assert fast[high].sum() > thermal[high].sum()

    def test_scales_linearly_with_weights(self, detector):
        E = detector.E_MeV
        one = fission_model(
            E, a1=0.2, a2=0.3, a3=0.1, b=0.1, beta=0.3, alpha=0.5, TF=1.4
        )
        two = fission_model(
            E, a1=0.4, a2=0.6, a3=0.2, b=0.1, beta=0.3, alpha=0.5, TF=1.4
        )
        assert_allclose(two, 2.0 * one, rtol=1e-12)


# ─── Generalized discrepancy alpha selection ──────────────────────


class TestGeneralizedDiscrepancy:
    @pytest.fixture
    def linear_problem(self, detector, fission_truth):
        """Noisy linear system with exactly known noise norm."""
        spectrum_bins, _params, _scale, _ln = fission_truth
        A = _response_matrix(detector)
        rng = np.random.default_rng(0)
        noise = 0.02 * np.linalg.norm(A @ spectrum_bins)
        noise_vec = noise * rng.standard_normal(A.shape[0]) / np.sqrt(A.shape[0])
        noise_vec *= noise / np.linalg.norm(noise_vec)
        b = A @ spectrum_bins + noise_vec
        return A, b, float(np.linalg.norm(noise_vec))

    def test_discrepancy_is_satisfied(self, linear_problem):
        A, b, delta = linear_problem
        info = alpha_finder_generalized_discrepancy(A, b, delta)
        assert info["status"] == STATUS_OK
        assert info["converged"]
        # rho(alpha*) = ||A z - b||^2 - delta^2 must vanish.
        assert abs(info["rho"]) / delta**2 < 1e-4
        assert info["alpha"] > 0

    def test_rho_monotone_in_alpha(self, linear_problem):
        A, b, delta = linear_problem
        n = A.shape[1]
        from bssunfold.core.unfold_tikhonov_sobolev_dp import (
            _penalty_matrix,
        )

        L = _penalty_matrix(n, "sobolev")
        delta_sq = delta**2
        rhos = [
            generalized_discrepancy(a, A.T @ A, L.T @ L, A.T @ b, A, b, delta_sq)
            for a in np.logspace(-6, 6, 13)
        ]
        diffs = np.diff(rhos)
        assert np.all(diffs > -1e-6 * delta_sq)

    def test_status_delta_too_small(self, linear_problem):
        A, b, _delta = linear_problem
        info = alpha_finder_generalized_discrepancy(A, b, 1e-15)
        assert info["status"] == STATUS_RHO_POSITIVE
        assert not info["converged"]

    def test_status_delta_too_large(self, linear_problem):
        A, b, _delta = linear_problem
        info = alpha_finder_generalized_discrepancy(A, b, 1e15)
        assert info["status"] == STATUS_RHO_NEGATIVE
        assert not info["converged"]

    def test_invalid_delta_raises(self, linear_problem):
        A, b, _delta = linear_problem
        with pytest.raises(ValueError):
            alpha_finder_generalized_discrepancy(A, b, 0.0)

    def test_invalid_alpha_range_raises(self, linear_problem):
        A, b, delta = linear_problem
        with pytest.raises(ValueError):
            alpha_finder_generalized_discrepancy(A, b, delta, alpha_range=(0.0, 1.0))

    def test_custom_identity_penalty(self, linear_problem):
        A, b, delta = linear_problem
        L = np.eye(A.shape[1])
        info = alpha_finder_generalized_discrepancy(A, b, delta, L=L)
        assert info["status"] == STATUS_OK
        assert abs(info["rho"]) / delta**2 < 1e-4

    def test_recovery_on_well_posed_problem(self):
        """On a well-posed linear system the method recovers the truth.

        The article's method targets classic spectroscopy systems with
        comparable numbers of data points and unknowns; on such systems
        the discrepancy-driven Tikhonov solution is accurate.
        """
        rng = np.random.default_rng(0)
        n = 60
        E = np.logspace(-9, 2, n)
        x_true = 1e6 * (E / (E + 1e-3)) * np.exp(-E / 2.0) + 0.5
        A = rng.uniform(0.2, 1.0, size=(n, n))
        noise = 0.01 * np.linalg.norm(A @ x_true)
        noise_vec = noise * rng.standard_normal(n) / np.sqrt(n)
        noise_vec *= noise / np.linalg.norm(noise_vec)
        b = A @ x_true + noise_vec

        spectrum, _it, converged = solve_tikhonov_sobolev_dp(
            A, b, delta=float(np.linalg.norm(noise_vec))
        )
        assert converged
        cosine = spectrum @ x_true / (np.linalg.norm(spectrum) * np.linalg.norm(x_true))
        assert cosine > 0.99
        rel_err = np.linalg.norm(spectrum - x_true) / np.linalg.norm(x_true)
        assert rel_err < 0.2


class TestSolveTikhonovSobolevDP:
    def test_returns_tuple_and_satisfies_dp(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        A = _response_matrix(detector)
        rng = np.random.default_rng(5)
        noise_vec = (
            0.02
            * np.linalg.norm(A @ spectrum_bins)
            * rng.standard_normal(A.shape[0])
            / np.sqrt(A.shape[0])
        )
        noise_vec *= np.linalg.norm(noise_vec) / np.linalg.norm(noise_vec)
        b = A @ spectrum_bins + noise_vec
        delta = float(np.linalg.norm(noise_vec))

        spectrum, n_iter, converged = solve_tikhonov_sobolev_dp(A, b, delta=delta)
        assert spectrum.shape == (detector.n_energy_bins,)
        assert n_iter > 0
        assert converged
        # ||A z - b|| matches delta (discrepancy principle).
        resid = np.linalg.norm(A @ spectrum - b)
        assert abs(resid - delta) / delta < 1e-4

    def test_explicit_delta_matches_noise_level(self, detector):
        A = _response_matrix(detector)[:4]
        b = np.ones(4)
        z1, _it1, _c1 = solve_tikhonov_sobolev_dp(A, b, noise_level=0.02)
        z2, _it2, _c2 = solve_tikhonov_sobolev_dp(A, b, delta=0.02 * np.linalg.norm(b))
        assert_allclose(z1, z2, rtol=1e-8)

    def test_penalty_variants_run(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins
        for penalty in ("sobolev", "curvature", "identity"):
            spectrum, _it, converged = solve_tikhonov_sobolev_dp(
                A, b, noise_level=0.02, penalty=penalty
            )
            assert spectrum.shape == (detector.n_energy_bins,)
            assert converged

    def test_invalid_penalty_raises(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        A = _response_matrix(detector)
        with pytest.raises(ValueError):
            solve_tikhonov_sobolev_dp(A, A @ spectrum_bins, penalty="bogus")


class TestUnfoldTikhonovSobolevDPDetector:
    def test_detector_workflow(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        readings = _quasi_real_readings(detector, spectrum_bins, noise=0.02, seed=42)
        A = _response_matrix(detector)
        b = np.array([readings[name] for name in detector.detector_names])
        # The article's delta is the known measurement error level.
        rng = np.random.default_rng(42)
        noise_vec = 0.02 * np.abs(b) * rng.uniform(-1.0, 1.0, size=b.shape)
        delta = float(np.linalg.norm(noise_vec))

        result = detector.unfold_tikhonov_sobolev_dp(
            readings, delta=delta, random_state=1
        )
        assert result["method"] == "TikhonovSobolevDP"
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        assert result["alpha"] > 0
        assert result["discrepancy_status"] == STATUS_OK
        assert result["dp_converged"]
        assert "doserates" in result

        # The raw (unclipped) DP solution matches the noise level
        # exactly: ||A z - b||^2 = delta^2 (article eq. 3.8).
        assert abs(result["residual_sq"] - delta**2) / delta**2 < 1e-3
        # The reported (non-negativity-clipped) spectrum folds to a
        # residual of the order of the discrepancy level.
        resid = np.linalg.norm(A @ result["spectrum"] - b)
        assert resid <= 0.3 * np.linalg.norm(b)

    def test_nonnegative_spectrum(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        readings = _quasi_real_readings(detector, spectrum_bins, seed=7)
        result = detector.unfold_tikhonov_sobolev_dp(readings)
        assert result["spectrum"].min() >= 0.0


# ─── Fission GA (BonnerFinder) ────────────────────────────────────


class TestSolveFissionGA:
    def test_noiseless_recovery(self, detector, fission_truth):
        spectrum_bins, params, phi_scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins

        spectrum, success, _msg, nfev, fitted = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, random_state=7
        )
        assert success
        assert nfev > 0
        assert spectrum.shape == spectrum_bins.shape

        rel_resid = np.linalg.norm(A @ spectrum - b) / np.linalg.norm(b)
        assert rel_resid < 1e-6

        rel_err = np.linalg.norm(spectrum - spectrum_bins) / np.linalg.norm(
            spectrum_bins
        )
        assert rel_err < 1e-3

        # Weight fractions are identifiable up to the model's shape
        # degeneracy: the ordering is preserved and each fraction is
        # recovered within a loose absolute tolerance.
        true_fr = np.array([params["a1"], params["a2"], params["a3"]])
        true_fr = true_fr / true_fr.sum()
        fitted_fr = fitted["weight_fractions"]
        fitted_vec = np.array([fitted_fr["a1"], fitted_fr["a2"], fitted_fr["a3"]])
        assert np.all(np.argsort(fitted_vec) == np.argsort(true_fr))
        assert_allclose(fitted_vec, true_fr, atol=0.15)
        # b parameter is well identified by the shape.
        assert abs(fitted["b"] - params["b"]) < 0.05

    def test_parameters_within_bounds(self, detector, fission_truth):
        spectrum_bins, _params, _scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins
        _spectrum, _success, _msg, _nfev, fitted = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, random_state=11
        )
        for name in FISSION_PARAM_NAMES:
            lo, hi = FISSION_PARAM_BOUNDS[name]
            assert lo <= fitted[name] <= hi

    def test_deterministic_with_seed(self, detector, fission_truth):
        spectrum_bins, _params, _scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins
        out1 = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, ga_maxiter=20, random_state=3
        )
        out2 = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, ga_maxiter=20, random_state=3
        )
        assert np.array_equal(out1[0], out2[0])
        assert out1[4]["cost"] == out2[4]["cost"]

    def test_noisy_readings_fit_within_noise(self, detector, fission_truth):
        spectrum_bins, _params, _scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = np.array(
            [
                readings
                for readings in _quasi_real_readings(
                    detector, spectrum_bins, noise=0.01, seed=1
                ).values()
            ]
        )
        spectrum, _success, _msg, _nfev, fitted = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, ga_maxiter=60, random_state=3
        )
        # Noise rms is ~0.58% of ||b|| (uniform +-1%).
        rel_resid = np.linalg.norm(A @ spectrum - b) / np.linalg.norm(b)
        assert rel_resid < 0.01

    def test_normalized_problem_without_scale(self, detector, fission_truth):
        """fit_scale=False on the model's natural (unscaled) output."""
        spectrum_bins, _params, _scale, ln_steps = fission_truth
        _shape_only = spectrum_bins / 1e4  # natural scale, phi_scale = 1
        A = _response_matrix(detector)
        b = A @ _shape_only
        spectrum, _success, _msg, _nfev, fitted = solve_fission_ga(
            A, b, detector.E_MeV, ln_steps, fit_scale=False, random_state=3
        )
        rel_resid = np.linalg.norm(A @ spectrum - b) / np.linalg.norm(b)
        assert rel_resid < 1e-3
        assert "phi_scale" not in fitted

    def test_lm_method_variant(self, detector, fission_truth):
        spectrum_bins, _params, _scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins
        spectrum, _success, _msg, _nfev, _fitted = solve_fission_ga(
            A,
            b,
            detector.E_MeV,
            ln_steps,
            lm_method="lm",
            ga_maxiter=40,
            random_state=2,
        )
        rel_resid = np.linalg.norm(A @ spectrum - b) / np.linalg.norm(b)
        assert rel_resid < 1e-3

    def test_initial_params_extra_start(self, detector, fission_truth):
        spectrum_bins, params, _scale, ln_steps = fission_truth
        A = _response_matrix(detector)
        b = A @ spectrum_bins
        start = dict(params)
        start["phi_scale"] = 1e4
        spectrum, _success, _msg, _nfev, fitted = solve_fission_ga(
            A,
            b,
            detector.E_MeV,
            ln_steps,
            initial_params=start,
            ga_maxiter=40,
            random_state=2,
        )
        rel_resid = np.linalg.norm(A @ spectrum - b) / np.linalg.norm(b)
        assert rel_resid < 1e-4
        assert fitted["phi_scale"] > 0


class TestUnfoldFissionGADetector:
    def test_detector_workflow_noiseless(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        readings = _quasi_real_readings(detector, spectrum_bins)
        result = detector.unfold_fission_ga(readings, random_state=3)
        assert result["method"] == "fission_ga"
        assert result["spectrum"].shape == (detector.n_energy_bins,)
        assert "doserates" in result

        rel_resid = result["residual_norm"] / np.linalg.norm(list(readings.values()))
        assert rel_resid < 1e-6

        validation = result["validation"]
        assert validation["passed"]
        assert validation["fom_percent"] < 1.0
        assert validation["max_relative_uncertainty"] < 0.05

        model_params = result["model_params"]
        assert set(FISSION_PARAM_NAMES).issubset(model_params)
        assert model_params["phi_scale"] > 0

    def test_detector_workflow_noisy(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        readings = _quasi_real_readings(detector, spectrum_bins, noise=0.01, seed=1)
        result = detector.unfold_fission_ga(readings, ga_maxiter=60, random_state=3)
        validation = result["validation"]
        # The article's validation criteria hold for the fit.
        assert validation["signs_mixed"]
        assert validation["max_relative_uncertainty"] <= 0.05
        rel_resid = result["residual_norm"] / np.linalg.norm(list(readings.values()))
        assert rel_resid < 0.01

    def test_max_neutron_energy_truncation(self, detector, fission_truth):
        spectrum_bins, _params, _scale, _ln = fission_truth
        readings = _quasi_real_readings(detector, spectrum_bins)
        result = detector.unfold_fission_ga(
            readings, random_state=3, max_neutron_energy=1.0
        )
        assert result["spectrum"].shape == (detector.n_energy_bins,)


# ─── Validation criteria helper ───────────────────────────────────


class TestValidateFit:
    def test_perfect_fit_passes(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        computed = b.copy()
        spectrum_bins = np.ones(10)
        report = _validate_fit(computed, b, spectrum_bins)
        assert report["passed"]
        assert report["fom_percent"] == 0.0
        assert report["residual_sign_changes"] == 0

    def test_alternating_small_residuals_pass(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        computed = b * (1.0 + np.array([0.01, -0.01, 0.01, -0.01]))
        spectrum_bins = np.ones(10)
        report = _validate_fit(computed, b, spectrum_bins)
        assert report["signs_mixed"]
        assert report["residual_sign_changes"] == 3
        assert report["passed"]

    def test_large_residual_fails(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        computed = b * 1.5
        spectrum_bins = np.ones(10)
        report = _validate_fit(computed, b, spectrum_bins)
        assert not report["passed"]

    def test_norm_check_normalized_mode(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        report = _validate_fit(b.copy(), b, np.full(4, 0.25), norm_range=(0.6, 1.2))
        assert report["norm_ok"]
        assert report["passed"]
        report_bad = _validate_fit(b.copy(), b, np.full(4, 5.0), norm_range=(0.6, 1.2))
        assert not report_bad["norm_ok"]
        assert not report_bad["passed"]


# ─── Package data (IAEA Compendium spectra) ───────────────────────


@pytest.fixture(scope="module")
def iaea_reference():
    return pd.read_csv(IAEA_CSV_PATH)


class TestIAEAPackageData:
    """Unfold spectra built from the packaged IAEA reference data."""

    def test_fission_ga_on_cf252(self, detector, iaea_reference):
        """Cf-252 is a Watt fission spectrum: the Fission model fits."""
        ref = iaea_reference
        spec = {"E_MeV": ref["E_MeV"].values, "Phi": ref["ISO_ref_Cf252"].values}
        readings = detector.get_effective_readings_for_spectra(spec)
        result = detector.unfold_fission_ga(readings, ga_maxiter=80, random_state=3)
        rel_resid = result["residual_norm"] / np.linalg.norm(list(readings.values()))
        # The Fission family approximates a Cf-252 field within ~5 %.
        assert rel_resid < 0.05
        assert result["validation"]["fom_percent"] < 15.0

    def test_tikhonov_dp_on_ambe(self, detector, iaea_reference):
        """AmBe source spectrum unfolded with Tikhonov + DP."""
        ref = iaea_reference
        spec = {"E_MeV": ref["E_MeV"].values, "Phi": ref["ISO_ref_AmBe"].values}
        readings = detector.get_effective_readings_for_spectra(spec)
        result = detector.unfold_tikhonov_sobolev_dp(readings, noise_level=0.02)
        assert result["discrepancy_status"] == STATUS_OK
        # With noiseless synthetic readings the discrepancy principle
        # drives the raw misfit to delta = 2 % of ||b||.
        b = np.array([readings[name] for name in detector.detector_names])
        delta = 0.02 * np.linalg.norm(b)
        assert abs(result["residual_sq"] - delta**2) / delta**2 < 1e-3

    def test_fission_ga_spectrum_shape_vs_cf252(self, detector, iaea_reference):
        """The unfolded Cf-252 spectrum correlates with the reference."""
        ref = iaea_reference
        spec = {"E_MeV": ref["E_MeV"].values, "Phi": ref["ISO_ref_Cf252"].values}
        interp = detector.discretize_spectra(spec)
        reference_bins = interp["Phi"].values
        readings = detector.get_effective_readings_for_spectra(spec)
        result = detector.unfold_fission_ga(readings, ga_maxiter=80, random_state=3)
        unfolded = result["spectrum"]
        corr = np.corrcoef(unfolded, reference_bins)[0, 1]
        assert corr > 0.7
