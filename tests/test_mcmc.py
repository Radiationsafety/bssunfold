"""Tests for the Bayesian MCMC/NUTS unfolding method.

``pymc`` and ``arviz`` are optional backends (installed in the dev group via
``bssunfold[mcmc]``). Tests that require a real sampler are skipped when the
packages are missing — mock-based tests cover the wrapper, output assembly and
the ImportError fallback in all environments.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bssunfold import Detector

MOD = importlib.import_module("bssunfold.core.unfold_mcmc")


# ---------------------------------------------------------------------------
# Fake PyMC / ArviZ stand-ins used to exercise the logic without real sampling
# ---------------------------------------------------------------------------


class _FakeMath:
    @staticmethod
    def matmul(a, b):
        a = np.asarray(a, dtype=float)
        return np.zeros(a.shape[0])

    @staticmethod
    def dot(a, b):
        a = np.asarray(a, dtype=float)
        return np.zeros(a.shape[1])

    @staticmethod
    def exp(x):
        return np.exp(np.asarray(x, dtype=float))


class _FakeRV:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __array__(self, dtype=None):
        return np.zeros(1)

    def __mul__(self, other):
        return np.asarray(other)

    __rmul__ = __mul__


class _FakeModel:
    def __init__(self, coords=None):
        self.coords = coords
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False


class _FakeTrace:
    def __init__(self, values):
        self.posterior = {"spectrum": SimpleNamespace(values=values)}


class FakePM:
    """Drop-in replacement exposing the pm.* surface used by unfold_mcmc."""

    Model = _FakeModel
    HalfCauchy = staticmethod(lambda name, **kw: _FakeRV(name, **kw))
    HalfNormal = staticmethod(lambda name, **kw: _FakeRV(name, **kw))
    Normal = staticmethod(lambda name, **kw: _FakeRV(name, **kw))
    Deterministic = staticmethod(
        lambda name, value, dims=None: _FakeRV(name, value=value, dims=dims)
    )
    math = _FakeMath()

    def __init__(self, values=None, fail_inferencedata=False, raise_on_sample=False):
        self.values = values
        self.fail_inferencedata = fail_inferencedata
        self.raise_on_sample = raise_on_sample
        self.sample_calls = []

    def sample(self, **kwargs):
        self.sample_calls.append(dict(kwargs))
        if self.raise_on_sample:
            raise RuntimeError("sampler exploded")
        if self.fail_inferencedata and "return_inferencedata" in kwargs:
            raise TypeError("unexpected keyword argument 'return_inferencedata'")
        assert kwargs["draws"] >= 1 and kwargs["tune"] >= 0
        assert kwargs["chains"] >= 1
        n_energy = self.values.shape[2]
        fill = self.values[0, 0, 0] if self.values is not None else 0.5
        return _FakeTrace(
            np.full((kwargs["chains"], kwargs["draws"], n_energy), fill)
        )


class _FakeVar:
    def __init__(self, arr):
        self.values = np.asarray(arr)


class _FakeAZ:
    @staticmethod
    def hdi(samples, prob=0.95, **kwargs):
        arr = np.asarray(samples, dtype=float)
        return np.stack([arr.min(axis=0), arr.max(axis=0)], axis=1)

    @staticmethod
    def rhat(trace):
        vals = trace.posterior["spectrum"].values
        return {"spectrum": _FakeVar(np.zeros(vals.shape[2]))}

    @staticmethod
    def ess(trace):
        vals = trace.posterior["spectrum"].values
        return {"spectrum": _FakeVar(np.full(vals.shape[2], 500.0))}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _stats(n_energy, chains=2, draws=50, tune=10):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(chains * draws, n_energy))
    return {
        "samples": samples,
        "mean": samples.mean(axis=0),
        "median": np.median(samples, axis=0),
        "std": samples.std(axis=0),
        "hpd_lower": samples.min(axis=0),
        "hpd_upper": samples.max(axis=0),
        "trace": None,
        "rhat": np.ones(n_energy),
        "ess": np.full(n_energy, 100.0),
        "n_samples_total": chains * draws,
        "n_chains": chains,
        "tune_samples": tune,
        "target_accept": 0.8,
        "use_hierarchical": False,
        "lengthscale": 3.0,
        "prior_center": np.ones(n_energy),
    }


def _install_fakes(monkeypatch, fake_pm, fake_az=None):
    monkeypatch.setattr(MOD, "PYMC_AVAILABLE", True)
    monkeypatch.setattr(MOD, "pm", fake_pm)
    monkeypatch.setattr(MOD, "az", _FakeAZ() if fake_az is None else fake_az)


# ---------------------------------------------------------------------------
# Exports and ImportError fallback
# ---------------------------------------------------------------------------


def test_core_exports():
    from bssunfold.core import solve_bayesian_mcmc, unfold_mcmc

    assert solve_bayesian_mcmc is not None
    assert unfold_mcmc is not None
    assert "solve_bayesian_mcmc" in MOD.__all__
    assert "unfold_mcmc" in MOD.__all__


def test_importerror_without_pymc(detector, readings):
    with patch.object(MOD, "PYMC_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyMC and ArviZ"):
            MOD.solve_bayesian_mcmc(
                np.ones((4, 60)), np.ones(4), np.ones(60), np.ones(60)
            )
        with pytest.raises(ImportError, match="PyMC and ArviZ"):
            MOD.unfold_mcmc(
                detector_names=detector.detector_names,
                n_energy_bins=detector.n_energy_bins,
                E_MeV=detector.E_MeV,
                sensitivities=detector.sensitivities,
                cc_icrp116=detector._get_interpolated_cc(),
                save_result_callback=detector._save_result,
                readings=readings,
            )
        with pytest.raises(ImportError, match="PyMC and ArviZ"):
            detector.unfold_mcmc(readings)


def test_module_importable_without_pymc():
    assert MOD.PYMC_AVAILABLE is True
    assert MOD.pm is not None
    assert MOD.az is not None


# ---------------------------------------------------------------------------
# solve_bayesian_mcmc with fake PyMC / ArviZ
# ---------------------------------------------------------------------------


def test_solve_bayesian_mcmc_stats(monkeypatch):
    n_energy = 8
    values = np.random.default_rng(1).normal(size=(2, 50, n_energy)) + 0.5
    fake_pm = FakePM(values=values)
    _install_fakes(monkeypatch, fake_pm)

    spectrum, stats = MOD.solve_bayesian_mcmc(
        A_matrix=np.ones((4, n_energy)),
        b_readings=np.ones(4),
        E=np.linspace(1e-3, 10, n_energy),
        log_steps=np.ones(n_energy),
        n_samples=50,
        tune=10,
        chains=2,
        random_state=42,
    )

    assert spectrum.shape == (n_energy,)
    assert np.asarray(stats["samples"]).shape == (100, n_energy)
    assert stats["n_samples_total"] == 100
    assert stats["n_chains"] == 2
    assert stats["tune_samples"] == 10
    for key in (
        "mean",
        "median",
        "std",
        "hpd_lower",
        "hpd_upper",
        "rhat",
        "ess",
    ):
        assert np.asarray(stats[key]).shape == (n_energy,)
    assert np.all(stats["hpd_lower"] <= stats["hpd_upper"])
    assert fake_pm.sample_calls[0]["draws"] == 50
    assert fake_pm.sample_calls[0]["chains"] == 2
    assert fake_pm.sample_calls[0]["random_seed"] == 42


def test_solve_bayesian_mcmc_hierarchical(monkeypatch):
    n_energy = 6
    fake_pm = FakePM(values=np.full((1, 20, n_energy), 0.7))
    _install_fakes(monkeypatch, fake_pm)

    spectrum, stats = MOD.solve_bayesian_mcmc(
        A_matrix=np.ones((3, n_energy)),
        b_readings=np.ones(3),
        E=np.linspace(1e-3, 10, n_energy),
        log_steps=np.ones(n_energy),
        n_samples=20,
        tune=5,
        chains=1,
        use_hierarchical=True,
        random_state=1,
    )

    assert spectrum.shape == (n_energy,)
    assert stats["use_hierarchical"] is True
    assert stats["n_chains"] == 1
    assert np.all(np.isfinite(spectrum))


def test_solve_bayesian_mcmc_retries_without_return_inferencedata(monkeypatch):
    """Newer PyMC versions drop 'return_inferencedata'; the retry succeeds."""
    n_energy = 4
    fake_pm = FakePM(values=np.full((1, 10, n_energy), 0.5), fail_inferencedata=True)
    _install_fakes(monkeypatch, fake_pm)

    spectrum, stats = MOD.solve_bayesian_mcmc(
        A_matrix=np.ones((2, n_energy)),
        b_readings=np.ones(2),
        E=np.linspace(1e-3, 10, n_energy),
        log_steps=np.ones(n_energy),
        n_samples=10,
        tune=5,
        chains=1,
        random_state=0,
    )

    assert spectrum.shape == (n_energy,)
    assert len(fake_pm.sample_calls) == 2
    assert "return_inferencedata" not in fake_pm.sample_calls[1]
    assert stats["n_samples_total"] == 10


def test_solve_bayesian_mcmc_sampling_error(monkeypatch):
    n_energy = 3
    fake_pm = FakePM(values=np.ones((1, 10, n_energy)), raise_on_sample=True)
    _install_fakes(monkeypatch, fake_pm)

    with pytest.raises(RuntimeError, match="MCMC sampling failed"):
        MOD.solve_bayesian_mcmc(
            A_matrix=np.ones((2, n_energy)),
            b_readings=np.ones(2),
            E=np.linspace(1e-3, 10, n_energy),
            log_steps=np.ones(n_energy),
            n_samples=10,
            tune=5,
            chains=1,
        )


# ---------------------------------------------------------------------------
# unfold_mcmc wrapper (solve_bayesian_mcmc mocked out)
# ---------------------------------------------------------------------------


def _mock_solve(monkeypatch, n_energy):
    def fake_solve(**kwargs):
        stats = _stats(n_energy)
        stats["use_hierarchical"] = kwargs.get("use_hierarchical", False)
        return stats["mean"], stats

    monkeypatch.setattr(MOD, "solve_bayesian_mcmc", fake_solve)
    return fake_solve


def test_unfold_mcmc_result_dict(detector, readings, monkeypatch):
    n_energy = detector.n_energy_bins
    _mock_solve(monkeypatch, n_energy)

    result = MOD.unfold_mcmc(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        save_result=False,
    )

    for key in (
        "energy",
        "spectrum",
        "spectrum_absolute",
        "effective_readings",
        "residual",
        "residual_norm",
        "method",
        "doserates",
        "spectrum_uncertainty",
        "spectrum_lower",
        "spectrum_upper",
        "mcmc_stats",
        "sigma_prior",
        "lambda_prior",
        "lengthscale",
        "n_samples",
        "chains",
    ):
        assert key in result
    assert result["method"] == "MCMC"
    assert len(result["spectrum"]) == n_energy
    assert np.all(result["spectrum"] >= 0)
    assert np.all(result["spectrum_lower"] >= 0)
    assert result["spectrum_lower"].shape == (n_energy,)

    stats = result["mcmc_stats"]
    for key in (
        "samples",
        "median",
        "hpd_lower",
        "hpd_upper",
        "rhat",
        "ess",
        "n_samples_total",
        "n_chains",
        "tune_samples",
        "target_accept",
        "use_hierarchical",
        "trace",
    ):
        assert key in stats


def test_unfold_mcmc_detector_method(detector, readings, monkeypatch):
    n_energy = detector.n_energy_bins
    captured = {}
    fake_solve = _mock_solve(monkeypatch, n_energy)

    def spy_solve(**kwargs):
        captured.update(kwargs)
        return fake_solve(**kwargs)

    monkeypatch.setattr(MOD, "solve_bayesian_mcmc", spy_solve)

    result = detector.unfold_mcmc(
        readings,
        n_samples=25,
        tune=10,
        chains=1,
        target_accept=0.9,
        lengthscale=2.0,
        use_hierarchical=True,
        random_state=7,
        progressbar=False,
        save_result=False,
    )

    assert result["method"] == "MCMC"
    assert captured["n_samples"] == 25
    assert captured["tune"] == 10
    assert captured["chains"] == 1
    assert captured["target_accept"] == 0.9
    assert captured["lengthscale"] == 2.0
    assert captured["use_hierarchical"] is True
    assert captured["random_state"] == 7
    assert captured["progressbar"] is False
    assert result["mcmc_stats"]["use_hierarchical"] is True


def test_unfold_mcmc_deterministic(detector, readings, monkeypatch):
    n_energy = detector.n_energy_bins
    _mock_solve(monkeypatch, n_energy)

    r1 = detector.unfold_mcmc(readings, save_result=False)
    r2 = detector.unfold_mcmc(readings, save_result=False)
    assert np.allclose(r1["spectrum"], r2["spectrum"])
    assert np.allclose(r1["residual_norm"], r2["residual_norm"])


def test_unfold_mcmc_save_result(detector, readings, monkeypatch):
    _mock_solve(monkeypatch, detector.n_energy_bins)

    detector.unfold_mcmc(readings, save_result=True)
    assert len(detector.results_history) == 1
    latest = detector.results_history[max(detector.results_history.keys())]
    assert latest["method"] == "MCMC"
    assert "mcmc_stats" in latest


def test_unfold_mcmc_calculate_errors(detector, readings, monkeypatch):
    n_energy = detector.n_energy_bins

    def const_solve(**kwargs):
        stats = _stats(n_energy, chains=1, draws=10, tune=5)
        spectrum = np.abs(stats["mean"]) + 0.1
        return spectrum, stats

    monkeypatch.setattr(MOD, "solve_bayesian_mcmc", const_solve)

    result = detector.unfold_mcmc(
        readings,
        calculate_errors=True,
        n_montecarlo=3,
        noise_level=0.05,
        save_result=False,
    )
    assert "spectrum_uncert_mean" in result
    assert "spectrum_uncert_std" in result
    assert len(result["spectrum_uncert_mean"]) == n_energy


def test_unfold_mcmc_initial_spectrum_validation(detector, readings, monkeypatch):
    _mock_solve(monkeypatch, detector.n_energy_bins)

    with pytest.raises(ValueError, match="Initial spectrum length"):
        detector.unfold_mcmc(
            readings,
            initial_spectrum=np.ones(3),
            save_result=False,
        )


def test_unfold_mcmc_wrapper_runtime_error(detector, readings, monkeypatch):
    def broken_solve(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(MOD, "solve_bayesian_mcmc", broken_solve)

    with pytest.raises(RuntimeError, match="boom"):
        detector.unfold_mcmc(readings, save_result=False)


# ---------------------------------------------------------------------------
# Real sampling smoke tests (skipped without pymc/arviz)
# ---------------------------------------------------------------------------


def _tiny_detector():
    E = np.logspace(-2, 2, 6)
    sensitivities = {
        "s1": np.array([0.5, 0.7, 0.6, 0.4, 0.2, 0.1]),
        "s2": np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.5]),
    }
    detector = Detector(E_MeV=E, sensitivities=sensitivities)
    readings = {
        "s1": float(np.sum(sensitivities["s1"])),
        "s2": float(np.sum(sensitivities["s2"])),
    }
    return detector, readings


def test_solve_bayesian_mcmc_smoke_real():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")

    n_energy = 6
    A = np.array(
        [[0.5, 0.7, 0.6, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.6, 0.7, 0.5]]
    )
    b = np.array([2.5, 2.5])
    E = np.logspace(-2, 2, n_energy)
    log_steps = np.ones(n_energy)

    spectrum, stats = MOD.solve_bayesian_mcmc(
        A_matrix=A,
        b_readings=b,
        E=E,
        log_steps=log_steps,
        n_samples=20,
        tune=10,
        chains=1,
        target_accept=0.8,
        random_state=0,
        progressbar=False,
    )

    assert spectrum.shape == (n_energy,)
    assert np.all(np.isfinite(spectrum))
    assert np.asarray(stats["samples"]).shape == (20, n_energy)
    assert stats["n_samples_total"] == 20
    assert stats["n_chains"] == 1
    # A single chain has no cross-chain R-hat/ESS by definition (NaN allowed).
    assert np.asarray(stats["rhat"]).shape == (n_energy,)
    assert np.asarray(stats["ess"]).shape == (n_energy,)


def test_unfold_mcmc_real_smoke():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")

    detector, readings = _tiny_detector()

    result = detector.unfold_mcmc(
        readings,
        n_samples=20,
        tune=10,
        chains=1,
        random_state=42,
        progressbar=False,
        save_result=False,
    )

    assert result["method"] == "MCMC"
    assert len(result["spectrum"]) == detector.n_energy_bins
    assert np.all(np.isfinite(result["spectrum"]))
    assert np.all(np.isfinite(result["spectrum_uncertainty"]))
    assert "mcmc_stats" in result
    assert np.asarray(result["mcmc_stats"]["rhat"]).shape == (
        detector.n_energy_bins,
    )
    assert result["residual_norm"] >= 0


def test_unfold_mcmc_matches_module_function_real():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")

    detector, readings = _tiny_detector()

    res_det = detector.unfold_mcmc(
        readings, n_samples=20, tune=10, chains=1, random_state=0, progressbar=False
    )
    res_fn = MOD.unfold_mcmc(
        detector_names=detector.detector_names,
        n_energy_bins=detector.n_energy_bins,
        E_MeV=detector.E_MeV,
        sensitivities=detector.sensitivities,
        cc_icrp116=detector._get_interpolated_cc(),
        save_result_callback=detector._save_result,
        readings=readings,
        n_samples=20,
        tune=10,
        chains=1,
        random_state=0,
        progressbar=False,
    )
    assert res_fn["method"] == "MCMC"
    assert np.allclose(res_det["spectrum"], res_fn["spectrum"])
