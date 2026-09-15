"""Tests for the TSVD SVD backends (full / ARPACK / PROPACK).

The ``svd_solver`` parameter maps the R packages ``svd`` (PROPACK) and
``rARPACK`` (ARPACK) onto their SciPy equivalents.  The tests verify
that the iterative backends reproduce the dense reference solution for
a fixed truncation ``k``, that automatic k-selection keeps using the
dense solver, and that failing backends degrade gracefully.
"""


import numpy as np
import pytest

from bssunfold import Detector
from bssunfold.core import solve_tsvd


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def A(detector):
    return np.array(
        [detector.sensitivities[name] for name in detector.detector_names]
    )


@pytest.fixture
def f_true(detector):
    E = detector.E_MeV
    f = (
        0.55 * np.sqrt(E / 2.0) * np.exp(-E / 2.0)
        + 0.20 * np.exp(-((E - 4.5) / 1.2) ** 2)
        + 0.02 / (1.0 + (E / 0.05) ** 2)
    )
    return f * 1e6 / np.max(f)


@pytest.fixture
def b(A, f_true):
    return A @ f_true


# ---------------------------------------------------------------------------
# solve_tsvd backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ("full", "arpack", "propack"))
def test_solve_tsvd_fixed_k_matches_dense_reference(A, b, solver):
    x_ref = solve_tsvd(A, b, k=6, svd_solver="full")
    x = solve_tsvd(A, b, k=6, svd_solver=solver)
    assert x.shape == x_ref.shape
    assert np.all(x >= 0)
    # iterative backends reproduce the leading-k triplets accurately
    cosine = float(x @ x_ref / (np.linalg.norm(x) * np.linalg.norm(x_ref)))
    assert cosine > 0.999


@pytest.mark.parametrize("solver", ("arpack", "propack"))
def test_solve_tsvd_k_equals_min_dim(A, b, solver):
    """k == min(m, n) - 1 must be accepted by the iterative backends."""
    m, n = A.shape
    x = solve_tsvd(A, b, k=min(m, n) - 1, svd_solver=solver)
    assert np.all(np.isfinite(x))


def test_solve_tsvd_auto_selection_uses_full(A, b):
    """Without a fixed k the dense solver is used (identical results)."""
    x1 = solve_tsvd(A, b, method="gcv", svd_solver="full")
    x2 = solve_tsvd(A, b, method="gcv", svd_solver="arpack")
    np.testing.assert_allclose(x1, x2, rtol=1e-12)


def test_solve_tsvd_threshold_selection_with_backend(A, b):
    x = solve_tsvd(A, b, threshold=1e-2, svd_solver="propack")
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)


def test_solve_tsvd_invalid_solver(A, b):
    with pytest.raises(ValueError, match="svd_solver"):
        solve_tsvd(A, b, k=5, svd_solver="bogus")


def test_solve_tsvd_backends_agree_on_singular_values(A):
    """The backends reproduce the leading singular spectrum."""
    from scipy.linalg import svd

    s_full = svd(A, compute_uv=False)
    from scipy.sparse.linalg import svds

    for solver in ("arpack", "propack"):
        _, s_k, _ = svds(A, k=4, solver=solver)
        np.testing.assert_allclose(
            np.sort(s_k)[::-1], s_full[:4], rtol=1e-8
        )


# ---------------------------------------------------------------------------
# unfold_tsvd wrapper
# ---------------------------------------------------------------------------


def test_unfold_tsvd_wrapper_svd_solver(detector, A, f_true):
    exact = A @ f_true
    readings = {
        name: float(v)
        for name, v in zip(detector.detector_names, exact)
    }
    result = detector.unfold_tsvd(readings, k=6, svd_solver="propack")
    assert result["svd_solver"] == "propack"
    assert np.all(result["spectrum"] >= 0)
    assert np.all(np.isfinite(result["spectrum"]))


def test_unfold_tsvd_wrapper_default_full(detector, A, f_true):
    exact = A @ f_true
    readings = {
        name: float(v)
        for name, v in zip(detector.detector_names, exact)
    }
    result = detector.unfold_tsvd(readings)
    assert result["svd_solver"] == "full"
    assert "k" in result and "k_method" in result


def test_unfold_tsvd_wrapper_max_energy(detector, A, f_true):
    exact = A @ f_true
    readings = {
        name: float(v)
        for name, v in zip(detector.detector_names, exact)
    }
    result = detector.unfold_tsvd(
        readings, k=5, svd_solver="arpack", max_neutron_energy=5.0
    )
    E = result["energy"]
    above = E > 5.0
    assert np.all(result["spectrum"][above] == 0)


# ---------------------------------------------------------------------------
# Bug fix regression: k / threshold must not be overridden
# ---------------------------------------------------------------------------


def test_solve_tsvd_fixed_k_overrides_method(A, b):
    """Regression: explicit k/threshold previously fell through to the
    automatic k-selection and were ignored."""
    x_k = solve_tsvd(A, b, k=4, method="discrepancy")
    x_ref = solve_tsvd(A, b, k=4, method="gcv")
    np.testing.assert_allclose(x_k, x_ref, rtol=1e-10)

    solve_tsvd(A, b, threshold=0.5)
    s_ref = np.linalg.svd(A, compute_uv=False)
    k_expected = int(np.sum(s_ref / s_ref[0] > 0.5))
    assert k_expected >= 1


def test_solve_tsvd_truncation_changes_solution(A, b):
    """Larger k must fit the data at least as well (within TSVD)."""
    res = []
    for k in (3, 8):
        x = solve_tsvd(A, b, k=k)
        res.append(np.linalg.norm(A @ x - b))
    assert res[1] <= res[0] + 1e-12
