"""Tests for the SSR-MLP port (R sisireg 1.2.1 ssrMLP).

Covers the error/factor building blocks of the partial sum criterion,
the network forward pass, the SGD training loop (including exact
agreement with the original R implementation on recorded fixtures),
prediction, factor importance and the input validation paths.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bssunfold.core.sisireg_mlp import (
    SSRMLPModel,
    calc_out,
    check_ps,
    d1sigmoid,
    err_lse,
    err_ps,
    err_ps_l1,
    err_ps_lse,
    fac_lse,
    fac_ps,
    fac_ps_l1,
    fac_ps_lse,
    fii_model,
    fii_prediction,
    sigmoid,
    ssrmlp_predict,
    ssrmlp_train,
)

_DATA = Path(__file__).parent / "data" / "sisireg_mlp"


class FixedPerm:
    """Replays a recorded sample permutation through the rng interface."""

    def __init__(self, perm):
        self.perm = np.asarray(perm, dtype=np.int64)

    def permutation(self, n):
        assert len(self.perm) == n
        return self.perm


def _load_mat(name):
    # R write.csv(row.names=FALSE) still writes a V1..Vk header line
    return pd.read_csv(_DATA / name).to_numpy()


@pytest.fixture(scope="module")
def mlp_fixtures():
    data = pd.read_csv(_DATA / "mlp_data.csv", header=0)
    X = data.iloc[:, :2].to_numpy()
    Y = data["Y"].to_numpy()
    bnd = pd.read_csv(_DATA / "mlp_bounds.csv")
    yb = pd.read_csv(_DATA / "mlp_ybounds.csv")
    init = SSRMLPModel(
        W0=_load_mat("mlp_W0_init.csv"),
        W1=_load_mat("mlp_W1_init.csv"),
        W2=_load_mat("mlp_W2_init.csv"),
        minX=bnd["minX"].to_numpy(),
        maxX=bnd["maxX"].to_numpy(),
        minY=float(yb["minY"].iloc[0]),
        maxY=float(yb["maxY"].iloc[0]),
    )
    mix = pd.read_csv(_DATA / "mlp_mix_002.csv")["mix"].to_numpy() - 1
    return X, Y, init, mix


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_sigmoid():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(np.array([-50.0, 0.0, 50.0])) == pytest.approx(
        [0.0, 0.5, 1.0], abs=1e-10
    )


def test_d1sigmoid():
    assert d1sigmoid(0.0) == pytest.approx(0.25)
    x = np.array([-2.0, 0.0, 3.0])
    s = sigmoid(x)
    assert d1sigmoid(x) == pytest.approx(s * (1 - s))


# ---------------------------------------------------------------------------
# Criterion building blocks
# ---------------------------------------------------------------------------


def _toy_nb():
    # three overlapping neighbourhoods (centre first)
    return [
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        np.array([2, 3, 0]),
    ]


def test_check_ps():
    Y = np.array([1.0, 1.0, 1.0, -1.0])
    Yp = np.zeros(4)
    nb = _toy_nb()
    # neighbourhood [0,1,2] sums to 3, [1,2,3] sums to 1, [2,3,0] sums to 1
    assert check_ps(nb, Y, Yp, 0, fn=4)
    assert not check_ps(nb, Y, Yp, 0, fn=2)  # 3 > 2
    assert check_ps(nb, Y, Yp, 1, fn=4)


def test_err_ps():
    Y = np.array([1.0, 1.0, 1.0, -1.0])
    Yp = np.zeros(4)
    nb = _toy_nb()
    # sums |3|, |1|, |1| -> max(0, 3-4)=0 with fn=4; with fn=2: 1 + 0 + 0
    assert err_ps(nb, None, None, Y, Yp, fn=4) == 0.0
    assert err_ps(nb, None, None, Y, Yp, fn=2) == 1.0
    assert err_ps(nb, None, None, Y, Yp, fn=1) == (2 + 0 + 0)


def test_fac_ps():
    Y = np.array([1.0, 1.0, 1.0, -1.0])
    nb = _toy_nb()
    Yp_ok = np.zeros(4)
    assert fac_ps(nb, None, None, Y, Yp_ok, 0, fn=4) == 0.0
    # violating configuration: sign(Y-Yp) * |d|^0.01
    Yp_bad = np.array([0.0, 0.5, 0.5, 0.0])
    expected = (1.0 - 0.0) * abs(1.0) ** 0.01
    assert fac_ps(nb, None, None, Y, Yp_bad, 0, fn=2) == pytest.approx(expected)


def test_err_fac_lse():
    Y = np.array([1.0, 2.0, 3.0])
    Yp = np.array([1.5, 2.0, 2.0])
    assert err_lse(None, None, None, Y, Yp) == pytest.approx(
        (0.25 + 0.0 + 1.0) / 3
    )
    assert fac_lse(None, None, None, Y, Yp, 2) == pytest.approx(1.0)


def test_err_fac_ps_lse():
    Y = np.array([1.0, 1.0, 1.0, -1.0])
    Yp = np.zeros(4)
    nb = _toy_nb()
    # pure ps part is 0 with fn=4; lse part = alpha * sum((Yp-Y)^2)
    alpha = 1e-4
    expected = alpha * 4.0
    assert err_ps_lse(nb, None, None, Y, Yp, fn=4, alpha=alpha) == pytest.approx(
        expected
    )
    # factor = alpha*(Y-Yp) when adequate
    assert fac_ps_lse(nb, None, None, Y, Yp, 0, fn=4, alpha=alpha) == pytest.approx(
        alpha * 1.0
    )


def test_err_fac_ps_l1():
    # explicit neighbourhoods with distances: centre first
    nb = [np.array([0, 1]), np.array([1, 0, 2]), np.array([2, 1, 3]), np.array([3, 2])]
    nb_dst = [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0]),
    ]
    Yp = np.array([0.0, 1.0, 1.0, 1.0])
    Y = Yp + 0.1  # adequate: small residuals, sums <= 4
    # err l1 part: sum over nb of squared slopes (dst != 0)
    slopes = [(0.0 - 1.0) / 1.0] + [
        (1.0 - 0.0) / 1.0,
        (1.0 - 1.0) / 1.0,
        (1.0 - 1.0) / 1.0,
        (1.0 - 1.0) / 1.0,
        (1.0 - 1.0) / 1.0,
    ]
    l1 = sum(s * s for s in slopes)
    assert err_ps_l1(nb, nb_dst, None, Y, Yp, fn=4, alpha=1e-4) == pytest.approx(
        1e-4 * l1
    )
    # fac for point 0: neighbourhoods where 0 is among the first dim=2
    # members: [0,1] and [1,0,2] -> linear slopes (skip centre)
    expected_fac = 1e-4 * ((0.0 - 1.0) / 1.0 + (1.0 - 0.0) / 1.0 + (1.0 - 1.0) / 1.0)
    assert fac_ps_l1(nb, nb_dst, 2, Y, Yp, 0, fn=4, alpha=1e-4) == pytest.approx(
        expected_fac
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_calc_out_matches_manual():
    rng = np.random.default_rng(0)
    X = rng.random((5, 3))
    W0 = rng.uniform(-0.5, 0.5, (7, 3))
    W1 = rng.uniform(-0.5, 0.5, (7, 7))
    W2 = rng.uniform(-0.5, 0.5, (1, 7))
    y = calc_out(X, W0, W1, W2)
    O1 = 1.0 / (1.0 + np.exp(-(W0 @ X.T)))  # (7, 5)
    O2 = 1.0 / (1.0 + np.exp(-(W1 @ O1)))
    y_manual = (W2 @ O2).ravel()
    assert y == pytest.approx(y_manual)


# ---------------------------------------------------------------------------
# Training: exact agreement with the original R implementation
# ---------------------------------------------------------------------------


def test_training_matches_r_fixture(mlp_fixtures):
    """Two SGD epochs reproduce the R ssrmlp_train result bit-exactly.

    The fixture was generated with the original R ``ssrmlp_train``
    (sisireg 1.2.1) through its re-training path with fixed initial
    weights and a recorded ``sample()`` permutation.  Agreement is
    expected to machine precision; longer training runs diverge in the
    last bits because the sign criterion is discontinuous (a residual
    sitting within BLAS/pow noise of zero can flip), which is inherent
    to any cross-platform reimplementation.
    """
    X, Y, init, mix = mlp_fixtures
    model = ssrmlp_train(
        X, Y, opt="ps", W=init, max_iter=2, rng=FixedPerm(mix)
    )
    assert np.max(np.abs(model.W0 - _load_mat("mlp_W0_002.csv"))) == 0.0
    assert np.max(np.abs(model.W1 - _load_mat("mlp_W1_002.csv"))) == 0.0
    assert np.max(np.abs(model.W2 - _load_mat("mlp_W2_002.csv"))) == 0.0
    yp_r = pd.read_csv(_DATA / "mlp_yp_002.csv")["yp"].to_numpy()
    assert ssrmlp_predict(X, model) == pytest.approx(yp_r, abs=1e-12)


# ---------------------------------------------------------------------------
# Training: statistical behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 40
    X = rng.random((n, 2))
    Y = 0.3 * X[:, 0] + 0.6 * X[:, 1] + rng.normal(0, 0.05, n)
    return X, Y


@pytest.mark.parametrize("opt", ["ps", "lse", "ps_lse", "ps_l1"])
def test_training_reduces_error(regression_data, opt):
    X, Y = regression_data
    model = ssrmlp_train(X, Y, opt=opt, max_iter=200, rng=7)
    yp = ssrmlp_predict(X, model)
    assert np.all(np.isfinite(yp))
    rmse = np.sqrt(np.mean((Y - yp) ** 2))
    rmse0 = np.sqrt(np.mean((Y - Y.mean()) ** 2))
    assert rmse < 0.5 * rmse0


def test_training_reproducible(regression_data):
    X, Y = regression_data
    m1 = ssrmlp_train(X, Y, max_iter=30, rng=123)
    m2 = ssrmlp_train(X, Y, max_iter=30, rng=123)
    assert np.array_equal(m1.W0, m2.W0)
    assert np.array_equal(m1.W1, m2.W1)


def test_training_unknown_opt_falls_back_to_ps(regression_data):
    X, Y = regression_data
    m_unknown = ssrmlp_train(X, Y, opt="bogus", max_iter=25, rng=5)
    m_ps = ssrmlp_train(X, Y, opt="ps", max_iter=25, rng=5)
    assert np.array_equal(m_unknown.W0, m_ps.W0)


def test_training_ext_matches_internal(regression_data):
    """opt='ext' with the public building blocks equals the fast path."""
    X, Y = regression_data
    m_int = ssrmlp_train(X, Y, opt="ps", max_iter=25, rng=5)
    m_ext = ssrmlp_train(
        X, Y, opt="ext", max_iter=25, rng=5, errfct_ex=err_ps, facfct_ex=fac_ps
    )
    assert np.max(np.abs(m_int.W0 - m_ext.W0)) == 0.0
    assert np.max(np.abs(m_int.W2 - m_ext.W2)) == 0.0


def test_training_retrain_path(regression_data, mlp_fixtures):
    X, Y = regression_data
    _, _, init, _ = mlp_fixtures
    model = ssrmlp_train(X, Y, opt="ps", W=init, max_iter=10, rng=3)
    assert np.all(np.isfinite(ssrmlp_predict(X, model)))
    # standardisation kept from the given model
    assert np.allclose(model.minX, init.minX)
    assert model.minY == init.minY


def test_training_std_false(regression_data):
    X, Y = regression_data
    model = ssrmlp_train(X, Y, std=False, max_iter=10, rng=3)
    assert np.allclose(model.minX, 0.0)
    assert np.allclose(model.maxX, 1.0)
    assert model.minY == 0.0 and model.maxY == 1.0


def test_training_zero_iterations(regression_data):
    X, Y = regression_data
    model = ssrmlp_train(X, Y, max_iter=0, rng=3)
    # returns the initialised (random) model; predictions are finite
    assert np.all(np.isfinite(ssrmlp_predict(X, model)))


def test_training_constant_column(regression_data):
    X, Y = regression_data
    X = np.hstack([X, np.full((len(X), 1), 2.5)])  # constant -> NaN -> 0
    model = ssrmlp_train(X, Y, max_iter=30, rng=3)
    assert np.all(np.isfinite(ssrmlp_predict(X, model)))


# ---------------------------------------------------------------------------
# Prediction / factor importance
# ---------------------------------------------------------------------------


def test_predict_destandardises(mlp_fixtures):
    X, Y, init, mix = mlp_fixtures
    model = ssrmlp_train(X, Y, opt="ps", W=init, max_iter=2, rng=FixedPerm(mix))
    yp = ssrmlp_predict(X, model)
    # manual: standardised forward pass, de-standardised output
    Xn = (X - model.minX) / (model.maxX - model.minX)
    Xb = np.hstack([Xn, np.ones((len(X), 1))])
    y = calc_out(Xb, model.W0, model.W1, model.W2)
    assert yp == pytest.approx(y * (model.maxY - model.minY) + model.minY)


def test_predict_constant_column(regression_data):
    X, Y = regression_data
    model = ssrmlp_train(X, Y, max_iter=20, rng=3)
    Xq = X.copy()
    Xq[:, 0] = 1.5  # constant column -> NaN after standardisation -> 0
    Xq = np.tile(np.array([[1.5, 0.5]]), (4, 1))
    assert np.all(np.isfinite(ssrmlp_predict(Xq, model)))


def test_fii_model_sums_to_one(mlp_fixtures):
    X, Y, init, mix = mlp_fixtures
    model = ssrmlp_train(X, Y, opt="ps", W=init, max_iter=2, rng=FixedPerm(mix))
    fm = fii_model(model)
    assert fm.shape == (3,)  # two inputs + bias
    assert fm.sum() == pytest.approx(1.0)


def test_fii_prediction(mlp_fixtures):
    X, Y, init, mix = mlp_fixtures
    model = ssrmlp_train(X, Y, opt="ps", W=init, max_iter=2, rng=FixedPerm(mix))
    fp = fii_prediction(model, X)
    assert fp.shape == (3,)
    assert fp.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_errors(regression_data):
    X, Y = regression_data
    with pytest.raises(ValueError, match="2-D"):
        ssrmlp_train(np.zeros(5), Y)
    with pytest.raises(ValueError, match="n rows"):
        ssrmlp_train(X, Y[:-1])
    with pytest.raises(ValueError, match="at least 2 samples"):
        ssrmlp_train(X[:1], Y[:1])
    with pytest.raises(ValueError, match="positive integer"):
        ssrmlp_train(X, Y, k=0)
    with pytest.raises(ValueError, match="hl\\[0\\] == hl\\[1\\]"):
        ssrmlp_train(X, Y, hl=[10, 8])
    with pytest.raises(ValueError, match="non-negative"):
        ssrmlp_train(X, Y, max_iter=-1)
    with pytest.raises(ValueError, match="requires errfct_ex"):
        ssrmlp_train(X, Y, opt="ext")
    with pytest.raises(ValueError, match="2-D"):
        ssrmlp_predict(np.zeros(5), ssrmlp_train(X, Y, max_iter=1, rng=1))
