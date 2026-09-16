"""Tests for the spatial SSR regression port (R sisireg 1.2.1 ssr3d).

Covers the neighbourhood construction (k-quadrant and 4k+1
neighbourhoods with the R value-matching tie semantics), the partial
sum statistic, the weighted means (including the 4-point minimal
surface interpolation), the ``ssr3d`` Gauss-Seidel solver (equivalence
with a literal transcription of the C kernel), and the predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bssunfold.core.sisireg3d import (
    SSR3DModel,
    max_run_quantile,
    near_neighbors,
    near_neighbors_grid,
    near_neighbors_grid_quadrant,
    near_neighbors_quadrant,
    partial_sum_quantile,
    point_distance,
    ps_max_3d,
    ps_statistic_3d,
    ssr3d,
    ssr3d_predict,
    wmean,
    wmean_exp,
    wmean_ms,
)

_DATA = Path(__file__).parent / "data" / "sisireg3d"


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------


class TestPointDistance:
    def test_values(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        d = point_distance(coords, [0.0, 0.0])
        assert np.allclose(d, [0.0, 5.0, np.sqrt(2.0)])

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="coords must have shape"):
            point_distance(np.zeros(3), [0.0, 0.0])
        with pytest.raises(ValueError, match="2 components"):
            point_distance(np.zeros((2, 2)), [1.0, 2.0, 3.0])


class TestNearNeighborsQuadrant:
    def test_single_neighbor_per_quadrant(self):
        # reference at the origin: NE=(0.5,0.5), NW=(0,1), SW=(-1,0), SE=(0,-1)
        coords = np.array(
            [
                [0.0, 0.0],  # reference
                [1.0, 0.0],  # NE candidate (farther)
                [0.0, 1.0],  # NW
                [-1.0, 0.0],  # SW
                [0.0, -1.0],  # SE
                [0.5, 0.5],  # NE (nearest)
            ]
        )
        nb = near_neighbors_quadrant([0.0, 0.0], coords, k=1)
        # order NE, NW, SW, SE; the reference itself is never a quadrant
        # neighbour and is excluded for k == 1
        assert nb.tolist() == [5, 2, 3, 4]

    def test_k2_includes_coincident_points_first(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],  # NE
                [0.0, 1.0],  # NW
                [-1.0, 0.0],  # SW
                [0.0, -1.0],  # SE
                [0.5, 0.5],  # NE (nearest)
            ]
        )
        nb = near_neighbors_quadrant([0.0, 0.0], coords, k=2)
        # k > 1 prepends the coincident point (distance 0); within a
        # quadrant the kept indices follow ascending index order (R which)
        assert nb[0] == 0
        assert nb[1:].tolist() == [1, 5, 2, 3, 4]

    def test_value_matching_ties_inflate(self):
        # two equidistant NE candidates with k = 1: both are kept because
        # R matches distance *values* (d %in% head(sort(d1), k))
        coords = np.array(
            [
                [0.0, 0.0],
                [0.6, 0.8],  # NE, d = 1.0
                [0.8, 0.6],  # NE, d = 1.0
                [0.0, 2.0],  # NW, d = 2.0
                [-2.0, 0.0],  # SW
                [0.0, -2.0],  # SE
            ]
        )
        nb = near_neighbors_quadrant([0.0, 0.0], coords, k=1)
        assert 1 in nb.tolist() and 2 in nb.tolist()
        assert nb.tolist() == [1, 2, 3, 4, 5]

    def test_fractional_k_truncates(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],  # NE d ~ 0.707
                [1.0, 0.0],  # NE d = 1
                [0.0, 1.0],  # NW
                [-1.0, 0.0],  # SW
                [0.0, -1.0],  # SE
            ]
        )
        # k = 1.9 truncates to one NE neighbour, but k != 1 triggers the
        # nb0 branch of the R aggregation (reference point at coords[0])
        nb = near_neighbors_quadrant([0.0, 0.0], coords, k=1.9)
        assert nb.tolist() == [0, 1, 3, 4, 5]

    def test_empty_quadrants(self):
        # all points strictly north-east of the reference: only Q1 is
        # populated and k=1 keeps its nearest member
        coords = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        nb = near_neighbors_quadrant([0.0, 0.0], coords, k=1)
        assert nb.tolist() == [0]

    def test_grid_consistency(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 10, size=(40, 2))
        for i in [0, 7, 23, 39]:
            nb1 = near_neighbors_quadrant(coords[i], coords, k=1)
            # the reference point itself is not among its quadrant neighbours
            assert i not in nb1.tolist()
            # every returned point lies in one of the four quadrants
            x0, y0 = coords[i]
            for j in nb1:
                xj, yj = coords[j]
                assert (
                    (xj > x0 and yj >= y0)
                    or (xj <= x0 and yj > y0)
                    or (xj < x0 and yj <= y0)
                    or (xj >= x0 and yj < y0)
                )


class TestNearNeighbors:
    def test_4k_plus_1_with_center(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [5.0, 5.0],
            ]
        )
        nb = near_neighbors([0.0, 0.0], coords, k=1)
        # the 5 nearest, distance ties inflate to all points with d <= 1
        assert set(nb.tolist()) == {0, 1, 2, 3, 4}

    def test_ties_inflate_beyond_4k_plus_1(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.7, 0.7],  # d = 0.99
            ]
        )
        nb = near_neighbors([0.0, 0.0], coords, k=1)
        # 4k+1 = 5 smallest values: 0, 0.99, 1, 1, 1 -> all 6 points match
        assert nb.size == 6

    def test_grid_all_points_for_small_k(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 1, size=(9, 2))
        nb = near_neighbors(coords[4], coords, k=100)
        assert nb.size == 9  # 4k+1 exceeds n: everything matches


def test_grid_functions_match_single_calls():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 5, size=(20, 2))
    nb_grid = near_neighbors_grid(coords, 1)
    nbq_grid = near_neighbors_grid_quadrant(coords, 1)
    assert len(nb_grid) == 20 and len(nbq_grid) == 20
    for i in range(20):
        assert np.array_equal(nb_grid[i], near_neighbors(coords[i], coords, k=1))
        assert np.array_equal(
            nbq_grid[i], near_neighbors_quadrant(coords[i], coords, k=1)
        )


# ---------------------------------------------------------------------------
# Partial sums
# ---------------------------------------------------------------------------


class TestPsMax3d:
    def test_basic(self):
        coords = np.array([[i, 0.0] for i in range(10)])
        z = np.array([1, 1, 1, -1, 1, 1, 1, 1, -1, -1], dtype=float)
        mu = np.zeros(10)
        nb = near_neighbors_grid(coords, 1)
        # neighbourhoods of size 4k+1 = 5 along the line; window sums of
        # the signs [+,+,+,-,+,+,+,+,-,-] are bounded by 3
        val = ps_max_3d(nb, z, mu, 1)
        assert val == 3

    def test_skips_oversized_neighborhoods(self):
        # neighbourhoods whose size differs from 4k+1 (e.g. inflated by
        # distance ties) are skipped exactly as in R psmax3dR
        z = np.ones(6)
        mu = np.zeros(6)
        nb = [
            np.array([0, 1, 2, 3, 4]),  # 5 members -> qualifies, sum = 5
            np.array([0, 1, 2, 3, 4, 5]),  # 6 members -> skipped
        ]
        assert ps_max_3d(nb, z, mu, 1) == 5
        assert ps_max_3d(nb[::-1], z, mu, 1) == 5
        # nothing qualifies when all sizes differ from 4k+1
        assert ps_max_3d([np.array([0, 1, 2, 3])], z, mu, 1) == 0

    def test_perfect_model_gives_zero(self):
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, 10, size=(50, 2))
        z = rng.normal(0, 1, 50)
        nb = near_neighbors_grid(coords, 2)
        assert ps_max_3d(nb, z, z.copy(), 2) == 0

    def test_length_validation(self):
        coords = np.zeros((5, 2))
        nb = near_neighbors_grid(coords, 1)
        with pytest.raises(ValueError, match="same length"):
            ps_max_3d(nb, np.zeros(5), np.zeros(4), 1)


class TestPsStatistic3d:
    def test_shapes_and_zero_model(self):
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, 10, size=(60, 2))
        z = rng.normal(0, 1, 60)
        ps, fn = ps_statistic_3d(coords, z, z.copy())
        assert ps.shape == (10,)  # max_int = int(max(60/20, 10)) = 10
        assert np.all(ps == 0)
        # fn evaluated at 5, 9, ..., 4*max_int+1
        assert fn.shape == ps.shape
        assert np.allclose(fn, partial_sum_quantile(60, np.arange(5, 4 * 10 + 2, 4)))

    def test_default_max_int(self):
        coords = np.zeros((200, 2))
        coords[:, 0] = np.arange(200)
        z = np.arange(200, dtype=float)
        ps, fn = ps_statistic_3d(coords, z, z * 0.5)  # model = z/2: signs all +
        assert ps.shape == (10,)  # max(200/20, 10) = 10
        assert np.all(ps >= 0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            ps_statistic_3d(np.zeros((5, 2)), np.zeros(5), np.zeros(4))


# ---------------------------------------------------------------------------
# Weighted means
# ---------------------------------------------------------------------------


class TestWeightedMeans:
    def test_wmean_coincident_point(self):
        coords = np.array([[1.0, 1.0], [2.0, 2.0]])
        mu = np.array([7.0, 9.0])
        assert wmean(coords, mu, [1.0, 1.0]) == 7.0

    def test_wmean_symmetric(self):
        coords = np.array([[1.0, 0.0], [-1.0, 0.0]])
        mu = np.array([4.0, 8.0])
        assert wmean(coords, mu, [0.0, 0.0]) == pytest.approx(6.0)

    def test_wmean_exp_symmetric(self):
        coords = np.array([[1.0, 0.0], [-1.0, 0.0]])
        mu = np.array([4.0, 8.0])
        assert wmean_exp(coords, mu, [0.0, 0.0]) == pytest.approx(6.0)

    def test_wmean_exp_validates(self):
        with pytest.raises(ValueError, match="same length"):
            wmean_exp(np.zeros((2, 2)), np.zeros(3), [0.0, 0.0])
        with pytest.raises(ValueError, match="same length"):
            wmean(np.zeros((2, 2)), np.zeros(3), [0.0, 0.0])

    def test_wmean_ms_exact_for_linear_on_square(self):
        # square corners in the aggregation order NE, NW, SW, SE; the
        # centered query has rx = ry = 0.5 and the interpolation is exact
        # for linear functions
        sq = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        a, b, c = 2.0, 3.0, 1.0
        mu = a * sq[:, 0] + b * sq[:, 1] + c
        val = wmean_ms(sq, mu, [0.5, 0.5])
        assert val == pytest.approx(a * 0.5 + b * 0.5 + c)

    def test_wmean_ms_constant(self):
        sq = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        assert wmean_ms(sq, np.full(4, 5.0), [0.3, 0.7]) == pytest.approx(5.0)

    def test_wmean_ms_requires_four_points(self):
        assert np.isnan(wmean_ms(np.zeros((3, 2)), np.zeros(3), [0.0, 0.0]))


# ---------------------------------------------------------------------------
# ssr3d solver
# ---------------------------------------------------------------------------


def _literal_ssr3d(coords, dat, k=None, fn=None, iter=1000):
    """Literal transcription of C ``ssr3dC`` (O(n^2) chi membership scans).

    Independent reference implementation used to validate the optimised
    incremental sign-sum bookkeeping of :func:`ssr3d`.
    """
    coords = np.asarray(coords, float)
    dat = np.asarray(dat, float).ravel()
    n = dat.shape[0]
    if k is None:
        k = max_run_quantile(n) / 2.0
    if fn is None:
        fn = float(partial_sum_quantile(n, 4.0 * k))
    nb = near_neighbors_grid(coords, k)
    nb1 = near_neighbors_grid_quadrant(coords, 1)

    def sgn(v):
        return int(v > 0) - int(v < 0)

    def chi(ind, mu):
        for members in nb:
            if ind in members.tolist():
                s = sum(sgn(dat[m] - mu[m]) for m in members)
                if abs(s) > fn:
                    return False
        return True

    def wmean_exp_nb(vals, i):
        members = nb1[i]
        d = np.sqrt(np.sum((coords[members] - coords[i]) ** 2, axis=1))
        keep = d != 0
        g = np.exp(-d[keep])
        return float(np.sum(g * vals[members][keep]) / np.sum(g))

    def wmean_exp_start(vals, i, neighbourhoods):
        members = np.asarray(neighbourhoods[i])
        d = np.sqrt(np.sum((coords[members] - coords[i]) ** 2, axis=1))
        keep = d != 0
        g = np.exp(-d[keep])
        return float(np.sum(g * vals[members][keep]) / np.sum(g))

    mu = np.array([wmean_exp_start(dat, i, nb) for i in range(n)])
    for i in range(n):
        if not chi(i, mu):
            mu[i] = dat[i]
    for _ in range(max(int(iter), 0)):
        change = False
        for i in range(n):
            oldval = mu[i]
            newval = wmean_exp_nb(mu, i)
            mu[i] = newval
            if not chi(i, mu):
                if sgn(dat[i] - newval) == sgn(dat[i] - oldval):
                    mu[i] = newval
                else:
                    mu[i] = dat[i]
            if abs(mu[i] - oldval) > 1e-7:
                change = True
        if not change:
            break
    return mu


class TestSsr3d:
    def test_model_fields(self):
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 5, size=(30, 2))
        dat = rng.normal(0, 1, 30)
        model = ssr3d(coords, dat, iter=50)
        assert isinstance(model, SSR3DModel)
        assert model.koord.shape == (30, 2)
        assert model.z.shape == (30,)
        assert model.mu.shape == (30,)
        assert np.array_equal(model.z, dat)

    def test_equivalence_with_literal_c_transcription(self):
        rng = np.random.default_rng(6)
        coords = rng.uniform(0, 6, size=(24, 2))
        dat = (
            1.5 * coords[:, 0]
            - 2.0 * coords[:, 1]
            + 0.3 * np.sin(2 * coords[:, 0])
            + rng.normal(0, 0.4, 24)
        )
        mu_ref = _literal_ssr3d(coords, dat)
        model = ssr3d(coords, dat)
        assert np.max(np.abs(mu_ref - model.mu)) < 1e-9

    def test_equivalence_explicit_k_fn(self):
        rng = np.random.default_rng(7)
        coords = rng.uniform(0, 5, size=(20, 2))
        dat = rng.normal(0, 1, 20)
        mu_ref = _literal_ssr3d(coords, dat, k=2, fn=3.0)
        model = ssr3d(coords, dat, k=2, fn=3.0)
        assert np.max(np.abs(mu_ref - model.mu)) < 1e-12

    def test_dense_data_tracks_truth(self):
        """On dense scattered data the minimal surface tracks the trend.

        Note the 3-D SSR surface is a *weak* smoother: the original R
        implementation does not necessarily reduce the RMSE on its own
        vignette example (verified against sisireg 1.2.1 output); the
        robust invariant is a high correlation with the underlying
        trend, together with the exact R fixture match above.
        """
        rng = np.random.default_rng(2)
        n = 900
        xy = rng.normal(0, 1, size=(n, 2))
        truth = np.arctan2(xy[:, 0], xy[:, 1])
        z = truth + rng.normal(0, 1.0, n)
        model = ssr3d(xy, z, iter=300)
        assert np.all(np.isfinite(model.mu))
        corr = np.corrcoef(model.mu, truth)[0, 1]
        assert corr > 0.8
        # bounded by the data range
        lo, hi = z.min() - 1.0, z.max() + 1.0
        assert np.all(model.mu > lo) and np.all(model.mu < hi)

    def test_bounded_by_data_range(self):
        rng = np.random.default_rng(9)
        coords = rng.uniform(0, 10, size=(60, 2))
        dat = 2.0 * coords[:, 0] + 3.0 * coords[:, 1] + rng.normal(0, 0.2, 60)
        model = ssr3d(coords, dat, iter=100)
        lo, hi = dat.min() - 1.0, dat.max() + 1.0
        assert np.all(model.mu > lo) and np.all(model.mu < hi)

    def test_validation(self):
        with pytest.raises(ValueError, match="same length"):
            ssr3d(np.zeros((5, 2)), np.zeros(4))
        with pytest.raises(ValueError, match="at least 5 data points"):
            ssr3d(np.zeros((4, 2)), np.zeros(4))
        with pytest.raises(ValueError, match=r"coords must have shape"):
            ssr3d(np.zeros(5), np.zeros(5))

    def test_zero_iterations_returns_start_values(self):
        rng = np.random.default_rng(10)
        coords = rng.uniform(0, 5, size=(25, 2))
        dat = rng.normal(0, 1, 25)
        model = ssr3d(coords, dat, iter=0)
        # start values: exponential means over the partial sum neighbourhoods
        nb = near_neighbors_grid(coords, max_run_quantile(25) / 2.0)
        for i in range(25):
            members = nb[i]
            d = np.sqrt(np.sum((coords[members] - coords[i]) ** 2, axis=1))
            keep = d != 0
            g = np.exp(-d[keep])
            expected = np.sum(g * dat[members][keep]) / np.sum(g)
            if abs(model.mu[i] - expected) > 1e-12:
                # reverted to the observation by the adequacy test
                assert model.mu[i] == dat[i]

    def test_matches_original_r_fixture(self):
        """Reproduces the original R ssr3d result to machine precision.

        The fixture was generated with the original R ``ssr3d``
        (sisireg 1.2.1, k = 3, fn = 2.5, 200 sweeps) on 40 scattered
        points.
        """
        df = pd.read_csv(_DATA / "model_k3.csv")
        coords = df[["x", "y"]].to_numpy()
        z = df["z"].to_numpy()
        model = ssr3d(coords, z, k=3, fn=2.5, iter=200)
        assert model.mu == pytest.approx(df["mu"].to_numpy(), abs=1e-10)

        q = pd.read_csv(_DATA / "predict_k3.csv")
        pred = ssr3d_predict(model, q[["x", "y"]].to_numpy())
        assert pred == pytest.approx(q["p_exp"].to_numpy(), abs=1e-10)
        pred_ms = ssr3d_predict(model, q[["x", "y"]].to_numpy(), ms=True)
        assert pred_ms == pytest.approx(q["p_ms"].to_numpy(), abs=1e-10)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestSsr3dPredict:
    def test_predict_near_model_values_dense(self):
        rng = np.random.default_rng(11)
        xy = rng.uniform(0, 10, size=(300, 2))
        z = np.sin(0.5 * xy[:, 0]) + 0.3 * xy[:, 1]
        model = ssr3d(xy, z, iter=100)
        pred = ssr3d_predict(model, xy)
        assert pred.shape == (300,)
        # with dense sampling the quadrant means approximate the surface
        assert np.max(np.abs(pred - model.mu)) < 0.5

    def test_predict_ms_runs(self):
        rng = np.random.default_rng(12)
        xy = rng.uniform(0, 10, size=(150, 2))
        z = rng.normal(0, 1, 150)
        model = ssr3d(xy, z, iter=50)
        pred = ssr3d_predict(model, xy[:20], ms=True)
        assert pred.shape == (20,)

    def test_predict_ms_requires_all_quadrants(self):
        # query far outside the hull: empty quadrants -> NaN
        rng = np.random.default_rng(13)
        xy = rng.uniform(0, 5, size=(30, 2))
        model = ssr3d(xy, rng.normal(0, 1, 30), iter=20)
        pred = ssr3d_predict(model, np.array([[100.0, 100.0]]), ms=True)
        assert np.isnan(pred[0])

    def test_predict_validation(self):
        model = ssr3d(np.random.default_rng(14).uniform(0, 5, (20, 2)),
                      np.random.default_rng(15).normal(0, 1, 20), iter=10)
        with pytest.raises(ValueError, match=r"xy must have shape"):
            ssr3d_predict(model, np.zeros(3))
        with pytest.raises(ValueError, match=r"xy must have shape"):
            ssr3d_predict(model, np.zeros((2, 3)))
