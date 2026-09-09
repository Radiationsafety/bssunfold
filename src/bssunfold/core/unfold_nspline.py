"""N-spline unfolding method for neutron spectrum reconstruction.

Implementation of the unfolding approach of R. F. Islamgulov and
V. D. Lartsev, ``Reconstruction of neutron spectra from activation
measurements in the form of N-splines``, Atomic Energy 104(5), 295-302
(May 2008) -- RFNC-VNIITF named after E. I. Zababakhin.

The method solves the activation-integral system

    Q_i = int sigma_i(E) phi(E) dE,   i = 1..N                (Eq. 1)

by *parameterising* the sought spectrum phi(E) with a specialised
"neutron" spline (N-spline) whose basis functions are

    N_k(E) = exp(a_k + q_k ln E + r_k E),  E_k <= E <= E_{k+1},
    k = 1..M                                                 (Eq. 2)

i.e. piecewise functions whose logarithm is linear both in ln E and in
E.  This family contains the classical model spectra (1/E, Maxwellian
evaporation exp(-E/T), fission-like sqrt(E) exp(-bE), two-component
Maxwell + slowed-down representations, ...) as particular members, so
the basis is close to complete for reactor and accelerator spectra and
only 3M parameters describe the whole spectrum.

The module implements the three components of the paper:

1. ``build_continuity_matrix`` / ``fit_nspline`` -- the N-spline
   itself.  C0/C1 continuity of N(E) at the interior knots (Eqs. 3-4)
   is imposed through the block matrix D (Eq. 5), and the pointwise
   approximation of a tabulated spectrum (Eqs. 6-7) reduces to a
   weighted linear least-squares problem in the log domain subject to
   the linear equality constraints D X = 0:

       G X = Y,   D X = 0,   X = (a, q, r)^T,

   solved here via the KKT (Lagrange multiplier) system.

2. ``solve_nspline_full`` -- the *directed divergence minimisation*
   loop (generalised MIRD algorithm of Lartsev, Preprint RFNC-VNIITF
   No. 216, 2005; Tarasko, Preprint FEI No. 1446, 1983).  With
   normalised measured activations p_i = Q_i / sum(Q) the functional

       H = sum_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0   (Eqs. 8-9)

   (pN_i = normalised calculated activations) is driven down by the
   flux-conserving gradient iteration

       phi_{n+1}(E) = phi_n(E) [1 - dmu_n (R_n(E) - Rbar_n)],

       R_n(E) = sum_i (p_i / Q_i) sigma_i(E) ln(pN_i / p_i),

   where Rbar_n is the flux-weighted mean of R_n (keeps the fluence
   constant) and the step dmu_n starts from the paper's conservative
   value 0.1 / sup|R_n - Rbar_n| and is halved (backtracking) until H
   decreases.  After *every* iteration the current spectrum is smoothed
   by re-fitting the N-spline (pointwise approximation of item 1),
   which is the key regularisation trick of the paper: the iteration
   effectively acts on 3M spline parameters instead of n bin values,
   avoiding the nonlinearity / local-minimum issues of a direct
   nonlinear fit of the spline parameters to the activation integrals.

3. Stopping criteria and quality control of the paper: iterations stop
   when H reaches the level corresponding to the measurement errors,

       H <= H_target = 0.5 * mean_i (dQ_i / Q_i)^2,

   or when the relative decrease of H per iteration falls below
   ``tol``.  The acceptability of the recovered spectrum is measured
   by the mean-squared residual

       nev = sqrt( 1/(N-1) sum_i ((Qr_i - Q_i)/dQ_i)^2 ),

   considered acceptable when nev <= 1 + 2/sqrt(N).

Knot sets used in the paper for the BARS-5, IGRIK (channel and
surface) and YAGUAR reactors are provided in ``NSPLINE_KNOT_PRESETS``;
``auto_knots`` builds a log-uniform default grid.  The module follows
the standard bssunfold solver API:

* ``solve_nspline(A, b, x0, E_MeV, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``solve_nspline_full(...)`` -- the same solver returning a rich
  diagnostics dictionary (H history, nev, spline parameters, ...);
* ``unfold_nspline(...)`` -- Detector-facing wrapper (also exposed as
  ``Detector.unfold_nspline``).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._base_unfolder import run_unfolding

__all__ = [
    "NSPLINE_KNOT_PRESETS",
    "auto_knots",
    "build_continuity_matrix",
    "fit_nspline",
    "nspline_eval",
    "directed_divergence",
    "solve_nspline",
    "solve_nspline_full",
    "unfold_nspline",
]

# Numerical floors / guards -------------------------------------------------
_PHI_FLOOR = 1e-300  # absolute floor for positive spectrum values
_LOG_CLIP = 50.0     # clip for ln(pN/p) ratio to tame outliers

_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # numpy < 2.0 fallback
    _trapezoid = np.trapz

# Knot presets (Eq. 2 knot sets, MeV) as used in the paper (section
# "Vosstanovlenie spektrov reaktorov BARS-5, IGRIK, YaGUAR").
NSPLINE_KNOT_PRESETS: dict[str, tuple[float, ...]] = {
    # BARS-5 reactor channel
    "BARS5_channel": (
        1e-10, 1.3e-7, 3.83e-7, 8e-6, 2e-5, 3e-5, 7.3e-5,
        3.2e-3, 0.38, 0.95, 7.0, 17.0, 20.0,
    ),
    # IGRIK reactor channel
    "IGRIK_channel": (
        1e-10, 2e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1.5e-4,
        3e-4, 6e-4, 6e-3, 0.27, 1.0, 2.7, 7.0, 13.0, 20.0,
    ),
    # IGRIK reactor surface
    "IGRIK_surface": (
        1e-10, 2e-8, 1e-7, 2e-7, 3e-6, 5e-6, 2.5e-4, 0.6,
        0.8, 1.5, 2.7, 7.0, 11.5, 14.0, 20.0,
    ),
    # YAGUAR reactor channel
    "YAGUAR_channel": (
        1e-10, 2e-8, 1e-7, 6e-7, 1e-6, 3e-6, 1e-5, 4.3e-5,
        1.8e-4, 6.3e-4, 5e-3, 0.6, 0.8, 1.0, 2.5, 7.0, 11.0,
        13.0, 20.0,
    ),
}

# ---------------------------------------------------------------------------
# Knot utilities
# ---------------------------------------------------------------------------


def auto_knots(E_MeV: np.ndarray, n_segments: int = 12) -> tuple[float, ...]:
    """Build a log-uniform knot grid spanning the energy range of ``E_MeV``.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid (MeV), must contain positive values.
    n_segments : int, optional
        Number of spline segments (default: 12); the returned tuple has
        ``n_segments + 1`` knots from ``min(E)`` to ``max(E)``.

    Returns
    -------
    Tuple[float, ...]
        Strictly increasing knot sequence (MeV).
    """
    E = np.asarray(E_MeV, dtype=float).ravel()
    Epos = E[E > 0]
    if Epos.size < 2:
        raise ValueError(
            "auto_knots requires at least two positive energy points, got "
            f"{Epos.size}"
        )
    emin = float(Epos.min())
    emax = float(Epos.max())
    if not np.isfinite(emin) or not np.isfinite(emax) or emin >= emax:
        raise ValueError(
            f"auto_knots requires finite min(E) < max(E), got [{emin}, {emax}]"
        )
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    return tuple(float(x) for x in np.geomspace(emin, emax, n_segments + 1))


def _resolve_knots(
    knots: str | Sequence[float] | None,
    E_MeV: np.ndarray,
    n_segments: int | None = None,
) -> tuple[tuple[float, ...], str]:
    """Resolve the knot specification to a valid knot tuple.

    ``knots`` may be ``None`` (auto log-uniform grid), a preset name from
    ``NSPLINE_KNOT_PRESETS`` or an explicit increasing sequence.  Explicit
    and preset knots are clipped to the energy range of ``E_MeV``.
    """
    E = np.asarray(E_MeV, dtype=float).ravel()
    emin = float(E[E > 0].min()) if np.any(E > 0) else 1e-10
    emax = float(E.max())

    if knots is None:
        if n_segments is None:
            n = max(E.size, 2)
            n_segments = int(min(12, max(4, n // 4)))
        return auto_knots(E, n_segments), "auto"

    if isinstance(knots, str):
        key = knots.strip()
        if key not in NSPLINE_KNOT_PRESETS:
            available = ", ".join(sorted(NSPLINE_KNOT_PRESETS))
            raise KeyError(
                f"Unknown N-spline knot preset '{key}'. "
                f"Available presets: {available}"
            )
        src, kn = f"preset:{key}", NSPLINE_KNOT_PRESETS[key]
    else:
        src, kn = "user", tuple(float(k) for k in knots)

    if len(kn) < 2:
        raise ValueError(f"N-spline needs at least 2 knots, got {len(kn)}")
    kn_arr = np.asarray(kn, dtype=float)
    if np.any(np.diff(kn_arr) <= 0):
        raise ValueError("N-spline knots must be strictly increasing")

    # Clip preset/user knots to the energy grid range and extend the
    # outer knots so that the spline domain spans the whole grid (the
    # paper defines the N-spline over the full energy scale).  Bins
    # outside the knot range would otherwise be extrapolated with the
    # boundary-segment parameters, which is numerically unsafe.
    kn_arr = np.clip(kn_arr, emin, emax)
    kn_arr = np.unique(kn_arr)
    if kn_arr[0] > emin:
        kn_arr[0] = emin
    if kn_arr[-1] < emax:
        kn_arr[-1] = emax
    if kn_arr.size < 2:
        kn_arr = np.array([emin, emax], dtype=float)
    return tuple(float(x) for x in kn_arr), src


def _segment_indices(E: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Map energy points onto spline segment indices 0..M-1."""
    k = np.searchsorted(np.asarray(knots), E, side="right") - 1
    return np.clip(k, 0, len(knots) - 2)


# ---------------------------------------------------------------------------
# N-spline definition (Eqs. 2-5)
# ---------------------------------------------------------------------------


def build_continuity_matrix(
    knots: Sequence[float],
    continuity: str = "C0C1",
) -> np.ndarray:
    """Build the spline continuity matrix ``D`` of Eq. (5).

    The N-spline parameter vector is ``X = (a, q, r)^T`` with
    ``a = (a_1..a_M)``, ``q = (q_1..q_M)``, ``r = (r_1..r_M)``.  The
    continuity conditions at the interior knots read (Eqs. 3-4):

        C0: a_k - a_{k+1} + u (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0
        C1: (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0,

    with ``u = ln E`` and ``E`` the knot value, and are assembled into

        D = [[A, B, C], [0, A, C]],   D X = 0.

    Parameters
    ----------
    knots : Sequence[float]
        Knot sequence (M = len(knots) - 1 segments).
    continuity : str, optional
        ``"C0C1"`` (default) -- continuous value and derivative;
        ``"C0"`` -- continuous value only (only the first block row of
        D is kept, as noted in the paper); ``"none"`` -- no continuity.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(rows, 3M)`` where ``rows`` is 0, M-1 or
        2(M-1) depending on ``continuity``.
    """
    kn = np.asarray(knots, dtype=float)
    M = kn.size - 1
    if M < 1:
        raise ValueError("knots must contain at least 2 values")
    cont = str(continuity).upper().replace(" ", "")
    if cont not in ("C0C1", "C0", "NONE"):
        raise ValueError(
            f"continuity must be one of 'C0C1', 'C0', 'none', got {continuity!r}"
        )

    n_int = M - 1  # interior knots
    if cont == "NONE" or n_int == 0:
        return np.zeros((0, 3 * M), dtype=float)

    rows_c0 = cont == "C0C1"
    n_rows = 2 * n_int if rows_c0 else n_int
    D = np.zeros((n_rows, 3 * M), dtype=float)
    for k in range(n_int):
        Ek = kn[k + 1]
        uk = np.log(Ek)
        # C0 row: a_k - a_{k+1} + u(q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
        D[k, M + k] = -uk
        D[k, M + k + 1] = uk
        D[k, 2 * M + k] = -Ek
        D[k, 2 * M + k + 1] = Ek
        if rows_c0:
            # C1 row: (q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
            row = n_int + k
            D[row, M + k] = -1.0
            D[row, M + k + 1] = 1.0
            D[row, 2 * M + k] = -Ek
            D[row, 2 * M + k + 1] = Ek
    return D


def nspline_eval(
    E: np.ndarray,
    a: Sequence[float],
    q: Sequence[float],
    r: Sequence[float],
    knots: Sequence[float],
) -> np.ndarray:
    """Evaluate the N-spline ``N(E) = exp(a_k + q_k ln E + r_k E)``.

    Parameters
    ----------
    E : np.ndarray
        Evaluation energies (MeV), must be positive.
    a, q, r : Sequence[float]
        Length-M parameter vectors.
    knots : Sequence[float]
        The M+1 knot values.

    Returns
    -------
    np.ndarray
        N-spline values (positive by construction).
    """
    E_arr = np.asarray(E, dtype=float).ravel()
    if np.any(E_arr <= 0):
        raise ValueError("nspline_eval requires strictly positive energies")
    a_arr = np.asarray(a, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    M = len(knots) - 1
    if not (a_arr.size == q_arr.size == r_arr.size == M):
        raise ValueError(
            f"a, q, r must all have length M={M} segments, got "
            f"{a_arr.size}, {q_arr.size}, {r_arr.size}"
        )
    k = _segment_indices(E_arr, tuple(knots))
    return np.exp(a_arr[k] + q_arr[k] * np.log(E_arr) + r_arr[k] * E_arr)


def directed_divergence(
    p_calc: np.ndarray,
    p_meas: np.ndarray,
) -> float:
    """Directed (Kullback-Leibler-type) divergence of Eq. (8-9).

        H = sum_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0,

    H = 0 iff the calculated activations equal the measured ones.

    Parameters
    ----------
    p_calc : np.ndarray
        Normalised calculated activations (pN).
    p_meas : np.ndarray
        Normalised measured activations (p).

    Returns
    -------
    float
        Non-negative divergence value.
    """
    pN = np.maximum(np.asarray(p_calc, dtype=float), 1e-300)
    p = np.maximum(np.asarray(p_meas, dtype=float), 1e-300)
    return float(np.sum(pN * np.log(pN / p) - pN + p))


# ---------------------------------------------------------------------------
# Pointwise N-spline approximation (Eqs. 6-7)
# ---------------------------------------------------------------------------


def fit_nspline(
    E: np.ndarray,
    phi: np.ndarray,
    knots: str | Sequence[float] | None = None,
    rel_err: np.ndarray | None = None,
    continuity: str = "C0C1",
    n_segments: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate a pointwise spectrum by an N-spline (Eqs. 2, 5-7).

    Solves the weighted log-domain least-squares problem with continuity
    constraints of the paper:

        min_X sum_j w_j^2 (a_kj + u_j q_kj + E_j r_kj - ln phi_j)^2
        s.t.  D X = 0,   w_j = 1 / eps_j,

    via the KKT (Lagrange multiplier) system

        [[G^T W G, D^T], [D, 0]] [X; lam] = [G^T W Y; 0].

    Parameters
    ----------
    E : np.ndarray
        Energy grid (MeV), positive values.
    phi : np.ndarray
        Spectrum values on the grid (non-negative; zeros are floored and
        down-weighted).
    knots : str / Sequence[float] / None, optional
        Knot preset name, explicit knots or ``None`` for an automatic
        log-uniform grid (default).
    rel_err : np.ndarray, optional
        Relative pointwise errors eps_j; weights w_j = 1/eps_j (paper's
        Eq. 7).  ``None`` means unit weights.
    continuity : str, optional
        ``"C0C1"`` (default), ``"C0"`` or ``"none"`` -- see
        :func:`build_continuity_matrix`.
    n_segments : int, optional
        Number of segments when ``knots=None`` (default: adaptive).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        ``(N_E, info)`` where ``N_E`` is the fitted spline on ``E`` and
        ``info`` carries ``knots``, ``a``/``q``/``r`` parameter arrays,
        the weighted RMS log-residual and the continuity option.
    """
    E_arr = np.asarray(E, dtype=float).ravel()
    phi_arr = np.asarray(phi, dtype=float).ravel()
    if E_arr.size != phi_arr.size:
        raise ValueError(
            f"E and phi length mismatch: {E_arr.size} vs {phi_arr.size}"
        )
    if np.any(E_arr <= 0):
        raise ValueError("fit_nspline requires strictly positive energies")
    if phi_arr.size < 3:
        raise ValueError("fit_nspline requires at least 3 spectrum points")

    kn, src = _resolve_knots(knots, E_arr, n_segments)
    M = len(kn) - 1
    n = E_arr.size

    # Point -> segment mapping and log-domain design matrix G (n x 3M).
    kseg = _segment_indices(E_arr, kn)
    u = np.log(E_arr)
    G = np.zeros((n, 3 * M), dtype=float)
    G[np.arange(n), kseg] = 1.0
    G[np.arange(n), M + kseg] = u
    G[np.arange(n), 2 * M + kseg] = E_arr

    # Floor tiny/zero bins and down-weight them so they do not drag the fit.
    phi_max = float(phi_arr.max())
    tiny = max(_PHI_FLOOR, 1e-12 * phi_max)
    floored = phi_arr < tiny
    y = np.log(np.where(floored, tiny, phi_arr))

    if rel_err is None:
        w = np.ones(n, dtype=float)
    else:
        w = 1.0 / np.maximum(np.asarray(rel_err, dtype=float), 1e-12)
    w = np.where(floored, 1e-3 * w, w)  # strong relative weight penalty

    D = build_continuity_matrix(kn, continuity)
    nc = D.shape[0]

    # Weighted normal equations + KKT constraint block.  No ridge is
    # added: np.linalg.lstsq already returns the minimum-norm solution
    # for rank-deficient (empty-segment) systems without biasing the fit.
    Gw = G * w[:, None]
    yw = y * w
    H_norm = Gw.T @ Gw
    KKT = np.zeros((3 * M + nc, 3 * M + nc), dtype=float)
    KKT[: 3 * M, : 3 * M] = H_norm
    if nc:
        KKT[: 3 * M, 3 * M:] = D.T
        KKT[3 * M:, : 3 * M] = D
    rhs = np.concatenate([Gw.T @ yw, np.zeros(nc)])

    sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
    X = sol[: 3 * M]

    N_E = np.exp(G @ X)
    resid = w * (G @ X - y)
    rms = float(np.sqrt(np.mean(resid**2)) / max(float(np.mean(w)), 1e-300))

    info: dict[str, Any] = {
        "knots": kn,
        "knots_source": src,
        "continuity": continuity,
        "a": X[:M].copy(),
        "q": X[M : 2 * M].copy(),
        "r": X[2 * M:].copy(),
        "log_rms_residual": rms,
    }
    return N_E, info


# ---------------------------------------------------------------------------
# Directed-divergence unfolding with per-iteration N-spline smoothing
# ---------------------------------------------------------------------------


def solve_nspline_full(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    E_MeV: np.ndarray | None = None,
    knots: str | Sequence[float] | None = None,
    sigma_rel: np.ndarray | None = None,
    continuity: str = "C0C1",
    max_iterations: int = 200,
    tol: float = 1e-3,
    step_theta: float = 0.1,
    smoothing: bool = True,
    n_segments: int | None = None,
) -> dict[str, Any]:
    """Full N-spline unfolding with diagnostics (Islamgulov & Lartsev, 2008).

    Iteratively minimises the directed divergence H between the measured
    and calculated normalised activations, smoothing the spectrum through
    an N-spline fit at every iteration (the paper's regularisation).
    Uses the paper's stopping criteria (H at the measurement-error level
    or stalled relative decrease) and reports the ``nev`` residual
    statistic with the acceptance bound ``nev <= 1 + 2/sqrt(N)``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)`` -- activation responses of the
        detectors on the energy grid.
    b : np.ndarray
        Measured readings / activation integrals ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess ``(n,)``.  ``None`` means a flat spectrum.
        In the spirit of the paper this should be a Monte-Carlo
        calculated spectrum when available.
    E_MeV : np.ndarray, optional
        Energy grid (MeV).  Required; must be positive.
    knots : str / Sequence[float] / None, optional
        Knot preset name (see ``NSPLINE_KNOT_PRESETS``), explicit knot
        sequence or ``None`` (automatic log-uniform grid).
    sigma_rel : np.ndarray, optional
        Relative measurement uncertainties dQ_i/Q_i ``(m,)`` used by the
        stopping criteria and ``nev`` (default: 0.1 for every detector).
    continuity : str, optional
        ``"C0C1"`` (default), ``"C0"`` or ``"none"``.
    max_iterations : int, optional
        Iteration budget (default: 200).
    tol : float, optional
        Relative-decrease stopping tolerance for H (default: 1e-3).
    step_theta : float, optional
        Conservative initial step factor: dmu = step_theta / sup|R-Rbar|
        (the paper's 0.1); backtracking halves it while H increases.
    smoothing : bool, optional
        Re-fit the N-spline after every iteration (default: True, the
        paper's procedure; ``False`` reduces to the plain MIRD loop).
    n_segments : int, optional
        Number of spline segments when ``knots=None``.

    Returns
    -------
    Dict[str, Any]
        Diagnostics dictionary with keys ``spectrum``, ``iterations``,
        ``converged``, ``stop_reason``, ``H``, ``H_history``,

        ``H_target``, ``nev``, ``nev_limit``, ``acceptable``, ``Qr``,
        ``relative_residuals``, ``fluence``, ``mean_energy``,
        ``knots``, ``knots_source``, ``continuity`` and ``params``.
    """
    A_arr = np.atleast_2d(np.asarray(A, dtype=float))
    b_arr = np.asarray(b, dtype=float).ravel()
    if A_arr.ndim != 2:
        raise ValueError(f"Response matrix A must be 2D, got {A_arr.ndim}D")
    m, n = A_arr.shape
    if b_arr.size != m:
        raise ValueError(f"b length ({b_arr.size}) does not match A rows ({m})")
    if m < 1:
        raise ValueError("At least one measurement is required")
    if E_MeV is None:
        raise ValueError("E_MeV (energy grid in MeV) is required")
    E = np.asarray(E_MeV, dtype=float).ravel()
    if E.size != n:
        raise ValueError(f"E_MeV length ({E.size}) does not match A columns ({n})")
    if np.any(E <= 0):
        raise ValueError("E_MeV must contain strictly positive energies")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if not 0.0 < step_theta <= 1.0:
        raise ValueError(f"step_theta must be in (0, 1], got {step_theta}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")

    # Detectors with positive readings only (zero measurements carry no
    # information for the divergence minimisation).
    valid = b_arr > 0
    if not np.any(valid):
        raise ValueError(
            "solve_nspline requires at least one positive measurement"
        )
    A_v = A_arr[valid]
    b_v = b_arr[valid]

    if sigma_rel is None:
        sigma_v = np.full(b_v.size, 0.1)
    else:
        sigma_v = np.maximum(
            np.asarray(sigma_rel, dtype=float).ravel()[valid], 1e-12
        )

    kn, knot_src = _resolve_knots(knots, E, n_segments)

    # Normalised measured activations and the paper's H target: the
    # expected directed divergence when all calculated activations sit
    # 1 sigma away from the measurements, E[H] ~ 0.5 sum_i p_i delta_i^2.
    p = b_v / float(b_v.sum())
    H_target = 0.5 * float(np.sum(p * sigma_v**2))

    # Initial spectrum: rescale x0 to the measured total response, then
    # apply the N-spline smoothing (the paper takes the spline of the
    # Monte-Carlo spectrum as the initial approximation).
    if x0 is None:
        x = np.full(n, 1.0)
    else:
        x = np.asarray(x0, dtype=float).ravel()
        if x.size != n:
            raise ValueError(f"x0 length ({x.size}) does not match A columns ({n})")
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.maximum(x, 0.0)
    if x.sum() <= 0:
        x = np.full(n, 1.0)

    Qc0 = A_v @ x
    scale = float(b_v.sum()) / max(float(Qc0.sum()), 1e-300)
    x = np.maximum(x * scale, _PHI_FLOOR)

    # Pointwise relative errors for the per-iteration smoothing fits
    # (the paper's w = 1/eps weighting, Eq. 7): bins with low total
    # detector sensitivity carry less information and get proportionally
    # larger assumed errors (Poisson-like sqrt scaling), so they cannot
    # drag the spline; insensitive regions are then shaped by the C0/C1
    # continuity instead of drifting freely.
    sens = A_v.sum(axis=0)
    sens_max = float(sens.max()) if sens.size else 0.0
    if smoothing and sens_max > 0:
        smooth_rel_err = np.sqrt(
            np.clip(sens_max / np.maximum(sens, 1e-300), 1.0, 1e12)
        )
    else:
        smooth_rel_err = None

    if smoothing:
        x, fit_info = fit_nspline(
            E, x, knots=kn, rel_err=smooth_rel_err, continuity=continuity
        )
        x = np.maximum(x, _PHI_FLOOR)
        # Keep the activation scale after the shape-only spline fit.
        x *= float(b_v.sum()) / max(float((A_v @ x).sum()), 1e-300)
    else:
        fit_info: dict[str, Any] = {}

    b_total = float(b_v.sum())
    eps_scale = 1e-12 * max(b_total, 1e-300)

    def _gauge(xx: np.ndarray) -> np.ndarray:
        """Pin the activation scale: sum(A x) = sum(b).

        The MIRD iteration and H depend only on *normalised* activations,
        so the overall scale of x is a free gauge; fixing it to the
        measured total after every (shape-only) spline smoothing keeps
        Qr consistent with Q in absolute terms.
        """
        return xx * (b_total / max(float((A_v @ xx).sum()), 1e-300))

    def _state(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        xx = np.maximum(xx, _PHI_FLOOR)
        Qc_ = np.maximum(A_v @ xx, eps_scale)
        pN_ = Qc_ / max(float(Qc_.sum()), 1e-300)
        H_ = directed_divergence(pN_, p)
        return xx, Qc_, pN_, H_

    x, _Qc, pN, H = _state(_gauge(x))
    H_history = [H]
    converged = False
    stop_reason = "max_iterations"
    iterations = 0

    if H <= H_target:
        converged = True
        stop_reason = "H_target (initial)"

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        # Gradient of H w.r.t. the spectrum (up to the constant 1/sum(Q)):
        # R(E) = sum_i (p_i / Q_i) sigma_i(E) ln(pN_i / p_i).
        ln_ratio = np.clip(np.log(pN / p), -_LOG_CLIP, _LOG_CLIP)
        R = (A_v.T @ ln_ratio) / float(b_v.sum())
        x_sum = float(x.sum())
        Rbar = float(x @ R) / max(x_sum, 1e-300)
        g = R - Rbar
        g_max = float(np.max(np.abs(g)))
        if not np.isfinite(g_max) or g_max <= 0.0:
            stop_reason = "stalled_gradient"
            iterations -= 1
            break

        # Conservative paper step (dmu0 = 0.1 / sup|R - Rbar|) with
        # backtracking halving until H does not increase.
        mu = step_theta / g_max
        accepted = False
        x_new = x
        Qc_new, pN_new, H_new = _Qc, pN, H
        for _bt in range(60):
            x_trial = x * (1.0 - mu * g)
            if smoothing:
                x_trial, _ = fit_nspline(
                    E, x_trial, knots=kn, rel_err=smooth_rel_err,
                    continuity=continuity,
                )
            # Gauge-fix the activation scale after the shape-only update.
            x_new, Qc_new, pN_new, H_new = _state(_gauge(x_trial))
            if np.isfinite(H_new) and H_new <= H + 1e-4 * max(H, 1e-300):
                accepted = True
                break
            mu *= 0.5
        if not accepted:
            stop_reason = "no_further_reduction"
            iterations -= 1
            break

        H_prev = H
        x, _Qc, pN, H = x_new, Qc_new, pN_new, H_new
        H_history.append(H)

        # Stopping criteria of the paper.
        if H <= H_target:
            converged = True
            stop_reason = "H_target"
            break
        if abs(H_prev - H) <= tol * max(H_prev, 1e-300):
            converged = True
            stop_reason = "relative_change"
            break

    # Paper's acceptability statistic: nev = RMS((Qr - Q)/dQ),
    # acceptable when nev <= 1 + 2/sqrt(N).
    Qr_full = A_arr @ x
    rel_res = np.zeros(m, dtype=float)
    denom = np.maximum(sigma_v * b_v, 1e-300)
    rel_res[valid] = (Qr_full[valid] - b_v) / denom
    cnt = int(valid.sum())
    div = cnt - 1 if cnt > 1 else cnt
    nev = float(np.sqrt(float(np.sum(rel_res[valid] ** 2)) / max(div, 1)))
    nev_limit = 1.0 + 2.0 / float(np.sqrt(cnt))
    acceptable = bool(nev <= nev_limit)

    fluence = float(_trapezoid(x, E))
    mean_energy = float(_trapezoid(E * x, E) / fluence) if fluence > 0 else float("nan")

    # Final spline parameterisation of the recovered spectrum.
    M = len(kn) - 1
    a_f = fit_info.get("a")
    if a_f is None:
        kseg = _segment_indices(E, kn)
        G = np.zeros((n, 3 * M), dtype=float)
        G[np.arange(n), kseg] = 1.0
        G[np.arange(n), M + kseg] = np.log(E)
        G[np.arange(n), 2 * M + kseg] = E
        Xl = np.linalg.lstsq(G, np.log(np.maximum(x, _PHI_FLOOR)), rcond=None)[0]
        params = {"a": Xl[:M], "q": Xl[M : 2 * M], "r": Xl[2 * M :]}
    else:
        params = {
            "a": fit_info.get("a"),
            "q": fit_info.get("q"),
            "r": fit_info.get("r"),
        }

    return {
        "spectrum": x,
        "iterations": int(iterations),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "H": H,
        "H_history": H_history,
        "H_target": H_target,
        "nev": nev,
        "nev_limit": nev_limit,
        "acceptable": acceptable,
        "Qr": Qr_full,
        "relative_residuals": rel_res,
        "fluence": fluence,
        "mean_energy": mean_energy,
        "knots": kn,
        "knots_source": knot_src,
        "continuity": continuity,
        "params": params,
    }


def solve_nspline(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    E_MeV: np.ndarray | None = None,
    knots: str | Sequence[float] | None = None,
    sigma_rel: np.ndarray | None = None,
    continuity: str = "C0C1",
    max_iterations: int = 200,
    tol: float = 1e-3,
    step_theta: float = 0.1,
    smoothing: bool = True,
    n_segments: int | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem using the N-spline method.

    Thin standard-API wrapper around :func:`solve_nspline_full` returning
    the usual ``(spectrum, iterations, converged)`` tuple used by the
    bssunfold solver protocol.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess ``(n,)`` (flat when ``None``).
    E_MeV : np.ndarray, optional
        Energy grid (MeV), positive; required.
    knots : str / Sequence[float] / None, optional
        Knot preset name, explicit knots or ``None`` (auto grid).
    sigma_rel : np.ndarray, optional
        Relative measurement uncertainties (default: 0.1).
    continuity : str, optional
        Spline continuity: ``"C0C1"`` (default), ``"C0"`` or ``"none"``.
    max_iterations : int, optional
        Iteration budget (default: 200).
    tol : float, optional
        Relative H-decrease stopping tolerance (default: 1e-3).
    step_theta : float, optional
        Conservative step factor (default: 0.1, the paper's value).
    smoothing : bool, optional
        Per-iteration N-spline smoothing (default: True).
    n_segments : int, optional
        Number of segments when ``knots=None``.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)``.
    """
    result = solve_nspline_full(
        A=A,
        b=b,
        x0=x0,
        E_MeV=E_MeV,
        knots=knots,
        sigma_rel=sigma_rel,
        continuity=continuity,
        max_iterations=max_iterations,
        tol=tol,
        step_theta=step_theta,
        smoothing=smoothing,
        n_segments=n_segments,
    )
    return result["spectrum"], result["iterations"], result["converged"]


def unfold_nspline(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    knots: str | Sequence[float] | None = None,
    continuity: str = "C0C1",
    relative_uncertainty: float = 0.1,
    max_iterations: int = 200,
    tol: float = 1e-3,
    step_theta: float = 0.1,
    smoothing: bool = True,
    n_segments: int | None = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using the N-spline method (2008).

    Detector-facing wrapper of the Islamgulov & Lartsev N-spline /
    directed-divergence unfolding algorithm (Atomic Energy 104(5), 2008).

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid (MeV).
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess (the paper recommends a Monte-Carlo
        calculated spectrum); flat when ``None``.
    knots : str / Sequence[float] / None, optional
        Knot preset name (``NSPLINE_KNOT_PRESETS``: "BARS5_channel",
        "IGRIK_channel", "IGRIK_surface", "YAGUAR_channel"), explicit
        knot sequence or ``None`` for an automatic log-uniform grid.
    continuity : str, optional
        Spline continuity: ``"C0C1"`` (default), ``"C0"`` or ``"none"``.
    relative_uncertainty : float, optional
        Relative measurement uncertainty dQ/Q used in the stopping
        criteria and the ``nev`` statistic (default: 0.1).
    max_iterations : int, optional
        Iteration budget (default: 200).
    tol : float, optional
        Relative H-decrease stopping tolerance (default: 1e-3).
    step_theta : float, optional
        Conservative step factor (default: 0.1).
    smoothing : bool, optional
        Per-iteration N-spline smoothing (default: True).
    n_segments : int, optional
        Number of spline segments when ``knots=None``.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding results dictionary enriched with the
        method diagnostics (``H``, ``H_history``, ``H_target``, ``nev``,
        ``nev_limit``, ``acceptable``, ``stop_reason``, ``fluence``,
        ``mean_energy``, ``knots``, ``knots_source``).
    """
    diag: dict[str, Any] = {
        "continuity": continuity,
        "relative_uncertainty": float(relative_uncertainty),
    }

    def solve_wrapper(A_mat: np.ndarray, b_vec: np.ndarray, **kwargs):
        out = solve_nspline_full(
            A_mat,
            b_vec,
            x0=kwargs.get("x0"),
            E_MeV=E_MeV,
            knots=knots,
            sigma_rel=np.full(np.asarray(b_vec).size, float(relative_uncertainty)),
            continuity=continuity,
            max_iterations=max_iterations,
            tol=tol,
            step_theta=step_theta,
            smoothing=smoothing,
            n_segments=n_segments,
        )
        diag.update(
            H=out["H"],
            H_history=out["H_history"],
            H_target=out["H_target"],
            nev=out["nev"],
            nev_limit=out["nev_limit"],
            acceptable=out["acceptable"],
            stop_reason=out["stop_reason"],
            fluence=out["fluence"],
            mean_energy=out["mean_energy"],
            knots=out["knots"],
            knots_source=out["knots_source"],
            Qr=out["Qr"],
            relative_residuals=out["relative_residuals"],
        )
        return out["spectrum"], out["iterations"], out["converged"]

    x0_default = np.ones(n_energy_bins) / 2.0

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="NSPLINE",
        extra_output=diag,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
