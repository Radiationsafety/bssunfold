"""Shared matrix utility functions for bssunfold core modules.

This module provides common matrix operations used across unfolding methods
and regularization modules, avoiding code duplication.
"""


import numpy as np
from scipy.sparse import csc_matrix, diags


def _log_bin_widths(E_MeV: np.ndarray, n: int) -> np.ndarray:
    """Natural-logarithmic interval widths ``h_i = ln(E_{i+1}) - ln(E_i)``.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid (n,), strictly positive and strictly increasing.
    n : int
        Expected grid length (used for error messages).

    Returns
    -------
    np.ndarray
        Widths (n-1,), all strictly positive.
    """
    E = np.asarray(E_MeV, dtype=float)
    if E.shape != (n,):
        raise ValueError(
            f"E_MeV must have length {n} to match the spectrum, got {E.shape}"
        )
    if np.any(E <= 0) or np.any(np.diff(E) <= 0):
        raise ValueError(
            "E_MeV must be strictly positive and strictly increasing "
            "for grid-aware regularization"
        )
    return np.diff(np.log(E))


def _quadrature_row_weights(n: int, order: int, h: np.ndarray) -> np.ndarray:
    """Lethargy quadrature weights for the rows of a derivative operator.

    The weighted norm ``sum_i w_i (L phi)_i^2`` approximates the integral
    ``int (d^order phi / dlnE^order)^2 dlnE`` over the energy range, making
    the smoothness penalty independent of the grid sampling.

    Parameters
    ----------
    n : int
        Spectrum length.
    order : int
        Derivative order (1 or 2).
    h : np.ndarray
        Natural-log interval widths (n-1,), ``h_i = ln(E_{i+1}) - ln(E_i)``.

    Returns
    -------
    np.ndarray
        Positive row weights of length ``n-1`` (order 1) or ``n-2``
        (order 2).
    """
    if order == 1:
        return h.copy()
    return 0.5 * (h[:-1] + h[1:])


def _grid_weighted_operator(
    n: int, order: int, E_MeV: np.ndarray
) -> csc_matrix:
    """Grid-aware derivative operator with lethargy quadrature weights.

    Returns ``sqrt(W) @ L`` where ``L`` approximates the ``order``-th
    derivative with respect to ``ln E`` on the (possibly non-uniform)
    logarithmic grid and ``W = diag(w)`` holds the row quadrature weights.
    The Tikhonov term ``||L_w phi||^2`` then approximates the rotationally
    invariant Sobolev integral and is comparable across energy grids.
    """
    h = _log_bin_widths(E_MeV, n)
    L = create_derivative_matrix(n, order, E_MeV=E_MeV)
    w = _quadrature_row_weights(n, order, h)
    return diags(np.sqrt(w), format="csc") @ L


def create_derivative_matrix(
    n: int, order: int, E_MeV: np.ndarray | None = None
) -> csc_matrix:
    """Create finite difference derivative matrix in csc format.

    By default the operator uses plain unit-spaced differences (grid
    independent, legacy behaviour). When ``E_MeV`` is provided the rows are
    scaled to approximate derivatives with respect to ``ln E`` on the
    (possibly non-uniform) logarithmic grid, so that the smoothness penalty
    measures roughness per unit ``d(ln E)`` instead of per bin:

    - order 1: ``dphi/dlnE_i = (phi_{i+1} - phi_i) / h_i`` with
      ``h_i = ln(E_{i+1}) - ln(E_i)``;
    - order 2: the standard non-uniform three-point second difference
      ``2 * [(phi_{i+1}-phi_i)/h_i - (phi_i-phi_{i-1})/h_{i-1}]
      / (h_{i-1} + h_i)``, which reduces to ``[1, -2, 1] / h^2`` for a
      uniform grid.

    Parameters
    ----------
    n : int
        Size of spectrum.
    order : int
        Derivative order (1 or 2).
    E_MeV : np.ndarray, optional
        Energy grid (n,) in MeV. When given, the returned operator
        approximates derivatives with respect to ``ln E`` (grid-aware
        regularization). When None, plain finite differences are returned
        (legacy behaviour).

    Returns
    -------
    csc_matrix
        Derivative matrix in csc format of shape (n-1, n) for order=1
        or (n-2, n) for order=2.

    Raises
    ------
    ValueError
        If order is not 1 or 2, or if E_MeV is not strictly positive and
        strictly increasing with length ``n``.
    """
    if order not in (1, 2):
        raise ValueError(f"Unsupported derivative order: {order}. Use 1 or 2.")

    if E_MeV is None:
        if order == 1:
            # First derivative: [-1, 1] on shifted rows
            data = np.concatenate([[-1] * (n - 1), [1] * (n - 1)])
            row = np.concatenate([np.arange(n - 1), np.arange(n - 1)])
            col = np.concatenate([np.arange(n - 1), np.arange(1, n)])
            return csc_matrix((data, (row, col)), shape=(n - 1, n))
        # Second derivative: [1, -2, 1] on diagonals - more efficient with diags
        return diags(
            [1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float
        )

    # Grid-aware operators on the ln E grid
    h = _log_bin_widths(E_MeV, n)
    if order == 1:
        # Row i: [-1/h_i, +1/h_i]
        data = np.concatenate([-1.0 / h, 1.0 / h])
        row = np.concatenate([np.arange(n - 1), np.arange(n - 1)])
        col = np.concatenate([np.arange(n - 1), np.arange(1, n)])
        return csc_matrix((data, (row, col)), shape=(n - 1, n))

    # Non-uniform second difference, rows i = 1..n-2
    h_prev = h[:-1]  # h_{i-1}
    h_next = h[1:]  # h_i
    c_prev = 2.0 / (h_prev * (h_prev + h_next))
    c_mid = -2.0 / (h_prev * h_next)
    c_next = 2.0 / (h_next * (h_prev + h_next))
    rows = np.arange(n - 2)
    row_idx = np.repeat(rows, 3)
    col_idx = np.stack([rows, rows + 1, rows + 2], axis=1).ravel()
    data = np.stack([c_prev, c_mid, c_next], axis=1).ravel()
    return csc_matrix((data, (row_idx, col_idx)), shape=(n - 2, n))


def build_smoothness_penalty(
    n: int,
    alpha: float,
    smoothness_order: int,
    smoothness_weight: float = 1.0,
    E_MeV: np.ndarray | None = None,
) -> csc_matrix | None:
    """Build the additive smoothness (Tikhonov) penalty matrix.

    Returns ``alpha * smoothness_weight * L.T @ L`` for the first/second-order
    finite-difference matrix ``L`` (orders 1 and 2). For ``smoothness_order == 0``
    returns ``None``: zeroth-order Tikhonov is the identity term, which callers
    add separately (and only when their method semantics require it).

    Parameters
    ----------
    n : int
        Spectrum length.
    alpha : float
        Regularization strength.
    smoothness_order : int
        Derivative order (1 or 2); use 0 to request no derivative penalty.
    smoothness_weight : float, optional
        Weight applied to the derivative penalty.
    E_MeV : np.ndarray, optional
        Energy grid (n,) enabling grid-aware regularization: the penalty
        approximates ``int (d^order phi / dlnE^order)^2 dlnE`` (derivatives
        with respect to ``ln E`` plus lethargy quadrature weights) instead
        of the bin-index sum, making the penalty comparable across
        different energy grids. None (default) keeps the legacy bin-index
        operator.

    Returns
    -------
    Optional[csc_matrix]
        Sparse penalty matrix, or ``None`` when ``smoothness_order == 0``.
    """
    if smoothness_order not in (1, 2):
        return None
    if E_MeV is not None:
        L = _grid_weighted_operator(n, smoothness_order, E_MeV)
    else:
        L = create_derivative_matrix(n, smoothness_order)
    return alpha * smoothness_weight * (L.T @ L)


def make_regularization_operator(
    n: int,
    smoothness_order: int,
    identity_for_zero: bool = True,
    E_MeV: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build the regularization operator L (dense) for a derivative order.

    A single source of truth for the repeated ``_make_regoperator`` helpers
    across the iterative solvers (CGLS, GKS, ...).

    Parameters
    ----------
    n : int
        Number of energy bins.
    smoothness_order : int
        Derivative order: 0 (identity or ``None``), 1 (first), 2 (second).
    identity_for_zero : bool, optional
        Control order-0 behaviour. If ``True`` (default) returns
        ``np.eye(n)``; if ``False`` returns ``None``. Solvers that apply the
        operator implicitly (e.g. plain CGLS) pass ``False`` so they can skip
        the regularization term entirely.
    E_MeV : np.ndarray, optional
        Energy grid (n,) enabling grid-aware regularization: the returned
        operator is ``sqrt(W) @ L`` with lethargy quadrature weights, so
        that ``||L phi||^2`` approximates the Sobolev integral over
        ``dlnE`` (see :func:`build_smoothness_penalty`). None (default)
        keeps the legacy bin-index differences.

    Returns
    -------
    Optional[np.ndarray]
        Operator ``L`` as a dense ndarray, or ``None`` for order 0 when
        ``identity_for_zero`` is ``False``.
    """
    if smoothness_order == 0:
        return np.eye(n) if identity_for_zero else None
    if smoothness_order not in (1, 2):
        raise ValueError(
            f"Unsupported smoothness_order: {smoothness_order}. Use 0, 1 or 2."
        )
    if E_MeV is not None:
        return _grid_weighted_operator(n, smoothness_order, E_MeV).toarray()
    return create_derivative_matrix(n, smoothness_order).toarray()


def build_tikhonov_system(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    L: np.ndarray,
) -> np.ndarray:
    """Build and solve a Tikhonov-regularized system: (A^T A + alpha * L^T L) x = A^T b.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    alpha : float
        Regularization parameter.
    L : np.ndarray
        Regularization matrix (e.g., identity or derivative matrix).

    Returns
    -------
    np.ndarray
        Solution vector x (n,), or None if solving fails.
    """
    try:
        P = A.T @ A + alpha * (L.T @ L)
        x = np.linalg.solve(P, A.T @ b)
        return np.maximum(x, 0)
    except np.linalg.LinAlgError:
        return None


def compute_svd_components(
    A: np.ndarray,
) -> tuple:
    """Compute SVD of A and return components needed for GCV and related computations.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).

    Returns
    -------
    tuple
        (U, s, Vt, s_sq) where s_sq = s**2 for reuse.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt, s**2


def compute_log_steps(E_MeV: np.ndarray, n_energy_bins: int) -> np.ndarray:
    """Compute logarithmic bin-width steps for an energy grid.

    Uses log10(energy + 1e-15) with edge differences at the boundaries and
    central differences for interior points, matching the convention used
    across the parametric unfolding modules.

    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    n_energy_bins : int
        Number of energy bins.

    Returns
    -------
    np.ndarray
        Log10 bin-width steps of length ``n_energy_bins``.
    """
    log_steps = np.zeros(n_energy_bins)
    log_e = np.log10(np.asarray(E_MeV, dtype=float) + 1e-15)

    if n_energy_bins > 1:
        log_steps[0] = log_e[1] - log_e[0]
        log_steps[-1] = log_e[-1] - log_e[-2]
    else:
        log_steps[0] = 1.0

    if n_energy_bins > 2:
        log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0

    return log_steps


def estimate_total_fluence(
    A: np.ndarray,
    b: np.ndarray,
    ln_steps: np.ndarray | None = None,
) -> float:
    """Estimate the total fluence of the unfolded spectrum.

    The estimate is obtained from an unconstrained non-negative least-squares
    fit ``min ||A x - b||^2, x >= 0``. The spectrum ``x`` is a differential
    fluence density per unit ``d(ln E)`` (matching the response matrix built
    by ``Detector._convert_rf_to_matrix_variable_step``, which pre-multiplies
    the response by the per-bin lethargy widths), so the physical total
    fluence is ``sum(x_i * dlnE_i)``:

    - with ``ln_steps`` given: returns ``sum(x_nnls * ln_steps)`` — the
      integrated fluence rate in the units of the readings
      (e.g. cm^-2 s^-1);
    - with ``ln_steps=None`` (default, backward compatible): returns
      ``sum(x_nnls)`` — the lethargy integral, which equals the total
      fluence only for a uniform dlnE grid with unit natural-log spacing.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n) including the dlnE weights.
    b : np.ndarray
        Measurement vector (m,).
    ln_steps : np.ndarray, optional
        Per-bin natural-logarithmic widths ``d(ln E)_i``. When given, the
        returned value is the true integrated fluence ``sum(x_i * dlnE_i)``.

    Returns
    -------
    float
        Positive estimate of the total fluence (or of the lethargy integral
        when ``ln_steps`` is None).
    """
    from scipy.optimize import nnls

    m, n = A.shape
    try:
        x_nnls, _ = nnls(A, b, maxiter=10 * n)
        if ln_steps is not None:
            total = float(np.sum(x_nnls * np.asarray(ln_steps, dtype=float)))
        else:
            total = float(x_nnls.sum())
    except Exception:  # nnls failure must not break fluence estimation
        total = 0.0
    if np.isfinite(total) and total > 0.0:
        return total
    # fallback: uniform-response heuristic
    mean_response = max(float(np.mean(A)), 1e-30)
    return float(np.mean(b) / mean_response * n)
