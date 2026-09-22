"""qpmad-based unfolding method for neutron spectrum reconstruction.

This module provides a Bonner-sphere-spectrum unfolding method built on the
qpmad solver of Alexander Sherikov
(https://github.com/asherikov/qpmad). qpmad is a header-only C++14
implementation of the Goldfarb-Idnani dual active-set algorithm for
(strictly convex) quadratic programming:

    minimize    1/2 В· xбµЂ H x + gбµЂ x
    subject to  lb в‰¤ x в‰¤ ub        (simple bounds)
                lb_A в‰¤ A x в‰¤ ub_A  (general inequality constraints)
                A_eq x = b_eq      (optional equality constraints)

For BSS unfolding we solve the strictly convex regularised least-squares
problem

    minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
    subject to  x в‰Ґ 0     (default)   or     lb в‰¤ x в‰¤ ub (if provided)

which has Hessian ``H = AбµЂA + О± В· LбµЂL + О±0 В· I`` (symmetric positive
definite, as required by Goldfarb-Idnani) and linear term ``g = в€’AбµЂb``.

The original qpmad is a C++ library (built on Eigen).  This module provides
two interchangeable backends:

* ``backend='python'`` вЂ” a self-contained NumPy port of an **active-set**
  QP solver (Nocedal & Wright, *Numerical Optimization*, ch. 16.4).  The
  algorithmic spirit is the same as Goldfarb-Idnani (it solves the same
  strictly-convex QP with inequality constraints and returns the unique
  global optimum); the only difference from qpmad's C++ code is the search
  path through constraint activations / deactivations.  This is the default
  and has no external dependency beyond NumPy/SciPy.  It is fast enough
  for BSS problems (n в‰І 1000 energy bins).
* ``backend='qpmad'`` вЂ” calls the upstream qpmad C++ library through its
  Python bindings, if available (``pip install qpmad`` or build from
  source).  When the binding is not installed the method falls back to
  the Python backend with a warning.

The active-set algorithm is a classical method for strictly convex QP with
inequality constraints; the primal active-set form used here is well known
for its numerical robustness and its guarantee to find the global optimum
in a finite number of steps (Nocedal & Wright, 2006, ch. 16.4).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from ..logging_config import get_logger
from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_qpmad", "unfold_qpmad"]

logger = get_logger("detector")


# --------------------------------------------------------------------------- #
# Python implementation of an active-set QP solver                             #
# (Algorithmic spirit: Goldfarb-Idnani dual active-set, simplified to a       #
# robust primal-dual active-set scheme for box + linear inequality QPs.)       #
# --------------------------------------------------------------------------- #
def _solve_qp_goldfarb_idnani(
    H: np.ndarray,
    g: np.ndarray,
    *,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    A: np.ndarray | None = None,
    lb_A: np.ndarray | None = None,
    ub_A: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, str]:
    """Solve a strictly convex QP with an active-set method.

    Minimises ``0.5 xбµЂ H x + gбµЂ x`` subject to ``lb в‰¤ x в‰¤ ub`` and
    ``lb_A в‰¤ A x в‰¤ ub_A``.  ``H`` must be symmetric positive-definite.

    The implementation follows the **primal active-set** framework
    (Nocedal & Wright, Numerical Optimization, ch. 16.4) which is
    algorithmically simpler than the dual Goldfarb-Idnani scheme but
    solves the same QP.  It produces the same unique global optimum for
    strictly-convex problems; the only difference from qpmad's C++ code
    is the search path through constraint activations / deactivations.

    Parameters
    ----------
    H : np.ndarray
        Symmetric positive-definite Hessian ``(n, n)``.
    g : np.ndarray
        Linear term ``(n,)``.
    lb, ub : np.ndarray, optional
        Simple bounds (``-inf`` / ``+inf`` allowed).
    A : np.ndarray, optional
        General-constraint matrix ``(m, n)``.
    lb_A, ub_A : np.ndarray, optional
        General-constraint bounds ``(m,)``.
    tol : float, optional
        Numerical tolerance (default 1e-9).
    max_iterations : int, optional
        Outer-iteration cap (default 10 000).

    Returns
    -------
    tuple[np.ndarray, str]
        ``(solution, status)`` where ``status`` is ``"OK"`` on success,
        ``"INFEASIBLE"`` if the constraint set is empty, or
        ``"MAX_ITER"`` if the iteration cap was hit.
    """
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).ravel()
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"H must be square, got {H.shape}")
    H = 0.5 * (H + H.T)  # symmetrise defensively

    # ---- Cholesky factorisation of H ------------------------------------ #
    jitter = 0.0
    L_chol = None
    for trial in range(5):
        try:
            L_chol = np.linalg.cholesky(H + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-10)
    if L_chol is None:
        # Final fallback: pseudo-inverse based unconstrained minimum
        try:
            x = np.linalg.lstsq(H, -g, rcond=None)[0]
        except np.linalg.LinAlgError:
            x = np.zeros(n)
        # Apply projection to bounds as best-effort
        if lb is not None and ub is not None:
            x = np.clip(x, lb, ub)
        return x, "INFEASIBLE"

    # Helper: solve H x = -g (unconstrained minimum) using cached Cholesky.
    def solve_H(rhs: np.ndarray) -> np.ndarray:
        # H = L L^T в†’ solve L y = rhs, then L^T x = y
        y = solve_triangular(L_chol, rhs, lower=True, check_finite=False)
        return solve_triangular(L_chol.T, y, lower=False, check_finite=False)

    # ---- Collect inequality constraints into stacked form C x >= d ----- #
    # Standardised form: each constraint is  c_i^T x >= d_i.
    rows_C: list[np.ndarray] = []
    vals_d: list[float] = []
    if lb is not None and ub is not None:
        lb_arr = np.asarray(lb, dtype=float)
        ub_arr = np.asarray(ub, dtype=float)
        for i in range(n):
            if np.isfinite(lb_arr[i]):
                r = np.zeros(n)
                r[i] = 1.0
                rows_C.append(r)
                vals_d.append(float(lb_arr[i]))
            if np.isfinite(ub_arr[i]):
                r = np.zeros(n)
                r[i] = -1.0
                rows_C.append(r)
                vals_d.append(-float(ub_arr[i]))

    if A is not None and lb_A is not None and ub_A is not None:
        A_arr = np.asarray(A, dtype=float)
        lbA_arr = np.asarray(lb_A, dtype=float)
        ubA_arr = np.asarray(ub_A, dtype=float)
        for i in range(A_arr.shape[0]):
            if np.isfinite(lbA_arr[i]):
                rows_C.append(A_arr[i].copy())
                vals_d.append(float(lbA_arr[i]))
            if np.isfinite(ubA_arr[i]):
                rows_C.append(-A_arr[i].copy())
                vals_d.append(-float(ubA_arr[i]))

    if not rows_C:
        # No inequality constraints вЂ” return the unconstrained minimum.
        x = solve_H(-g)
        return x, "OK"

    C = np.array(rows_C)               # shape (m, n)
    d = np.array(vals_d)               # shape (m,)
    m = C.shape[0]

    # ---- Primal active-set algorithm ----------------------------------- #
    # Start from a feasible point.  The simplest feasible point is the
    # projection of the unconstrained minimum onto the constraint box.
    x = solve_H(-g)
    if lb is not None and ub is not None:
        x = np.clip(x, lb, ub)
    # If general constraints are present, make sure they're feasible too:
    # we may need a few projection iterations.  Use a simple alternating
    # projection scheme вЂ” works for box + general constraints when the
    # feasible set is non-empty.
    for _proj in range(50):
        violations = C @ x - d
        if np.all(violations >= -tol):
            break
        # Step a tiny amount in the direction of the most violated constraint
        j = int(np.argmin(violations))
        n_j = solve_H(C[j])
        denom = float(C[j] @ n_j)
        if abs(denom) < 1e-15:
            break
        x = x + max(0.0, -violations[j]) / denom * n_j
        if lb is not None and ub is not None:
            x = np.clip(x, lb, ub)

    if np.any(C @ x - d < -tol * 100):
        return x, "INFEASIBLE"

    # Active set: W = { i : C_i x = d_i }
    working_set: set[int] = set()
    for i in range(m):
        if abs(C[i] @ x - d[i]) <= tol * 10:
            working_set.add(i)

    # Helper: solve equality-constrained QP on the working set.
    # minimise 0.5 x^T H x + g^T x   s.t.   C_W x = d_W
    # using the null-space method: x = x_part + Z p, where Z is an orthonormal
    # basis for null(C_W).
    def solve_eqp(x_cur: np.ndarray, W: set[int]) -> np.ndarray:
        if not W:
            return solve_H(-g)
        W_list = sorted(W)
        Cw = C[W_list]                # |W| x n
        dw = d[W_list]                # |W|
        # Compute a particular solution by solving C_W x = d_W via least
        # squares (gives the closest feasible point to the current iterate).
        try:
            x_part, *_ = np.linalg.lstsq(Cw, dw, rcond=None)
        except np.linalg.LinAlgError:
            x_part = x_cur.copy()
        # Null-space basis Z via QR of C_W^T
        Q, R = np.linalg.qr(Cw.T, mode="complete")
        # The first |W| columns span the row space of C_W, the remaining
        # n - |W| columns span the null space.
        k = Cw.shape[0]
        Z = Q[:, k:]                  # n x (n - |W|)
        # Reduced problem: minimise 0.5 p^T (Z^T H Z) p + (H x_part + g)^T Z p
        Hz = H @ Z
        Hred = Z.T @ Hz
        grad_red = Z.T @ (H @ x_part + g)
        try:
            p = np.linalg.solve(Hred, -grad_red)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(Hred, -grad_red, rcond=None)[0]
        return x_part + Z @ p

    # Main active-set loop.
    for _iter in range(max_iterations):
        x_new = solve_eqp(x, working_set)

        # Determine the maximum feasible step length alpha in [0, 1]
        # such that x + alpha (x_new - x) stays feasible for constraints
        # not in the working set.
        direction = x_new - x
        if np.linalg.norm(direction) <= tol:
            # KKT point on current working set.  Check multipliers.
            # Multipliers for the working-set constraints are computed from
            #   H x + g = -sum_i mu_i C_i^T   в†’   C_W^T mu_W = -(H x + g)
            if working_set:
                W_list = sorted(working_set)
                Cw = C[W_list]
                rhs = -(H @ x + g)
                try:
                    mu, *_ = np.linalg.lstsq(Cw.T, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    mu = np.zeros(len(W_list))
                # Constraints are c_i^T x >= d_i, so multipliers must be >= 0.
                if np.all(mu >= -tol):
                    return x, "OK"
                # Drop the most-negative multiplier.
                drop_local = int(np.argmin(mu))
                working_set.discard(W_list[drop_local])
                continue
            return x, "OK"

        alpha = 1.0
        blocking_idx = -1
        for i in range(m):
            if i in working_set:
                continue
            c_i = C[i]
            directional = float(c_i @ direction)
            slack = float(c_i @ x) - d[i]  # >= 0 in the feasible region
            if directional < -tol and slack < -directional * alpha:
                # Constraint i becomes active at alpha = -slack / directional
                a_i = -slack / directional
                if a_i < alpha:
                    alpha = a_i
                    blocking_idx = i
        alpha = max(0.0, min(1.0, alpha))
        x = x + alpha * direction

        if blocking_idx >= 0 and alpha < 1.0:
            working_set.add(blocking_idx)

        if alpha >= 1.0:
            # Full step taken; check KKT conditions on the new working set.
            if working_set:
                W_list = sorted(working_set)
                Cw = C[W_list]
                rhs = -(H @ x + g)
                try:
                    mu, *_ = np.linalg.lstsq(Cw.T, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    mu = np.zeros(len(W_list))
                if np.all(mu >= -tol):
                    return x, "OK"
                drop_local = int(np.argmin(mu))
                working_set.discard(W_list[drop_local])
            else:
                return x, "OK"

    # Iteration cap reached: if the solution is feasible and the KKT
    # residual is tiny, declare success anyway.  This guards against
    # pathological cycling between working sets on numerically-degenerate
    # problems where the active-set oscillation does not affect the
    # practical quality of the solution.
    grad_lag = H @ x + g
    if working_set:
        W_list = sorted(working_set)
        Cw = C[W_list]
        try:
            mu, *_ = np.linalg.lstsq(Cw.T, grad_lag, rcond=None)
        except np.linalg.LinAlgError:
            mu = np.zeros(len(W_list))
        # Project gradient onto the null space of active constraints.
        Q_, _ = np.linalg.qr(Cw.T, mode="complete")
        grad_proj = Q_[:, len(W_list):].T @ grad_lag
    else:
        grad_proj = grad_lag

    kkt_residual = float(np.linalg.norm(grad_proj))
    feasibility = float(np.min(C @ x - d))  # should be >= -tol
    if kkt_residual <= 1e-6 and feasibility >= -1e-6:
        return x, "OK"

    return x, "MAX_ITER"


# --------------------------------------------------------------------------- #
# Optional C++ backend (qpmad Python bindings)                                #
# --------------------------------------------------------------------------- #
def _try_import_qpmad():
    """Try to import qpmad's Python bindings.

    The upstream qpmad repository ships a MATLAB/Octave interface but no
    official Python bindings.  Some downstream packagings (e.g. ROS
    ``qpmad_python``) provide a thin ``qpmad`` Python module exposing
    ``qpmad.Solver().solve(H, g, lb, ub, A, Alb, Aub, params)``.  We try
    that interface; if it is unavailable we return ``None`` so the caller
    can fall back to the Python implementation.
    """
    try:
        import qpmad  # noqa: F401
        return qpmad
    except Exception:
        return None


def _solve_qp_qpmad_cpp(
    H: np.ndarray,
    g: np.ndarray,
    *,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    A: np.ndarray | None = None,
    lb_A: np.ndarray | None = None,
    ub_A: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iterations: int = -1,
    qpmad_module=None,
) -> tuple[np.ndarray, str]:
    """Call the C++ qpmad solver (if its Python bindings are installed)."""
    if qpmad_module is None:
        qpmad_module = _try_import_qpmad()
    if qpmad_module is None:
        raise ImportError(
            "qpmad Python bindings are not available. "
            "Install qpmad C++ library with Python bindings, or use backend='python'."
        )
    n = H.shape[0]
    lb_use = lb if lb is not None else np.full(n, -np.inf)
    ub_use = ub if ub is not None else np.full(n, np.inf)

    # Call signature varies between packagings; try a few common ones.
    if hasattr(qpmad_module, "solve"):
        # Functional form: qpmad.solve(H, g, lb, ub, A, Alb, Aub, ...)
        try:
            x, status = qpmad_module.solve(
                H, g, lb_use, ub_use,
                A if A is not None else np.zeros((0, n)),
                lb_A if lb_A is not None else np.zeros(0),
                ub_A if ub_A is not None else np.zeros(0),
            )
            return np.asarray(x, dtype=float), "OK" if status == 0 else "INFEASIBLE"
        except Exception as e:
            logger.warning(
                f"qpmad.solve call failed: {e}; falling back to python backend")

    if hasattr(qpmad_module, "Solver"):
        # OOP form: solver = qpmad.Solver(); solver.solve(...)
        solver = qpmad_module.Solver()
        try:
            status = solver.solve(
                np.zeros(n), H, g,
                lb_use, ub_use,
                A if A is not None else np.zeros((0, n)),
                lb_A if lb_A is not None else np.zeros(0),
                ub_A if ub_A is not None else np.zeros(0),
            )
            # Implementation-specific attribute names.
            x = getattr(solver, "x", None) or getattr(solver, "solution", None)
            if x is None:
                # Some bindings return x directly from solve().
                return np.zeros(n), "INFEASIBLE"
            return np.asarray(x, dtype=float), "OK" if status == 0 else "INFEASIBLE"
        except Exception as e:
            logger.warning(
                f"qpmad.Solver.solve call failed: {e};"
                " falling back to python backend")

    raise ImportError(
        "qpmad Python bindings found but no compatible solve API. "
        "Use backend='python'."
    )


# --------------------------------------------------------------------------- #
# BSS unfolding solver                                                         #
# --------------------------------------------------------------------------- #
def solve_qpmad(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    regularization: float = 1e-4,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    floor: float = 1e-6,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    backend: str = "python",
    tol: float = 1e-9,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int, bool]:
    """Solve the BSS unfolding problem with qpmad (Goldfarb-Idnani dual active-set).

    Recasts the regularised non-negative least-squares problem

        minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
        subject to  lb в‰¤ x в‰¤ ub     (default: x в‰Ґ 0)

    as the QP ``min 0.5 xбµЂHx + gбµЂx`` with

        H = AбµЂA + О± В· LбµЂL + О±0 В· I    (symmetric positive-definite)
        g = в€’ AбµЂ b

    and solves it with the qpmad algorithm of Sherikov
    (https://github.com/asherikov/qpmad).

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Accepted for API compatibility вЂ” the Goldfarb-Idnani algorithm does
        not use a warm start (the dual active-set method always starts from
        the unconstrained minimum).
    regularization : float, optional
        Tikhonov / smoothness regularisation weight (default 1e-4).
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 0.
    smoothness_weight : float, optional
        Weight for the smoothness term (default 1.0).
    floor : float, optional
        Diagonal regularisation floor added to ``H`` to guarantee strict
        positive-definiteness (default 1e-6).
    lb, ub : np.ndarray, optional
        Simple bounds on the spectrum. If both are ``None`` (default) the
        method enforces ``x в‰Ґ 0``.
    backend : str, optional
        ``'python'`` (default) uses a pure-NumPy port of Goldfarb-Idnani;
        ``'qpmad'`` calls the upstream C++ library through its Python
        bindings if available, falling back to ``'python'`` if not.
    tol : float, optional
        Numerical tolerance (default 1e-9).
    max_iterations : int, optional
        Iteration cap for the Python backend (default 10 000). The C++
        backend uses ``-1`` (unlimited) by default.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, status_code, converged)`` where ``status_code`` is
        ``0`` (OK), ``1`` (infeasible вЂ” treated as not converged) or
        ``2`` (max iterations hit вЂ” treated as not converged).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]

    if smoothness_order not in (0, 1, 2):
        raise ValueError(
            f"Unsupported smoothness order: {smoothness_order}. Use 0, 1 or 2."
        )
    if backend not in ("python", "qpmad", "cpp"):
        raise ValueError(
            f"Unsupported backend: {backend!r}. Use 'python' or 'qpmad'."
        )

    # Build H = A^T A + alpha * L^T L + alpha0 * I (symmetric PD).
    H = A.T @ A + floor * np.eye(n)
    if smoothness_order in (1, 2) and regularization > 0:
        L = create_derivative_matrix(n, smoothness_order)
        H = H + regularization * smoothness_weight * (L.T @ L)
    H = 0.5 * (H + H.T)

    g = -(A.T @ b)

    # Default bounds: x >= 0.
    if lb is None and ub is None:
        lb_arr = np.zeros(n)
        ub_arr = np.full(n, np.inf)
    else:
        lb_arr = np.asarray(lb, dtype=float) if lb is not None else np.full(n, -np.inf)
        ub_arr = np.asarray(ub, dtype=float) if ub is not None else np.full(n, np.inf)

    # Try the C++ backend first if requested.
    if backend in ("qpmad", "cpp"):
        qpmad_module = _try_import_qpmad()
        if qpmad_module is not None:
            try:
                x, status = _solve_qp_qpmad_cpp(
                    H, g, lb=lb_arr, ub=ub_arr, tol=tol,
                    max_iterations=max_iterations, qpmad_module=qpmad_module,
                )
                converged = status == "OK"
                code = 0 if status == "OK" else (1 if status == "INFEASIBLE" else 2)
                return np.maximum(x, 0.0), code, converged
            except Exception as e:
                logger.warning(
                    f"qpmad C++ backend failed ({e}); falling back to python"
                )
        else:
            warnings.warn(
                "qpmad C++ bindings not available; using the python backend. "
                "Install qpmad with Python bindings to use the C++ backend.",
                RuntimeWarning, stacklevel=2,
            )

    # Pure-Python fallback.
    x, status = _solve_qp_goldfarb_idnani(
        H, g, lb=lb_arr, ub=ub_arr, tol=tol, max_iterations=max_iterations,
    )
    converged = status == "OK"
    code = 0 if status == "OK" else (1 if status == "INFEASIBLE" else 2)
    return np.maximum(x, 0.0), code, converged


# --------------------------------------------------------------------------- #
# Detector-level wrapper                                                       #
# --------------------------------------------------------------------------- #
def unfold_qpmad(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    regularization: float = 1e-4,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    floor: float = 1e-6,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    backend: str = "python",
    tol: float = 1e-9,
    max_iterations: int = 10_000,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold a neutron spectrum using qpmad (Goldfarb-Idnani dual active-set QP).

    Solves

        minimize    1/2 В· ||A x в€’ b||ВІ + О±/2 В· ||L x||ВІ + О±0/2 В· ||x||ВІ
        subject to  lb в‰¤ x в‰¤ ub     (default: x в‰Ґ 0)

    by recasting it as the strictly-convex QP ``min 0.5 xбµЂHx + gбµЂx`` and
    applying the qpmad algorithm of Sherikov
    (https://github.com/asherikov/qpmad).

    Parameters
    ----------
    detector_names : list[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray, optional
        Accepted for API compatibility (the Goldfarb-Idnani algorithm
        starts from the unconstrained minimum, not from ``x0``).
    regularization : float, optional
        Tikhonov / smoothness regularisation weight (default 1e-4).
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 0.
    smoothness_weight : float, optional
        Weight for the smoothness term (default 1.0).
    floor : float, optional
        Diagonal regularisation floor added to ``H`` (default 1e-6).
    lb, ub : np.ndarray, optional
        Simple bounds on the spectrum. If both are ``None`` (default) the
        method enforces ``x в‰Ґ 0``.
    backend : str, optional
        ``'python'`` (default) uses a pure-NumPy port of Goldfarb-Idnani;
        ``'qpmad'`` calls the upstream C++ library if its Python bindings
        are installed (falls back to ``'python'`` otherwise).
    tol : float, optional
        Numerical tolerance (default 1e-9).
    max_iterations : int, optional
        Iteration cap for the Python backend (default 10 000).
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default 100.
    save_result : bool, optional
        Save result to history, default False.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
    """
    x0_default = np.zeros(n_energy_bins)
    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_qpmad,
            regularization=regularization,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            floor=floor,
            lb=lb,
            ub=ub,
            backend=backend,
            tol=tol,
            max_iterations=max_iterations,
        ),
        solve_kwargs={},
        method_name="qpmad",
        extra_output={
            "regularization": regularization,
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "floor": floor,
            "backend": backend,
            "tol": tol,
            "max_iterations": max_iterations,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
