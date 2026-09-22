"""AMG- and stationary-preconditioned Krylov unfolding method.

Python analogue of the R packages ``Rlinsolve`` (stationary and Krylov
iterative solvers for sparse linear systems) and ``pyamg``-style
algebraic multigrid: the unfolding least-squares problem is solved as a
preconditioned Krylov iteration on the normal equations

    (A^T A) x = A^T b,

where the preconditioner approximates ``(A^T A)^-1`` and can be built
from:

* ``"amg"``    -- algebraic multigrid (smoothed aggregation, optional
  dependency ``pyamg``; falls back to Jacobi with a warning when not
  installed);
* ``"jacobi"`` -- diagonal (Jacobi) preconditioner, the same iteration
  as ``Rlinsolve::lsolve.jacobi``;
* ``"gs"``     -- one forward Gauss-Seidel sweep per application
  (``Rlinsolve::lsolve.gs``);
* ``"sor"``    -- successive over-relaxation sweep with relaxation
  factor ``omega`` (``Rlinsolve::lsolve.sor``);
* ``"ssor"``   -- symmetric SOR sweep (``Rlinsolve::lsolve.ssor``);
* ``"none"``   -- unpreconditioned Krylov iteration.

The Krylov solver itself is one of ``cg``, ``bicgstab`` or ``gmres``
(scipy.sparse.linalg).  Non-negativity of the fluence is enforced by
projected outer restarts: after each Krylov solve the spectrum is
clamped to ``x >= 0`` and the iteration is restarted on the residual of
the clamped iterate, ``outer_iterations`` times.  This both stabilises
the physical admissibility of the solution and acts as an accelerated
fixed-point refinement of the Krylov answer.

Module API (standard bssunfold solver conventions):

* ``build_preconditioner(A, kind, omega)`` -- returns a
  ``scipy.sparse.linalg.LinearOperator`` approximating
  ``(A^T A)^-1`` (and the availability flag for AMG);
* ``solve_amg(A, b, ...)`` -- core solver returning
  ``(spectrum, iterations, converged)``;
* ``unfold_amg(...)`` -- Detector-facing wrapper (also exposed as
  ``Detector.unfold_amg``).

Optional dependency
-------------------
``pyamg`` enables the ``"amg"`` preconditioner
(``pip install bssunfold[amg]``).  Without it the module still works:
``"amg"`` transparently degrades to the Jacobi preconditioner with a
warning, and the stationary preconditioners (``jacobi``/``gs``/``sor``/
``ssor``) only need scipy.
"""

import warnings
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags, tril, triu
from scipy.sparse.linalg import (
    LinearOperator,
    bicgstab,
    cg,
    gmres,
    spsolve_triangular,
)

from ..logging_config import get_logger
from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = [
    "solve_amg",
    "unfold_amg",
    "build_preconditioner",
    "AMG_AVAILABLE",
]

logger = get_logger("unfold_amg")

_VALID_METHODS = ("cg", "bicgstab", "gmres")
_VALID_PRECONDITIONERS = ("amg", "jacobi", "gs", "sor", "ssor", "none")

# Preconditioners whose application matrix is symmetric positive definite
# (compatible with CG); "gs" and "sor" are nonsymmetric and are only
# used with bicgstab/gmres.
_CG_COMPATIBLE = ("amg", "jacobi", "ssor", "none")

# Auto Tikhonov damping (fraction of the mean diagonal of A^T A) used
# when ``regularization=None``.  A tiny damping makes the normal matrix
# nonsingular, which stabilises AMG hierarchy construction and the
# stationary-splitting preconditioners on rank-deficient systems.
_AUTO_REG_FACTOR = 1e-4

try:  # optional dependency for the "amg" preconditioner
    import pyamg  # noqa: F401

    AMG_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via block_import
    pyamg = None
    AMG_AVAILABLE = False


def _stationary_apply(
    N: csr_matrix, kind: str, omega: float, r: np.ndarray
) -> np.ndarray:
    """Apply one stationary sweep of kind ``kind`` to system ``N``.

    Implements the preconditioned-residual recurrences of the classical
    splitting iterations (Jacobi / Gauss-Seidel / SOR / SSOR) used by
    ``Rlinsolve``: given the splitting ``N = D + L + U`` the application
    returns ``M^-1 r`` with

    * ``jacobi``: ``M = D``;
    * ``gs``: ``M = D + L`` (forward substitution);
    * ``sor``: ``M = (D + omega L) / omega``;
    * ``ssor``: ``M = (D + omega L) D^-1 (D + omega U) / (omega (2 - omega))``.
    """
    D = N.diagonal()
    D_safe = np.where(np.abs(D) > 0, D, 1.0)
    if kind == "jacobi":
        return r / D_safe

    m, n = N.shape
    lower = tril(N, k=-1).tocsr()
    upper = triu(N, k=1).tocsr()
    Dm = diags(D_safe)

    if kind == "gs":
        M = (Dm + lower).tocsr()
        return spsolve_triangular(M, r, lower=True)

    if kind == "sor":
        M = (Dm + omega * lower).tocsr()
        return omega * spsolve_triangular(M, r, lower=True)

    if kind == "ssor":
        Mf = (Dm + omega * lower).tocsr()
        Mb = (Dm + omega * upper).tocsr()
        t = spsolve_triangular(Mf, r, lower=True)
        t = D_safe * t
        t = spsolve_triangular(Mb, t, lower=False)
        return omega * (2.0 - omega) * t

    raise ValueError(f"Unknown stationary kind: {kind!r}")


def build_preconditioner(
    A: np.ndarray,
    kind: str = "amg",
    omega: float = 1.0,
    damping: float = 0.0,
) -> LinearOperator:
    """Build a preconditioner approximating ``(A^T A + damping I)^-1``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    kind : str, optional
        ``"amg"`` (default), ``"jacobi"``, ``"gs"``, ``"sor"``, ``"ssor"``
        or ``"none"``.
    omega : float, optional
        Relaxation factor for ``"sor"`` / ``"ssor"`` (default: 1.0).
    damping : float, optional
        Tikhonov damping added to the diagonal of the normal matrix
        before the preconditioner is built (default: 0.0).

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Operator with matvec ``M^-1 r`` acting on ``(n,)`` vectors.

    Notes
    -----
    When ``kind="amg"`` and ``pyamg`` is not installed, a Jacobi
    preconditioner is returned and a :exc:`RuntimeWarning` is emitted
    (graceful degradation, mirroring the numba fallback convention of
    this package).
    """
    A = np.asarray(A, dtype=float)
    if kind not in _VALID_PRECONDITIONERS:
        raise ValueError(
            f"preconditioner must be one of {_VALID_PRECONDITIONERS}, "
            f"got {kind!r}"
        )
    if kind == "none":
        n = A.shape[1]
        return LinearOperator((n, n), matvec=lambda r: np.asarray(r, float))

    N = csr_matrix(A.T @ A + float(damping) * np.eye(A.shape[1]))
    n = N.shape[0]

    if kind == "amg":
        if AMG_AVAILABLE:
            # Deterministic setup: pyamg draws from the global numpy RNG
            # during hierarchy construction (spectral radius estimates),
            # which would otherwise make the unfolding result run-dependent.
            # Deterministic tentative prolongator candidates (ones) are
            # used as well.
            rng_state = np.random.get_state()
            try:
                np.random.seed(0)
                B_candidates = np.ones((N.shape[0], 1))
                ml = pyamg.smoothed_aggregation_solver(N, B=B_candidates)
            finally:
                np.random.set_state(rng_state)
            return ml.aspreconditioner()
        warnings.warn(
            "pyamg is not installed -- AMG preconditioner falls back to "
            "Jacobi. Install with: pip install bssunfold[amg]",
            RuntimeWarning,
            stacklevel=2,
        )
        kind = "jacobi"

    return LinearOperator(
        (n, n),
        matvec=lambda r: _stationary_apply(N, kind, omega, np.asarray(r, float)),
    )


def solve_amg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    method: str = "cg",
    preconditioner: str = "amg",
    omega: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
    outer_iterations: int = 3,
    nonnegativity: bool = True,
    regularization: float | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve unfolding problem with a preconditioned Krylov method.

    The (optionally damped) normal equations ``(A^T A + reg I) x =
    A^T b`` are solved with the chosen Krylov solver and preconditioner;
    non-negativity is enforced with projected outer restarts (see
    module docstring).

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial guess ``(n,)``; zeros when ``None``.
    method : str, optional
        Krylov solver: ``"cg"`` (default), ``"bicgstab"`` or ``"gmres"``.
        ``"cg"`` requires a symmetric positive-definite preconditioner;
        the nonsymmetric ``"gs"`` / ``"sor"`` preconditioners are
        transparently replaced by ``"ssor"`` (with a warning) for CG.
    preconditioner : str, optional
        ``"amg"`` (default), ``"jacobi"``, ``"gs"``, ``"sor"``, ``"ssor"``
        or ``"none"``.
    omega : float, optional
        Relaxation factor for ``"sor"`` / ``"ssor"`` (default: 1.0).
    max_iterations : int, optional
        Maximum Krylov iterations per outer restart (default: 200).
    tolerance : float, optional
        Relative residual tolerance of the normal equations
        (default: 1e-10).
    outer_iterations : int, optional
        Number of projected restarts (default: 3).
    nonnegativity : bool, optional
        Clamp the spectrum to ``x >= 0`` between restarts (default:
        ``True``).
    regularization : float or None, optional
        Tikhonov damping added to the diagonal of ``A^T A``.  ``None``
        (default) selects ``1e-4 * mean(diag(A^T A))`` automatically,
        which keeps the damped system nonsingular and stabilises the
        AMG and stationary-splitting preconditioners on rank-deficient
        systems; pass ``0.0`` for the pure (undamped) normal equations.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        ``(spectrum, iterations, converged)`` with the estimated total
        Krylov iteration count across restarts.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS}, got {method!r}"
        )
    if preconditioner not in _VALID_PRECONDITIONERS:
        raise ValueError(
            f"preconditioner must be one of {_VALID_PRECONDITIONERS}, "
            f"got {preconditioner!r}"
        )
    if method == "cg" and preconditioner not in _CG_COMPATIBLE:
        warnings.warn(
            f"preconditioner={preconditioner!r} is nonsymmetric and "
            "incompatible with method='cg'; switching to 'ssor'",
            RuntimeWarning,
            stacklevel=2,
        )
        preconditioner = "ssor"
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    if outer_iterations < 1:
        raise ValueError(
            f"outer_iterations must be >= 1, got {outer_iterations}"
        )
    if not (0.0 < omega <= 2.0):
        warnings.warn(
            f"omega={omega} outside the recommended range (0, 2]",
            RuntimeWarning,
            stacklevel=2,
        )

    n = A.shape[1]
    AT_A = A.T @ A
    AT_b = A.T @ b
    if regularization is None:
        damping = _AUTO_REG_FACTOR * float(np.mean(np.diag(AT_A)))
    else:
        damping = float(regularization)
        if damping < 0:
            raise ValueError(
                f"regularization must be non-negative, got {regularization}"
            )
    AT_A_solver = AT_A + damping * np.eye(n)
    M_raw = build_preconditioner(A, kind=preconditioner, omega=omega,
                                 damping=damping)
    M, matvec_counter = _counting_operator(M_raw)
    # cg/gmres apply the preconditioner once per iteration, bicgstab twice
    applications_per_iteration = 2 if method == "bicgstab" else 1

    x = x0.copy() if x0 is not None else np.zeros(n)
    converged = False
    inner_converged = False

    runners = {
        "cg": cg,
        "bicgstab": bicgstab,
        "gmres": gmres,
    }
    runner = runners[method]
    b_norm = max(np.linalg.norm(AT_b), 1e-300)

    for _outer in range(outer_iterations):
        residual = AT_b - AT_A_solver @ x
        if np.linalg.norm(residual) <= tolerance * b_norm:
            converged = True
            break
        dx, info = runner(
            AT_A_solver,
            residual,
            x0=np.zeros(n),
            rtol=tolerance,
            maxiter=max_iterations,
            M=M,
        )
        if not np.all(np.isfinite(dx)):
            dx = np.zeros(n)
        x = x + dx
        if nonnegativity:
            x = np.maximum(x, 0.0)
        if info == 0:
            inner_converged = True
            # Full success requires the (possibly clamped) iterate to
            # satisfy the tolerance as well.
            if np.linalg.norm(AT_b - AT_A_solver @ x) <= tolerance * b_norm:
                converged = True
                break
        elif info < 0:
            # Krylov breakdown (illegal input / loss of orthogonality):
            # further restarts on the same system are unlikely to help.
            break

    # The projected restarts succeed when the Krylov solves themselves
    # converged (the clamped iterate is then the projected solution) or
    # when the clamped iterate satisfies the residual tolerance.
    converged = converged or inner_converged
    total_iterations = min(
        matvec_counter[0] // applications_per_iteration,
        outer_iterations * max_iterations,
    )
    if nonnegativity:
        x = np.maximum(x, 0.0)
    return x, total_iterations, converged


def _counting_operator(M: LinearOperator) -> tuple[LinearOperator, list[int]]:
    """Wrap a preconditioner with a matvec counter.

    Returns ``(M_counted, counter)`` where ``counter`` is a one-element
    list holding the number of matvec applications (used to estimate
    the Krylov iteration count, which scipy does not report).
    """
    counter = [0]

    def matvec(r):
        counter[0] += 1
        return M.matvec(r)

    M_counted = LinearOperator(M.shape, matvec=matvec, dtype=M.dtype)
    return M_counted, counter


def unfold_amg(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    method: str = "cg",
    preconditioner: str = "amg",
    omega: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
    outer_iterations: int = 3,
    nonnegativity: bool = True,
    regularization: float | None = None,
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
    """Unfold neutron spectrum with AMG/stationary-preconditioned Krylov.

    Python analogue of the ``Rlinsolve`` iterative-solver family and of
    algebraic-multigrid preconditioning: the normal equations of the
    least-squares unfolding problem are solved with ``cg``/``bicgstab``/
    ``gmres`` accelerated by an algebraic multigrid or classical
    stationary-iteration (Jacobi/Gauss-Seidel/SOR/SSOR) preconditioner,
    with projected non-negativity restarts.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial guess for the Krylov iteration.
    method : str, optional
        Krylov solver: ``"cg"`` (default), ``"bicgstab"`` or ``"gmres"``.
    preconditioner : str, optional
        ``"amg"`` (default), ``"jacobi"``, ``"gs"``, ``"sor"``, ``"ssor"``
        or ``"none"``.
    omega : float, optional
        Relaxation factor for SOR/SSOR preconditioning (default: 1.0).
    max_iterations : int, optional
        Maximum Krylov iterations per restart (default: 200).
    tolerance : float, optional
        Relative residual tolerance (default: 1e-10).
    outer_iterations : int, optional
        Number of projected non-negativity restarts (default: 3).
    nonnegativity : bool, optional
        Clamp between restarts (default: True).
    regularization : float or None, optional
        Tikhonov damping for the normal equations; ``None`` (default)
        selects ``1e-4 * mean(diag(A^T A))`` automatically, ``0.0``
        disables the damping.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result dictionary.
    """
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
        default_initial=np.zeros(n_energy_bins),
        solve_func=make_solve_wrapper(
            solve_amg,
            method=method,
            preconditioner=preconditioner,
            omega=omega,
            max_iterations=max_iterations,
            tolerance=tolerance,
            outer_iterations=outer_iterations,
            nonnegativity=nonnegativity,
            regularization=regularization,
        ),
        solve_kwargs={},
        method_name="AMG-Krylov",
        extra_output={
            "krylov_method": method,
            "preconditioner": preconditioner,
            "omega": omega,
            "outer_iterations": outer_iterations,
            "regularization": regularization,
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
