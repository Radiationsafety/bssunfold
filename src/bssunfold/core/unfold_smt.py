"""SMT-based unfolding method.

This module ports the linear equation solver from the Haskell/SBV library
``linearEqSolver`` (https://github.com/LeventErkok/linearEqSolver) to a
Python implementation backed by the ``z3-solver`` package. It provides exact
solvers for systems of linear equations over the integers and the rationals,
both single and multiple solution variants, plus the ``solve_smt`` /
``unfold_smt`` pair for the Detector class.
"""

import fractions
import warnings

import numpy as np
from typing import Dict, Optional, Any, List

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = [
    "solve_integer_linear_eqs",
    "solve_integer_linear_eqs_all",
    "solve_rational_linear_eqs",
    "solve_rational_linear_eqs_all",
    "solve_smt",
    "unfold_smt",
]

_IMPORT_ERROR_MSG = (
    "z3-solver is required for SMT-based unfolding. "
    "Install with: pip install z3-solver"
)


def _import_z3():
    """Import and return the z3 module, raising a helpful ImportError."""
    try:
        import z3
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return z3


def _to_real_value(v, z3):
    """Convert a number to an exact rational z3 expression."""
    fr = fractions.Fraction(v).limit_denominator()
    return z3.RealVal(fr)


def _build_constraints(xs, coeffs, res, z3, to_value, nonneg=False):
    """Build the equation constraints ``A x = b``.

    Port of ``buildConstraints`` from the linearEqSolver library. ``to_value``
    maps a numeric coefficient to the corresponding z3 expression.
    """
    coeffs = [list(row) for row in coeffs]
    res = list(res)
    m = len(coeffs)
    if m == 0 or any(len(row) != len(xs) for row in coeffs) or m != len(res):
        raise ValueError("SMT solver: received ill-formed input.")
    constraints = []
    for row, r in zip(coeffs, res):
        lhs = z3.Sum([xs[j] * to_value(row[j], z3) for j in range(len(xs))])
        constraints.append(lhs == to_value(r, z3))
    if nonneg:
        constraints.extend([x >= 0 for x in xs])
    return constraints


def _all_solutions(solver, xs, z3, max_solutions, to_python):
    """Enumerate up to ``max_solutions`` models of ``solver``."""
    solutions = []
    while len(solutions) < max_solutions:
        if solver.check() != z3.sat:
            break
        model = solver.model()
        solutions.append([to_python(model[x]) for x in xs])
        solver.add(z3.Or([xs[j] != model[xs[j]] for j in range(len(xs))]))
    return solutions


def _validate_system(A, b):
    """Validate the coefficient matrix and result vector."""
    coeffs = np.asarray(A, dtype=float)
    res = np.asarray(b, dtype=float)
    if coeffs.ndim != 2 or res.ndim != 1 or coeffs.shape[0] != res.shape[0]:
        raise ValueError("SMT solver: received ill-formed input.")
    return coeffs, res


def solve_integer_linear_eqs(
    A: np.ndarray, b: np.ndarray
) -> Optional[List[int]]:
    """Solve a system of linear equations over the integers.

    Port of ``solveIntegerLinearEqs`` from the linearEqSolver library.
    The coefficients and the result vector are expected to be integers
    (or integral floats).

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of size (m, n).
    b : np.ndarray
        Result vector of size (m,).

    Returns
    -------
    Optional[List[int]]
        A solution ``x`` to ``A x = b`` over the integers, or None if no
        solution exists.
    """
    z3 = _import_z3()
    coeffs, res = _validate_system(A, b)
    n = coeffs.shape[1]
    xs = [z3.Int(f"x{i}") for i in range(n)]

    def to_int_value(v, _z3):
        return _z3.IntVal(int(v))

    solver = z3.Solver()
    solver.add(_build_constraints(xs, coeffs, res, z3, to_int_value))
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    return [model[x].as_long() for x in xs]


def solve_integer_linear_eqs_all(
    A: np.ndarray, b: np.ndarray, max_solutions: int = 10
) -> List[List[int]]:
    """Solve a system of linear equations over the integers, all solutions.

    Port of ``solveIntegerLinearEqsAll`` from the linearEqSolver library.
    If the system is underspecified it has infinitely many solutions; this
    returns up to ``max_solutions`` of them.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of size (m, n).
    b : np.ndarray
        Result vector of size (m,).
    max_solutions : int, optional
        Maximum number of solutions to return (default: 10).

    Returns
    -------
    List[List[int]]
        Distinct solutions to ``A x = b`` over the integers.
    """
    z3 = _import_z3()
    coeffs, res = _validate_system(A, b)
    n = coeffs.shape[1]
    xs = [z3.Int(f"x{i}") for i in range(n)]

    def to_int_value(v, _z3):
        return _z3.IntVal(int(v))

    solver = z3.Solver()
    solver.add(_build_constraints(xs, coeffs, res, z3, to_int_value))
    return _all_solutions(
        solver, xs, z3, int(max_solutions), lambda v: v.as_long()
    )


def solve_rational_linear_eqs(
    A: np.ndarray, b: np.ndarray
) -> Optional[List[float]]:
    """Solve a system of linear equations over the rationals.

    Port of ``solveRationalLinearEqs`` from the linearEqSolver library.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of size (m, n).
    b : np.ndarray
        Result vector of size (m,).

    Returns
    -------
    Optional[List[float]]
        A solution ``x`` to ``A x = b`` over the rationals, or None if no
        solution exists.
    """
    z3 = _import_z3()
    coeffs, res = _validate_system(A, b)
    n = coeffs.shape[1]
    xs = [z3.Real(f"x{i}") for i in range(n)]

    solver = z3.Solver()
    solver.add(_build_constraints(xs, coeffs, res, z3, _to_real_value))
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    return [float(model[x].as_fraction()) for x in xs]


def solve_rational_linear_eqs_all(
    A: np.ndarray, b: np.ndarray, max_solutions: int = 10
) -> List[List[float]]:
    """Solve a system of linear equations over the rationals, all solutions.

    Port of ``solveRationalLinearEqsAll`` from the linearEqSolver library.
    If the system is underspecified it has infinitely many solutions; this
    returns up to ``max_solutions`` of them.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of size (m, n).
    b : np.ndarray
        Result vector of size (m,).
    max_solutions : int, optional
        Maximum number of solutions to return (default: 10).

    Returns
    -------
    List[List[float]]
        Distinct solutions to ``A x = b`` over the rationals.
    """
    z3 = _import_z3()
    coeffs, res = _validate_system(A, b)
    n = coeffs.shape[1]
    xs = [z3.Real(f"x{i}") for i in range(n)]

    solver = z3.Solver()
    solver.add(_build_constraints(xs, coeffs, res, z3, _to_real_value))
    return _all_solutions(
        solver, xs, z3, int(max_solutions), lambda v: float(v.as_fraction())
    )


def _solve_smt_l1(
    A: np.ndarray,
    b: np.ndarray,
    nonneg: bool,
    timeout_ms: int,
    z3,
) -> np.ndarray:
    """Minimize the L1 residual ``||A x - b||_1`` then the total fluence.

    This is the original lexicographic objective of ``solve_smt``: the
    residual is bounded by non-negative slack variables and the slacks are
    minimized first, followed by ``sum(x)``.
    """
    n = A.shape[1]
    xs = [z3.Real(f"x{i}") for i in range(n)]
    slacks = [z3.Real(f"s{i}") for i in range(A.shape[0])]

    opt = z3.Optimize()
    opt.set("timeout", int(timeout_ms))
    for row, r, slack in zip(A, b, slacks):
        lhs = z3.Sum([xs[j] * _to_real_value(row[j], z3) for j in range(n)])
        rhs = _to_real_value(r, z3)
        opt.add(lhs - slack <= rhs)
        opt.add(lhs + slack >= rhs)
        opt.add(slack >= 0)
    if nonneg:
        for x in xs:
            opt.add(x >= 0)

    obj_residual = opt.minimize(z3.Sum(slacks))
    if opt.check() != z3.sat:
        warnings.warn(
            "SMT solver could not find a solution. Returning zero vector."
        )
        return np.zeros(n)

    opt.add(z3.Sum(slacks) == opt.lower(obj_residual))
    opt.minimize(z3.Sum(xs))
    if opt.check() != z3.sat:
        warnings.warn(
            "SMT solver could not refine the solution. Returning zero vector."
        )
        return np.zeros(n)

    model = opt.model()
    return np.array(
        [
            float(model.eval(x, model_completion=True).as_fraction())
            for x in xs
        ]
    )


def _solve_smt_l2(
    A: np.ndarray,
    b: np.ndarray,
    nonneg: bool,
    timeout_ms: int,
    z3,
) -> Optional[np.ndarray]:
    """Minimize the L2 residual ``||A x - b||_2`` via the KKT conditions.

    The non-negative least-squares optimum ``min ||A x - b||_2^2 s.t.
    x >= 0`` is characterized by linear constraints (a linear complementarity
    problem), so Z3 can solve it exactly without non-linear arithmetic:

    - stationarity: ``mu = 2 A^T (A x - b)``
    - dual feasibility: ``mu >= 0``
    - complementarity: ``x_i * mu_i = 0`` via the disjunction
      ``x_i == 0 OR mu_i == 0``.

    For ``nonneg=False`` only the stationarity ``mu == 0`` (normal equations)
    is imposed. The unconstrained least-squares residual is then returned.

    Among the L2-optimal solutions the total fluence ``sum(x)`` is minimized.
    Returns ``None`` on ``unknown``/``unsat``/error so the caller can fall
    back to the L1 objective.
    """
    try:
        n = A.shape[1]
        gram = 2.0 * (A.T @ A)
        rhs = 2.0 * (A.T @ b)

        xs = [z3.Real(f"x{i}") for i in range(n)]
        mu = [
            z3.Sum(
                [xs[j] * _to_real_value(gram[i, j], z3) for j in range(n)]
            )
            - _to_real_value(rhs[i], z3)
            for i in range(n)
        ]

        opt = z3.Optimize()
        opt.set("timeout", int(timeout_ms))
        if nonneg:
            for i in range(n):
                opt.add(xs[i] >= 0)
                opt.add(mu[i] >= 0)
                opt.add(z3.Or(xs[i] == 0, mu[i] == 0))
        else:
            for i in range(n):
                opt.add(mu[i] == 0)

        opt.minimize(z3.Sum(xs))
        if opt.check() != z3.sat:
            return None

        model = opt.model()
        return np.array(
            [
                float(model.eval(x, model_completion=True).as_fraction())
                for x in xs
            ]
        )
    except Exception as exc:  # fall back to L1 on any solver failure
        warnings.warn(
            f"SMT L2 objective failed ({exc}); falling back to L1."
        )
        return None


def solve_smt(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    nonneg: bool = True,
    timeout_ms: int = 10000,
    random_state: Optional[int] = None,
    objective: str = "l2",
) -> np.ndarray:
    """Solve the unfolding problem with an SMT solver.

    Minimizes the L2 residual ``||A x - b||_2`` and then the total fluence
    ``sum(x)`` over the non-negative orthant using the Z3 optimizer (via the
    exact KKT characterization of the non-negative least-squares optimum).
    If the L2 solve does not converge within its (bounded) time budget,
    e.g. on large systems, the solver falls back to the L1 residual
    ``||A x - b||_1``.

    The system ``A x = b`` is usually underdetermined (fewer detectors than
    energy bins), so the lexicographic objective selects a deterministic
    solution.

    Parameters
    ----------
    A : np.ndarray
        Response matrix of size (m, n).
    b : np.ndarray
        Measurement vector of size (m,).
    x0 : np.ndarray, optional
        Not used (provided for API compatibility). Z3 has no warm start.
    nonneg : bool, optional
        Constrain the solution to ``x >= 0`` (default: True).
    timeout_ms : int, optional
        SMT solver timeout in milliseconds (default: 10000).
    random_state : int, optional
        Random seed for the SMT solver, for reproducibility.
    objective : str, optional
        Residual objective: ``'l2'`` (default, least squares) or ``'l1'``.
        On a non-converging L2 solve the L1 objective is used as a fallback.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,). Returns a zero vector if solving failed.
    """
    z3 = _import_z3()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("SMT solver: received ill-formed input.")
    n = A.shape[1]

    if objective not in ("l1", "l2"):
        warnings.warn(
            f"SMT: unknown objective {objective!r}; using 'l2'."
        )
        objective = "l2"

    try:
        if random_state is not None:
            z3.set_param("smt.random_seed", int(random_state))
        if objective == "l2":
            l2_timeout_ms = max(1, min(timeout_ms // 2, 2000))
            solution = _solve_smt_l2(
                A, b, nonneg, l2_timeout_ms, z3
            )
            if solution is not None:
                return solution
        return _solve_smt_l1(A, b, nonneg, timeout_ms, z3)
    except Exception as exc:
        warnings.warn(f"SMT solver failed: {exc}. Returning zero vector.")
        return np.zeros(n)


def unfold_smt(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    nonneg: bool = True,
    timeout_ms: int = 10000,
    objective: str = "l2",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum using an SMT solver.

    Minimizes the L2 residual ``||A x - b||_2`` (via the exact KKT
    characterization of the least-squares optimum) and then the total
    fluence ``sum(x)``. Falls back to the L1 residual on non-converging
    solves.

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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess (accepted for API compatibility).
    nonneg : bool, optional
        Constrain the spectrum to be non-negative (default: True).
    timeout_ms : int, optional
        SMT solver timeout in milliseconds (default: 10000).
    objective : str, optional
        Residual objective: ``'l2'`` (default) or ``'l1'``.
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default: 100.
    save_result : bool, optional
        Save result to history, default: True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
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
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_smt,
            nonneg=nonneg,
            timeout_ms=timeout_ms,
            objective=objective,
        ),
        solve_kwargs={},
        method_name="SMT",
        extra_output={
            "nonneg": nonneg,
            "timeout_ms": timeout_ms,
            "objective": objective,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
