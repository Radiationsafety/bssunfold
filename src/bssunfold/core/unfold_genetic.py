"""Meta-heuristic (evolutionary) unfolding methods via MEALPY.

This module implements neutron spectrum unfolding using population-based
meta-heuristic algorithms from the ``mealpy`` package
(https://pypi.org/project/mealpy/). The approach is inspired by several
published genetic / evolutionary unfolding works:

- Shahabinejad & Sohrabpour, Rad. Phys. Chem. 136 (2017): Particle Swarm
  Optimization (SDPSO) with chaotic inertia weight and the cost
  ``||b - A x||^2 / ||b||^2 + lambda * ||x||^2``.
- Suman & Sarkar, BARC/2013/E/005 and Indian J. Pure Appl. Phys. 50 (2012):
  a genetic algorithm with second-difference smoothing
  ``sum((x_{j-1} - 2 x_j + x_{j+1})^2)``.
- Woo et al., Prog. Nucl. Sci. Technol. 6 (2019): a multi-objective
  formulation that also maximises the Shannon information entropy.
- Mukherjee, Radiat. Prot. Dosim. 110 (2004): ANDI-03, a GA tool for
  activation-detector data that requires no prior guess spectrum.

The ``unfold_genetic`` / ``solve_genetic`` pair exposes a ``solver``
selector that maps to different meta-heuristic algorithms implemented in
MEALPY (PSO, GA, DE, ES, EP, ABC, GWO, CMA-ES).
"""

import warnings

import numpy as np
from typing import Dict, Optional, Any, List, Callable

from ._matrix_utils import create_derivative_matrix
from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = ["solve_genetic", "unfold_genetic"]

_IMPORT_ERROR_MSG = (
    "mealpy is required for unfold_genetic. "
    "Install with: pip install mealpy"
)

# Supported meta-heuristic solvers (registry keys).
_SUPPORTED_SOLVERS = ("pso", "ga", "de", "es", "ep", "abc", "gwo", "cmaes")

# Long-form aliases for the solver names.
_SOLVER_ALIASES = {
    "particle_swarm": "pso",
    "genetic": "ga",
    "genetic_algorithm": "ga",
    "differential_evolution": "de",
    "evolution_strategy": "es",
    "evolutionary_programming": "ep",
    "bee_colony": "abc",
    "grey_wolf": "gwo",
    "gray_wolf": "gwo",
    "cma_es": "cmaes",
}


def _import_mealpy():
    """Import the MEALPY modules, raising a helpful ImportError if missing."""
    try:
        from mealpy import FloatVar, PSO, GA, DE, ES, EP, ABC, GWO
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return FloatVar, PSO, GA, DE, ES, EP, ABC, GWO


def _normalize_solver(solver: str) -> str:
    """Resolve a solver name (or alias) to a registry key, warning on unknown."""
    name = (solver or "").strip().lower()
    if name in _SOLVER_ALIASES:
        name = _SOLVER_ALIASES[name]
    if name not in _SUPPORTED_SOLVERS:
        warnings.warn(
            f"Solver '{solver}' not supported. "
            f"Available solvers: {_SUPPORTED_SOLVERS}. Using 'pso'."
        )
        name = "pso"
    return name


def _build_bounds(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray]):
    """Build non-negativity upper bounds from the response matrix and readings."""
    n = A.shape[1]
    x0_arr = np.zeros(n)
    if x0 is not None:
        x0_arr = np.maximum(np.asarray(x0, dtype=float), 0)
    col_norm = float(np.max(np.linalg.norm(A, axis=0))) or 1.0
    scale = float(np.max(np.abs(b))) / max(col_norm, 1e-12)
    ub = np.maximum(2.0 * np.abs(x0_arr), scale)
    ub = np.maximum(ub, 1e-3)
    return x0_arr, ub


def _build_fitness(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int,
    L,
    smoothness_weight: float,
    entropy_weight: float,
) -> Callable[[np.ndarray], float]:
    """Build the unfolding objective function.

    ``f(x) = ||b - A x||^2 / ||b||^2 + alpha * ||x||_norm
            + alpha * smoothness_weight * ||L x||^2
            - entropy_weight * H(x)``

    where ``H`` is the Shannon information entropy of the (normalised)
    spectrum. Positive weights only; a zero/negative weight disables the
    corresponding term.
    """
    denom = float(np.dot(b, b))
    if denom <= 0.0:
        denom = 1.0

    def fitness(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 0.0)
        residual = A @ x - b
        value = float(np.dot(residual, residual)) / denom
        if alpha > 0:
            if norm == 2:
                value += alpha * float(np.dot(x, x))
            elif norm == 1:
                value += alpha * float(np.sum(np.abs(x)))
        if L is not None and alpha > 0 and smoothness_weight > 0:
            Lx = L @ x
            value += alpha * smoothness_weight * float(np.dot(Lx, Lx))
        if entropy_weight > 0:
            total = float(np.sum(x))
            if total > 0:
                p = x / total
                logp = np.log(np.maximum(p, 1e-300))
                value -= entropy_weight * float(np.dot(p, logp))
        return value

    return fitness


def _make_starting_solutions(
    x0: np.ndarray, ub: np.ndarray, pop_size: int
) -> Optional[np.ndarray]:
    """Build a starting-solutions matrix with ``x0`` as the first individual.

    Returns None when ``x0`` is effectively zero so that MEALPY performs a
    pure random initialisation (the default, matching the published works
    that need no initial guess).
    """
    if np.all(np.asarray(x0, dtype=float) == 0.0):
        return None
    n = len(x0)
    rng = np.random.default_rng(0)
    starting = rng.uniform(0.0, ub, size=(pop_size, n))
    starting[0] = np.asarray(x0, dtype=float)
    return starting


def _build_model(mealpy, solver: str, epoch: int, pop_size: int):
    """Instantiate the MEALPY optimizer for the given solver key."""
    FloatVar, PSO, GA, DE, ES, EP, ABC, GWO = mealpy
    if solver == "pso":
        # Chaotic PSO with SDPSO-style inertia weight range.
        return PSO.C_PSO(
            epoch=epoch, pop_size=pop_size, c1=2.0, c2=2.0,
            w_min=0.4, w_max=0.9,
        )
    if solver == "ga":
        # Suman & Sarkar-style GA (pc=0.7, pm=0.3).
        return GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.7, pm=0.3)
    if solver == "de":
        return DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.7, cr=0.9)
    if solver == "es":
        return ES.OriginalES(epoch=epoch, pop_size=pop_size, lamda=0.75)
    if solver == "ep":
        return EP.OriginalEP(epoch=epoch, pop_size=pop_size, bout_size=0.05)
    if solver == "abc":
        return ABC.OriginalABC(epoch=epoch, pop_size=pop_size)
    if solver == "gwo":
        return GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
    if solver == "cmaes":
        return ES.CMA_ES(epoch=epoch, pop_size=pop_size)
    raise ValueError(f"Unsupported solver: {solver}")


def solve_genetic(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    solver: str = "pso",
    epoch: int = 500,
    pop_size: int = 50,
    regularization: float = 1e-4,
    norm: int = 2,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    n_runs: int = 1,
    early_stop: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Solve the unfolding problem using a meta-heuristic optimizer.

    Minimises the relative-residual objective described in the module
    docstring subject to ``0 <= x`` using MEALPY population-based solvers.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None (or all zeros), the population is
        initialised randomly (no prior spectrum required).
    solver : str, optional
        Meta-heuristic algorithm: 'pso', 'ga', 'de', 'es', 'ep', 'abc',
        'gwo' or 'cmaes' (default: 'pso').
    epoch : int, optional
        Maximum number of generations/iterations (default: 500).
    pop_size : int, optional
        Population size (default: 50).
    regularization : float, optional
        Tikhonov regularisation weight alpha (default: 1e-4).
    norm : int, optional
        Norm for the regularisation term (1 for L1, 2 for L2), default: 2.
    smoothness_order : int, optional
        Second-difference smoothing order (0, 1 or 2), default: 0.
    smoothness_weight : float, optional
        Weight of the smoothing term relative to the regularisation
        (default: 1.0).
    entropy_weight : float, optional
        Weight of the negative Shannon-entropy objective (0 disables it).
    n_runs : int, optional
        Number of independent optimisation runs; results are averaged
        (default: 1).
    early_stop : int, optional
        Stop if the global best does not improve for this many consecutive
        epochs (MEALPY early stopping).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, MEALPY logs the optimisation progress to the console.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,). Returns a zero vector if solving failed.
    """
    try:
        return _solve_genetic_impl(
            A=A, b=b, x0=x0, solver=solver, epoch=epoch, pop_size=pop_size,
            regularization=regularization, norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            entropy_weight=entropy_weight, n_runs=n_runs,
            early_stop=early_stop, random_state=random_state,
            verbose=verbose,
        )
    except ImportError:
        raise
    except Exception as exc:
        warnings.warn(
            f"Genetic solver '{solver}' failed: {exc}. Returning zero vector."
        )
        n = np.asarray(A, dtype=float).shape[1]
        return np.zeros(n)


def _solve_genetic_impl(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray],
    solver: str,
    epoch: int,
    pop_size: int,
    regularization: float,
    norm: int,
    smoothness_order: int,
    smoothness_weight: float,
    entropy_weight: float,
    n_runs: int,
    early_stop: Optional[int],
    random_state: Optional[int],
    verbose: bool,
) -> np.ndarray:
    mealpy = _import_mealpy()
    FloatVar, PSO, GA, DE, ES, EP, ABC, GWO = mealpy
    solver = _normalize_solver(solver)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    if norm not in (1, 2):
        raise ValueError(f"Unsupported norm type: {norm}")
    if smoothness_order not in (0, 1, 2):
        raise ValueError(
            f"Unsupported smoothness order: {smoothness_order}"
        )

    L = None
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)

    fitness = _build_fitness(
        A, b, regularization, norm, L, smoothness_weight, entropy_weight
    )
    x0_arr, ub = _build_bounds(A, b, x0)

    problem = {
        "obj_func": fitness,
        "bounds": FloatVar(lb=np.zeros(n), ub=ub),
        "minmax": "min",
        "log_to": "console" if verbose else None,
    }

    termination = None
    if early_stop is not None:
        termination = {"max_early_stop": int(early_stop)}

    starting = _make_starting_solutions(x0_arr, ub, pop_size)
    runs = max(1, int(n_runs))
    spectra = []
    for run in range(runs):
        model = _build_model(mealpy, solver, int(epoch), int(pop_size))
        seed = None
        if random_state is not None:
            seed = int(random_state) + run
        g_best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g_best = model.solve(
                problem,
                seed=seed,
                termination=termination,
                starting_solutions=starting,
            )
        spectra.append(np.asarray(g_best.solution, dtype=float))

    spectrum = np.mean(spectra, axis=0) if runs > 1 else spectra[0]
    return np.maximum(spectrum, 0)


def unfold_genetic(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    solver: str = "pso",
    epoch: int = 500,
    pop_size: int = 50,
    regularization: float = 1e-4,
    norm: int = 2,
    smoothness_order: int = 0,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    n_runs: int = 1,
    early_stop: Optional[int] = None,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum using a meta-heuristic algorithm.

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
        Initial spectrum guess. If None, no prior is used.
    solver : str, optional
        Meta-heuristic algorithm: 'pso', 'ga', 'de', 'es', 'ep', 'abc',
        'gwo' or 'cmaes' (default: 'pso').
    epoch : int, optional
        Maximum number of generations (default: 500).
    pop_size : int, optional
        Population size (default: 50).
    regularization : float, optional
        Tikhonov regularisation weight (default: 1e-4).
    norm : int, optional
        Norm for the regularisation term (1 or 2), default: 2.
    smoothness_order : int, optional
        Second-difference smoothing order (0, 1 or 2), default: 0.
    smoothness_weight : float, optional
        Weight of the smoothing term (default: 1.0).
    entropy_weight : float, optional
        Weight of the negative Shannon-entropy objective (default: 0).
    n_runs : int, optional
        Number of independent runs to average (default: 1).
    early_stop : int, optional
        Early-stopping patience (epochs without improvement).
    calculate_errors : bool, optional
        If True, calculate Monte-Carlo uncertainty, default: False.
    noise_level : float, optional
        Noise level for Monte-Carlo, default: 0.01.
    n_montecarlo : int, optional
        Number of Monte-Carlo samples, default: 100.
    save_result : bool, optional
        Save result to history, default: False.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, print the MEALPY optimisation progress.

    Returns
    -------
    Dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
    """
    x0_default = np.zeros(n_energy_bins)
    solver = _normalize_solver(solver)

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
            solve_genetic,
            solver=solver,
            epoch=epoch,
            pop_size=pop_size,
            regularization=regularization,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            entropy_weight=entropy_weight,
            n_runs=n_runs,
            early_stop=early_stop,
            random_state=random_state,
            verbose=verbose,
        ),
        solve_kwargs={},
        method_name=f"genetic_{solver}",
        extra_output={
            "solver": solver,
            "epoch": epoch,
            "pop_size": pop_size,
            "regularization": regularization,
            "norm": norm,
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "entropy_weight": entropy_weight,
            "n_runs": n_runs,
            "early_stop": early_stop,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
