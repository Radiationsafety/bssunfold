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

Numerical strategy
------------------
The unfolding problem is severely ill-posed (many more energy bins than
detectors), so a naive population-based search over a wide linear range
converges to a noisy, arbitrary spectrum. To obtain results comparable to
deterministic methods (landweber, cvxpy, MLEM) the optimizer:

- searches in **log space** (``y = log(x)``) so the wide dynamic range of
  neutron spectra is handled naturally;
- is **seeded with a Landweber warm-start solution** (or a user-provided
  ``initial_spectrum``) so the population starts from a smooth,
  physically-plausible spectrum;
- is bounded to ``log(seed) +/- half_range`` decades;
- minimises a **scale-consistent objective** in which the residual,
  regularisation and second-difference smoothness terms are all
  dimensionless and comparable, preventing the optimizer from inflating
  ``x`` without penalty.
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


def _build_seed(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray]) -> np.ndarray:
    """Build the seed spectrum used to initialise the population.

    If a non-trivial initial guess ``x0`` is provided it is used directly.
    Otherwise a Landweber warm-start solution (from a zero initial guess) is
    computed, which gives the meta-heuristic a smooth, physically-plausible
    starting point. This is essential because the unfolding problem is
    severely ill-posed (many more energy bins than detectors) and a purely
    random population converges to a noisy, arbitrary spectrum.
    """
    n = A.shape[1]
    if x0 is not None and np.any(np.asarray(x0, dtype=float) > 0):
        return np.maximum(np.asarray(x0, dtype=float), 1e-12)
    try:
        from .unfold_landweber import solve_landweber
        lw, _, _ = solve_landweber(A, b, np.zeros(n), max_iterations=500)
        seed = np.maximum(np.asarray(lw, dtype=float), 1e-12)
    except Exception:
        # Fallback: flat spectrum scaled to match the readings.
        A_fro = float(np.linalg.norm(A)) or 1.0
        x_scale = float(np.linalg.norm(b)) / A_fro
        seed = np.full(n, max(x_scale / np.sqrt(n), 1e-12))
    return seed


def _build_log_bounds(seed: np.ndarray, half_range: float):
    """Build log-space bounds around the seed.

    The optimizer searches in log space (``y = log(x)``) so that the wide
    dynamic range of neutron spectra (many orders of magnitude) is handled
    naturally. Bounds are ``log(seed) +/- half_range`` decades.
    """
    y0 = np.log(np.maximum(np.asarray(seed, dtype=float), 1e-300))
    lb = y0 - half_range * np.log(10.0)
    ub = y0 + half_range * np.log(10.0)
    return lb, ub


def _build_fitness(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int,
    L,
    smoothness_weight: float,
    entropy_weight: float,
) -> Callable[[np.ndarray], float]:
    """Build the unfolding objective function in log space.

    The optimizer searches for ``y`` with ``x = exp(y)``. All terms are
    normalised by their natural scales so that the residual, regularisation
    and smoothness contributions are dimensionless and comparable:

    ``f(y) = ||b - A exp(y)||^2 / ||b||^2
           + alpha * ||exp(y)||_norm / x_scale
           + smoothness_weight * ||L exp(y)||^2 / x_scale^2
           - entropy_weight * H(exp(y))``

    where ``x_scale = ||b|| / ||A||`` is the characteristic magnitude of a
    spectrum that reproduces the readings. This scale-consistent formulation
    fixes the previous scaling bug where the relative residual was
    dimensionless but the regularisation term carried units of ``x^2``,
    letting the optimizer inflate ``x`` without penalty and produce noise.
    """
    denom = float(np.dot(b, b))
    if denom <= 0.0:
        denom = 1.0
    A_fro = float(np.linalg.norm(A))
    if A_fro <= 0.0:
        A_fro = 1.0
    x_scale = np.sqrt(denom) / A_fro
    x_scale2 = x_scale * x_scale

    def fitness(y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        x = np.exp(y)
        residual = A @ x - b
        value = float(np.dot(residual, residual)) / denom
        if alpha > 0:
            if norm == 2:
                value += alpha * float(np.dot(x, x)) / x_scale2
            elif norm == 1:
                value += alpha * float(np.sum(np.abs(x))) / x_scale
        if L is not None and smoothness_weight > 0:
            Lx = L @ x
            value += smoothness_weight * float(np.dot(Lx, Lx)) / x_scale2
        if entropy_weight > 0:
            total = float(np.sum(x))
            if total > 0:
                p = x / total
                logp = np.log(np.maximum(p, 1e-300))
                value -= entropy_weight * float(np.dot(p, logp))
        return value

    return fitness


def _make_starting_solutions(
    seed: np.ndarray, lb: np.ndarray, ub: np.ndarray, pop_size: int
) -> np.ndarray:
    """Build a starting-solutions matrix with the seed as the first individual.

    The population is sampled uniformly in log space within the bounds, with
    the seed (``log(seed)``) placed as the first individual so the optimizer
    always starts from a good warm-start solution.
    """
    n = len(seed)
    rng = np.random.default_rng(0)
    starting = rng.uniform(lb, ub, size=(pop_size, n))
    starting[0] = np.log(np.maximum(np.asarray(seed, dtype=float), 1e-300))
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
    regularization: float = 1e-2,
    norm: int = 2,
    smoothness_order: int = 2,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    n_runs: int = 1,
    early_stop: Optional[int] = None,
    half_range: float = 2.0,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Solve the unfolding problem using a meta-heuristic optimizer.

    The optimizer searches in log space (``y = log(x)``) so that the wide
    dynamic range of neutron spectra is handled naturally. The population is
    seeded with a Landweber warm-start solution (or the provided ``x0``) and
    bounded to ``log(seed) +/- half_range`` decades. All objective terms are
    scale-consistent (dimensionless), which prevents the optimizer from
    producing a noisy, arbitrary spectrum.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None (or all zeros), a Landweber
        warm-start solution is used to seed the population.
    solver : str, optional
        Meta-heuristic algorithm: 'pso', 'ga', 'de', 'es', 'ep', 'abc',
        'gwo' or 'cmaes' (default: 'pso').
    epoch : int, optional
        Maximum number of generations/iterations (default: 500).
    pop_size : int, optional
        Population size (default: 50).
    regularization : float, optional
        Tikhonov regularisation weight alpha (default: 1e-2).
    norm : int, optional
        Norm for the regularisation term (1 for L1, 2 for L2), default: 2.
    smoothness_order : int, optional
        Second-difference smoothing order (0, 1 or 2), default: 2.
    smoothness_weight : float, optional
        Weight of the smoothing term (default: 1.0).
    entropy_weight : float, optional
        Weight of the negative Shannon-entropy objective (0 disables it).
    n_runs : int, optional
        Number of independent optimisation runs; results are averaged
        (default: 1).
    early_stop : int, optional
        Stop if the global best does not improve for this many consecutive
        epochs (MEALPY early stopping).
    half_range : float, optional
        Half-width of the log-space search bounds in decades around the seed
        (default: 2.0).
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
            early_stop=early_stop, half_range=half_range,
            random_state=random_state, verbose=verbose,
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
    half_range: float,
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
    seed = _build_seed(A, b, x0)
    lb, ub = _build_log_bounds(seed, half_range)

    problem = {
        "obj_func": fitness,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min",
        "log_to": "console" if verbose else None,
    }

    termination = None
    if early_stop is not None:
        termination = {"max_early_stop": int(early_stop)}

    starting = _make_starting_solutions(seed, lb, ub, pop_size)
    runs = max(1, int(n_runs))
    spectra = []
    for run in range(runs):
        model = _build_model(mealpy, solver, int(epoch), int(pop_size))
        seed_val = None
        if random_state is not None:
            seed_val = int(random_state) + run
        g_best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g_best = model.solve(
                problem,
                seed=seed_val,
                termination=termination,
                starting_solutions=starting,
            )
        # Convert back from log space to the actual spectrum.
        spectra.append(np.exp(np.asarray(g_best.solution, dtype=float)))

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
    regularization: float = 1e-2,
    norm: int = 2,
    smoothness_order: int = 2,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    n_runs: int = 1,
    early_stop: Optional[int] = None,
    half_range: float = 2.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Unfold a neutron spectrum using a meta-heuristic algorithm.

    The optimizer searches in log space seeded with a Landweber warm-start
    solution (or the provided ``initial_spectrum``), bounded to
    ``log(seed) +/- half_range`` decades, with a scale-consistent objective.

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
        Initial spectrum guess. If None, a Landweber warm-start solution is
        used to seed the population.
    solver : str, optional
        Meta-heuristic algorithm: 'pso', 'ga', 'de', 'es', 'ep', 'abc',
        'gwo' or 'cmaes' (default: 'pso').
    epoch : int, optional
        Maximum number of generations (default: 500).
    pop_size : int, optional
        Population size (default: 50).
    regularization : float, optional
        Tikhonov regularisation weight (default: 1e-2).
    norm : int, optional
        Norm for the regularisation term (1 or 2), default: 2.
    smoothness_order : int, optional
        Second-difference smoothing order (0, 1 or 2), default: 2.
    smoothness_weight : float, optional
        Weight of the smoothing term (default: 1.0).
    entropy_weight : float, optional
        Weight of the negative Shannon-entropy objective (default: 0).
    n_runs : int, optional
        Number of independent runs to average (default: 1).
    early_stop : int, optional
        Early-stopping patience (epochs without improvement).
    half_range : float, optional
        Half-width of the log-space search bounds in decades around the seed
        (default: 2.0).
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
            half_range=half_range,
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
            "half_range": half_range,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
