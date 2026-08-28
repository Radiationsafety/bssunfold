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
MEALPY (PSO, GA, DE, ES, EP, ABC, GWO, CMA-ES) plus two self-contained
numpy engines that extend the published approaches:

- ``nsga2``: a real-coded NSGA-II multi-objective engine (Deb et al., 2002)
  that derives the Pareto front of two objectives -- the relative response
  error ``||b - A x||^2 / ||b||^2`` and the negative Shannon entropy
  ``-H(x)`` -- as in Woo et al., Prog. Nucl. Sci. Technol. 6 (2019).  A
  single solution is returned from the front (knee by default).
- ``two_step`` mode: a two-step genetic scheme inspired by the TGASU code
  (Shahabinejad et al., NIMA 811 (2016)) in which the unfolding problem is
  first solved on a coarse energy grid (roughly 60% of the variables) and
  the result is interpolated back to seed the full-resolution population.
- TGASU-style genetic operators: arithmetic crossover (beta-weighted
  averaging) and iterative mutation with a generation-decreasing step, plus
  a two-stage smoother (Gaussian + multiplicative bias correction) for
  oscillation reduction.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding
from ._matrix_utils import create_derivative_matrix
from ._multires import _coarsen_columns, _split_coarse  # re-exported for callers

__all__ = ["solve_genetic", "unfold_genetic"]

_IMPORT_ERROR_MSG = (
    "mealpy is required for unfold_genetic. Install with: pip install mealpy"
)

# Supported meta-heuristic solvers (registry keys).
_SUPPORTED_SOLVERS = (
    "pso",
    "ga",
    "de",
    "es",
    "ep",
    "abc",
    "gwo",
    "cmaes",
    "nsga2",
)

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
    "non_dominated_sorting_genetic_algorithm_ii": "nsga2",
    "nondominated_sorting_genetic_algorithm_ii": "nsga2",
    "pareto": "nsga2",
    "multi_objective": "nsga2",
}


def _import_mealpy():
    """Import the MEALPY modules, raising a helpful ImportError if missing."""
    try:
        from mealpy import ABC, DE, EP, ES, GA, GWO, PSO, FloatVar
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return FloatVar, PSO, GA, DE, ES, EP, ABC, GWO


def _normalize_solver(solver: str) -> str:
    """Resolve a solver name (or alias) to a registry key, warning on unknown."""
    name = (solver or "").strip().lower()
    if name in _SOLVER_ALIASES:
        name = _SOLVER_ALIASES.get(name, name)
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
    seed: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int,
    extra: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a starting-solutions matrix with the seed as the first individual.

    The population is sampled uniformly in log space within the bounds, with
    the seed (``log(seed)``) placed as the first individual so the optimizer
    always starts from a good warm-start solution.  When ``extra`` is given,
    it is placed as the second individual (clipped to the bounds), allowing a
    two-step scheme to inject a coarse solution without shifting the search
    box that is defined by the seed.
    """
    n = len(seed)
    rng = np.random.default_rng(0)
    starting = rng.uniform(lb, ub, size=(pop_size, n))
    starting[0] = np.log(np.maximum(np.asarray(seed, dtype=float), 1e-300))
    if extra is not None:
        y_extra = np.log(np.maximum(np.asarray(extra, dtype=float), 1e-300))
        starting[1] = np.clip(y_extra, lb, ub)
    return starting


def _build_model(mealpy, solver: str, epoch: int, pop_size: int):
    """Instantiate the MEALPY optimizer for the given solver key."""
    _, PSO, GA, DE, ES, EP, ABC, GWO = mealpy
    if solver == "pso":
        # Chaotic PSO with SDPSO-style inertia weight range.
        return PSO.C_PSO(
            epoch=epoch,
            pop_size=pop_size,
            c1=2.0,
            c2=2.0,
            w_min=0.4,
            w_max=0.9,
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


def _normalize_smoother(smoother: Optional[str]) -> str:
    """Normalise the post-processing smoother name, warning on unknown."""
    name = (smoother or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "no": "none",
        "off": "none",
        "gauss": "gaussian",
        "gaussian_multiplicative_bias_correction": "gaussian_mbc",
        "gauss_mbc": "gaussian_mbc",
        "mbc": "gaussian_mbc",
        "2nd_difference": "second_difference",
        "seconddifference": "second_difference",
        "d2": "second_difference",
    }
    name = aliases.get(name, name)
    valid = ("none", "gaussian", "mbc", "gaussian_mbc", "second_difference")
    if name not in valid:
        warnings.warn(
            f"Smoother '{smoother}' not supported. "
            f"Available smoothers: {valid}. Using 'none'."
        )
        name = "none"
    return name


def _apply_smoother(
    x: np.ndarray,
    smoother: str,
    sigma: float = 2.0,
    smoothing_weight: float = 1.0,
) -> np.ndarray:
    """Post-process the spectrum with a two-stage oscillation-reduction smoother.

    Implements the smoothing strategies discussed in Suman & Sarkar (2012)
    and the two-stage Gaussian + multiplicative bias correction (MBC) scheme
    of the TGASU code (Shahabinejad et al., 2016).  The smoothed spectrum is
    clipped to non-negative values and rescaled to preserve the total
    fluence of the input.

    Parameters
    ----------
    x : np.ndarray
        Spectrum to smooth.
    smoother : str
        One of 'none', 'gaussian', 'mbc', 'gaussian_mbc' or
        'second_difference'.
    sigma : float, optional
        Standard deviation of the Gaussian filter (default: 2.0).
    smoothing_weight : float, optional
        Weight of the second-difference penalty used by the
        'second_difference' smoother (default: 1.0).

    Returns
    -------
    np.ndarray
        Smoothed spectrum with the same shape and total fluence.
    """
    if smoother == "none":
        return x

    from scipy.ndimage import gaussian_filter1d

    x_arr = np.asarray(x, dtype=float)
    n = x_arr.shape[0]

    if smoother == "gaussian":
        s = gaussian_filter1d(x_arr, sigma=sigma, mode="nearest")
    elif smoother == "mbc":
        s = gaussian_filter1d(x_arr, sigma=sigma, mode="nearest")
        bias = x_arr / np.maximum(s, 1e-300)
        correction = gaussian_filter1d(bias, sigma=sigma, mode="nearest")
        s = x_arr * correction
    elif smoother == "gaussian_mbc":
        s = gaussian_filter1d(x_arr, sigma=sigma, mode="nearest")
        bias = x_arr / np.maximum(s, 1e-300)
        correction = gaussian_filter1d(bias, sigma=sigma, mode="nearest")
        s = s * correction
    elif smoother == "second_difference":
        L = create_derivative_matrix(n, 2)
        M = np.eye(n) + smoothing_weight * (L.T @ L).toarray()
        try:
            s = np.linalg.solve(M, x_arr)
        except np.linalg.LinAlgError:
            s = x_arr.copy()
    else:
        return x_arr

    s = np.maximum(s, 0.0)
    total_in = float(np.sum(x_arr))
    total_out = float(np.sum(s))
    if total_out > 0 and total_in > 0:
        s = s * (total_in / total_out)
    return s


# _coarsen_columns / _split_coarse are defined in core._multires and
# re-exported here for backward compatibility (see tests/test_genetic_improvements.py).


def _run_numpy_ga(
    A: np.ndarray,
    b: np.ndarray,
    fitness,
    seed: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    epoch: int,
    pop_size: int,
    crossover: str,
    mutation: str,
    pc: float,
    pm: float,
    random_state: Optional[int],
    verbose: bool,
) -> np.ndarray:
    """Self-contained numpy genetic algorithm with TGASU-style operators.

    Searches in log space (``y`` with ``x = exp(y)``).  Supports single-point
    or arithmetic (beta-weighted) crossover and random or iterative
    (generation-decreasing step) mutation, with elitist preservation.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    fitness : callable
        Objective ``f(y)`` to minimise (built by :func:`_build_fitness`).
    seed : np.ndarray
        Seed spectrum used as the first individual.
    lb, ub : np.ndarray
        Log-space lower/upper bounds.
    epoch : int
        Number of generations.
    pop_size : int
        Population size.
    crossover : str
        'single' or 'arithmetic'.
    mutation : str
        'random' or 'iterative'.
    pc : float
        Crossover probability.
    pm : float
        Mutation probability.
    random_state : int, optional
        Random seed.
    verbose : bool
        Unused; kept for interface symmetry.

    Returns
    -------
    np.ndarray
        Best spectrum (n,) in linear space.
    """
    rng = np.random.default_rng(random_state)
    n = seed.shape[0]
    y0 = np.log(np.maximum(seed, 1e-300))
    pop = rng.uniform(lb, ub, size=(pop_size, n))
    pop[0] = y0
    elite = max(1, pop_size // 10)
    scale0 = 0.3

    def _tournament_select() -> Tuple[int, int]:
        idx = rng.integers(0, pop_size, size=4)
        a, b1 = idx[0], idx[1]
        c, d = idx[2], idx[3]
        w1 = a if fitness(pop[a]) <= fitness(pop[b1]) else b1
        w2 = c if fitness(pop[c]) <= fitness(pop[d]) else d
        return w1, w2

    for gen in range(int(epoch)):
        vals = np.array([fitness(y) for y in pop])
        order = np.argsort(vals)
        elites_ = pop[order[:elite]].copy()
        scale = scale0 * (1.0 - gen / max(int(epoch), 1))
        offspring = []
        while len(offspring) < pop_size - elite:
            p1, p2 = _tournament_select()
            y1, y2 = pop[p1], pop[p2]
            if rng.random() < pc:
                if crossover == "arithmetic":
                    beta = rng.random()
                    c1 = beta * y1 + (1.0 - beta) * y2
                    c2 = (1.0 - beta) * y1 + beta * y2
                else:
                    point = int(rng.integers(1, n))
                    c1 = np.concatenate([y1[:point], y2[point:]])
                    c2 = np.concatenate([y2[:point], y1[point:]])
            else:
                c1, c2 = y1.copy(), y2.copy()
            for child in (c1, c2):
                mask = rng.random(n) < pm
                if np.any(mask):
                    if mutation == "iterative":
                        # phi_new = phi_old * (1 + scale * beta), beta in [-1, 1]
                        beta = rng.uniform(-1.0, 1.0, size=n)
                        child = child + np.log(np.maximum(1.0 + scale * beta, 1e-12))
                    else:
                        child = child + rng.normal(0.0, scale, size=n)
                    child = np.clip(child, lb, ub)
                offspring.append(child)
                if len(offspring) >= pop_size - elite:
                    break
        pop = np.vstack([elites_, np.asarray(offspring)[: pop_size - elite]])

    vals = np.array([fitness(y) for y in pop])
    best = pop[int(np.argmin(vals))]
    return np.maximum(np.exp(best), 0.0)


def _fast_non_dominated_sort(fvals: np.ndarray) -> List[np.ndarray]:
    """Return the Pareto fronts of a population (minimisation).

    Parameters
    ----------
    fvals : np.ndarray
        Objective values (N x n_obj).

    Returns
    -------
    List[np.ndarray]
        Fronts, each an array of individual indices; fronts[0] is the
        non-dominated (Pareto) front.
    """
    N = fvals.shape[0]
    # Vectorized domination check using broadcasting
    # dominates[i,j] = all(fvals[i] <= fvals[j]) and any(fvals[i] < fvals[j])
    leq = fvals[:, None, :] <= fvals[None, :, :]  # (N, N, n_obj)
    lt = fvals[:, None, :] < fvals[None, :, :]
    dominates = np.all(leq, axis=2) & np.any(lt, axis=2)  # (N, N)
    np.fill_diagonal(dominates, False)

    fronts: List[np.ndarray] = []
    remaining = np.arange(N)
    while remaining.size:
        # A remaining individual i is non-dominated if no other remaining j dominates i
        # i.e. not np.any(dominates[remaining, i]) for i in remaining
        dom_remaining = dominates[np.ix_(remaining, remaining)]
        is_nondominated = ~np.any(dom_remaining, axis=0)
        front = remaining[is_nondominated]
        fronts.append(front)
        remaining = remaining[~is_nondominated]
    return fronts


def _crowding_distance(fvals: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Assign crowding distances to the individuals of a front."""
    m = front.shape[0]
    dist = np.zeros(m)
    if m <= 2:
        return np.full(m, np.inf)
    n_obj = fvals.shape[1]
    front_vals = fvals[front]  # (m, n_obj)
    for obj in range(n_obj):
        order = np.argsort(front_vals[:, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin = front_vals[0, obj]  # min after sort
        fmax = front_vals[-1, obj]  # max after sort
        spread = fmax - fmin
        if spread <= 0:
            continue
        # Vectorized crowding distance for interior points
        dist[order[1:-1]] += (
            front_vals[order[2:], obj] - front_vals[order[:-2], obj]
        ) / spread
    return dist


def _sbx_crossover(p1, p2, lb, ub, rng, eta_c: float = 15.0):
    """Simulated binary crossover with clipping to the bounds."""
    n = p1.shape[0]
    mask = rng.random(n) < 0.5
    u = rng.random(n)
    with np.errstate(divide="ignore"):
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
        )
    sum_p = p1 + p2
    diff = p1 - p2
    c1 = 0.5 * (sum_p - beta * diff)
    c2 = 0.5 * (sum_p + beta * diff)
    c1 = np.where(mask, c1, p1)
    c2 = np.where(mask, c2, p2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def _polynomial_mutation(p, lb, ub, rng, eta_m: float = 20.0):
    """Polynomial mutation with clipping to the bounds."""
    n = p.shape[0]
    pm = 1.0 / n
    mask = rng.random(n) < pm
    if not np.any(mask):
        return p.copy()
    u = rng.random(n)
    with np.errstate(divide="ignore"):
        delta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0)),
        )
    p = p + delta * (ub - lb)
    return np.clip(p, lb, ub)


def _select_knee(fvals: np.ndarray) -> int:
    """Return the knee index (closest to the ideal point) of a Pareto front."""
    ideal = fvals.min(axis=0)
    spread = fvals.max(axis=0) - ideal
    spread = np.where(spread == 0, 1.0, spread)
    norm = (fvals - ideal) / spread
    return int(np.argmin(np.linalg.norm(norm, axis=1)))


def _run_nsga2(
    A: np.ndarray,
    b: np.ndarray,
    seed: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    epoch: int,
    pop_size: int,
    random_state: Optional[int],
    pareto_select: str,
    entropy_weight: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run a real-coded NSGA-II over two objectives (relative error, entropy).

    The two objectives (both minimised) are:

    * ``f1 = ||b - A exp(y)||^2 / ||b||^2`` (relative response error);
    * ``f2 = -entropy_weight * H(exp(y))`` (negative Shannon entropy).

    After evolution, the Pareto front is extracted and a single solution is
    selected from it according to ``pareto_select``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    seed : np.ndarray
        Seed spectrum (first individual).
    lb, ub : np.ndarray
        Log-space bounds.
    epoch : int
        Number of generations.
    pop_size : int
        Population size.
    random_state : int, optional
        Random seed.
    pareto_select : str
        'knee', 'min_residual' or 'max_entropy'.
    entropy_weight : float, optional
        Weight of the entropy objective (default: 1.0).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (selected spectrum, diagnostics with front residual/entropy).
    """
    rng = np.random.default_rng(random_state)
    n = seed.shape[0]
    denom = float(np.dot(b, b))
    if denom <= 0.0:
        denom = 1.0

    def _objectives(pop: np.ndarray) -> np.ndarray:
        x = np.exp(pop)
        resid = (A @ x.T).T - b
        f1 = np.sum(resid * resid, axis=1) / denom
        total = np.maximum(np.sum(x, axis=1), 1e-300)
        p = x / total[:, None]
        logp = np.log(np.maximum(p, 1e-300))
        ent = -np.sum(p * logp, axis=1)
        f2 = -entropy_weight * ent
        return np.column_stack([f1, f2])

    pop = rng.uniform(lb, ub, size=(pop_size, n))
    pop[0] = np.log(np.maximum(seed, 1e-300))

    for _ in range(int(epoch)):
        fvals = _objectives(pop)
        fronts = _fast_non_dominated_sort(fvals)
        rank = np.empty(pop_size, dtype=int)
        for r, front in enumerate(fronts):
            rank[front] = r
        crowding = np.empty(pop_size)
        for front in fronts:
            crowding[front] = _crowding_distance(fvals, front)

        # Vectorized binary tournament selection by (rank, -crowding).
        idx_all = rng.integers(0, pop_size, size=(pop_size, 4))
        rank_a = rank[idx_all[:, 0]]
        rank_b = rank[idx_all[:, 1]]
        rank_c = rank[idx_all[:, 2]]
        rank_d = rank[idx_all[:, 3]]
        crowd_a = crowding[idx_all[:, 0]]
        crowd_b = crowding[idx_all[:, 1]]
        crowd_c = crowding[idx_all[:, 2]]
        crowd_d = crowding[idx_all[:, 3]]
        # w1: pick a if (rank_a, -crowd_a) < (rank_b, -crowd_b)
        w1_less = (rank_a < rank_b) | ((rank_a == rank_b) & (crowd_a > crowd_b))
        w1 = np.where(w1_less, idx_all[:, 0], idx_all[:, 1])
        # w2: pick c if (rank_c, -crowd_c) < (rank_d, -crowd_d)
        w2_less = (rank_c < rank_d) | ((rank_c == rank_d) & (crowd_c > crowd_d))
        w2 = np.where(w2_less, idx_all[:, 2], idx_all[:, 3])
        # Final: pick w1 if (rank[w1], -crowd[w1]) <= (rank[w2], -crowd[w2])
        rw1 = rank[w1]
        cw1 = crowding[w1]
        rw2 = rank[w2]
        cw2 = crowding[w2]
        final_less = (rw1 < rw2) | ((rw1 == rw2) & (cw1 >= cw2))
        winner = np.where(final_less, w1, w2)
        selected = pop[winner]

        # SBX + polynomial mutation.
        offspring = []
        for i in range(0, pop_size, 2):
            c1, c2 = _sbx_crossover(selected[i], selected[i + 1], lb, ub, rng)
            c1 = _polynomial_mutation(c1, lb, ub, rng)
            c2 = _polynomial_mutation(c2, lb, ub, rng)
            offspring.extend([c1, c2])
        offspring = np.asarray(offspring)[:pop_size]

        combined = np.vstack([pop, offspring])
        cf = _objectives(combined)
        cfronts = _fast_non_dominated_sort(cf)
        cfronts = [f for f in cfronts if f.size]
        next_pop = []
        remaining = 2 * pop_size
        for front in cfronts:
            if len(next_pop) + front.size <= pop_size:
                next_pop.extend(front.tolist())
                remaining -= front.size
            else:
                cd = _crowding_distance(cf, front)
                order = np.argsort(-cd)
                needed = pop_size - len(next_pop)
                next_pop.extend(front[order[:needed]].tolist())
                break
        pop = combined[next_pop]

    fvals = _objectives(pop)
    fronts = _fast_non_dominated_sort(fvals)
    front0 = fronts[0]
    f0 = fvals[front0]
    if pareto_select == "min_residual":
        idx = int(front0[int(np.argmin(f0[:, 0]))])
    elif pareto_select == "max_entropy":
        idx = int(front0[int(np.argmin(f0[:, 1]))])
    else:  # 'knee'
        idx = int(front0[_select_knee(f0)])
    spectrum = np.maximum(np.exp(pop[idx]), 0.0)
    diagnostics = {
        "pareto_front_size": int(front0.size),
        "pareto_min_residual": float(np.min(f0[:, 0])),
        "pareto_max_entropy": float(np.max(-f0[:, 1])),
        "pareto_select": pareto_select,
    }
    return spectrum, diagnostics


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
    two_step: bool = False,
    n_coarse: Optional[int] = None,
    smoother: str = "none",
    sigma_smooth: float = 2.0,
    crossover: str = "single",
    mutation: str = "random",
    pareto_select: str = "knee",
    random_state: Optional[int] = None,
    verbose: bool = False,
    extra_starting: Optional[np.ndarray] = None,
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
        'gwo', 'cmaes' or 'nsga2' (default: 'pso').
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
        (default: 1). Not used by the 'nsga2' solver.
    early_stop : int, optional
        Stop if the global best does not improve for this many consecutive
        epochs (MEALPY early stopping). Not used by the numpy engines.
    half_range : float, optional
        Half-width of the log-space search bounds in decades around the seed
        (default: 2.0).
    two_step : bool, optional
        If True, run the two-step genetic scheme (TGASU-style): the problem
        is first solved on a coarse energy grid and the result is
        interpolated back to seed the full-resolution population
        (default: False).
    n_coarse : int, optional
        Number of coarse bins for the ``two_step`` mode. When None, it is
        chosen as ``max(8, n // 4)``.
    smoother : str, optional
        Post-processing smoother: 'none', 'gaussian', 'mbc',
        'gaussian_mbc' or 'second_difference' (default: 'none').
    sigma_smooth : float, optional
        Gaussian filter sigma for the smoothers (default: 2.0).
    crossover : str, optional
        GA crossover operator: 'single' (single-point) or 'arithmetic'
        (beta-weighted, TGASU). Only used by the numpy GA engine
        (default: 'single').
    mutation : str, optional
        GA mutation operator: 'random' or 'iterative' (generation-decreasing
        step, TGASU). Only used by the numpy GA engine (default: 'random').
    pareto_select : str, optional
        Selection from the Pareto front for the 'nsga2' solver: 'knee',
        'min_residual' or 'max_entropy' (default: 'knee').
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, MEALPY logs the optimisation progress to the console.
    extra_starting : np.ndarray, optional
        Additional starting individual (in linear spectrum units) injected
        into the initial population without shifting the search box. Used
        internally by the two-step scheme.

    Returns
    -------
    np.ndarray
        Unfolded spectrum (n,). Returns a zero vector if solving failed.
    """
    try:
        return _solve_genetic_impl(
            A=A,
            b=b,
            x0=x0,
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
            two_step=two_step,
            n_coarse=n_coarse,
            smoother=smoother,
            sigma_smooth=sigma_smooth,
            crossover=crossover,
            mutation=mutation,
            pareto_select=pareto_select,
            random_state=random_state,
            verbose=verbose,
            extra_starting=extra_starting,
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
    two_step: bool,
    n_coarse: Optional[int],
    smoother: str,
    sigma_smooth: float,
    crossover: str,
    mutation: str,
    pareto_select: str,
    random_state: Optional[int],
    verbose: bool,
    extra_starting: Optional[np.ndarray] = None,
) -> np.ndarray:
    mealpy = _import_mealpy()
    FloatVar, _, _, _, _, _, _, _ = mealpy
    solver = _normalize_solver(solver)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    if norm not in (1, 2):
        raise ValueError(f"Unsupported norm type: {norm}")
    if smoothness_order not in (0, 1, 2):
        raise ValueError(f"Unsupported smoothness order: {smoothness_order}")
    if crossover not in ("single", "arithmetic"):
        raise ValueError(
            f"Unsupported crossover operator: {crossover}. "
            f"Use 'single' or 'arithmetic'."
        )
    if mutation not in ("random", "iterative"):
        raise ValueError(
            f"Unsupported mutation operator: {mutation}. Use 'random' or 'iterative'."
        )
    if pareto_select not in ("knee", "min_residual", "max_entropy"):
        raise ValueError(
            f"Unsupported pareto_select: {pareto_select}. "
            f"Use 'knee', 'min_residual' or 'max_entropy'."
        )

    if two_step:
        n_coarse_ = n_coarse or max(8, n // 4)
        if n_coarse_ >= n:
            n_coarse_ = max(1, n // 2)
        A_coarse = _coarsen_columns(A, n_coarse_)
        if x0 is not None and np.any(np.asarray(x0, dtype=float) > 0):
            coarse_x0 = _coarsen_columns(
                np.asarray(x0, dtype=float)[None, :], n_coarse_
            )[0]
        else:
            coarse_x0 = np.ones(n_coarse_)
        coarse = _solve_genetic_impl(
            A=A_coarse,
            b=b,
            x0=coarse_x0,
            solver=solver,
            epoch=max(20, epoch // 2),
            pop_size=pop_size,
            regularization=regularization,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            entropy_weight=entropy_weight,
            n_runs=n_runs,
            early_stop=early_stop,
            half_range=half_range,
            two_step=False,
            n_coarse=None,
            smoother="none",
            sigma_smooth=sigma_smooth,
            crossover=crossover,
            mutation=mutation,
            pareto_select=pareto_select,
            random_state=random_state,
            verbose=verbose,
        )
        x0_seed = _split_coarse(np.maximum(coarse, 0.0), n)
        # The full-resolution run keeps the Landweber-defined search box and
        # injects the coarse solution as an additional starting individual.
        return _solve_genetic_impl(
            A=A,
            b=b,
            x0=x0,
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
            two_step=False,
            n_coarse=None,
            smoother="none",
            sigma_smooth=sigma_smooth,
            crossover=crossover,
            mutation=mutation,
            pareto_select=pareto_select,
            random_state=random_state,
            verbose=verbose,
            extra_starting=x0_seed,
        )

    L = None
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)

    if solver == "nsga2":
        seed = _build_seed(A, b, x0)
        lb, ub = _build_log_bounds(seed, half_range)
        spectrum, _diag = _run_nsga2(
            A=A,
            b=b,
            seed=seed,
            lb=lb,
            ub=ub,
            epoch=epoch,
            pop_size=pop_size,
            random_state=random_state,
            pareto_select=pareto_select,
            entropy_weight=entropy_weight if entropy_weight > 0 else 1.0,
        )
        smoother = _normalize_smoother(smoother)
        if smoother != "none":
            spectrum = _apply_smoother(
                spectrum,
                smoother,
                sigma=sigma_smooth,
                smoothing_weight=smoothness_weight,
            )
        return np.maximum(spectrum, 0.0)

    if solver == "ga" and (crossover != "single" or mutation != "random"):
        seed = _build_seed(A, b, x0)
        lb, ub = _build_log_bounds(seed, half_range)
        fitness = _build_fitness(
            A, b, regularization, norm, L, smoothness_weight, entropy_weight
        )
        spectrum = _run_numpy_ga(
            A=A,
            b=b,
            fitness=fitness,
            seed=seed,
            lb=lb,
            ub=ub,
            epoch=epoch,
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            pc=0.9,
            pm=0.05,
            random_state=random_state,
            verbose=verbose,
        )
        smoother = _normalize_smoother(smoother)
        if smoother != "none":
            spectrum = _apply_smoother(
                spectrum,
                smoother,
                sigma=sigma_smooth,
                smoothing_weight=smoothness_weight,
            )
        return np.maximum(spectrum, 0.0)

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

    starting = _make_starting_solutions(seed, lb, ub, pop_size, extra=extra_starting)
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
    smoother = _normalize_smoother(smoother)
    if smoother != "none":
        spectrum = _apply_smoother(
            spectrum,
            smoother,
            sigma=sigma_smooth,
            smoothing_weight=smoothness_weight,
        )
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
    two_step: bool = False,
    n_coarse: Optional[int] = None,
    smoother: str = "none",
    sigma_smooth: float = 2.0,
    crossover: str = "single",
    mutation: str = "random",
    pareto_select: str = "knee",
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
        'gwo', 'cmaes' or 'nsga2' (default: 'pso').
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
        Number of independent runs to average (default: 1). Not used by the
        'nsga2' solver.
    early_stop : int, optional
        Early-stopping patience (epochs without improvement).
    half_range : float, optional
        Half-width of the log-space search bounds in decades around the seed
        (default: 2.0).
    two_step : bool, optional
        If True, run the two-step genetic scheme (TGASU-style) with a coarse
        first step seeding the full-resolution population (default: False).
    n_coarse : int, optional
        Number of coarse bins for the ``two_step`` mode (default: None, i.e.
        ``max(8, n // 4)``).
    smoother : str, optional
        Post-processing smoother: 'none', 'gaussian', 'mbc',
        'gaussian_mbc' or 'second_difference' (default: 'none').
    sigma_smooth : float, optional
        Gaussian filter sigma for the smoothers (default: 2.0).
    crossover : str, optional
        GA crossover operator: 'single' or 'arithmetic' (TGASU); only used by
        the numpy GA engine (default: 'single').
    mutation : str, optional
        GA mutation operator: 'random' or 'iterative' (TGASU, decreasing
        step); only used by the numpy GA engine (default: 'random').
    pareto_select : str, optional
        Selection from the Pareto front for the 'nsga2' solver: 'knee',
        'min_residual' or 'max_entropy' (default: 'knee').
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
    smoother = _normalize_smoother(smoother)
    if crossover not in ("single", "arithmetic"):
        raise ValueError(
            f"Unsupported crossover operator: {crossover}. "
            f"Use 'single' or 'arithmetic'."
        )
    if mutation not in ("random", "iterative"):
        raise ValueError(
            f"Unsupported mutation operator: {mutation}. Use 'random' or 'iterative'."
        )
    if pareto_select not in ("knee", "min_residual", "max_entropy"):
        raise ValueError(
            f"Unsupported pareto_select: {pareto_select}. "
            f"Use 'knee', 'min_residual' or 'max_entropy'."
        )

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
            two_step=two_step,
            n_coarse=n_coarse,
            smoother=smoother,
            sigma_smooth=sigma_smooth,
            crossover=crossover,
            mutation=mutation,
            pareto_select=pareto_select,
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
            "two_step": two_step,
            "smoother": smoother,
            "crossover": crossover,
            "mutation": mutation,
            "pareto_select": pareto_select,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
