"""Pure-Python 3 port of the Gnowee hybrid metaheuristic optimizer.

This module is a self-contained re-implementation of the *continuous-variable*
subset of the Gnowee algorithm described in

    Bevins, J. and Parsons, D., "Gnowee: A rapidly converging hybrid
    metaheuristic optimization algorithm", UC Berkeley / Slaybaugh Lab
    (https://github.com/SlaybaughLab/Gnowee).

Gnowee combines four complementary heuristics to balance diversification and
intensification:

* **Lévy flights** (Cuckoo Search) — heavy-tailed random walks that allow
  large, occasionally wild, jumps to escape local optima.  Sampled via the
  Mantegna (1994) algorithm.
* **Crossover** — golden-ratio weighted recombination between the best parent
  and an elite partner (Walton 2011 / Storn 1997).
* **Mutation** — DE-style differential perturbation biased toward worse
  individuals (Yang 2010 / Storn 1997).
* **Scatter search** — path-relinking combination of two elite parents
  (Egea 2009).

The population is updated with elitism, a Metropolis-Hastings acceptance
fallback, and a stall-driven restart that reinitialises stuck individuals.

Only the continuous-variable heuristics are ported here, because Bonner-sphere
spectrum unfolding works on a real-valued log-space spectrum.  The discrete,
integer and combinatorial heuristics of the original Gnowee are intentionally
omitted.

This port is faithful to the original algorithm but written in idiomatic
Python 3, removes the hard dependency on ``pyDOE`` (we offer 'random' and
'lhc' initial samplers, the latter via ``scipy.stats.qmc.LatinHypercube``),
and uses NumPy's ``default_rng`` for reproducibility.
"""

from __future__ import annotations

import copy as cp
from collections.abc import Callable
from dataclasses import dataclass, field
from math import gamma, sqrt

import numpy as np

__all__ = [
    "Parent",
    "Event",
    "GnoweeSettings",
    "GnoweeHeuristics",
    "levy",
    "tlf",
    "simple_bounds",
    "rejection_bounds",
    "run_gnowee",
]


# --------------------------------------------------------------------------- #
# Sampling primitives (ports of Gnowee's Sampling.py)                          #
# --------------------------------------------------------------------------- #
def levy(nc: int, nr: int = 0, alpha: float = 1.5, gam: float = 1.0, n: int = 1,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample the symmetric Lévy stable distribution via the Mantegna algorithm.

    Mirrors ``Sampling.levy`` from the original Gnowee code, but uses
    ``numpy.random.Generator`` for reproducibility.

    Parameters
    ----------
    nc : int
        Number of columns of Lévy values returned.
    nr : int, optional
        Number of rows (default 0 → 1-D array of shape ``(nc,)``).
    alpha : float, optional
        Lévy exponent, ``0.3 < alpha < 1.99`` (default 1.5).
    gam : float, optional
        Scale of the process (default 1.0).
    n : int, optional
        Number of independent variables used to reduce sampling variance
        (default 1).
    rng : numpy.random.Generator, optional
        Random generator.  If ``None`` a fresh one is created.

    Returns
    -------
    np.ndarray
        Lévy samples of shape ``(nc,)`` if ``nr == 0`` else ``(nr, nc)``.
    """
    if not 0.3 < alpha < 1.99:
        raise ValueError(f"alpha must be in (0.3, 1.99), got {alpha}")
    if gam < 0:
        raise ValueError("gamma must be non-negative")
    if n < 1:
        raise ValueError("n must be positive")

    if rng is None:
        rng = np.random.default_rng()

    invalpha = 1.0 / alpha
    sigx = (
        (
            gamma(1.0 + alpha) * np.sin(np.pi * alpha / 2.0)
        )
        / (
            gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ** ((alpha - 1.0) / 2.0)
        )
    ) ** invalpha

    if nr != 0:
        v = sigx * rng.standard_normal((n, nr, nc)) / (
            np.abs(rng.standard_normal((n, nr, nc))) ** invalpha
        )
    else:
        v = sigx * rng.standard_normal((n, nc)) / (
            np.abs(rng.standard_normal((n, nc))) ** invalpha
        )

    kappa = (
        (alpha * gamma((alpha + 1.0) / (2.0 * alpha)))
        / gamma(invalpha)
        * (
            (alpha * gamma((alpha + 1.0) / 2.0))
            / (gamma(1.0 + alpha) * np.sin(np.pi * alpha / 2.0))
        ) ** invalpha
    )
    # Mantegna's polynomial fit for the temperature parameter c(alpha)
    p = [-17.7767, 113.3855, -281.5879, 337.5439, -193.5494, 44.8754]
    c = float(np.polyval(p, alpha))
    w = ((kappa - 1.0) * np.exp(-np.abs(v) / c) + 1.0) * v

    if n > 1:
        z = (1.0 / n ** invalpha) * np.sum(w, axis=0)
    else:
        z = w[0] if n == 1 else w

    z = gam ** invalpha * z
    if nr != 0:
        return z.reshape(nr, nc)
    return z.reshape(nc)


def tlf(num_row: int = 1, num_col: int = 1, alpha: float = 1.5,
        gam: float = 1.0, cut_point: float = 10.0,
        rng: np.random.Generator | None = None,
        max_resamples: int = 50) -> np.ndarray:
    """Truncated Lévy flight on the interval (0, 1).

    Port of ``Sampling.tlf``: samples a Lévy, divides by ``cut_point`` and
    resamples any value that exceeds 1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = np.abs(levy(num_row, num_col, alpha=alpha, gam=gam, rng=rng) / cut_point)
    z = z.reshape(num_row, num_col)

    # Resample values above 1.0
    mask = z > 1.0
    n_attempts = 0
    while np.any(mask) and n_attempts < max_resamples:
        n_bad = int(mask.sum())
        if n_bad == 0:
            break
        replacements = np.abs(
            levy(n_bad, 1, alpha=alpha, gam=gam, rng=rng) / cut_point
        ).reshape(-1)
        z[mask] = replacements
        mask = z > 1.0
        n_attempts += 1
    # If still over (rare), just clip
    return np.minimum(z, 1.0)


# --------------------------------------------------------------------------- #
# Boundary helpers                                                            #
# --------------------------------------------------------------------------- #
def simple_bounds(child: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Clip ``child`` to ``[lb, ub]`` (port of ``GnoweeHeuristics.simple_bounds``)."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return np.clip(np.asarray(child, dtype=float), lb, ub)


def rejection_bounds(parent: np.ndarray, child: np.ndarray, step_size: np.ndarray,
                     lb: np.ndarray, ub: np.ndarray,
                     max_reductions: int = 5) -> np.ndarray:
    """Iteratively halve the step until the child lies in ``[lb, ub]``.

    Port of ``GnoweeHeuristics.rejection_bounds``.  If the step cannot be made
    feasible in ``max_reductions`` halvings, the offending components fall back
    to their parent value.
    """
    parent = np.asarray(parent, dtype=float)
    child = np.asarray(child, dtype=float).copy()
    step = np.asarray(step_size, dtype=float).copy()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    out_of_bounds = (child < lb) | (child > ub)
    for _ in range(max_reductions):
        if not np.any(out_of_bounds):
            break
        step[out_of_bounds] *= 0.5
        child[out_of_bounds] -= step[out_of_bounds]
        out_of_bounds = (child < lb) | (child > ub)
    # Final fall-back to the parent for anything still out of bounds
    child[out_of_bounds] = parent[out_of_bounds]
    return child


# --------------------------------------------------------------------------- #
# Population bookkeeping                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Parent:
    """A single population member.

    Attributes mirror Gnowee's ``Parent`` class.
    """
    variables: np.ndarray
    fitness: float = 1.0e15
    change_count: int = 0
    stall_count: int = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Parent(variables={self.variables}, fitness={self.fitness}, "
            f"change_count={self.change_count}, stall_count={self.stall_count})"
        )


@dataclass
class Event:
    """Optimization timeline snapshot."""
    generation: int
    evaluations: int
    fitness: float
    design: np.ndarray


@dataclass
class GnoweeSettings:
    """Hyper-parameters of the Gnowee optimizer.

    Defaults follow the original Gnowee paper but ``max_gens`` and
    ``max_fevals`` are scaled down for the much smaller BSS-unfolding
    budget.  ``stall_limit`` is also reduced — Gnowee's 10 000 evaluations
    of stall makes no sense when ``max_fevals`` is a few thousand.
    """
    population: int = 25
    init_sampling: str = "lhc"            # 'lhc' or 'random'
    frac_mutation: float = 0.2
    frac_elite: float = 0.2
    frac_levy: float = 1.0
    alpha: float = 1.5                    # Lévy exponent
    gamma: float = 1.0                   # Lévy scale
    n: int = 1                           # independent Lévy samples
    scaling_factor: float = 10.0        # Lévy step length scale
    penalty: float = 0.0
    max_gens: int = 200
    max_fevals: int = 5_000
    conv_tol: float = 1.0e-6
    stall_limit: int = 200
    opt_conv_tol: float = 1.0e-2
    optimum: float = 0.0
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Heuristics                                                                   #
# --------------------------------------------------------------------------- #
class GnoweeHeuristics:
    """Continuous-variable Gnowee heuristics.

    The class mirrors ``GnoweeHeuristics`` of the original repository but
    only the parts needed for continuous variables are ported.  Discrete /
    integer / combinatorial heuristics are omitted (BSS unfolding does not
    use them).
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray,
                 objective: Callable[[np.ndarray], float],
                 settings: GnoweeSettings | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("lb and ub must have the same shape")
        self.objective = objective
        self.s = settings or GnoweeSettings()
        self.rng = rng or np.random.default_rng()
        if self.s.frac_mutation < 0 or self.s.frac_mutation > 1:
            raise ValueError("frac_mutation must lie in [0, 1]")
        if self.s.frac_elite < 0 or self.s.frac_elite > 1:
            raise ValueError("frac_elite must lie in [0, 1]")
        if self.s.frac_levy < 0 or self.s.frac_levy > 1:
            raise ValueError("frac_levy must lie in [0, 1]")

    # ----------------------------------------------------------------- #
    # Initial population                                                 #
    # ----------------------------------------------------------------- #
    def initialize(self, num_samples: int, method: str | None = None) -> np.ndarray:
        """Generate the initial population within ``[lb, ub]``.

        Two samplers are supported: ``'lhc'`` (Latin-hypercube, default) and
        ``'random'`` (uniform).  The original Gnowee offers several more
        (NOLH variants) but they are not needed for BSS unfolding.
        """
        method = (method or self.s.init_sampling or "lhc").lower()
        n_dim = len(self.lb)
        if method == "random":
            samples = self.lb + (self.ub - self.lb) * self.rng.random(
                (num_samples, n_dim)
            )
        elif method in ("lhc", "lhs"):
            try:
                from scipy.stats.qmc import LatinHypercube
                sampler = LatinHypercube(d=n_dim, seed=int(self.rng.integers(0, 2**31 - 1)))
                unit = sampler.random(n=num_samples)
            except Exception:
                # Fallback: crude Latin-hypercube via permutation per dimension
                unit = np.empty((num_samples, n_dim), dtype=float)
                for j in range(n_dim):
                    perm = self.rng.permutation(num_samples)
                    unit[:, j] = (perm + self.rng.random(num_samples)) / num_samples
            samples = self.lb + (self.ub - self.lb) * unit
        else:
            raise ValueError(
                f"Unsupported init_sampling '{method}'. Use 'lhc' or 'random'."
            )
        return samples

    # ----------------------------------------------------------------- #
    # Lévy flight                                                        #
    # ----------------------------------------------------------------- #
    def cont_levy_flight(self, pop: list[Parent]) -> tuple[list[np.ndarray], list[int]]:
        """Continuous Lévy flight children (port of ``cont_levy_flight``).

        ``x_r^{g+1} = x_r^g + (1/scaling_factor) * L_{alpha,gamma}``

        The Lévy step is masked onto the continuous variables (all of them
        here).  ``rejection_bounds`` enforces the box constraints.
        """
        if not pop:
            return [], []
        n_take = max(1, int(self.s.frac_levy * self.s.population))
        n_take = min(n_take, len(pop))
        # Pick parents without replacement
        idx = self.rng.choice(len(pop), size=n_take, replace=False)

        # levy(nc, nr, ...) returns shape (nr, nc) = (n_take, n_dim).
        step = levy(len(pop[0].variables), n_take, alpha=self.s.alpha,
                    gam=self.s.gamma, n=self.s.n, rng=self.rng)
        children: list[np.ndarray] = []
        used: list[int] = []
        for k, parent_idx in enumerate(idx):
            base = np.array(pop[parent_idx].variables, dtype=float)
            step_size = step[k] / self.s.scaling_factor
            child = base + step_size
            child = rejection_bounds(base, child, step_size, self.lb, self.ub)
            children.append(child)
            used.append(int(parent_idx))
        return children, used

    # ----------------------------------------------------------------- #
    # Crossover (golden-ratio weighted recombination)                    #
    # ----------------------------------------------------------------- #
    def crossover(self, pop: list[Parent]) -> tuple[list[np.ndarray], list[int]]:
        """Crossover the top parent with elite partners (port of ``crossover``).

        ``child = elite_partner + |parent - elite_partner| / phi``
        """
        n_take = max(0, int(self.s.frac_elite * len(pop)))
        if n_take == 0 or len(pop) < 2:
            return [], []
        golden = (1.0 + sqrt(5.0)) / 2.0
        children: list[np.ndarray] = []
        used: list[int] = []
        for i in range(n_take):
            r = self.rng.integers(0, len(pop))
            attempts = 0
            while r == i and attempts < 10:
                r = self.rng.integers(0, len(pop))
                attempts += 1
            if r == i:
                continue
            used.append(i)
            base = np.array(pop[r].variables, dtype=float)
            dx = np.abs(np.array(pop[i].variables, dtype=float) - base) / golden
            child = base + dx
            child = simple_bounds(child, self.lb, self.ub)
            children.append(child)
        return children, used

    # ----------------------------------------------------------------- #
    # Scatter search                                                     #
    # ----------------------------------------------------------------- #
    def scatter_search(self, pop: list[Parent]) -> tuple[list[np.ndarray], list[int]]:
        """Scatter search (Egea 2009) — port of ``scatter_search``."""
        n_take = max(0, int(self.s.frac_elite * len(pop)))
        if n_take == 0 or len(pop) < 2:
            return [], []
        children: list[np.ndarray] = []
        used: list[int] = []
        for i in range(n_take):
            j = self.rng.integers(0, len(pop))
            attempts = 0
            while (j == i or j in used) and attempts < 10:
                j = self.rng.integers(0, len(pop))
                attempts += 1
            if j == i or j in used:
                continue
            used.append(i)
            xi = np.array(pop[i].variables, dtype=float)
            xj = np.array(pop[j].variables, dtype=float)
            d = (xj - xi) / 2.0
            alpha_ = 1.0 if i < j else -1.0
            beta = (abs(j - i) - 1) / max(len(pop) - 2, 1)
            c1 = xi - d * (1.0 + alpha_ * beta)
            c2 = xi + d * (1.0 - alpha_ * beta)
            r = self.rng.random(len(xi))
            child = c1 + (c2 - c1) * r
            child = simple_bounds(child, self.lb, self.ub)
            children.append(child)
        return children, used

    # ----------------------------------------------------------------- #
    # Mutation (DE-style)                                                #
    # ----------------------------------------------------------------- #
    def mutate(self, pop: list[Parent]) -> list[np.ndarray]:
        """DE-style differential mutation (port of ``mutate``).

        ``child = parent + r * (perm1 - perm2) * k`` where ``k`` is a
        per-component mask driven by ``frac_mutation``.
        """
        if not pop:
            return []
        n = len(pop)
        dim = len(pop[0].variables)
        pop_arr = np.array([p.variables for p in pop], dtype=float)
        perm1 = self.rng.permutation(n)
        perm2 = self.rng.permutation(n)
        r = float(self.rng.random())
        k = self.rng.random((n, dim)) > (self.s.frac_mutation * float(self.rng.random()))
        diff = pop_arr[perm1] - pop_arr[perm2]
        step = r * diff
        children = pop_arr + step * k
        # Enforce bounds
        return [simple_bounds(c, self.lb, self.ub) for c in children]

    # ----------------------------------------------------------------- #
    # Population update                                                   #
    # ----------------------------------------------------------------- #
    def population_update(self, parents: list[Parent], children: list[np.ndarray],
                          timeline: list[Event] | None = None,
                          adopted_parents: list[int] | None = None,
                          mh_frac: float = 0.0,
                          random_parents: bool = False) -> tuple[list[Parent], int, list[Event] | None]:
        """Evaluate children and replace worse parents (port of ``population_update``).

        Keeps the parents sorted ascending by fitness.  Implements:
        - elite replacement when a child beats its (adopted) parent
        - Metropolis-Hastings acceptance fallback for otherwise-rejected children
        - stall-driven restart of parents stuck for ``stall_limit`` generations
        - "changeCount" reset for long-stalled non-elite parents
        """
        if adopted_parents is None:
            adopted_parents = []
        n_parents = len(parents)
        n_children = len(children)
        if n_children == 0:
            return parents, 0, timeline

        replace = 0
        feval = 0
        worst = max((p.fitness for p in parents), default=self.s.penalty)
        for i, child_vars in enumerate(children):
            fnew = float(self.objective(np.asarray(child_vars, dtype=float)))
            if fnew > self.s.penalty:
                self.s.penalty = fnew
            feval += 1

            # Decide which parent to compare against
            if random_parents:
                j = int(self.rng.integers(0, n_parents))
            elif len(adopted_parents) == n_children:
                j = int(adopted_parents[i])
            else:
                j = i if i < n_parents else int(self.rng.integers(0, n_parents))

            if fnew < parents[j].fitness:
                parents[j].fitness = fnew
                parents[j].variables = np.array(child_vars, dtype=float)
                parents[j].change_count += 1
                parents[j].stall_count = 0
                replace += 1
                # Reinitialise frequently-improving non-elite parents
                if (parents[j].change_count >= 25
                        and j >= int(self.s.population * self.s.frac_elite)):
                    new_vars = self.initialize(1, "random")[0]
                    fnew2 = float(self.objective(new_vars))
                    parents[j].variables = new_vars
                    parents[j].fitness = fnew2
                    parents[j].change_count = 0
                    feval += 1
            else:
                parents[j].stall_count += 1
                if (parents[j].stall_count > 50_000 and j != 0):
                    new_vars = self.initialize(1, "random")[0]
                    fnew2 = float(self.objective(new_vars))
                    parents[j].variables = new_vars
                    parents[j].fitness = fnew2
                    parents[j].change_count = 0
                    parents[j].stall_count = 0
                    feval += 1
                # Metropolis-Hastings fallback — adopt if it beats a random peer
                if mh_frac > 0.0 and self.rng.random() < mh_frac:
                    r2 = int(self.rng.integers(0, n_parents))
                    if fnew < parents[r2].fitness:
                        parents[r2].fitness = fnew
                        parents[r2].variables = np.array(child_vars, dtype=float)
                        parents[r2].change_count += 1
                        parents[r2].stall_count += 1
                        replace += 1

        parents.sort(key=lambda p: p.fitness)

        if timeline is not None:
            if len(timeline) < 2:
                timeline.append(Event(1, feval, parents[0].fitness,
                                     np.array(parents[0].variables, dtype=float)))
            elif (parents[0].fitness < timeline[-1].fitness
                  and abs((timeline[-1].fitness - parents[0].fitness)
                          / max(abs(parents[0].fitness), 1.0e-300)) > self.s.conv_tol):
                timeline.append(Event(timeline[-1].generation,
                                      timeline[-1].evaluations + feval,
                                      parents[0].fitness,
                                      np.array(parents[0].variables, dtype=float)))
            else:
                timeline[-1].generation += 0  # incremented by caller via genUpdate
                timeline[-1].evaluations += feval
        return parents, replace, timeline


# --------------------------------------------------------------------------- #
# Main optimizer loop                                                          #
# --------------------------------------------------------------------------- #
def run_gnowee(lb: np.ndarray, ub: np.ndarray,
               objective: Callable[[np.ndarray], float],
               settings: GnoweeSettings | None = None,
               rng: np.random.Generator | None = None,
               seed_solution: np.ndarray | None = None,
               extra_starting: np.ndarray | None = None) -> tuple[np.ndarray, float, list[Event]]:
    """Run the Gnowee optimizer on a continuous problem.

    Parameters
    ----------
    lb, ub : np.ndarray
        Box bounds (same shape).
    objective : callable
        ``f(x) -> float`` to minimise.
    settings : GnoweeSettings, optional
        Hyper-parameters. Defaults are used if not provided.
    rng : numpy.random.Generator, optional
        Random generator. If ``None`` a fresh one is created.
    seed_solution : np.ndarray, optional
        If provided, the first population member is initialised to this
        point (clipped to the bounds) instead of being sampled.
    extra_starting : np.ndarray, optional
        Additional starting individuals (e.g. a coarse-step solution) injected
        into the initial population after the seed.

    Returns
    -------
    tuple[np.ndarray, float, list[Event]]
        Best solution, best fitness, timeline of improvements.
    """
    settings = settings or GnoweeSettings()
    rng = rng or np.random.default_rng()
    gh = GnoweeHeuristics(lb=lb, ub=ub, objective=objective,
                          settings=settings, rng=rng)

    # ----- Initial population ------------------------------------------ #
    init_num = max(settings.population * 2, len(gh.lb) * 10)
    init_vars = gh.initialize(init_num, settings.init_sampling)
    if seed_solution is not None:
        seed_clipped = simple_bounds(seed_solution, gh.lb, gh.ub)
        init_vars[0] = seed_clipped
        if extra_starting is not None and len(init_vars) > 1:
            init_vars[1] = simple_bounds(extra_starting, gh.lb, gh.ub)
    init_num = min(init_num, len(init_vars))

    pop: list[Parent] = [Parent(variables=np.array(v, dtype=float)) for v in init_vars]
    # Evaluate and trim to the configured population
    for p in pop:
        p.fitness = float(objective(p.variables))
    pop.sort(key=lambda p: p.fitness)
    if len(pop) > settings.population:
        pop = pop[: settings.population]
    else:
        settings.population = len(pop)

    timeline: list[Event] = []
    # Seed the timeline with the best initial fitness
    timeline.append(Event(0, len(pop), pop[0].fitness,
                          np.array(pop[0].variables, dtype=float)))

    fe = gh.s.frac_elite
    fl = gh.s.frac_levy
    converge = False
    while not converge:
        # Gnowee re-samples the elite/levy fractions each generation for MI
        # problems; for continuous-only we keep them fixed (no MI vars).

        # 1. Lévy flights
        children, ind = gh.cont_levy_flight(pop)
        if children:
            pop, _, timeline = gh.population_update(
                pop, children, timeline=timeline,
                adopted_parents=ind, mh_frac=0.2, random_parents=True,
            )

        # 2. Crossover
        children, ind = gh.crossover(pop)
        if children:
            pop, _, timeline = gh.population_update(
                pop, children, timeline=timeline,
            )

        # 3. Scatter search
        children, ind = gh.scatter_search(pop)
        if children:
            pop, _, timeline = gh.population_update(
                pop, children, timeline=timeline,
                adopted_parents=ind,
            )

        # 4. Mutation
        children = gh.mutate(pop)
        if children:
            pop, _, timeline = gh.population_update(
                pop, children, timeline=timeline,
            )

        # ----- Convergence tests --------------------------------------- #
        gen = timeline[-1].generation + 1
        evals = timeline[-1].evaluations
        if settings.verbose and gen % 10 == 0:
            print(f"Gnowee gen={gen} evals={evals} best={pop[0].fitness:.6e}")

        if evals > settings.stall_limit and len(timeline) >= 2:
            if evals > timeline[-2].evaluations + settings.stall_limit:
                converge = True
                if settings.verbose:
                    print(f"Gnowee: stall at evaluation #{evals}")
        if gen > settings.max_gens:
            converge = True
            if settings.verbose:
                print("Gnowee: max generations reached.")
        if evals > settings.max_fevals:
            converge = True
            if settings.verbose:
                print("Gnowee: max function evaluations reached.")

        # Fitness convergence
        if settings.optimum == 0.0:
            if pop[0].fitness < settings.opt_conv_tol:
                converge = True
                if settings.verbose:
                    print("Gnowee: fitness convergence (absolute).")
        elif abs((pop[0].fitness - settings.optimum) / settings.optimum) <= settings.opt_conv_tol:
            converge = True
            if settings.verbose:
                print("Gnowee: fitness convergence (relative).")
        elif pop[0].fitness < settings.optimum:
            converge = True
            if settings.verbose:
                print("Gnowee: fitness below optimum.")

        timeline[-1].generation = gen

    best_x = np.array(pop[0].variables, dtype=float)
    best_f = float(pop[0].fitness)
    return best_x, best_f, timeline
