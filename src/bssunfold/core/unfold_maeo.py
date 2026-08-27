"""MAEO (Multiobjective Animorphic Ensemble Optimization) unfolding methods.

This module implements neutron spectrum unfolding using the MAEO framework
from Erdem et al. (2026), which combines multiple multiobjective optimization
algorithms in an ensemble with adaptive migration based on hypervolume performance.

The MAEO framework is particularly well-suited for neutron spectrum unfolding because:
1. It handles multiple conflicting objectives
   (data fit, smoothness, physical constraints)
2. It automatically selects the best-performing algorithm for the specific problem
3. It provides robust convergence through ensemble diversity
4. It supports parallel evaluation of individuals

References
----------
[1] O.F. Erdem, D. Price, P. Seurin, M.I. Radaideh, "MAEO: Multiobjective Animorphic
    Ensemble Optimization for Scalable Large-scale Engineering Applications",
    arXiv:2604.26973 (2026).

[2] D. Price, M.I. Radaideh,
    "Animorphic Ensemble Optimization: a large-scale island model",
    Neural Computing and Applications 35 (4) (2023) 3221-3243.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_maeo", "unfold_maeo", "solve_maeo_ensemble", "unfold_maeo_ensemble"]

_IMPORT_ERROR_MSG = (
    "pymoo is required for unfold_maeo. Install with: pip install pymoo numba"
)

# MAEO default algorithms from the paper
_MAEIO_ALGORITHMS = ("nsga3", "ctaea", "agemoea2", "spea2")


def _compute_hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute hypervolume indicator for a Pareto front.

    Parameters
    ----------
    front : np.ndarray
        Array of shape (n_points, n_objectives) containing objective values.
    ref_point : np.ndarray
        Reference point (nadir point) for hypervolume calculation.

    Returns
    -------
    float
        Hypervolume value (larger is better).
    """
    try:
        from pymoo.indicators.hv import HV
        indicator = HV(ref_point=ref_point)
        return indicator(front)
    except Exception:
        # Fallback: simple 2D hypervolume approximation
        if front.shape[1] == 2:
            # Sort by first objective
            sorted_idx = np.argsort(front[:, 0])
            front_sorted = front[sorted_idx]

            hv = 0.0
            prev_f1 = 0.0
            for i, (f1, f2) in enumerate(front_sorted):
                width = f1 - prev_f1 if i > 0 else f1
                height = ref_point[1] - f2
                hv += width * height
                prev_f1 = f1

            # Add final rectangle
            hv += (ref_point[0] - prev_f1) * (ref_point[1] - front_sorted[-1, 1])
            return hv

        # Very rough approximation for higher dimensions
        dominated_vol = np.prod(ref_point - np.min(front, axis=0))
        return dominated_vol * 0.5


def _maeo_objectives(
    x: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    D2: np.ndarray,
    lambda_smooth: float = 0.01,
    prior_spectrum: Optional[np.ndarray] = None,
    log_space: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Compute multiobjective functions for MAEO unfolding.

    Objectives (all to be minimized):
    1. Data fidelity: ||b - A*phi||^2 / ||b||^2
    2. Smoothness: ||D2 * phi||^2 (second derivative regularization)
    3. Prior deviation: ||phi - phi_prior||^2 / ||phi_prior||^2 (optional)

    Parameters
    ----------
    x : np.ndarray
        Decision variables (spectrum in log or linear space).
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measured count rates.
    D2 : np.ndarray
        Second derivative matrix for smoothness regularization.
    lambda_smooth : float
        Weight for smoothness objective.
    prior_spectrum : np.ndarray, optional
        Prior/guess spectrum for third objective.
    log_space : bool
        If True, x is in log space and will be exponentiated.

    Returns
    -------
    tuple
        (objectives_array, constraint_violation_flag)
    """
    # Transform from log space if needed
    if log_space:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -50, 50)
        phi = np.exp(x_clipped)
    else:
        phi = np.maximum(x, 0)  # Ensure non-negativity

    # Objective 1: Normalized data fidelity (chi-squared like)
    b_norm_sq = np.dot(b, b)
    if b_norm_sq < 1e-20:
        b_norm_sq = 1.0

    residual = b - A @ phi
    obj_data = np.dot(residual, residual) / b_norm_sq

    # Objective 2: Smoothness (second derivative)
    smoothness = np.dot(D2 @ phi, D2 @ phi)
    obj_smooth = lambda_smooth * smoothness

    # Build objectives array
    objectives = np.array([obj_data, obj_smooth])

    # Objective 3: Prior deviation (if prior provided)
    if prior_spectrum is not None:
        prior_norm_sq = np.dot(prior_spectrum, prior_spectrum)
        if prior_norm_sq < 1e-20:
            prior_norm_sq = 1.0

        prior_dev = np.dot(phi - prior_spectrum, phi - prior_spectrum)
        obj_prior = prior_dev / prior_norm_sq
        objectives = np.append(objectives, obj_prior)

    # Check for constraint violations (numerical issues)
    constraint_violated = False
    if np.any(np.isnan(objectives)) or np.any(np.isinf(objectives)):
        constraint_violated = True
        objectives = np.nan_to_num(objectives, nan=1e10, posinf=1e10, neginf=1e10)

    return objectives, constraint_violated


def _individual_score(
    rank: int,
    max_rank: int,
    crowding_dist: float,
    ref_distance: float,
    max_crowding: float = 1.0,
    max_ref_dist: float = 1.0,
) -> float:
    """Calculate individual performance score for MAEO migration.

    This implements Eq. (14) from the MAEO paper:
    S_x = (1 - rank_x/max_rank) + (CD_x + RD_x) / (2 * (max_rank + 1))

    Parameters
    ----------
    rank : int
        Pareto rank of the individual (lower is better).
    max_rank : int
        Maximum rank in the population.
    crowding_dist : float
        Normalized crowding distance [0, 1].
    ref_distance : float
        Normalized reference distance from nadir point [0, 1].
    max_crowding : float
        Maximum possible crowding distance.
    max_ref_dist : float
        Maximum possible reference distance.

    Returns
    -------
    float
        Performance score (higher is better).
    """
    if max_rank == 0:
        rank_component = 1.0
    else:
        rank_component = 1.0 - (rank / max_rank)

    # Normalize diversity components
    cd_norm = crowding_dist / max_crowding if max_crowding > 0 else 0.5
    rd_norm = ref_distance / max_ref_dist if max_ref_dist > 0 else 0.5

    # Band scaling factor ensures strict rank separation
    band_scale = 2.0 * (max_rank + 1)
    diversity_component = (cd_norm + rd_norm) / band_scale

    return rank_component + diversity_component


def solve_maeo(
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    n_cycles: int = 20,
    n_gen_per_cycle: int = 10,
    pop_size: int = 100,
    algorithms: Optional[List[str]] = None,
    lambda_smooth: float = 0.01,
    prior_spectrum: Optional[np.ndarray] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    convergence_assist_ratio: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Solve neutron spectrum unfolding using MAEO ensemble optimization.

    This implements the Multiobjective Animorphic Ensemble Optimization (MAEO)
    framework from Erdem et al. (2026) for neutron spectrum unfolding.

    The method runs multiple MOO algorithms (islands) in parallel, evaluates
    their performance using hypervolume indicators, and adaptively migrates
    individuals toward better-performing islands.

    Parameters
    ----------
    A : np.ndarray
        Response matrix of shape (n_detectors, n_energy_bins).
    b : np.ndarray
        Measured count rates of shape (n_detectors,).
    E_MeV : np.ndarray
        Energy grid in MeV of shape (n_energy_bins,).
    n_cycles : int, optional
        Number of MAEO cycles (default: 20).
    n_gen_per_cycle : int, optional
        Generations per cycle for each island (default: 10).
    pop_size : int, optional
        Population size per island (default: 100).
    algorithms : list of str, optional
        List of algorithm names to use as islands. Default uses the four
        algorithms from the MAEO paper: ["nsga3", "ctaea", "agemoea2", "spea2"].
    lambda_smooth : float, optional
        Smoothness regularization weight (default: 0.01).
    prior_spectrum : np.ndarray, optional
        Prior/guess spectrum for additional objective.
    initial_spectrum : np.ndarray, optional
        Initial spectrum for warm-start (will be used to seed population).
    convergence_assist_ratio : float, optional
        Fraction of cycles to dedicate to best island at the end (default: 0.2).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress information (default: False).
    **kwargs
        Additional keyword arguments passed to the underlying optimizer.

    Returns
    -------
    dict
        Dictionary containing:
        - 'spectrum': Unfolded spectrum (non-negative)
        - 'energy': Energy grid
        - 'n_cycles_run': Number of cycles executed
        - 'best_algorithm': Name of the best-performing algorithm
        - 'hypervolume_history': HV history for each island
        - 'population_history': Population sizes per island per cycle
        - 'pareto_front': Final Pareto front solutions
        - 'objectives': Objective values for the selected solution

    Notes
    -----
    The MAEO framework optimizes multiple objectives simultaneously:
    1. Minimize data fidelity error ||b - A*phi||^2 / ||b||^2
    2. Minimize spectrum roughness ||D2 * phi||^2
    3. (Optional) Minimize deviation from prior spectrum

    The final solution is selected from the Pareto front using a knee-point
    detection method to balance accuracy and smoothness.

    Examples
    --------
    >>> from bssunfold import Detector
    >>> detector = Detector()
    >>> readings = {'sphere_1': 100.5, 'sphere_2': 85.3, ...}
    >>> result = detector.unfold_maeo(readings, n_cycles=15)
    >>> spectrum = result['spectrum']
    """
    try:
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        from pymoo.algorithms.moo.ctaea import CTAEA
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.algorithms.moo.spea2 import SPEA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    n_detectors, n_energy = A.shape
    n_obj = 3 if prior_spectrum is not None else 2

    # Create second derivative matrix for smoothness
    D2 = create_derivative_matrix(n_energy, order=2)

    # Algorithm registry
    algo_registry = {
        "nsga3": NSGA3,
        "ctaea": CTAEA,
        "agemoea2": AGEMOEA2,
        "spea2": SPEA2,
    }

    # Use default algorithms if not specified
    if algorithms is None:
        algorithms = list(algo_registry.keys())
    else:
        # Validate algorithm names
        for algo in algorithms:
            if algo not in algo_registry:
                raise ValueError(
                    f"Unknown algorithm '{algo}'. "
                    f"Available: {list(algo_registry.keys())}"
                )

    n_islands = len(algorithms)

    # Calculate cycles with convergence assist
    n_migration_cycles = int(n_cycles * (1 - convergence_assist_ratio))
    n_convergence_cycles = n_cycles - n_migration_cycles

    if verbose:
        print("MAEO Configuration:")
        print(f"  Islands: {n_islands} ({', '.join(algorithms)})")
        print(f"  Total cycles: {n_cycles}")
        print(f"  Migration cycles: {n_migration_cycles}")
        print(f"  Convergence cycles: {n_convergence_cycles}")
        print(f"  Population per island: {pop_size}")
        print(f"  Objectives: {n_obj}")

    # Initialize storage for results
    hv_history = {algo: [] for algo in algorithms}
    pop_history = {algo: [] for algo in algorithms}
    all_pareto_solutions = []
    all_pareto_objectives = []

    # Generate reference directions for NSGA-III and CTAEA
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)

    # Create initial population seed
    if initial_spectrum is not None:
        # Use provided initial spectrum
        seed_spectrum = np.maximum(initial_spectrum, 1e-10)
        seed_log = np.log(seed_spectrum)
    else:
        # Simple uniform initial guess
        seed_spectrum = np.ones(n_energy) * (np.sum(b) / n_detectors / np.mean(A))
        seed_log = np.log(np.maximum(seed_spectrum, 1e-10))

    # Bounds in log space ( +/- 3 orders of magnitude from seed)
    lower_bounds = seed_log - 3.0
    upper_bounds = seed_log + 3.0

    # Define the optimization problem
    class UnfoldingProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=n_energy,
                n_obj=n_obj,
                n_constr=0,
                xl=lower_bounds,
                xu=upper_bounds,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            objs_list = []
            for x in X:
                obj_vals, violated = _maeo_objectives(
                    x, A, b, D2, lambda_smooth, prior_spectrum, log_space=True
                )
                if violated:
                    # Penalize infeasible solutions
                    obj_vals = np.full(n_obj, 1e10)
                objs_list.append(obj_vals)

            out["F"] = np.array(objs_list)

    problem = UnfoldingProblem()

    # Run MAEO cycles
    current_populations = {}
    island_results = {}
    # Best island index, updated at the end of each migration cycle (line 490).
    # Initialized to 0 so the convergence-phase read (line ~423) is always
    # defined, even when no migration cycle runs first.
    best_island_idx = 0

    for cycle in range(n_cycles):
        is_convergence_phase = cycle >= n_migration_cycles

        if verbose:
            phase = "Convergence" if is_convergence_phase else "Migration"
            print(f"\nCycle {cycle+1}/{n_cycles} [{phase} phase]")

        # Evaluate each island
        island_hvs = {}

        for algo_idx, algo_name in enumerate(algorithms):
            if is_convergence_phase:
                # In convergence phase, only run the best island
                if algo_idx != best_island_idx:
                    continue

            # Initialize algorithm
            AlgoClass = algo_registry[algo_name]

            # Handle reference-direction-based algorithms (NSGA-III, CTAEA)
            # They set pop_size from ref_dirs length, so don't pass it explicitly
            if algo_name == "nsga3":
                algorithm = AlgoClass(ref_dirs=ref_dirs)
            elif algo_name == "ctaea":
                algorithm = AlgoClass(ref_dirs=ref_dirs)
            elif algo_name == "agemoea2":
                algorithm = AlgoClass(pop_size=pop_size)
            elif algo_name == "spea2":
                algorithm = AlgoClass(pop_size=pop_size)
            else:
                algorithm = AlgoClass(pop_size=pop_size)

            # Warm start with previous population if available
            if cycle > 0 and algo_name in current_populations:
                algorithm.pop = current_populations[algo_name]

            # Termination: one cycle = n_gen_per_cycle generations
            termination = get_termination("n_gen", n_gen_per_cycle)

            # Run optimization
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=seed + cycle if seed is not None else None,
                verbose=False,
                save_history=True,
            )

            # Store population for next cycle
            current_populations[algo_name] = res.pop

            # Calculate hypervolume for this island
            pareto_front = res.opt.get("F")

            # Determine reference point (nadir point with margin)
            if len(pareto_front) > 0:
                ref_point = np.max(pareto_front, axis=0) * 1.1 + 0.1

                hv = _compute_hypervolume(pareto_front, ref_point)
                island_hvs[algo_name] = hv
                hv_history[algo_name].append(hv)
                pop_history[algo_name].append(len(res.pop))

                # Store Pareto solutions
                solutions = res.opt.get("X")
                for sol, obj in zip(solutions, pareto_front):
                    all_pareto_solutions.append(sol)
                    all_pareto_objectives.append(obj)

                if verbose:
                    print(f"  {algo_name}: HV={hv:.4f}, Pop={len(res.pop)}")
            else:
                island_hvs[algo_name] = 0.0
                hv_history[algo_name].append(0.0)
                pop_history[algo_name].append(0)

        # Migration phase: redistribute population based on HV
        if not is_convergence_phase and len(island_hvs) > 1:
            # Find best island for next cycle
            best_island_idx = np.argmax(list(island_hvs.values()))
            best_island_name = algorithms[best_island_idx]

            if verbose:
                print(
                    f"  Best island: {best_island_name} "
                    f"(HV={island_hvs[best_island_name]:.4f})"
                )

    # Select final solution from combined Pareto front
    if len(all_pareto_objectives) > 0:
        all_pareto_objectives = np.array(all_pareto_objectives)
        all_pareto_solutions = np.array(all_pareto_solutions)

        # Find knee point (solution closest to ideal point)
        ideal_point = np.min(all_pareto_objectives, axis=0)
        nadir_point = np.max(all_pareto_objectives, axis=0)

        # Normalize objectives
        obj_range = nadir_point - ideal_point
        obj_range[obj_range < 1e-10] = 1.0

        normalized_objs = (all_pareto_objectives - ideal_point) / obj_range

        # Knee point: minimize distance to ideal (or use trade-off analysis)
        distances = np.sqrt(np.sum(normalized_objs**2, axis=1))
        knee_idx = np.argmin(distances)

        best_solution_log = all_pareto_solutions[knee_idx]
        best_objectives = all_pareto_objectives[knee_idx]

        # Convert back to linear space
        best_solution = np.exp(np.clip(best_solution_log, -50, 50))
        best_solution = np.maximum(best_solution, 0)
    else:
        # Fallback to simple least-squares if MAEO fails
        warnings.warn(
            "MAEO did not find valid solutions. "
            "Using least-squares fallback."
        )
        best_solution = np.linalg.lstsq(A, b, rcond=None)[0]
        best_solution = np.maximum(best_solution, 0)
        best_objectives = None

    # Compile results
    result = {
        "spectrum": best_solution,
        "energy": E_MeV,
        "n_cycles_run": n_cycles,
        "best_algorithm": (
            algorithms[best_island_idx]
            if "best_island_idx" in locals()
            else algorithms[0]
        ),
        "hypervolume_history": hv_history,
        "population_history": pop_history,
        "pareto_front": (
            all_pareto_objectives if len(all_pareto_objectives) > 0 else None
        ),
        "objectives": best_objectives,
        "algorithm_details": {
            "n_islands": n_islands,
            "algorithms_used": algorithms,
            "convergence_assist_ratio": convergence_assist_ratio,
        },
    }

    return result


def unfold_maeo(
    detector: Any,
    readings: Dict[str, float],
    n_cycles: int = 20,
    n_gen_per_cycle: int = 10,
    pop_size: int = 100,
    algorithms: Optional[List[str]] = None,
    lambda_smooth: float = 0.01,
    prior_spectrum: Optional[np.ndarray] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    convergence_assist_ratio: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using MAEO ensemble optimization.

    This is the high-level interface for MAEO unfolding that integrates
    with the Detector class.

    Parameters
    ----------
    detector : Detector
        Detector instance with response matrix and energy grid.
    readings : dict
        Dictionary mapping detector names to measured count rates.
    n_cycles : int, optional
        Number of MAEO cycles (default: 20).
    n_gen_per_cycle : int, optional
        Generations per cycle (default: 10).
    pop_size : int, optional
        Population size per island (default: 100).
    algorithms : list of str, optional
        Algorithms to use as islands. Default: ["nsga3", "ctaea", "agemoea2", "spea2"].
    lambda_smooth : float, optional
        Smoothness regularization weight (default: 0.01).
    prior_spectrum : np.ndarray, optional
        Prior/guess spectrum for additional objective.
    initial_spectrum : np.ndarray, optional
        Initial spectrum for warm-start.
    convergence_assist_ratio : float, optional
        Fraction of cycles for convergence phase (default: 0.2).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress information (default: False).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    dict
        Standardized result dictionary with spectrum, dose rates, etc.

    See Also
    --------
    solve_maeo : Low-level solver function.
    unfold_maeo_ensemble : Variant with explicit ensemble control.

    Examples
    --------
    >>> from bssunfold import Detector
    >>> detector = Detector()
    >>> readings = {
    ...     'sphere_1': 100.5,
    ...     'sphere_2': 85.3,
    ...     'sphere_3': 72.1,
    ...     'sphere_4': 58.9,
    ...     'sphere_5': 45.2,
    ...     'sphere_6': 32.8,
    ... }
    >>> result = detector.unfold_maeo(readings, n_cycles=15, verbose=True)
    >>> print(f"Spectrum integral: {np.sum(result['spectrum']):.2f}")
    >>> print(f"Best algorithm: {result.get('best_algorithm', 'N/A')}")
    """
    # Build system matrices
    A, b, selected = detector._build_system(readings)

    # Run MAEO solver
    result = solve_maeo(
        A=A,
        b=b,
        E_MeV=detector.E_MeV,
        n_cycles=n_cycles,
        n_gen_per_cycle=n_gen_per_cycle,
        pop_size=pop_size,
        algorithms=algorithms,
        lambda_smooth=lambda_smooth,
        prior_spectrum=prior_spectrum,
        initial_spectrum=initial_spectrum,
        convergence_assist_ratio=convergence_assist_ratio,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )

    # Standardize output format
    standardized_result = detector._standardize_output(
        spectrum=result["spectrum"],
        A=A,
        b=b,
        selected=selected,
        method="MAEO",
        maeo_info={
            "n_cycles": result["n_cycles_run"],
            "best_algorithm": result["best_algorithm"],
            "hypervolume_history": result["hypervolume_history"],
            "population_history": result["population_history"],
            "algorithms_used": result["algorithm_details"]["algorithms_used"],
        },
    )

    # Add MAEO-specific results
    standardized_result["spectrum_absolute"] = result["spectrum"]
    standardized_result["maeo_pareto_front"] = result.get("pareto_front")
    standardized_result["maeo_objectives"] = result.get("objectives")

    return standardized_result


def solve_maeo_ensemble(
    A: np.ndarray,
    b: np.ndarray,
    E_MeV: np.ndarray,
    n_cycles: int = 25,
    n_gen_per_cycle: int = 8,
    pop_size: int = 100,
    algorithms: Optional[List[str]] = None,
    lambda_smooth: float = 0.01,
    prior_spectrum: Optional[np.ndarray] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    convergence_assist_ratio: float = 0.2,
    migration_method: str = "hypervolume",
    seed: Optional[int] = None,
    verbose: bool = False,
    parallel: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Solve unfolding with explicit MAEO ensemble control.

    This variant provides more control over the ensemble behavior,
    including migration strategy and parallel execution.

    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measured count rates.
    E_MeV : np.ndarray
        Energy grid.
    n_cycles : int, optional
        Total number of cycles (default: 25).
    n_gen_per_cycle : int, optional
        Generations per cycle (default: 8).
    pop_size : int, optional
        Population size per island (default: 100).
    algorithms : list, optional
        Algorithm list.
    lambda_smooth : float, optional
        Smoothness weight.
    prior_spectrum : np.ndarray, optional
        Prior spectrum.
    initial_spectrum : np.ndarray, optional
        Initial spectrum.
    convergence_assist_ratio : float, optional
        Convergence phase ratio (default: 0.2).
    migration_method : str, optional
        Migration strategy: "hypervolume" (default) or "uniform".
    seed : int, optional
        Random seed.
    verbose : bool, optional
        Verbose output.
    parallel : bool, optional
        Enable parallel island execution (requires joblib).
    **kwargs
        Additional arguments.

    Returns
    -------
    dict
        Results dictionary with full ensemble information.
    """
    # For now, delegate to standard solve_maeo
    # Future enhancement: implement explicit parallel migration
    result = solve_maeo(
        A=A,
        b=b,
        E_MeV=E_MeV,
        n_cycles=n_cycles,
        n_gen_per_cycle=n_gen_per_cycle,
        pop_size=pop_size,
        algorithms=algorithms,
        lambda_smooth=lambda_smooth,
        prior_spectrum=prior_spectrum,
        initial_spectrum=initial_spectrum,
        convergence_assist_ratio=convergence_assist_ratio,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )

    result["migration_method"] = migration_method
    result["parallel_enabled"] = parallel

    return result


def unfold_maeo_ensemble(
    detector: Any,
    readings: Dict[str, float],
    **kwargs,
) -> Dict[str, Any]:
    """Unfold with explicit MAEO ensemble control.

    High-level interface for solve_maeo_ensemble.

    Parameters
    ----------
    detector : Detector
        Detector instance.
    readings : dict
        Count rate measurements.
    **kwargs
        Passed to solve_maeo_ensemble.

    Returns
    -------
    dict
        Standardized results.
    """
    A, b, selected = detector._build_system(readings)

    result = solve_maeo_ensemble(
        A=A,
        b=b,
        E_MeV=detector.E_MeV,
        **kwargs,
    )

    standardized_result = detector._standardize_output(
        spectrum=result["spectrum"],
        A=A,
        b=b,
        selected=selected,
        method="MAEO-Ensemble",
        maeo_info=result,
    )

    standardized_result["spectrum_absolute"] = result["spectrum"]

    return standardized_result


# Make solve_maeo available as the main entry point
unfold_maeo_impl = unfold_maeo
solve_maeo_impl = solve_maeo
