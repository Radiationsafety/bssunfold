"""Gnowee-based unfolding method for neutron spectrum reconstruction.

This module provides a Bonner-sphere-spectrum unfolding method built on the
Gnowee hybrid metaheuristic optimizer
(https://github.com/SlaybaughLab/Gnowee, Bevins & Parsons, UC Berkeley /
Slaybaugh Lab).  Gnowee combines Lévy flights (Cuckoo Search), golden-ratio
crossover (Modified Cuckoo Search / Differential Evolution), scatter search
(Egea 2009) and DE-style mutation in an elitist population with
Metropolis-Hastings acceptance and stall-driven restarts.

The numerical strategy mirrors the proven approach used by
:func:`bssunfold.unfold_genetic`:

* search in **log space** (``y = log(x)``) so positivity is enforced and the
  wide dynamic range of neutron spectra is handled naturally;
* seed the population with a **Landweber warm-start solution** (or the
  user-provided ``initial_spectrum``);
* bound the search to ``log(seed) ± half_range`` decades;
* minimise a **scale-consistent objective** in which the relative residual,
  Tikhonov regularisation and second-difference smoothness terms are all
  dimensionless and comparable (this prevents the optimizer from inflating
  ``x`` to artificially lower the chi-squared term).

Post-processing smoothers (Gaussian / multiplicative bias correction /
second-difference) and Monte-Carlo uncertainty estimation are provided
through the standard :func:`run_unfolding` workflow.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ..logging_config import get_logger
from ._base_unfolder import run_unfolding
from ._gnowee import GnoweeSettings, run_gnowee
from ._matrix_utils import create_derivative_matrix

__all__ = ["solve_gnowee", "unfold_gnowee"]

logger = get_logger("detector")


# --------------------------------------------------------------------------- #
# Helpers (kept consistent with unfold_genetic so the two methods can be      #
# benchmarked against each other)                                              #
# --------------------------------------------------------------------------- #
def _build_seed(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
    """Build a warm-start seed spectrum.

    If the user supplies a non-trivial ``initial_spectrum`` it is used
    directly; otherwise a short Landweber iteration produces a smooth,
    physically-plausible starting point — important because the unfolding
    problem is severely under-determined (many more energy bins than
    detectors) and a purely random population converges to a noisy spectrum.
    """
    n = A.shape[1]
    if x0 is not None and np.any(np.asarray(x0, dtype=float) > 0):
        return np.maximum(np.asarray(x0, dtype=float), 1e-12)
    try:
        from .unfold_landweber import solve_landweber

        lw, _, _ = solve_landweber(A, b, np.zeros(n), max_iterations=500)
        seed = np.maximum(np.asarray(lw, dtype=float), 1e-12)
    except Exception:
        A_fro = float(np.linalg.norm(A)) or 1.0
        x_scale = float(np.linalg.norm(b)) / A_fro
        seed = np.full(n, max(x_scale / np.sqrt(n), 1e-12))
    return seed


def _build_log_bounds(seed: np.ndarray, half_range: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lb, ub)`` in log space centred on the seed."""
    y0 = np.log(np.maximum(np.asarray(seed, dtype=float), 1e-300))
    span = half_range * np.log(10.0)
    return y0 - span, y0 + span


def _build_fitness(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    norm: int,
    L,
    smoothness_weight: float,
    entropy_weight: float,
) -> Callable[[np.ndarray], float]:
    """Build the scale-consistent unfolding objective ``f(y)``.

    All terms are dimensionless and on the same scale so the optimizer
    cannot trivially drive one term to zero at the expense of the others:

    ``f(y) = ||b - A exp(y)||^2 / ||b||^2
           + alpha * ||exp(y)||_norm / x_scale
           + smoothness_weight * ||L exp(y)||^2 / x_scale^2
           - entropy_weight * H(exp(y))``
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


# --------------------------------------------------------------------------- #
# Core solver                                                                  #
# --------------------------------------------------------------------------- #
def solve_gnowee(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    population: int = 25,
    max_gens: int = 200,
    max_fevals: int = 5_000,
    stall_limit: int = 200,
    conv_tol: float = 1e-6,
    opt_conv_tol: float = 1e-2,
    frac_elite: float = 0.2,
    frac_levy: float = 1.0,
    frac_mutation: float = 0.2,
    alpha_levy: float = 1.5,
    gamma_levy: float = 1.0,
    n_levy: int = 1,
    scaling_factor: float = 10.0,
    init_sampling: str = "lhc",
    regularization: float = 1e-2,
    norm: int = 2,
    smoothness_order: int = 2,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    half_range: float = 2.0,
    random_state: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, int, bool, dict[str, Any]]:
    """Solve the unfolding problem with the Gnowee metaheuristic.

    The optimizer searches in log space (``y`` with ``x = exp(y)``)
    seeded with a Landweber warm-start (or the provided ``initial_spectrum``),
    bounded to ``log(seed) ± half_range`` decades, with a scale-consistent
    objective.

    Parameters
    ----------
    A : np.ndarray
        Response matrix ``(m, n)``.
    b : np.ndarray
        Measurement vector ``(m,)``.
    x0 : np.ndarray, optional
        Initial spectrum guess. If ``None`` or all-zero, a Landweber
        warm-start is used to seed the population.
    population : int, optional
        Population size (default: 25, Gnowee's recommended value).
    max_gens : int, optional
        Maximum number of generations (default: 200).
    max_fevals : int, optional
        Maximum number of fitness evaluations (default: 5_000).
    stall_limit : int, optional
        Evaluations without improvement before termination (default: 200).
    conv_tol : float, optional
        Relative improvement in best fitness required to extend the
        timeline (default: 1e-6).
    opt_conv_tol : float, optional
        Tolerance on the optimum value for fitness convergence
        (default: 1e-2).  Since the global optimum is unknown for BSS
        unfolding, the default is loose.
    frac_elite : float, optional
        Elite fraction (crossover / scatter search), default 0.2.
    frac_levy : float, optional
        Lévy flight fraction, default 1.0.
    frac_mutation : float, optional
        Mutation discovery probability, default 0.2.
    alpha_levy : float, optional
        Lévy exponent, default 1.5.
    gamma_levy : float, optional
        Lévy scale, default 1.0.
    n_levy : int, optional
        Number of independent Lévy samples, default 1.
    scaling_factor : float, optional
        Lévy step scale, default 10.0.
    init_sampling : str, optional
        Initial population sampler: ``'lhc'`` or ``'random'``, default ``'lhc'``.
    regularization : float, optional
        Tikhonov regularisation weight (default 1e-2).
    norm : int, optional
        Norm for the regularisation term (1 or 2), default 2.
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 2.
    smoothness_weight : float, optional
        Weight for the smoothness term, default 1.0.
    entropy_weight : float, optional
        Weight of the negative Shannon entropy objective (0 disables it).
    half_range : float, optional
        Half-width of the log-space search bounds in decades around the
        seed, default 2.0.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print Gnowee progress every 10 generations.

    Returns
    -------
    tuple[np.ndarray, int, bool, dict]
        ``(spectrum, n_evals, converged, diagnostics)`` where ``spectrum``
        is the unfolded spectrum in linear space, ``n_evals`` is the total
        number of objective evaluations, ``converged`` indicates whether
        Gnowee hit a fitness / stall convergence criterion (rather than the
        max-iteration cap), and ``diagnostics`` carries extra metadata
        (best fitness, generations, timeline length).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    if norm not in (1, 2):
        raise ValueError(f"Unsupported norm type: {norm}. Use 1 or 2.")
    if smoothness_order not in (0, 1, 2):
        raise ValueError(
            f"Unsupported smoothness order: {smoothness_order}. Use 0, 1 or 2."
        )
    if init_sampling.lower() not in ("lhc", "lhs", "random"):
        raise ValueError(
            f"Unsupported init_sampling: {init_sampling}. Use 'lhc' or 'random'."
        )

    L = None
    if smoothness_order in (1, 2):
        L = create_derivative_matrix(n, smoothness_order)

    fitness = _build_fitness(
        A, b, regularization, norm, L, smoothness_weight, entropy_weight
    )
    seed = _build_seed(A, b, x0)
    lb, ub = _build_log_bounds(seed, half_range)

    rng = np.random.default_rng(random_state)

    settings = GnoweeSettings(
        population=int(population),
        init_sampling="lhc" if init_sampling.lower() in ("lhc", "lhs") else "random",
        frac_mutation=float(frac_mutation),
        frac_elite=float(frac_elite),
        frac_levy=float(frac_levy),
        alpha=float(alpha_levy),
        gamma=float(gamma_levy),
        n=int(n_levy),
        scaling_factor=float(scaling_factor),
        max_gens=int(max_gens),
        max_fevals=int(max_fevals),
        conv_tol=float(conv_tol),
        stall_limit=int(stall_limit),
        opt_conv_tol=float(opt_conv_tol),
        verbose=bool(verbose),
    )

    best_y, best_f, timeline = run_gnowee(
        lb=lb, ub=ub, objective=fitness, settings=settings, rng=rng,
        seed_solution=np.log(np.maximum(seed, 1e-300)),
    )
    spectrum = np.maximum(np.exp(best_y), 0.0)

    n_evals = timeline[-1].evaluations if timeline else 0
    # "Converged" means we did not hit the max_fevals / max_gens caps
    hit_feval_cap = n_evals >= settings.max_fevals
    hit_gen_cap = (timeline[-1].generation if timeline else 0) >= settings.max_gens
    converged = not (hit_feval_cap or hit_gen_cap)

    diagnostics = {
        "best_fitness": float(best_f),
        "generations": int(timeline[-1].generation) if timeline else 0,
        "evaluations": int(n_evals),
        "timeline_len": len(timeline),
    }
    return spectrum, int(n_evals), converged, diagnostics


# --------------------------------------------------------------------------- #
# Detector-level wrapper                                                      #
# --------------------------------------------------------------------------- #
def unfold_gnowee(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    population: int = 25,
    max_gens: int = 200,
    max_fevals: int = 5_000,
    stall_limit: int = 200,
    conv_tol: float = 1e-6,
    opt_conv_tol: float = 1e-2,
    frac_elite: float = 0.2,
    frac_levy: float = 1.0,
    frac_mutation: float = 0.2,
    alpha_levy: float = 1.5,
    gamma_levy: float = 1.0,
    n_levy: int = 1,
    scaling_factor: float = 10.0,
    init_sampling: str = "lhc",
    regularization: float = 1e-2,
    norm: int = 2,
    smoothness_order: int = 2,
    smoothness_weight: float = 1.0,
    entropy_weight: float = 0.0,
    half_range: float = 2.0,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Unfold a neutron spectrum using the Gnowee metaheuristic optimizer.

    The optimizer searches in log space seeded with a Landweber warm-start
    solution (or the provided ``initial_spectrum``), bounded to
    ``log(seed) ± half_range`` decades, with a scale-consistent objective
    combining the relative L2 residual, Tikhonov regularisation,
    second-difference smoothness and (optionally) negative Shannon entropy.

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
        Initial spectrum guess. If ``None``, a Landweber warm-start solution
        is used to seed the population.
    population : int, optional
        Population size, default 25.
    max_gens : int, optional
        Maximum generations, default 200.
    max_fevals : int, optional
        Maximum fitness evaluations, default 5_000.
    stall_limit : int, optional
        Stall-based termination threshold (evaluations), default 200.
    conv_tol : float, optional
        Relative improvement tolerance for timeline extension, default 1e-6.
    opt_conv_tol : float, optional
        Tolerance on the optimum value for fitness convergence, default 1e-2.
    frac_elite : float, optional
        Elite fraction for crossover/scatter-search, default 0.2.
    frac_levy : float, optional
        Lévy flight fraction, default 1.0.
    frac_mutation : float, optional
        Mutation discovery probability, default 0.2.
    alpha_levy : float, optional
        Lévy exponent, default 1.5.
    gamma_levy : float, optional
        Lévy scale, default 1.0.
    n_levy : int, optional
        Independent Lévy samples, default 1.
    scaling_factor : float, optional
        Lévy step scale, default 10.0.
    init_sampling : str, optional
        Initial sampler: ``'lhc'`` or ``'random'``, default ``'lhc'``.
    regularization : float, optional
        Tikhonov regularisation weight, default 1e-2.
    norm : int, optional
        Norm for the regularisation term (1 or 2), default 2.
    smoothness_order : int, optional
        Smoothness penalty order (0, 1 or 2), default 2.
    smoothness_weight : float, optional
        Weight for the smoothness term, default 1.0.
    entropy_weight : float, optional
        Weight of the negative Shannon-entropy objective (0 disables it).
    half_range : float, optional
        Half-width of the log-space search bounds in decades, default 2.0.
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
    verbose : bool, optional
        Print Gnowee progress.

    Returns
    -------
    dict[str, Any]
        Unfolding results including spectrum, residuals, and metadata.
    """
    x0_default = np.zeros(n_energy_bins)

    # Wrap solve_gnowee so that run_unfolding can pass x0 through kwargs.
    # solve_gnowee returns (spectrum, n_evals, converged, diagnostics); the
    # base unfolder only inspects the first three positions of a tuple.
    def _solve(A, b, **kwargs):
        x0_in = kwargs.pop("x0", None)
        spectrum, n_evals, converged, diag = solve_gnowee(
            A, b, x0=x0_in,
            population=population,
            max_gens=max_gens,
            max_fevals=max_fevals,
            stall_limit=stall_limit,
            conv_tol=conv_tol,
            opt_conv_tol=opt_conv_tol,
            frac_elite=frac_elite,
            frac_levy=frac_levy,
            frac_mutation=frac_mutation,
            alpha_levy=alpha_levy,
            gamma_levy=gamma_levy,
            n_levy=n_levy,
            scaling_factor=scaling_factor,
            init_sampling=init_sampling,
            regularization=regularization,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            entropy_weight=entropy_weight,
            half_range=half_range,
            random_state=random_state,
            verbose=verbose,
        )
        # Stash the diagnostics on the function so the wrapper below can
        # retrieve them — solve_func is called once per Monte-Carlo sample
        # so we keep only the latest.
        _solve.last_diagnostics = diag
        return spectrum, n_evals, converged

    _solve.last_diagnostics = {}

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
        solve_func=_solve,
        solve_kwargs={},
        method_name="Gnowee",
        extra_output={
            "population": population,
            "max_gens": max_gens,
            "max_fevals": max_fevals,
            "stall_limit": stall_limit,
            "frac_elite": frac_elite,
            "frac_levy": frac_levy,
            "frac_mutation": frac_mutation,
            "alpha_levy": alpha_levy,
            "gamma_levy": gamma_levy,
            "n_levy": n_levy,
            "scaling_factor": scaling_factor,
            "init_sampling": init_sampling,
            "regularization": regularization,
            "norm": norm,
            "smoothness_order": smoothness_order,
            "smoothness_weight": smoothness_weight,
            "entropy_weight": entropy_weight,
            "half_range": half_range,
            "best_fitness": getattr(_solve, "last_diagnostics", {}).get("best_fitness"),
            "evaluations": getattr(_solve, "last_diagnostics", {}).get("evaluations"),
            "generations": getattr(_solve, "last_diagnostics", {}).get("generations"),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
