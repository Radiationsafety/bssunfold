"""Method metadata and tier classification for BSSUnfold.

This module provides metadata about all unfolding methods including:
- Tier classification (stable/experimental/deprecated)
- Numba JIT acceleration status
- Recommended use cases
- Known limitations

Tiers:
------
- Tier 1 (stable): Production-ready for radiation protection applications.
  Requirements: explicit noise covariance, positivity constraints, 
  validation on IAEA TRS-403 benchmarks.
  
- Tier 2 (research): Advanced methods requiring careful hyperparameter tuning.
  Suitable for research with proper uncertainty quantification.
  
- Tier 3 (experimental): Not recommended for final reports without 
  additional validation. Includes parametric models, dictionary learning,
  metaheuristics, and statistically questionable methods for single measurements.
"""

from typing import Any

__all__ = ["get_method_metadata", "METHOD_TIERS", "NUMBA_ACCELERATED_METHODS"]

# Tier 1: Production Ready (Radiation Protection)
TIER_1_STABLE = [
    "unfold_cvxpy",
    "unfold_qpsolvers",
    "unfold_maxed",
    "unfold_bayes",
    "unfold_tsvd",
]

# Tier 2: Research & Advanced UQ
TIER_2_RESEARCH = [
    "unfold_mcmc",
    "unfold_cuqi",
    "unfold_lanczos",
    "unfold_gks",
    "unfold_mlem_bs",
    "unfold_amg",
    "unfold_cgls",
    "unfold_fista",
    "unfold_admm",
    "unfold_lbfgsb",
    "unfold_mirror_descent",
    "unfold_pgd",
    "unfold_extragradient",
    "unfold_coordinate_descent",
    "unfold_subgradient",
    "unfold_landweber",
    "unfold_kaczmarz",
    "unfold_randomized_kaczmarz",
    "unfold_sart",
    "unfold_osem",
    "unfold_gravel",
    "unfold_mlem",
    "unfold_doroshenko",
    "unfold_bunki",
    "unfold_bunkiut",
    "unfold_rebunki",
    "unfold_imaxed",
    "unfold_amaxed",
    "unfold_directed_divergence",
    "unfold_mapem",
    "unfold_bsrem",
    "unfold_frank_wolfe",
    "unfold_nnqp",
    "unfold_qpmad",
    "unfold_scipy_direct_method",
    "unfold_lmfit",
    "unfold_zfit",
    "unfold_pspline_reml",
    "unfold_nspline",
    "unfold_bayes_spline_regularization",
    "unfold_statreg",
    "unfold_tikhonov_tv",
    "unfold_tikhonov_legendre",
    "unfold_tikhonov_sobolev_dp",
    "unfold_iterative_refinement",
    "unfold_hybrid_gmres",
    "unfold_ensemble",
    "unfold_combined",
    "unfold_composite",
    "unfold_cascade",
    "unfold_binned",
    "unfold_epic",
    "unfold_express",
    "unfold_interpret",
    "unfold_eki",
    "unfold_uno",
    "unfold_mystic",
    "unfold_docplex",
    "unfold_scip",
    "unfold_qubo",  # Note: QUBO is experimental but uses stable backend
    "unfold_smt",  # Note: SMT is experimental but uses stable backend
]

# Tier 3: Experimental / Diagnostic Only
TIER_3_EXPERIMENTAL = [
    # Parametric methods - only if field type is known a priori
    "unfold_parametric",
    "unfold_parametric2",
    "unfold_fruit_like",
    "unfold_bayesian_parametric",
    "unfold_hybrid_parametric",
    "unfold_fission_ga",
    "unfold_bon95",
    # Dictionary/Sparse methods - high model bias risk
    "unfold_nnksvd",
    "unfold_cs",
    # Metaheuristics - non-deterministic, slow, no natural uncertainty
    "unfold_genetic",
    "unfold_maeo",
    "unfold_gnowee",
    # Statistically questionable for single measurements
    "unfold_gee",
    "unfold_ssr",
    # Classic reimplementations - legacy compatibility only
    "unfold_crystal_ball",
    "unfold_staysl",
    "unfold_sandii",
    "unfold_ferdor",
    "unfold_nsduaz",
    "unfold_rfsp_jul",
    "unfold_odl_advanced",
    "unfold_mlem_odl",
    "unfold_mlem_stop",
    "unfold_bunkiut",
    "unfold_reconst",
]

# Methods with Numba JIT acceleration
# Verified by checking for _numba_jit imports and @njit decorators
NUMBA_ACCELERATED_METHODS = [
    "unfold_mlem",
    "unfold_kaczmarz",
    "unfold_doroshenko",
    "unfold_gravel",
    "unfold_bayes",
    "unfold_landweber",
    # Note: The following methods are listed in docs but NOT numba-accelerated:
    # unfold_sart, unfold_osem, unfold_randomized_kaczmarz, unfold_mlem_bs
    # They use pure Python or other backends.
]

# Method metadata with descriptions and warnings
METHOD_METADATA: dict[str, dict[str, Any]] = {
    # Tier 1
    "unfold_cvxpy": {
        "tier": "stable",
        "numba_accelerated": False,
        "description": "Convex optimization via CVXPY (Tikhonov L2/L1)",
        "use_case": "Production radiation protection, regulatory compliance",
        "limitations": "Requires CVXPY installation, slower for large problems",
    },
    "unfold_qpsolvers": {
        "tier": "stable",
        "numba_accelerated": False,
        "description": "Quadratic programming solvers",
        "use_case": "Production with positivity constraints",
        "limitations": "May require commercial solver licenses",
    },
    "unfold_maxed": {
        "tier": "stable",
        "numba_accelerated": False,
        "description": "Maximum entropy method (classic BSS standard)",
        "use_case": "Well-characterized fields, IAEA benchmarks",
        "limitations": "Sensitive to default spectrum choice",
    },
    "unfold_bayes": {
        "tier": "stable",
        "numba_accelerated": True,
        "description": "Bayesian unfolding (D'Agostini)",
        "use_case": "Interpretable results with stopping rule",
        "limitations": "Sensitive to prior and iteration count",
    },
    "unfold_tsvd": {
        "tier": "stable",
        "numba_accelerated": False,
        "description": "Truncated SVD (diagnostic use)",
        "use_case": "Ill-conditioning analysis, regularization parameter selection",
        "limitations": "Assumes white noise; use weighted version for BSS",
    },
    
    # Tier 2 examples
    "unfold_mlem": {
        "tier": "research",
        "numba_accelerated": True,
        "description": "Maximum likelihood EM for Poisson data",
        "use_case": "Low-count measurements, Poisson statistics",
        "limitations": "Requires positive readings; fails with background subtraction negatives",
    },
    "unfold_osem": {
        "tier": "research",
        "numba_accelerated": False,
        "description": "Ordered subsets EM (accelerated convergence)",
        "use_case": "Fast iterative reconstruction",
        "limitations": "May introduce ring artifacts; monotonicity not guaranteed",
    },
    "unfold_mlem_bs": {
        "tier": "research",
        "numba_accelerated": False,
        "description": "B-spline MLEM hybrid",
        "use_case": "Smooth spectra with reduced dimensionality",
        "limitations": "Spline basis may miss sharp features",
    },
    
    # Tier 3 examples with warnings
    "unfold_gee": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Generalized Estimating Equations",
        "use_case": "NOT RECOMMENDED for single BSS measurements",
        "limitations": "GEE requires many clusters; BSS has only one cluster. "
                      "Sandwich estimator inconsistent for small cluster counts.",
        "warning": "Statistically invalid for typical BSS applications",
    },
    "unfold_ssr": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Subset-based statistical regression",
        "use_case": "NOT RECOMMENDED without validation",
        "limitations": "Similar issues to GEE with single-cluster data",
        "warning": "Use only for exploratory analysis",
    },
    "unfold_parametric": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Parametric spectral model fitting",
        "use_case": "ONLY when field type is known (reactor, calibration facility)",
        "limitations": "Cannot recover unknown spectral features; model bias",
        "warning": "Will produce smooth but potentially incorrect spectra for unknown fields",
    },
    "unfold_nnksvd": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Dictionary learning with K-SVD",
        "use_case": "Research with representative training data",
        "limitations": "High model bias if dictionary doesn't match physics",
        "warning": "Requires user-provided training spectra from IAEA database or similar",
    },
    "unfold_genetic": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Genetic algorithm optimization",
        "use_case": "Non-convex landscapes, global search",
        "limitations": "Non-deterministic, slow, no natural uncertainty measure",
        "warning": "Fix random_state for reproducibility; not for production reports",
    },
    "unfold_qubo": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Quadratic Unconstrained Binary Optimization",
        "use_case": "Demonstration of quantum-inspired methods",
        "limitations": "Binary encoding loses precision; requires many bits",
        "warning": "Not suitable for quantitative dose assessment",
    },
    "unfold_smt": {
        "tier": "experimental",
        "numba_accelerated": False,
        "description": "Satisfiability Modulo Theories (Z3)",
        "use_case": "Exact arithmetic demonstration",
        "limitations": "Contradiction: exact solving impossible for noisy incompatible data",
        "warning": "Minimizing residual makes this inexact; conceptual mismatch",
    },
}


def get_method_metadata(method_name: str) -> dict[str, Any]:
    """Get metadata for a specific unfolding method.
    
    Parameters
    ----------
    method_name : str
        Name of the unfolding method (e.g., 'unfold_cvxpy').
    
    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with keys:
        - tier: 'stable', 'research', or 'experimental'
        - numba_accelerated: bool
        - description: str
        - use_case: str
        - limitations: str
        - warning: str (optional, for experimental methods)
    
    Raises
    ------
    KeyError
        If method_name is not recognized.
    """
    if method_name in METHOD_METADATA:
        return METHOD_METADATA[method_name].copy()
    
    # Infer from tier lists if not in explicit metadata
    if method_name in TIER_1_STABLE:
        return {
            "tier": "stable",
            "numba_accelerated": method_name in NUMBA_ACCELERATED_METHODS,
            "description": f"Production-ready method: {method_name}",
            "use_case": "Radiation protection applications",
            "limitations": "See method-specific documentation",
        }
    elif method_name in TIER_2_RESEARCH:
        return {
            "tier": "research",
            "numba_accelerated": method_name in NUMBA_ACCELERATED_METHODS,
            "description": f"Research method: {method_name}",
            "use_case": "Advanced research with careful tuning",
            "limitations": "Requires hyperparameter optimization",
        }
    elif method_name in TIER_3_EXPERIMENTAL:
        return {
            "tier": "experimental",
            "numba_accelerated": method_name in NUMBA_ACCELERATED_METHODS,
            "description": f"Experimental method: {method_name}",
            "use_case": "Diagnostic/exploratory only",
            "limitations": "Not validated for production use",
            "warning": "Do not use for final radiation protection reports without validation",
        }
    
    raise KeyError(
        f"Unknown method: {method_name}. "
        f"Available methods: {list(METHOD_METADATA.keys())[:10]}..."
    )


# Convenience sets for quick checks
METHOD_TIERS = {
    "stable": set(TIER_1_STABLE),
    "research": set(TIER_2_RESEARCH),
    "experimental": set(TIER_3_EXPERIMENTAL),
}
