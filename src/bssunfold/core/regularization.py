"""Regularization parameter selection module for bssunfold package.

This module provides methods for selecting optimal regularization parameters
using various heuristics: L-curve, GCV, Discrepancy Principle.
"""

import numpy as np
import warnings
from typing import Optional, Dict, Any, Tuple, List

__all__ = [
    "select_regularization_parameter",
    "lcurve_selection",
    "gcv_selection",
    "discrepancy_principle_selection",
    "cosine_similarity_selection",
    "compare_regularization_methods",
    "randomization_experiment",
]


def _create_identity_matrix(n: int) -> np.ndarray:
    """Create identity matrix of size n."""
    return np.eye(n)


def _estimate_noise_variance(
    A: np.ndarray,
    b: np.ndarray,
) -> float:
    """Estimate noise variance from least squares residual.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    
    Returns
    -------
    float
        Estimated noise variance.
    """
    x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual = b - A @ x_ls
    return float(np.var(residual))


def select_regularization_parameter(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "lcurve",
    noise_var: Optional[float] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Select regularization parameter using specified method.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    method : str, optional
        Selection method: 'lcurve', 'gcv', 'dp', 'cosine' (default: 'lcurve').
    noise_var : float, optional
        Noise variance for discrepancy principle.
    initial_spectrum : np.ndarray, optional
        Initial spectrum for cosine similarity method.
    **kwargs : dict
        Additional method-specific arguments.
    
    Returns
    -------
    float
        Selected regularization parameter (lambda).
    
    Raises
    ------
    ValueError
        If method is unknown or selection fails.
    """
    if method == "lcurve":
        return lcurve_selection(A, b, **kwargs)
    elif method == "gcv":
        return gcv_selection(A, b, **kwargs)
    elif method == "dp":
        return discrepancy_principle_selection(A, b, noise_var=noise_var, **kwargs)
    elif method == "cosine":
        return cosine_similarity_selection(A, b, initial_spectrum, **kwargs)
    else:
        raise ValueError(
            f"Unknown regularization selection method: {method}. "
            "Choose from 'lcurve', 'gcv', 'dp', 'cosine'."
        )


def lcurve_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using L-curve corner heuristic.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).
    
    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk
        
        n = A.shape[1]
        L = _create_identity_matrix(n)
        
        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.lcorner(fam)
        lam = result.get("opt_lambdah")
        
        if lam is None:
            raise ValueError("L-curve heuristic did not return lambda.")
        
        return float(lam)
    
    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback L-curve implementation.",
            ImportWarning,
        )
        return _lcurve_fallback(A, b, n_alphas, alpha_range)


def _lcurve_fallback(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback L-curve implementation without pytikhonov."""
    
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    residuals = []
    norms = []
    
    for alpha in alphas:
        # Solve regularized problem
        n = A.shape[1]
        L = _create_identity_matrix(n)
        P = A.T @ A + alpha * (L.T @ L)
        q = -A.T @ b
        
        try:
            x = np.linalg.solve(P, -q)
            x = np.maximum(x, 0)  # Non-negativity
            residual = np.linalg.norm(A @ x - b)
            norm = np.linalg.norm(L @ x)
            residuals.append(residual)
            norms.append(norm)
        except np.linalg.LinAlgError:
            continue
    
    if len(residuals) < 3:
        return 1.0  # Default value
    
    # Find corner using maximum curvature
    log_res = np.log(residuals)
    log_norm = np.log(norms)
    
    # Simple corner detection: point with maximum distance from line
    # connecting endpoints
    p1 = np.array([log_res[0], log_norm[0]])
    p2 = np.array([log_res[-1], log_norm[-1]])
    
    distances = []
    for i in range(len(residuals)):
        p = np.array([log_res[i], log_norm[i]])
        # Distance from point to line
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    
    idx_max = np.argmax(distances)
    return float(alphas[idx_max])


def gcv_selection(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using Generalized Cross Validation.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).
    
    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk
        
        n = A.shape[1]
        L = _create_identity_matrix(n)
        
        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.gcvmin(fam)
        lam = result.get("opt_lambdah")
        
        if lam is None:
            raise ValueError("GCV minimization did not return lambda.")
        
        return float(lam)
    
    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback GCV implementation.",
            ImportWarning,
        )
        return _gcv_fallback(A, b, n_alphas, alpha_range)


def _gcv_fallback(
    A: np.ndarray,
    b: np.ndarray,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback GCV implementation without pytikhonov."""
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    gcv_values = []
    
    m, n = A.shape
    
    for alpha in alphas:
        # Compute GCV function
        # GCV(alpha) = ||A x_alpha - b||^2 / (m - trace(A A^+_alpha))^2
        try:
            L = _create_identity_matrix(n)
            P = A.T @ A + alpha * (L.T @ L)
            x = np.linalg.solve(P, A.T @ b)
            x = np.maximum(x, 0)
            
            residual = np.linalg.norm(A @ x - b) ** 2
            
            # Approximate trace of influence matrix
            # Using simplified formula for Tikhonov regularization
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            trace_term = np.sum(s**2 / (s**2 + alpha))
            
            gcv = residual / (m - trace_term) ** 2
            gcv_values.append(gcv)
        except (np.linalg.LinAlgError, ValueError):
            gcv_values.append(np.inf)
    
    if not gcv_values or all(v == np.inf for v in gcv_values):
        return 1.0  # Default value
    
    idx_min = np.argmin(gcv_values)
    return float(alphas[idx_min])


def discrepancy_principle_selection(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Select regularization parameter using Discrepancy Principle.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    noise_var : float, optional
        Noise variance. If None, estimated from data.
    n_alphas : int, optional
        Number of alpha values to test (default: 50).
    alpha_range : Tuple[float, float], optional
        Range of alpha values (default: (1e-9, 1e2)).
    
    Returns
    -------
    float
        Selected regularization parameter.
    """
    try:
        import pytikhonov as ptk
        
        n = A.shape[1]
        L = _create_identity_matrix(n)
        
        if noise_var is None:
            noise_var = _estimate_noise_variance(A, b)
        
        delta = np.sqrt(noise_var)
        
        fam = ptk.TikhonovFamily(A, L, b)
        result = ptk.discrepancy_principle(fam, delta=delta)
        lam = result.get("opt_lambdah")
        
        if lam is None:
            raise ValueError("Discrepancy principle did not return lambda.")
        
        return float(lam)
    
    except ImportError:
        warnings.warn(
            "pytikhonov not available. Using fallback DP implementation.",
            ImportWarning,
        )
        if noise_var is None:
            noise_var = _estimate_noise_variance(A, b)
        return _dp_fallback(A, b, noise_var, n_alphas, alpha_range)


def _dp_fallback(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: float,
    n_alphas: int = 50,
    alpha_range: Tuple[float, float] = (1e-9, 1e2),
) -> float:
    """Fallback Discrepancy Principle implementation."""
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    delta = np.sqrt(noise_var)
    m = len(b)
    target_residual = delta * np.sqrt(m)
    
    residuals = []
    
    for alpha in alphas:
        n = A.shape[1]
        L = _create_identity_matrix(n)
        P = A.T @ A + alpha * (L.T @ L)
        
        try:
            x = np.linalg.solve(P, A.T @ b)
            x = np.maximum(x, 0)
            residual = np.linalg.norm(A @ x - b)
            residuals.append(residual)
        except np.linalg.LinAlgError:
            residuals.append(np.inf)
    
    # Find alpha where residual is closest to target
    residuals = np.array(residuals)
    idx = np.argmin(np.abs(residuals - target_residual))
    
    return float(alphas[idx])


def cosine_similarity_selection(
    A: np.ndarray,
    b: np.ndarray,
    initial_spectrum: np.ndarray,
    n_alphas: int = 100,
    alpha_range: Tuple[float, float] = (-9, 2),
    norm: int = 2,
) -> float:
    """Select regularization parameter by maximizing cosine similarity.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix.
    b : np.ndarray
        Measurement vector.
    initial_spectrum : np.ndarray
        Initial/reference spectrum for similarity comparison.
    n_alphas : int, optional
        Number of alpha values to test (default: 100).
    alpha_range : Tuple[float, float], optional
        Log range of alpha values (default: (-9, 2)).
    norm : int, optional
        Norm type for regularization (default: 2).
    
    Returns
    -------
    float
        Selected regularization parameter.
    """
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    similarities = []
    
    # Normalize initial spectrum
    norm_init = np.linalg.norm(initial_spectrum)
    if norm_init == 0:
        raise ValueError("Initial spectrum has zero norm.")
    initial_normalized = initial_spectrum / norm_init
    
    for alpha in alphas:
        try:
            n = A.shape[1]
            L = _create_identity_matrix(n)
            P = A.T @ A + alpha * (L.T @ L)
            
            x = np.linalg.solve(P, A.T @ b)
            x = np.maximum(x, 0)
            
            # Compute cosine similarity
            norm_x = np.linalg.norm(x)
            if norm_x == 0:
                similarities.append(0.0)
            else:
                sim = np.dot(x, initial_normalized) / norm_x
                similarities.append(sim)
        except np.linalg.LinAlgError:
            similarities.append(-1.0)
    
    idx_max = np.argmax(similarities)
    return float(alphas[idx_max])


def compare_regularization_methods(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    plot: bool = False,
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare regularization selection methods for given system.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    noise_var : float, optional
        Noise variance for discrepancy principle.
        If None, estimated from residual of unregularized solution.
    plot : bool, optional
        If True, generate comparison plot.
    plot_path : str, optional
        Path to save the plot.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'lcurve': dict from lcorner()
        - 'dp': dict from discrepancy_principle()
        - 'gcv': dict from gcvmin()
        - 'all_data': dict from pytikhonov.all_regparam_methods()
        - 'selected': dict mapping method name to selected lambda.
    
    Raises
    ------
    ImportError
        If pytikhonov is not available.
    """
    try:
        import pytikhonov as ptk
    except ImportError as e:
        raise ImportError(
            "pytikhonov is required for compare_regularization_methods. "
            "Install with: pip install pytikhonov"
        ) from e
    
    n = A.shape[1]
    L = np.eye(n)
    fam = ptk.TikhonovFamily(A, L, b)
    
    # Compute each method
    lc_res = ptk.lcorner(fam)
    if noise_var is None:
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        noise_var = np.var(b - A @ x_ls)
    delta = np.sqrt(noise_var)
    dp_res = ptk.discrepancy_principle(fam, delta=delta)
    gcv_res = ptk.gcvmin(fam)
    all_data = ptk.all_regparam_methods(fam)
    
    selected = {
        "lcurve": lc_res.get("opt_lambdah"),
        "dp": dp_res.get("opt_lambdah"),
        "gcv": gcv_res.get("opt_lambdah"),
    }
    
    if plot:
        ptk.plot_all_methods(all_data, plot_path=plot_path)
    
    return {
        "lcurve": lc_res,
        "dp": dp_res,
        "gcv": gcv_res,
        "all_data": all_data,
        "selected": selected,
    }


def randomization_experiment(
    A: np.ndarray,
    b: np.ndarray,
    noise_var: Optional[float] = None,
    n_samples: int = 10,
    rseed: int = 0,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run randomization experiments for regularization parameter selection.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    noise_var : float, optional
        Noise variance for generating perturbed measurements.
        If None, estimated from residual of unregularized solution.
    n_samples : int, optional
        Number of random samples for each method, default 10.
    rseed : int, optional
        Random seed for reproducibility, default 0.
    methods : list of str, optional
        List of methods to run: 'lcurve', 'dp', 'gcv', 'lcurve_full'.
        If None, runs all four.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys for each method, each containing:
        - 'lambdas': array of selected lambdas per sample.
        - 'mean': mean of lambdas.
        - 'std': standard deviation.
        - 'median': median.
        - 'min', 'max': range.
        - 'cv': coefficient of variation (std/mean).
        - 'raw_result': raw output from pytikhonov function.
    
    Raises
    ------
    ImportError
        If pytikhonov is not available.
    """
    try:
        import pytikhonov as ptk
    except ImportError as e:
        raise ImportError(
            "pytikhonov is required for randomization_experiment. "
            "Install with: pip install pytikhonov"
        ) from e
    
    n = A.shape[1]
    L = np.eye(n)
    
    # Estimate noise variance if not provided
    if noise_var is None:
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        noise_var = np.var(b - A @ x_ls)
    
    # Create TikhonovFamily with btrue = b (assumed true signal)
    fam = ptk.TikhonovFamily(A, L, b, btrue=b, noise_var=noise_var)
    
    if methods is None:
        methods = ["lcurve", "dp", "gcv", "lcurve_full"]
    
    results = {}
    for method in methods:
        if method == "lcurve":
            raw = ptk.rand_lcorner(fam, n_samples=n_samples, rseed=rseed)
            lambdas = np.array(raw[0])  # first element is list of lambdas
        elif method == "dp":
            raw = ptk.rand_discrepancy_principle(
                fam, n_samples=n_samples, tau=1.01, rseed=rseed
            )
            lambdas = np.array(raw[0])
        elif method == "gcv":
            raw = ptk.rand_gcvmin(fam, n_samples=n_samples, rseed=rseed)
            lambdas = np.array(raw[0])
        elif method == "lcurve_full":
            raw = ptk.rand_lcurve(
                fam, lambdahs=None, n_samples=n_samples, rseed=rseed
            )
            lambdas = np.array(raw[0])
        else:
            warnings.warn(f"Unknown method: {method}. Skipping.")
            continue
        
        # Compute statistics
        mean = float(np.mean(lambdas))
        std = float(np.std(lambdas))
        median = float(np.median(lambdas))
        min_val = float(np.min(lambdas))
        max_val = float(np.max(lambdas))
        cv = std / mean if mean != 0 else np.inf
        
        results[method] = {
            "lambdas": lambdas,
            "mean": mean,
            "std": std,
            "median": median,
            "min": min_val,
            "max": max_val,
            "cv": cv,
            "raw_result": raw,
        }
    
    return results
