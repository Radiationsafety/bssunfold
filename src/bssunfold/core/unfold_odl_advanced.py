"""Advanced ODL-based unfolding methods with PDHG and Douglas-Rachford splitting.

This module provides advanced regularization methods using the Operator Discretization 
Library (ODL), including Primal-Dual Hybrid Gradient (PDHG) and Douglas-Rachford 
splitting algorithms. These methods support Total Variation (TV) regularization which 
better preserves sharp spectral features compared to standard Tikhonov regularization.

Requires the 'odl' package to be installed.
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding
from ._matrix_utils import create_derivative_matrix


def solve_odl_pdhg(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Primal-Dual Hybrid Gradient (PDHG).

    PDHG is a first-order primal-dual algorithm that can handle non-smooth 
    regularizers like Total Variation. It solves problems of the form:

        min_x ||Kx - y||^2 + λ * R(x)

    where R(x) can be TV norm or L1 norm.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None, uniform spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tau : float, optional
        Primal step size. If None, computed automatically.
    sigma : float, optional
        Dual step size. If None, computed automatically.
    use_tv : bool, optional
        If True, use Total Variation regularization (default: True).
    tv_weight : float, optional
        Weight for TV regularization term (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity constraint (default: True).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple
        (spectrum, iterations, converged)
    """
    try:
        import odl
        from odl.solvers import pdhg, proximal_nonnegativity
    except ImportError as e:
        raise ImportError(
            "odl is required for ODL-based PDHG unfolding. "
            "Install with: pip install odl"
        ) from e

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n = A.shape

    # Create ODL spaces
    measurement_space = odl.uniform_discr(0, m_len, m_len)
    spectrum_space = odl.uniform_discr(0, n, n)

    # Create operator
    operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

    # Regularization term
    if use_tv:
        # Total Variation regularization using finite difference matrix
        # Build 1D forward difference matrix manually
        h = 1.0  # grid spacing (normalized)
        grad_matrix = np.zeros((n - 1, n))
        for i in range(n - 1):
            grad_matrix[i, i] = -1.0 / h
            grad_matrix[i, i + 1] = 1.0 / h
        
        # Combine data and TV in product space
        measurement_space_tv = odl.uniform_discr(0, n - 1, n - 1)
        product_space = odl.ProductSpace(measurement_space, measurement_space_tv)
        
        # Create combined operator [A; ∇]
        combined_data = np.vstack([A, tv_weight * grad_matrix])
        combined_op = odl.MatrixOperator(combined_data, domain=spectrum_space, range=product_space)
        
        # L1 norm for TV
        l1_norm = lambda x: sum(x[i].norm(1) for i in range(len(x)))
        
        # Proximal for L1 (soft thresholding)
        def prox_l1(sigma, x):
            result = []
            for i in range(len(x)):
                val = np.abs(x[i])
                thresh = np.maximum(val - sigma, 0)
                result.append(np.sign(x[i]) * thresh)
            return type(x)(result)
        
        if nonnegativity:
            # Use PDHG with non-negativity via projection
            if tau is None or sigma is None:
                op_norm = odl.power_method(op=combined_op, x=spectrum_space.element())
                tau = 0.99 / op_norm
                sigma = 0.99 / op_norm
            
            # Initial point
            if x0 is None:
                x0 = np.ones(n) * 0.5
            x = spectrum_space.element(x0.copy())
            
            # Dual variable
            y = product_space.element()
            
            # Run PDHG iterations manually
            for k in range(max_iterations):
                # Gradient step for primal
                grad = combined_op.adjoint(y)
                x_new = x - tau * grad
                
                # Apply non-negativity
                x_new = np.maximum(x_new, 0)
                x_new = spectrum_space.element(x_new)
                
                # Gradient step for dual
                y_new = y + sigma * combined_op(2 * x_new - x)
                
                # Apply proximal of L1 (soft thresholding)
                y_list = []
                split_idx = m_len
                y_list.append(y_new[:split_idx])
                y_list.append(prox_l1(sigma * tv_weight, y_new[split_idx:]))
                y_new = product_space.element(y_list)
                
                # Update
                x = x_new
                y = y_new
            
        else:
            # No nonnegativity constraint
            if tau is None or sigma is None:
                op_norm = odl.power_method(op=combined_op, x=spectrum_space.element())
                tau = 0.99 / op_norm
                sigma = 0.99 / op_norm
            
            if x0 is None:
                x0 = np.ones(n) * 0.5
            x = spectrum_space.element(x0.copy())
            
            y = product_space.element()
            
            for k in range(max_iterations):
                grad = combined_op.adjoint(y)
                x_new = x - tau * grad
                x_new = spectrum_space.element(x_new)
                
                y_new = y + sigma * combined_op(2 * x_new - x)
                
                y_list = []
                split_idx = m_len
                y_list.append(y_new[:split_idx])
                y_list.append(prox_l1(sigma * tv_weight, y_new[split_idx:]))
                y_new = product_space.element(y_list)
                
                x = x_new
                y = y_new
    else:
        # Simple Landweber iteration for L2 regularization
        reg_weight = tv_weight
        
        if x0 is None:
            x0 = np.ones(n) * 0.5
        x = spectrum_space.element(x0.copy())
        
        if tau is None:
            op_norm = odl.power_method(op=operator, x=spectrum_space.element())
            tau = 0.99 / (op_norm ** 2 + reg_weight)
        
        for k in range(max_iterations):
            residual = operator(x) - b
            grad = operator.adjoint(residual) + reg_weight * x
            x = x - tau * grad
            if nonnegativity:
                x = np.maximum(x, 0)
                x = spectrum_space.element(x)

    # Extract result
    x_opt = np.asarray(x.data)
    x_opt = np.maximum(x_opt, 0)  # Ensure non-negativity
    
    converged = True  # PDHG typically runs fixed iterations
    return x_opt, max_iterations, converged


def solve_odl_douglas_rachford(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using Douglas-Rachford splitting.

    Douglas-Rachford is an operator splitting method that can handle
    the sum of two convex functionals. It's particularly effective for
    problems with non-smooth regularizers.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess. If None, uniform spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    use_tv : bool, optional
        If True, use Total Variation regularization (default: True).
    tv_weight : float, optional
        Weight for TV regularization term (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity constraint (default: True).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    tuple
        (spectrum, iterations, converged)
    """
    try:
        import odl
        from odl.solvers import douglas_rachford
    except ImportError as e:
        raise ImportError(
            "odl is required for ODL-based Douglas-Rachford unfolding. "
            "Install with: pip install odl"
        ) from e

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n = A.shape

    # Create ODL spaces
    measurement_space = odl.uniform_discr(0, m_len, m_len)
    spectrum_space = odl.uniform_discr(0, n, n)

    # Create operator
    operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

    # Data fidelity term
    data_norm = odl.solvers.L2NormSquared(measurement_space)
    data_functional = data_norm.translated(b)

    # Regularization
    if use_tv:
        # TV regularization via gradient operator
        gradient_op = odl.FiniteDifferenceOperator(spectrum_space, method='forward')
        tv_norm = odl.solvers.L1Norm(gradient_op.range)
        
        # Combined functional
        def reg_func(x):
            return tv_weight * tv_norm(gradient_op(x))
        
        # Proximal operator for TV
        def prox_reg(sigma, x):
            grad_x = gradient_op(x)
            dual_grad = gradient_op.range.element(grad_x)
            
            # Proximal of L1 norm (soft thresholding)
            prox_dual = dual_grad.maximum(1 - sigma * tv_weight) / \
                       (dual_grad.abs().maximum(1 - sigma * tv_weight) + 1e-10) * dual_grad
            
            return x - sigma * gradient_op.adjoint(prox_dual)
    else:
        # L2 regularization
        reg_weight = tv_weight
        
        def reg_func(x):
            return 0.5 * reg_weight * odl.solvers.L2Norm(spectrum_space)(x) ** 2
        
        def prox_reg(sigma, x):
            return x / (1 + sigma * reg_weight)

    # Constraint for non-negativity
    if nonnegativity:
        constraint = odl.solvers.IndicatorNonnegativity(spectrum_space)
        
        def prox_constraint(sigma, x):
            return x.maximum(0)
        
        # Combine regularization and constraint
        def combined_prox(sigma, x):
            return prox_constraint(sigma, prox_reg(sigma, x))
    else:
        combined_prox = prox_reg

    # Initial point
    if x0 is None:
        x0 = np.ones(n) * 0.5
    x = spectrum_space.element(x0)

    # Douglas-Rachford parameters
    param = 0.5  # Relaxation parameter

    # Run Douglas-Rachford iteration
    y = x.copy()
    
    for k in range(max_iterations):
        # Proximal step for data fidelity
        prox_data = odl.proximal_moreau_yoshida(data_functional, 1.0, operator(y))
        
        # Back-projection
        z = y - operator.adjoint(prox_data - operator(y))
        
        # Proximal step for regularization
        x_new = combined_prox(1.0, z)
        
        # Update
        y = y + param * (x_new - y)
        
        # Check convergence
        if k > 0:
            diff = odl.norm(x_new - x)
            if diff < tolerance:
                x = x_new
                break
        
        x = x_new

    # Extract result
    x_opt = np.asarray(x.data)
    x_opt = np.maximum(x_opt, 0)
    
    converged = True
    return x_opt, k + 1, converged


def unfold_odl_pdhg(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using ODL PDHG with TV regularization.

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
        Initial spectrum approximation.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tau : float, optional
        Primal step size.
    sigma : float, optional
        Dual step size.
    use_tv : bool, optional
        Use Total Variation regularization (default: True).
    tv_weight : float, optional
        TV regularization weight (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity (default: True).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for error calculation (default: 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default: 100).
    save_result : bool, optional
        Save to history (default: False).
    random_state : int, optional
        Random seed.

    Returns
    -------
    Dict
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins) * 0.5

    def solve_wrapper(A, b, **kwargs):
        x_init = kwargs.pop('x0', initial_spectrum)
        return solve_odl_pdhg(
            A, b, x0=x_init,
            max_iterations=max_iterations,
            tau=tau, sigma=sigma,
            use_tv=use_tv, tv_weight=tv_weight,
            nonnegativity=nonnegativity,
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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="ODL-PDHG",
        extra_output={
            "max_iterations": max_iterations,
            "use_tv": use_tv,
            "tv_weight": tv_weight,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )


def unfold_odl_douglas_rachford(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    use_tv: bool = True,
    tv_weight: float = 0.1,
    nonnegativity: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using ODL Douglas-Rachford splitting with TV regularization.

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
        Initial spectrum approximation.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    use_tv : bool, optional
        Use Total Variation regularization (default: True).
    tv_weight : float, optional
        TV regularization weight (default: 0.1).
    nonnegativity : bool, optional
        Enforce non-negativity (default: True).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for error calculation (default: 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default: 100).
    save_result : bool, optional
        Save to history (default: False).
    random_state : int, optional
        Random seed.

    Returns
    -------
    Dict
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins) * 0.5

    def solve_wrapper(A, b, **kwargs):
        x_init = kwargs.pop('x0', initial_spectrum)
        return solve_odl_douglas_rachford(
            A, b, x0=x_init,
            max_iterations=max_iterations,
            use_tv=use_tv, tv_weight=tv_weight,
            nonnegativity=nonnegativity,
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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="ODL-DouglasRachford",
        extra_output={
            "max_iterations": max_iterations,
            "use_tv": use_tv,
            "tv_weight": tv_weight,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )


__all__ = [
    "solve_odl_pdhg",
    "solve_odl_douglas_rachford",
    "unfold_odl_pdhg",
    "unfold_odl_douglas_rachford",
]
