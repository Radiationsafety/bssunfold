"""
OccamPy-based unfolding methods for neutron spectrum reconstruction.

This module provides scalable iterative methods from the OccamPy library,
including LSQR, L-BFGS, ISTA, and Split-Bregman algorithms.
These methods are particularly effective for large-scale inverse problems
with many energy groups.

References:
    - OccamPy: https://github.com/fpicetti/occamypy
    - Picetti, F., et al. "OccamPy: A Python library for large-scale inverse problems."
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from scipy.sparse import csr_matrix, issparse

try:
    import occamypy as occ
    from occamypy import operator, solver, problem
    OCCAMPY_AVAILABLE = True
except ImportError:
    OCCAMPY_AVAILABLE = False
    occ = None
    operator = None
    solver = None
    problem = None


class OccamPyUnfolder:
    """
    Neutron spectrum unfolding using OccamPy scalable iterative methods.
    
    OccamPy provides efficient implementations of modern solvers for large-scale
    inverse problems. This class wraps several algorithms suitable for neutron
    spectrum reconstruction:
    
    - **LSQR**: Iterative least-squares solver for sparse systems
    - **L-BFGS-B**: Limited-memory BFGS with bounds
    - **ISTA**: Iterative Shrinkage-Thresholding Algorithm (L1 regularization)
    - **Split-Bregman**: Efficient solver for L1-regularized problems
    - **CGLS**: Conjugate Gradient Least Squares
    - **FISTA**: Fast ISTA with acceleration
    
    Parameters
    ----------
    response_matrix : array_like
        Response matrix R of shape (n_channels, n_energy_groups).
    counts : array_like
        Measured counts of shape (n_channels,).
    counts_unc : array_like
        Uncertainties in counts of shape (n_channels,).
    method : str, optional
        Unfolding method to use. Options:
        - 'lsqr': LSQR iterative solver (default)
        - 'lbfgs': L-BFGS-B optimization
        - 'ista': ISTA with L1 regularization
        - 'splitbregman': Split-Bregman for L1 problems
        - 'cgls': CGLS solver
        - 'fista': Fast ISTA
    **kwargs : dict
        Additional keyword arguments passed to the solver.
    
    Attributes
    ----------
    spectrum : ndarray
        Unfolded spectrum of shape (n_energy_groups,).
    spectrum_unc : ndarray
        Uncertainties in unfolded spectrum.
    convergence_info : dict
        Information about convergence (iterations, residual, etc.).
    
    Examples
    --------
    >>> from bssunfold.core.unfold_occamy import OccamPyUnfolder
    >>> # Create response matrix and counts
    >>> R = np.random.rand(10, 20) * 0.1
    >>> counts = np.dot(R, np.exp(-np.linspace(0, 5, 20))) + 100
    >>> counts_unc = np.sqrt(counts)
    >>> # Unfold using LSQR
    >>> unfolder = OccamPyUnfolder(R, counts, counts_unc, method='lsqr')
    >>> spectrum = unfolder.unfold()
    """
    
    def __init__(
        self,
        response_matrix: np.ndarray,
        counts: np.ndarray,
        counts_unc: np.ndarray,
        method: str = 'lsqr',
        **kwargs
    ):
        if not OCCAMPY_AVAILABLE:
            raise ImportError(
                "occamypy is required for OccamPyUnfolder. "
                "Install it with: pip install occamypy pylops"
            )
        
        self.response_matrix = np.asarray(response_matrix)
        self.counts = np.asarray(counts)
        self.counts_unc = np.asarray(counts_unc)
        
        # Validate shapes
        if self.response_matrix.ndim != 2:
            raise ValueError("response_matrix must be 2D")
        if len(self.counts) != self.response_matrix.shape[0]:
            raise ValueError("counts length must match response_matrix rows")
        if len(self.counts_unc) != len(self.counts):
            raise ValueError("counts_unc length must match counts")
        
        self.method = method.lower()
        valid_methods = ['lsqr', 'lbfgs', 'lbfgsb', 'ista', 'splitbregman', 
                        'cgls', 'fista', 'sd', 'nlcg']
        if self.method not in valid_methods:
            raise ValueError(
                f"Method '{method}' not recognized. Choose from: {valid_methods}"
            )
        
        # Solver parameters
        self.solver_kwargs = kwargs
        
        # Results storage
        self.spectrum = None
        self.spectrum_unc = None
        self.convergence_info = {}
    
    def _setup_weighted_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up weighted least-squares problem.
        
        Returns
        -------
        R_weighted : ndarray
            Weighted response matrix.
        y_weighted : ndarray
            Weighted counts vector.
        """
        # Weight by inverse uncertainty
        weights = 1.0 / np.maximum(self.counts_unc, 1e-10)
        R_weighted = self.response_matrix * weights[:, np.newaxis]
        y_weighted = self.counts * weights
        return R_weighted, y_weighted
    
    def _create_operator(self, R: np.ndarray) -> occ.Operator:
        """
        Create OccamPy operator from response matrix.
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        
        Returns
        -------
        op : occ.Operator
            OccamPy operator object.
        """
        # Convert to sparse if beneficial
        if issparse(R):
            R_sparse = R
        else:
            R_sparse = csr_matrix(R)
        
        # Create Matrix operator
        op = operator.Matrix(R_sparse, transpose=True)
        return op
    
    def _solve_lsqr(
        self, 
        R: np.ndarray, 
        y: np.ndarray,
        n_iter: int = 100,
        tol: float = 1e-6,
        damp: float = 0.0,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using LSQR algorithm.
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        damp : float
            Damping parameter for regularization.
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create least-squares problem
        prob = problem.LeastSquares(data, op)
        
        # Solve with LSQR
        lsqr_solver = solver.LSQR(
            damp=damp,
            iter_lim=n_iter,
            tol=tol,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = lsqr_solver.run(prob, x0)
        x = result.vector[:]
        
        info = {
            'method': 'lsqr',
            'iterations': result.info.get('iter', n_iter),
            'residual': result.info.get('residual', np.linalg.norm(R @ x - y)),
            'damping': damp
        }
        
        return x, info
    
    def _solve_lbfgs(
        self,
        R: np.ndarray,
        y: np.ndarray,
        n_iter: int = 100,
        tol: float = 1e-6,
        bounds: Optional[Tuple[float, float]] = (0, None),
        m: int = 10,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using L-BFGS-B optimization.
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        bounds : tuple
            Bounds for solution (min, max).
        m : int
            Number of corrections for L-BFGS.
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create least-squares problem
        prob = problem.LeastSquares(data, op)
        
        # Setup bounds
        if bounds is not None:
            lower = bounds[0] if bounds[0] is not None else -np.inf
            upper = bounds[1] if bounds[1] is not None else np.inf
            bounds_occ = occ.Bounds(lower, upper)
        else:
            bounds_occ = None
        
        # Solve with L-BFGS-B
        lbfgs_solver = solver.LBFGSB(
            bounds=bounds_occ,
            m=m,
            iter_lim=n_iter,
            tol=tol,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = lbfgs_solver.run(prob, x0)
        x = result.vector[:]
        
        info = {
            'method': 'lbfgsb',
            'iterations': result.info.get('iter', n_iter),
            'residual': result.info.get('residual', np.linalg.norm(R @ x - y)),
            'm_corrections': m
        }
        
        return x, info
    
    def _solve_ista(
        self,
        R: np.ndarray,
        y: np.ndarray,
        n_iter: int = 200,
        tol: float = 1e-5,
        tau: float = 0.01,
        mu: float = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using ISTA (Iterative Shrinkage-Thresholding Algorithm).
        
        ISTA solves: min ||R*x - y||^2 + tau*||x||_1
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        tau : float
            L1 regularization parameter.
        mu : float
            Step size (if None, computed automatically).
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create L1-regularized least-squares problem
        l1_term = occ.Lasso(tau)
        prob = problem.LeastSquaresRegularized(data, op, l1_term)
        
        # Solve with ISTA
        ista_solver = solver.ISTA(
            mu=mu,
            iter_lim=n_iter,
            tol=tol,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = ista_solver.run(prob, x0)
        x = result.vector[:]
        
        # Ensure non-negativity
        x = np.maximum(x, 0)
        
        info = {
            'method': 'ista',
            'iterations': result.info.get('iter', n_iter),
            'residual': np.linalg.norm(R @ x - y),
            'tau_l1': tau,
            'step_size': mu
        }
        
        return x, info
    
    def _solve_splitbregman(
        self,
        R: np.ndarray,
        y: np.ndarray,
        n_iter: int = 100,
        n_inner: int = 10,
        tau: float = 0.01,
        mu: float = 1.0,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using Split-Bregman algorithm for L1-regularized problems.
        
        Split-Bregman efficiently solves: min ||R*x - y||^2 + tau*||x||_1
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum outer iterations.
        n_inner : int
            Inner iterations per outer step.
        tau : float
            L1 regularization parameter.
        mu : float
            Augmented Lagrangian parameter.
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create L1 regularization term
        l1_term = occ.Lasso(tau)
        prob = problem.LeastSquaresRegularized(data, op, l1_term)
        
        # Solve with Split-Bregman
        sb_solver = solver.SplitBregman(
            mu=mu,
            iter_lim=n_iter,
            inner_iter=n_inner,
            tol=1e-5,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = sb_solver.run(prob, x0)
        x = result.vector[:]
        
        # Ensure non-negativity
        x = np.maximum(x, 0)
        
        info = {
            'method': 'splitbregman',
            'iterations': result.info.get('iter', n_iter),
            'inner_iterations': n_inner,
            'residual': np.linalg.norm(R @ x - y),
            'tau_l1': tau,
            'mu': mu
        }
        
        return x, info
    
    def _solve_cgls(
        self,
        R: np.ndarray,
        y: np.ndarray,
        n_iter: int = 100,
        tol: float = 1e-6,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using CGLS (Conjugate Gradient Least Squares).
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create least-squares problem
        prob = problem.LeastSquares(data, op)
        
        # Solve with CG
        cg_solver = solver.CG(
            iter_lim=n_iter,
            tol=tol,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = cg_solver.run(prob, x0)
        x = result.vector[:]
        
        info = {
            'method': 'cgls',
            'iterations': result.info.get('iter', n_iter),
            'residual': np.linalg.norm(R @ x - y)
        }
        
        return x, info
    
    def _solve_fista(
        self,
        R: np.ndarray,
        y: np.ndarray,
        n_iter: int = 200,
        tol: float = 1e-5,
        tau: float = 0.01,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using FISTA (Fast ISTA with Nesterov acceleration).
        
        Parameters
        ----------
        R : ndarray
            Response matrix.
        y : ndarray
            Data vector.
        n_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        tau : float
            L1 regularization parameter.
        
        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Convergence information.
        """
        op = self._create_operator(R)
        data = occ.Vector(y)
        
        # Create L1-regularized problem
        l1_term = occ.Lasso(tau)
        prob = problem.LeastSquaresRegularized(data, op, l1_term)
        
        # Solve with FISTA
        fista_solver = solver.FISTA(
            iter_lim=n_iter,
            tol=tol,
            show=False
        )
        
        # Initial guess
        x0 = occ.Vector(np.ones(R.shape[1]) * np.mean(y) / np.mean(R))
        
        result = fista_solver.run(prob, x0)
        x = result.vector[:]
        
        # Ensure non-negativity
        x = np.maximum(x, 0)
        
        info = {
            'method': 'fista',
            'iterations': result.info.get('iter', n_iter),
            'residual': np.linalg.norm(R @ x - y),
            'tau_l1': tau
        }
        
        return x, info
    
    def unfold(self, **kwargs) -> np.ndarray:
        """
        Perform spectrum unfolding.
        
        Parameters
        ----------
        **kwargs : dict
            Override solver parameters.
        
        Returns
        -------
        spectrum : ndarray
            Unfolded spectrum of shape (n_energy_groups,).
        """
        # Merge kwargs
        params = {**self.solver_kwargs, **kwargs}
        
        # Setup weighted problem
        R_w, y_w = self._setup_weighted_problem()
        
        # Select and run solver
        if self.method == 'lsqr':
            x, info = self._solve_lsqr(R_w, y_w, **params)
        elif self.method in ['lbfgs', 'lbfgsb']:
            x, info = self._solve_lbfgs(R_w, y_w, **params)
        elif self.method == 'ista':
            x, info = self._solve_ista(R_w, y_w, **params)
        elif self.method == 'splitbregman':
            x, info = self._solve_splitbregman(R_w, y_w, **params)
        elif self.method == 'cgls':
            x, info = self._solve_cgls(R_w, y_w, **params)
        elif self.method == 'fista':
            x, info = self._solve_fista(R_w, y_w, **params)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store results
        self.spectrum = x
        self.convergence_info = info
        
        # Estimate uncertainties using diagonal of covariance approximation
        # For iterative methods, use residual-based estimate
        residual = np.linalg.norm(R_w @ x - y_w)
        dof = max(len(y_w) - len(x), 1)
        variance_factor = (residual ** 2) / dof
        
        # Simple uncertainty estimate based on sensitivity
        R_pinv = np.linalg.pinv(R_w)
        self.spectrum_unc = np.sqrt(np.diag(R_pinv @ R_pinv.T) * variance_factor)
        
        return self.spectrum
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence information.
        
        Returns
        -------
        info : dict
            Dictionary with convergence details.
        """
        return self.convergence_info


def unfold_occamy_lsqr(
    response_matrix: np.ndarray,
    counts: np.ndarray,
    counts_unc: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-6,
    damp: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unfold spectrum using OccamPy LSQR algorithm.
    
    Parameters
    ----------
    response_matrix : ndarray
        Response matrix of shape (n_channels, n_energy_groups).
    counts : ndarray
        Measured counts of shape (n_channels,).
    counts_unc : ndarray
        Count uncertainties of shape (n_channels,).
    n_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    damp : float
        Damping parameter.
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    spectrum : ndarray
        Unfolded spectrum.
    spectrum_unc : ndarray
        Spectrum uncertainties.
    info : dict
        Convergence information.
    """
    unfolder = OccamPyUnfolder(
        response_matrix, counts, counts_unc,
        method='lsqr',
        n_iter=n_iter,
        tol=tol,
        damp=damp,
        **kwargs
    )
    spectrum = unfolder.unfold()
    return spectrum, unfolder.spectrum_unc, unfolder.get_convergence_info()


def unfold_occamy_lbfgs(
    response_matrix: np.ndarray,
    counts: np.ndarray,
    counts_unc: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-6,
    bounds: Tuple[float, float] = (0, None),
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unfold spectrum using OccamPy L-BFGS-B algorithm.
    
    Parameters
    ----------
    response_matrix : ndarray
        Response matrix of shape (n_channels, n_energy_groups).
    counts : ndarray
        Measured counts of shape (n_channels,).
    counts_unc : ndarray
        Count uncertainties of shape (n_channels,).
    n_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    bounds : tuple
        Solution bounds (min, max).
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    spectrum : ndarray
        Unfolded spectrum.
    spectrum_unc : ndarray
        Spectrum uncertainties.
    info : dict
        Convergence information.
    """
    unfolder = OccamPyUnfolder(
        response_matrix, counts, counts_unc,
        method='lbfgs',
        n_iter=n_iter,
        tol=tol,
        bounds=bounds,
        **kwargs
    )
    spectrum = unfolder.unfold()
    return spectrum, unfolder.spectrum_unc, unfolder.get_convergence_info()


def unfold_occamy_ista(
    response_matrix: np.ndarray,
    counts: np.ndarray,
    counts_unc: np.ndarray,
    n_iter: int = 200,
    tau: float = 0.01,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unfold spectrum using OccamPy ISTA algorithm with L1 regularization.
    
    Parameters
    ----------
    response_matrix : ndarray
        Response matrix of shape (n_channels, n_energy_groups).
    counts : ndarray
        Measured counts of shape (n_channels,).
    counts_unc : ndarray
        Count uncertainties of shape (n_channels,).
    n_iter : int
        Maximum iterations.
    tau : float
        L1 regularization parameter.
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    spectrum : ndarray
        Unfolded spectrum.
    spectrum_unc : ndarray
        Spectrum uncertainties.
    info : dict
        Convergence information.
    """
    unfolder = OccamPyUnfolder(
        response_matrix, counts, counts_unc,
        method='ista',
        n_iter=n_iter,
        tau=tau,
        **kwargs
    )
    spectrum = unfolder.unfold()
    return spectrum, unfolder.spectrum_unc, unfolder.get_convergence_info()


def unfold_occamy_splitbregman(
    response_matrix: np.ndarray,
    counts: np.ndarray,
    counts_unc: np.ndarray,
    n_iter: int = 100,
    tau: float = 0.01,
    mu: float = 1.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unfold spectrum using OccamPy Split-Bregman algorithm.
    
    Parameters
    ----------
    response_matrix : ndarray
        Response matrix of shape (n_channels, n_energy_groups).
    counts : ndarray
        Measured counts of shape (n_channels,).
    counts_unc : ndarray
        Count uncertainties of shape (n_channels,).
    n_iter : int
        Maximum iterations.
    tau : float
        L1 regularization parameter.
    mu : float
        Augmented Lagrangian parameter.
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    spectrum : ndarray
        Unfolded spectrum.
    spectrum_unc : ndarray
        Spectrum uncertainties.
    info : dict
        Convergence information.
    """
    unfolder = OccamPyUnfolder(
        response_matrix, counts, counts_unc,
        method='splitbregman',
        n_iter=n_iter,
        tau=tau,
        mu=mu,
        **kwargs
    )
    spectrum = unfolder.unfold()
    return spectrum, unfolder.spectrum_unc, unfolder.get_convergence_info()
