"""Detector class with unfolding methods."""
from __future__ import annotations

from datetime import datetime
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import odl
import pandas as pd
from scipy.interpolate import pchip
from scipy.optimize import minimize, nnls

from .unfolding_helpers import calculate_spectrum_maxed_core, gravel, gravel_with_errors

logger = logging.getLogger(__name__)


class Detector:
    """
    Class for neutron detector operations and spectrum unfolding.

    This class provides methods for neutron spectrum unfolding using various
    algorithms and includes tools for dose rate calculations based on ICRP-116
    conversion coefficients.

    Parameters
    ----------
    response_functions_df : pd.DataFrame
        DataFrame containing detector response functions. The first column should
        be 'E_MeV' (energy in MeV) and subsequent columns contain response
        functions for different detector spheres.

    Attributes
    ----------
    Amat : np.ndarray
        Response matrix with logarithmic energy step corrections
    E_MeV : np.ndarray
        Energy grid in MeV
    detector_names : List[str]
        Names of available detectors/spheres
    log_steps : np.ndarray
        Logarithmic steps for each energy point
    sensitivities : Dict[str, np.ndarray]
        Dictionary mapping detector names to their sensitivity arrays
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients for dose calculation
    n_detectors : int
        Number of available detectors (property)
    n_energy_bins : int
        Number of energy bins (property)

    Examples
    --------
    >>> import pandas as pd
    >>> from bssunfold import Detector
    >>> # Load response functions from CSV
    >>> rf_df = pd.read_csv('response_functions.csv')
    >>> detector = Detector(rf_df)
    >>> # Perform unfolding
    >>> readings = {'sphere_1': 100.5, 'sphere_2': 85.3}
    >>> result = detector.unfold_cvxpy(readings)
    """

    def __init__(self, response_functions_df):
        """
        Initialize Detector with response functions.

        Parameters
        ----------
        response_functions_df : pd.DataFrame
            DataFrame containing response functions with 'E_MeV'
            as first column
            and detector names as subsequent columns.

        Raises
        ------
        ValueError
            If E_MeV is not a 1D array or has less than 2 energy points
        """
        Amat, E_MeV, detector_names, log_steps = (
            self._convert_rf_to_matrix_variable_step(
                response_functions_df, Emin=1e-9
            )
        )

        self.Amat = Amat
        self.E_MeV = np.asarray(E_MeV, dtype=float)
        self.detector_names = detector_names
        self.log_steps = log_steps

        if self.E_MeV.ndim != 1:
            raise ValueError("E_MeV must be a 1D array")
        if len(self.E_MeV) < 2:
            raise ValueError("At least 2 energy bins are required")

        self.sensitivities = {
            self.detector_names[i]: np.array(Amat[:, i])
            for i in range(len(self.detector_names))
        }
        self.cc_icrp116 = self._load_icrp116_coefficients()

        # Initialize results storage
        self.results_history = {}
        self.current_result = None

    def __str__(self) -> str:
        """
        User-friendly string representation of the detector.

        Returns
        -------
        str
            Human-readable information about the detector
        """
        energy_range = f"{self.E_MeV[0]:.3e} - {self.E_MeV[-1]:.3e} MeV"
        return (
            f"Detector(energy bins: {self.n_energy_bins}, "
            f"detectors: {self.n_detectors}, "
            f"range: {energy_range})"
        )

    def __repr__(self) -> str:
        """
        Technical string representation for object recreation.

        Returns
        -------
        str
            String that can be used to recreate the object
        """
        return f"Detector(E_MeV={self.E_MeV.tolist()}, sensitivities={self.sensitivities})"

    @property
    def n_detectors(self) -> int:
        """Number of available detectors."""
        return len(self.detector_names)

    @property
    def n_energy_bins(self) -> int:
        """Number of energy bins."""
        return len(self.E_MeV)
    
    def get_response_matrix(self, readings: Dict[str, float]) -> np.ndarray:
        """Return response matrix for given readings (spheres)."""
        selected = [name for name in self.detector_names if name in readings]
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)
        return A

    def _save_result(self, result: Dict[str, Any]) -> str:
        """
        Save unfolding result to history with timestamp.

        Parameters
        ----------
        result : Dict[str, Any]
            Unfolding result dictionary

        Returns
        -------
        str
            Key under which result was saved (timestamp + method)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = result.get("method", "unknown")
        key = f"{timestamp}_{method}"

        # Add timestamp to result
        result["timestamp"] = timestamp
        result["saved_key"] = key

        # Store in history
        self.results_history[key] = result.copy()
        self.current_result = result

        logger.info("Result saved with key: %s", key)
        return key

    def get_result(self, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get unfolding result from history.

        Parameters
        ----------
        key : Optional[str], optional
            Result key. If None, returns current result

        Returns
        -------
        Optional[Dict[str, Any]]
            Unfolding result dictionary or None if not found

        Examples
        --------
        >>> result = detector.get_result('20240115_143022_cvxpy')
        >>> detector.get_result()  # Returns current result
        """
        if key is None:
            return self.current_result
        return self.results_history.get(key)

    def list_results(self) -> List[str]:
        """
        List all saved result keys.

        Returns
        -------
        List[str]
            List of result keys sorted by timestamp

        Examples
        --------
        >>> keys = detector.list_results()
        >>> for key in keys:
        ...     print(key)
        """
        return sorted(self.results_history.keys())

    def clear_results(self) -> None:
        """Clear all saved results."""
        self.results_history.clear()
        self.current_result = None
        logger.info("All results cleared.")

    def _validate_readings(
        self, readings: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Validate detector readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings dictionary

        Returns
        -------
        Dict[str, float]
            Validated readings

        Raises
        ------
        ValueError
            If readings are negative or no detector readings are provided
        """
        valid = {}
        for det in self.detector_names:
            if det in readings:
                val = float(readings[det])
                if val < 0:
                    raise ValueError(f"Reading '{det}' is negative: {val}")
                valid[det] = val
        if not valid:
            raise ValueError("No detector readings provided")
        return valid

    def _build_system(
        self, readings: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build response matrix A and measurement vector b.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            A: Response matrix
            b: Measurement vector
            selected: List of selected detector names
        """
        selected = [name for name in self.detector_names if name in readings]
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array(
            [self.sensitivities[name] for name in selected], dtype=float
        )
        return A, b, selected

    def _standardize_output(
        self,
        spectrum: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        selected: List[str],
        method: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create standardized output dictionary for all unfolding methods.

        Parameters
        ----------
        spectrum : np.ndarray
            Unfolded spectrum
        A : np.ndarray
            Response matrix
        b : np.ndarray
            Measurement vector
        selected : List[str]
            Selected detector names
        method : str
            Unfolding method name
        **kwargs : dict
            Additional parameters to include in output

        Returns
        -------
        Dict[str, Any]
            Standardized output dictionary
        """
        # Ensure non-negative spectrum
        spectrum_nonneg = np.maximum(spectrum, 0)

        computed_readings = A @ spectrum_nonneg
        residual = b - computed_readings

        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": spectrum_nonneg.copy(),
            "spectrum_absolute": spectrum_nonneg.copy(),
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, computed_readings)
            },
            "residual": residual.copy(),
            "residual_norm": float(np.linalg.norm(residual)),
            "method": method,
            "doserates": self._calculate_doserates(spectrum_nonneg),
        }
        output.update(kwargs)
        return output

    def _validate_initial_spectrum(
        self,
        initial_spectrum: Optional[np.ndarray],
        default_value: float = 0.0,
    ) -> np.ndarray:
        """Validate and prepare an initial spectrum guess."""
        if initial_spectrum is None:
            return np.full(self.n_energy_bins, default_value, dtype=float)
        if len(initial_spectrum) != self.n_energy_bins:
            raise ValueError(
                f"Initial spectrum length ({len(initial_spectrum)}) "
                f"must match number of energy bins ({self.n_energy_bins})"
            )
        return np.maximum(np.asarray(initial_spectrum, dtype=float), 0.0)

    @staticmethod
    def _uncertainty_stats(samples: np.ndarray) -> Dict[str, Any]:
        """Compute summary statistics from Monte-Carlo samples."""
        return {
            "spectrum_uncert_mean": np.mean(samples, axis=0),
            "spectrum_uncert_std": np.std(samples, axis=0),
            "spectrum_uncert_min": np.min(samples, axis=0),
            "spectrum_uncert_max": np.max(samples, axis=0),
            "spectrum_uncert_median": np.median(samples, axis=0),
            "spectrum_uncert_percentile_5": np.percentile(samples, 5, axis=0),
            "spectrum_uncert_percentile_95": np.percentile(samples, 95, axis=0),
            "spectrum_uncert_all": samples,
        }

    def _monte_carlo_uncertainty(
        self,
        readings: Dict[str, float],
        solver_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        noise_level: float,
        n_montecarlo: int,
    ) -> Dict[str, Any]:
        """Run Monte-Carlo uncertainty propagation for a given solver."""
        samples = np.zeros((n_montecarlo, self.n_energy_bins))
        for i in range(n_montecarlo):
            noisy_readings = self._add_noise(readings, noise_level)
            A_noisy, b_noisy, _ = self._build_system(noisy_readings)
            samples[i] = solver_fn(A_noisy, b_noisy)

        stats = self._uncertainty_stats(samples)
        stats["montecarlo_samples"] = n_montecarlo
        stats["noise_level"] = noise_level
        return stats

    @staticmethod
    def _import_optional(module_name: str, purpose: str) -> Any:
        """
        Import an optional dependency with a clear error message.

        Parameters
        ----------
        module_name : str
            Module name to import.
        purpose : str
            Short description of what the dependency is needed for.

        Returns
        -------
        Any
            Imported module.
        """
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Optional dependency '{module_name}' is required for {purpose}."
            ) from exc

    def _convert_rf_to_matrix_variable_step(self, rf_df, Emin=1e-9) -> tuple:
        """
        Convert response functions DataFrame to matrix with variable step correction.

        Multiplies by np.log(10) and individual logarithmic energy step for each point.

        Parameters
        ----------
        rf_df : pd.DataFrame
            DataFrame with response functions.
            First column 'E_MeV' contains energies in MeV.
            Other columns contain response functions for different spheres.
        Emin : float, optional
            Minimum energy for logarithmic scaling, default: 1e-9

        Returns
        -------
        tuple: (matrix, energies, sphere_names, log_steps)
            matrix : np.ndarray
                Matrix of size (n_energies, n_spheres)
            energies : np.ndarray
                Array of energies in MeV
            sphere_names : list
                List of sphere names
            log_steps : np.ndarray
                Array of logarithmic steps for each point
        """
        # Extract energies
        if "E_MeV" in rf_df.columns:
            energies = rf_df["E_MeV"].values
            rf_data = rf_df.drop("E_MeV", axis=1)
        else:
            # Assume first column is energy
            energies = rf_df.iloc[:, 0].values
            rf_data = rf_df.iloc[:, 1:]

        # Get sphere names
        sphere_names = rf_data.columns.tolist()

        # Convert to numpy array
        rf_array = rf_data.values  # size: (n_energies, n_spheres)

        # Calculate energy logarithms
        log_energies = np.log10(energies / Emin)

        # Calculate logarithmic steps for each point
        n_points = len(energies)
        log_steps = np.zeros(n_points)

        # For first point
        log_steps[0] = log_energies[1] - log_energies[0]

        # For last point
        log_steps[-1] = log_energies[-1] - log_energies[-2]

        # For interior points
        for i in range(1, n_points - 1):
            # Average step between left and right intervals
            left_step = log_energies[i] - log_energies[i - 1]
            right_step = log_energies[i + 1] - log_energies[i]
            log_steps[i] = (left_step + right_step) / 2

        # Convert to natural logarithm steps
        ln_steps = log_steps * np.log(10)

        # Multiply each row by corresponding step
        rf_matrix = rf_array * ln_steps[:, np.newaxis]

        return rf_matrix, energies, sphere_names, log_steps

    def unfold_cvxpy(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "default",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using convex optimization (cvxpy).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, uniform spectrum is used.
        regularization : float, optional
            Regularization parameter, default: 1e-4
        norm : int, optional
            Norm type for regularization (1 for L1, 2 for L2), default: 2
        solver : str, optional
            Solver to use ('ECOS' or 'default'), default: 'default'
        calculate_errors : bool, optional
            Flag to calculate unfolding errors via Monte-Carlo, default: False

        Returns
        -------
        Dict[str, Any]
            Dictionary containing unfolding results with keys:
            - 'energy': Energy grid [MeV]
            - 'spectrum': Unfolded spectrum [counts/bin]
            - 'spectrum_absolute': Absolute unfolded spectrum
            - 'effective_readings': Calculated readings from unfolded spectrum
            - 'residual': Difference between measured and calculated readings
            - 'residual_norm': L2 norm of residual
            - 'method': Method name ('cvxpy')
            - 'doserates': Dose rates for different geometries [pSv/s]
            - 'spectrum_uncert_*': Uncertainty estimates (if calculate_errors=True)

        Raises
        ------
        ValueError
            If readings are invalid or dimensions mismatch

        Examples
        --------
        >>> readings = {'sphere_1': 150.2, 'sphere_2': 120.5, 'sphere_3': 95.7}
        >>> result = detector.unfold_cvxpy(
        ...     readings,
        ...     regularization=0.001,
        ...     calculate_errors=True
        ... )
        >>> print(f"Spectrum length: {len(result['spectrum'])}")
        >>> print(f"Residual norm: {result['residual_norm']:.3f}")
        """

        def _solve_problem(
            A: np.ndarray, b: np.ndarray, use_solver: Optional[str] = None
        ) -> np.ndarray:
            """Solve optimization problem."""
            x = cp.Variable(A.shape[1], nonneg=True)
            objective = cp.Minimize(
                cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm)
            )
            problem = cp.Problem(objective)

            if use_solver == "ECOS":
                problem.solve(solver=cp.ECOS)
            else:
                problem.solve()

            logger.info("CVXPY status: %s", problem.status)
            logger.debug("CVXPY objective value: %s", problem.value)
            return x.value

        # Validate and solve
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        alpha = regularization
        _ = initial_spectrum  # kept for API compatibility

        # Main solution
        x_value = _solve_problem(A, b, solver)

        # Create main output
        output = self._standardize_output(
            spectrum=x_value,
            A=A,
            b=b,
            selected=selected,
            method="cvxpy",
            norm=norm,
            solver=solver,
        )

        # Monte-Carlo error estimation
        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    readings,
                    solver_fn=lambda A_mc, b_mc: _solve_problem(
                        A_mc, b_mc, solver
                    ),
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")

        self._save_result(output)
        return output
 
    def unfold_landweber(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Landweber iteration method.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings (counts or dose rates)
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, uniform spectrum is used
        max_iterations : int, optional
            Maximum number of iterations, default: 1000
        tolerance : float, optional
            Convergence tolerance for residual norm, default: 1e-6
        calculate_errors : bool, optional
            Flag to calculate uncertainty via Monte-Carlo, default: False
        noise_level : float, optional
            Noise level for Monte-Carlo uncertainty calculation, default: 0.01
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'energy': Energy grid [MeV]
            - 'spectrum': Unfolded spectrum [counts/bin]
            - 'spectrum_absolute': Absolute unfolded spectrum
            - 'effective_readings': Calculated readings from unfolded spectrum
            - 'residual': Difference between measured and calculated readings
            - 'residual_norm': L2 norm of residual
            - 'method': Method name ('Landweber')
            - 'iterations': Number of iterations performed
            - 'converged': Whether convergence was achieved
            - 'doserates': Dose rates for different geometries [pSv/s]
            - 'spectrum_uncert_*': Monte-Carlo uncertainty estimates
                (if calculate_errors=True)

        Raises
        ------
        ValueError
            If readings are invalid or dimensions mismatch

        Examples
        --------
        >>> readings = {'sphere_1': 150.2, 'sphere_2': 120.5}
        >>> result = detector.unfold_landweber(
        ...     readings,
        ...     max_iterations=500,
        ...     tolerance=1e-5,
        ...     calculate_errors=True
        ... )
        >>> print(f"Converged: {result['converged']}")
        >>> print(f"Iterations: {result['iterations']}")
        """

        def _landweber_iteration(
            A: np.ndarray,
            b: np.ndarray,
            x0: np.ndarray,
            max_iter: int,
            tol: float,
        ) -> Tuple[np.ndarray, int, bool]:
            """Core Landweber iteration implementation."""
            # n = A.shape[1]
            x = x0.copy()

            # Calculate optimal step size
            sigma_max = np.linalg.norm(A, 2)
            step_size = 1.0 / (sigma_max**2)

            # Precompute A^T for efficiency
            AT = A.T

            converged = False
            iterations = 0

            for i in range(max_iter):
                # Compute residual
                residual = A @ x - b
                residual_norm = np.linalg.norm(residual)

                # Check convergence
                if residual_norm < tol:
                    converged = True
                    iterations = i
                    break

                # Landweber update
                x = x - step_size * (AT @ residual)

                # Apply non-negativity constraint
                x = np.maximum(x, 0)

            if not converged:
                iterations = max_iter

            return x, iterations, converged

        # Validate and prepare data
        validated_readings = self._validate_readings(readings)
        A, b, selected = self._build_system(validated_readings)

        # Set initial spectrum
        x0 = self._validate_initial_spectrum(initial_spectrum, default_value=0.0)

        # Main Landweber iteration
        x_opt, n_iter, converged = _landweber_iteration(
            A, b, x0, max_iterations, tolerance
        )

        # Create standard output
        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method="Landweber",
            iterations=n_iter,
            converged=converged,
            tolerance=tolerance,
        )

        # Monte-Carlo uncertainty estimation
        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    validated_readings,
                    solver_fn=lambda A_mc, b_mc: _landweber_iteration(
                        A_mc, b_mc, x0, max_iterations, tolerance
                    )[0],
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")
        self._save_result(output)
        return output

    def unfold_mlem_odl(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict:
        """
        Unfold the neutron spectrum using the Maximum
        Likelihood Expectation Maximization algorithm.
        poisson_log_likelihood – Poisson log-likelihood.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum approximation. If None, a uniform spectrum is used.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.
        max_iterations : int, optional
            Maximum number of iterations. Default is 1000.
        calculate_errors : bool, optional
            Flag for calculating restoration errors. Default is False.
        noise_level : float, optional
            Noise level for error calculation. Default is 0.01.
        n_montecarlo : int, optional
            Number of Monte Carlo samples for error calculation. Default is 100.

        Returns
        -------
        Dict
            Dictionary containing the spectrum restoration results.
        """
        def _create_odl_spaces(b_size: int):
            """Create ODL spaces for the given measurement size."""
            measurement_space = odl.uniform_discr(0, b_size - 1, b_size)
            spectrum_space = odl.uniform_discr(
                np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
            )
            return measurement_space, spectrum_space

        def _initialize_spectrum(spectrum_space, initial_spectrum=None):
            if initial_spectrum is None:
                # 0.5 prevents collapse to zero in ODL MLEM.
                return spectrum_space.element(0.5)
            return spectrum_space.element(initial_spectrum)

        def _run_mlem(A_matrix, b_vector, initial_spectrum_vals=None):
            """Run MLEM for the given system."""
            measurement_space, spectrum_space = _create_odl_spaces(len(b_vector))

            operator = odl.MatrixOperator(
                A_matrix, 
                domain=spectrum_space, 
                range=measurement_space
            )

            y = measurement_space.element(b_vector)
            x = _initialize_spectrum(spectrum_space, initial_spectrum_vals)

            odl.solvers.mlem(
                operator, 
                x, 
                y, 
                niter=max_iterations, 
                callback=callback
            )

            return x.asarray(), A_matrix, b_vector

        callback = odl.solvers.CallbackPrintIteration()

        # Validate and prepare data
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        # Main MLEM run
        unfolded_spectrum, A, b = _run_mlem(A, b, initial_spectrum)

        # Main output
        output = self._standardize_output(
            spectrum=unfolded_spectrum,
            A=A,
            b=b,
            selected=selected,
            method="MLEM (ODL)",
            iterations=max_iterations,
        )

        # Monte-Carlo uncertainty estimation
        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    readings,
                    solver_fn=lambda A_mc, b_mc: _run_mlem(
                        A_mc, b_mc, initial_spectrum
                    )[0],
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")
        
        self._save_result(output)
        return output

    def unfold_nnls(
        self,
        readings: Dict[str, float],
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Non-Negative Least Squares (NNLS).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        calculate_errors : bool, optional
            Flag to calculate uncertainty via Monte-Carlo, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        spectrum, _ = nnls(A, b)

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="NNLS",
        )

        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    readings,
                    solver_fn=lambda A_mc, b_mc: nnls(A_mc, b_mc)[0],
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")

        self._save_result(output)
        return output

    def unfold_tikhonov(
        self,
        readings: Dict[str, float],
        regularization: float = 1e-2,
        regularization_matrix: str = "identity",
        nonneg: bool = True,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Tikhonov regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        regularization : float, optional
            Regularization strength, default: 1e-2.
        regularization_matrix : str, optional
            Regularization operator: 'identity', 'first_derivative',
            or 'second_derivative'. Default: 'identity'.
        nonneg : bool, optional
            Enforce non-negativity after solution, default: True.
        calculate_errors : bool, optional
            Flag to calculate uncertainty via Monte-Carlo, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        def _build_regularization_matrix(kind: str, n: int) -> np.ndarray:
            if kind == "identity":
                return np.eye(n)
            if kind == "first_derivative":
                if n < 2:
                    raise ValueError("first_derivative requires at least 2 bins")
                L = np.zeros((n - 1, n))
                for i in range(n - 1):
                    L[i, i] = -1.0
                    L[i, i + 1] = 1.0
                return L
            if kind == "second_derivative":
                if n < 3:
                    raise ValueError("second_derivative requires at least 3 bins")
                L = np.zeros((n - 2, n))
                for i in range(n - 2):
                    L[i, i] = 1.0
                    L[i, i + 1] = -2.0
                    L[i, i + 2] = 1.0
                return L
            raise ValueError(
                "regularization_matrix must be one of: "
                "'identity', 'first_derivative', 'second_derivative'"
            )

        def _solve_tikhonov(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            n = A.shape[1]
            L = _build_regularization_matrix(regularization_matrix, n)
            lhs = A.T @ A + regularization * (L.T @ L)
            rhs = A.T @ b
            try:
                x = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            if nonneg:
                x = np.maximum(x, 0.0)
            return x

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        spectrum = _solve_tikhonov(A, b)

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="Tikhonov",
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            nonneg=nonneg,
        )

        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    readings,
                    solver_fn=_solve_tikhonov,
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")

        self._save_result(output)
        return output

    def unfold_tsvd(
        self,
        readings: Dict[str, float],
        n_components: Optional[int] = None,
        cutoff: Optional[float] = 1e-3,
        nonneg: bool = True,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Truncated SVD (TSVD).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        n_components : Optional[int], optional
            Number of singular values to keep. If None, use cutoff.
        cutoff : Optional[float], optional
            Relative cutoff for singular values (s / s_max). Default: 1e-3.
        nonneg : bool, optional
            Enforce non-negativity after solution, default: True.
        calculate_errors : bool, optional
            Flag to calculate uncertainty via Monte-Carlo, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        def _solve_tsvd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            if s.size == 0 or s[0] == 0:
                return np.zeros(A.shape[1], dtype=float)
            if n_components is not None:
                k = max(1, min(int(n_components), len(s)))
            else:
                if cutoff is None:
                    k = len(s)
                else:
                    s_rel = s / s[0]
                    k = max(1, int(np.sum(s_rel >= cutoff)))

            Uk = U[:, :k]
            sk = s[:k]
            Vk = Vt[:k, :]
            x = Vk.T @ ((Uk.T @ b) / sk)
            if nonneg:
                x = np.maximum(x, 0.0)
            return x

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        spectrum = _solve_tsvd(A, b)

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="TSVD",
            n_components=n_components,
            cutoff=cutoff,
            nonneg=nonneg,
        )

        if calculate_errors:
            logger.info(
                "Calculating uncertainty with %d Monte-Carlo samples...",
                n_montecarlo,
            )
            output.update(
                self._monte_carlo_uncertainty(
                    readings,
                    solver_fn=_solve_tsvd,
                    noise_level=noise_level,
                    n_montecarlo=n_montecarlo,
                )
            )
            logger.info("Uncertainty calculation completed.")

        self._save_result(output)
        return output

    # --- Unfolding methods: external/advanced ---

    def unfold_tikhonov_legendre(
        self,
        readings: Dict[str, float],
        delta: float = 0.05,
        n_polynomials: int = 15,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Tikhonov regularization with Legendre basis.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        delta : float, optional
            Regularization parameter, by default 0.05.
        n_polynomials : int, optional
            Number of Legendre polynomials, by default 15.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        try:
            from spectrum_recovery import calculate_spectrum_tikhonov_core
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'spectrum_recovery' for Tikhonov-Legendre."
            ) from exc

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        result = calculate_spectrum_tikhonov_core(
            E_MeV=self.E_MeV,
            K_j=A,
            Q_j=b,
            delta=delta,
            n_polynomials=n_polynomials,
        )

        spectrum = np.asarray(result.get("spectrum"), dtype=float)
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="Tikhonov-Legendre",
            delta=delta,
            n_polynomials=n_polynomials,
        )
        output.update(
            {
                "alpha": result.get("alpha"),
                "coefficients": result.get("coefficients"),
            }
        )
        self._save_result(output)
        return output

    def unfold_maxed(
        self,
        readings: Dict[str, float],
        reference_spectrum: Optional[Dict[str, np.ndarray]] = None,
        sigma_factor: float = 0.01,
        omega: float = 1.0,
        maxiter: int = 5000,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the MAXED (maximum entropy) method.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        reference_spectrum : Optional[Dict[str, np.ndarray]], optional
            Reference spectrum with keys 'E_MeV' and 'Phi'. If None, a default
            reference spectrum is used.
        sigma_factor : float, optional
            Relative measurement uncertainty, by default 0.01.
        omega : float, optional
            Chi-square constraint parameter, by default 1.0.
        maxiter : int, optional
            Maximum iterations, by default 5000.
        tol : float, optional
            Convergence tolerance, by default 1e-6.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        phi_0 = None
        if reference_spectrum is not None:
            ref_E = np.asarray(reference_spectrum["E_MeV"], dtype=float)
            ref_phi = np.asarray(reference_spectrum["Phi"], dtype=float)
            if len(ref_E) != len(ref_phi):
                raise ValueError("reference_spectrum E_MeV and Phi must match in length")
            phi_0 = np.interp(
                np.log10(self.E_MeV),
                np.log10(np.maximum(ref_E, self.E_MeV.min())),
                ref_phi,
                left=1e-10,
                right=1e-10,
            )
            phi_0 = np.maximum(phi_0, 1e-10)

        result = calculate_spectrum_maxed_core(
            E_MeV=self.E_MeV,
            K_j=A,
            Q_j=b,
            phi_0=phi_0,
            sigma_factor=sigma_factor,
            omega=omega,
            maxiter=maxiter,
            tol=tol,
        )

        spectrum = np.asarray(result["spectrum"], dtype=float)
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="MAXED",
            omega=result.get("omega"),
            mu=result.get("mu"),
            chi_square=result.get("chi_square"),
        )
        self._save_result(output)
        return output

    def unfold_gravel(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the GRAVEL algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, uses a flat spectrum.
        tolerance : float, optional
            Convergence tolerance, by default 1e-8.
        max_iterations : int, optional
            Maximum iterations, by default 1000.
        regularization : float, optional
            Regularization strength, by default 0.0.
        calculate_errors : bool, optional
            Whether to estimate spectrum errors, by default False.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            x0 = np.ones(self.n_energy_bins, dtype=float)
        else:
            x0 = np.asarray(initial_spectrum, dtype=float)
            if len(x0) != self.n_energy_bins:
                raise ValueError(
                    "initial_spectrum length must match number of energy bins"
                )

        if calculate_errors:
            results = gravel_with_errors(
                S=A,
                measurements=b,
                x0=x0,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
            )
        else:
            results = gravel(
                S=A,
                measurements=b,
                x0=x0,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
            )

        spectrum = np.asarray(results["spectrum_absolute"], dtype=float)
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="GRAVEL",
            iterations=results.get("iterations"),
            converged=results.get("converged"),
        )

        for key in (
            "error_history",
            "chi_sq_history",
            "spectrum_errors",
            "covariance_matrix",
            "correlation_matrix",
        ):
            if key in results:
                output[key] = results[key]

        self._save_result(output)
        return output

    def unfold_tikhonov_legendre_maxed(
        self,
        readings: Dict[str, float],
        delta: float = 0.05,
        n_polynomials: int = 15,
        sigma_factor: float = 0.01,
        omega: float = 1.0,
        maxiter: int = 5000,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Hybrid unfolding: Tikhonov-Legendre followed by MAXED.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        delta : float, optional
            Tikhonov regularization parameter.
        n_polynomials : int, optional
            Number of Legendre polynomials.
        sigma_factor : float, optional
            Relative measurement uncertainty for MAXED.
        omega : float, optional
            Chi-square constraint parameter.
        maxiter : int, optional
            Maximum iterations for MAXED.
        tol : float, optional
            Convergence tolerance for MAXED.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        tikh = self.unfold_tikhonov_legendre(readings, delta, n_polynomials)
        prior = {"E_MeV": tikh["energy"], "Phi": tikh["spectrum"]}
        maxed = self.unfold_maxed(
            readings,
            reference_spectrum=prior,
            sigma_factor=sigma_factor,
            omega=omega,
            maxiter=maxiter,
            tol=tol,
        )
        maxed["tikhonov_alpha"] = tikh.get("alpha")
        maxed["tikhonov_coefficients"] = tikh.get("coefficients")
        return maxed

    def unfold_tikhonov_legendre_gravel(
        self,
        readings: Dict[str, float],
        delta: float = 0.05,
        n_polynomials: int = 15,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict[str, Any]:
        """
        Hybrid unfolding: Tikhonov-Legendre followed by GRAVEL.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        delta : float, optional
            Tikhonov regularization parameter.
        n_polynomials : int, optional
            Number of Legendre polynomials.
        tolerance : float, optional
            GRAVEL convergence tolerance.
        max_iterations : int, optional
            GRAVEL maximum iterations.
        regularization : float, optional
            GRAVEL regularization strength.
        calculate_errors : bool, optional
            Whether to compute GRAVEL error estimates.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        tikh = self.unfold_tikhonov_legendre(readings, delta, n_polynomials)
        gravel_result = self.unfold_gravel(
            readings,
            initial_spectrum=tikh["spectrum"],
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            calculate_errors=calculate_errors,
        )
        gravel_result["tikhonov_alpha"] = tikh.get("alpha")
        gravel_result["tikhonov_coefficients"] = tikh.get("coefficients")
        gravel_result["method"] = "Tikhonov-Legendre + GRAVEL"
        return gravel_result

    def unfold_mlem(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Maximum Likelihood (MLE) optimization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        method : str, optional
            SciPy optimizer method name.
        max_iterations : int, optional
            Maximum optimization iterations.
        tolerance : float, optional
            Optimization tolerance.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            spectrum_initial = np.ones(self.n_energy_bins) * np.mean(b) / np.mean(A)
        else:
            spectrum_initial = np.asarray(initial_spectrum, dtype=float)
            if len(spectrum_initial) != self.n_energy_bins:
                raise ValueError(
                    "initial_spectrum length must match number of energy bins"
                )

        def negative_log_likelihood(spectrum: np.ndarray) -> float:
            predicted = A @ spectrum
            epsilon = 1e-10
            log_likelihood = np.sum(b * np.log(predicted + epsilon) - predicted)
            return -log_likelihood

        def gradient_negative_log_likelihood(spectrum: np.ndarray) -> np.ndarray:
            predicted = A @ spectrum
            grad = -A.T @ (b / (predicted + 1e-10) - 1)
            return grad

        bounds = [(1e-10, None)] * self.n_energy_bins
        result = minimize(
            negative_log_likelihood,
            spectrum_initial,
            method=method,
            bounds=bounds,
            jac=gradient_negative_log_likelihood,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        output = self._standardize_output(
            spectrum=result.x,
            A=A,
            b=b,
            selected=selected,
            method="MLE (scipy)",
            iterations=result.nit,
            converged=result.success,
            optimization_result={
                "fun": result.fun,
                "message": result.message,
                "nfev": result.nfev,
                "njev": result.njev,
            },
        )
        self._save_result(output)
        return output

    def unfold_doroshenko(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Doroshenko coordinate update method.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        tolerance : float, optional
            Convergence tolerance.
        max_iterations : int, optional
            Maximum iterations.
        regularization : float, optional
            Regularization strength.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            x = np.ones(self.n_energy_bins)
        else:
            x = np.asarray(initial_spectrum, dtype=float)
            if x.size != self.n_energy_bins:
                raise ValueError(
                    "initial_spectrum length must match number of energy bins"
                )

        denominator_cache = np.sum(A * A, axis=0)

        for iteration in range(max_iterations):
            x_old = x.copy()
            for i in range(x.size):
                ax_without_i = A[:, :i] @ x[:i] + A[:, i + 1 :] @ x[i + 1 :]
                numerator = np.dot(A[:, i], b - ax_without_i)
                denominator = denominator_cache[i] + regularization
                if denominator > 0:
                    x[i] = max(0.0, numerator / denominator)
            if np.linalg.norm(x - x_old) < tolerance:
                break

        output = self._standardize_output(
            spectrum=x,
            A=A,
            b=b,
            selected=selected,
            method="Doroshenko",
            iterations=iteration + 1,
        )
        self._save_result(output)
        return output

    def unfold_doroshenko_matrix(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Doroshenko matrix form.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        tolerance : float, optional
            Convergence tolerance.
        max_iterations : int, optional
            Maximum iterations.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            x = np.ones(self.n_energy_bins)
        else:
            x = np.asarray(initial_spectrum, dtype=float)
            if len(x) != self.n_energy_bins:
                raise ValueError(
                    "initial_spectrum length must match number of energy bins"
                )

        ATA = A.T @ A
        ATb = A.T @ b

        for iteration in range(max_iterations):
            x_old = x.copy()
            for i in range(ATA.shape[0]):
                sum_without_i = np.sum(ATA[i, :] * x) - ATA[i, i] * x[i]
                numerator = ATb[i] - sum_without_i
                if ATA[i, i] > 0:
                    x[i] = max(0.0, numerator / ATA[i, i])
            if np.linalg.norm(x - x_old) < tolerance:
                break

        output = self._standardize_output(
            spectrum=x,
            A=A,
            b=b,
            selected=selected,
            method="Doroshenko-matrix",
            iterations=iteration + 1,
        )
        self._save_result(output)
        return output

    def unfold_cvxopt(
        self,
        readings: Dict[str, float],
        regularization: float = 1e-4,
        solver: str = "ECOS",
        abstol: float = 1e-10,
        reltol: float = 1e-10,
        feastol: float = 1e-10,
        max_iterations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using CVXOPT quadratic programming.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        regularization : float, optional
            Regularization strength.
        solver : str, optional
            CVXOPT solver name.
        abstol : float, optional
            Absolute tolerance.
        reltol : float, optional
            Relative tolerance.
        feastol : float, optional
            Feasibility tolerance.
        max_iterations : int, optional
            Maximum iterations.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        cvxopt = self._import_optional("cvxopt", "CVXOPT-based unfolding")
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        m, n = A.shape
        P = A.T @ A + regularization * np.eye(n)
        q = -A.T @ b

        P_cvx = cvxopt.matrix(P.astype(float))
        q_cvx = cvxopt.matrix(q.astype(float))
        G = cvxopt.matrix(-np.eye(n).astype(float))
        h = cvxopt.matrix(np.zeros(n).astype(float))

        cvxopt.solvers.options.clear()
        cvxopt.solvers.options["abstol"] = abstol
        cvxopt.solvers.options["reltol"] = reltol
        cvxopt.solvers.options["feastol"] = feastol
        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["maxit"] = max_iterations

        try:
            solution = cvxopt.solvers.qp(P_cvx, q_cvx, G, h, solver=solver.lower())
        except Exception:
            solution = cvxopt.solvers.qp(P_cvx, q_cvx, G, h)

        if solution["status"] != "optimal":
            raise ValueError(f"CVXOPT failed with status: {solution['status']}")

        spectrum = np.array(solution["x"]).flatten()
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="CVXOPT",
            regularization=regularization,
            solver=solver,
        )
        self._save_result(output)
        return output

    def unfold_cuqipy(
        self,
        readings: Dict[str, float],
        readings_error: float = 0.05,
        spectrum_error: float = 0.1,
        method: str = "Gaussian",
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using CUQIpy Bayesian inversion.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        readings_error : float, optional
            Measurement error scale.
        spectrum_error : float, optional
            Prior spectrum error scale.
        method : str, optional
            Prior type: 'Gaussian', 'LMRF', or 'CMRF'.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        cuqi_model = self._import_optional("cuqi.model", "CUQIpy unfolding")
        cuqi_distribution = self._import_optional("cuqi.distribution", "CUQIpy unfolding")
        cuqi_problem = self._import_optional("cuqi.problem", "CUQIpy unfolding")

        LinearModel = cuqi_model.LinearModel
        Gaussian = cuqi_distribution.Gaussian
        LMRF = cuqi_distribution.LMRF
        CMRF = cuqi_distribution.CMRF
        BayesianProblem = cuqi_problem.BayesianProblem

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        Amodel = LinearModel(A)
        if method == "Gaussian":
            x = Gaussian(np.zeros(A.shape[1]), spectrum_error)
        elif method == "LMRF":
            x = LMRF(0, spectrum_error, geometry=A.shape[1], bc_type="zero")
        elif method == "CMRF":
            x = CMRF(0, spectrum_error, geometry=A.shape[1], bc_type="zero")
        else:
            raise ValueError("method must be one of: Gaussian, LMRF, CMRF")

        y = Gaussian(Amodel @ x, readings_error)
        ip = BayesianProblem(y, x).set_data(y=b)
        samples = ip.UQ()

        spectrum = samples.samples.mean(axis=1)
        spectrum_err = samples.samples.std(axis=1)

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method=f"CUQIpy ({method})",
        )
        output["spectrum_error"] = spectrum_err
        self._save_result(output)
        return output

    def unfold_gurobi(
        self,
        readings: Dict[str, float],
        tolerance: float = 1e-6,
        method: str = "MinNormNonnegative",
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Gurobi optimization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        tolerance : float, optional
            Constraint tolerance.
        method : str, optional
            Model name/label for Gurobi.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        gp = self._import_optional("gurobipy", "Gurobi-based unfolding")
        GRB = gp.GRB

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        model = gp.Model(method)
        x = model.addVars(A.shape[1], lb=0.0, name="x")

        for i in range(A.shape[0]):
            expr = gp.LinExpr()
            for j in range(A.shape[1]):
                expr += A[i, j] * x[j]
            model.addConstr(expr >= b[i] - tolerance, f"constr_lower_{i}")
            model.addConstr(expr <= b[i] + tolerance, f"constr_upper_{i}")

        obj = gp.QuadExpr()
        for j in range(A.shape[1]):
            obj += x[j] * x[j]
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise ValueError(f"Gurobi failed with status: {model.status}")

        spectrum = np.array([x[i].X for i in range(A.shape[1])])
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method=f"Gurobi ({method})",
            tolerance=tolerance,
        )
        self._save_result(output)
        return output

    def unfold_bayes(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using iterative Bayes (D'Agostini).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior spectrum.
        max_iterations : int, optional
            Maximum iterations.
        tolerance : float, optional
            Stopping tolerance.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        pyunfold = self._import_optional("pyunfold", "Bayesian unfolding")
        pyunfold_callbacks = self._import_optional("pyunfold.callbacks", "Bayesian unfolding")

        iterative_unfold = pyunfold.iterative_unfold
        Logger = pyunfold_callbacks.Logger

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        efficiencies = [1] * A.shape[1]
        response_err = np.zeros_like(A)
        efficiencies_err = [0.05] * A.shape[1]
        data_err = [0.05] * A.shape[0]

        result = iterative_unfold(
            data=b,
            data_err=data_err,
            response=A,
            response_err=response_err,
            efficiencies=efficiencies,
            efficiencies_err=efficiencies_err,
            max_iter=max_iterations,
            callbacks=[Logger()],
            prior=initial_spectrum,
            ts_stopping=tolerance,
        )

        output = self._standardize_output(
            spectrum=result["unfolded"],
            A=A,
            b=b,
            selected=selected,
            method="Bayes (D'Agostini)",
            iterations=result["num_iterations"],
        )
        output["spectrum_statistical_uncertainties"] = result["stat_err"]
        output["spectrum_systematic_uncertainties"] = result["sys_err"]
        self._save_result(output)
        return output

    def unfold_bayes_spline_regularization(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
        spline_degree: int = 3,
        spline_smooth: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Bayes (D'Agostini) with spline regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior spectrum.
        max_iterations : int, optional
            Maximum iterations.
        tolerance : float, optional
            Stopping tolerance.
        spline_degree : int, optional
            Spline degree.
        spline_smooth : float, optional
            Spline smoothing parameter.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        pyunfold = self._import_optional("pyunfold", "Bayesian unfolding")
        pyunfold_callbacks = self._import_optional("pyunfold.callbacks", "Bayesian unfolding")

        iterative_unfold = pyunfold.iterative_unfold
        Logger = pyunfold_callbacks.Logger
        SplineRegularizer = pyunfold_callbacks.SplineRegularizer

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        spline_reg = SplineRegularizer(degree=spline_degree, smooth=spline_smooth)

        efficiencies = [1] * A.shape[1]
        response_err = np.zeros_like(A)
        efficiencies_err = [0.05] * A.shape[1]
        data_err = [0.05] * A.shape[0]

        result = iterative_unfold(
            data=b,
            data_err=data_err,
            response=A,
            response_err=response_err,
            efficiencies=efficiencies,
            efficiencies_err=efficiencies_err,
            max_iter=max_iterations,
            callbacks=[Logger(), spline_reg],
            prior=initial_spectrum,
            ts_stopping=tolerance,
        )

        output = self._standardize_output(
            spectrum=result["unfolded"],
            A=A,
            b=b,
            selected=selected,
            method="Bayes (D'Agostini) + spline",
            iterations=result["num_iterations"],
            spline_degree=spline_degree,
            spline_smooth=spline_smooth,
        )
        output["spectrum_statistical_uncertainties"] = result["stat_err"]
        output["spectrum_systematic_uncertainties"] = result["sys_err"]
        self._save_result(output)
        return output

    def unfold_statreg(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        unfoldermethod: str = "EmpiricalBayes",
        regularization: Optional[np.ndarray] = None,
        basis_name: str = "CubicSplines",
        boundary: Optional[str] = None,
        derivative_degree: int = 2,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using statistical regularization (Turchin's method).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior spectrum.
        unfoldermethod : str, optional
            Unfolder method ('EmpiricalBayes' or 'User').
        regularization : Optional[np.ndarray], optional
            Regularization parameter(s) for 'User' mode.
        basis_name : str, optional
            Basis type (only 'CubicSplines' supported).
        boundary : Optional[str], optional
            Boundary condition.
        derivative_degree : int, optional
            Regularization derivative degree.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        statreg_model = self._import_optional(
            "statreg.model", "statistical regularization unfolding"
        )
        statreg_basis = self._import_optional(
            "statreg.basis", "statistical regularization unfolding"
        )

        GaussErrorMatrixUnfolder = statreg_model.GaussErrorMatrixUnfolder
        CubicSplines = statreg_basis.CubicSplines

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        b_err = b * 0.05
        Emin = np.min(self.E_MeV)

        if basis_name != "CubicSplines":
            raise ValueError("Only CubicSplines basis is currently supported")

        basis = CubicSplines(np.log10(self.E_MeV / Emin), boundary=boundary)
        omega = basis.omega(derivative_degree)

        if unfoldermethod == "EmpiricalBayes":
            model = GaussErrorMatrixUnfolder(omega, method=unfoldermethod)
        elif unfoldermethod == "User":
            if regularization is None:
                regularization = 1e-4
            model = GaussErrorMatrixUnfolder(
                omega, method=unfoldermethod, alphas=regularization
            )
        else:
            raise ValueError("unfoldermethod must be 'EmpiricalBayes' or 'User'")

        result = model.solve(A, b, b_err)

        output = self._standardize_output(
            spectrum=result.phi,
            A=A,
            b=b,
            selected=selected,
            method="Statistical regularization (Turchin)",
            alphas=result["alphas"],
            basis_name=basis_name,
            boundary=boundary,
            derivative_degree=derivative_degree,
        )
        output["covariance"] = result["covariance"]
        output["uncertainties"] = np.sqrt(np.diag(result["covariance"]))
        self._save_result(output)
        return output

    def unfold_scipy_direct_method(
        self,
        readings: Dict[str, float],
        tolerance: float = 1e-8,
        max_iterations: int = 4000,
        method: str = "cg",
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using SciPy sparse solvers on normal equations.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        tolerance : float, optional
            Solver tolerance.
        max_iterations : int, optional
            Maximum iterations.
        method : str, optional
            Solver name.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        scipy_sparse = self._import_optional(
            "scipy.sparse.linalg", "SciPy sparse linear solvers"
        )

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        AT_A = A.T @ A
        AT_b = A.T @ b

        solvers = {
            "cg": lambda: scipy_sparse.cg(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "cgs": lambda: scipy_sparse.cgs(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "bicgstab": lambda: scipy_sparse.bicgstab(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "gmres": lambda: scipy_sparse.gmres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "lgmres": lambda: scipy_sparse.lgmres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "minres": lambda: scipy_sparse.minres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "qmr": lambda: scipy_sparse.qmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "gcrotmk": lambda: scipy_sparse.gcrotmk(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "tfqmr": lambda: scipy_sparse.tfqmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "lsqr": lambda: scipy_sparse.lsqr(A, b, atol=tolerance),
            "lsmr": lambda: scipy_sparse.lsmr(A, b, atol=tolerance, maxiter=max_iterations),
        }

        if method not in solvers:
            raise ValueError(f"Unknown method: {method}")

        x = solvers[method]()[0]
        x = np.maximum(x, 0)

        output = self._standardize_output(
            spectrum=x,
            A=A,
            b=b,
            selected=selected,
            method=f"SciPy solver ({method})",
            iterations=max_iterations,
            tolerance=tolerance,
        )
        self._save_result(output)
        return output

    def unfold_mcmc(
        self,
        readings: Dict[str, float],
        prior_type: str = "gamma",
        n_samples: int = 5000,
        tune: int = 5000,
        target_accept: float = 0.9,
        cpucores: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using MCMC (PyMC).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        prior_type : str, optional
            Prior distribution type.
        n_samples : int, optional
            Number of MCMC samples.
        tune : int, optional
            Tuning steps.
        target_accept : float, optional
            Target acceptance rate.
        cpucores : Optional[int], optional
            Number of CPU cores.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        pm = self._import_optional("pymc", "MCMC unfolding")

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        m = A.shape[1]

        with pm.Model() as model:
            if prior_type == "truncated_normal":
                x = pm.TruncatedNormal("x", mu=0, sigma=1, lower=0, shape=m)
            elif prior_type == "exponential":
                x = pm.Exponential("x", lam=1, shape=m)
            elif prior_type == "half_normal":
                x = pm.HalfNormal("x", sigma=1, shape=m)
            elif prior_type == "gamma":
                x = pm.Gamma("x", alpha=1, beta=1, shape=m)
            else:
                raise ValueError("prior_type must be one of: truncated_normal, exponential, half_normal, gamma")

            b_pred = pm.math.dot(A, x)
            sigma = pm.HalfNormal("sigma", sigma=1)
            pm.Normal("b_obs", mu=b_pred, sigma=sigma, observed=b)

            trace = pm.sample(
                n_samples, tune=tune, target_accept=target_accept, cores=cpucores
            )

        result = np.array(trace.posterior["x"].mean(axis=(0, 1)))
        output = self._standardize_output(
            spectrum=result,
            A=A,
            b=b,
            selected=selected,
            method="MCMC (PyMC)",
            prior_type=prior_type,
            n_samples=n_samples,
        )
        self._save_result(output)
        return output

    def unfold_ridge(
        self,
        readings: Dict[str, float],
        method: str = "ridgecv",
        regularization: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using scikit-learn linear models.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        method : str, optional
            Model name: ridge, ridgecv, lasso, bayesianridge.
        regularization : float, optional
            Regularization strength.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        sklearn_linear_model = self._import_optional(
            "sklearn.linear_model", "scikit-learn ridge/lasso unfolding"
        )

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if method == "ridge":
            reg = sklearn_linear_model.Ridge(alpha=regularization, positive=True)
        elif method == "ridgecv":
            reg = sklearn_linear_model.RidgeCV(alphas=np.logspace(-19, 2, 50))
        elif method == "lasso":
            reg = sklearn_linear_model.Lasso(alpha=regularization, positive=True)
        elif method == "bayesianridge":
            reg = sklearn_linear_model.BayesianRidge(
                alpha_init=regularization, lambda_init=regularization
            )
        else:
            raise ValueError(
                "method must be one of: ridge, ridgecv, lasso, bayesianridge"
            )

        reg.fit(A, b)
        if method == "ridgecv":
            regularization = reg.alpha_

        output = self._standardize_output(
            spectrum=reg.coef_,
            A=A,
            b=b,
            selected=selected,
            method=f"sklearn ({method})",
            regularization=regularization,
        )
        self._save_result(output)
        return output

    def unfold_pyomo(
        self,
        readings: Dict[str, float],
        solver_name: str = "gurobi",
        regularization: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using Pyomo optimization models.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        solver_name : str, optional
            Pyomo solver name.
        regularization : float, optional
            Regularization strength.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        pyo = self._import_optional("pyomo.environ", "Pyomo-based unfolding")

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        model = pyo.ConcreteModel()
        model.I = pyo.Set(initialize=range(A.shape[0]))
        model.J = pyo.Set(initialize=range(A.shape[1]))
        model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

        def objective_rule(model):
            residual_sum = 0.0
            for i in model.I:
                linear_comb = 0.0
                for j in model.J:
                    linear_comb += A[i, j] * model.x[j]
                residual_sum += (linear_comb - b[i]) ** 2
            reg_sum = 0.0
            for j in model.J:
                reg_sum += regularization * model.x[j] ** 2
            return residual_sum + reg_sum

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError(f"Pyomo solver failed: {results.solver.termination_condition}")

        spectrum = np.array([pyo.value(model.x[j]) for j in range(A.shape[1])])
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method=f"Pyomo ({solver_name})",
            regularization=regularization,
        )
        self._save_result(output)
        return output

    def unfold_lmfit(
        self,
        readings: Dict[str, float],
        method: str = "leastsq",
        model_name: str = "elastic",
        regularization: float = 1e-4,
        regularization2: float = 1e-4,
        l1_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using lmfit with L1/L2/Elastic regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        method : str, optional
            lmfit solver name.
        model_name : str, optional
            Regularization model: elastic, lasso, ridge.
        regularization : float, optional
            L1 regularization strength.
        regularization2 : float, optional
            L2 regularization strength.
        l1_weight : float, optional
            L1 weight for elastic net.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        lmfit = self._import_optional("lmfit", "lmfit-based unfolding")

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        m = A.shape[1]

        params = lmfit.Parameters()
        init_values = np.ones(m) * np.mean(b) / np.mean(A.sum(axis=1))
        for i in range(m):
            params.add(f"x{i}", value=max(init_values[i], 1e-10), min=0.0)

        def residual_lasso(params, A, b, regularization):
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b
            if method == "leastsq":
                reg_residual = np.sqrt(regularization) * np.sqrt(m) * x
                return np.concatenate([residual, reg_residual])
            return np.sum(residual**2) + regularization * np.sum(np.abs(x))

        def residual_ridge(params, A, b, regularization):
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b
            if method == "leastsq":
                reg_residual = np.sqrt(regularization) * x
                return np.concatenate([residual, reg_residual])
            return np.sum(residual**2) + regularization * np.sum(x**2)

        def residual_elastic(params, A, b, regularization, regularization2, l1_weight):
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b
            if method == "leastsq":
                l1_residual = (
                    np.sqrt(regularization * l1_weight) * np.sqrt(m) * np.abs(x)
                )
                l2_residual = np.sqrt(regularization2 * (1 - l1_weight)) * x
                reg_residual = np.concatenate([l1_residual, l2_residual])
                return np.concatenate([residual, reg_residual])
            l1_penalty = regularization * l1_weight * np.sum(np.abs(x))
            l2_penalty = regularization2 * (1 - l1_weight) * np.sum(x**2)
            return np.sum(residual**2) + l1_penalty + l2_penalty

        gradient_methods = {"newton", "tnc", "cg", "bfgs", "lbfgsb"}

        if model_name == "lasso":
            result = lmfit.minimize(
                residual_lasso,
                params,
                args=(A, b, regularization),
                method=method,
            )
        elif model_name == "ridge":
            result = lmfit.minimize(
                residual_ridge,
                params,
                args=(A, b, regularization),
                method=method,
            )
        elif model_name == "elastic":
            result = lmfit.minimize(
                residual_elastic,
                params,
                args=(A, b, regularization, regularization2, l1_weight),
                method=method,
            )
        else:
            raise ValueError("model_name must be one of: elastic, lasso, ridge")

        spectrum = np.array([result.params[f"x{i}"].value for i in range(m)])
        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method=f"lmfit ({method})",
            model_name=model_name,
        )
        output.update(
            {
                "regularization": regularization,
                "regularization2": regularization2 if model_name == "elastic" else None,
                "l1_weight": l1_weight if model_name == "elastic" else None,
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
            }
        )
        self._save_result(output)
        return output

    def unfold_gauss_newton(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using ODL Gauss-Newton iterations.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        max_iterations : int, optional
            Maximum iterations.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        measurement_space = odl.uniform_discr(0, len(b) - 1, len(b))
        spectrum_space = odl.uniform_discr(
            np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
        )
        operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)
        y = measurement_space.element(b)
        x = spectrum_space.element(0.5 if initial_spectrum is None else initial_spectrum)

        if hasattr(odl.solvers, "gauss_newton"):
            odl.solvers.gauss_newton(operator, x, y, niter=max_iterations, callback=None)
        elif hasattr(odl.solvers, "iterative") and hasattr(
            odl.solvers.iterative, "gauss_newton"
        ):
            odl.solvers.iterative.gauss_newton(
                operator, x, y, niter=max_iterations, callback=None
            )
        else:
            raise ImportError("ODL gauss_newton solver is not available in this version.")

        spectrum = x.asarray()
        spectrum[spectrum < 0] = 0

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="Gauss-Newton (ODL)",
            iterations=max_iterations,
        )
        self._save_result(output)
        return output

    def unfold_kaczmarz(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        omega: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Kaczmarz algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        max_iterations : int, optional
            Maximum iterations.
        omega : float, optional
            Relaxation parameter.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        m, n = A.shape

        x = np.zeros(n) if initial_spectrum is None else np.asarray(initial_spectrum)

        for k in range(max_iterations):
            i = k % m
            a_i = A[i, :]
            denominator = np.dot(a_i, a_i)
            if denominator > 0:
                update = (b[i] - np.dot(a_i, x)) / denominator
                x = x + omega * update * a_i

        output = self._standardize_output(
            spectrum=x,
            A=A,
            b=b,
            selected=selected,
            method="Kaczmarz",
            iterations=max_iterations,
            omega=omega,
        )
        self._save_result(output)
        return output

    def unfold_kaczmarz2(
        self,
        readings: Dict[str, float],
        max_iterations: int = 4000,
        tolerance: float = 1e-8,
        rule: str = "maxdistance",
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the kaczmarz-algorithms package.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        max_iterations : int, optional
            Maximum iterations.
        tolerance : float, optional
            Convergence tolerance.
        rule : str, optional
            Selection rule: 'maxdistance' or 'cyclic'.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        kaczmarz = self._import_optional("kaczmarz", "kaczmarz-algorithms unfolding")
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if rule == "maxdistance":
            x = kaczmarz.MaxDistance.solve(A, b, maxiter=max_iterations, tol=tolerance)
        elif rule == "cyclic":
            x = kaczmarz.Cyclic.solve(A, b, maxiter=max_iterations, tol=tolerance)
        else:
            raise ValueError("rule must be 'maxdistance' or 'cyclic'")

        output = self._standardize_output(
            spectrum=x,
            A=A,
            b=b,
            selected=selected,
            method=f"Kaczmarz ({rule})",
            iterations=max_iterations,
        )
        self._save_result(output)
        return output

    def unfold_evolutionary(
        self,
        readings: Dict[str, float],
        regularization: float = 1.0,
        penalty_weight: int = 1000,
        population_size: int = 300,
        generations: int = 300,
        mu: float = 0.0,
        mate_param: float = 0.3,
        tournsize: int = 3,
        mutation_strength: float = 0.3,
        individ_probability: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using a DEAP-based evolutionary algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        regularization : float, optional
            L2 regularization strength.
        penalty_weight : int, optional
            Penalty weight for negative values.
        population_size : int, optional
            Population size.
        generations : int, optional
            Number of generations.
        mu : float, optional
            Mutation mean.
        mate_param : float, optional
            Crossover blending parameter.
        tournsize : int, optional
            Tournament size.
        mutation_strength : float, optional
            Mutation standard deviation.
        individ_probability : float, optional
            Mutation probability per gene.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        deap_base = self._import_optional("deap.base", "DEAP evolutionary unfolding")
        deap_creator = self._import_optional("deap.creator", "DEAP evolutionary unfolding")
        deap_tools = self._import_optional("deap.tools", "DEAP evolutionary unfolding")
        deap_algorithms = self._import_optional("deap.algorithms", "DEAP evolutionary unfolding")
        import random

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        def evaluate_solution_penalty(individual):
            x = np.array(individual)
            negative_penalty = penalty_weight * np.sum(np.minimum(0, x) ** 2)
            residual = A @ x - b
            error = np.sum(residual**2) + negative_penalty + regularization * np.sum(x**2)
            return (error,)

        def nonnegative_mutation(individual, mu, sigma, indpb):
            for i in range(len(individual)):
                if random.random() < indpb:
                    new_value = individual[i] + random.gauss(mu, sigma)
                    individual[i] = max(0, new_value)
            return (individual,)

        if not hasattr(deap_creator, "FitnessMin"):
            deap_creator.create("FitnessMin", deap_base.Fitness, weights=(-1.0,))
        if not hasattr(deap_creator, "Individual"):
            deap_creator.create("Individual", list, fitness=deap_creator.FitnessMin)

        toolbox = deap_base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.0, 5.0)
        toolbox.register(
            "individual",
            deap_tools.initRepeat,
            deap_creator.Individual,
            toolbox.attr_float,
            n=A.shape[1],
        )
        toolbox.register("population", deap_tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_solution_penalty)
        toolbox.register("mate", deap_tools.cxBlend, alpha=mate_param)
        toolbox.register(
            "mutate",
            nonnegative_mutation,
            mu=mu,
            sigma=mutation_strength,
            indpb=individ_probability,
        )
        toolbox.register("select", deap_tools.selTournament, tournsize=tournsize)

        population = toolbox.population(n=population_size)
        population, _ = deap_algorithms.eaSimple(
            population,
            toolbox,
            cxpb=mate_param,
            mutpb=mutation_strength,
            ngen=generations,
            stats=None,
            verbose=False,
        )

        best_individual = deap_tools.selBest(population, k=1)[0]
        spectrum = np.maximum(0, np.array(best_individual))

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="Evolutionary (DEAP)",
            population_size=population_size,
            generations=generations,
        )
        self._save_result(output)
        return output

    def unfold_qubo(
        self,
        readings: Dict[str, float],
        regularization: float = 1e-4,
        num_reads: int = 200,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using QUnfold QUBO formulation.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        regularization : float, optional
            Regularization strength.
        num_reads : int, optional
            Number of annealing reads.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result dictionary.
        """
        qunfold = self._import_optional("qunfold", "QUnfold QUBO unfolding")
        QUnfolder = qunfold.QUnfolder

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        binning = np.append(np.array(self.E_MeV), np.array(np.max(self.E_MeV)))
        unfolder = QUnfolder(A.T @ A, A.T @ b, binning, lam=regularization)
        unfolder.initialize_qubo_model()
        spectrum, cov = unfolder.solve_simulated_annealing(num_reads=num_reads)

        output = self._standardize_output(
            spectrum=spectrum,
            A=A,
            b=b,
            selected=selected,
            method="QUBO (QUnfold)",
            regularization=regularization,
            num_reads=num_reads,
        )
        output["spectrum_error"] = np.sqrt(np.diag(cov))
        output["covariance"] = cov
        self._save_result(output)
        return output

    def _calculate_doserates(
        self, spectrum: np.ndarray, dlnE: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate dose rates using ICRP-116 conversion coefficients.

        Uses uniform logarithmic step of 0.2 for integration.

        Parameters
        ----------
        spectrum : np.ndarray
            Unfolded neutron spectrum
        dlnE : float, optional
            Logarithmic energy step for integration, default: 0.2

        Returns
        -------
        Dict[str, float]
            Dictionary of dose rates for different geometries:
            - 'AP': Anterior-Posterior
            - 'PA': Posterior-Anterior
            - 'LLAT': Left Lateral
            - 'RLAT': Right Lateral
            - 'ISO': Isotropic
            - 'ROT': Rotational
            Values are in pico-Sievert per second (pSv/s)
        """
        if not self.cc_icrp116:
            return {
                geom: 0.0 for geom in ["AP", "PA", "LLAT", "RLAT", "ISO", "ROT"]
            }

        doserates = {}
        for geom, k in self.cc_icrp116.items():
            if geom != "E_MeV":
                integrand = k * spectrum * dlnE
                dose = np.log(10) * np.sum(integrand)
                doserates[geom] = float(dose)  # pSv/s
        return doserates

    def _load_icrp116_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Load ICRP-116 conversion coefficients.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of conversion coefficients for different geometries
        """
        try:
            from .constants import ICRP116_COEFF_EFFECTIVE_DOSE

            return ICRP116_COEFF_EFFECTIVE_DOSE
        except ImportError:
            warnings.warn(
                "ICRP-116 coefficients not found. Dose calculations will return zeros."
            )
            return {}

    def _add_noise(
        self, readings: Dict[str, float], noise_level: float = 0.01
    ) -> Dict[str, float]:
        """
        Add Gaussian noise to readings dictionary.

        Parameters
        ----------
        readings : Dict[str, float]
            Original readings dictionary
        noise_level : float, optional
            Noise level as fraction (e.g., 0.01 = 1% noise), default: 0.01

        Returns
        -------
        Dict[str, float]
            Noisy readings dictionary
        """
        readings_noisy = {}
        for key, value in readings.items():
            # Generate Gaussian noise
            noise = np.random.normal(loc=0, scale=noise_level)
            # Apply noise
            readings_noisy[key] = value * (1 + noise)
        return readings_noisy

    # --- UTILS ---
    def plot_response_functions(self):
        """ plot all response functions"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for key, rf in self.sensitivities.items():
            ax.plot(
                self.E_MeV,
                rf,
                label=key,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Energy, MeV")
        ax.set_ylabel("Response, cm²")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Response functions of the detector")
        plt.show()
        plt.close()

    def discretize_spectra(self, spectra):
        '''
        Interpolate spectra onto the target energy grid using PCHIP interpolation.
        Extrapolation is avoided and replaced with zeros.
        
        Parameters:
        -----------
        spectra : pandas.DataFrame or dict
            Input spectra to be discretized. If DataFrame, it should contain energy 
            values. If dict, it should have 'E_MeV' and 'Phi' keys.
            
        Returns:
        --------
        pandas.DataFrame
            Discretized spectra with columns 'E_MeV' and 'Phi'
        '''
        # Get target energy grid parameters
        Emin = np.min(self.E_MeV)
        new_ebins = self.E_MeV
        
        # Initialize variables for input spectra
        energies = None
        spectre_data = None
        
        # Handle dictionary input
        if isinstance(spectra, dict):
            # Convert dictionary to DataFrame for uniform processing
            spectre_df = pd.DataFrame(spectra)
        # Handle DataFrame input
        elif isinstance(spectra, pd.DataFrame):
            spectre_df = spectra.copy()
        else:
            raise TypeError("Input spectra must be either a pandas DataFrame or a dictionary")
        
        # Extract energy values from the input
        if "E_MeV" in spectre_df.columns:
            energies = spectre_df["E_MeV"].values
            # Assuming the rest of the columns are spectral data
            spectre_data = spectre_df.drop("E_MeV", axis=1).values
        else:
            # Assume the first column contains energy values
            energies = spectre_df.iloc[:, 0].values
            spectre_data = spectre_df.iloc[:, 1:].values
        
        # Convert string energy values to float if needed
        if len(energies) > 0 and isinstance(energies[0], str):
            energies = energies.astype(float)
        
        # Check if target grid exceeds input grid bounds
        if new_ebins.min() < np.min(energies):
            print("Warning: Target energy bins extend below the input grid minimum. Setting values to zero.")
        
        if new_ebins.max() > np.max(energies):
            print("Warning: Target energy bins extend above the input grid maximum. Setting values to zero.")
        
        # Convert to logarithmic scale relative to minimum energy
        u = np.log10(energies / Emin)
        u_new = np.log10(new_ebins / Emin)  # New grid in log scale
        
        # Initialize array for interpolated values
        interpolated_values = np.zeros((len(new_ebins), spectre_data.shape[1]))
        
        # Interpolate each spectral component separately
        for i in range(spectre_data.shape[1]):
            # Create PCHIP interpolator for this spectral component
            interpolator = pchip(u, spectre_data[:, i])
            
            # Interpolate onto new grid
            interp_vals = interpolator(u_new)
            
            # Replace extrapolated values with zeros
            interp_vals[(u_new < u.min()) | (u_new > u.max())] = 0
            
            # Replace negative values with zeros
            interp_vals[interp_vals < 0] = 0
            
            interpolated_values[:, i] = interp_vals
        
        # Create result DataFrame
        new_spectra = pd.DataFrame()
        new_spectra["E_MeV"] = new_ebins
        
        # Add interpolated spectral components
        if spectre_data.shape[1] == 1:
            # Single spectrum case
            new_spectra["Phi"] = interpolated_values[:, 0]
        else:
            # Multiple spectra case
            for i in range(spectre_data.shape[1]):
                if isinstance(spectra, dict) and i == 0:
                    # For dictionary input, use the original column names
                    if 'Phi' in spectra:
                        new_spectra[f"Phi_{i}"] = interpolated_values[:, i]
                    else:
                        # Try to use original column names
                        col_name = spectre_df.columns[1] if len(spectre_df.columns) > 1 else f"Phi_{i}"
                        new_spectra[col_name] = interpolated_values[:, i]
                else:
                    # For multiple spectra from DataFrame, preserve original names
                    if i < len(spectre_df.columns) - 1:
                        col_name = spectre_df.columns[i + 1]
                    else:
                        col_name = f"Phi_{i}"
                    new_spectra[col_name] = interpolated_values[:, i]
        
        return new_spectra
    
    def get_effective_readings_for_spectra(self, spectra):
        """
        Calculate effective readings for a given spectrum.
        
        This function interpolates the input spectrum onto the detector's energy grid,
        then calculates what readings each detector sphere would measure for that spectrum
        by integrating the product of the spectrum and response functions.
        
        Parameters
        ----------
        spectra : pandas.DataFrame or dict
            Input spectra to calculate effective readings for.
            - If DataFrame: Should contain 'E_MeV' column for energies and one or more 
            columns for spectral data (e.g., 'Phi').
            - If dict: Should have 'E_MeV' and spectral data keys.
        
        Returns
        -------
        dict
            Dictionary of effective readings for each detector sphere in the format:
            {sphere_name: reading_value, ...}
            
        Raises
        ------
        TypeError
            If input is not a pandas DataFrame or dictionary
        ValueError
            If energy grid doesn't match or spectrum has invalid shape
        
        Examples
        --------
        >>> # Using DataFrame input
        >>> spectrum_df = pd.DataFrame({
        ...     'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        ...     'Phi': [1.0, 0.8, 0.5, 0.3, 0.1]
        ... })
        >>> readings = detector.get_effective_readings_for_spectra(spectrum_df)
        
        >>> # Using dictionary input
        >>> spectrum_dict = {
        ...     'E_MeV': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        ...     'Phi': [1.0, 0.8, 0.5, 0.3, 0.1]
        ... }
        >>> readings = detector.get_effective_readings_for_spectra(spectrum_dict)
        """       
        # Handle dictionary input
        if isinstance(spectra, dict):
            # Convert dictionary to DataFrame for uniform processing
            spectra_df = pd.DataFrame(spectra)
        # Handle DataFrame input
        elif isinstance(spectra, pd.DataFrame):
            spectra_df = spectra.copy()
        else:
            raise TypeError(
                "Input spectra must be either a pandas DataFrame or a dictionary. "
                f"Got type: {type(spectra)}"
            )
        
        # Check if input has the same energy grid as detector
        if "E_MeV" in spectra_df.columns:
            input_energies = spectra_df["E_MeV"].values
        else:
            # Assume first column is energy
            input_energies = spectra_df.iloc[:, 0].values
        
        # Convert string energy values to float if needed
        if len(input_energies) > 0 and isinstance(input_energies[0], str):
            input_energies = input_energies.astype(float)
        
        # Check if we need to interpolate
        need_interpolation = not np.array_equal(
            np.round(input_energies, 12), 
            np.round(self.E_MeV, 12)
        )
        
        if need_interpolation:
            # Interpolate spectrum onto detector's energy grid
            interp_spectra_df = self.discretize_spectra(spectra)
            
            # Extract spectrum values from the interpolated DataFrame
            if "Phi" in interp_spectra_df.columns:
                spectrum_values = interp_spectra_df["Phi"].values
            elif interp_spectra_df.shape[1] > 1:
                # Use the first non-energy column
                spectrum_values = interp_spectra_df.iloc[:, 1].values
            else:
                raise ValueError(
                    "Interpolated spectra doesn't contain spectrum data. "
                    f"Columns: {list(interp_spectra_df.columns)}"
                )
        else:
            # Extract spectrum values directly
            if "Phi" in spectra_df.columns:
                spectrum_values = spectra_df["Phi"].values
            else:
                # Assume the second column is spectrum data
                if spectra_df.shape[1] < 2:
                    raise ValueError(
                        f"Spectrum DataFrame must have at least 2 columns. "
                        f"Got {spectra_df.shape[1]} columns."
                    )
                spectrum_values = spectra_df.iloc[:, 1].values
        
        # Ensure spectrum values have the right shape
        if len(spectrum_values) != len(self.E_MeV):
            raise ValueError(
                f"Spectrum length ({len(spectrum_values)}) must match "
                f"energy grid length ({len(self.E_MeV)})"
            )
        
        # Calculate effective readings by integrating over energy
        # For each detector: reading = ∫ Φ(E) * R(E) dE
        # where R(E) is the response function (already includes log steps in Amat)
        
        effective_readings = {}
        
        for i, detector_name in enumerate(self.detector_names):
            # Get response function for this detector
            response_func = self.Amat[:, i]  # Already includes dlnE factor
            
            # Calculate reading: dot product of spectrum and response function
            # This is equivalent to ∫ Φ(E) * R(E) dE over the energy grid
            reading = np.sum(spectrum_values * response_func)
            
            # Ensure reading is non-negative (physical constraint)
            reading = max(0.0, reading)
            
            effective_readings[detector_name] = float(reading)
        
        return effective_readings
