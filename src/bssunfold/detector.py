"""Detector class with unfolding methods."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip

from typing import Dict, Optional, List, Tuple, Any, Union
import cvxpy as cp
import odl
import warnings
from datetime import datetime


class Detector:
    """
    Class for neutron detector operations and spectrum unfolding.

    This class provides methods for neutron spectrum unfolding using various
    algorithms and includes tools for dose rate calculations based on ICRP-116
    conversion coefficients.

    Parameters
    ----------
    response_functions : pd.DataFrame, dict, optional
        Response functions data. Can be:
        - pandas DataFrame with 'E_MeV' column and detector columns.
        - dict with 'E_MeV' key (array) and detector names as keys (arrays).
        If None, default GSF response functions are used.
    E_MeV : np.ndarray, optional
        Energy grid in MeV. Required if `response_functions` is not provided
        and `sensitivities` is provided.
    sensitivities : dict or np.ndarray, optional
        Detector sensitivities. If dict, keys are detector names and
        values
        are arrays of same length as E_MeV. If 2D array, shape (n_energy, n_detectors).
        Required if `response_functions` is not provided and `E_MeV` is provided.

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
    >>> from bssunfold import Detector
    >>> # Create detector with default GSF response functions
    >>> detector = Detector()
    >>> # Or from a DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'E_MeV': [1e-9, 1e-8, 1e-7],
    ...     'sphere_1': [0.1, 0.2, 0.3],
    ...     'sphere_2': [0.4, 0.5, 0.6]
    ... })
    >>> detector = Detector(df)
    >>> # Or from a dictionary
    >>> rf_dict = {
    ...     'E_MeV': [1e-9, 1e-8, 1e-7],
    ...     'sphere_1': [0.1, 0.2, 0.3],
    ...     'sphere_2': [0.4, 0.5, 0.6]
    ... }
    >>> detector = Detector(rf_dict)
    >>> # Or using numpy arrays
    >>> import numpy as np
    >>> E = np.array([1e-9, 1e-8, 1e-7])
    >>> sens = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    >>> detector = Detector(E_MeV=E, sensitivities=sens)
    >>> # Perform unfolding
    >>> readings = {'sphere_1': 100.5, 'sphere_2': 85.3}
    >>> result = detector.unfold_cvxpy(readings)
    """

    def __init__(self, response_functions=None, E_MeV=None, sensitivities=None):
        """
        Initialize Detector with response functions.

        Parameters
        ----------
        response_functions : pd.DataFrame, dict, optional
            Response functions data. Can be:
            - pandas DataFrame with 'E_MeV' column and detector columns.
            - dict with 'E_MeV' key (array) and detector names as keys (arrays).
            If None, default GSF response functions are used.
        E_MeV : np.ndarray, optional
            Energy grid in MeV. Required if `response_functions` is not provided
            and `sensitivities` is provided.
        sensitivities : dict or np.ndarray, optional
            Detector sensitivities. If dict, keys are detector names and
        values
            are arrays of same length as E_MeV. If 2D array, shape (n_energy, n_detectors).
            Required if `response_functions` is not provided and `E_MeV` is provided.

        Raises
        ------
        ValueError
            If E_MeV is not a 1D array or has less than 2 energy points,
            or if input data is inconsistent.
        """
        rf_df = self._process_input(response_functions, E_MeV, sensitivities)
        Amat, E_MeV, detector_names, log_steps = (
            self._convert_rf_to_matrix_variable_step(rf_df, Emin=1e-9)
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
        self.results_history: Dict[str, Dict[str, Any]] = {}
        self.current_result: Optional[Dict[str, Any]] = None

    def _process_input(self, response_functions, E_MeV, sensitivities):
        """
        Convert various input formats to a unified DataFrame.

        Parameters
        ----------
        response_functions : pd.DataFrame, dict, or None
        E_MeV : np.ndarray or None
        sensitivities : dict, np.ndarray, or None

        Returns
        -------
        pd.DataFrame
            DataFrame with 'E_MeV' column and detector columns.
        """
        import pandas as pd
        import numpy as np
        from .constants import RF_GSF

        # Case 1: response_functions is a DataFrame
        if isinstance(response_functions, pd.DataFrame):
            return response_functions.copy()

        # Case 2: response_functions is a dict
        if isinstance(response_functions, dict):
            # Ensure 'E_MeV' key exists
            if 'E_MeV' not in response_functions:
                raise ValueError("Dictionary must contain 'E_MeV' key")
            return pd.DataFrame(response_functions)

        # Case 3: E_MeV and sensitivities provided
        if E_MeV is not None and sensitivities is not None:
            if isinstance(sensitivities, dict):
                # Convert dict to DataFrame
                data = {'E_MeV': E_MeV}
                for det_name, sens_arr in sensitivities.items():
                    if len(sens_arr) != len(E_MeV):
                        raise ValueError(
                            f"Sensitivity array length for '{det_name}' "
                            f"must match E_MeV length"
                        )
                    data[det_name] = sens_arr
                return pd.DataFrame(data)
            elif isinstance(sensitivities, np.ndarray):
                if sensitivities.ndim != 2:
                    raise ValueError("sensitivities must be 2D array (n_energy, n_detectors)")
                if sensitivities.shape[0] != len(E_MeV):
                    raise ValueError("Number of rows in sensitivities must match length of E_MeV")
                # Create detector names
                detector_names = [f"det_{i}" for i in range(sensitivities.shape[1])]
                data = {'E_MeV': E_MeV}
                for i, name in enumerate(detector_names):
                    data[name] = sensitivities[:, i]
                return pd.DataFrame(data)
            else:
                raise TypeError("sensitivities must be dict or np.ndarray")

        # Case 4: No arguments, use default
        if response_functions is None and E_MeV is None and sensitivities is None:
            # Use GSF response functions as default
            return pd.DataFrame(RF_GSF)

        # If none of the above, raise error
        raise ValueError(
            "Invalid input combination. Provide either response_functions "
            "(DataFrame/dict) or both E_MeV and sensitivities."
        )

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

        print(f"Result saved with key: {key}")
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
        print("All results cleared.")

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
        save_result: bool = True,
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
        noise_level : float, optional
            Noise level for Monte-Carlo error estimation, default: 0.01
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100
        save_result : bool, optional
            If True, save result to internal history, default: True

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
            A: np.ndarray, b: np.ndarray, use_solver: str = None
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

            print(f"Status: {problem.status}")
            print(f"Objective value: {problem.value}")
            return x.value

        # Validate and solve
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        alpha = regularization
        
        # Normalize initial spectrum if provided
        if initial_spectrum is not None:
            initial_spectrum = self._normalize_initial_spectrum(initial_spectrum)
            # Currently not used in cvxpy solver, but kept for compatibility
            # Could be used as initial guess for x in future improvements
        
        n = A.shape[1]

        # Main solution
        x_value = _solve_problem(A, b, solver)
        computed_readings = A @ x_value
        residual = b - computed_readings
        residual_norm = np.linalg.norm(residual)
        print(f"Residual norm: {residual_norm:.6f}")

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
            print("Calculating uncertainty with Monte-Carlo...")
            x_montecarlo = np.empty((n_montecarlo, n))

            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_montecarlo[i] = _solve_problem(A_noisy, b_noisy, solver)

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                    "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                    "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                    "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
                    "spectrum_uncert_all": x_montecarlo,
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )
            print("...uncertainty calculated.")

        if save_result:
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
        save_result: bool = True,
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
        save_result : bool, optional
            If True, save result to internal history, default: True

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

        # Set initial spectrum (normalize if needed)
        if initial_spectrum is None:
            x0 = np.zeros(self.n_energy_bins)
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                x0 = np.zeros(self.n_energy_bins)

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
            print(
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo samples..."
            )

            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))

            for i in range(n_montecarlo):
                # Add noise to readings
                noisy_readings = self._add_noise(
                    validated_readings, noise_level
                )

                # Rebuild system with noisy readings
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)

                # Run Landweber with same parameters
                x_sample, _, _ = _landweber_iteration(
                    A_noisy, b_noisy, x0, max_iterations, tolerance
                )
                spectra_samples[i] = x_sample

            # Calculate uncertainty statistics
            output.update(
                {
                    "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
                    "spectrum_uncert_std": np.std(spectra_samples, axis=0),
                    "spectrum_uncert_min": np.min(spectra_samples, axis=0),
                    "spectrum_uncert_max": np.max(spectra_samples, axis=0),
                    "spectrum_uncert_median": np.median(
                        spectra_samples, axis=0
                    ),
                    "spectrum_uncert_percentile_5": np.percentile(
                        spectra_samples, 5, axis=0
                    ),
                    "spectrum_uncert_percentile_95": np.percentile(
                        spectra_samples, 95, axis=0
                    ),
                    "spectrum_uncert_all": spectra_samples,
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )
            print("...uncertainty calculation completed.")
        if save_result:
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
        save_result: bool = True,
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
        save_result : bool, optional
            If True, save result to internal history. Default is True.

        Returns
        -------
        Dict
            Dictionary containing the spectrum restoration results.
        """
        # Вспомогательная функция для создания ODL пространств
        def _create_odl_spaces(b_size: int):
            """Создание пространств ODL для заданного размера вектора измерений."""
            measurement_space = odl.uniform_discr(0, b_size - 1, b_size)
            spectrum_space = odl.uniform_discr(
                np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
            )
            return measurement_space, spectrum_space

        # Вспомогательная функция для инициализации спектра
        def _initialize_spectrum(spectrum_space, initial_spectrum=None):
            """Инициализация спектра в заданном пространстве."""
            if initial_spectrum is None:
                # 0.5 - хорошо, иначе при 1 или 0 получаем нули.
                return spectrum_space.element(0.5)
            return spectrum_space.element(initial_spectrum)

        # Основная функция выполнения MLEM
        def _run_mlem(A_matrix, b_vector, initial_spectrum_vals=None):
            """Запуск алгоритма MLEM для заданной системы уравнений."""
            # Создаем пространства ODL
            measurement_space, spectrum_space = _create_odl_spaces(len(b_vector))
            
            # Создаем оператор (матрицу чувствительности)
            operator = odl.MatrixOperator(
                A_matrix, 
                domain=spectrum_space, 
                range=measurement_space
            )
            
            y = measurement_space.element(b_vector)
            x = _initialize_spectrum(spectrum_space, initial_spectrum_vals)
            
            # Запускаем алгоритм MLEM
            odl.solvers.mlem(
                operator, 
                x, 
                y, 
                niter=max_iterations, 
                callback=callback
            )
            
            return x.asarray(), A_matrix, b_vector

        # Инициализация
        callback = odl.solvers.CallbackPrintIteration()
        
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        
        # Нормализация начального спектра
        normalized_initial = None
        if initial_spectrum is not None:
            normalized_initial = self._normalize_initial_spectrum(
                initial_spectrum
            )
        
        # Основной запуск MLEM
        unfolded_spectrum, A, b = _run_mlem(A, b, normalized_initial)
        
        # Создание основного результата
        output = self._standardize_output(
            spectrum=unfolded_spectrum,
            A=A,
            b=b,
            selected=selected,
            method="MLEM (ODL)",
            iterations=max_iterations,
        )
        
        # Monte-Carlo оценка неопределенности
        if calculate_errors:
            print(f"Calculating uncertainty with {n_montecarlo} Monte-Carlo samples...")
            
            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))
            
            for i in range(n_montecarlo):
                # Добавляем шум к показаниям
                noisy_readings = self._add_noise(readings, noise_level)
                
                # Перестраиваем систему с зашумленными данными
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)
                
                # Запускаем MLEM с теми же параметрами
                spectrum_sample, _, _ = _run_mlem(
                    A_noisy, b_noisy, normalized_initial
                )
                spectra_samples[i] = spectrum_sample
            
            # Вычисляем статистики неопределенности
            uncertainty_stats = {
                "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
                "spectrum_uncert_std": np.std(spectra_samples, axis=0),
                "spectrum_uncert_min": np.min(spectra_samples, axis=0),
                "spectrum_uncert_max": np.max(spectra_samples, axis=0),
                "spectrum_uncert_median": np.median(spectra_samples, axis=0),
                "spectrum_uncert_percentile_5": np.percentile(spectra_samples, 5, axis=0),
                "spectrum_uncert_percentile_95": np.percentile(spectra_samples, 95, axis=0),
                "spectrum_uncert_all": spectra_samples,
                "montecarlo_samples": n_montecarlo,
                "noise_level": noise_level,
            }
            
            output.update(uncertainty_stats)
            print("...uncertainty calculation completed.")
        
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

    def _normalize_initial_spectrum(
        self,
        initial_spectrum: Optional[Union[np.ndarray, Dict, pd.DataFrame]],
    ) -> Optional[np.ndarray]:
        """
        Normalize initial spectrum to detector's energy grid.

        If initial_spectrum is None, returns None.
        If it's a numpy array, assumes it's already on self.E_MeV grid.
        If lengths differ, raises ValueError (cannot interpolate without energies).
        If it's a dict or DataFrame, uses discretize_spectra to interpolate onto self.E_MeV,
        ensuring non-negative values.

        Parameters
        ----------
        initial_spectrum : Optional[Union[np.ndarray, Dict, pd.DataFrame]]
            Initial spectrum guess. Can be:
            - None: returns None.
            - np.ndarray: 1D array of length n_energy_bins (must match).
            - dict: with 'E_MeV' and 'Phi' keys (or similar).
            - pd.DataFrame: with 'E_MeV' column and at least one spectrum column.

        Returns
        -------
        Optional[np.ndarray]
            Normalized spectrum as 1D numpy array of length self.n_energy_bins,
            with negative values replaced by 0. Returns None if input is None.

        Raises
        ------
        ValueError
            If initial_spectrum is np.ndarray with wrong length, or if dict/DataFrame
            cannot be interpolated.
        """
        if initial_spectrum is None:
            return None

        # Case 1: numpy array
        if isinstance(initial_spectrum, np.ndarray):
            if len(initial_spectrum) != self.n_energy_bins:
                raise ValueError(
                    f"Initial spectrum length ({len(initial_spectrum)}) "
                    f"must match number of energy bins ({self.n_energy_bins}). "
                    "If you have a spectrum on a different energy grid, "
                    "provide it as a dict or DataFrame with 'E_MeV' column."
                )
            # Ensure non-negative
            return np.maximum(initial_spectrum, 0)

        # Case 2: dict or DataFrame
        # Use existing discretize_spectra method which does PCHIP interpolation
        # and replaces negatives with zero.
        if isinstance(initial_spectrum, (dict, pd.DataFrame)):
            # discretize_spectra returns DataFrame with columns 'E_MeV' and at least one spectrum column
            discretized = self.discretize_spectra(initial_spectrum)
            # Extract the first spectrum column (assuming single spectrum)
            # If multiple columns, take the first non-energy column
            if 'Phi' in discretized.columns:
                spectrum_col = 'Phi'
            else:
                # Find first column that is not 'E_MeV'
                non_energy_cols = [c for c in discretized.columns if c != 'E_MeV']
                if not non_energy_cols:
                    raise ValueError(
                        "No spectrum column found in discretized result."
                    )
                spectrum_col = non_energy_cols[0]
            spectrum = discretized[spectrum_col].values
            # Ensure non-negative (already done by discretize_spectra, but double-check)
            spectrum = np.maximum(spectrum, 0)
            return spectrum

        raise TypeError(
            f"initial_spectrum must be None, np.ndarray, dict, or pd.DataFrame. "
            f"Got {type(initial_spectrum)}"
        )
    def _save_figure(
        self,
        fig: plt.Figure,
        save_to: Optional[str] = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **savefig_kwargs,
    ) -> None:
        """
        Save figure to file with support for multiple formats.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to save.
        save_to : str, optional
            File path for saving. Supported extensions: .png, .jpg, .jpeg, .eps, .pdf.
            If None, figure is not saved.
        dpi : int, optional
            Resolution in dots per inch, default 300.
        bbox_inches : str, optional
            Bounding box inches, default "tight".
        **savefig_kwargs : dict
            Additional keyword arguments passed to fig.savefig().
        """
        if save_to is None:
            return
        # Validate extension
        allowed_extensions = (".png", ".jpg", ".jpeg", ".eps", ".pdf")
        if not any(save_to.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(
                f"Unsupported file extension. Allowed: {allowed_extensions}"
            )
        fig.savefig(
            save_to,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )
        print(f"Figure saved to: {save_to}")
    # --- UTILS ---
    def plot_response_functions(
        self,
        save_to: Optional[str] = None,
        show: bool = True,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **savefig_kwargs,
    ) -> None:
        """
        Plot all response functions.

        Parameters
        ----------
        save_to : str, optional
            File path to save the figure. Supported extensions: .png, .jpg, .jpeg, .eps, .pdf.
            If None, figure is not saved.
        show : bool, optional
            If True, display the figure with plt.show().
        dpi : int, optional
            Resolution for saved figure, default 300.
        bbox_inches : str, optional
            Bounding box inches, default "tight".
        **savefig_kwargs : dict
            Additional keyword arguments passed to fig.savefig().
        """
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

        # Save figure if requested
        self._save_figure(
            fig,
            save_to=save_to,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )

        if show:
            plt.show()
        plt.close()

    def plot_with_uncertainty(
        self,
        result: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        reference_spectrum: Optional[Union[pd.DataFrame, Dict]] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 8),
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_to: Optional[str] = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **savefig_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot unfolded spectrum with uncertainty range.

        Parameters
        ----------
        result : Dict[str, Any], optional
            Single unfolding result dictionary (must contain 'energy', 'spectrum',
            and optionally 'spectrum_uncert_min', 'spectrum_uncert_max').
            If not provided, uses self.current_result.
        results : Dict[str, Dict[str, Any]], optional
            Dictionary of multiple results (key: method name, value: result dict).
            If provided, plots all spectra with uncertainty ranges.
        reference_spectrum : pandas.DataFrame or dict, optional
            Reference spectrum with columns 'E_MeV' and 'Phi'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : tuple, optional
            Figure size (width, height) in inches, default (12, 8).
        colors : list of str, optional
            Colors for each spectrum (including reference). If None, uses default palette.
        title : str, optional
            Plot title. If None, generates automatic title.
        show : bool, optional
            If True, calls plt.show().
        save_to : str, optional
            File path to save the figure. Supported extensions: .png, .jpg, .jpeg, .eps, .pdf.
            If None, figure is not saved.
        dpi : int, optional
            Resolution for saved figure, default 300.
        bbox_inches : str, optional
            Bounding box inches, default "tight".
        **savefig_kwargs : dict
            Additional keyword arguments passed to fig.savefig().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.

        Examples
        --------
        >>> # Plot single result with uncertainty
        >>> result = detector.unfold_cvxpy(readings, calculate_errors=True)
        >>> detector.plot_with_uncertainty(result)
        >>> # Plot multiple results
        >>> results = {
        ...     'MLEM': result_mlem,
        ...     'cvxpy': result_cvxpy
        ... }
        >>> detector.plot_with_uncertainty(results=results)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Determine what to plot
        if results is not None:
            # Multiple results
            plot_multiple = True
            result_dict = results
        else:
            # Single result
            plot_multiple = False
            if result is None:
                result = self.current_result
                if result is None:
                    raise ValueError("No result provided and no current result available.")
            result_dict = {"result": result}

        # Prepare colors
        if colors is None:
            # Default palette: black for reference, then tab10
            default_colors = ["black", "#1f77b4", "#e68910", "#589c43", "indianred",
                              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
            # If more spectra than colors, cycle
            colors = default_colors

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # Plot reference spectrum if provided
        if reference_spectrum is not None:
            if isinstance(reference_spectrum, pd.DataFrame):
                ref_E = reference_spectrum["E_MeV"].values
                ref_Phi = reference_spectrum["Phi"].values
            elif isinstance(reference_spectrum, dict):
                ref_E = reference_spectrum["E_MeV"]
                ref_Phi = reference_spectrum["Phi"]
            else:
                raise TypeError("reference_spectrum must be DataFrame or dict")
            ax.plot(
                ref_E,
                ref_Phi,
                label="reference",
                linewidth=1,
                linestyle=":",
                color=colors[0],
            )

        # Plot each result
        for i, (method, res) in enumerate(result_dict.items()):
            color_idx = i + 1 if reference_spectrum is not None else i
            color = colors[color_idx % len(colors)]

            # Extract data
            energy = res.get("energy", self.E_MeV)
            spectrum = res.get("spectrum")
            if spectrum is None:
                raise ValueError(f"Result for '{method}' missing 'spectrum' key.")
            # Uncertainty ranges
            uncert_min = res.get("spectrum_uncert_min")
            uncert_max = res.get("spectrum_uncert_max")

            # Plot uncertainty region if available
            if uncert_min is not None and uncert_max is not None:
                ax.fill_between(
                    energy,
                    uncert_min,
                    uncert_max,
                    alpha=0.5,
                    color=color,
                    label=f"{method} uncertainty range",
                )

            # Plot spectrum line
            ax.plot(
                energy,
                spectrum,
                label=method,
                color=color,
                ls="-",
                linewidth=0.8,
                alpha=1,
            )

        # Set labels and scales
        ax.set_xlabel("Energy, MeV")
        ax.set_ylabel("Fluence per unit lethargy, F(E)E, neutron/(cm² ∙ s)")
        ax.set_xscale("log")
        # Adjust ylim
        ymax = ax.get_ylim()[1]
        if reference_spectrum is not None:
            ymax = max(ymax, np.max(ref_Phi) * 1.5)
        ax.set_ylim(0, ymax)
        ax.legend(loc="upper left", borderaxespad=0.0, fontsize=8)
        ax.grid(True, which="both", ls=":")

        # Title
        if title is None:
            if plot_multiple:
                title = "Unfolded spectra with uncertainty ranges"
            else:
                method = list(result_dict.keys())[0]
                title = f"Unfolded spectrum ({method}) with uncertainty range"
        ax.set_title(title, fontsize=14)

        # Save figure if requested
        self._save_figure(
            fig,
            save_to=save_to,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )

        if show:
            plt.show()

        return fig, ax

    def discretize_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
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
    
    def get_effective_readings_for_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> Dict[str, float]:
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