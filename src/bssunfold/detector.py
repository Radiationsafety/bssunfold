"""Detector class with unfolding methods."""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import cvxpy as cp
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
            DataFrame containing response functions with 'E_MeV' as first column
            and detector names as subsequent columns.

        Raises
        ------
        ValueError
            If E_MeV is not a 1D array or has less than 2 energy points
        """
        Amat, E_MeV, detector_names, log_steps = (
            self._convert_rf_to_matrix_variable_step(response_functions_df, Emin=1e-9)
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
        return (
            f"Detector(E_MeV={self.E_MeV.tolist()}, sensitivities={self.sensitivities})"
        )

    @property
    def n_detectors(self) -> int:
        """Number of available detectors."""
        return len(self.detector_names)

    @property
    def n_energy_bins(self) -> int:
        """Number of energy bins."""
        return len(self.E_MeV)

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

    def _validate_readings(self, readings: Dict[str, float]) -> Dict[str, float]:
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
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)
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
                name: float(val) for name, val in zip(selected, computed_readings)
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
            A: np.ndarray, b: np.ndarray, use_solver: str = None
        ) -> np.ndarray:
            """Solve optimization problem."""
            x = cp.Variable(A.shape[1], nonneg=True)
            objective = cp.Minimize(cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm))
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

            n_montecarlo = 1000
            x_montecarlo = np.empty((n_montecarlo, n))

            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_montecarlo[i] = _solve_problem(A_noisy, b_noisy, solver)

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                    "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                    "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                    "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
                }
            )
            print("...uncertainty calculated.")

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
            - 'spectrum_uncert_*': Monte-Carlo uncertainty estimates (if calculate_errors=True)

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
            A: np.ndarray, b: np.ndarray, x0: np.ndarray, max_iter: int, tol: float
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
        if initial_spectrum is None:
            # x0 = np.ones(self.n_energy_bins) * np.mean(b) / np.mean(A)
            x0 = np.zeros(self.n_energy_bins)
        else:
            if len(initial_spectrum) != self.n_energy_bins:
                raise ValueError(
                    f"Initial spectrum length ({len(initial_spectrum)}) "
                    f"must match number of energy bins ({self.n_energy_bins})"
                )
            x0 = np.maximum(initial_spectrum, 0)

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
            print(f"Calculating uncertainty with {n_montecarlo} Monte-Carlo samples...")

            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))

            for i in range(n_montecarlo):
                # Add noise to readings
                noisy_readings = self._add_noise(validated_readings, noise_level)

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
                    "spectrum_uncert_median": np.median(spectra_samples, axis=0),
                    "spectrum_uncert_percentile_5": np.percentile(
                        spectra_samples, 5, axis=0
                    ),
                    "spectrum_uncert_percentile_95": np.percentile(
                        spectra_samples, 95, axis=0
                    ),
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )
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
            return {geom: 0.0 for geom in ["AP", "PA", "LLAT", "RLAT", "ISO", "ROT"]}

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
