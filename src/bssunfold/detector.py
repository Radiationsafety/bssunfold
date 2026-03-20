"""Detector class with unfolding methods."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
from scipy.interpolate import pchip

from typing import Dict, Optional, List, Tuple, Any, Union
import cvxpy as cp
import odl
import warnings
from datetime import datetime
import pytikhonov as ptk
from .utils.math_utils import cosine_similarity, create_first_derivative_matrix, create_second_derivative_matrix


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
        values are arrays of same length as E_MeV. If 2D array,
        shape (n_energy, n_detectors).
        Required if `response_functions` is not provided
        and `E_MeV` is provided.

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

    def __init__(
        self, response_functions=None, E_MeV=None, sensitivities=None
    ):
        """
        Initialize Detector with response functions.

        Parameters
        ----------
        response_functions : pd.DataFrame, dict, optional
            Response functions data. Can be:
            - pandas DataFrame with 'E_MeV' column and detector columns.
            - dict with 'E_MeV' key (array) and detector names as keys
              (arrays).
            If None, default GSF response functions are used.
        E_MeV : np.ndarray, optional
            Energy grid in MeV. Required if `response_functions` is not
            provided
            and `sensitivities` is provided.
        sensitivities : dict or np.ndarray, optional
            Detector sensitivities. If dict, keys are detector names and
        values
            are arrays of same length as E_MeV. If 2D array,
            shape (n_energy, n_detectors).
            Required if `response_functions` is not provided
            and `E_MeV` is provided.

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
            if "E_MeV" not in response_functions:
                raise ValueError("Dictionary must contain 'E_MeV' key")
            return pd.DataFrame(response_functions)

        # Case 3: E_MeV and sensitivities provided
        if E_MeV is not None and sensitivities is not None:
            if isinstance(sensitivities, dict):
                # Convert dict to DataFrame
                data = {"E_MeV": E_MeV}
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
                    raise ValueError(
                        "sensitivities must be 2D array "
                        "(n_energy, n_detectors)"
                    )
                if sensitivities.shape[0] != len(E_MeV):
                    raise ValueError(
                        "Number of rows in sensitivities must match length "
                        "of E_MeV"
                    )
                # Create detector names
                detector_names = [
                    f"det_{i}" for i in range(sensitivities.shape[1])
                ]
                data = {"E_MeV": E_MeV}
                for i, name in enumerate(detector_names):
                    data[name] = sensitivities[:, i]
                return pd.DataFrame(data)
            else:
                raise TypeError("sensitivities must be dict or np.ndarray")

        # Case 4: No arguments, use default
        if (
            response_functions is None
            and E_MeV is None
            and sensitivities is None
        ):
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
        return (f"Detector(E_MeV={self.E_MeV.tolist()}, "
                f"sensitivities={self.sensitivities})")

    @property
    def n_detectors(self) -> int:
        """Number of available detectors."""
        return len(self.detector_names)

    @property
    def n_energy_bins(self) -> int:
        """Number of energy bins."""
        return len(self.E_MeV)



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

    def get_response_matrix(self, readings: Dict[str, float]) -> np.ndarray:
        """Return response matrix for given readings (spheres)."""
        selected = [name for name in self.detector_names if name in readings]
        A = np.array(
            [self.sensitivities[name] for name in selected], dtype=float
        )
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

    def get_result(
        self, key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
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
        Convert response functions DataFrame to matrix with variable step
        correction.

        Multiplies by np.log(10) and individual logarithmic energy step for
        each point.

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

    # def _select_smoothness_parameter(self, A, b, smoothness_order=1):
    #     """Выбор параметра гладкости методом L-кривой"""
    #     n = A.shape[1]
        
    #     # Создаем матрицу для выбранного порядка производной
    #     if smoothness_order == 1:
    #         L = create_first_derivative_matrix(n)
    #     elif smoothness_order == 2:
    #         L = create_second_derivative_matrix(n)
    #     else:
    #         raise ValueError("smoothness_order должен быть 1 или 2")
        
    #     # Пробуем разные значения параметра регуляризации
    #     alphas = np.logspace(-6, 2, 50)
    #     residuals = []
    #     smoothness_terms = []
        
    #     for alpha in alphas:
    #         # Решаем задачу с регуляризацией гладкости
    #         P = A.T @ A + alpha * (L.T @ L)
    #         q = -A.T @ b
            
    #         # Решаем QP задачу
    #         x = solve_qp(P=P, q=q, G=-np.eye(n), h=np.zeros(n), solver="proxqp", verbose=False)
            
    #         if x is not None:
    #             residual = np.linalg.norm(A @ x - b)
    #             smoothness = np.linalg.norm(L @ x)
    #             residuals.append(residual)
    #             smoothness_terms.append(smoothness)
        
    #     # Находим точку максимальной кривизны L-кривой
    #     if len(residuals) > 2:
    #         # Преобразуем в логарифмический масштаб
    #         log_res = np.log(residuals)
    #         log_smooth = np.log(smoothness_terms)
            
    #         # Вычисляем кривизну
    #         from scipy import interpolate
    #         from scipy.signal import savgol_filter
            
    #         # Интерполируем для равномерной параметризации
    #         tck, u = interpolate.splprep([log_res, log_smooth], s=0)
    #         unew = np.linspace(0, 1, 100)
    #         out = interpolate.splev(unew, tck)
            
    #         # Вычисляем кривизну
    #         dx = np.gradient(out[0])
    #         dy = np.gradient(out[1])
    #         ddx = np.gradient(dx)
    #         ddy = np.gradient(dy)
    #         curvature = np.abs(ddx*dy - ddy*dx) / (dx**2 + dy**2)**1.5
            
    #         # Выбираем alpha в точке максимальной кривизны
    #         idx_max_curv = np.argmax(curvature)
    #         alpha_optimal = alphas[min(idx_max_curv, len(alphas)-1)]
            
    #         return alpha_optimal
        
    #     return 1.0  # значение по умолчанию

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
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
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
            Regularization parameter, default: 1e-4. Used only when
            regularization_method='manual'.
        norm : int, optional
            Norm type for regularization (1 for L1, 2 for L2), default: 2.
            Note: automatic regularization selection methods assume L2 norm.
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
        regularization_method : str, optional
            Method for selecting regularization parameter. Options:
            - 'manual': use the `regularization` parameter (default).
            - 'lcurve': L-curve corner heuristic.
            - 'dp': Discrepancy principle.
            - 'gcv': Generalized cross validation.
            - 'all': run all methods and pick L-curve value.
            - 'cosine': maximize cosine similarity with initial_spectrum.
                Requires `initial_spectrum` parameter.
        noise_var : float, optional
            Noise variance for discrepancy principle. If None, estimated from
            residual of unregularized solution. Only used for 'dp' method.

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
            - 'spectrum_uncert_*': Uncertainty estimates (if
              calculate_errors=True)
            - 'regularization_method': method used for selecting lambda.
            - 'selected_regularization': selected lambda value.

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

            status = problem.status
            print(f"Status: {status}")
            print(f"Objective value: {problem.value}")

            if x.value is None:
                warnings.warn(
                    f"Solution variable is None. Problem status: {status}. "
                    "Returning zero vector."
                )
                return np.zeros(A.shape[1])
            # Check for non-optimal status
            if status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(
                    f"Problem status is not optimal: {status}. "
                    "Solution may be inaccurate."
                )
            return x.value

        # Validate and solve
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        # Select regularization parameter
        if regularization_method == "manual":
            alpha = regularization
            selected_lambda = alpha
        elif regularization_method == "cosine":
            if initial_spectrum is None:
                raise ValueError(
                    "For 'cosine' regularization method, initial_spectrum must be provided."
                )
            # Warn if norm != 2 (cosine method assumes L2 for consistency)
            if norm != 2:
                warnings.warn(
                    f"Cosine regularization selection method assumes L2 "
                    f"norm, but norm={norm} was requested. Using L2 for "
                    f"selection."
                )
            # Normalize initial spectrum
            initial_spectrum_norm = self._normalize_initial_spectrum(initial_spectrum)
            # Grid of alpha values
            alphas = np.logspace(-9, 2, 100)
            cosine_similarities = []
            all_unfolded_spectra = []
            # Temporary variable to store original alpha
            original_alpha = alpha if 'alpha' in locals() else None
            for alpha_val in alphas:
                alpha = alpha_val
                x_temp = _solve_problem(A, b, solver)
                all_unfolded_spectra.append(x_temp)
                cos_sim = cosine_similarity(x_temp, initial_spectrum_norm)
                cosine_similarities.append(cos_sim)
            # Restore alpha (will be set to optimal later)
            if original_alpha is not None:
                alpha = original_alpha
            # Find optimal alpha
            optimal_idx = np.argmax(cosine_similarities)
            selected_lambda = alphas[optimal_idx]
            alpha = selected_lambda
            print(
                f"Selected regularization (method=cosine): "
                f"{selected_lambda:.3e}"
            )
        else:
            # Warn if norm != 2 (automatic methods assume L2)
            if norm != 2:
                warnings.warn(
                    f"Automatic regularization selection methods assume L2 "
                    f"norm, but norm={norm} was requested. Using L2 for "
                    f"selection."
                )
            try:
                selected_lambda = self._select_regularization(
                    A, b, method=regularization_method, noise_var=noise_var
                )
            except Exception as e:
                raise ValueError(
                    f"Regularization selection failed: {e}. "
                    "Consider using manual regularization."
                )
            alpha = selected_lambda
            print(
                f"Selected regularization (method={regularization_method}): "
                f"{selected_lambda:.3e}"
            )

        # Normalize initial spectrum if provided
        if initial_spectrum is not None:
            initial_spectrum = self._normalize_initial_spectrum(
                initial_spectrum
            )
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
            regularization_method=regularization_method,
            selected_regularization=selected_lambda,
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
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo "
                "samples..."
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
            Initial spectrum approximation. If None, a uniform spectrum is
            used.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.
        max_iterations : int, optional
            Maximum number of iterations. Default is 1000.
        calculate_errors : bool, optional
            Flag for calculating restoration errors. Default is False.
        noise_level : float, optional
            Noise level for error calculation. Default is 0.01.
        n_montecarlo : int, optional
            Number of Monte Carlo samples for error calculation. Default is
            100.
        save_result : bool, optional
            If True, save result to internal history. Default is True.

        Returns
        -------
        Dict
            Dictionary containing the spectrum restoration results.
        """

        # Вспомогательная функция для создания ODL пространств
        def _create_odl_spaces(b_size: int):
            """Создание пространств ODL для заданного размера вектора
            измерений."""
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
            measurement_space, spectrum_space = _create_odl_spaces(
                len(b_vector)
            )

            # Создаем оператор (матрицу чувствительности)
            operator = odl.MatrixOperator(
                A_matrix, domain=spectrum_space, range=measurement_space
            )

            y = measurement_space.element(b_vector)
            x = _initialize_spectrum(spectrum_space, initial_spectrum_vals)

            # Запускаем алгоритм MLEM
            odl.solvers.mlem(
                operator, x, y, niter=max_iterations, callback=callback
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
            print(
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo "
                "samples..."
            )

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
                geom: 0.0
                for geom in ["AP", "PA", "LLAT", "RLAT", "ISO", "ROT"]
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
                "ICRP-116 coefficients not found. Dose calculations will "
                "return zeros."
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
        If lengths differ, raises ValueError (cannot interpolate without
        energies).
        If it's a dict or DataFrame, uses discretize_spectra to interpolate
        onto self.E_MeV,
        ensuring non-negative values.

        Parameters
        ----------
        initial_spectrum : Optional[Union[np.ndarray, Dict, pd.DataFrame]]
            Initial spectrum guess. Can be:
            - None: returns None.
            - np.ndarray: 1D array of length n_energy_bins (must match).
            - dict: with 'E_MeV' and 'Phi' keys (or similar).
            - pd.DataFrame: with 'E_MeV' column and at least one spectrum
              column.

        Returns
        -------
        Optional[np.ndarray]
            Normalized spectrum as 1D numpy array of length self.n_energy_bins,
            with negative values replaced by 0. Returns None if input is None.

        Raises
        ------
        ValueError
            If initial_spectrum is np.ndarray with wrong length, or if
            dict/DataFrame
            cannot be interpolated.
        """
        if initial_spectrum is None:
            return None

        # Case 1: numpy array
        if isinstance(initial_spectrum, np.ndarray):
            if len(initial_spectrum) != self.n_energy_bins:
                raise ValueError(
                    f"Initial spectrum length ({len(initial_spectrum)}) ",
                    f"must match number of energy bins ({self.n_energy_bins}). ",
                    "If you have a spectrum on a different energy grid, ",
                    "provide it as a dict or DataFrame with 'E_MeV' column."
                )
            # Ensure non-negative
            return np.maximum(initial_spectrum, 0)

        # Case 2: dict or DataFrame
        # Use existing discretize_spectra method which does PCHIP interpolation
        # and replaces negatives with zero.
        if isinstance(initial_spectrum, (dict, pd.DataFrame)):
            # discretize_spectra returns DataFrame with columns 'E_MeV' and at
            # least one spectrum column
            discretized = self.discretize_spectra(initial_spectrum)
            # Extract the first spectrum column (assuming single spectrum)
            # If multiple columns, take the first non-energy column
            if "Phi" in discretized.columns:
                spectrum_col = "Phi"
            else:
                # Find first column that is not 'E_MeV'
                non_energy_cols = [
                    c for c in discretized.columns if c != "E_MeV"
                ]
                if not non_energy_cols:
                    raise ValueError(
                        "No spectrum column found in discretized result."
                    )
                spectrum_col = non_energy_cols[0]
            spectrum = discretized[spectrum_col].values
            # Ensure non-negative (already done by discretize_spectra, but
            # double-check)
            spectrum = np.maximum(spectrum, 0)
            return spectrum

        raise TypeError(
            f"initial_spectrum must be None, np.ndarray, dict, or "
            f"pd.DataFrame. "
            f"Got {type(initial_spectrum)}"
        )

    def _select_regularization(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: str = "lcurve",
        noise_var: Optional[float] = None,
    ) -> float:
        """
        Select regularization parameter using pytikhonov methods.

        Parameters
        ----------
        A : np.ndarray
            Response matrix (m x n).
        b : np.ndarray
            Measurement vector (m,).
        method : str, optional
            Selection method: 'lcurve', 'dp', 'gcv', 'all'. Default 'lcurve'.
        noise_var : float, optional
            Noise variance (relative) for discrepancy principle.
            If None, estimated from residual of unregularized solution.

        Returns
        -------
        float
            Selected regularization parameter (lambda).

        Raises
        ------
        ValueError
            If method is unknown or pytikhonov fails.
        """
        import pytikhonov as ptk

        # Ensure L = I (identity) for standard Tikhonov
        n = A.shape[1]
        L = np.eye(n)

        # Create Tikhonov family
        try:
            fam = ptk.TikhonovFamily(A, L, b)
        except Exception as e:
            raise ValueError(f"Failed to create TikhonovFamily: {e}")

        if method == "lcurve":
            result = ptk.lcorner(fam)
            lam = result.get("opt_lambdah")
            if lam is None:
                raise ValueError("L-curner heuristic did not return lambda.")
        elif method == "dp":
            if noise_var is None:
                # Estimate noise variance from residual of
                # unregularized solution
                # Use least squares solution (without regularization)
                x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                residual = b - A @ x_ls
                noise_var = np.var(residual)
            delta = np.sqrt(noise_var)  # standard deviation
            result = ptk.discrepancy_principle(fam, delta=delta)
            lam = result.get("opt_lambdah")
            if lam is None:
                raise ValueError(
                    "Discrepancy principle did not return lambda."
                )
        elif method == "gcv":
            result = ptk.gcvmin(fam)
            lam = result.get("opt_lambdah")
            if lam is None:
                raise ValueError("GCV minimization did not return lambda.")
        elif method == "all":
            # Run all methods and return a dictionary
            # For compatibility, we return the L-curve value
            result = ptk.all_regparam_methods(fam)
            # result is dict with keys 'lcurve_data', 'gcv_data'
            # (DP not included)
            lcurve_data = result.get("lcurve_data", {})
            lam = lcurve_data.get("opt_lambdah")
            if lam is None:
                raise ValueError("All methods did not return lambda.")
        else:
            raise ValueError(
                f"Unknown regularization selection method: {method}. "
                "Choose from 'lcurve', 'dp', 'gcv', 'all'."
            )

        lam = float(lam)

        # Clamp extreme values to avoid numerical issues
        if lam > 1e6:
            warnings.warn(
                f"Selected regularization parameter ({lam:.3e}) is too "
                f"large. Clamping to 1e6."
            )
            lam = 1e6
        if lam < 1e-12:
            warnings.warn(
                f"Selected regularization parameter ({lam:.3e}) is too "
                f"small. Clamping to 1e-12."
            )
            lam = 1e-12

        return lam

    def _compare_regularization_methods(
        self,
        A: np.ndarray,
        b: np.ndarray,
        noise_var: Optional[float] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare all regularization parameter selection methods.

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
            If True, generate comparison plot using "
            "pytikhonov.plot_all_methods.
        plot_path : str, optional
            Path to save the plot. If None, plot is shown (if plot=True).

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'lcurve': dict from fam.lcorner()
            - 'dp': dict from fam.discrepancy_principle()
            - 'gcv': dict from fam.gcvmin()
            - 'all_data': dict from pytikhonov.all_regparam_methods()
            - 'selected': dict mapping method name to selected lambda.
        """

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

    def compare_regularization_methods(
        self,
        readings: Dict[str, float],
        noise_var: Optional[float] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Public method to compare regularization selection methods for given
        readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
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
            Comparison results as returned by _compare_regularization_methods.
        """
        readings = self._validate_readings(readings)
        A, b, _ = self._build_system(readings)
        return self._compare_regularization_methods(
            A, b, noise_var=noise_var, plot=plot, plot_path=plot_path
        )

    def _randomization_experiment(
        self,
        A: np.ndarray,
        b: np.ndarray,
        noise_var: Optional[float] = None,
        n_samples: int = 10,
        rseed: int = 0,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run randomization experiments for regularization parameter selection.

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
        """
        import pytikhonov as ptk

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
                raise ValueError(f"Unknown randomization method: {method}")

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

    def randomization_experiment(
        self,
        readings: Dict[str, float],
        noise_var: Optional[float] = None,
        n_samples: int = 10,
        rseed: int = 0,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Public method to run randomization experiments for given readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
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
            Results as returned by _randomization_experiment.
        """
        readings = self._validate_readings(readings)
        A, b, _ = self._build_system(readings)
        return self._randomization_experiment(
            A,
            b,
            noise_var=noise_var,
            n_samples=n_samples,
            rseed=rseed,
            methods=methods,
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
        Wrapper for the standalone save_figure function from bssunfold.plots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to save.
        save_to : str, optional
            File path for saving. Supported extensions: .png, .jpg, .jpeg, "
            ".eps, .pdf."
            If None, figure is not saved.
        dpi : int, optional
            Resolution in dots per inch, default 300.
        bbox_inches : str, optional
            Bounding box inches, default "tight".
        **savefig_kwargs : dict
            Additional keyword arguments passed to fig.savefig().
        """
        from bssunfold.plots import save_figure
        return save_figure(fig, save_to=save_to, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)

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
            File path to save the figure. Supported extensions: .png, .jpg, "
            ".jpeg, .eps, .pdf."
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
        from bssunfold.plots import plot_response_functions as plot_rf
        plot_rf(self, save_to=save_to, show=show, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)

    def plot_with_uncertainty(
        self,
        result: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        reference_spectrum: Optional[Union[pd.DataFrame, Dict]] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 8),
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        plot_style: str = 'fill_between',  # Новый параметр: 'fill_between' или 'errorbar'
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
            Single unfolding result dictionary (must contain 'energy',
            'spectrum',
            and optionally 'spectrum_uncert_min', 'spectrum_uncert_max').
            If not provided, uses self.current_result.
        results : Dict[str, Dict[str, Any]], optional
            Dictionary of multiple results (key: method name, value: result
            dict).
            If provided, plots all spectra with uncertainty ranges.
        reference_spectrum : pandas.DataFrame or dict, optional
            Reference spectrum with columns 'E_MeV' and 'Phi'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : tuple, optional
            Figure size (width, height) in inches, default (12, 8).
        colors : list of str, optional
            Colors for each spectrum (including reference). If None, uses
            default palette.
        title : str, optional
            Plot title. If None, generates automatic title.
        plot_style : str, optional
            Style for uncertainty visualization:
            - 'fill_between' - filled region between min and max
            - 'errorbar' - error bars using standard deviation
            Default 'fill_between'.
        show : bool, optional
            If True, calls plt.show().
        save_to : str, optional
            File path to save the figure. Supported extensions: .png, .jpg, "
            ".jpeg, .eps, .pdf."
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
        >>> # Plot single result with uncertainty as fill_between
        >>> result = detector.unfold_cvxpy(readings, calculate_errors=True)
        >>> detector.plot_with_uncertainty(result)
        >>>
        >>> # Plot with error bars using standard deviation
        >>> detector.plot_with_uncertainty(result, plot_style='errorbar')
        >>>
        >>> # Plot multiple results
        >>> results = {
        ...     'MLEM': result_mlem,
        ...     'cvxpy': result_cvxpy
        ... }
        >>> detector.plot_with_uncertainty(results=results, plot_style='errorbar')
        """
        from bssunfold.plots import plot_with_uncertainty as plot_uncertainty
        return plot_uncertainty(
            self,
            result=result,
            results=results,
            reference_spectrum=reference_spectrum,
            ax=ax,
            figsize=figsize,
            colors=colors,
            title=title,
            plot_style=plot_style,
            show=show,
            save_to=save_to,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )

    def discretize_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """
        Interpolate spectra onto the target energy grid using PCHIP
        interpolation.
        Extrapolation is avoided and replaced with zeros.

        Parameters:
        -----------
        spectra : pandas.DataFrame or dict
            Input spectra to be discretized. If DataFrame, it should contain
            energy
            values. If dict, it should have 'E_MeV' and 'Phi' keys.

        Returns:
        --------
        pandas.DataFrame
            Discretized spectra with columns 'E_MeV' and 'Phi'
        """
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
            raise TypeError(
                "Input spectra must be either a pandas DataFrame or a "
                "dictionary"
            )

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
            print(
                "Warning: Target energy bins extend below the input grid "
                "minimum. Setting values to zero."
            )

        if new_ebins.max() > np.max(energies):
            print(
                "Warning: Target energy bins extend above the input grid "
                "maximum. Setting values to zero."
            )

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
                    if "Phi" in spectra:
                        new_spectra[f"Phi_{i}"] = interpolated_values[:, i]
                    else:
                        # Try to use original column names
                        col_name = (
                            spectre_df.columns[1]
                            if len(spectre_df.columns) > 1
                            else f"Phi_{i}"
                        )
                        new_spectra[col_name] = interpolated_values[:, i]
                else:
                    # For multiple spectra from DataFrame, preserve original
                    # names
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

        This function interpolates the input spectrum onto the detector's "
        "energy grid,
        then calculates what readings each detector sphere would measure for "
        "that spectrum
        by integrating the product of the spectrum and response functions.

        Parameters
        ----------
        spectra : pandas.DataFrame or dict
            Input spectra to calculate effective readings for.
            - If DataFrame: Should contain 'E_MeV' column for energies and "
            "one or more
            columns for spectral data (e.g., 'Phi').
            - If dict: Should have 'E_MeV' and spectral data keys.

        Returns
        -------
        dict
            Dictionary of effective readings for each detector sphere in the "
            "format:
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
        >>> readings = detector.get_effective_readings_for_spectra(
        ...     spectrum_dict)
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
                "Input spectra must be either a pandas DataFrame or a "
                f"dictionary. Got type: {type(spectra)}"
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
            np.round(input_energies, 12), np.round(self.E_MeV, 12)
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
        # where R(E) is the response function (already includes log steps in
        # Amat)

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

    def unfold_qpsolvers(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "proxqp",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = True,
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        smoothness_order: int = 0,  
        smoothness_weight: float = 1.0, 
    ) -> Dict:
        """
        Восстановление спектра с помощью библиотеки qpsolvers.
        ... (документация)
        """
        import warnings
        from qpsolvers import available_solvers, solve_qp
        

        def _solve_problem_qpsolvers(
            A: np.ndarray, 
            b: np.ndarray, 
            alpha: float, 
            norm_type: int,
            solver_name: str,
            x_init: Optional[np.ndarray] = None,
            smoothness_order: int = 0,
            smoothness_weight: float = 1.0
        ) -> np.ndarray:
            """Решение задачи оптимизации с поддержкой условий гладкости."""
            n_vars = A.shape[1]

            # Базовая задача МНК: 0.5 * ||Ax - b||^2
            P = A.T @ A
            q = -A.T @ b

            # Проверка доступности солвера
            if solver_name not in available_solvers:
                raise ValueError(
                    f"Солвер '{solver_name}' не доступен. "
                    f"Установленные солверы: {available_solvers}."
                )

            # Базовая настройка ограничений: x >= 0
            G = -np.eye(n_vars)
            h = np.zeros(n_vars)

            # Обработка разных типов регуляризации
            if norm_type == 2:
                # L2 регуляризация с возможностью добавления гладкости
                
                # Создаем матрицу для регуляризации гладкости
                if smoothness_order == 1:
                    L = create_first_derivative_matrix(n_vars)
                    # Добавляем гладкость: alpha * ||Lx||^2
                    P += alpha * smoothness_weight * (L.T @ L)
                elif smoothness_order == 2:
                    L = create_second_derivative_matrix(n_vars)
                    P += alpha * smoothness_weight * (L.T @ L)
                else:
                    # Стандартная регуляризация Тихонова
                    P += alpha * np.eye(n_vars)

                # Решение задачи QP
                x = solve_qp(
                    P=P,
                    q=q,
                    G=G,
                    h=h,
                    solver=solver_name,
                    initvals=x_init,
                    verbose=False
                )

            elif norm_type == 1:
                # L1 регуляризация через расширение переменных
                n_extended = 2 * n_vars
                P_ext = np.zeros((n_extended, n_extended))
                P_ext[:n_vars, :n_vars] = P
                
                # Добавляем гладкость если нужно
                if smoothness_order == 1:
                    L = create_first_derivative_matrix(n_vars)
                    P_ext[:n_vars, :n_vars] += alpha * smoothness_weight * (L.T @ L)
                elif smoothness_order == 2:
                    L = create_second_derivative_matrix(n_vars)
                    P_ext[:n_vars, :n_vars] += alpha * smoothness_weight * (L.T @ L)
                
                q_ext = np.zeros(n_extended)
                q_ext[:n_vars] = q
                q_ext[n_vars:] = alpha * np.ones(n_vars)

                # Ограничения: x - t <= 0, -x - t <= 0
                G_ext = np.zeros((2 * n_vars, n_extended))
                G_ext[:n_vars, :n_vars] = np.eye(n_vars)
                G_ext[:n_vars, n_vars:] = -np.eye(n_vars)
                G_ext[n_vars:, :n_vars] = -np.eye(n_vars)
                G_ext[n_vars:, n_vars:] = -np.eye(n_vars)
                
                h_ext = np.zeros(2 * n_vars)

                # Дополнительное ограничение: x >= 0
                G_x = np.zeros((n_vars, n_extended))
                G_x[:, :n_vars] = -np.eye(n_vars)
                G_ext = np.vstack([G_ext, G_x])
                h_ext = np.append(h_ext, np.zeros(n_vars))

                # Начальное приближение для расширенной задачи
                x_init_ext = None
                if x_init is not None:
                    x_init_ext = np.concatenate([x_init, np.abs(x_init)])

                x_ext = solve_qp(
                    P=P_ext,
                    q=q_ext,
                    G=G_ext,
                    h=h_ext,
                    solver=solver_name,
                    initvals=x_init_ext,
                    verbose=False
                )

                if x_ext is None:
                    return None
                x = x_ext[:n_vars]

            else:
                raise ValueError(f"Неподдерживаемый тип нормы: {norm_type}")

            if x is None:
                print(f"Предупреждение: {solver_name} не нашел решение.")
                return None

            return x

        # Основной код метода
        
        
        # Валидация и основное решение
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        n = A.shape[1]

        # Выбор метода регуляризации
        if regularization_method == "manual":
            alpha = regularization
            selected_lambda = alpha
        elif regularization_method == "cosine":
            if initial_spectrum is None:
                raise ValueError(
                    "For 'cosine' regularization method, initial_spectrum must be provided."
                )
            if norm != 2:
                warnings.warn(
                    f"Cosine regularization selection method assumes L2 "
                    f"norm, but norm={norm} was requested. Using L2 for "
                    f"selection."
                )
            initial_spectrum_norm = self._normalize_initial_spectrum(initial_spectrum)
            alphas = np.logspace(-9, 2, 100)
            cosine_similarities = []
            
            for alpha_val in alphas:
                x_temp = _solve_problem_qpsolvers(
                    A, b, alpha_val, 2, solver, 
                    x_init=initial_spectrum_norm,
                    smoothness_order=smoothness_order,
                    smoothness_weight=smoothness_weight
                )
                if x_temp is not None:
                    cos_sim = cosine_similarity(x_temp, initial_spectrum_norm)
                    cosine_similarities.append(cos_sim)
                else:
                    cosine_similarities.append(-1)  # штраф за не найденное решение
            
            optimal_idx = np.argmax(cosine_similarities)
            selected_lambda = alphas[optimal_idx]
            alpha = selected_lambda
            print(f"Selected regularization (method=cosine): {selected_lambda:.3e}")
        else:
            if norm != 2:
                warnings.warn(
                    f"Automatic regularization selection methods assume L2 "
                    f"norm, but norm={norm} was requested. Using L2 for "
                    f"selection."
                )
            try:
                selected_lambda = self._select_regularization(
                    A, b, method=regularization_method, noise_var=noise_var
                )
            except Exception as e:
                raise ValueError(
                    f"Regularization selection failed: {e}. "
                    "Consider using manual regularization."
                )
            alpha = selected_lambda
            print(f"Selected regularization (method={regularization_method}): {selected_lambda:.3e}")

        # Основное решение с учетом гладкости
        x_value = _solve_problem_qpsolvers(
            A, b, alpha, norm, solver, 
            x_init=initial_spectrum,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight
        )

        if x_value is None:
            x_value = np.zeros(n)
            print("Внимание: решение не найдено, возвращен нулевой спектр.")

        computed_readings = A @ x_value
        residual = b - computed_readings
        residual_norm = np.linalg.norm(residual)
        print(f"Норма невязки: {residual_norm:.6f}")

        # Формирование основных результатов
        output = self._standardize_output(
            spectrum=x_value,
            A=A,
            b=b,
            selected=selected,
            method=f"qpsolvers_{solver}",
            norm=norm,
            regularization=regularization,
            regularization_method=regularization_method,
            selected_regularization=selected_lambda,
        )

        # Добавляем информацию о гладкости в выходные данные
        output['smoothness_order'] = smoothness_order
        output['smoothness_weight'] = smoothness_weight

        # Расчет погрешностей методом Монте-Карло
        if calculate_errors:
            print("Calculating uncertainty with Monte-Carlo...")
            
            x_montecarlo = np.empty((n_montecarlo, n))
            
            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_i = _solve_problem_qpsolvers(
                    A_noisy, b_noisy, regularization, norm, solver,
                    smoothness_order=smoothness_order,
                    smoothness_weight=smoothness_weight
                )
                if x_i is not None:
                    x_montecarlo[i] = x_i
                else:
                    x_montecarlo[i] = x_montecarlo[i-1] if i > 0 else np.zeros(n)
            
            output.update({
                "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
            })
            print("...uncertainty calculated.")

        return output
        
    def unfold_combined(
        self,
        readings: Dict[str, float],
        pipeline: List[Dict[str, Any]],
        calculate_errors: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Комбинированный метод восстановления спектра, позволяющий последовательно
        применять несколько методов, используя результат предыдущего как начальное 
        приближение для следующего.
        
        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        pipeline : List[Dict[str, Any]]
            Список методов для последовательного применения. Каждый элемент словаря должен содержать:
            - 'method': str - название метода (например, 'cvxpy', 'landweber', 'gravel', и т.д.)
            - 'params': dict - параметры для данного метода
            - 'use_as_initial' : bool (опционально) - использовать ли результат как начальное 
                                приближение для следующего метода (по умолчанию True)
            - 'store_intermediate' : bool (опционально) - сохранять ли промежуточный результат
                                    в выходном словаре (по умолчанию False)
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления для последнего метода
        verbose : bool, optional
            Флаг вывода отладочной информации
        
        Возвращает:
        Dict
            Словарь с результатами восстановления спектра, включая:
            - 'final_result': результат последнего метода
            - 'intermediate_results': словарь с промежуточными результатами (если store_intermediate=True)
            - 'pipeline_info': информация о применённых методах
        """
        
        readings = self._validate_readings(readings)
        current_spectrum = None
        intermediate_results = {}
        final_result = None
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Combined algorithm, methods = {len(pipeline)} ")
            print(f"{'='*60}")
        
        for i, stage in enumerate(pipeline):
            method = stage['method']
            params = stage.get('params', {}).copy()
            use_as_initial = stage.get('use_as_initial', True)
            store_intermediate = stage.get('store_intermediate', False)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Stage {i+1}/{len(pipeline)}: {method}")
                print(f"{'='*60}")
                print(f"Parameters: {params}")
            
            # Если есть текущий спектр, передаём его как начальное приближение
            if current_spectrum is not None and use_as_initial:
                # Определяем, какой параметр отвечает за начальное приближение для каждого метода
                initial_param_names = {
                    'landweber': 'initial_spectrum',
                    'mlem': 'initial_spectrum',
                    'cvxpy': 'initial_spectrum',
                    'qpsolvers': 'initial_spectrum',
                    'pytikhonov': 'initial_spectrum',
                }
                
                if method in initial_param_names:
                    params[initial_param_names[method]] = current_spectrum.copy()
                    if verbose:
                        print(f" Previous results is used as initial spectrum")
            
            # Вызов соответствующего метода
            method_func = getattr(self, f'unfold_{method}', None)
            if method_func is None:
                raise ValueError(f"Method '{method}'not found in Detector class")
            
            # Добавляем calculate_errors только для последнего метода, если нужно
            if i == len(pipeline) - 1 and calculate_errors:
                params['calculate_errors'] = True
            else:
                params['calculate_errors'] = False
            
            # Выполняем метод
            try:
                result = method_func(readings, **params)
            except Exception as e:
                print(f"Ошибка при выполнении метода {method}: {e}")
                raise
            
            # Обновляем текущий спектр
            if 'spectrum' in result:
                current_spectrum = result['spectrum'].copy()
                if verbose:
                    print(f" Spectrum norm: {np.linalg.norm(current_spectrum):.6f}")
                    print(f" Residual normи: {result.get('residual norm', 'N/A')}")
            
            # Сохраняем промежуточный результат, если нужно
            if store_intermediate:
                intermediate_results[f'stage_{i+1}_{method}'] = result.copy()
            
            # Запоминаем финальный результат
            final_result = result
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Combined method finished")
            print(f"final residual norm: {final_result.get('residual norm', 'N/A')}")
            print(f"{'='*60}")
        
        # Формируем итоговый результат
        output = final_result.copy()
        output['pipeline_info'] = {
            'stages': [stage['method'] for stage in pipeline],
            'params': [stage.get('params', {}) for stage in pipeline]
        }
        
        if intermediate_results:
            output['intermediate_results'] = intermediate_results
        
        return output


    def unfold_doroshenko(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: float = 0.0,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Doroshenko coordinate update method.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings (counts or dose rates)
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, uniform spectrum is used
        max_iterations : int, optional
            Maximum number of iterations, default: 1000
        tolerance : float, optional
            Convergence tolerance for solution change, default: 1e-6
        regularization : float, optional
            Regularization strength to prevent division by zero, default: 0.0
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
            - 'method': Method name ('Doroshenko')
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
        >>> result = detector.unfold_doroshenko(
        ...     readings,
        ...     max_iterations=500,
        ...     tolerance=1e-5,
        ...     regularization=0.001,
        ...     calculate_errors=True
        ... )
        >>> print(f"Converged: {result['converged']}")
        >>> print(f"Iterations: {result['iterations']}")
        """

        def _doroshenko_iteration(
            A: np.ndarray,
            b: np.ndarray,
            x0: np.ndarray,
            max_iter: int,
            tol: float,
            reg: float,
        ) -> Tuple[np.ndarray, int, bool]:
            """Core Doroshenko coordinate update iteration implementation."""
            x = x0.copy()
            
            # Precompute denominators for each coordinate (sum of squares of column A[:,i])
            denominator_cache = np.sum(A * A, axis=0) + reg

            converged = False
            iterations = 0

            for i in range(max_iter):
                x_old = x.copy()
                
                # Update each coordinate sequentially
                for j in range(x.size):
                    # Calculate contribution from all coordinates except current one
                    ax_without_j = A[:, :j] @ x[:j] + A[:, j + 1:] @ x[j + 1:]
                    
                    # Compute numerator for the update
                    numerator = np.dot(A[:, j], b - ax_without_j)
                    
                    # Update coordinate with non-negativity constraint
                    if denominator_cache[j] > 0:
                        x[j] = max(0.0, numerator / denominator_cache[j])
                
                # Check convergence based on change in solution
                if np.linalg.norm(x - x_old) < tol:
                    converged = True
                    iterations = i + 1
                    break

            if not converged:
                iterations = max_iter

            return x, iterations, converged

        # Validate and prepare data
        validated_readings = self._validate_readings(readings)
        A, b, selected = self._build_system(validated_readings)

        # Set initial spectrum (normalize if needed)
        if initial_spectrum is None:
            x0 = np.ones(self.n_energy_bins)
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                x0 = np.ones(self.n_energy_bins)

        # Main Doroshenko iteration
        x_opt, n_iter, converged = _doroshenko_iteration(
            A, b, x0, max_iterations, tolerance, regularization
        )

        # Create standard output
        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method="Doroshenko",
            iterations=n_iter,
            converged=converged,
            tolerance=tolerance,
            regularization=regularization,
        )

        # Monte-Carlo uncertainty estimation
        if calculate_errors:
            print(
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo "
                "samples..."
            )

            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))

            for i in range(n_montecarlo):
                # Add noise to readings
                noisy_readings = self._add_noise(
                    validated_readings, noise_level
                )

                # Rebuild system with noisy readings
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)

                # Run Doroshenko with same parameters
                x_sample, _, _ = _doroshenko_iteration(
                    A_noisy, b_noisy, x0, max_iterations, tolerance, regularization
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

    def unfold_lmfit(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "lbfgsb",
        model_name: str = "elastic",
        regularization: float = 1e-4,
        regularization2: float = 1e-4,
        l1_weight: float = 0.5,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using lmfit with L1/L2/Elastic regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings (counts or dose rates)
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, uniform spectrum based on mean readings is used
        method : str, optional
            lmfit solver name (leastsq, newton, tnc, cg, bfgs, lbfgsb), default: "lbfgsb"
        model_name : str, optional
            Regularization model: elastic, lasso, ridge, default: "elastic"
        regularization : float, optional
            L1 regularization strength, default: 1e-4
        regularization2 : float, optional
            L2 regularization strength for elastic net, default: 1e-4
        l1_weight : float, optional
            L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5
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
            - 'method': Method name ('lmfit (method)')
            - 'model_name': Regularization model used
            - 'regularization': Regularization parameter(s)
            - 'success': Whether optimization succeeded
            - 'message': Optimization status message
            - 'nfev': Number of function evaluations
            - 'doserates': Dose rates for different geometries [pSv/s]
            - 'initial_spectrum': Initial spectrum used
            - 'spectrum_uncert_*': Monte-Carlo uncertainty estimates
                (if calculate_errors=True)

        Raises
        ------
        ValueError
            If readings are invalid, dimensions mismatch, or model_name unknown
        ImportError
            If lmfit is not installed

        Examples
        --------
        >>> readings = {'sphere_1': 150.2, 'sphere_2': 120.5}
        >>> initial_guess = np.ones(40) * 10.0
        >>> result = detector.unfold_lmfit(
        ...     readings,
        ...     initial_spectrum=initial_guess,
        ...     method='lbfgsb',
        ...     model_name='elastic',
        ...     regularization=1e-3,
        ...     regularization2=1e-3,
        ...     l1_weight=0.3,
        ...     calculate_errors=True
        ... )
        >>> print(f"Success: {result['success']}")
        >>> print(f"Function evaluations: {result['nfev']}")
        """

        def _lmfit_optimization(
            A: np.ndarray,
            b: np.ndarray,
            x0: np.ndarray,
            method: str,
            model_name: str,
            reg: float,
            reg2: float,
            l1_w: float,
        ) -> Tuple[np.ndarray, bool, str, int]:
            """Core lmfit optimization implementation."""
            lmfit = self._import_optional("lmfit", "lmfit-based unfolding")
            m = A.shape[1]

            # Initialize parameters with initial spectrum
            params = lmfit.Parameters()
            for i in range(m):
                params.add(f"x{i}", value=max(x0[i], 1e-10), min=0.0)

            # Define residual functions
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

            # Run optimization based on model
            if model_name == "lasso":
                result = lmfit.minimize(
                    residual_lasso,
                    params,
                    args=(A, b, reg),
                    method=method,
                )
            elif model_name == "ridge":
                result = lmfit.minimize(
                    residual_ridge,
                    params,
                    args=(A, b, reg),
                    method=method,
                )
            elif model_name == "elastic":
                result = lmfit.minimize(
                    residual_elastic,
                    params,
                    args=(A, b, reg, reg2, l1_w),
                    method=method,
                )
            else:
                raise ValueError("model_name must be one of: elastic, lasso, ridge")

            spectrum = np.array([result.params[f"x{i}"].value for i in range(m)])
            return spectrum, result.success, result.message, result.nfev

        # Validate and prepare data
        validated_readings = self._validate_readings(readings)
        A, b, selected = self._build_system(validated_readings)

        # Set initial spectrum (normalize if needed)
        if initial_spectrum is None:
            # Default initialization: uniform spectrum based on mean readings
            x0 = np.ones(self.n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                # Fallback to default if normalization fails
                x0 = np.ones(self.n_energy_bins) * np.mean(b) / np.mean(A.sum(axis=1))
                print("Warning: Initial spectrum normalization failed, using default initialization")

        # Main lmfit optimization
        x_opt, success, message, nfev = _lmfit_optimization(
            A, b, x0, method, model_name, regularization, regularization2, l1_weight
        )

        # Create standard output
        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method=f"lmfit ({method})",
            model_name=model_name,
        )
        
        # Add lmfit-specific information
        output.update(
            {
                "regularization": regularization,
                "regularization2": regularization2 if model_name == "elastic" else None,
                "l1_weight": l1_weight if model_name == "elastic" else None,
                "success": success,
                "message": message,
                "nfev": nfev,
                "initial_spectrum": x0.copy(),  # Save initial spectrum used
            }
        )

        # Monte-Carlo uncertainty estimation
        if calculate_errors:
            print(
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo "
                "samples..."
            )

            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))

            for i in range(n_montecarlo):
                # Add noise to readings
                noisy_readings = self._add_noise(
                    validated_readings, noise_level
                )

                # Rebuild system with noisy readings
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)

                # Run lmfit with same parameters and initial spectrum
                x_sample, _, _, _ = _lmfit_optimization(
                    A_noisy, b_noisy, x0, method, model_name, 
                    regularization, regularization2, l1_weight
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


    def unfold_kaczmarz(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        omega: float = 1.0,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Unfold neutron spectrum using the Kaczmarz algorithm (ART).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings (counts or dose rates)
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, zero spectrum is used
        max_iterations : int, optional
            Maximum number of iterations, default: 1000
        omega : float, optional
            Relaxation parameter (0 < omega <= 2), default: 1.0
        tolerance : float, optional
            Convergence tolerance for solution change, default: 1e-6
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
            - 'method': Method name ('Kaczmarz')
            - 'iterations': Number of iterations performed
            - 'converged': Whether convergence was achieved
            - 'omega': Relaxation parameter used
            - 'doserates': Dose rates for different geometries [pSv/s]
            - 'spectrum_uncert_*': Monte-Carlo uncertainty estimates
                (if calculate_errors=True)

        Raises
        ------
        ValueError
            If readings are invalid, dimensions mismatch, or omega out of range

        Examples
        --------
        >>> readings = {'sphere_1': 150.2, 'sphere_2': 120.5}
        >>> result = detector.unfold_kaczmarz(
        ...     readings,
        ...     max_iterations=1000,
        ...     omega=1.5,
        ...     tolerance=1e-5,
        ...     calculate_errors=True
        ... )
        >>> print(f"Converged: {result['converged']}")
        >>> print(f"Residual norm: {result['residual_norm']:.2e}")
        """

        def _kaczmarz_iteration(
            A: np.ndarray,
            b: np.ndarray,
            x0: np.ndarray,
            max_iter: int,
            omega: float,
            tol: float,
        ) -> Tuple[np.ndarray, int, bool]:
            """Core Kaczmarz iteration implementation."""
            m, n = A.shape
            x = x0.copy()
            
            # Validate relaxation parameter
            if omega <= 0 or omega > 2:
                print(f"Warning: omega={omega} outside recommended range (0,2]")
            
            # Precompute squared norms of rows for efficiency
            row_norms_sq = np.sum(A * A, axis=1)
            
            converged = False
            iterations = 0
            x_old = x.copy()

            for k in range(max_iter):
                i = k % m  # Cyclic access pattern
                
                # Skip rows with zero norm
                if row_norms_sq[i] > 0:
                    # Compute update
                    update = (b[i] - np.dot(A[i], x)) / row_norms_sq[i]
                    x = x + omega * update * A[i]
                    
                    # Apply non-negativity constraint
                    x = np.maximum(x, 0)
                
                # Check convergence after each full cycle
                if (k + 1) % m == 0:
                    if np.linalg.norm(x - x_old) < tol:
                        converged = True
                        iterations = k + 1
                        break
                    x_old = x.copy()

            if not converged:
                iterations = max_iter

            return x, iterations, converged

        # Validate and prepare data
        validated_readings = self._validate_readings(readings)
        A, b, selected = self._build_system(validated_readings)

        # Set initial spectrum
        if initial_spectrum is None:
            x0 = np.zeros(self.n_energy_bins)
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                x0 = np.zeros(self.n_energy_bins)

        # Main Kaczmarz iteration
        x_opt, n_iter, converged = _kaczmarz_iteration(
            A, b, x0, max_iterations, omega, tolerance
        )

        # Create standard output
        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method="Kaczmarz",
            iterations=n_iter,
            converged=converged,
            tolerance=tolerance,
            omega=omega,
        )

        # Monte-Carlo uncertainty estimation
        if calculate_errors:
            print(
                f"Calculating uncertainty with {n_montecarlo} Monte-Carlo "
                "samples..."
            )

            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))

            for i in range(n_montecarlo):
                # Add noise to readings
                noisy_readings = self._add_noise(
                    validated_readings, noise_level
                )

                # Rebuild system with noisy readings
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)

                # Run Kaczmarz with same parameters
                x_sample, _, _ = _kaczmarz_iteration(
                    A_noisy, b_noisy, x0, max_iterations, omega, tolerance
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