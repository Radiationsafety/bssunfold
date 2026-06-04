"""Detector class for neutron spectrum unfolding.

This module contains the main Detector class which provides methods for
neutron spectrum unfolding using various algorithms.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any, Union
from scipy.interpolate import PchipInterpolator as pchip
import warnings

from ..constants import RF_GSF, ICRP116_COEFF_EFFECTIVE_DOSE
from ..logging_config import get_logger
from ..platform_check import get_recommended_solver
from ..utils.validators import validate_readings as validate_readings_util
from ..utils.converters import convert_to_dataframe, convert_sensitivities_to_matrix
from ..utils.interpolation import interpolate_spectrum, discretize_spectra as discretize_spectra_util
from .dose_calculation import calculate_dose_rates, get_icrp116_coefficients
from .regularization import select_regularization_parameter
from .unfolding_methods import (
    solve_cvxpy,
    solve_landweber,
    solve_mlem,
    solve_qpsolvers,
)

__all__ = ["Detector"]

logger = get_logger("detector")


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
    >>> # Perform unfolding
    >>> readings = {'sphere_1': 100.5, 'sphere_2': 85.3}
    >>> result = detector.unfold_cvxpy(readings)
    """

    def __init__(
        self,
        response_functions: Optional[Union[pd.DataFrame, Dict]] = None,
        E_MeV: Optional[np.ndarray] = None,
        sensitivities: Optional[Union[Dict, np.ndarray]] = None,
    ):
        """Initialize Detector with response functions.

        Parameters
        ----------
        response_functions : pd.DataFrame, dict, optional
            Response functions data.
        E_MeV : np.ndarray, optional
            Energy grid in MeV.
        sensitivities : dict or np.ndarray, optional
            Detector sensitivities.

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
        self.cc_icrp116 = get_icrp116_coefficients()

        # Initialize results storage
        self.results_history: Dict[str, Dict[str, Any]] = {}
        self.current_result: Optional[Dict[str, Any]] = None

    def _process_input(
        self,
        response_functions: Optional[Union[pd.DataFrame, Dict]],
        E_MeV: Optional[np.ndarray],
        sensitivities: Optional[Union[Dict, np.ndarray]],
    ) -> pd.DataFrame:
        """Convert various input formats to a unified DataFrame."""
        # Case 1: response_functions is a DataFrame
        if isinstance(response_functions, pd.DataFrame):
            return response_functions.copy()

        # Case 2: response_functions is a dict
        if isinstance(response_functions, dict):
            if "E_MeV" not in response_functions:
                raise ValueError("Dictionary must contain 'E_MeV' key")
            return pd.DataFrame(response_functions)

        # Case 3: E_MeV and sensitivities provided
        if E_MeV is not None and sensitivities is not None:
            if isinstance(sensitivities, dict):
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
                        "sensitivities must be 2D array (n_energy, n_detectors)"
                    )
                if sensitivities.shape[0] != len(E_MeV):
                    raise ValueError(
                        "Number of rows in sensitivities must match "
                        "length of E_MeV"
                    )
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
        if response_functions is None and E_MeV is None and sensitivities is None:
            return pd.DataFrame(RF_GSF)

        raise ValueError(
            "Invalid input combination. Provide either response_functions "
            "(DataFrame/dict) or both E_MeV and sensitivities."
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        energy_range = f"{self.E_MeV[0]:.3e} - {self.E_MeV[-1]:.3e} MeV"
        return (
            f"Detector(energy bins: {self.n_energy_bins}, "
            f"detectors: {self.n_detectors}, "
            f"range: {energy_range})"
        )

    def __repr__(self) -> str:
        """Technical string representation."""
        return (
            f"Detector(E_MeV={self.E_MeV.tolist()}, "
            f"sensitivities={self.sensitivities})"
        )

    @property
    def n_detectors(self) -> int:
        """Number of available detectors."""
        return len(self.detector_names)

    @property
    def n_energy_bins(self) -> int:
        """Number of energy bins."""
        return len(self.E_MeV)

    def _validate_readings(
        self, readings: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate detector readings."""
        return validate_readings_util(readings, self.detector_names)

    def _build_system(
        self, readings: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build response matrix A and measurement vector b."""
        selected = [
            name for name in self.detector_names if name in readings
        ]
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
        """Create standardized output dictionary."""
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
            "doserates": calculate_dose_rates(spectrum_nonneg, self.cc_icrp116),
        }
        output.update(kwargs)
        return output

    def _convert_rf_to_matrix_variable_step(
        self, rf_df: pd.DataFrame, Emin: float = 1e-9
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Convert response functions to matrix with variable step correction."""
        if "E_MeV" in rf_df.columns:
            energies = rf_df["E_MeV"].values
            rf_data = rf_df.drop("E_MeV", axis=1)
        else:
            energies = rf_df.iloc[:, 0].values
            rf_data = rf_df.iloc[:, 1:]

        sphere_names = rf_data.columns.tolist()
        rf_array = rf_data.values

        log_energies = np.log10(energies / Emin)
        n_points = len(energies)
        log_steps = np.zeros(n_points)

        log_steps[0] = log_energies[1] - log_energies[0]
        log_steps[-1] = log_energies[-1] - log_energies[-2]

        for i in range(1, n_points - 1):
            left_step = log_energies[i] - log_energies[i - 1]
            right_step = log_energies[i + 1] - log_energies[i]
            log_steps[i] = (left_step + right_step) / 2

        ln_steps = log_steps * np.log(10)
        rf_matrix = rf_array * ln_steps[:, np.newaxis]

        return rf_matrix, energies, sphere_names, log_steps

    def _save_result(self, result: Dict[str, Any]) -> str:
        """Save unfolding result to history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = result.get("method", "unknown")
        key = f"{timestamp}_{method}"

        result["timestamp"] = timestamp
        result["saved_key"] = key
        self.results_history[key] = result.copy()
        self.current_result = result

        logger.info(f"Result saved with key: {key}")
        return key

    def get_result(
        self, key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get unfolding result from history."""
        if key is None:
            return self.current_result
        return self.results_history.get(key)

    def list_results(self) -> List[str]:
        """List all saved result keys."""
        return sorted(self.results_history.keys())

    def clear_results(self) -> None:
        """Clear all saved results."""
        self.results_history.clear()
        self.current_result = None
        logger.info("All results cleared.")

    def _normalize_initial_spectrum(
        self,
        initial_spectrum: Optional[Union[np.ndarray, Dict, pd.DataFrame]],
    ) -> Optional[np.ndarray]:
        """Normalize initial spectrum to detector's energy grid."""
        if initial_spectrum is None:
            return None

        if isinstance(initial_spectrum, np.ndarray):
            if len(initial_spectrum) != self.n_energy_bins:
                raise ValueError(
                    f"Initial spectrum length ({len(initial_spectrum)}) "
                    f"must match number of energy bins ({self.n_energy_bins})"
                )
            return np.maximum(initial_spectrum, 0)

        if isinstance(initial_spectrum, (dict, pd.DataFrame)):
            discretized = self.discretize_spectra(initial_spectrum)
            if "Phi" in discretized.columns:
                spectrum_col = "Phi"
            else:
                non_energy_cols = [
                    c for c in discretized.columns if c != "E_MeV"
                ]
                if not non_energy_cols:
                    raise ValueError("No spectrum column found")
                spectrum_col = non_energy_cols[0]
            spectrum = discretized[spectrum_col].values
            return np.maximum(spectrum, 0)

        raise TypeError(
            f"initial_spectrum must be None, np.ndarray, dict, or "
            f"pd.DataFrame. Got {type(initial_spectrum)}"
        )

    def _cosine_similarity(
        self, spectrum1: np.ndarray, spectrum2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two spectra."""
        norm1 = np.linalg.norm(spectrum1)
        norm2 = np.linalg.norm(spectrum2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(spectrum1, spectrum2) / (norm1 * norm2))

    def _add_noise(
        self, readings: Dict[str, float], noise_level: float = 0.01
    ) -> Dict[str, float]:
        """Add Gaussian noise to readings."""
        readings_noisy = {}
        for key, value in readings.items():
            noise = np.random.normal(loc=0, scale=noise_level)
            readings_noisy[key] = value * (1 + noise)
        return readings_noisy

    # Public methods delegated to unfolding_methods
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
        """Unfold neutron spectrum using convex optimization (cvxpy).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        regularization : float, optional
            Regularization parameter (default: 1e-4).
        norm : int, optional
            Norm type (1 for L1, 2 for L2), default: 2.
        solver : str, optional
            Solver to use ('ECOS' or 'default').
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: True).
        regularization_method : str, optional
            Method for selecting regularization parameter.
        noise_var : float, optional
            Noise variance for discrepancy principle.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        if solver == "default":
            solver = get_recommended_solver()

        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        # Handle regularization selection
        if regularization_method == "manual":
            alpha = regularization
        elif regularization_method == "cosine":
            if initial_spectrum is None:
                raise ValueError(
                    "For 'cosine' method, initial_spectrum must be provided."
                )
            initial_spectrum_norm = self._normalize_initial_spectrum(
                initial_spectrum
            )
            alpha = select_regularization_parameter(
                A, b, method="cosine", initial_spectrum=initial_spectrum_norm
            )
        else:
            alpha = select_regularization_parameter(
                A, b, method=regularization_method, noise_var=noise_var
            )

        # Solve using cvxpy
        x_value = solve_cvxpy(A, b, alpha, norm, solver)

        # Create output
        output = self._standardize_output(
            spectrum=x_value,
            A=A,
            b=b,
            selected=selected,
            method="cvxpy",
            norm=norm,
            solver=solver,
            regularization_method=regularization_method,
            selected_regularization=alpha,
        )

        # Monte-Carlo error estimation
        if calculate_errors:
            n = A.shape[1]
            x_montecarlo = np.empty((n_montecarlo, n))

            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_montecarlo[i] = solve_cvxpy(A_noisy, b_noisy, alpha, norm, solver)

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                    "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                    "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                    "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )

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
        """Unfold using Landweber iteration method."""
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            x0 = np.zeros(self.n_energy_bins)
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                x0 = np.zeros(self.n_energy_bins)

        x_opt, n_iter, converged = solve_landweber(
            A, b, x0, max_iterations, tolerance
        )

        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method="Landweber",
            iterations=n_iter,
            converged=converged,
        )

        if calculate_errors:
            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))
            for i in range(n_montecarlo):
                noisy_readings = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)
                x_sample, _, _ = solve_landweber(
                    A_noisy, b_noisy, x0, max_iterations, tolerance
                )
                spectra_samples[i] = x_sample

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
                    "spectrum_uncert_std": np.std(spectra_samples, axis=0),
                    "spectrum_uncert_min": np.min(spectra_samples, axis=0),
                    "spectrum_uncert_max": np.max(spectra_samples, axis=0),
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )

        if save_result:
            self._save_result(output)

        return output

    def unfold_mlem(
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
        """Unfold using MLEM algorithm."""
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)

        if initial_spectrum is None:
            x0 = np.ones(self.n_energy_bins) * 0.5
        else:
            x0 = self._normalize_initial_spectrum(initial_spectrum)
            if x0 is None:
                x0 = np.ones(self.n_energy_bins) * 0.5

        x_opt, n_iter, converged = solve_mlem(
            A, b, x0, max_iterations, tolerance
        )

        output = self._standardize_output(
            spectrum=x_opt,
            A=A,
            b=b,
            selected=selected,
            method="MLEM",
            iterations=n_iter,
            converged=converged,
        )

        if calculate_errors:
            spectra_samples = np.zeros((n_montecarlo, self.n_energy_bins))
            for i in range(n_montecarlo):
                noisy_readings = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(noisy_readings)
                x_sample, _, _ = solve_mlem(
                    A_noisy, b_noisy, x0, max_iterations, tolerance
                )
                spectra_samples[i] = x_sample

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(spectra_samples, axis=0),
                    "spectrum_uncert_std": np.std(spectra_samples, axis=0),
                    "spectrum_uncert_min": np.min(spectra_samples, axis=0),
                    "spectrum_uncert_max": np.max(spectra_samples, axis=0),
                    "montecarlo_samples": n_montecarlo,
                    "noise_level": noise_level,
                }
            )

        if save_result:
            self._save_result(output)

        return output

    def unfold_qpsolvers(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "osqp",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = True,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Unfold using qpsolvers."""
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        n = A.shape[1]

        x_value = solve_qpsolvers(
            A,
            b,
            regularization,
            norm,
            solver,
            initial_spectrum,
            smoothness_order,
            smoothness_weight,
        )

        if x_value is None:
            x_value = np.zeros(n)
            warnings.warn("Solution not found, returning zero spectrum.")

        output = self._standardize_output(
            spectrum=x_value,
            A=A,
            b=b,
            selected=selected,
            method=f"qpsolvers_{solver}",
            norm=norm,
            regularization=regularization,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
        )

        if calculate_errors:
            x_montecarlo = np.empty((n_montecarlo, n))
            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings, noise_level)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_i = solve_qpsolvers(
                    A_noisy,
                    b_noisy,
                    regularization,
                    norm,
                    solver,
                    initial_spectrum,
                    smoothness_order,
                    smoothness_weight,
                )
                x_montecarlo[i] = x_i if x_i is not None else np.zeros(n)

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                    "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                    "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                    "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
                }
            )

        if save_result:
            self._save_result(output)

        return output

    # Utility methods
    def discretize_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """Interpolate spectra onto target energy grid."""
        return discretize_spectra_util(spectra, self.E_MeV)

    def get_effective_readings_for_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> Dict[str, float]:
        """Calculate effective readings for a given spectrum."""
        if isinstance(spectra, dict):
            spectra_df = pd.DataFrame(spectra)
        elif isinstance(spectra, pd.DataFrame):
            spectra_df = spectra.copy()
        else:
            raise TypeError(
                "Input spectra must be DataFrame or dict. "
                f"Got type: {type(spectra)}"
            )

        if "E_MeV" in spectra_df.columns:
            input_energies = spectra_df["E_MeV"].values
        else:
            input_energies = spectra_df.iloc[:, 0].values

        need_interpolation = not np.array_equal(
            np.round(input_energies, 12), np.round(self.E_MeV, 12)
        )

        if need_interpolation:
            interp_spectra_df = self.discretize_spectra(spectra)
            if "Phi" in interp_spectra_df.columns:
                spectrum_values = interp_spectra_df["Phi"].values
            else:
                spectrum_values = interp_spectra_df.iloc[:, 1].values
        else:
            if "Phi" in spectra_df.columns:
                spectrum_values = spectra_df["Phi"].values
            else:
                spectrum_values = spectra_df.iloc[:, 1].values

        if len(spectrum_values) != len(self.E_MeV):
            raise ValueError(
                f"Spectrum length ({len(spectrum_values)}) must match "
                f"energy grid length ({len(self.E_MeV)})"
            )

        effective_readings = {}
        for i, detector_name in enumerate(self.detector_names):
            response_func = self.Amat[:, i]
            reading = np.sum(spectrum_values * response_func)
            reading = max(0.0, reading)
            effective_readings[detector_name] = float(reading)

        return effective_readings
