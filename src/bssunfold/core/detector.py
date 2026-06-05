"""Detector class for neutron spectrum unfolding.

This module contains the main Detector class which provides methods for
neutron spectrum unfolding using various algorithms.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any, Union

from ..constants import RF_GSF
from ..logging_config import get_logger
from ..utils.validators import validate_readings
from ..utils.interpolation import discretize_spectra
from ..utils.plotting import plot_with_uncertainty
from .dose_calculation import calculate_dose_rates, get_icrp116_coefficients
from .regularization import (
    compare_regularization_methods as compare_reg_util,
    randomization_experiment as rand_exp_util,
)
from .unfold_cvxpy import unfold_cvxpy as unfold_cvxpy_impl
from .unfold_landweber import unfold_landweber as unfold_landweber_impl
from .unfold_mlem import unfold_mlem as unfold_mlem_impl
from .unfold_qpsolvers import unfold_qpsolvers as unfold_qpsolvers_impl
from .unfold_doroshenko import unfold_doroshenko as unfold_doroshenko_impl
from .unfold_kaczmarz import unfold_kaczmarz as unfold_kaczmarz_impl
from .unfold_lmfit import unfold_lmfit as unfold_lmfit_impl
from .unfold_mlem_odl import unfold_mlem_odl as unfold_mlem_odl_impl
from .unfold_combined import unfold_combined as unfold_combined_impl

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
        return validate_readings(readings, self.detector_names)

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

        # Vectorized computation of logarithmic steps
        log_steps[0] = log_energies[1] - log_energies[0]
        log_steps[-1] = log_energies[-1] - log_energies[-2]
        # Central differences for interior points: (E[i+1] - E[i-1]) / 2
        log_steps[1:-1] = (log_energies[2:] - log_energies[:-2]) / 2

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
        self,
        readings: Dict[str, float],
        noise_level: float = 0.01,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Add Gaussian noise to readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Original readings.
        noise_level : float, optional
            Relative noise level (default: 0.01).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, float]
            Noisy readings.
        """
        rng = np.random.default_rng(random_state)
        return {
            key: value * (1 + rng.normal(loc=0, scale=noise_level))
            for key, value in readings.items()
        }

    # Public methods delegated to unfolding modules
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
        random_state: Optional[int] = None,
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
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_cvxpy_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            solver=solver,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            regularization_method=regularization_method,
            noise_var=noise_var,
            random_state=random_state,
        )

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using Landweber iteration method.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        max_iterations : int, optional
            Maximum iterations (default: 1000).
        tolerance : float, optional
            Convergence tolerance (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: True).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_landweber_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using MLEM algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        max_iterations : int, optional
            Maximum iterations (default: 1000).
        tolerance : float, optional
            Convergence tolerance (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: True).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_mlem_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

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
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using qpsolvers with regularization selection.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess.
        regularization : float, optional
            Regularization parameter, default: 1e-4.
        norm : int, optional
            Norm type (1 for L1, 2 for L2), default: 2.
        solver : str, optional
            QP solver name, default: 'osqp'.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: True.
        regularization_method : str, optional
            Method for selecting regularization parameter.
            Options: 'manual', 'cosine', 'gcv', 'lcurve', 'dp'.
        noise_var : float, optional
            Noise variance for discrepancy principle ('dp' method).
        smoothness_order : int, optional
            Smoothness constraint order (0, 1, or 2), default: 0.
        smoothness_weight : float, optional
            Weight for smoothness term, default: 1.0.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_qpsolvers_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            solver=solver,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            regularization_method=regularization_method,
            noise_var=noise_var,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            random_state=random_state,
        )

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using lmfit with L1/L2/Elastic regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings (counts or dose rates)
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        method : str, optional
            lmfit solver name (leastsq, lbfgsb, etc.), default: "lbfgsb".
        model_name : str, optional
            Regularization model: elastic, lasso, ridge, default: "elastic".
        regularization : float, optional
            L1 regularization strength, default: 1e-4.
        regularization2 : float, optional
            L2 regularization strength for elastic net, default: 1e-4.
        l1_weight : float, optional
            L1 weight for elastic net (0=pure L2, 1=pure L1), default: 0.5.
        calculate_errors : bool, optional
            Flag to calculate uncertainty via Monte-Carlo, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo uncertainty calculation, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples for error estimation, default: 100.
        save_result : bool, optional
            If True, save result to internal history, default: True.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing unfolding results.
        """
        return unfold_lmfit_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            method=method,
            model_name=model_name,
            regularization=regularization,
            regularization2=regularization2,
            l1_weight=l1_weight,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using MLEM with ODL (Operator Discretization Library).

        Requires the 'odl' package to be installed.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum approximation.
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
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict
            Dictionary containing the spectrum restoration results.
        """
        return unfold_mlem_odl_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            tolerance=tolerance,
            max_iterations=max_iterations,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_combined(
        self,
        readings: Dict[str, float],
        pipeline: List[Dict[str, Any]],
        calculate_errors: bool = False,
        verbose: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Combined unfolding method applying multiple methods sequentially.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings
        pipeline : List[Dict[str, Any]]
            List of methods for sequential application.
        calculate_errors : bool, optional
            Flag to calculate errors for the last method.
        verbose : bool, optional
            Flag to print debug information.

        Returns
        -------
        Dict
            Dictionary with unfolding results.
        """
        return unfold_combined_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            pipeline=pipeline,
            calculate_errors=calculate_errors,
            verbose=verbose,
        )

    # Utility methods
    def discretize_spectra(
        self, spectra: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """Interpolate spectra onto target energy grid."""
        return discretize_spectra(spectra, self.E_MeV)

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

    @staticmethod
    def _import_optional(module_name: str, purpose: str) -> Any:
        """Import optional dependency with informative error message."""
        try:
            return __import__(module_name)
        except ImportError as e:
            raise ImportError(
                f"{module_name} is required for {purpose}. "
                f"Install with: pip install {module_name}"
            ) from e

    def _save_figure(
        self,
        fig: "Any",
        save_to: Optional[str] = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **savefig_kwargs,
    ) -> None:
        """Save figure to file with support for multiple formats."""
        if save_to is None:
            return
        allowed_extensions = (".png", ".jpg", ".jpeg", ".eps", ".pdf")
        if not any(
            save_to.lower().endswith(ext) for ext in allowed_extensions
        ):
            raise ValueError(
                f"Unsupported file extension. Allowed: {allowed_extensions}"
            )
        fig.savefig(
            save_to,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )
        logger.info(f"Figure saved to: {save_to}")

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the Doroshenko coordinate update method.

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
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing unfolding results.
        """
        return unfold_doroshenko_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

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
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the Kaczmarz algorithm (ART).

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
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing unfolding results.
        """
        return unfold_kaczmarz_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self.cc_icrp116,
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            omega=omega,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def plot_response_functions(
        self,
        save_to: Optional[str] = None,
        show: bool = True,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **savefig_kwargs,
    ) -> None:
        """Plot all detector response functions."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for name in self.detector_names:
            ax.plot(self.E_MeV, self.sensitivities[name], label=name)

        ax.set_xlabel("Energy, MeV")
        ax.set_ylabel("Response, cm²")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Response functions of the detector")

        self._save_figure(fig, save_to, dpi, bbox_inches, **savefig_kwargs)

        if show:
            plt.show()
        plt.close(fig)

    def plot_with_uncertainty(
        self,
        result: Dict[str, Any],
        reference_spectrum: Optional[Dict[str, np.ndarray]] = None,
        save_to: Optional[str] = None,
        show: bool = True,
        **plot_kwargs,
    ) -> Tuple["Any", "Any"]:
        """Plot unfolded spectrum with uncertainty range.

        Parameters
        ----------
        result : Dict[str, Any]
            Unfolding result dictionary containing 'energy', 'spectrum',
            and optionally 'spectrum_uncert_min', 'spectrum_uncert_max',
            'spectrum_uncert_std'.
        reference_spectrum : Dict[str, np.ndarray], optional
            Reference spectrum with 'E_MeV' and 'Phi' keys.
        save_to : str, optional
            Path to save figure.
        show : bool, optional
            Call plt.show() (default: True).
        **plot_kwargs : dict
            Additional keyword arguments for plotting.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        E_MeV = result.get("energy", self.E_MeV)
        spectrum = result.get("spectrum", np.zeros_like(E_MeV))
        uncert_min = result.get("spectrum_uncert_min")
        uncert_max = result.get("spectrum_uncert_max")
        uncert_std = result.get("spectrum_uncert_std")

        return plot_with_uncertainty(
            E_MeV=E_MeV,
            spectrum=spectrum,
            uncert_min=uncert_min,
            uncert_max=uncert_max,
            uncert_std=uncert_std,
            reference_spectrum=reference_spectrum,
            save_to=save_to,
            show=show,
            **plot_kwargs,
        )

    def compare_regularization_methods(
        self,
        readings: Dict[str, float],
        noise_var: Optional[float] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare regularization selection methods for given readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        noise_var : float, optional
            Noise variance for discrepancy principle.
        plot : bool, optional
            If True, generate comparison plot.
        plot_path : str, optional
            Path to save the plot.

        Returns
        -------
        Dict[str, Any]
            Comparison results.
        """
        readings = self._validate_readings(readings)
        A, b, _ = self._build_system(readings)
        return compare_reg_util(
            A, b, noise_var=noise_var, plot=plot, plot_path=plot_path
        )

    def randomization_experiment(
        self,
        readings: Dict[str, float],
        noise_var: Optional[float] = None,
        n_samples: int = 10,
        rseed: int = 0,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run randomization experiments for given readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        noise_var : float, optional
            Noise variance for generating perturbed measurements.
        n_samples : int, optional
            Number of random samples for each method, default 10.
        rseed : int, optional
            Random seed for reproducibility, default 0.
        methods : list of str, optional
            List of methods to run: 'lcurve', 'dp', 'gcv', 'lcurve_full'.

        Returns
        -------
        Dict[str, Any]
            Randomization experiment results.
        """
        readings = self._validate_readings(readings)
        A, b, _ = self._build_system(readings)
        return rand_exp_util(
            A, b,
            noise_var=noise_var,
            n_samples=n_samples,
            rseed=rseed,
            methods=methods,
        )
