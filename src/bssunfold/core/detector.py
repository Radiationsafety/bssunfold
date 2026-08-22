"""Detector class for neutron spectrum unfolding.

This module contains the main Detector class which provides methods for
neutron spectrum unfolding using various algorithms.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any, Union, Callable

from ..constants import RF_GSF
from ..logging_config import get_logger
from ..utils.validators import validate_readings
from ..utils.interpolation import discretize_spectra
from ..utils.plotting import plot_with_uncertainty
from .dose_calculation import (
    calculate_dose_rates,
    get_coefficients,
    interpolate_coefficients,
)
from .regularization import (
    compare_regularization_methods as compare_reg_util,
    randomization_experiment as rand_exp_util,
)
from .unfold_cvxpy import unfold_cvxpy as unfold_cvxpy_impl
from .unfold_landweber import unfold_landweber as unfold_landweber_impl
from .unfold_mlem import unfold_mlem as unfold_mlem_impl
from .unfold_qpsolvers import unfold_qpsolvers as unfold_qpsolvers_impl
from .unfold_mystic import unfold_mystic as unfold_mystic_impl
from .unfold_genetic import unfold_genetic as unfold_genetic_impl
from .unfold_reconst import unfold_reconst as unfold_reconst_impl
from .unfold_doroshenko import unfold_doroshenko as unfold_doroshenko_impl
from .unfold_kaczmarz import unfold_kaczmarz as unfold_kaczmarz_impl
from .unfold_lmfit import unfold_lmfit as unfold_lmfit_impl
from .unfold_mlem_odl import unfold_mlem_odl as unfold_mlem_odl_impl
from .unfold_mlem_stop import unfold_mlem_stop as unfold_mlem_stop_impl
from .unfold_imaxed import unfold_imaxed as unfold_imaxed_impl
from .unfold_amaxed import unfold_amaxed as unfold_amaxed_impl
from .unfold_amaxed_regularization import (
    unfold_amaxed_regularization as unfold_amaxed_regularization_impl,
)
from .unfold_odl_advanced import (
    unfold_odl_pdhg as unfold_odl_pdhg_impl,
    unfold_odl_douglas_rachford as unfold_odl_douglas_rachford_impl,
)
from .unfold_qubo import unfold_qubo as unfold_qubo_impl
from .unfold_zfit import unfold_zfit as unfold_zfit_impl
from .unfold_combined import unfold_combined as unfold_combined_impl
from .unfold_gravel import unfold_gravel as unfold_gravel_impl
from .unfold_maxed import unfold_maxed as unfold_maxed_impl
from .unfold_tikhonov_legendre import (
    unfold_tikhonov_legendre as unfold_tikhonov_legendre_impl,
)
from .unfold_bayes import unfold_bayes as unfold_bayes_impl
from .unfold_bayes_spline_regularization import (
    unfold_bayes_spline_regularization as unfold_bayes_spline_impl,
)
from .unfold_statreg import unfold_statreg as unfold_statreg_impl
from .unfold_scipy_direct_method import (
    unfold_scipy_direct_method as unfold_scipy_direct_impl,
)
from .unfold_tsvd import unfold_tsvd as unfold_tsvd_impl
from .unfold_lanczos import unfold_lanczos as unfold_lanczos_impl
from .unfold_fruit_like import unfold_fruit_like as unfold_fruit_like_impl
from .unfold_hybrid_parametric import (
    unfold_hybrid_parametric as unfold_hybrid_parametric_impl,
)
from .unfold_bayesian_parametric import (
    unfold_bayesian_parametric as unfold_bayesian_parametric_impl,
)
from .unfold_parametric import unfold_parametric as unfold_parametric_impl
from .unfold_parametric2 import unfold_parametric2 as unfold_parametric2_impl
from .unfold_smt import unfold_smt as unfold_smt_impl
from .unfold_scip import unfold_scip as unfold_scip_impl
from .unfold_docplex import unfold_docplex as unfold_docplex_impl
from .unfold_cs import unfold_cs as unfold_cs_impl
from .unfold_epic import unfold_epic as unfold_epic_impl
from .unfold_cgls import unfold_cgls as unfold_cgls_impl
from .unfold_gks import unfold_gks as unfold_gks_impl
from .unfold_tikhonov_tv import unfold_tikhonov_tv as unfold_tikhonov_tv_impl
from .unfold_sandii import unfold_sandii as unfold_sandii_impl
from .unfold_bunki import unfold_bunki as unfold_bunki_impl
from .unfold_bunkiut import unfold_bunkiut as unfold_bunkiut_impl
from .unfold_ferdor import unfold_ferdor as unfold_ferdor_impl
from .unfold_rebunki import unfold_rebunki as unfold_rebunki_impl
from .unfold_nsduaz import unfold_nsduaz as unfold_nsduaz_impl
from .unfold_osem import unfold_osem as unfold_osem_impl
from .unfold_mapem import unfold_mapem as unfold_mapem_impl
from .unfold_bsrem import unfold_bsrem as unfold_bsrem_impl
from .unfold_sart import unfold_sart as unfold_sart_impl
from ._base_unfolder import _build_system
from .unfold_interpret import (
    interpret_qp as interpret_qp_impl,
    unfold_interpret as unfold_interpret_impl,
)
from .unfold_fista import unfold_fista as unfold_fista_impl
from .unfold_hybrid_gmres import unfold_hybrid_gmres as unfold_hybrid_gmres_impl
from .unfold_mcmc import unfold_mcmc as unfold_mcmc_impl
from .unfold_maeo import unfold_maeo as unfold_maeo_impl

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
        Raw (non-interpolated) conversion coefficients for dose calculation
    cc_type : str
        Name of the dose conversion coefficient dataset (default: "ICRP116")
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
        cc_type: str = "ICRP116",
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
        cc_type : str, optional
            Name of the dose conversion coefficient dataset to use.
            Options: "ICRP116", "ICRP74_effective", "NRB99_2009_effective",
            "ICRP74_operational". Default: "ICRP116".

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
        self.cc_type = cc_type
        self.cc_icrp116 = get_coefficients(cc_type)

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
            if isinstance(sensitivities, np.ndarray):
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
            raise TypeError("sensitivities must be dict or np.ndarray")

        # Case 4: No arguments, use default
        if (
            response_functions is None
            and E_MeV is None
            and sensitivities is None
        ):
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

    def set_dose_coefficients(self, name: str) -> None:
        """Change the dose conversion coefficient dataset.

        Parameters
        ----------
        name : str
            Name of the coefficient dataset. Options:

            - ``"ICRP116"``: ICRP-116 effective dose (default)
            - ``"ICRP74_effective"``: ICRP-74 effective dose
            - ``"NRB99_2009_effective"``: NRB99-2009 effective dose
            - ``"ICRP74_operational"``: ICRP-74 operational quantities

        Raises
        ------
        ValueError
            If the coefficient name is not found.

        Examples
        --------
        >>> detector = Detector()
        >>> detector.set_dose_coefficients("ICRP74_effective")
        >>> detector.cc_type
        'ICRP74_effective'
        """
        self.cc_icrp116 = get_coefficients(name)
        self.cc_type = name

    def _get_interpolated_cc(self) -> Dict[str, np.ndarray]:
        """Get conversion coefficients interpolated to this detector's energy grid.

        Returns
        -------
        Dict[str, np.ndarray]
            Interpolated conversion coefficients on self.E_MeV.
        """
        return interpolate_coefficients(self.cc_icrp116, self.E_MeV)

    def _validate_readings(
        self, readings: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate detector readings."""
        return validate_readings(readings, self.detector_names)

    def _build_system(
        self, readings: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build response matrix A and measurement vector b."""
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
            "doserates": calculate_dose_rates(
                spectrum_nonneg, self._get_interpolated_cc()
            ),
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

    def get_result(self, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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

    def unfold_mystic(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "fmin_powell",
        maxiter: Optional[int] = 2000,
        maxfun: Optional[int] = 20000,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using mystic with regularization selection.

        Solves ``min ||A x - b||^2 + alpha * ||x||_norm`` subject to
        ``x >= 0`` with the constrained-optimization framework `mystic`.

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
            Mystic solver name: 'fmin', 'fmin_powell', 'diffev' or
            'diffev2', default: 'fmin_powell'.
        maxiter : int, optional
            Maximum number of solver iterations, default: 2000.
        maxfun : int, optional
            Maximum number of function evaluations, default: 20000.
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
        return unfold_mystic_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            solver=solver,
            maxiter=maxiter,
            maxfun=maxfun,
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

    def unfold_genetic(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        solver: str = "pso",
        epoch: int = 500,
        pop_size: int = 50,
        regularization: float = 1e-2,
        norm: int = 2,
        smoothness_order: int = 2,
        smoothness_weight: float = 1.0,
        entropy_weight: float = 0.0,
        n_runs: int = 1,
        early_stop: Optional[int] = None,
        half_range: float = 2.0,
        two_step: bool = False,
        n_coarse: Optional[int] = None,
        smoother: str = "none",
        sigma_smooth: float = 2.0,
        crossover: str = "single",
        mutation: str = "random",
        pareto_select: str = "knee",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Unfold using a meta-heuristic (evolutionary) algorithm.

        The optimizer searches in log space seeded with a Landweber
        warm-start solution (or the provided ``initial_spectrum``), bounded
        to ``log(seed) +/- half_range`` decades, with a scale-consistent
        objective. Inspired by the genetic / PSO unfolding works of
        Shahabinejad & Sohrabpour (2017), Suman & Sarkar (2012), Woo et al.
        (2019) and Mukherjee (2004).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess. If None, a Landweber warm-start solution
            is used to seed the population.
        solver : str, optional
            Meta-heuristic algorithm: 'pso', 'ga', 'de', 'es', 'ep', 'abc',
            'gwo', 'cmaes' or 'nsga2', default: 'pso'.
        epoch : int, optional
            Maximum number of generations, default: 500.
        pop_size : int, optional
            Population size, default: 50.
        regularization : float, optional
            Tikhonov regularization weight, default: 1e-2.
        norm : int, optional
            Norm for the regularization term (1 or 2), default: 2.
        smoothness_order : int, optional
            Smoothness constraint order (0, 1, or 2), default: 2.
        smoothness_weight : float, optional
            Weight for the smoothness term, default: 1.0.
        entropy_weight : float, optional
            Weight of the negative Shannon-entropy objective (0 disables it).
        n_runs : int, optional
            Number of independent runs whose results are averaged,
            default: 1. Not used by the 'nsga2' solver.
        early_stop : int, optional
            Stop if the global best does not improve for this many
            consecutive epochs.
        half_range : float, optional
            Half-width of the log-space search bounds in decades around the
            seed, default: 2.0.
        two_step : bool, optional
            Run the two-step genetic scheme (TGASU-style): a coarse first
            step seeds the full-resolution population, default: False.
        n_coarse : int, optional
            Number of coarse bins for ``two_step`` mode (default: None, i.e.
            ``max(8, n // 4)``).
        smoother : str, optional
            Post-processing smoother: 'none', 'gaussian', 'mbc',
            'gaussian_mbc' or 'second_difference', default: 'none'.
        sigma_smooth : float, optional
            Gaussian filter sigma for the smoothers, default: 2.0.
        crossover : str, optional
            GA crossover operator: 'single' or 'arithmetic' (TGASU); used by
            the numpy GA engine, default: 'single'.
        mutation : str, optional
            GA mutation operator: 'random' or 'iterative' (TGASU, decreasing
            step); used by the numpy GA engine, default: 'random'.
        pareto_select : str, optional
            Selection from the Pareto front for the 'nsga2' solver: 'knee',
            'min_residual' or 'max_entropy', default: 'knee'.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: False.
        random_state : int, optional
            Random seed for reproducibility.
        verbose : bool, optional
            If True, print the MEALPY optimization progress.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_genetic_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            solver=solver,
            epoch=epoch,
            pop_size=pop_size,
            regularization=regularization,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            entropy_weight=entropy_weight,
            n_runs=n_runs,
            early_stop=early_stop,
            half_range=half_range,
            two_step=two_step,
            n_coarse=n_coarse,
            smoother=smoother,
            sigma_smooth=sigma_smooth,
            crossover=crossover,
            mutation=mutation,
            pareto_select=pareto_select,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
            verbose=verbose,
        )

    def unfold_smt(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        nonneg: bool = True,
        timeout_ms: int = 10000,
        objective: str = "l2",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold a neutron spectrum using an SMT solver.

        Minimizes ``||A x - b||_2`` and then the total fluence ``sum(x)``
        over the non-negative orthant using the Z3 optimizer
        (z3-solver package, optional dependency). Falls back to the L1
        residual on non-converging solves.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess (accepted for API compatibility).
        nonneg : bool, optional
            Constrain the spectrum to be non-negative, default: True.
        timeout_ms : int, optional
            SMT solver timeout in milliseconds, default: 10000.
        objective : str, optional
            Residual objective: ``'l2'`` (default) or ``'l1'``.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: True.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_smt_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            nonneg=nonneg,
            timeout_ms=timeout_ms,
            objective=objective,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_scip(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        timeout: float = 10.0,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        nonneg: bool = True,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold a neutron spectrum using the SCIP optimizer.

        Minimizes the Tikhonov-regularized least-squares objective
        ``0.5 * ||A x - b||^2 + penalty(x)`` with the SCIP Optimization
        Suite (pyscipopt package, optional dependency).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess, used as a warm start.
        regularization : float, optional
            Regularization parameter, default: 1e-4.
        norm : int, optional
            Norm type (1 for L1, 2 for L2), default: 2.
        timeout : float, optional
            Time limit in seconds, default: 10.0.
        smoothness_order : int, optional
            Smoothness constraint order (0, 1, or 2), default: 0.
        smoothness_weight : float, optional
            Weight for the smoothness term, default: 1.0.
        nonneg : bool, optional
            Constrain the spectrum to be non-negative, default: True.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: True.
        regularization_method : str, optional
            Method for selecting the regularization parameter
            ('manual', 'cosine', 'lcurve', 'gcv', 'dp'), default: 'manual'.
        noise_var : float, optional
            Noise variance for discrepancy principle ('dp' method).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_scip_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            timeout=timeout,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            nonneg=nonneg,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            regularization_method=regularization_method,
            noise_var=noise_var,
            random_state=random_state,
        )

    def unfold_docplex(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        timeout: float = 10.0,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        nonneg: bool = True,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold a neutron spectrum using CPLEX (docplex).

        Minimizes the Tikhonov-regularized least-squares objective
        ``0.5 * ||A x - b||^2 + penalty(x)`` with IBM Decision Optimization
        CPLEX Modeling for Python (docplex + cplex packages, optional
        dependencies).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess (accepted for API compatibility).
        regularization : float, optional
            Regularization parameter, default: 1e-4.
        norm : int, optional
            Norm type (1 for L1, 2 for L2), default: 2.
        timeout : float, optional
            Time limit in seconds, default: 10.0.
        smoothness_order : int, optional
            Smoothness constraint order (0, 1, or 2), default: 0.
        smoothness_weight : float, optional
            Weight for the smoothness term, default: 1.0.
        nonneg : bool, optional
            Constrain the spectrum to be non-negative, default: True.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: True.
        regularization_method : str, optional
            Method for selecting the regularization parameter
            ('manual', 'cosine', 'lcurve', 'gcv', 'dp'), default: 'manual'.
        noise_var : float, optional
            Noise variance for discrepancy principle ('dp' method).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_docplex_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            timeout=timeout,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            nonneg=nonneg,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            regularization_method=regularization_method,
            noise_var=noise_var,
            random_state=random_state,
        )

    def unfold_cs(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        n_atoms: Optional[int] = None,
        sparsity: Optional[int] = None,
        dictionary: Optional[np.ndarray] = None,
        n_dictionary_iterations: int = 20,
        sigma_min: float = 0.01,
        sigma_decrease_factor: float = 0.5,
        mu_0: float = 1.0,
        L: int = 3,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Compressive Sensing (CS).

        The spectrum is represented sparsely in a learned dictionary (K-SVD),
        sparse coding is performed with OMP, and reconstruction is done with
        the SL0 algorithm. This method is well suited for the highly
        underdetermined problem where the number of energy groups greatly
        exceeds the number of detector readings.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        n_atoms : int, optional
            Number of dictionary atoms.
        sparsity : int, optional
            Target sparsity for dictionary learning.
        dictionary : np.ndarray, optional
            Pre-learned dictionary (n x n_atoms).
        n_dictionary_iterations : int, optional
            Number of K-SVD iterations (default: 20).
        sigma_min : float, optional
            SL0 minimum sigma (default: 0.01).
        sigma_decrease_factor : float, optional
            SL0 sigma decrease factor (default: 0.5).
        mu_0 : float, optional
            SL0 step-size factor (default: 1.0).
        L : int, optional
            SL0 inner iterations per sigma (default: 3).
        max_iterations : int, optional
            SL0 maximum outer iterations (default: 1000).
        tolerance : float, optional
            Convergence tolerance (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_cs_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            n_atoms=n_atoms,
            sparsity=sparsity,
            dictionary=dictionary,
            n_dictionary_iterations=n_dictionary_iterations,
            sigma_min=sigma_min,
            sigma_decrease_factor=sigma_decrease_factor,
            mu_0=mu_0,
            L=L,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_reconst(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        pp: float = 1e-3,
        alpha: float = -1.0,
        beta: float = 0.0,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Turchin's statistical regularization.

        Pure numpy port of the RECONST.FOR algorithm (STREG1).
        Solves  (B * beta + Omega * alpha) * f = A_vec * beta
        with automatic alpha/beta selection.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Ignored (for API compatibility).
        pp : float, optional
            PP parameter for the smoothing matrix (default: 1e-3).
        alpha : float, optional
            Regularization parameter. >0 fixed, <0 auto-select absolute value
            (default: -1).
        beta : float, optional
            Data fidelity weight. >0 fixed, <=0 auto-select (default: 0).
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
        return unfold_reconst_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            pp=pp,
            alpha=alpha,
            beta=beta,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
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
        regularization_method: str = "manual",
        lambda_range: Tuple[float, float] = (1e-6, 1e-1),
        n_lambda: int = 30,
        verbose: bool = True,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
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
        regularization_method : str, optional
            How to choose the regularization parameter. Options: 'manual'
            (use the supplied ``regularization``/``regularization2``), or an
            information criterion 'aic', 'aicc' or 'bic'. For non-manual
            selection the regularization parameter is swept over
            ``lambda_range`` and the candidate minimizing the chosen
            criterion is used. Default: 'manual'.
        lambda_range : Tuple[float, float], optional
            Log-spaced range of lambda candidates for information-criterion
            selection, default: (1e-6, 1e-1).
        n_lambda : int, optional
            Number of lambda candidates for information-criterion selection,
            default: 30.
        verbose : bool, optional
            Print the regularization selection summary, default: True.
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
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            method=method,
            model_name=model_name,
            regularization=regularization,
            regularization2=regularization2,
            l1_weight=l1_weight,
            regularization_method=regularization_method,
            lambda_range=lambda_range,
            n_lambda=n_lambda,
            verbose=verbose,
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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

    def unfold_imaxed(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma_factor: float = 0.1,
        max_iterations: int = 5000,
        tolerance: float = 1e-8,
        line_search_tol: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using the IMAXED algorithm (Wong 2024)."""
        return unfold_imaxed_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma_factor=sigma_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_amaxed(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma_factor: float = 0.1,
        target_chi2: Optional[float] = None,
        max_iterations: int = 5000,
        tolerance: float = 1e-8,
        line_search_tol: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using the AMAXED algorithm (Wong 2024)."""
        return unfold_amaxed_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma_factor=sigma_factor,
            target_chi2=target_chi2,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_amaxed_regularization(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma_factor: float = 0.1,
        tau: float = 1.0,
        max_iterations: int = 5000,
        tolerance: float = 1e-8,
        line_search_tol: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using the AMAXED-Regularization algorithm (Wong 2024)."""
        return unfold_amaxed_regularization_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma_factor=sigma_factor,
            tau=tau,
            max_iterations=max_iterations,
            tolerance=tolerance,
            line_search_tol=line_search_tol,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_odl_pdhg(
        self,
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
        """Unfold using the Primal-Dual Hybrid Gradient (PDHG) algorithm."""
        return unfold_odl_pdhg_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tau=tau,
            sigma=sigma,
            use_tv=use_tv,
            tv_weight=tv_weight,
            nonnegativity=nonnegativity,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_odl_douglas_rachford(
        self,
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
        """Unfold using Douglas-Rachford splitting."""
        return unfold_odl_douglas_rachford_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            use_tv=use_tv,
            tv_weight=tv_weight,
            nonnegativity=nonnegativity,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_qubo(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        n_bits: int = 6,
        max_value: Optional[float] = None,
        regularization: float = 0.01,
        max_iterations: int = 1000,
        annealing_time: int = 1000,
        num_reads: int = 10,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 50,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using a QUBO formulation with quantum-inspired annealing."""
        return unfold_qubo_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            n_bits=n_bits,
            max_value=max_value,
            regularization=regularization,
            max_iterations=max_iterations,
            annealing_time=annealing_time,
            num_reads=num_reads,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_zfit(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        use_mcmc: bool = False,
        n_samples: int = 1000,
        regularization: float = 0.1,
        smoothness_weight: float = 0.01,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using zfit Bayesian inference."""
        return unfold_zfit_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            use_mcmc=use_mcmc,
            n_samples=n_samples,
            regularization=regularization,
            smoothness_weight=smoothness_weight,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_mlem_stop(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 15000,
        cps_crossover: float = 30000.0,
        j_threshold: Optional[float] = None,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold using MLEM-STOP with J-factor early stopping criterion.

        Uses the modified MLEM-STOP method from Montgomery et al. (2020).
        The J-factor indicator (Bouallegue et al. 2013) is computed at each
        iteration: J = sum((meas - est)^2) / sum(est). The algorithm stops
        when J falls below the threshold (mean(measurements) / cps_crossover).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        max_iterations : int, optional
            Maximum iterations (default: 15000).
        cps_crossover : float, optional
            Crossover CPS value for automatic J threshold (default: 30000).
        j_threshold : float, optional
            Explicit J threshold. If None, computed from cps_crossover.
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
        return unfold_mlem_stop_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            cps_crossover=cps_crossover,
            j_threshold=j_threshold,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_epic(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        target_sigmas: Optional[np.ndarray] = None,
        sigma_frac: float = 0.1,
        regularization_order: int = 1,
        non_neg: bool = True,
        noise_var: Optional[float] = None,
        homogeneous_step: bool = True,
        regularize: Optional[Dict[str, Any]] = None,
        beta_shift_k: float = 0,
        beta_distance: float = 2,
        EPIC_bool: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        LSQpar: Optional[Dict[str, Any]] = None,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold a neutron spectrum using EPIC Tikhonov regularization.

        Selects the prior variances of the regularization operator such that the
        a posteriori variances of the model parameters match the target sigmas
        (Equal Posterior Information Condition), then solves the weighted least
        squares problem. Port of the EPIC_LS method of Ortega-Culaciati et al.
        (2021), https://github.com/frortega/EPIC_LS.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess (unused by this method).
        target_sigmas : np.ndarray, optional
            Target a posteriori standard deviations of the model parameters. If
            None, derived as ``sigma_frac`` times the magnitude of the naive
            least-squares solution.
        sigma_frac : float, optional
            Fraction used to derive the default target sigmas, default: 0.1.
        regularization_order : int, optional
            Regularization operator order: 0 (identity), 1 (first derivative,
            default) or 2 (second derivative).
        non_neg : bool, optional
            Constrain the spectrum to be non-negative, default: True.
        noise_var : float, optional
            Variance of the i.i.d. misfit errors used to build Cx, default:
            None (identity Cx).
        homogeneous_step : bool, optional
            Run a preliminary homogeneous Ch search, default: True.
        regularize : dict, optional
            If given (can be empty), damp the EPIC weights towards a
            minimum-norm solution.
        beta_shift_k : float, optional
            Center shift for the beta bounds, default: 0.
        beta_distance : float, optional
            Distance kept from the representability limit, default: 2.
        EPIC_bool : np.ndarray, optional
            Boolean mask of which parameters are subject to the EPIC.
        V : np.ndarray, optional
            Matrix mapping the searched betas to the regularization rows, beta = V @ y (shape (H.shape[0], len(y))).
        LSQpar : dict, optional
            Tuning parameters for the nonlinear least-squares solver.
        calculate_errors : bool, optional
            If True, calculate Monte-Carlo uncertainty, default: False.
        noise_level : float, optional
            Noise level for Monte-Carlo, default: 0.01.
        n_montecarlo : int, optional
            Number of Monte-Carlo samples, default: 100.
        save_result : bool, optional
            Save result to history, default: False.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results including spectrum, residuals, and metadata.
        """
        return unfold_epic_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            target_sigmas=target_sigmas,
            sigma_frac=sigma_frac,
            regularization_order=regularization_order,
            non_neg=non_neg,
            noise_var=noise_var,
            homogeneous_step=homogeneous_step,
            regularize=regularize,
            beta_shift_k=beta_shift_k,
            beta_distance=beta_distance,
            EPIC_bool=EPIC_bool,
            V=V,
            LSQpar=LSQpar,
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
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            pipeline=pipeline,
            calculate_errors=calculate_errors,
            verbose=verbose,
        )

    def unfold_interpret(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        enforce_norm: bool = False,
        norm_value: float = 1.0,
        regularization_method: str = "manual",
        noise_var: Optional[float] = None,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
        tolerance: float = 1e-8,
        interpret_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Unfold a neutron spectrum and interpret the solution with pyoptexplain.

        Solves the same unfolding QP as :meth:`unfold_qpsolvers` through
        pyoptexplain and attaches an interpretation report. The returned dict is
        the standard bssunfold result with two extra keys:

        - ``report``                 -- Markdown interpretation report.
        - ``interpretation_metrics`` -- JSON-friendly metrics dictionary.
        - ``interpretation_spectrum`` -- interpreted (zeroed) spectrum.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : np.ndarray, optional
            Initial spectrum guess (used by the 'cosine' regularization
            method).
        regularization : float, optional
            Regularization parameter (default: 1e-4).
        norm : int, optional
            Penalty norm, 1 or 2 (default: 2).
        smoothness_order : int, optional
            Smoothness derivative order, 0, 1 or 2 (default: 0).
        smoothness_weight : float, optional
            Weight of the smoothness term (default: 1.0).
        enforce_norm : bool, optional
            Add ``sum(x) == norm_value`` (default: False).
        norm_value : float, optional
            Target total fluence (default: 1.0).
        regularization_method : str, optional
            Method for selecting the regularization parameter.
        noise_var : float, optional
            Noise variance for the discrepancy principle ('dp' method).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.
        tolerance : float, optional
            Solver feasibility/optimality tolerance (default: 1e-8). Relax it
            (e.g. 1e-5) if pyoptexplain's backend reports ``iteration_limit``.
        interpret_options : dict, optional
            Extra keyword arguments forwarded to :func:`interpret_qp`.

        Returns
        -------
        Dict[str, Any]
            Standardized unfolding result plus ``report`` and
            ``interpretation_metrics`` keys.
        """
        return unfold_interpret_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization=regularization,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            enforce_norm=enforce_norm,
            norm_value=norm_value,
            regularization_method=regularization_method,
            noise_var=noise_var,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
            tolerance=tolerance,
            interpret_options=interpret_options,
        )

    def interpret_result(
        self,
        readings: Dict[str, float],
        alpha: float = 1e-4,
        norm: int = 2,
        smoothness_order: int = 0,
        smoothness_weight: float = 1.0,
        enforce_norm: bool = False,
        norm_value: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Interpret a set of detector readings without unfolding.

        Builds the response matrix from ``readings`` and runs
        :func:`interpret_qp` directly, returning the report, metrics, tables and
        interpreted spectrum.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        alpha : float, optional
            Regularization parameter (default: 1e-4).
        norm : int, optional
            Penalty norm, 1 or 2 (default: 2).
        smoothness_order : int, optional
            Smoothness derivative order, 0, 1 or 2 (default: 0).
        smoothness_weight : float, optional
            Weight of the smoothness term (default: 1.0).
        enforce_norm : bool, optional
            Add ``sum(x) == norm_value`` (default: False).
        norm_value : float, optional
            Target total fluence (default: 1.0).
        **kwargs
            Extra keyword arguments forwarded to :func:`interpret_qp`.

        Returns
        -------
        Dict[str, Any]
            Dictionary with ``report``, ``metrics``, ``tables`` and
            ``spectrum`` keys.
        """
        A, b, selected = _build_system(
            readings, self.detector_names, self.sensitivities
        )
        result = interpret_qp_impl(
            A,
            b,
            alpha,
            norm=norm,
            smoothness_order=smoothness_order,
            smoothness_weight=smoothness_weight,
            enforce_norm=enforce_norm,
            norm_value=norm_value,
            E_MeV=self.E_MeV,
            detector_names=selected,
            **kwargs,
        )
        return {
            "report": result.report,
            "metrics": result.metrics,
            "tables": result.tables,
            "spectrum": np.asarray(result.spectrum, dtype=float),
        }

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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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
        save_result: bool = False,
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
            cc_icrp116=self._get_interpolated_cc(),
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

    def unfold_gravel(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the GRAVEL algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, default initial spectrum is used.
        tolerance : float, optional
            Convergence tolerance (default: 1e-8).
        max_iterations : int, optional
            Maximum iterations (default: 1000).
        regularization : float, optional
            Regularization parameter (default: 0.0).
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
        return unfold_gravel_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_maxed(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma_factor: float = 0.01,
        max_iterations: int = 5000,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the MAXED algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Reference spectrum. If None, a flat reference is used.
        sigma_factor : float, optional
            Relative measurement uncertainty (default: 0.01).
        max_iterations : int, optional
            Maximum L-BFGS-B iterations (default: 5000).
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
        return unfold_maxed_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma_factor=sigma_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_tikhonov_legendre(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        delta: float = 0.05,
        n_polynomials: int = 15,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Tikhonov regularization with Legendre basis.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Not used (provided for API consistency).
        delta : float, optional
            Regularization parameter (default: 0.05).
        n_polynomials : int, optional
            Number of Legendre polynomials (default: 15).
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
        return unfold_tikhonov_legendre_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            delta=delta,
            n_polynomials=n_polynomials,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_bayes(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Bayesian iterative unfolding (D'Agostini).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior spectrum. If None, uniform prior is used.
        max_iterations : int, optional
            Maximum iterations (default: 4000).
        tolerance : float, optional
            Convergence tolerance (default: 1e-3).
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
        return unfold_bayes_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
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

    def unfold_bayes_spline_regularization(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
        spline_degree: int = 3,
        spline_smooth: float = 1e-2,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Bayesian iterative unfolding with spline regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior spectrum. If None, uniform prior is used.
        max_iterations : int, optional
            Maximum iterations (default: 4000).
        tolerance : float, optional
            Convergence tolerance (default: 1e-3).
        spline_degree : int, optional
            Spline degree (default: 3).
        spline_smooth : float, optional
            Spline smoothing parameter (default: 1e-2).
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
        return unfold_bayes_spline_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            spline_degree=spline_degree,
            spline_smooth=spline_smooth,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_statreg(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        unfoldermethod: str = "EmpiricalBayes",
        regularization: Optional[float] = None,
        basis_name: str = "CubicSplines",
        boundary: Optional[str] = None,
        derivative_degree: int = 2,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Turchin's method of statistical regularization.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        unfoldermethod : str, optional
            Regularization method: 'EmpiricalBayes' or 'User' (default: 'EmpiricalBayes').
        regularization : float, optional
            Regularization parameter for 'User' method.
        basis_name : str, optional
            Basis type (default: 'CubicSplines').
        boundary : str, optional
            Boundary condition, None or 'dirichlet'.
        derivative_degree : int, optional
            Derivative degree (1, 2, 3), default: 2.
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
        return unfold_statreg_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            unfoldermethod=unfoldermethod,
            regularization=regularization,
            basis_name=basis_name,
            boundary=boundary,
            derivative_degree=derivative_degree,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_scipy_direct_method(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 4000,
        method: str = "cg",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using scipy linear solvers.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        tolerance : float, optional
            Solver tolerance (default: 1e-8).
        max_iterations : int, optional
            Maximum solver iterations (default: 4000).
        method : str, optional
            Solver method: 'cg', 'cgs', 'bicgstab', 'gmres', etc. (default: 'cg').
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
        return unfold_scipy_direct_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            tolerance=tolerance,
            max_iterations=max_iterations,
            method=method,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_tsvd(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "discrepancy",
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        noise_level: Optional[float] = None,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Truncated SVD (TSVD).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        method : str, optional
            K-selection method: 'discrepancy', 'l_curve', 'gcv', 'energy',
            'threshold_ratio', 'median_threshold', 'donoho' (default: 'discrepancy').
        k : int, optional
            Fixed number of singular values to keep.
        threshold : float, optional
            Threshold ratio for singular value truncation.
        noise_level : float, optional
            Noise level for discrepancy principle.
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
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
        return unfold_tsvd_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            method=method,
            k=k,
            threshold=threshold,
            noise_level=noise_level,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_lanczos(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization_method: str = "gcv",
        max_iterations: Optional[int] = None,
        regularization: float = 1e-8,
        noise_level: Optional[float] = None,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the Lanczos-hybrid (Krylov) method.

        Performs Golub-Kahan (Lanczos-type) bidiagonalization of the
        response matrix, building a sequence of Krylov subspaces. At each
        iteration a new approximation is obtained by solving the projected
        Tikhonov problem, with the regularization parameter selected
        automatically on the projected problem by Generalized Cross
        Validation (GCV). No a-priori spectrum is required.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (accepted for API compatibility).
        regularization_method : str, optional
            Method for selecting the regularization parameter. Only
            ``'gcv'`` is supported (default: 'gcv').
        max_iterations : int, optional
            Maximum Krylov dimension. Defaults to
            ``min(n_detectors, n_energy_bins)``.
        regularization : float, optional
            Fallback regularization parameter (default: 1e-8).
        noise_level : float, optional
            Relative noise level used for discrepancy-principle early
            stopping.
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_lanczos_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            regularization_method=regularization_method,
            max_iterations=max_iterations,
            regularization=regularization,
            noise_level=noise_level,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_cgls(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-12,
        regularization: float = 0.0,
        smoothness_order: int = 0,
        noise_level: Optional[float] = None,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the CGLS iterative method.

        CGLS (Conjugate Gradient for Least Squares) solves the least
        squares problem ``min ||A x - b||^2`` with a truncated-CG
        iteration, optionally regularized by a ``||L x||^2`` Tikhonov term
        and/or stopped early by the discrepancy principle.  Nonnegativity
        is enforced by clamping at each iteration.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.  If None, a zero vector is used
            (the CGLS iteration does not depend on the initial guess).
        max_iterations : int, optional
            Maximum number of CG iterations (default: 100).
        tolerance : float, optional
            Relative tolerance on the normal-equation residual
            (default: 1e-12).
        regularization : float, optional
            Tikhonov regularization parameter for the ``||L x||^2`` term
            (default: 0.0 = no extra regularization).
        smoothness_order : int, optional
            Order of the derivative matrix used as the regularization
            operator ``L`` (0 = identity, 1 = first derivative,
            2 = second derivative).  Ignored when ``regularization`` is 0.
        noise_level : float, optional
            Relative noise level used for discrepancy-principle stopping.
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_cgls_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
            smoothness_order=smoothness_order,
            noise_level=noise_level,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_gks(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
        smoothness_order: int = 2,
        regularization_method: str = "gcv",
        regularization: float = 1e-8,
        noise_level: Optional[float] = None,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the GKS Krylov-hybrid method.

        GKS (Golub-Kahan hybrid) performs Lanczos-type bidiagonalization
        of the response matrix, building a Krylov subspace of modest
        dimension.  At each iteration the regularized problem is projected
        onto the subspace and solved, with the regularization parameter
        chosen automatically on the projected problem by GCV, the
        discrepancy principle, or the L-curve.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (accepted for API compatibility).
        max_iterations : int, optional
            Maximum Krylov dimension.  Defaults to
            ``min(n_detectors, n_energy_bins)``.
        smoothness_order : int, optional
            Order of the derivative matrix used for regularization
            (default: 2).
        regularization_method : str, optional
            Method for selecting the regularization parameter:
            ``'gcv'``, ``'dp'``, ``'lcurve'`` or ``'manual'``
            (default: 'gcv').
        regularization : float, optional
            Fallback/manual regularization parameter (default: 1e-8).
        noise_level : float, optional
            Relative noise level used by the discrepancy principle when
            ``regularization_method='dp'``.
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_gks_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            smoothness_order=smoothness_order,
            regularization_method=regularization_method,
            regularization=regularization,
            noise_level=noise_level,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_tikhonov_tv(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        mu: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        max_iterations: int = 100,
        type_: str = "TT",
        beta: float = 1.0,
        zthr: float = 2.5,
        tolerance: float = 1e-4,
        noise_level: Optional[float] = None,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum with noise-constrained Tikhonov-TV.

        Solves ``min f(x)`` subject to ``||A x - b||^2 = epsilon`` with the
        ADMM scheme of Gazzola & Gholami adapted to 1D spectra.  The
        regularizer ``f`` is a blend of total variation on the first
        derivative and Tikhonov smoothing on the second derivative, with
        the balancing parameter ``beta`` either fixed or estimated
        adaptively.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (accepted for API compatibility).
        epsilon : float, optional
            Estimate of the squared 2-norm of the noise.  If None, derived
            from ``noise_level`` (``(noise_level * ||b||)^2``) or from the
            residuals of an unregularized least-squares solve.
        mu : Tuple[float, float, float], optional
            Penalty parameters ``(mu1, mu2, mu3)`` (default: (1, 1, 1)).
        max_iterations : int, optional
            Maximum number of ADMM iterations (default: 100).
        type_ : str, optional
            Optimization problem: ``'TT'`` (TV + Tikhonov), ``'TV'`` (pure
            total variation) or ``'T'`` (pure Tikhonov) (default: 'TT').
        beta : float, optional
            Balancing parameter between TV and Tikhonov terms, or
            ``'adapt'`` for adaptive estimation (default: 1.0).
        zthr : float, optional
            Threshold for the adaptive beta estimation (default: 2.5).
        tolerance : float, optional
            Stabilization stopping criterion (default: 1e-4).
        noise_level : float, optional
            Relative noise level used to derive a default ``epsilon``.
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_tikhonov_tv_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            epsilon=epsilon,
            mu=mu,
            max_iterations=max_iterations,
            type_=type_,
            beta=beta,
            zthr=zthr,
            tolerance=tolerance,
            noise_level=noise_level,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_sandii(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        chi_fac: int = 1,
        relative_uncertainty: float = 0.1,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the SAND-II algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        max_iterations : int, optional
            Maximum number of iterations (default: 50).
        tolerance : float, optional
            Maximum relative spectrum change used when ``chi_fac=0``
            (default: 1e-3).
        chi_fac : int, optional
            Convergence criterion: ``1`` = chi-square based, ``0`` = maximum
            relative deviation based (default: 1).
        relative_uncertainty : float, optional
            Relative measurement uncertainty for the chi-square criterion
            (default: 0.1).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_sandii_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            chi_fac=chi_fac,
            relative_uncertainty=relative_uncertainty,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_bunki(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        smoothing: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the BUNKI (SPUNIT) algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        smoothing : float, optional
            Three-point smoothing factor (default: 0.1).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_bunki_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_bunkiut(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        smoothing: float = 0.05,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the BUNKI-UT (BON31G) algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        smoothing : float, optional
            Three-point smoothing factor (default: 0.05).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_bunkiut_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_ferdor(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        smoothing: float = 1e-3,
        chi_squared_target: float = 1.0,
        relative_uncertainty: float = 0.1,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the FERDOR algorithm.

        FERDOR (ORNL; Burrus, ORNL-4154) is a constrained least-squares
        unfolding code with second-difference smoothing. The smoothing weight
        is adjusted iteratively so the reduced chi-square of the fit reaches
        ``chi_squared_target`` (discrepancy principle), and the final
        spectrum is non-negative.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        max_iterations : int, optional
            Maximum number of smoothing-weight iterations (default: 100).
        tolerance : float, optional
            Relative tolerance on the reduced chi-square (default: 1e-3).
        smoothing : float, optional
            Initial smoothing weight alpha (default: 1e-3).
        chi_squared_target : float, optional
            Target reduced chi-square per degree of freedom (default: 1.0).
        relative_uncertainty : float, optional
            Relative measurement uncertainty for the chi-square criterion
            (default: 0.1).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_ferdor_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            smoothing=smoothing,
            chi_squared_target=chi_squared_target,
            relative_uncertainty=relative_uncertainty,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_rebunki(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        smoothing: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 0.01,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the ReBUNKI (SPUNIT) algorithm.

        ReBUNKI (Lacerda et al., 2018) is a modern open reimplementation of
        the BUNKI code; its Python version supports the SPUNIT iterative
        algorithm. The default tolerance matches the ~1% relative-error
        convergence recommended by ReBUNKI.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        smoothing : float, optional
            Three-point smoothing factor (default: 0.1).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 0.01).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_rebunki_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_nsduaz(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        catalogue: Optional[Dict[str, np.ndarray]] = None,
        use_catalogue: bool = True,
        reference_name: Optional[str] = None,
        smoothing: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 0.01,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the NSDUAZ algorithm.

        NSDUAZ (Universidad Autonoma de Zacatecas; Ortiz-Rodriguez &
        Vega-Carrillo, 2012) uses the SPUNIT iterative algorithm with an
        initial spectrum selected from a catalogue of standard spectra by a
        statistical test on count-rate ratios relative to the reference
        (20.32 cm) sphere.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Explicit initial spectrum guess. When given, it overrides the
            catalogue selection.
        catalogue : Optional[Dict[str, np.ndarray]], optional
            User-supplied catalogue of candidate initial spectra (label ->
            spectrum on the detector energy grid). When None, the built-in
            mini-catalogue is used.
        use_catalogue : bool, optional
            If True (default), select the initial spectrum from the catalogue
            when ``initial_spectrum`` is not provided; if False, a flat
            spectrum is used.
        reference_name : str, optional
            Reference sphere name for the catalogue test (default:
            auto-detect 20.32 cm sphere).
        smoothing : float, optional
            Three-point smoothing factor (default: 0.1).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 0.01).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_nsduaz_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            catalogue=catalogue,
            use_catalogue=use_catalogue,
            reference_name=reference_name,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_mcmc(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma_prior: float = 0.05,
        lambda_prior: float = 0.5,
        lengthscale: float = 3.0,
        n_samples: int = 2000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.95,
        use_hierarchical: bool = False,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
        progressbar: bool = False,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Bayesian MCMC with NUTS sampler.

        Full Bayesian unfolding with the No-U-Turn Sampler (NUTS). The method
        returns the mean posterior spectrum together with 95% HPD credible
        intervals, per-bin posterior standard deviations and convergence
        diagnostics (R-hat, effective sample size) under ``result['mcmc_stats']``.

        The spectrum is modelled on the log scale with a smoothness
        (Ornstein-Uhlenbeck) prior anchored on a data-driven center (the
        non-negative least-squares solution, or a user-supplied
        ``initial_spectrum``), which keeps the underdetermined unfolding
        problem well posed for NUTS.

        Requires optional ``pymc`` and ``arviz`` (``pip install bssunfold[mcmc]``
        or ``pip install pymc arviz``).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Prior center guess for the spectrum. When None, the non-negative
            least-squares solution of ``A @ x = b`` is used as the prior center.
        sigma_prior : float, optional
            Relative likelihood noise scale (default: 0.05). With
            ``use_hierarchical=False`` the noise is fixed at
            ``sigma_prior * |b|``; with ``use_hierarchical=True`` it is the
            prior scale of the estimated relative noise.
        lambda_prior : float, optional
            Prior scale of the log-spectrum spatial amplitude (default: 0.5).
        lengthscale : float, optional
            OU smoothness correlation length in energy bins (default: 3.0).
        n_samples : int, optional
            Number of MCMC samples per chain after tuning (default: 2000).
        tune : int, optional
            Number of tuning (warmup) samples per chain (default: 1000).
        chains : int, optional
            Number of independent MCMC chains (default: 2).
        target_accept : float, optional
            Target acceptance rate for NUTS (default: 0.95).
        use_hierarchical : bool, optional
            Estimate the likelihood noise from the data (default: False).
        calculate_errors : bool, optional
            Calculate additional Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for additional Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of additional Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.
        progressbar : bool, optional
            Show sampling progress bar (default: False).

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary. Includes the standard keys
            (``energy``, ``spectrum``, ``effective_readings``, ``residual``,
            ``residual_norm``, ``method``, ``doserates``) plus MCMC-specific
            keys ``spectrum_uncertainty``, ``spectrum_lower``,
            ``spectrum_upper`` and ``mcmc_stats``.

        Raises
        ------
        ImportError
            If PyMC or ArviZ is not installed.
        RuntimeError
            If MCMC sampling fails.
        """
        return unfold_mcmc_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma_prior=sigma_prior,
            lambda_prior=lambda_prior,
            lengthscale=lengthscale,
            n_samples=n_samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            use_hierarchical=use_hierarchical,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
            progressbar=progressbar,
        )

    def unfold_maeo(
        self,
        readings: Dict[str, float],
        n_cycles: int = 20,
        n_gen_per_cycle: int = 10,
        pop_size: int = 100,
        algorithms: Optional[List[str]] = None,
        lambda_smooth: float = 0.01,
        prior_spectrum: Optional[np.ndarray] = None,
        initial_spectrum: Optional[np.ndarray] = None,
        convergence_assist_ratio: float = 0.2,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_result: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using MAEO ensemble optimization.

        This method implements the Multiobjective Animorphic Ensemble Optimization
        (MAEO) framework from Erdem et al. (2026), which combines multiple
        multiobjective optimization algorithms (NSGA-III, CTAEA, AGEMOEA2, SPEA2)
        in an ensemble with adaptive migration based on hypervolume performance.

        The MAEO framework is particularly effective for neutron spectrum unfolding
        because it:
        - Handles multiple conflicting objectives (data fit vs. smoothness)
        - Automatically selects the best-performing algorithm for the problem
        - Provides robust convergence through ensemble diversity
        - Supports parallel evaluation of individuals

        Parameters
        ----------
        readings : dict
            Dictionary mapping detector names to measured count rates.
        n_cycles : int, optional
            Number of MAEO cycles (default: 20). Each cycle runs n_gen_per_cycle
            generations for each island algorithm.
        n_gen_per_cycle : int, optional
            Generations per cycle for each island (default: 10).
        pop_size : int, optional
            Population size per island (default: 100).
        algorithms : list of str, optional
            List of algorithm names to use as islands. Default uses the four
            algorithms from the MAEO paper: ["nsga3", "ctaea", "agemoea2", "spea2"].
            Available options: "nsga3", "ctaea", "agemoea2", "spea2".
        lambda_smooth : float, optional
            Smoothness regularization weight (default: 0.01). Controls the trade-off
            between data fidelity and spectrum smoothness.
        prior_spectrum : np.ndarray, optional
            Prior/guess spectrum for additional objective. If provided, adds a third
            objective to minimize deviation from this prior.
        initial_spectrum : np.ndarray, optional
            Initial spectrum for warm-start. Used to seed the population in log space.
        convergence_assist_ratio : float, optional
            Fraction of cycles to dedicate to the best-performing island at the end
            (default: 0.2). Implements the "convergence assist" mechanism from MAEO.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, optional
            Print progress information including hypervolume history (default: False).
        save_result : bool, optional
            Save result to history (default: False).
        **kwargs
            Additional keyword arguments passed to the underlying optimizer.

        Returns
        -------
        dict
            Standardized result dictionary containing:
            - 'energy': Energy grid in MeV
            - 'spectrum': Unfolded spectrum (non-negative)
            - 'spectrum_absolute': Absolute flux values
            - 'effective_readings': Computed readings from unfolded spectrum
            - 'residual': Difference between measured and computed readings
            - 'residual_norm': L2 norm of residual
            - 'method': 'MAEO'
            - 'doserates': Dose rates calculated from spectrum
            - 'maeo_info': Dictionary with MAEO-specific information:
                - 'n_cycles': Number of cycles executed
                - 'best_algorithm': Name of best-performing algorithm
                - 'hypervolume_history': HV history for each island
                - 'population_history': Population sizes per island per cycle
                - 'algorithms_used': List of algorithms used
            - 'maeo_pareto_front': Final Pareto front objectives (if available)
            - 'maeo_objectives': Objectives for selected solution

        Notes
        -----
        The MAEO framework optimizes multiple objectives simultaneously:
        1. Minimize data fidelity error ||b - A*phi||^2 / ||b||^2
        2. Minimize spectrum roughness ||D2 * phi||^2 (second derivative)
        3. (Optional) Minimize deviation from prior spectrum

        The final solution is selected from the combined Pareto front using a
        knee-point detection method to balance accuracy and smoothness.

        The algorithm runs in two phases:
        1. Migration phase: All islands run in parallel, with individuals migrating
           toward better-performing islands based on hypervolume indicators.
        2. Convergence phase: Only the best-performing island continues, focusing
           computational resources on exploitation.

        References
        ----------
        [1] O.F. Erdem, D. Price, P. Seurin, M.I. Radaideh, "MAEO: Multiobjective
            Animorphic Ensemble Optimization for Scalable Large-scale Engineering
            Applications", arXiv:2604.26973 (2026).

        [2] D. Price, M.I. Radaideh, "Animorphic Ensemble Optimization: a large-scale
            island model", Neural Computing and Applications 35 (4) (2023) 3221-3243.

        Examples
        --------
        >>> from bssunfold import Detector
        >>> detector = Detector()
        >>> readings = {
        ...     'sphere_1': 100.5,
        ...     'sphere_2': 85.3,
        ...     'sphere_3': 72.1,
        ...     'sphere_4': 58.9,
        ...     'sphere_5': 45.2,
        ...     'sphere_6': 32.8,
        ... }
        >>> # Run MAEO with default settings
        >>> result = detector.unfold_maeo(readings, n_cycles=15)
        >>> print(f"Spectrum integral: {np.sum(result['spectrum']):.2f}")
        >>> print(f"Best algorithm: {result['maeo_info']['best_algorithm']}")
        >>>
        >>> # Run with custom algorithms and verbose output
        >>> result = detector.unfold_maeo(
        ...     readings,
        ...     algorithms=["nsga3", "spea2"],
        ...     n_cycles=10,
        ...     verbose=True,
        ... )
        """
        result = unfold_maeo_impl(
            detector=self,
            readings=readings,
            n_cycles=n_cycles,
            n_gen_per_cycle=n_gen_per_cycle,
            pop_size=pop_size,
            algorithms=algorithms,
            lambda_smooth=lambda_smooth,
            prior_spectrum=prior_spectrum,
            initial_spectrum=initial_spectrum,
            convergence_assist_ratio=convergence_assist_ratio,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

        if save_result:
            self._save_result(result)

        return result

    def unfold_osem(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        n_subsets: int = 1,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the OSEM algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        max_iterations : int, optional
            Maximum number of iterations (default: 50).
        n_subsets : int, optional
            Number of ordered subsets over the detector readings
            (default: 1, i.e. standard MLEM).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_osem_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            n_subsets=n_subsets,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_mapem(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        prior: str = "quadratic",
        beta: float = 1e-3,
        prior_delta: float = 1.0,
        gamma: float = 1.0,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using penalised EM (MAP-EM).

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        prior : str, optional
            Prior type: ``'none'``, ``'quadratic'``, ``'logcosh'`` or
            ``'relative_difference'`` (default: ``'quadratic'``).
        beta : float, optional
            Prior weight (default: 1e-3).
        prior_delta : float, optional
            Width parameter of the quadratic/logcosh priors and additive
            floor of the relative-difference prior (default: 1.0).
        gamma : float, optional
            Edge-preservation parameter of the relative-difference prior
            (default: 1.0).
        max_iterations : int, optional
            Maximum number of iterations (default: 50).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_mapem_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            prior=prior,
            beta=beta,
            prior_delta=prior_delta,
            gamma=gamma,
            max_iterations=max_iterations,
            tolerance=tolerance,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_bsrem(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        prior: str = "none",
        beta: float = 1e-3,
        prior_delta: float = 1.0,
        gamma: float = 1.0,
        max_iterations: int = 50,
        n_subsets: int = 1,
        tolerance: float = 1e-6,
        relaxation: Optional[Union[float, Callable[[int], float]]] = None,
        addition_after_iteration: float = 1e-4,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the BSREM algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        prior : str, optional
            Prior type: ``'none'``, ``'quadratic'``, ``'logcosh'`` or
            ``'relative_difference'`` (default: ``'none'``).
        beta : float, optional
            Prior weight (default: 1e-3).
        prior_delta : float, optional
            Width parameter of the quadratic/logcosh priors and additive
            floor of the relative-difference prior (default: 1.0).
        gamma : float, optional
            Edge-preservation parameter of the relative-difference prior
            (default: 1.0).
        max_iterations : int, optional
            Maximum number of iterations (default: 50).
        n_subsets : int, optional
            Number of ordered subsets over the detector readings (default: 1).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        relaxation : float or callable, optional
            Relaxation sequence (default: None -> constant 1).
        addition_after_iteration : float, optional
            Floor value for spectrum bins (default: 1e-4).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_bsrem_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            prior=prior,
            beta=beta,
            prior_delta=prior_delta,
            gamma=gamma,
            max_iterations=max_iterations,
            n_subsets=n_subsets,
            tolerance=tolerance,
            relaxation=relaxation,
            addition_after_iteration=addition_after_iteration,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_sart(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        relaxation: Optional[Union[float, Callable[[int], float]]] = None,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the SART algorithm.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess. If None, a flat spectrum is used.
        max_iterations : int, optional
            Maximum number of iterations (default: 50).
        tolerance : float, optional
            Relative change tolerance for early stopping (default: 1e-6).
        relaxation : float or callable, optional
            Relaxation sequence (default: None -> constant 0.8).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        noise_level : float, optional
            Noise level for Monte-Carlo (default: 0.01).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_sart_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            relaxation=relaxation,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_fista(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 500,
        tolerance: float = 1e-8,
        regularization: float = 0.0,
        l1_penalty: float = 0.0,
        tv_penalty: float = 0.0,
        nonnegativity: bool = True,
        x_min: float = 0.0,
        x_max: float = np.inf,
        noise_level: Optional[float] = None,
        eta: float = 1.01,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using FISTA algorithm.

        The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) is an
        accelerated proximal gradient method that achieves O(1/k^2) convergence
        rate for convex optimization problems. It can handle L1 regularization
        (sparsity), TV regularization, and box constraints.

        Based on IRtools IRfista.m by Silvia Gazzola et al.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial guess for spectrum. If None, a flat spectrum is used.
        max_iterations : int, optional
            Maximum number of iterations (default: 500).
        tolerance : float, optional
            Convergence tolerance (default: 1e-8).
        regularization : float, optional
            Tikhonov regularization parameter (default: 0.0).
        l1_penalty : float, optional
            L1 regularization penalty parameter for sparsity (default: 0.0).
        tv_penalty : float, optional
            Total variation penalty parameter (default: 0.0).
        nonnegativity : bool, optional
            Apply nonnegativity constraints (default: True).
        x_min : float, optional
            Lower bound for solution (default: 0.0).
        x_max : float, optional
            Upper bound for solution (default: inf).
        noise_level : float, optional
            Relative noise level for discrepancy principle stopping.
        eta : float, optional
            Safety factor for discrepancy principle (default: 1.01).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_fista_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
            l1_penalty=l1_penalty,
            tv_penalty=tv_penalty,
            nonnegativity=nonnegativity,
            x_min=x_min,
            x_max=x_max,
            noise_level=noise_level,
            eta=eta,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_hybrid_gmres(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        regularization_method: str = "gcv",
        regularization: float = 0.0,
        noise_level: Optional[float] = None,
        eta: float = 1.01,
        reorthogonalization: bool = True,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Hybrid GMRES method.

        The hybrid GMRES method combines the GMRES iterative solver with
        Tikhonov regularization applied to the projected problem at each
        iteration. The regularization parameter is selected automatically
        using GCV or discrepancy principle.

        Based on IRtools IRhybrid_gmres.m by Silvia Gazzola et al.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial guess for spectrum. If None, zero vector is used.
        max_iterations : int, optional
            Maximum Krylov dimension (default: 100).
        regularization_method : str, optional
            Method for selecting regularization parameter:
            'gcv', 'modgcv', 'discrep' (default: 'gcv').
        regularization : float, optional
            Fixed regularization parameter (used if not auto-selected).
        noise_level : float, optional
            Relative noise level for discrepancy principle.
        eta : float, optional
            Safety factor for discrepancy principle (default: 1.01).
        reorthogonalization : bool, optional
            Apply full reorthogonalization (default: True).
        calculate_errors : bool, optional
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int, optional
            Number of Monte-Carlo samples (default: 100).
        save_result : bool, optional
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_hybrid_gmres_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            max_iterations=max_iterations,
            regularization_method=regularization_method,
            regularization=regularization,
            noise_level=noise_level,
            eta=eta,
            reorthogonalization=reorthogonalization,
            calculate_errors=calculate_errors,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_fruit_like(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        initial_params: Optional[Dict[str, float]] = None,
        method: str = "leastsq",
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using FRUIT-like parametric method.

        Uses a parametric model with Maxwellian thermal component,
        1/E epithermal component, and evaporation spectrum for fast neutrons.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (unused in parametric method).
        initial_params : Optional[Dict[str, float]], optional
            Initial parameter values for the parametric model.
            Keys: A_th, T_th, A_epi, A_f, T_ev.
        method : str, optional
            lmfit solver method (default: "leastsq").
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
        return unfold_fruit_like_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            initial_params=initial_params,
            method=method,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_hybrid_parametric(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        refinement_method: str = "landweber",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        step_size: float = 0.01,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using hybrid parametric-nonparametric method.

        Combines parametric initial guess with iterative refinement using
        Landweber or MLEM iteration.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess.
        refinement_method : str, optional
            Refinement method: "landweber" or "mlem" (default: "landweber").
        max_iterations : int, optional
            Maximum iterations (default: 100).
        tolerance : float, optional
            Convergence tolerance (default: 1e-6).
        step_size : float, optional
            Step size for Landweber (default: 0.01).
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
        return unfold_hybrid_parametric_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            refinement_method=refinement_method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            step_size=step_size,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_bayesian_parametric(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        sigma: float = 0.02,
        n_samples: int = 1000,
        burn_in: int = 200,
        proposal_scale: float = 0.1,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using Bayesian parametric method.

        Uses Bayesian inference with MCMC sampling to estimate spectral
        parameters and quantify uncertainty.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (unused).
        sigma : float, optional
            Measurement uncertainty (default: 0.02).
        n_samples : int, optional
            Number of MCMC samples (default: 1000).
        burn_in : int, optional
            Burn-in samples (default: 200).
        proposal_scale : float, optional
            Proposal scale (default: 0.1).
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
        return unfold_bayesian_parametric_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            sigma=sigma,
            n_samples=n_samples,
            burn_in=burn_in,
            proposal_scale=proposal_scale,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_parametric(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        initial_params: Optional[Dict[str, float]] = None,
        method: str = "leastsq",
        optimizer: str = "lmfit",
        alpha: float = 1e-4,
        alpha_auto: bool = False,
        solver_backend: str = "auto",
        max_iter: int = 50,
        tol: float = 1e-6,
        calculate_errors: bool = False,
        noise_level: float = 0.01,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the FRUIT-based parametric method.

        Uses the three-component parameterization from Bedogni FRUIT /
        Pyshkina B3S: thermal (Maxwellian), epithermal (1/E with
        exponential cutoffs), and fast (power-law x exponential).

        The ``optimizer`` parameter selects the backend:

        * ``"lmfit"``     -- classic lmfit least-squares (default).
        * ``"cvxpy"``     -- sequential QP via cvxpy (SQP).
        * ``"qpsolvers"`` -- sequential QP via qpsolvers (SQP).
        * ``"combined"``  -- lmfit first, then QP refinement.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (unused in parametric method).
        initial_params : Optional[Dict[str, float]], optional
            Initial parameter values for the parametric model.
            Keys: b, beta_prime, alpha, beta, P_th, P_epi.
        method : str, optional
            lmfit solver method (default: "leastsq").
        optimizer : str, optional
            Backend optimizer (default: "lmfit").
        alpha : float, optional
            Regularization weight for QP-based optimizers (default: 1e-4).
        alpha_auto : bool, optional
            If True, select alpha automatically via GCV for the lmfit
            optimizer (default: False).
        solver_backend : str, optional
            QP solver backend: "auto", "cvxpy", "cvxpy:ECOS",
            "qpsolvers", "qpsolvers:osqp", etc. (default: "auto").
        max_iter : int, optional
            Max SQP iterations (default: 50).
        tol : float, optional
            Convergence tolerance for SQP (default: 1e-6).
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
        return unfold_parametric_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            initial_params=initial_params,
            method=method,
            optimizer=optimizer,
            alpha=alpha,
            alpha_auto=alpha_auto,
            solver_backend=solver_backend,
            max_iter=max_iter,
            tol=tol,
            calculate_errors=calculate_errors,
            noise_level=noise_level,
            n_montecarlo=n_montecarlo,
            save_result=save_result,
            random_state=random_state,
        )

    def unfold_parametric2(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        optimizer: str = "grid",
        b_range: Tuple[float, float, int] = (0.5, 2.0, 5),
        Tf_range: Tuple[float, float, int] = (0.5, 10.0, 5),
        c_range: Tuple[float, float, int] = (0.5, 3.0, 4),
        alpha: float = 1e-4,
        solver_backend: str = "auto",
        max_iter_qp: int = 50,
        tol_qp: float = 1e-6,
        noise_level: float = 0.05,
        max_iter: int = 200,
        tol_chi2: float = 1.0,
        calculate_errors: bool = False,
        n_montecarlo: int = 100,
        save_result: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unfold neutron spectrum using the BON95 parametric method.

        Uses the four-component parameterization from Sannikov BON95:
        thermal (Maxwellian), epithermal (1/E), intermediate, and
        fast (evaporation/cascade) components. After parametric fitting,
        the result is refined by directed-divergence iterations.

        The ``optimizer`` parameter selects the parametric fit backend:

        * ``"grid"``      -- grid search + NLS (default, no extra deps).
        * ``"cvxpy"``     -- SQP via cvxpy.
        * ``"qpsolvers"`` -- SQP via qpsolvers.
        * ``"combined"``  -- grid search + SQP refinement.

        Parameters
        ----------
        readings : Dict[str, float]
            Detector readings.
        initial_spectrum : Optional[np.ndarray], optional
            Initial spectrum guess (unused in parametric method).
        optimizer : str
            Parametric fit optimizer (default: "grid").
        b_range : tuple
            Grid range for b: (min, max, n_points). Used by "grid"/"combined".
        Tf_range : tuple
            Grid range for Tf (MeV): (min, max, n_points). Used by "grid"/"combined".
        c_range : tuple
            Grid range for c: (min, max, n_points). Used by "grid"/"combined".
        alpha : float
            Tikhonov regularization for SQP (default: 1e-4).
        solver_backend : str
            QP backend for SQP (default: "auto").
        max_iter_qp : int
            Max SQP iterations (default: 50).
        tol_qp : float
            SQP convergence tolerance (default: 1e-6).
        noise_level : float
            Relative uncertainty for measurements (default: 0.05 = 5%).
        max_iter : int
            Max directed-divergence iterations (default: 200).
        tol_chi2 : float
            Chi-squared convergence threshold (default: 1.0).
        calculate_errors : bool
            Calculate Monte-Carlo errors (default: False).
        n_montecarlo : int
            Number of Monte-Carlo samples (default: 100).
        save_result : bool
            Save result to history (default: False).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Unfolding results dictionary.
        """
        return unfold_parametric2_impl(
            detector_names=self.detector_names,
            n_energy_bins=self.n_energy_bins,
            E_MeV=self.E_MeV,
            sensitivities=self.sensitivities,
            cc_icrp116=self._get_interpolated_cc(),
            save_result_callback=self._save_result,
            readings=readings,
            initial_spectrum=initial_spectrum,
            optimizer=optimizer,
            b_range=b_range,
            Tf_range=Tf_range,
            c_range=c_range,
            alpha=alpha,
            solver_backend=solver_backend,
            max_iter_qp=max_iter_qp,
            tol_qp=tol_qp,
            noise_level=noise_level,
            max_iter=max_iter,
            tol_chi2=tol_chi2,
            calculate_errors=calculate_errors,
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
            A,
            b,
            noise_var=noise_var,
            n_samples=n_samples,
            rseed=rseed,
            methods=methods,
        )

    def compare(
        self,
        *spectra: Any,
        metrics: Optional[Union[str, List[str]]] = None,
        labels: Optional[List[str]] = None,
        readings1: Optional[np.ndarray] = None,
        readings2: Optional[np.ndarray] = None,
        response_matrix: Optional[np.ndarray] = None,
        plot: bool = False,
        save_to: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[int, int] = (14, 5),
        return_fig: bool = False,
        **plot_kwargs,
    ) -> Union[
        Dict[str, float],
        pd.DataFrame,
        Tuple[Union[Dict[str, float], pd.DataFrame], Any, Any],
    ]:
        """Compare two or more spectra using comparison metrics.

        Each spectrum can be provided as:
        - np.ndarray of length matching ``self.n_energy_bins``
        - dict with a ``'spectrum'`` key (e.g. an unfolding result)
        - result dictionary returned by any ``unfold_*`` method

        When the energy grid is available, EURADOS-style metrics (dose
        differences, peak errors, log-lethargy correlation, etc.) are
        computed automatically.

        Parameters
        ----------
        *spectra : np.ndarray or dict
            Two or more spectra to compare.
        metrics : str, list of str, or None
            Metric(s) to compute. If None, all metrics are used.
        labels : list of str, optional
            Labels for each spectrum. Required for 3+ spectra.
        readings1, readings2 : np.ndarray, optional
            Measured readings for response-matrix consistency check.
            If a spectrum is a result dict containing ``'readings'``,
            those values are used as a fallback.
        response_matrix : np.ndarray, optional
            Response matrix for the consistency check.  If a spectrum is
            a result dict containing ``'response_matrix'``, that value is
            used as a fallback.
        plot : bool, optional
            If True, generate a comparison figure with spectra overlay
            and metric bar chart.
        save_to : str, optional
            Path to save the figure (png/jpg/eps/pdf).
        dpi : int, optional
            Figure DPI (default: 300).
        figsize : tuple, optional
            Figure size (default: (14, 5)).
        return_fig : bool, optional
            If True, return (result, fig, ax) tuple.
        **plot_kwargs : dict
            Additional keyword arguments passed to matplotlib/seaborn plots.

        Returns
        -------
        dict or pd.DataFrame or tuple
            If two spectra: dict {metric: value}.
            If three or more: pd.DataFrame with metrics as rows and
            comparison pairs as columns.
            If return_fig=True: (result, fig, ax).
        """
        from ..utils.comparison import compare_spectra

        parsed = []
        extra_readings = [None, None]
        extra_rm = [None, None]
        _meta_keys = {
            "E_MeV",
            "energy",
            "readings",
            "response_matrix",
            "effective_readings",
            "doserates",
            "spectrum_uncert_min",
            "spectrum_uncert_max",
            "spectrum_uncert_std",
            "spectrum_uncert_mean",
        }
        for i, s in enumerate(spectra):
            if isinstance(s, dict):
                if "spectrum" in s:
                    parsed.append(np.asarray(s["spectrum"], dtype=float))
                else:
                    spectrum_key = None
                    if "Phi" in s:
                        spectrum_key = "Phi"
                    else:
                        for key in s:
                            if key not in _meta_keys and isinstance(
                                s[key], (np.ndarray, list, tuple)
                            ):
                                spectrum_key = key
                                break
                    if spectrum_key is not None:
                        parsed.append(np.asarray(s[spectrum_key], dtype=float))
                    else:
                        raise ValueError(
                            f"Spectrum {i} is a dict but has no recognizable "
                            f"spectrum key. Available keys: {list(s.keys())}"
                        )
                if i < 2:
                    if readings1 is None and "readings" in s and i == 0:
                        extra_readings[0] = np.asarray(
                            s["readings"], dtype=float
                        )
                    if readings2 is None and "readings" in s and i == 1:
                        extra_readings[1] = np.asarray(
                            s["readings"], dtype=float
                        )
                    if response_matrix is None and "response_matrix" in s:
                        extra_rm[i] = np.asarray(
                            s["response_matrix"], dtype=float
                        )
            elif isinstance(s, np.ndarray):
                if s.ndim != 1:
                    raise ValueError(
                        f"Spectrum {i} must be 1-D, got shape {s.shape}"
                    )
                parsed.append(s)
            else:
                raise TypeError(
                    f"Spectrum {i} must be ndarray or dict, got {type(s)}"
                )

        if len(parsed) < 2:
            raise ValueError("At least two spectra required for comparison")

        n_bins = self.n_energy_bins
        for i, s in enumerate(parsed):
            if len(s) != n_bins:
                raise ValueError(
                    f"Spectrum {i} has {len(s)} bins, expected {n_bins} "
                    f"(matching detector energy grid)"
                )

        # Default labels
        if labels is None:
            if len(parsed) == 2:
                labels = ["Reference", "Comparison"]
            else:
                labels = [f"Spectrum {i}" for i in range(len(parsed))]
        if len(labels) != len(parsed):
            raise ValueError(
                f"Expected {len(parsed)} labels, got {len(labels)}"
            )

        # Resolve readings / response_matrix for EURADOS metrics
        r1 = readings1 if readings1 is not None else extra_readings[0]
        r2 = readings2 if readings2 is not None else extra_readings[1]
        rm = (
            response_matrix
            if response_matrix is not None
            else (extra_rm[0] if extra_rm[0] is not None else extra_rm[1])
        )
        use_energy = self.E_MeV
        use_cc = self._get_interpolated_cc()

        # Single-pair comparison
        if len(parsed) == 2:
            result = compare_spectra(
                parsed[0],
                parsed[1],
                metrics=metrics,
                energy=use_energy,
                cc_icrp116=use_cc,
                readings1=r1,
                readings2=r2,
                response_matrix=rm,
            )
        else:
            pairs = {}
            ref = parsed[0]
            for i in range(1, len(parsed)):
                key = f"{labels[0]} vs {labels[i]}"
                pairs[key] = compare_spectra(
                    ref,
                    parsed[i],
                    metrics=metrics,
                    energy=use_energy,
                    cc_icrp116=use_cc,
                    readings1=r1,
                    readings2=r2,
                    response_matrix=rm,
                )
            result_df = pd.DataFrame(pairs)
            result = result_df

        # Plotting
        fig = ax_left = ax_right = None
        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

            # Left: spectra overlay
            colors = sns.color_palette("husl", n_colors=len(parsed))
            for i, s in enumerate(parsed):
                ax_left.semilogy(
                    self.E_MeV,
                    np.maximum(s, 1e-20),
                    label=labels[i],
                    color=colors[i],
                    **plot_kwargs,
                )
            ax_left.set_xlabel("Energy, MeV")
            ax_left.set_ylabel("Fluence per unit lethargy, F(E)E")
            ax_left.set_xscale("log")
            ax_left.legend(fontsize=8)
            ax_left.grid(True, which="both", alpha=0.3)
            ax_left.set_title("Spectra comparison")

            # Right: metric bar chart
            if isinstance(result, dict):
                plot_data = result
                title = "Comparison metrics"
            else:
                plot_data = result.iloc[:, 0].to_dict()
                title = f"Comparison metrics ({labels[0]} vs {labels[1]})"

            if plot_data:
                names = list(plot_data.keys())
                values = list(plot_data.values())
                colors_bars = sns.color_palette("viridis", n_colors=len(names))
                bars = ax_right.barh(names, values, color=colors_bars)
                ax_right.axvline(
                    x=0, color="gray", linestyle="--", linewidth=0.5
                )
                ax_right.set_xlabel("Metric value")
                ax_right.set_title(title)
                ax_right.grid(True, axis="x", alpha=0.3)

                # Annotate bars
                for patch, val in zip(bars, values):
                    if val != 0:
                        lbl = f"{val:.4f}"
                        ax_right.text(
                            val,
                            patch.get_y() + patch.get_height() / 2,
                            lbl,
                            va="center",
                            ha="left" if val > 0 else "right",
                            fontsize=7,
                        )

            fig.tight_layout()

            if save_to is not None:
                self._save_figure(fig, save_to, dpi=dpi)
                plt.close(fig)

        if return_fig:
            return result, fig, ax_left, ax_right
        return result
