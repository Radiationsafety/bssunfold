"""NSDUAZ unfolding method for neutron spectrum reconstruction.

NSDUAZ ("Neutron Spectrometry and Dosimetry from the Universidad Autonoma
de Zacatecas"; Ortiz-Rodriguez & Vega-Carrillo, 2012) is a Bonner-sphere
unfolding code based on the SPUNIT iterative algorithm (Doroshenko et al.,
1977; the same iteration used by BUNKI).  Its distinctive feature is the
automatic selection of the initial guess spectrum from a *catalogue* of
standard neutron spectra: the experimental count rates are normalised to
the reading of the 20.32 cm-diameter sphere and compared (statistical test)
with the count rates predicted by each catalogue spectrum.  The catalogue
entry that best reproduces the measured relative count-rate pattern is used
as the initial spectrum for the SPUNIT iteration, which then runs until the
relative change of the solution drops below ~1%.

This module implements:

* ``solve_nsduaz`` -- the SPUNIT iteration (thin wrapper over the BUNKI
  SPUNIT solver) with the NSDUAZ default convergence tolerance;
* ``select_catalogue_initial`` -- the catalogue initial-spectrum selection
  (statistical test on count-rate ratios relative to the reference sphere);
* ``unfold_nsduaz`` -- the Detector-facing wrapper.

The reference sphere is identified by name (looking for "20.32" / "20" /
"8in" in the detector name, as in the UTA/IAEA response-matrix conventions)
and falls back to the detector with the largest reading.  A user-supplied
``catalogue`` (dict of name -> spectrum array on the detector energy grid)
overrides the built-in mini-catalogue of analytic standard spectra
(241Am/9Be, 252Cf, thermal + 1/E + fission reactor-like).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding
from .unfold_bunki import solve_bunki

__all__ = [
    "solve_nsduaz",
    "unfold_nsduaz",
    "select_catalogue_initial",
    "builtin_catalogue",
]


def _watt_spectrum(E_MeV: np.ndarray, a: float = 1.025, b: float = 2.926) -> np.ndarray:
    """Analytic Watt fission spectrum (e.g. 252Cf): exp(-E/a) sinh(sqrt(b E))."""
    E = np.asarray(E_MeV, dtype=float)
    E = np.maximum(E, 1e-9)
    w = np.exp(-E / a) * np.sinh(np.sqrt(b * E))
    total = float(np.sum(w))
    return w / total if total > 0 else np.full_like(w, 1.0 / len(w))


def _ambe_spectrum(E_MeV: np.ndarray) -> np.ndarray:
    """Analytic 241Am/9Be(alpha,n) shape: evaporation continuum + 4.2 MeV peak."""
    E = np.asarray(E_MeV, dtype=float)
    E = np.maximum(E, 1e-9)
    continuum = np.exp(-E / 2.0)
    peak = np.exp(-0.5 * ((E - 4.2) / 1.2) ** 2)
    spec = continuum + 3.5 * peak
    total = float(np.sum(spec))
    return spec / total if total > 0 else np.full_like(spec, 1.0 / len(spec))


def _reactor_spectrum(E_MeV: np.ndarray) -> np.ndarray:
    """Analytic reactor-like shape: thermal Maxwellian + 1/E + fast fission."""
    E = np.asarray(E_MeV, dtype=float)
    E = np.maximum(E, 1e-9)
    kT = 0.0253e-6  # 0.0253 eV in MeV
    thermal = (E / kT) * np.exp(-E / kT)
    epithermal = np.where(E > 1e-6, 1.0 / np.maximum(E, 1e-9), 0.0)
    fast = _watt_spectrum(E)
    spec = 1e-3 * thermal + 0.1 * epithermal + fast
    total = float(np.sum(spec))
    return spec / total if total > 0 else np.full_like(spec, 1.0 / len(spec))


def builtin_catalogue(E_MeV: np.ndarray) -> Dict[str, np.ndarray]:
    """Build the built-in mini-catalogue of analytic standard spectra.

    Parameters
    ----------
    E_MeV : np.ndarray
        Detector energy grid (MeV).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of catalogue label to a normalised spectrum on the energy
        grid: ``'ambe'``, ``'cf252'`` and ``'reactor'``.
    """
    return {
        "ambe": _ambe_spectrum(E_MeV),
        "cf252": _watt_spectrum(E_MeV),
        "reactor": _reactor_spectrum(E_MeV),
    }


def _find_reference_index(detector_names: List[str], A: np.ndarray) -> int:
    """Locate the reference sphere (20.32 cm diameter) index.

    Searches the detector names for the usual UTA/IAEA conventions
    ("20.32", "20", "8in", "8 in") and falls back to the detector row with
    the largest response norm (a proxy for the largest moderating sphere).
    """
    for i, name in enumerate(detector_names):
        lowered = name.lower()
        if (
            "20.32" in lowered
            or "20in" in lowered
            or "8in" in lowered
            or "8 in" in lowered
        ):
            return i
    # Fallback: detector with the largest integrated response.
    return int(np.argmax(np.sum(np.abs(A), axis=1)))


def select_catalogue_initial(
    readings: Dict[str, float],
    detector_names: List[str],
    sensitivities: Dict[str, np.ndarray],
    catalogue: Optional[Dict[str, np.ndarray]] = None,
    reference_name: Optional[str] = None,
    E_MeV: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """Select the initial spectrum from a catalogue using a statistical test.

    The experimental count rates are normalised to the reading of the
    reference sphere (20.32 cm by default) and compared with the relative
    count-rate pattern predicted by each catalogue spectrum folded through
    the response matrix.  The entry minimising the weighted chi-square of
    the relative ratios is chosen and rescaled so that its predicted
    reference reading matches the measured one.

    Parameters
    ----------
    readings : Dict[str, float]
        Detector readings.
    detector_names : List[str]
        Names of available detectors.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    catalogue : Dict[str, np.ndarray], optional
        Mapping of label -> spectrum on the detector energy grid.  When None,
        :func:`builtin_catalogue` is used.
    reference_name : str, optional
        Name of the reference detector.  When None, the 20.32 cm sphere is
        located automatically.
    E_MeV : np.ndarray, optional
        Energy grid used to build the built-in catalogue shapes.  When None,
        a representative log grid over the sensitivity length is used.

    Returns
    -------
    Tuple[np.ndarray, str]
        (selected initial spectrum, catalogue label).
    """
    selected = [name for name in detector_names if name in readings]
    if not selected:
        raise ValueError("No detector readings available for catalogue selection")
    b = np.array([readings[name] for name in selected], dtype=float)
    A = np.array([sensitivities[name] for name in selected], dtype=float)

    if reference_name is not None:
        if reference_name not in readings:
            raise ValueError(
                f"reference_name '{reference_name}' is not present in readings"
            )
        ref_idx = int(selected.index(reference_name))
    else:
        ref_idx = _find_reference_index(selected, A)

    if catalogue is None:
        n_bins = A.shape[1]
        if E_MeV is not None:
            E_MeV_arr = np.asarray(E_MeV, dtype=float)
            if E_MeV_arr.shape == (n_bins,):
                cat = builtin_catalogue(E_MeV_arr)
            else:
                E_rep = np.logspace(np.log10(1e-9), np.log10(1e2), n_bins)
                cat = builtin_catalogue(E_rep)
        else:
            E_rep = np.logspace(np.log10(1e-9), np.log10(1e2), n_bins)
            cat = builtin_catalogue(E_rep)
        catalogue = cat

    b_ref = b[ref_idx]
    if b_ref <= 0:
        raise ValueError("Reference sphere reading must be strictly positive")
    r_ratio = b / b_ref

    best_label = None
    best_chi = np.inf
    best_scale = 1.0
    best_spec = None

    for label, spec in catalogue.items():
        spec = np.asarray(spec, dtype=float)
        if spec.shape != (A.shape[1],):
            raise ValueError(
                f"Catalogue spectrum '{label}' has shape {spec.shape}, "
                f"expected ({A.shape[1]},)"
            )
        if not np.any(spec > 0):
            continue
        c = A @ np.maximum(spec, 0)
        c_ref = c[ref_idx]
        if c_ref <= 0:
            continue
        s_ratio = c / c_ref
        denom = np.maximum(s_ratio, 1e-12)
        chi = float(np.sum(((r_ratio - s_ratio) / denom) ** 2))
        if chi < best_chi:
            best_chi = chi
            best_label = label
            best_scale = b_ref / c_ref
            best_spec = spec

    if best_spec is None:
        raise ValueError("Catalogue is empty or has no usable spectrum")
    return np.maximum(best_scale * best_spec, 0.0), best_label


def solve_nsduaz(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    smoothing: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 0.01,
    lethargy_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using the NSDUAZ (SPUNIT) iteration.

    This is the SPUNIT iterative algorithm with the NSDUAZ default
    convergence tolerance (~1% relative change).  See
    :func:`bssunfold.core.unfold_bunki.solve_bunki` for the iteration
    details.  The initial spectrum ``x0`` is typically obtained via
    :func:`select_catalogue_initial` (or provided by the user).

    Parameters
    ----------
    A : np.ndarray
        Lethargy-weighted response matrix (m x n) as built by the Detector
        class.
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    smoothing : float, optional
        Three-point smoothing factor (default: 0.1).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 0.01).
    lethargy_weights : np.ndarray, optional
        Per-bin lethargy widths. Only needed when ``A`` is supplied as a
        per-bin (non-lethargy-weighted) response matrix.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A, b, x0 = validate_system(
        A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance
    )
    return solve_bunki(
        A=A,
        b=b,
        x0=x0,
        smoothing=smoothing,
        max_iterations=max_iterations,
        tolerance=tolerance,
        lethargy_weights=lethargy_weights,
    )


def unfold_nsduaz(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
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
        Explicit initial spectrum guess.  When given, it overrides the
        catalogue selection.
    catalogue : Optional[Dict[str, np.ndarray]], optional
        User-supplied catalogue of candidate initial spectra (label ->
        spectrum on the detector energy grid).  When None, the built-in
        mini-catalogue is used.
    use_catalogue : bool, optional
        If True (default), the initial spectrum is selected from the
        catalogue when ``initial_spectrum`` is not provided; if False, a
        flat spectrum is used (NSDUAZ "flat spectrum" mode).
    reference_name : str, optional
        Reference sphere name for the catalogue statistical test (default:
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
    x0_default = np.ones(n_energy_bins)
    cat_label = None

    if initial_spectrum is None and use_catalogue:
        initial_spectrum, cat_label = select_catalogue_initial(
            readings=readings,
            detector_names=detector_names,
            sensitivities=sensitivities,
            catalogue=catalogue,
            reference_name=reference_name,
            E_MeV=E_MeV,
        )

    extra = {
        "smoothing": float(smoothing),
        "catalogue": cat_label,
    }

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
        solve_func=make_solve_wrapper(
            solve_nsduaz,
            smoothing=smoothing,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
        solve_kwargs={},
        method_name="NSDUAZ",
        extra_output=extra,
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
