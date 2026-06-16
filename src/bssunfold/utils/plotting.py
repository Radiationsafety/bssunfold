"""Plotting utilities for bssunfold package.

This module provides functions for visualizing spectra, response functions,
and unfolding results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

__all__ = [
    "plot_spectrum",
    "plot_response_functions",
    "plot_with_uncertainty",
    "plot_residuals",
    "plot_comparison",
]


def plot_spectrum(
    E_MeV: np.ndarray,
    spectrum: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    log_x: bool = True,
    log_y: bool = False,
    show: bool = True,
    save_to: Optional[str] = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a spectrum.
    
    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    spectrum : np.ndarray
        Spectrum values.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    label : str, optional
        Label for the spectrum.
    log_x : bool, optional
        Use logarithmic x-axis (default: True).
    log_y : bool, optional
        Use logarithmic y-axis (default: False).
    show : bool, optional
        Call plt.show() (default: True).
    save_to : str, optional
        Path to save figure. If None, not saved.
    **plot_kwargs : dict
        Additional keyword arguments for plt.plot().
    
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure
    
    ax.plot(E_MeV, spectrum, label=label, **plot_kwargs)
    
    ax.set_xlabel("Energy, MeV")
    ax.set_ylabel("Fluence per unit lethargy, F(E)E")
    
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    
    ax.grid(True, which="both", alpha=0.3)
    
    if label:
        ax.legend()
    
    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_response_functions(
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    log_x: bool = True,
    show: bool = True,
    save_to: Optional[str] = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot detector response functions.
    
    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Dictionary mapping detector names to sensitivity arrays.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    log_x : bool, optional
        Use logarithmic x-axis (default: True).
    show : bool, optional
        Call plt.show() (default: True).
    save_to : str, optional
        Path to save figure. If None, not saved.
    **plot_kwargs : dict
        Additional keyword arguments for plt.plot().
    
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure
    
    for name, sens in sensitivities.items():
        ax.plot(E_MeV, sens, label=name, **plot_kwargs)
    
    ax.set_xlabel("Energy, MeV")
    ax.set_ylabel("Response, cm²")
    
    if log_x:
        ax.set_xscale("log")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Response functions of the detector")
    
    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_with_uncertainty(
    E_MeV: np.ndarray,
    spectrum: np.ndarray,
    uncert_min: Optional[np.ndarray] = None,
    uncert_max: Optional[np.ndarray] = None,
    uncert_std: Optional[np.ndarray] = None,
    reference_spectrum: Optional[Dict[str, np.ndarray]] = None,
    ax: Optional[plt.Axes] = None,
    plot_style: str = "fill_between",
    show: bool = True,
    save_to: Optional[str] = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot spectrum with uncertainty range.
    
    Parameters
    ----------
    E_MeV : np.ndarray
        Energy grid in MeV.
    spectrum : np.ndarray
        Central spectrum values.
    uncert_min : np.ndarray, optional
        Minimum uncertainty bound.
    uncert_max : np.ndarray, optional
        Maximum uncertainty bound.
    uncert_std : np.ndarray, optional
        Standard deviation for error bars.
    reference_spectrum : Dict[str, np.ndarray], optional
        Reference spectrum with 'E_MeV' and 'Phi' keys.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    plot_style : str, optional
        Style for uncertainty: 'fill_between' or 'errorbar' (default: 'fill_between').
    show : bool, optional
        Call plt.show() (default: True).
    save_to : str, optional
        Path to save figure. If None, not saved.
    **plot_kwargs : dict
        Additional keyword arguments for plotting.
    
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Plot reference if provided
    if reference_spectrum is not None:
        ax.plot(
            reference_spectrum["E_MeV"],
            reference_spectrum["Phi"],
            label="reference",
            linestyle=":",
            color="black",
        )
    
    # Plot uncertainty
    if plot_style == "fill_between" and uncert_min is not None and uncert_max is not None:
        ax.fill_between(
            E_MeV,
            uncert_min,
            uncert_max,
            alpha=0.3,
            hatch="\\",
            label="uncertainty range",
            **plot_kwargs,
        )
    elif plot_style == "errorbar" and uncert_std is not None:
        # Subsample for error bars if too many points
        n_points = len(E_MeV)
        if n_points > 50:
            step = n_points // 50
            indices = np.arange(0, n_points, step)
            if indices[-1] != n_points - 1:
                indices = np.append(indices, n_points - 1)
            ax.errorbar(
                E_MeV[indices],
                spectrum[indices],
                yerr=uncert_std[indices],
                fmt="none",
                ecolor=plot_kwargs.get("color", "blue"),
                capsize=2,
                alpha=0.5,
                label="±1σ",
            )
        else:
            ax.errorbar(
                E_MeV,
                spectrum,
                yerr=uncert_std,
                fmt="none",
                ecolor=plot_kwargs.get("color", "blue"),
                capsize=2,
                alpha=0.5,
                label="±1σ",
            )
    
    # Plot spectrum
    ax.plot(E_MeV, spectrum, label="spectrum", **plot_kwargs)
    
    ax.set_xlabel("Energy, MeV")
    ax.set_ylabel("Fluence per unit lethargy, F(E)E")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    
    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_residuals(
    measured: np.ndarray,
    calculated: np.ndarray,
    detector_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_to: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot residuals between measured and calculated readings.
    
    Parameters
    ----------
    measured : np.ndarray
        Measured readings.
    calculated : np.ndarray
        Calculated readings from unfolded spectrum.
    detector_names : List[str], optional
        Detector names for x-axis labels.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, optional
        Call plt.show() (default: True).
    save_to : str, optional
        Path to save figure. If None, not saved.
    
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure
    
    residuals = measured - calculated
    n_detectors = len(residuals)
    
    if detector_names is None:
        detector_names = [f"det_{i}" for i in range(n_detectors)]
    
    x_positions = np.arange(n_detectors)
    
    ax.bar(x_positions, residuals, alpha=0.7)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Detector")
    ax.set_ylabel("Residual (measured - calculated)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(detector_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Unfolding Residuals")

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_comparison(
    results: Dict[str, Dict],
    readings: Dict[str, float],
    reference_spectrum: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (8, 8),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    show: bool = True,
    save_to: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Compare multiple unfolded spectra and their effective readings.

    Creates a two-panel figure: unfolded spectra (top) and a grouped bar chart
    of effective readings (bottom).

    Parameters
    ----------
    results : Dict[str, Dict]
        Mapping of method names to result dictionaries. Each result must contain
        ``"energy"`` (np.ndarray), ``"spectrum"`` (np.ndarray), and
        ``"effective_readings"`` (Dict[str, float]).
    readings : Dict[str, float]
        Measured (ground-truth) readings keyed by detector name.
    reference_spectrum : Dict[str, np.ndarray], optional
        Reference spectrum with ``"E_MeV"`` and ``"Phi"`` keys.
    figsize : Tuple[int, int], optional
        Figure size (default: (8, 8)).
    colors : List[str], optional
        Colors for reference + each method. Defaults to a built-in palette.
    markers : List[str], optional
        Markers for each method on the spectra plot. Defaults to a built-in set.
    show : bool, optional
        Call ``plt.show()`` (default: True).
    save_to : str, optional
        Path to save figure. If None, not saved.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of two Axes objects.
    """
    default_colors = ["black", "green", "tan", "blue", "indianred", "red"]
    default_markers = ["o", "s", "^", "*", "D", "v", "*"]
    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    method_names = list(results.keys())

    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # --- Top panel: spectra ---
    for i, method in enumerate(method_names):
        ax[0].plot(
            results[method]["energy"],
            results[method]["spectrum"],
            label=method,
            color=colors[(i + 1) % len(colors)],
            ls="-",
            marker=markers[i % len(markers)],
            markersize=3,
            linewidth=0.6,
            alpha=1,
        )

    if reference_spectrum is not None:
        ax[0].plot(
            reference_spectrum["E_MeV"],
            reference_spectrum["Phi"],
            label="reference",
            linewidth=1,
            linestyle=":",
            color=colors[0],
        )

    ax[0].set_xlabel("Energy, MeV")
    ax[0].set_ylabel("Fluence per unit lethargy, F(E)E, neutron/(cm² ∙ s)")
    ax[0].set_xscale("log")
    ax[0].legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=8
    )
    ax[0].grid(True, which="both", ls=":")
    ax[0].set_title("Unfolded spectra from noisy readings", fontsize=14)

    # --- Bottom panel: effective readings bar chart ---
    data_sources = {}
    if reference_spectrum is not None:
        data_sources["reference"] = list(readings.values())
    for method in method_names:
        data_sources[method] = [
            results[method]["effective_readings"][det] for det in readings.keys()
        ]

    labels = list(readings.keys())
    x = np.arange(len(labels))
    n_groups = len(data_sources)
    width = 0.8 / (n_groups * 1.5)

    for i, (label, values) in enumerate(data_sources.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        ax[1].bar(x + offset, values, width, label=label, alpha=1,
                  color=colors[i % len(colors)])

    ax[1].set_xticks(x, labels)
    ax[1].set_xlabel("Moderator sphere")
    ax[1].set_ylabel("Readings")
    ax[1].set_title("Effective readings", fontsize=14)
    ax[1].legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=8
    )
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
