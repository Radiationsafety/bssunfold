"""
Модуль для построения графиков, связанных с детектором и восстановлением спектров.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union


def save_figure(
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
        File path for saving. Supported extensions: .png, .jpg, .jpeg,
        .eps, .pdf.
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
    print(f"Figure saved to: {save_to}")


def plot_response_functions(
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
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
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Dictionary mapping detector names to response functions.
    save_to : str, optional
        File path to save the figure. Supported extensions: .png, .jpg,
        .jpeg, .eps, .pdf.
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
    for key, rf in sensitivities.items():
        ax.plot(
            E_MeV,
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
    save_figure(
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
    E_MeV: np.ndarray,
    result: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Dict[str, Any]]] = None,
    reference_spectrum: Optional[Union[pd.DataFrame, Dict]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    plot_style: str = 'fill_between',
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
    E_MeV : np.ndarray
        Energy grid in MeV.
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
        File path to save the figure. Supported extensions: .png, .jpg,
        .jpeg, .eps, .pdf.
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
    """
    # Validate plot_style
    valid_styles = ['fill_between', 'errorbar']
    if plot_style not in valid_styles:
        raise ValueError(
            f"plot_style must be one of {valid_styles}, got '{plot_style}'"
        )

    # Determine what to plot
    if results is not None:
        # Multiple results
        plot_multiple = True
        result_dict = results
    else:
        # Single result
        plot_multiple = False
        if result is None:
            raise ValueError(
                "No result provided and no current result available."
            )
        result_dict = {"result": result}

    # Prepare colors
    if colors is None:
        # Default palette: black for reference, then tab10
        default_colors = [
            "black",
            "#1f77b4",
            "#e68910",
            "#589c43",
            "indianred",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
        ]
        # If more spectra than colors, cycle
        colors = default_colors

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure  # type: ignore

    # Plot reference spectrum if provided
    if reference_spectrum is not None:
        if isinstance(reference_spectrum, pd.DataFrame):
            ref_E = reference_spectrum["E_MeV"].values
            ref_Phi = reference_spectrum["Phi"].values
        elif isinstance(reference_spectrum, dict):
            ref_E = np.asarray(reference_spectrum["E_MeV"])
            ref_Phi = np.asarray(reference_spectrum["Phi"])
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
        energy = res.get("energy", E_MeV)
        spectrum = res.get("spectrum")
        if spectrum is None:
            raise ValueError(
                f"Result for '{method}' missing 'spectrum' key."
            )

        # Plot uncertainty based on style
        if plot_style == 'fill_between':
            # Use min/max for fill_between
            uncert_min = res.get("spectrum_uncert_min")
            uncert_max = res.get("spectrum_uncert_max")

            if uncert_min is not None and uncert_max is not None:
                ax.fill_between(
                    energy,
                    uncert_min,
                    uncert_max,
                    alpha=0.3,
                    hatch='\\',
                    facecolor=color,
                    color=color,
                    label=f"{method} uncertainty range",
                )

        elif plot_style == 'errorbar':
            # Use standard deviation for error bars
            uncert_std = res.get("spectrum_uncert_std")

            if uncert_std is not None:
                # Calculate subsample for error bars to avoid overcrowding
                n_points = len(energy)
                if n_points > 50:  # If too many points, subsample
                    step = n_points // 50
                    indices = np.arange(0, n_points, step)
                    # Ensure last point is included
                    if indices[-1] != n_points - 1:
                        indices = np.append(indices, n_points - 1)

                    e_plot = energy[indices]
                    s_plot = spectrum[indices]
                    err_plot = uncert_std[indices]
                else:
                    e_plot = energy
                    s_plot = spectrum
                    err_plot = uncert_std

                # Plot error bars
                ax.errorbar(
                    e_plot,
                    s_plot,
                    yerr=err_plot,
                    fmt='none',  # No markers
                    ecolor=color,
                    capsize=2,
                    capthick=1,
                    alpha=0.5,
                    label=f"{method} ±1σ" if len(result_dict) == 1 else None
                )

        # Plot spectrum line
        ax.plot(
            energy,
            spectrum,
            label=method,
            color=color,
            ls="-",
            linewidth=1.3,
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

    # For errorbar style, consider adding some headroom for error bars
    if plot_style == 'errorbar':
        ymax *= 1.1

    ax.set_ylim(0, ymax)

    # Legend handling for errorbar style
    if plot_style == 'errorbar' and len(result_dict) > 1:
        # Don't show individual error bar labels for multiple results
        handles, labels = ax.get_legend_handles_labels()
        # Keep only spectrum lines and reference in legend
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if '±1σ' not in l:  # Skip error bar entries
                filtered_handles.append(h)
                filtered_labels.append(l)
        ax.legend(filtered_handles, filtered_labels, loc="upper left",
                  borderaxespad=0.0, fontsize=8)
    else:
        ax.legend(loc="upper left", borderaxespad=0.0, fontsize=8)

    ax.grid(True, which="both", ls=":")

    # Title
    if title is None:
        if plot_multiple:
            title = f"Unfolded spectra with uncertainty ({plot_style})"
        else:
            method = list(result_dict.keys())[0]
            title = (
                f"Unfolded spectrum ({method}) with uncertainty ({plot_style})"
            )
    ax.set_title(title, fontsize=14)

    # Save figure if requested
    save_figure(
        fig,
        save_to=save_to,
        dpi=dpi,
        bbox_inches=bbox_inches,
        **savefig_kwargs,
    )

    if show:
        plt.show()

    return fig, ax