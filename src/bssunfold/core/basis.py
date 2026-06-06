"""Spectral basis abstraction for neutron spectrum unfolding.

This module provides a pluggable basis system that allows unfolding methods
to work in different representation spaces (bins, Legendre polynomials,
Fourier series, etc.) instead of the default piecewise-constant bin basis.

The basis transformation converts the linear system from:

    A (m × n) @ x (n) = b (m)

to:

    A_proj (m × k) @ c (k) = b (m)

where k << n is the number of basis coefficients, and the final spectrum
is reconstructed as x = Phi @ c with Phi (n × k) being the basis matrix.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = [
    "SpectralBasis",
    "BinBasis",
    "LegendreBasis",
    "FourierBasis",
]


class SpectralBasis(ABC):
    """Abstract base class for spectral basis expansions.

    A basis defines how a neutron spectrum (a vector of length n_energy)
    is represented as a linear combination of k basis functions:

        spectrum = Phi @ coefficients

    where Phi is the (n_energy × k) basis matrix.
    """

    @property
    @abstractmethod
    def n_coeffs(self) -> int:
        """Number of basis coefficients."""
        ...

    @abstractmethod
    def build_matrix(self, n_energy: int, E_MeV: np.ndarray) -> np.ndarray:
        """Build the basis matrix Phi (n_energy × n_coeffs).

        Parameters
        ----------
        n_energy : int
            Number of energy bins.
        E_MeV : np.ndarray
            Energy grid in MeV. Used for correct axis mapping.

        Returns
        -------
        np.ndarray
            Basis matrix of shape (n_energy, n_coeffs).
        """
        ...

    def to_coeffs(
        self, spectrum: np.ndarray, E_MeV: np.ndarray
    ) -> np.ndarray:
        """Forward transform: spectrum → coefficients.

        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum in bin space (n_energy,).
        E_MeV : np.ndarray
            Energy grid in MeV.

        Returns
        -------
        np.ndarray
            Coefficients in basis space (n_coeffs,).
        """
        Phi = self.build_matrix(len(spectrum), E_MeV)
        coeffs, _, _, _ = np.linalg.lstsq(Phi, spectrum, rcond=None)
        return coeffs

    def to_spectrum(
        self, coeffs: np.ndarray, E_MeV: np.ndarray
    ) -> np.ndarray:
        """Inverse transform: coefficients → spectrum.

        Parameters
        ----------
        coeffs : np.ndarray
            Coefficients in basis space (n_coeffs,).
        E_MeV : np.ndarray
            Energy grid in MeV.

        Returns
        -------
        np.ndarray
            Spectrum in bin space (n_energy,).
        """
        Phi = self.build_matrix(len(E_MeV), E_MeV)
        return Phi @ coeffs


class BinBasis(SpectralBasis):
    """Standard bin-by-bin basis (identity transform).

    Each basis function corresponds to a single energy bin.
    This is the default representation: the spectrum is a piecewise-constant
    function over the energy grid.

    Parameters
    ----------
    n_coeffs : int, optional
        Number of bins. If None, inferred from the energy grid at runtime.
    """

    def __init__(self, n_coeffs: Optional[int] = None):
        self._n_coeffs = n_coeffs

    @property
    def n_coeffs(self) -> int:
        if self._n_coeffs is None:
            raise ValueError(
                "BinBasis.n_coeffs is not set. "
                "Pass n_coeffs to the constructor or use build_matrix()."
            )
        return self._n_coeffs

    def build_matrix(self, n_energy: int, E_MeV: np.ndarray) -> np.ndarray:
        """Return identity matrix (n_energy × n_energy)."""
        return np.eye(n_energy)

    def to_coeffs(
        self, spectrum: np.ndarray, E_MeV: np.ndarray
    ) -> np.ndarray:
        """Identity transform — returns spectrum copy."""
        return spectrum.copy()

    def to_spectrum(
        self, coeffs: np.ndarray, E_MeV: np.ndarray
    ) -> np.ndarray:
        """Identity transform — returns coeffs copy."""
        return coeffs.copy()


class LegendreBasis(SpectralBasis):
    """Legendre polynomial basis.

    The energy axis is mapped from [E_min, E_max] to [-1, 1] in log10
    space, and Legendre polynomials P_0, P_1, ..., P_{k-1} are evaluated
    on this mapped grid.

    Parameters
    ----------
    n_polynomials : int, optional
        Number of Legendre polynomials (default: 15).
    """

    def __init__(self, n_polynomials: int = 15):
        self.n_polynomials = n_polynomials

    @property
    def n_coeffs(self) -> int:
        return self.n_polynomials

    def build_matrix(self, n_energy: int, E_MeV: np.ndarray) -> np.ndarray:
        """Build Legendre basis matrix.

        Maps log10(E) to [-1, 1] so that the polynomials are defined
        over the actual energy range, not just bin indices.
        """
        from numpy.polynomial.legendre import Legendre

        log_E = np.log10(np.maximum(E_MeV, 1e-30))
        log_E_min, log_E_max = log_E.min(), log_E.max()

        if log_E_max > log_E_min + 1e-15:
            x = 2.0 * (log_E - log_E_min) / (log_E_max - log_E_min) - 1.0
        else:
            x = np.linspace(-1, 1, n_energy)

        Phi = np.zeros((n_energy, self.n_polynomials))
        for i in range(self.n_polynomials):
            coeffs = np.zeros(i + 1)
            coeffs[-1] = 1.0
            Phi[:, i] = Legendre(coeffs)(x)
        return Phi


class FourierBasis(SpectralBasis):
    """Fourier (sin/cos) basis.

    The energy axis is mapped from [E_min, E_max] to [0, 1] in log10
    space. Basis functions are:

        f_0(x) = 1                    (DC component)
        f_1(x) = cos(2*pi*x)
        f_2(x) = sin(2*pi*x)
        f_3(x) = cos(4*pi*x)
        f_4(x) = sin(4*pi*x)
        ...

    Parameters
    ----------
    n_terms : int, optional
        Number of Fourier terms (default: 15). Must be >= 1.
    """

    def __init__(self, n_terms: int = 15):
        if n_terms < 1:
            raise ValueError("n_terms must be >= 1")
        self.n_terms = n_terms

    @property
    def n_coeffs(self) -> int:
        return self.n_terms

    def build_matrix(self, n_energy: int, E_MeV: np.ndarray) -> np.ndarray:
        """Build Fourier basis matrix.

        Maps log10(E) to [0, 1] so that the sinusoids are defined
        over the actual energy range.
        """
        log_E = np.log10(np.maximum(E_MeV, 1e-30))
        log_E_min, log_E_max = log_E.min(), log_E.max()

        if log_E_max > log_E_min + 1e-15:
            x = (log_E - log_E_min) / (log_E_max - log_E_min)
        else:
            x = np.linspace(0, 1, n_energy)

        Phi = np.ones((n_energy, 1))
        for k in range(1, self.n_terms):
            freq = (k + 1) // 2
            if k % 2 == 1:
                Phi = np.column_stack([Phi, np.cos(2 * np.pi * freq * x)])
            else:
                Phi = np.column_stack([Phi, np.sin(2 * np.pi * freq * x)])
        return Phi[:, : self.n_terms]
