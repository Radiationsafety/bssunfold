"""
Mathematical utilities for spectrum unfolding and general numerical operations.
"""

import numpy as np
from typing import Optional, Tuple, Union


def cosine_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute cosine similarity between two spectra.

    Parameters
    ----------
    spectrum1 : np.ndarray
        First spectrum
    spectrum2 : np.ndarray
        Second spectrum

    Returns
    -------
    float
        Cosine similarity in range [-1, 1]
    """
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(spectrum1, spectrum2) / (norm1 * norm2)


def create_first_derivative_matrix(n: int) -> np.ndarray:
    """
    Create first‑order finite‑difference matrix (n‑1 × n).

    Parameters
    ----------
    n : int
        Number of variables (size of vector)

    Returns
    -------
    np.ndarray
        Matrix L of shape (n‑1, n) where L @ x approximates x[i+1] - x[i]
    """
    L = np.zeros((n - 1, n))
    for i in range(n - 1):
        L[i, i] = -1
        L[i, i + 1] = 1
    return L


def create_second_derivative_matrix(n: int) -> np.ndarray:
    """
    Create second‑order finite‑difference matrix (n‑2 × n).

    Parameters
    ----------
    n : int
        Number of variables (size of vector)

    Returns
    -------
    np.ndarray
        Matrix L of shape (n‑2, n) where L @ x approximates x[i] - 2*x[i+1] + x[i+2]
    """
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1
        L[i, i + 1] = -2
        L[i, i + 2] = 1
    return L


def add_gaussian_noise(
    readings: dict, noise_level: float = 0.01, rng: Optional[np.random.Generator] = None
) -> dict:
    """
    Add independent Gaussian noise to each value in a readings dictionary.

    Parameters
    ----------
    readings : dict
        Dictionary mapping keys to numeric values
    noise_level : float, optional
        Relative standard deviation (e.g., 0.01 = 1% noise). Default 0.01.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().

    Returns
    -------
    dict
        Noisy readings with same keys
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = {}
    for key, value in readings.items():
        noise = rng.normal(loc=0, scale=noise_level)
        noisy[key] = value * (1 + noise)
    return noisy


def normalize_to_unit_interval(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize array to [0, 1] range (min‑max scaling).

    Parameters
    ----------
    arr : np.ndarray
        Input array
    eps : float, optional
        Small constant to avoid division by zero

    Returns
    -------
    np.ndarray
        Scaled array with values in [0, 1]
    """
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < eps:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft‑thresholding (proximal operator for L1 norm).

    Parameters
    ----------
    x : np.ndarray
        Input vector
    threshold : float
        Non‑negative threshold

    Returns
    -------
    np.ndarray
        Soft‑thresholded vector: sign(x) * max(|x| - threshold, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)